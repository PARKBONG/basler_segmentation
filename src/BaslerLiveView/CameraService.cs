using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Basler.Pylon;

namespace BaslerLiveView;

/// <summary>
/// Thin wrapper around a single Basler <see cref="Camera"/> that grabs images
/// continuously and hands each frame to the UI as a BGRA byte buffer.
///
/// Lifecycle: Open() -> Start() -> (FrameReady events) -> Stop() -> Close().
/// <see cref="FrameReady"/> is raised on the pylon grab-loop thread, so the
/// listener must marshal onto the UI thread before touching WPF objects.
/// </summary>
public sealed class CameraService : IDisposable
{
    private Camera? _camera;

    // Reused across frames: converts whatever the sensor delivers (Mono8, Bayer,
    // YUV, ...) into 32-bit BGRA, which maps directly onto WPF's Bgra32 bitmap.
    private readonly PixelDataConverter _converter = new()
    {
        OutputPixelFormat = PixelType.BGRA8packed
    };

    private byte[] _buffer = Array.Empty<byte>();

    // --- DIAGNOSTIC (temporary): distinguish transport packet loss from UI
    // blocking when fps randomly drops. Writes to "grab-stats.log" next to the
    // exe once per second. Remove once the cause is confirmed.
    private readonly Stopwatch _statClock = new();
    private long _grabbedSinceLog;   // buffers actually delivered to us
    private long _lastLogMs;
    private readonly string _statLogPath =
        Path.Combine(AppContext.BaseDirectory, "grab-stats.log");

    /// <summary>Raised per grabbed frame: (width, height, bgraBuffer).
    /// The buffer is reused, so copy out of it before returning.</summary>
    public event Action<int, int, byte[]>? FrameReady;

    /// <summary>Human-readable status text.</summary>
    public event Action<string>? StatusChanged;

    /// <summary>Raised for failures on the grab thread.</summary>
    public event Action<Exception>? ErrorOccurred;

    public bool IsOpen => _camera?.IsOpen ?? false;
    public bool IsGrabbing => _camera?.StreamGrabber?.IsGrabbing ?? false;

    /// <summary>Enumerate all cameras visible to every installed transport layer.</summary>
    public static List<ICameraInfo> Enumerate() => CameraFinder.Enumerate();

    /// <summary>Open a camera. Pass null to open the first available device.</summary>
    public void Open(ICameraInfo? info = null)
    {
        Close();

        _camera = info != null ? new Camera(info) : new Camera();

        // Registers the standard "acquire continuous" node-map setup, applied
        // automatically once the camera opens. Wrapped in a lambda so the
        // nullable-annotated event signature matches AcquireContinuous.
        _camera.CameraOpened += (s, e) => Configuration.AcquireContinuous(s!, e);
        _camera.Open();

        // Apply app config (config\config.xml) so acquisition settings are
        // deterministic instead of inheriting whatever is persisted in the
        // camera — the cause of the random 80fps↔3fps swings (stale exposure).
        ApplyConfig(CameraConfig.Load());

        var camInfo = _camera!.CameraInfo!;
        var model = camInfo[CameraInfoKey.ModelName];
        var serial = camInfo[CameraInfoKey.SerialNumber];
        StatusChanged?.Invoke($"Connected: {model} (SN {serial})");
    }

    /// <summary>Push the config's acquisition settings onto the open camera so
    /// the frame rate is deterministic instead of inheriting stale camera state.</summary>
    private void ApplyConfig(CameraConfig config)
    {
        if (_camera == null || config.FrameRate <= 0) return;
        var p = _camera.Parameters;

        double periodUs = 1_000_000.0 / config.FrameRate; // frame period at target fps

        // 1) A fixed exposure that fits inside the frame period (10% headroom for
        //    sensor readout) — otherwise a long stale exposure caps the real fps.
        try { p[PLCamera.ExposureAuto].SetValue("Off"); }
        catch (Exception ex) { StatusChanged?.Invoke("ExposureAuto set failed: " + ex.Message); }
        SetExposureUs(periodUs * 0.9);

        // 2) Pin the acquisition frame rate to the target (enable the control first).
        try { p[PLCamera.AcquisitionFrameRateEnable].SetValue(true); }
        catch (Exception ex) { StatusChanged?.Invoke("FrameRateEnable set failed: " + ex.Message); }
        SetFrameRate(config.FrameRate);
    }

    /// <summary>Crop the sensor to <paramref name="targetW"/>×<paramref name="targetH"/>
    /// via the hardware ROI (Width/Height/OffsetX/OffsetY). Either dimension ≤ 0
    /// disables cropping — which still writes the ROI, restoring the full sensor,
    /// because the camera keeps the previously applied ROI in its power-up set.
    /// <paramref name="pctX"/>/<paramref name="pctY"/> place the window within the
    /// available travel (0–100, 50 = centered; a smaller Y nudges it upward).
    /// Sizes/offsets are clamped to the camera's valid range and rounded down to
    /// its required increment.</summary>
    private void SetCenteredRoi(int targetW, int targetH, double pctX, double pctY)
    {
        var p = _camera!.Parameters;
        try
        {
            var wNode = p[PLCamera.Width];
            var hNode = p[PLCamera.Height];
            var oxNode = p[PLCamera.OffsetX];
            var oyNode = p[PLCamera.OffsetY];

            // Rounds v down to the nearest increment step, then clamps to [min, max].
            static long Align(long v, long inc, long min, long max)
            {
                v = Math.Clamp(v, min, max);
                if (inc > 1) v -= (v - min) % inc;
                return v;
            }

            // Zero the offsets first so the current offset doesn't cap the max size
            // we're allowed to request.
            try { oxNode.SetValue(oxNode.GetMinimum()); } catch { /* offset may be read-only until size set */ }
            try { oyNode.SetValue(oyNode.GetMinimum()); } catch { }

            // Cropping disabled (0) → request the full sensor, i.e. each node's max.
            // Never skip the write: the camera persists the ROI from the last run,
            // so leaving it untouched would silently keep the old crop.
            bool full = targetW <= 0 || targetH <= 0;
            long reqW = full ? wNode.GetMaximum() : targetW;
            long reqH = full ? hNode.GetMaximum() : targetH;

            wNode.SetValue(Align(reqW, wNode.GetIncrement(), wNode.GetMinimum(), wNode.GetMaximum()));
            hNode.SetValue(Align(reqH, hNode.GetIncrement(), hNode.GetMinimum(), hNode.GetMaximum()));

            // At full size the offset travel is zero; the offsets stay at their
            // minimum set above, so skip the placement step entirely.
            if (full) return;

            // With the size set, OffsetX/Y max == sensorMax − size (the full travel).
            // pct% of that travel places the window: 50% centers it, <50% on Y moves
            // it toward the top. Clamp the percentage defensively before applying.
            static long AtPct(double pct, long inc, long min, long max)
            {
                double t = Math.Clamp(pct, 0, 100) / 100.0;
                long v = min + (long)Math.Round((max - min) * t);
                v = Math.Clamp(v, min, max);
                if (inc > 1) v -= (v - min) % inc;
                return v;
            }

            oxNode.SetValue(AtPct(pctX, oxNode.GetIncrement(), oxNode.GetMinimum(), oxNode.GetMaximum()));
            oyNode.SetValue(AtPct(pctY, oyNode.GetIncrement(), oyNode.GetMinimum(), oyNode.GetMaximum()));
        }
        catch (Exception ex)
        {
            StatusChanged?.Invoke("ROI (center crop) set failed: " + ex.Message);
        }
    }

    /// <summary>Set the acquisition frame rate (fps), clamped to the camera's
    /// valid range. Handles current (AcquisitionFrameRate) and legacy (…Abs) names.</summary>
    private void SetFrameRate(double fps)
    {
        var p = _camera!.Parameters;
        foreach (var key in new[] { PLCamera.AcquisitionFrameRate, PLCamera.AcquisitionFrameRateAbs })
        {
            try
            {
                var f = p[key];
                f.SetValue(Math.Clamp(fps, f.GetMinimum(), f.GetMaximum()));
                return;
            }
            catch { /* try the next name */ }
        }
        StatusChanged?.Invoke("Frame rate set failed: no writable AcquisitionFrameRate node.");
    }

    /// <summary>Set exposure (µs), clamped to the camera's valid range. Handles
    /// both current (ExposureTime) and legacy (ExposureTimeAbs) SFNC names.</summary>
    private void SetExposureUs(double us)
    {
        var p = _camera!.Parameters;
        foreach (var key in new[] { PLCamera.ExposureTime, PLCamera.ExposureTimeAbs })
        {
            try
            {
                var e = p[key];
                e.SetValue(Math.Clamp(us, e.GetMinimum(), e.GetMaximum()));
                return;
            }
            catch { /* try the next name */ }
        }
        StatusChanged?.Invoke("Exposure set failed: no writable ExposureTime node.");
    }

    public void Start()
    {
        if (_camera == null)
            throw new InvalidOperationException("Open a camera before starting the grab.");

        _camera.StreamGrabber!.ImageGrabbed += OnImageGrabbed;

        // LatestImages = keep only the freshest frames, drop backlog under load.
        // ProvidedByStreamGrabber = pylon runs its own grab-loop thread and
        // raises ImageGrabbed for us (no manual RetrieveResult loop needed).
        // DIAGNOSTIC: reset counters and start the once-per-second stats dump.
        _grabbedSinceLog = 0;
        _lastLogMs = 0;
        _statClock.Restart();
        File.AppendAllText(_statLogPath,
            $"===== grab started {DateTime.Now:yyyy-MM-dd HH:mm:ss} =====" + Environment.NewLine);
        DumpSettings();   // DIAGNOSTIC: which knob is capping the frame rate?

        _camera.StreamGrabber.Start(GrabStrategy.LatestImages, GrabLoop.ProvidedByStreamGrabber);
        StatusChanged?.Invoke("Grabbing…");
    }

    public void Stop()
    {
        if (_camera == null) return;

        DumpStats("stop");   // DIAGNOSTIC: final snapshot before teardown.

        if (_camera!.StreamGrabber!.IsGrabbing)
            _camera.StreamGrabber.Stop();

        _camera.StreamGrabber.ImageGrabbed -= OnImageGrabbed;
        StatusChanged?.Invoke("Stopped.");
    }

    private void OnImageGrabbed(object? sender, ImageGrabbedEventArgs e)
    {
        try
        {
            IGrabResult result = e.GrabResult;
            if (!result.GrabSucceeded)
            {
                StatusChanged?.Invoke($"Grab failed: {result.ErrorCode} {result.ErrorDescription}");
                return;
            }

            int width = result.Width;
            int height = result.Height;
            int needed = width * height * 4; // BGRA = 4 bytes/pixel

            if (_buffer.Length < needed)
                _buffer = new byte[needed];

            // Convert directly into the reusable managed buffer.
            _converter.Convert(_buffer, result);

            // DIAGNOSTIC: count this delivered buffer and dump stats ~1×/sec.
            _grabbedSinceLog++;
            if (_statClock.ElapsedMilliseconds - _lastLogMs >= 1000)
                DumpStats("run");

            FrameReady?.Invoke(width, height, _buffer);
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(ex);
        }
    }

    // DIAGNOSTIC (temporary): logs the camera's configured output rate vs. what
    // actually arrives, plus the GigE loss/resend counters. Interpretation:
    //   • camFps≈80 but grabbed≈3 and missed climbing  → UI/render blocking.
    //   • camFps low with failed/resend/failedPkt climbing → transport loss.
    //   • camFps itself low with clean counters → camera acquisition settings
    //     (exposure / frame-rate cap / bandwidth) — see DumpSettings().
    private void DumpStats(string tag)
    {
        if (_camera == null) return;

        long elapsed = _statClock.ElapsedMilliseconds;
        double secs = Math.Max(1, elapsed - _lastLogMs) / 1000.0;
        double grabbedFps = _grabbedSinceLog / secs;
        _grabbedSinceLog = 0;
        _lastLogMs = elapsed;

        long I(IntegerName key)
        {
            try { return (long)_camera!.Parameters[key].GetValue(); }
            catch { return -1; }
        }
        double F(FloatName key)
        {
            try { return _camera!.Parameters[key].GetValue(); }
            catch { return double.NaN; }
        }

        string line =
            $"[{tag} t={elapsed / 1000.0:F1}s] " +
            $"grabbedFps={grabbedFps:F1} camFps={F(PLCamera.ResultingFrameRate):F1} " +
            $"total={I(PLStream.Statistic_Total_Buffer_Count)} " +
            $"failed={I(PLStream.Statistic_Failed_Buffer_Count)} " +
            $"missed={I(PLStream.Statistic_Missed_Frame_Count)} " +
            $"underrun={I(PLStream.Statistic_Buffer_Underrun_Count)} " +
            $"resendReq={I(PLStream.Statistic_Resend_Request_Count)} " +
            $"resendPkt={I(PLStream.Statistic_Resend_Packet_Count)} " +
            $"failedPkt={I(PLStream.Statistic_Failed_Packet_Count)} " +
            $"packetSize={I(PLCamera.GevSCPSPacketSize)}";

        try { File.AppendAllText(_statLogPath, line + Environment.NewLine); }
        catch { /* logging must never break the grab loop */ }
    }

    // DIAGNOSTIC (temporary): one-shot dump of the acquisition knobs that decide
    // ResultingFrameRate. Handles both current and legacy (*Abs) SFNC names.
    private void DumpSettings()
    {
        if (_camera == null) return;
        var p = _camera.Parameters;

        string Get(string label, Func<string> read)
        {
            try { return $"{label}={read()}"; } catch { return $"{label}=n/a"; }
        }
        string ExposureUs()
        {
            try { return p[PLCamera.ExposureTime].GetValue().ToString("F0"); }
            catch { return p[PLCamera.ExposureTimeAbs].GetValue().ToString("F0"); }
        }
        string FrameRate()
        {
            try { return p[PLCamera.AcquisitionFrameRate].GetValue().ToString("F1"); }
            catch { return p[PLCamera.AcquisitionFrameRateAbs].GetValue().ToString("F1"); }
        }

        var parts = new List<string>
        {
            Get("ROI", () => $"{p[PLCamera.Width].GetValue()}x{p[PLCamera.Height].GetValue()}" +
                             $"+{p[PLCamera.OffsetX].GetValue()}+{p[PLCamera.OffsetY].GetValue()}"),
            Get("ExposureAuto",         () => p[PLCamera.ExposureAuto].GetValue()),
            Get("ExposureTime(us)",     ExposureUs),
            Get("FrameRateEnable",      () => p[PLCamera.AcquisitionFrameRateEnable].GetValue().ToString()),
            Get("FrameRate",            FrameRate),
            Get("ResultingFrameRate",   () => p[PLCamera.ResultingFrameRate].GetValue().ToString("F1")),
            Get("ThroughputLimitMode",  () => p[PLCamera.DeviceLinkThroughputLimitMode].GetValue()),
            Get("ThroughputLimit",      () => p[PLCamera.DeviceLinkThroughputLimit].GetValue().ToString()),
        };

        try { File.AppendAllText(_statLogPath, "  SETTINGS " + string.Join("  ", parts) + Environment.NewLine); }
        catch { /* ignore */ }
    }

    public void Close()
    {
        if (_camera == null) return;

        try
        {
            Stop();
            if (_camera.IsOpen)
                _camera.Close();
        }
        finally
        {
            _camera.Dispose();
            _camera = null;
        }
    }

    public void Dispose()
    {
        Close();
        _converter.Dispose();
    }
}
