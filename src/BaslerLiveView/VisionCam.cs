using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Basler.Pylon;
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.DirectML;
using YoloDotNet.Models;

namespace BaslerLiveView;

/// <summary>
/// A Basler camera plus (optionally) YOLO instance segmentation behind one pull API:
/// <c>connect()</c> → <c>get()</c> → <c>disconnect()</c>.
///
/// Pull, not push: the camera never calls back into the consumer, so there is no
/// callback threading contract, no shared buffer to guard and no teardown race —
/// whoever wants a frame asks for one. The pylon driver keeps receiving in the
/// background; the "latest images" strategy with a one-deep output queue means
/// <c>get()</c> always returns the newest frame and stale ones are dropped, which is
/// what a live monitor wants.
///
/// Inference runs inline inside <c>get()</c>. That keeps pixels and detections in the
/// same <see cref="Frame"/> (they can never be mismatched) and spends GPU time only on
/// frames somebody actually reads.
///
/// Single-consumer: one thread calls <c>get()</c> in a loop. Fan-out to display,
/// logging or measurement is the caller's job.
/// </summary>
public sealed class VisionCam : IDisposable
{
    /// <summary>An enumerated camera. Keeps pylon's ICameraInfo out of the public API.</summary>
    public sealed record Device(string Serial, string Model)
    {
        public override string ToString() => $"{Model}  [{Serial}]";
    }

    private readonly float _fps;
    private readonly bool _segmentation;
    private readonly int _roiW;
    private readonly int _roiH;
    private readonly string _modelPath;
    private readonly int _gpuId;

    private Camera? _camera;
    private Yolo? _yolo;

    // Whatever the sensor delivers (Mono8, Bayer, YUV, ...) is normalised to packed
    // Mono8. Buffers are reused across frames; only the per-frame Frame.Gray is fresh.
    private readonly PixelDataConverter _toMono = new() { OutputPixelFormat = PixelType.Mono8 };
    private byte[] _mono = Array.Empty<byte>();
    private byte[] _bgra = Array.Empty<byte>();

    // Camera timestamp ticks per second; 0 = unknown, fall back to the host clock.
    private double _ticksPerSecond;
    private readonly Stopwatch _hostClock = new();

    /// <param name="fps">Target acquisition rate. Also caps exposure (which must fit
    /// inside the frame period) — with segmentation on, set this near the achievable
    /// inference rate: extra frames are transferred only to be discarded.</param>
    /// <param name="segmentation">Run YOLO on every grabbed frame.</param>
    /// <param name="roiWidth">Hardware ROI width; 0 = full sensor.</param>
    /// <param name="roiHeight">Hardware ROI height; 0 = full sensor.</param>
    /// <param name="modelPath">ONNX model; defaults to Models\yolo26s-seg.onnx beside the exe.</param>
    /// <param name="gpuId">DirectML device id, or -1 for CPU.</param>
    /// <remarks>The ROI defaults to the model's 640×640 input on purpose: grabbing at
    /// exactly the input size means no resize happens, so detection boxes come back in
    /// sensor pixel coordinates with no rescaling error. Widening the ROI reintroduces
    /// a resize — and with it a pixel-to-millimetre scale that must be recalibrated.</remarks>
    public VisionCam(float fps, bool segmentation,
                     int roiWidth = 640, int roiHeight = 640,
                     string? modelPath = null, int gpuId = 0)
    {
        if (fps <= 0) throw new ArgumentOutOfRangeException(nameof(fps), "fps must be > 0.");

        _fps = fps;
        _segmentation = segmentation;
        _roiW = roiWidth;
        _roiH = roiHeight;
        _gpuId = gpuId;
        _modelPath = modelPath ?? Path.Combine(AppContext.BaseDirectory, "Models", "yolo26s-seg.onnx");
    }

    /// <summary>Description of the connected device and inference backend.</summary>
    public string Info { get; private set; } = "not connected";

    /// <summary>True between a successful <see cref="connect"/> and <see cref="disconnect"/>.</summary>
    public bool IsConnected => _camera?.StreamGrabber?.IsGrabbing ?? false;

    /// <summary>Cameras visible to every installed transport layer.</summary>
    public static IReadOnlyList<Device> Enumerate() =>
        CameraFinder.Enumerate()
                    .Select(i => new Device(i[CameraInfoKey.SerialNumber] ?? "?",
                                            i[CameraInfoKey.ModelName] ?? "unknown"))
                    .ToList();

    /// <summary>Open the camera, apply all acquisition settings and start streaming.
    /// Pass a serial to pick a specific device, or null for the first one found.
    /// Loading the ONNX model blocks for a second or two on the first connect.</summary>
    public void connect(string? serial = null)
    {
        disconnect();

        ICameraInfo? info = null;
        if (serial != null)
        {
            info = CameraFinder.Enumerate().FirstOrDefault(i => i[CameraInfoKey.SerialNumber] == serial)
                   ?? throw new InvalidOperationException($"No camera with serial {serial}.");
        }

        var camera = info != null ? new Camera(info) : new Camera();
        _camera = camera;
        camera.CameraOpened += (s, e) => Configuration.AcquireContinuous(s!, e);
        camera.Open();

        ConfigureAcquisition();

        if (_segmentation)
            _yolo ??= CreateYolo();

        // "Latest images" with a one-deep output queue and two buffers is pylon's
        // documented way to say latest-image-only: anything the consumer is too slow
        // to read is discarded instead of queueing up staleness.
        Try(() => camera.Parameters[PLCameraInstance.MaxNumBuffer].SetValue(2));
        Try(() => camera.Parameters[PLCameraInstance.OutputQueueSize].SetValue(1));

        _hostClock.Restart();
        camera.StreamGrabber!.Start(GrabStrategy.LatestImages, GrabLoop.ProvidedByUser);

        var ci = camera.CameraInfo!;
        string backend = _segmentation ? (_gpuId >= 0 ? $" · GPU DirectML {_gpuId}" : " · CPU") : "";
        Info = $"{ci[CameraInfoKey.ModelName]} (SN {ci[CameraInfoKey.SerialNumber]}){backend}";
    }

    /// <summary>Fetch the newest frame, blocking until one arrives.
    /// Returns null on timeout — a stalled link is a "no signal" state for a live
    /// monitor, not a fatal error, so the caller's loop stays alive and can say so.
    /// Incomplete grabs are retried within the same time budget.</summary>
    public Frame? get(int timeoutMs = 1000)
    {
        var cam = _camera ?? throw new InvalidOperationException("connect() before get().");
        var grabber = cam.StreamGrabber!;
        if (!grabber.IsGrabbing) throw new InvalidOperationException("Not streaming — connect() first.");

        var budget = Stopwatch.StartNew();
        int remaining = timeoutMs;

        while (true)
        {
            using IGrabResult? result = grabber.RetrieveResult(remaining, TimeoutHandling.Return);
            if (result == null) return null;

            if (result.GrabSucceeded)
                return Compose(result);

            remaining = timeoutMs - (int)budget.ElapsedMilliseconds;
            if (remaining <= 0) return null;
        }
    }

    /// <summary>Stop streaming and release the camera. Idempotent. The loaded ONNX
    /// session is kept so a reconnect does not pay the model-load cost again;
    /// <see cref="Dispose"/> releases it.</summary>
    public void disconnect()
    {
        var camera = _camera;
        if (camera == null) return;

        try
        {
            if (camera.StreamGrabber!.IsGrabbing) camera.StreamGrabber.Stop();
            if (camera.IsOpen) camera.Close();
        }
        finally
        {
            camera.Dispose();
            _camera = null;
            Info = "not connected";
        }
    }

    public void Dispose()
    {
        disconnect();
        _yolo?.Dispose();
        _yolo = null;
        _toMono.Dispose();
    }

    // --- acquisition setup -------------------------------------------------

    /// <summary>Order matters: the ROI comes first because a smaller region is what
    /// frees the bandwidth the target frame rate needs, and exposure comes before the
    /// frame rate because a stale long exposure would otherwise cap it.</summary>
    private void ConfigureAcquisition()
    {
        var p = _camera!.Parameters;

        SetRoi(_roiW, _roiH);

        // Mono8 straight off the sensor: one byte per pixel, no debayering, and a
        // quarter of the bandwidth of BGRA. Colour cameras that refuse it are handled
        // by the converter in Compose() instead.
        Try(() => p[PLCamera.PixelFormat].SetValue("Mono8"));

        double periodUs = 1_000_000.0 / _fps;
        Try(() => p[PLCamera.ExposureAuto].SetValue("Off"));
        SetFloat(periodUs * 0.9, PLCamera.ExposureTime, PLCamera.ExposureTimeAbs);

        Try(() => p[PLCamera.AcquisitionFrameRateEnable].SetValue(true));
        SetFloat(_fps, PLCamera.AcquisitionFrameRate, PLCamera.AcquisitionFrameRateAbs);

        // Camera timestamps are tick counts; without the tick frequency they cannot be
        // turned into seconds, so fall back to the host clock in that case.
        _ticksPerSecond = 0;
        try { _ticksPerSecond = p[PLCamera.GevTimestampTickFrequency].GetValue(); }
        catch { /* not a GigE device, or the node is unavailable */ }
    }

    /// <summary>Apply the centered hardware ROI. A size of 0 means "full sensor" —
    /// which still writes the nodes, restoring the maximum, because the camera persists
    /// the ROI from the previous session and would otherwise keep the old crop.</summary>
    private void SetRoi(int targetW, int targetH)
    {
        var p = _camera!.Parameters;
        try
        {
            var wNode = p[PLCamera.Width];
            var hNode = p[PLCamera.Height];
            var oxNode = p[PLCamera.OffsetX];
            var oyNode = p[PLCamera.OffsetY];

            // Round down to the camera's increment, then clamp into range.
            static long Align(long v, long inc, long min, long max)
            {
                v = Math.Clamp(v, min, max);
                if (inc > 1) v -= (v - min) % inc;
                return v;
            }

            // Zero the offsets first, or the current offset caps the size we may ask for.
            Try(() => oxNode.SetValue(oxNode.GetMinimum()));
            Try(() => oyNode.SetValue(oyNode.GetMinimum()));

            bool full = targetW <= 0 || targetH <= 0;
            long reqW = full ? wNode.GetMaximum() : targetW;
            long reqH = full ? hNode.GetMaximum() : targetH;

            wNode.SetValue(Align(reqW, wNode.GetIncrement(), wNode.GetMinimum(), wNode.GetMaximum()));
            hNode.SetValue(Align(reqH, hNode.GetIncrement(), hNode.GetMinimum(), hNode.GetMaximum()));

            if (full) return; // at full size there is no travel left to center within

            // With the size set, OffsetX/Y max == sensor − size, i.e. the full travel;
            // half of it puts the window in the middle of the sensor.
            Try(() => oxNode.SetValue(Align(oxNode.GetMaximum() / 2, oxNode.GetIncrement(),
                                            oxNode.GetMinimum(), oxNode.GetMaximum())));
            Try(() => oyNode.SetValue(Align(oyNode.GetMaximum() / 2, oyNode.GetIncrement(),
                                            oyNode.GetMinimum(), oyNode.GetMaximum())));
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"ROI {targetW}×{targetH} could not be applied: {ex.Message}", ex);
        }
    }

    /// <summary>Write a float node, clamped to its range, trying the current SFNC name
    /// first and the legacy *Abs name second.</summary>
    private void SetFloat(double value, params FloatName[] keys)
    {
        var p = _camera!.Parameters;
        foreach (var key in keys)
        {
            try
            {
                var node = p[key];
                node.SetValue(Math.Clamp(value, node.GetMinimum(), node.GetMaximum()));
                return;
            }
            catch { /* try the next name */ }
        }
    }

    private static void Try(Action action)
    {
        try { action(); } catch { /* optional node: not all models expose it */ }
    }

    // --- per-frame work ----------------------------------------------------

    private Frame Compose(IGrabResult result)
    {
        int w = result.Width;
        int h = result.Height;
        int count = w * h;

        if (_mono.Length < count) _mono = new byte[count];
        _toMono.Convert(_mono, result);

        // Buffer.BlockCopy works on multidimensional arrays, so the row-major [h, w]
        // layout is filled in one memcpy.
        var gray = new byte[h, w];
        Buffer.BlockCopy(_mono, 0, gray, 0, count);

        double ts = _ticksPerSecond > 0
            ? result.Timestamp / _ticksPerSecond
            : _hostClock.Elapsed.TotalSeconds;

        IReadOnlyList<Instance> instances = _segmentation && _yolo != null
            ? Detect(w, h, count)
            : Array.Empty<Instance>();

        return new Frame((long)result.BlockID, result.SkippedImageCount, ts, w, h, gray, instances);
    }

    private IReadOnlyList<Instance> Detect(int w, int h, int count)
    {
        // YoloDotNet's preprocessing path expects a colour bitmap, so widen Mono8 to
        // BGRA. Roughly a megabyte of straight-line writes — negligible next to inference.
        int bytes = count * 4;
        if (_bgra.Length < bytes) _bgra = new byte[bytes];
        for (int i = 0, j = 0; i < count; i++, j += 4)
        {
            byte v = _mono[i];
            _bgra[j] = v;
            _bgra[j + 1] = v;
            _bgra[j + 2] = v;
            _bgra[j + 3] = 255;
        }

        var info = new SKImageInfo(w, h, SKColorType.Bgra8888, SKAlphaType.Unpremul);
        using var bitmap = new SKBitmap(info);
        Marshal.Copy(_bgra, 0, bitmap.GetPixels(), bytes);

        var results = _yolo!.RunSegmentation(bitmap, confidence: 0.24, pixelConfedence: 0.5, iou: 0.7);

        var instances = new List<Instance>(results.Count);
        foreach (var r in results)
        {
            var b = r.BoundingBox;
            instances.Add(new Instance(r.Label.Index, r.Label.Name, r.Confidence,
                                       new BBox(b.Left, b.Top, b.Width, b.Height)));
        }
        return instances;
    }

    private Yolo CreateYolo()
    {
        if (!File.Exists(_modelPath))
            throw new FileNotFoundException($"Segmentation model not found: {_modelPath}", _modelPath);

        return new Yolo(new YoloOptions
        {
            ExecutionProvider = new DirectMLExecutionProvider(_modelPath, _gpuId),
            // Proportional (letterbox), never stretched: stretching scales x and y
            // differently, which would make a pixel distance mean different real
            // distances depending on direction — fatal for a geometric measurement.
            // At the default ROI the frame already matches the input, so nothing resizes.
            ImageResize = ImageResize.Proportional,
            SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None),
        });
    }
}

// --- data contract ---------------------------------------------------------

/// <summary>
/// One immutable snapshot handed over by <see cref="VisionCam.get"/> — pixels and
/// detections for the same instant, so a measurement can never be paired with the
/// wrong frame. The arrays are freshly allocated per frame and owned by the caller.
/// </summary>
/// <param name="Seq">Camera-side frame counter. Jumps whenever the latest-only grab
/// strategy discards frames the consumer was too slow to read — normal, not an error.</param>
/// <param name="Skipped">Frames the driver skipped since the previous <c>get()</c>.</param>
/// <param name="TimestampSec">Exposure time of this frame, in seconds, on a monotonic
/// clock (the camera's own when its tick frequency is known, else a host stopwatch).
/// Always use this — never wall-clock time — to compute intervals: frames arrive
/// irregularly, so assuming a fixed period is wrong.</param>
/// <param name="Gray">Mono8 pixels as <c>[Height, Width]</c>, row-major.</param>
/// <param name="Instances">Detections, empty when segmentation is off.</param>
public sealed record Frame(
    long Seq,
    long Skipped,
    double TimestampSec,
    int Width,
    int Height,
    byte[,] Gray,
    IReadOnlyList<Instance> Instances);

/// <summary>A single detected object. Geometry is in sensor pixel coordinates — the
/// ROI is grabbed at the model's input size, so no resampling happens and the box
/// needs no back-mapping.</summary>
public sealed record Instance(int ClassId, string Label, double Confidence, BBox Box);

/// <summary>Axis-aligned box in pixel coordinates. Deliberately not SkiaSharp's
/// SKRectI, so the inference library stays behind <see cref="VisionCam"/>.</summary>
public readonly record struct BBox(int X, int Y, int Width, int Height)
{
    public int Left => X;
    public int Top => Y;
    public int Right => X + Width;
    public int Bottom => Y + Height;
    public double CenterX => X + Width / 2.0;
    public double CenterY => Y + Height / 2.0;
}

/// <summary>One CTWD reading, tagged with the frame it came from so a value can always
/// be traced back to the pixels it was measured on. The estimator that produces these
/// lands in this file once the tip/work classes are settled.</summary>
/// <param name="RateMmPerSec">Rate of change; positive = moving away. Must be derived
/// from <see cref="Frame.TimestampSec"/> differences, not from a frame count.</param>
public sealed record Measurement(long Seq, double TimestampSec,
                                 double CtwdMm, double RateMmPerSec, double Confidence);
