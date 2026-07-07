using System;
using System.Collections.Generic;
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

    /// <summary>Raised per grabbed frame: (width, height, bgraBuffer).
    /// The buffer is reused, so copy out of it before returning.</summary>
    public event Action<int, int, byte[]>? FrameReady;

    /// <summary>Human-readable status text.</summary>
    public event Action<string>? StatusChanged;

    /// <summary>Raised for failures on the grab thread.</summary>
    public event Action<Exception>? ErrorOccurred;

    public bool IsOpen => _camera?.IsOpen ?? false;
    public bool IsGrabbing => _camera?.StreamGrabber.IsGrabbing ?? false;

    /// <summary>Enumerate all cameras visible to every installed transport layer.</summary>
    public static List<ICameraInfo> Enumerate() => CameraFinder.Enumerate();

    /// <summary>Open a camera. Pass null to open the first available device.</summary>
    public void Open(ICameraInfo? info = null)
    {
        Close();

        _camera = info != null ? new Camera(info) : new Camera();

        // Registers the standard "acquire continuous" node-map setup, applied
        // automatically once the camera opens.
        _camera.CameraOpened += Configuration.AcquireContinuous;
        _camera.Open();

        var model = _camera.CameraInfo[CameraInfoKey.ModelName];
        var serial = _camera.CameraInfo[CameraInfoKey.SerialNumber];
        StatusChanged?.Invoke($"Connected: {model} (SN {serial})");
    }

    public void Start()
    {
        if (_camera == null)
            throw new InvalidOperationException("Open a camera before starting the grab.");

        _camera.StreamGrabber.ImageGrabbed += OnImageGrabbed;

        // LatestImages = keep only the freshest frames, drop backlog under load.
        // ProvidedByStreamGrabber = pylon runs its own grab-loop thread and
        // raises ImageGrabbed for us (no manual RetrieveResult loop needed).
        _camera.StreamGrabber.Start(GrabStrategy.LatestImages, GrabLoop.ProvidedByStreamGrabber);
        StatusChanged?.Invoke("Grabbing…");
    }

    public void Stop()
    {
        if (_camera == null) return;

        if (_camera.StreamGrabber.IsGrabbing)
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

            FrameReady?.Invoke(width, height, _buffer);
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(ex);
        }
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
