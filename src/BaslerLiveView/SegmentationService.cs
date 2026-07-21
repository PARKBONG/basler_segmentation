using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.ExecutionProvider.DirectML;
using YoloDotNet.Models;

namespace BaslerLiveView;

/// <summary>
/// Runs YOLO instance segmentation on incoming BGRA camera frames using YoloDotNet
/// on the DirectML (GPU) execution provider.
///
/// Inference is far slower than the grab rate, so frames are processed on a private
/// worker thread with a "latest frame wins" policy: <see cref="Submit"/> only ever
/// keeps the newest frame and drops any backlog, so the grab loop and UI never block
/// waiting on the GPU. Each finished frame is published via <see cref="FrameProcessed"/>.
/// </summary>
public sealed class SegmentationService : IDisposable
{
    private readonly Yolo _yolo;
    private readonly SegmentationDrawingOptions _drawing;
    private readonly Thread _worker;
    private readonly AutoResetEvent _frameSignal = new(false);
    private readonly object _lock = new();

    // Latest submitted frame, owned by this service (copied out of the reused camera buffer).
    private byte[] _pending = Array.Empty<byte>();
    private int _pendingWidth;
    private int _pendingHeight;
    private bool _hasPending;
    private volatile bool _running = true;

    /// <summary>Raised per processed frame: (width, height, annotatedBgraBuffer).
    /// The buffer belongs to the caller after the event (a fresh array per frame).</summary>
    public event Action<int, int, byte[]>? FrameProcessed;

    /// <summary>Raised for inference failures on the worker thread.</summary>
    public event Action<Exception>? ErrorOccurred;

    /// <param name="modelPath">Path to the ONNX segmentation model.</param>
    /// <param name="gpuId">DirectML device id (0 = default GPU, -1 = CPU fallback).</param>
    public SegmentationService(string modelPath, int gpuId = 0)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Segmentation model not found: {modelPath}", modelPath);

        _yolo = new Yolo(new YoloOptions
        {
            ExecutionProvider = new DirectMLExecutionProvider(modelPath, gpuId),
            // Stretch to the model's input size; matches the segmentation demo defaults.
            ImageResize = ImageResize.Stretched,
            SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None),
        });

        _drawing = new SegmentationDrawingOptions
        {
            DrawBoundingBoxes = true,
            DrawLabels = true,
            DrawConfidenceScore = true,
            DrawLabelBackground = true,
            EnableFontShadow = true,
            Font = SKTypeface.Default,
            FontSize = 18,
            FontColor = SKColors.White,
            BorderThickness = 2,
            BoundingBoxOpacity = 128,
            EnableDynamicScaling = true,
            DrawSegmentationPixelMask = true,
        };

        _worker = new Thread(WorkerLoop)
        {
            IsBackground = true,
            Name = "YoloSegmentation",
        };
        _worker.Start();
    }

    /// <summary>Human-readable model description (type, version, input size, ...).</summary>
    public string ModelInfo => _yolo.ModelInfo.ToString() ?? "unknown";

    /// <summary>
    /// Hand the newest frame to the inference worker. Copies the pixels out of the
    /// (reused) camera buffer immediately, so the caller may return at once. If a
    /// previous frame is still being processed it is simply overwritten.
    /// </summary>
    public void Submit(int width, int height, byte[] bgra)
    {
        int needed = width * height * 4;
        lock (_lock)
        {
            if (_pending.Length < needed)
                _pending = new byte[needed];

            Buffer.BlockCopy(bgra, 0, _pending, 0, needed);
            _pendingWidth = width;
            _pendingHeight = height;
            _hasPending = true;
        }
        _frameSignal.Set();
    }

    private void WorkerLoop()
    {
        while (_running)
        {
            _frameSignal.WaitOne();
            if (!_running) return;

            int width, height, needed;
            byte[] frame;
            lock (_lock)
            {
                if (!_hasPending) continue;
                _hasPending = false;
                width = _pendingWidth;
                height = _pendingHeight;
                needed = width * height * 4;
                frame = new byte[needed];
                Buffer.BlockCopy(_pending, 0, frame, 0, needed);
            }

            try
            {
                byte[] annotated = RunInference(width, height, frame);
                FrameProcessed?.Invoke(width, height, annotated);
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(ex);
            }
        }
    }

    private byte[] RunInference(int width, int height, byte[] bgra)
    {
        // Camera frames are BGRA8 packed, which maps directly onto Skia's Bgra8888
        // memory layout — no channel conversion needed. Alpha is fully opaque.
        var info = new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Unpremul);
        using var bitmap = new SKBitmap(info);
        Marshal.Copy(bgra, 0, bitmap.GetPixels(), bgra.Length);

        var results = _yolo.RunSegmentation(bitmap, confidence: 0.24, pixelConfedence: 0.5, iou: 0.7);

        // Draw() mutates the SKBitmap in place, overlaying masks/boxes/labels.
        bitmap.Draw(results, _drawing);

        // Read the annotated pixels back out (still Bgra8888) for the WriteableBitmap.
        var annotated = new byte[bgra.Length];
        Marshal.Copy(bitmap.GetPixels(), annotated, 0, annotated.Length);
        return annotated;
    }

    public void Dispose()
    {
        _running = false;
        _frameSignal.Set();
        _worker.Join(TimeSpan.FromSeconds(2));
        _frameSignal.Dispose();
        _yolo.Dispose();
    }
}
