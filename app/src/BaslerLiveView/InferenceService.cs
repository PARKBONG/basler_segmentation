using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
/// Runs YOLO on incoming BGRA camera frames using YoloDotNet on the DirectML (GPU)
/// execution provider. The task follows the model file: an instance-segmentation model
/// draws masks, an oriented-bounding-box (OBB) model draws rotated boxes — both come out
/// of training/ (train.seg_all.yaml / train.obb_all.yaml), so switching is just pointing
/// config.xml &lt;Inference&gt;&lt;Model&gt; at the other .onnx.
///
/// Inference is far slower than the grab rate, so frames are processed on a private
/// worker thread with a "latest frame wins" policy: <see cref="Submit"/> only ever
/// keeps the newest frame and drops any backlog, so the grab loop and UI never block
/// waiting on the GPU. Each finished frame is published via <see cref="FrameProcessed"/>.
/// </summary>
public sealed class InferenceService : IDisposable
{
    private readonly Yolo _yolo;
    private readonly SegmentationDrawingOptions _drawing;
    private readonly DetectionDrawingOptions _obbDrawing;
    private readonly Thread _worker;
    private readonly AutoResetEvent _frameSignal = new(false);
    private readonly object _lock = new();

    // The rig has exactly one wire in view. YOLO26 can still emit a second, overlapping
    // instance of it (IoU below the NMS threshold) — keep only the most confident one.
    private const int MaxDetections = 1;

    // Latest submitted frame, owned by this service (copied out of the reused camera buffer).
    private byte[] _pending = Array.Empty<byte>();
    private int _pendingWidth;
    private int _pendingHeight;
    private (int X, int Y)? _pendingOrigin;
    private bool _hasPending;
    private volatile bool _running = true;

    // Worker-thread buffers, reused across frames: a crop is ~6.5 MB (1280×1280 BGRA), and
    // allocating that per frame lands on the Large Object Heap, which is only reclaimed by
    // full (gen2) GCs — at tens of frames per second those show up as stutters.
    private SKBitmap? _bitmap;
    private readonly byte[][] _outputs = { Array.Empty<byte>(), Array.Empty<byte>() };
    private int _nextOutput;

    /// <summary>Raised per processed frame: (width, height, annotatedBgraBuffer).
    /// Two output buffers alternate, so a buffer stays valid only until the next event —
    /// the receiver must drop its reference to the previous one by the time the handler
    /// returns (MainWindow swaps it under its overlay lock).</summary>
    public event Action<int, int, byte[]>? FrameProcessed;

    /// <summary>Detection text of the latest processed frame (class · score · wire tip),
    /// one item per line. Set on the worker thread right before <see cref="FrameProcessed"/>.</summary>
    public string Info { get; private set; } = "";

    /// <summary>Raised for inference failures on the worker thread.</summary>
    public event Action<Exception>? ErrorOccurred;

    /// <param name="modelPath">Path to the ONNX model (segmentation or OBB).</param>
    /// <param name="gpuId">DirectML device id (0 = default GPU, -1 = CPU fallback).</param>
    public InferenceService(string modelPath, int gpuId = 0)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model not found: {modelPath}", modelPath);

        // The execution backend is decided purely by gpuId in the provider:
        // >= 0 uses DirectML on that GPU device, < 0 runs on CPU.
        Backend = gpuId >= 0 ? $"GPU (DirectML dev {gpuId})" : "CPU";

        _yolo = new Yolo(new YoloOptions
        {
            ExecutionProvider = new DirectMLExecutionProvider(modelPath, gpuId),
            // Stretch to the model's input size; matches the segmentation demo defaults.
            ImageResize = ImageResize.Stretched,
            SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None),
        });

        // YoloDotNet reads the task from the ONNX metadata but only exposes it through
        // ModelInfo ("<ModelType> (yolo vNN)"), so parse the leading enum name back.
        ModelTask = Enum.TryParse(_yolo.ModelInfo.Split(' ')[0], out ModelType task)
            ? task
            : ModelType.Segmentation;
        if (ModelTask is not (ModelType.Segmentation or ModelType.ObbDetection))
        {
            _yolo.Dispose();
            throw new NotSupportedException(
                $"Unsupported model task {ModelTask} ({Path.GetFileName(modelPath)}) — " +
                "expected a segmentation or OBB model.");
        }

        // No axis-aligned box, a faint mask (32/255 ≈ 12%) under a thick outline, and the
        // wire tip as a red dot (Annotate()); class · score · tip text is shown by MainWindow.
        // An OBB model's rotated box is the prediction itself, so that one is drawn.
        _drawing = new SegmentationDrawingOptions
        {
            DrawBoundingBoxes = false,
            DrawLabels = false,
            DrawConfidenceScore = false,
            BorderThickness = 5,
            BoundingBoxOpacity = 128,
            EnableDynamicScaling = true,
            DrawSegmentationPixelMask = true,
            PixelMaskOpacity = 32,
            DrawContour = true,
            ContourThickness = 5,
        };
        _obbDrawing = (DetectionDrawingOptions)_drawing;
        _obbDrawing.DrawBoundingBoxes = true;

        _worker = new Thread(WorkerLoop)
        {
            IsBackground = true,
            Name = "YoloInference",
        };
        _worker.Start();
    }

    /// <summary>Execution backend in use: "GPU (DirectML dev N)" or "CPU".</summary>
    public string Backend { get; }

    /// <summary>The model's task: <see cref="ModelType.Segmentation"/> or
    /// <see cref="ModelType.ObbDetection"/>.</summary>
    public ModelType ModelTask { get; }

    /// <summary>Human-readable model description (type, version, input size, ...).</summary>
    public string ModelInfo => _yolo.ModelInfo ?? "unknown";

    /// <summary>
    /// Hand the newest frame to the inference worker. Copies the pixels out of the
    /// (reused) camera buffer immediately, so the caller may return at once. If a
    /// previous frame is still being processed it is simply overwritten.
    /// <paramref name="origin"/> is where this crop sits in the full sensor frame (null =
    /// not cropped); it is only used to also print the wire tip in full-frame pixels.
    /// </summary>
    public void Submit(int width, int height, byte[] bgra, (int X, int Y)? origin = null)
    {
        int needed = width * height * 4;
        lock (_lock)
        {
            if (_pending.Length < needed)
                _pending = new byte[needed];

            Buffer.BlockCopy(bgra, 0, _pending, 0, needed);
            _pendingWidth = width;
            _pendingHeight = height;
            _pendingOrigin = origin;
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

            int width, height;
            (int X, int Y)? origin;
            SKBitmap bitmap;
            lock (_lock)
            {
                if (!_hasPending) continue;
                _hasPending = false;
                width = _pendingWidth;
                height = _pendingHeight;
                origin = _pendingOrigin;
                if (_bitmap == null || _bitmap.Width != width || _bitmap.Height != height)
                {
                    // Camera frames are BGRA8 packed, which maps directly onto Skia's
                    // Bgra8888 memory layout — no channel conversion needed.
                    _bitmap?.Dispose();
                    _bitmap = new SKBitmap(new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Unpremul));
                }
                bitmap = _bitmap;
                Marshal.Copy(_pending, 0, bitmap.GetPixels(), width * height * 4);
            }

            try
            {
                byte[] annotated = RunInference(bitmap, origin);
                FrameProcessed?.Invoke(width, height, annotated);
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(ex);
            }
        }
    }

    private byte[] RunInference(SKBitmap bitmap, (int X, int Y)? origin)
    {
        // Draw() mutates the SKBitmap in place, overlaying masks / rotated boxes.
        var found = new List<(LabelModel Label, double Confidence, SKPoint? Tip)>();
        if (ModelTask == ModelType.ObbDetection)
        {
            var results = _yolo.RunObbDetection(bitmap, confidence: 0.24, iou: 0.7)
                .OrderByDescending(r => r.Confidence).Take(MaxDetections).ToList();
            bitmap.Draw(results, _obbDrawing);
            found.AddRange(results.Select(r => (r.Label, r.Confidence,
                (SKPoint?)WireTip.FromObb(r.BoundingBox, r.OrientationAngle))));
        }
        else
        {
            var results = _yolo.RunSegmentation(bitmap, confidence: 0.24, pixelConfedence: 0.5, iou: 0.7)
                .OrderByDescending(r => r.Confidence).Take(MaxDetections).ToList();
            bitmap.Draw(results, _drawing);
            found.AddRange(results.Select(r => (r.Label, r.Confidence,
                WireTip.FromMask(r.BitPackedPixelMask, r.BoundingBox))));
        }
        Info = Annotate(bitmap, found, origin);

        // Read the annotated pixels back out (still Bgra8888) into the output buffer the
        // receiver is not holding (see FrameProcessed).
        int size = bitmap.ByteCount;
        ref byte[] annotated = ref _outputs[_nextOutput];
        if (annotated.Length != size)
            annotated = new byte[size];
        Marshal.Copy(bitmap.GetPixels(), annotated, 0, size);
        _nextOutput ^= 1;
        return annotated;
    }

    /// <summary>
    /// Wire tip as a red dot on the crop, and per detection the lines
    /// "wire 0.75" / "crop (498.0, 334.8)" / "full (1082.0, 718.8)" — crop in crop pixels
    /// (only when cropped), full in sensor pixels. The text is not burnt into the crop
    /// (it came out too small once the crop is letterboxed); MainWindow shows it top-left
    /// of the view instead.
    /// </summary>
    private static string Annotate(SKBitmap bitmap, List<(LabelModel Label, double Confidence, SKPoint? Tip)> found,
        (int X, int Y)? origin)
    {
        using var canvas = new SKCanvas(bitmap);
        using var paint = new SKPaint { IsAntialias = true };
        float radius = Math.Max(6, bitmap.Width / 100f);

        var lines = new List<string>();
        foreach (var (label, confidence, tip) in found)
        {
            lines.Add(FormattableString.Invariant($"{label.Name} {confidence:0.00}"));
            if (tip is SKPoint p)
            {
                paint.Color = SKColors.White;
                canvas.DrawCircle(p, radius + Math.Max(2, radius / 3), paint);
                paint.Color = SKColors.Red;
                canvas.DrawCircle(p, radius, paint);
                var (ox, oy) = origin ?? (0, 0);
                if (origin is not null)
                    lines.Add(FormattableString.Invariant($"crop ({p.X:0.0}, {p.Y:0.0})"));
                lines.Add(FormattableString.Invariant($"full ({p.X + ox:0.0}, {p.Y + oy:0.0})"));
            }
        }
        if (lines.Count == 0)
            lines.Add("no detection");
        return string.Join(Environment.NewLine, lines);
    }

    public void Dispose()
    {
        _running = false;
        _frameSignal.Set();
        bool stopped = _worker.Join(TimeSpan.FromSeconds(2));
        _frameSignal.Dispose();
        _yolo.Dispose();
        if (stopped)
            _bitmap?.Dispose();   // still in use if the worker is stuck in inference
    }
}
