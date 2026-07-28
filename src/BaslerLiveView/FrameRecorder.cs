using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using SkiaSharp;

namespace BaslerLiveView;

/// <summary>
/// Saves live frames to disk as PNGs for training-data capture.
///
/// PNG encoding is far slower than the grab rate, so <see cref="Submit"/> only
/// copies the pixels into a bounded queue and returns; a private worker thread
/// does the encoding and file I/O. If the disk cannot keep up the queue fills and
/// the newest frames are dropped (counted in <see cref="DroppedCount"/>) — the
/// grab loop and the UI are never blocked by recording.
///
/// Consecutive frames of a live scene are nearly identical, so saving all of them
/// mostly costs disk and labelling time. <see cref="SaveFps"/> samples the grab
/// stream down to a useful rate (2 fps by default) before anything is queued.
///
/// Frames are written exactly as the camera delivered them — full sensor resolution,
/// before the crop stage and never the segmentation overlay. Cropping to the training
/// size is finetuner/preprocess.py's job, because it must crop the polygon labels
/// along with the image; pixels discarded here could never be recovered.
/// </summary>
public sealed class FrameRecorder : IDisposable
{
    private sealed record Frame(int Width, int Height, byte[] Bgra, int Index);

    private readonly BlockingCollection<Frame> _queue;
    private readonly Thread _worker;

    // Sampling clock. Only ever read/written from the grab thread inside Submit()
    // (and reset in Start(), before any frame is accepted), so it needs no locking.
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private long _nextSaveTicks;

    private volatile bool _recording;
    private string _session = "";
    private int _accepted;   // frames the sampler let through this session (= file index)
    private int _saved;
    private int _dropped;

    /// <summary>Raised for encode/write failures on the worker thread.</summary>
    public event Action<Exception>? ErrorOccurred;

    /// <param name="directory">Destination folder (already resolved to an absolute path).</param>
    /// <param name="queueCapacity">Max frames buffered before new ones are dropped.</param>
    public FrameRecorder(string directory, int queueCapacity = 120)
    {
        Directory = directory;
        _queue = new BlockingCollection<Frame>(Math.Max(1, queueCapacity));
        _worker = new Thread(WorkerLoop) { IsBackground = true, Name = "FrameRecorder" };
        _worker.Start();
    }

    /// <summary>Absolute destination folder.</summary>
    public string Directory { get; }

    /// <summary>
    /// How many frames per second to keep. Grab-rate frames arriving between two
    /// slots are ignored outright — they are not counted as dropped, because
    /// skipping them is the point. 0 or less saves every frame.
    /// </summary>
    public double SaveFps { get; set; }

    public bool IsRecording => _recording;
    public int SavedCount => Volatile.Read(ref _saved);
    public int DroppedCount => Volatile.Read(ref _dropped);
    public int PendingCount => _queue.Count;

    /// <summary>
    /// Resolve the configured destination. Absolute paths are used as-is; relative
    /// ones are anchored at the repo root — the nearest ancestor of the exe holding
    /// <c>.git</c> — so the default <c>datasets/raw/images/train</c> lands exactly
    /// where finetuner/preprocess.py reads the <c>raw</c> source from.
    ///
    /// The marker is <c>.git</c> rather than any pipeline file on purpose: renaming
    /// or reorganising the finetuner scripts must not silently redirect recordings.
    /// </summary>
    public static string ResolveDirectory(string configured)
    {
        if (string.IsNullOrWhiteSpace(configured))
            configured = "datasets/raw/images/train";

        if (Path.IsPathRooted(configured))
            return Path.GetFullPath(configured);

        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            // A submodule's .git is a file, not a directory — accept either.
            var marker = Path.Combine(dir.FullName, ".git");
            if (System.IO.Directory.Exists(marker) || File.Exists(marker))
                return Path.GetFullPath(Path.Combine(dir.FullName, configured));
        }

        // Repo marker not found (e.g. the app was copied elsewhere) → beside the exe.
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, configured));
    }

    /// <summary>Begin a capture session. Creates the destination folder if needed.</summary>
    public void Start()
    {
        if (_recording) return;

        System.IO.Directory.CreateDirectory(Directory);
        // One timestamp per session → files from different sessions never collide,
        // and a plain name sort is also a chronological sort.
        _session = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        _nextSaveTicks = _clock.ElapsedTicks;   // first frame of a session is always kept
        Volatile.Write(ref _accepted, 0);
        Volatile.Write(ref _saved, 0);
        Volatile.Write(ref _dropped, 0);
        _recording = true;
    }

    /// <summary>Stop accepting frames. Already-queued frames are still written out.</summary>
    public void Stop() => _recording = false;

    /// <summary>
    /// Offer a frame to the recorder. A no-op unless recording, and unless the
    /// <see cref="SaveFps"/> sampler is due for one. The pixels are copied
    /// immediately, so the caller may reuse its buffer on return.
    /// </summary>
    public void Submit(int width, int height, byte[] bgra)
    {
        if (!_recording) return;

        double fps = SaveFps;
        if (fps > 0)
        {
            long now = _clock.ElapsedTicks;
            if (now < _nextSaveTicks) return;

            // Advance by whole periods so the average rate stays exact, but never
            // schedule into the past: after a stall that would fire a burst of
            // back-to-back saves to "catch up".
            long period = Math.Max(1, (long)(Stopwatch.Frequency / fps));
            _nextSaveTicks = Math.Max(now, _nextSaveTicks) + period;
        }

        int needed = width * height * 4;
        var copy = new byte[needed];
        Buffer.BlockCopy(bgra, 0, copy, 0, needed);

        // Index counts accepted frames, so a gap in the file numbering is a
        // visible record of frames the disk could not keep up with.
        int index = Interlocked.Increment(ref _accepted);
        if (!_queue.TryAdd(new Frame(width, height, copy, index)))
            Interlocked.Increment(ref _dropped);
    }

    private void WorkerLoop()
    {
        foreach (var frame in _queue.GetConsumingEnumerable())
        {
            try
            {
                Save(frame);
                Interlocked.Increment(ref _saved);
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(ex);
            }
        }
    }

    private void Save(Frame frame)
    {
        // Camera frames are BGRA8 packed → Skia's Bgra8888 layout, no conversion.
        var info = new SKImageInfo(frame.Width, frame.Height, SKColorType.Bgra8888, SKAlphaType.Opaque);
        using var bitmap = new SKBitmap(info);
        Marshal.Copy(frame.Bgra, 0, bitmap.GetPixels(), frame.Bgra.Length);

        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);

        var path = Path.Combine(Directory, $"cap_{_session}_{frame.Index:D5}.png");
        using var file = File.Create(path);
        data.SaveTo(file);
    }

    public void Dispose()
    {
        _recording = false;
        _queue.CompleteAdding();          // lets the worker drain, then exit
        _worker.Join(TimeSpan.FromSeconds(5));
        _queue.Dispose();
    }
}
