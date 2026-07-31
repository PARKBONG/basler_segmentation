using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Basler.Pylon;

namespace BaslerLiveView;

public partial class MainWindow : Window
{
    private readonly CameraService _camera = new();
    private readonly CameraConfig _config = CameraConfig.Load();

    // YOLO segmentation overlay (lazily created when first enabled).
    private SegmentationService? _seg;
    private volatile bool _segEnabled;
    // Model file name comes from config\config.xml (<Segmentation><Model>),
    // resolved under Models\ beside the exe. Falls back to the config default.
    private readonly string _modelPath;

    // The crop window feeds segmentation and mirrors what finetuner/preprocess.py
    // will cut the recorded PNGs down to. The live view is never cropped — it shows
    // the whole frame with this window outlined. Recording queues frames to a worker.
    private readonly FrameCropper _cropper = new();
    private readonly FrameRecorder _recorder;

    // Latest annotated crop from the segmentation worker, pasted back into the full
    // frame at the crop window's position. Written by the segmentation thread, read
    // by the grab thread → guarded by _overlayLock.
    private readonly object _overlayLock = new();
    private byte[] _overlay = Array.Empty<byte>();
    private int _overlayWidth;
    private int _overlayHeight;
    private bool _hasOverlay;

    // Scratch full-frame buffer used to composite the overlay; grab thread only.
    private byte[] _composed = Array.Empty<byte>();

    // Set once the crop controls hold real values; guards the ValueChanged/
    // TextChanged handlers that fire while we are still populating them.
    private bool _cropUiReady;

    private WriteableBitmap? _bitmap;
    private int _bmpWidth;
    private int _bmpHeight;

    // FPS measurement.
    private readonly Stopwatch _fpsClock = Stopwatch.StartNew();
    private int _frameCount;

    // Set once the window starts closing so late grab-thread frames stop
    // touching the (soon-to-be-gone) Dispatcher.
    private bool _closing;

    /// <summary>ComboBox row: wraps an ICameraInfo with a friendly label.</summary>
    private sealed record CameraItem(ICameraInfo Info, string DisplayName);

    public MainWindow()
    {
        InitializeComponent();

        _modelPath = Path.Combine(AppContext.BaseDirectory, "Models", _config.SegModel);
        _recorder = new FrameRecorder(
            FrameRecorder.ResolveDirectory(_config.RecordDir), _config.RecordQueueCapacity)
        {
            SaveFps = _config.RecordFps,
        };
        _recorder.ErrorOccurred += ex =>
            Dispatcher.BeginInvoke(() => StatusText.Text = "Recording error: " + ex.Message);

        // Seed the crop controls from config; the handler pushes them onto _cropper.
        CropCheck.IsChecked = _config.CropEnabled;
        CropWidthBox.Text = _config.CropWidth.ToString();
        CropHeightBox.Text = _config.CropHeight.ToString();
        CropXSlider.Value = _config.CropCenterX;
        CropYSlider.Value = _config.CropCenterY;
        _cropUiReady = true;
        ApplyCropSettings();

        _camera.StatusChanged += msg => Dispatcher.BeginInvoke(() => StatusText.Text = msg);
        _camera.ErrorOccurred += ex => Dispatcher.BeginInvoke(() => StatusText.Text = "Error: " + ex.Message);
        _camera.FrameReady += OnFrameReady;

        Loaded += (_, _) => RefreshCameras();
    }

    /// <summary>Short one-line description of the configured segmentation model:
    /// which file is expected (from config) and whether it is actually present.
    /// The real YOLO version is confirmed on the exe side once segmentation is
    /// toggled on (see <see cref="Segment_Toggled"/>).</summary>
    private string DescribeSegModel()
    {
        var file = Path.GetFileName(_modelPath);
        if (File.Exists(_modelPath))
        {
            var mb = new FileInfo(_modelPath).Length / (1024.0 * 1024.0);
            return $"Seg model: {file} ✓ ({mb:0.#} MB) — toggle Segmentation to load & confirm version";
        }
        return $"Seg model: {file} ✗ MISSING — place it in Models\\ (see config.xml)";
    }

    private void Refresh_Click(object sender, RoutedEventArgs e) => RefreshCameras();

    private void RefreshCameras()
    {
        try
        {
            CameraCombo.Items.Clear();
            foreach (var info in CameraService.Enumerate())
            {
                var model = info[CameraInfoKey.ModelName];
                var serial = info[CameraInfoKey.SerialNumber];
                CameraCombo.Items.Add(new CameraItem(info, $"{model}  [{serial}]"));
            }

            if (CameraCombo.Items.Count > 0)
            {
                CameraCombo.SelectedIndex = 0;
                StatusText.Text = $"Found {CameraCombo.Items.Count} camera(s).  |  {DescribeSegModel()}";
            }
            else
            {
                StatusText.Text = $"No cameras found. Connect a device, or relaunch with --emulate for the software emulator.  |  {DescribeSegModel()}";
            }
        }
        catch (Exception ex)
        {
            StatusText.Text = "Enumeration failed: " + ex.Message;
        }
    }

    private void Connect_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var item = CameraCombo.SelectedItem as CameraItem;
            _camera.Open(item?.Info);

            ConnectButton.IsEnabled = false;
            DisconnectButton.IsEnabled = true;
            StartButton.IsEnabled = true;
            CameraCombo.IsEnabled = false;
            RefreshButton.IsEnabled = false;
        }
        catch (Exception ex)
        {
            StatusText.Text = "Connect failed: " + ex.Message;
        }
    }

    private void Disconnect_Click(object sender, RoutedEventArgs e)
    {
        _camera.Close();
        ConnectButton.IsEnabled = true;
        DisconnectButton.IsEnabled = false;
        StartButton.IsEnabled = false;
        StopButton.IsEnabled = false;
        CameraCombo.IsEnabled = true;
        RefreshButton.IsEnabled = true;
    }

    private void Start_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            _fpsClock.Restart();
            _frameCount = 0;
            _camera.Start();
            StartButton.IsEnabled = false;
            StopButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            StatusText.Text = "Start failed: " + ex.Message;
        }
    }

    private void Stop_Click(object sender, RoutedEventArgs e)
    {
        _camera.Stop();
        StartButton.IsEnabled = true;
        StopButton.IsEnabled = false;
        FpsText.Text = "";
    }

    // --- Crop -------------------------------------------------------------

    private void Crop_Changed(object sender, RoutedEventArgs e) => ApplyCropSettings();

    private void CropText_Changed(object sender, TextChangedEventArgs e) => ApplyCropSettings();

    private void CropSlider_Changed(object sender, RoutedPropertyChangedEventArgs<double> e) => ApplyCropSettings();

    private void CropCenter_Click(object sender, RoutedEventArgs e)
    {
        CropXSlider.Value = 50;
        CropYSlider.Value = 50;
    }

    /// <summary>Push the toolbar's crop controls onto the cropper. Invalid or empty
    /// size text is simply ignored, so typing "6" on the way to "640" is harmless.
    /// 0 is accepted and means the full frame on that axis (preprocess.py convention).</summary>
    private void ApplyCropSettings()
    {
        if (!_cropUiReady) return;

        _cropper.Enabled = CropCheck.IsChecked == true;
        if (int.TryParse(CropWidthBox.Text, out int w) && w >= 0) _cropper.Width = w;
        if (int.TryParse(CropHeightBox.Text, out int h) && h >= 0) _cropper.Height = h;
        _cropper.CenterXPercent = CropXSlider.Value;
        _cropper.CenterYPercent = CropYSlider.Value;

        CropXText.Text = $"{CropXSlider.Value:F0}%";
        CropYText.Text = $"{CropYSlider.Value:F0}%";

        UpdateCropOverlay();
    }

    private void ViewHost_SizeChanged(object sender, SizeChangedEventArgs e) => UpdateCropOverlay();

    /// <summary>
    /// Draw the crop window over the live image. The image is shown with
    /// <c>Stretch="Uniform"</c>, so it is letterboxed inside the host: reproduce that
    /// fit here to map crop pixels onto screen coordinates. Hidden until a frame has
    /// arrived — the source resolution is unknown before that.
    /// </summary>
    private void UpdateCropOverlay()
    {
        int x = 0, y = 0, w = 0, h = 0;
        bool show = _bmpWidth > 0 && _bmpHeight > 0
                    && ViewHost.ActualWidth > 0 && ViewHost.ActualHeight > 0
                    && _cropper.TryGetWindow(_bmpWidth, _bmpHeight, out x, out y, out w, out h);

        var visibility = show ? Visibility.Visible : Visibility.Collapsed;
        CropRectShadow.Visibility = visibility;
        CropRectLine.Visibility = visibility;
        CropRectLabelBox.Visibility = visibility;
        if (!show) return;

        double scale = Math.Min(ViewHost.ActualWidth / _bmpWidth, ViewHost.ActualHeight / _bmpHeight);
        double originX = (ViewHost.ActualWidth - _bmpWidth * scale) / 2;
        double originY = (ViewHost.ActualHeight - _bmpHeight * scale) / 2;

        double left = originX + x * scale;
        double top = originY + y * scale;
        double rectW = w * scale;
        double rectH = h * scale;

        foreach (var rect in new[] { CropRectShadow, CropRectLine })
        {
            Canvas.SetLeft(rect, left);
            Canvas.SetTop(rect, top);
            rect.Width = rectW;
            rect.Height = rectH;
        }

        CropRectLabel.Text = $"{w}×{h}  ({CropXSlider.Value:F0}%, {CropYSlider.Value:F0}%)";
        Canvas.SetLeft(CropRectLabelBox, left);
        // Above the box normally; tucked inside when it would fall off the top edge.
        CropRectLabelBox.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
        double labelHeight = CropRectLabelBox.DesiredSize.Height;
        Canvas.SetTop(CropRectLabelBox, top - labelHeight - 2 >= 0 ? top - labelHeight - 2 : top + 2);
    }

    // --- Recording --------------------------------------------------------

    private void Record_Toggled(object sender, RoutedEventArgs e)
    {
        if (RecordButton.IsChecked == true)
        {
            try
            {
                _recorder.Start();
                var rate = _recorder.SaveFps > 0 ? $"{_recorder.SaveFps:0.##} fps" : "every frame";
                StatusText.Text = $"REC ({rate}) → {_recorder.SessionDirectory}";
            }
            catch (Exception ex)
            {
                RecordButton.IsChecked = false;
                StatusText.Text = "Recording start failed: " + ex.Message;
            }
        }
        else
        {
            _recorder.Stop();
            StatusText.Text = $"REC stopped — {_recorder.SavedCount + _recorder.PendingCount} frame(s) → {_recorder.SessionDirectory}";
        }
    }

    // Raised on the pylon grab-loop thread → marshal to UI, then blit.
    private void OnFrameReady(int width, int height, byte[] bgra)
    {
        // Once the window is closing the Dispatcher is going away; don't block
        // the grab thread on it (that races shutdown and crashes the process).
        if (_closing || Dispatcher.HasShutdownStarted)
            return;

        // Recording saves the frame as grabbed: training data is collected at full
        // sensor resolution and cropped later by finetuner/preprocess.py, which crops
        // the polygon labels along with the image. Cropping here would throw away
        // pixels that can never be recovered.
        // No-op unless recording and the save-rate sampler is due; copies the pixels
        // out before returning.
        _recorder.Submit(width, height, bgra);

        var seg = _seg;
        if (_segEnabled && seg != null)
        {
            // Segmentation runs on the crop only — the same region the training
            // images will be cut down to, so the model sees at inference exactly the
            // framing it was trained on. Use the toolbar sliders to find it, then put
            // the same numbers into preprocess_config.yaml: identical 0–100% convention.
            var cropped = _cropper.Crop(width, height, bgra, out int cw, out int ch);

            // Submit() copies the pixels out of the reused buffer immediately, so we
            // can return without blocking the grab loop; the annotated result arrives
            // later via OnSegmentedFrame and is pasted in by Compose() below.
            seg.Submit(cw, ch, cropped);
            bgra = Compose(width, height, bgra);
        }

        // Synchronous Invoke: keeps the grab thread paused until WPF has copied
        // the pixels out of the shared buffer, avoiding tearing/overwrite races.
        try
        {
            Dispatcher.Invoke(() => RenderFrame(width, height, bgra));
        }
        catch (OperationCanceledException)
        {
            // Dispatcher shut down between the check above and the Invoke.
        }
    }

    /// <summary>
    /// Copy the full frame into a scratch buffer and paste the newest annotated crop
    /// back over its own region, so the operator sees the whole sensor image with the
    /// segmentation drawn inside the crop window. Grab thread only.
    ///
    /// Inference lags the grab rate, so the pasted region is a few frames old — the
    /// same staleness the annotated view always had, now confined to the crop. If the
    /// crop size changed since the result was produced the overlay is skipped rather
    /// than stretched; the next inference brings it back a fraction of a second later.
    /// </summary>
    private byte[] Compose(int width, int height, byte[] bgra)
    {
        int needed = width * height * 4;
        if (_composed.Length < needed)
            _composed = new byte[needed];
        Buffer.BlockCopy(bgra, 0, _composed, 0, needed);

        lock (_overlayLock)
        {
            _cropper.TryGetWindow(width, height, out int x, out int y, out int cw, out int ch);
            if (_hasOverlay && cw == _overlayWidth && ch == _overlayHeight)
            {
                int srcStride = cw * 4;
                int dstStride = width * 4;
                for (int row = 0; row < ch; row++)
                    Buffer.BlockCopy(_overlay, row * srcStride, _composed, (y + row) * dstStride + x * 4, srcStride);
            }
        }

        return _composed;
    }

    // Raised on the segmentation worker thread with a fresh (owned) buffer. It is not
    // rendered directly: display is driven entirely by the grab loop, which pastes
    // this in at the crop window's position (see Compose).
    private void OnSegmentedFrame(int width, int height, byte[] bgra)
    {
        lock (_overlayLock)
        {
            _overlay = bgra;
            _overlayWidth = width;
            _overlayHeight = height;
            _hasOverlay = true;
        }
    }

    private void Segment_Toggled(object sender, RoutedEventArgs e)
    {
        if (SegmentCheck.IsChecked == true)
        {
            try
            {
                if (_seg == null)
                {
                    // Model load + DirectML session init blocks briefly (~1-2s).
                    _seg = new SegmentationService(_modelPath);
                    _seg.FrameProcessed += OnSegmentedFrame;
                    _seg.ErrorOccurred += ex =>
                        Dispatcher.BeginInvoke(() => StatusText.Text = "Segmentation error: " + ex.Message);
                }
                _segEnabled = true;
                StatusText.Text = $"Segmentation ON — {_seg.ModelInfo} · {_seg.Backend}  [{Path.GetFileName(_modelPath)}]";
            }
            catch (Exception ex)
            {
                _segEnabled = false;
                SegmentCheck.IsChecked = false;
                StatusText.Text = "Segmentation load failed: " + ex.Message;
            }
        }
        else
        {
            _segEnabled = false;
            // Drop the last annotated crop, or it would stay frozen on screen.
            lock (_overlayLock) _hasOverlay = false;
            StatusText.Text = "Segmentation OFF.";
        }
    }

    private void RenderFrame(int width, int height, byte[] bgra)
    {
        if (_bitmap == null || _bmpWidth != width || _bmpHeight != height)
        {
            _bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
            _bmpWidth = width;
            _bmpHeight = height;
            LiveImage.Source = _bitmap;

            // Only now is the source resolution known, so the crop rectangle can be
            // placed. Size is stable afterwards, so this stays off the per-frame path.
            UpdateCropOverlay();
        }

        _bitmap.WritePixels(new Int32Rect(0, 0, width, height), bgra, width * 4, 0);

        _frameCount++;
        if (_fpsClock.ElapsedMilliseconds >= 500)
        {
            double fps = _frameCount * 1000.0 / _fpsClock.ElapsedMilliseconds;
            FpsText.Text = $"{fps:F1} fps   {width}×{height}";
            _frameCount = 0;
            _fpsClock.Restart();

            // Piggy-backs on the fps tick so the counters cost nothing per frame.
            RecText.Text = _recorder.IsRecording || _recorder.PendingCount > 0
                ? $"● REC  {_recorder.SavedCount} saved" +
                  (_recorder.PendingCount > 0 ? $"  (+{_recorder.PendingCount} queued)" : "") +
                  (_recorder.DroppedCount > 0 ? $"  {_recorder.DroppedCount} dropped" : "")
                : "";
        }
    }

    // Closing via the window's X does an automatic Stop → Disconnect. The camera
    // teardown blocks until the grab loop drains, so we hold the close, run it
    // off the UI thread (keeping the Dispatcher free to release any in-flight
    // frame callback → no deadlock), then close for real.
    private async void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        if (_closing)
            return; // cleanup already ran on the first pass; let the window close.

        _closing = true;
        e.Cancel = true;                    // hold the close until teardown finishes
        _segEnabled = false;                // stop routing frames to the segmentation worker
        _camera.FrameReady -= OnFrameReady; // stop feeding frames to the UI
        _recorder.Stop();                   // stop accepting frames (queued ones still flush)

        try
        {
            // Stop()/Close() + segmentation worker teardown off the UI thread, so
            // the Dispatcher stays free to release any in-flight frame callback.
            // The recorder is disposed last so its queue drains to disk first.
            await Task.Run(() =>
            {
                _camera.Dispose();
                _seg?.Dispose();
                _recorder.Dispose();
            });
        }
        catch (Exception ex)
        {
            Debug.WriteLine("Shutdown teardown failed: " + ex);
        }

        Close(); // re-enters Window_Closing with _closing == true → window closes
    }
}
