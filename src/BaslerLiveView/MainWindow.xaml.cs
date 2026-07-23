using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Basler.Pylon;

namespace BaslerLiveView;

public partial class MainWindow : Window
{
    private readonly CameraService _camera = new();

    // YOLO segmentation overlay (lazily created when first enabled).
    private SegmentationService? _seg;
    private volatile bool _segEnabled;
    // Model file name comes from config\config.xml (<Segmentation><Model>),
    // resolved under Models\ beside the exe. Falls back to the config default.
    private readonly string _modelPath =
        Path.Combine(AppContext.BaseDirectory, "Models", CameraConfig.Load().SegModel);

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

    // Raised on the pylon grab-loop thread → marshal to UI, then blit.
    private void OnFrameReady(int width, int height, byte[] bgra)
    {
        // Once the window is closing the Dispatcher is going away; don't block
        // the grab thread on it (that races shutdown and crashes the process).
        if (_closing || Dispatcher.HasShutdownStarted)
            return;

        var seg = _seg;
        if (_segEnabled && seg != null)
        {
            // Hand the frame to the GPU worker. Submit() copies the pixels out of
            // the reused buffer immediately, so we can return without blocking the
            // grab loop; the annotated result arrives later via OnSegmentedFrame.
            seg.Submit(width, height, bgra);
            return;
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

    // Raised on the segmentation worker thread with a fresh (owned) buffer → safe to
    // marshal asynchronously.
    private void OnSegmentedFrame(int width, int height, byte[] bgra)
    {
        Dispatcher.BeginInvoke(() => RenderFrame(width, height, bgra));
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
        }

        _bitmap.WritePixels(new Int32Rect(0, 0, width, height), bgra, width * 4, 0);

        _frameCount++;
        if (_fpsClock.ElapsedMilliseconds >= 500)
        {
            double fps = _frameCount * 1000.0 / _fpsClock.ElapsedMilliseconds;
            FpsText.Text = $"{fps:F1} fps   {width}×{height}";
            _frameCount = 0;
            _fpsClock.Restart();
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

        try
        {
            // Stop()/Close() + segmentation worker teardown off the UI thread, so
            // the Dispatcher stays free to release any in-flight frame callback.
            await Task.Run(() =>
            {
                _camera.Dispose();
                _seg?.Dispose();
            });
        }
        catch (Exception ex)
        {
            Debug.WriteLine("Shutdown teardown failed: " + ex);
        }

        Close(); // re-enters Window_Closing with _closing == true → window closes
    }
}
