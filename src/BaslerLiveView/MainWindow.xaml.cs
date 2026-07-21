using System;
using System.Diagnostics;
using System.IO;
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
    private readonly string _modelPath =
        Path.Combine(AppContext.BaseDirectory, "Models", "yolov11s-seg.onnx");

    private WriteableBitmap? _bitmap;
    private int _bmpWidth;
    private int _bmpHeight;

    // FPS measurement.
    private readonly Stopwatch _fpsClock = Stopwatch.StartNew();
    private int _frameCount;

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
                StatusText.Text = $"Found {CameraCombo.Items.Count} camera(s).";
            }
            else
            {
                StatusText.Text = "No cameras found. Connect a device, or relaunch with --emulate for the software emulator.";
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
        Dispatcher.Invoke(() => RenderFrame(width, height, bgra));
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
                StatusText.Text = "Segmentation ON — " + _seg.ModelInfo;
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

    private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        _camera.Dispose();
        _seg?.Dispose();
    }
}
