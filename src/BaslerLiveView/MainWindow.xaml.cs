using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Basler.Pylon;

namespace BaslerLiveView;

public partial class MainWindow : Window
{
    private readonly CameraService _camera = new();

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
        // Synchronous Invoke: keeps the grab thread paused until WPF has copied
        // the pixels out of the shared buffer, avoiding tearing/overwrite races.
        Dispatcher.Invoke(() => RenderFrame(width, height, bgra));
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
    }
}
