using System;
using System.Diagnostics;
using System.Threading.Tasks;
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
        // Once the window is closing the Dispatcher is going away; don't block
        // the grab thread on it (that races shutdown and crashes the process).
        if (_closing || Dispatcher.HasShutdownStarted)
            return;

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
        _camera.FrameReady -= OnFrameReady; // stop feeding frames to the UI

        try
        {
            await Task.Run(() => _camera.Dispose()); // Stop() + Close() off the UI thread
        }
        catch (Exception ex)
        {
            Debug.WriteLine("Shutdown teardown failed: " + ex);
        }

        Close(); // re-enters Window_Closing with _closing == true → window closes
    }
}
