using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace BaslerLiveView;

/// <summary>
/// The application layer: it owns the one pump loop that reads <see cref="VisionCam"/>
/// and hands each <see cref="Frame"/> to the display. The camera class knows nothing
/// about WPF; pulling from a single place also means future consumers (measurement,
/// logging) can be fed from this same loop without competing for frames.
/// </summary>
public partial class MainWindow : Window
{
    // Acquisition settings live here, not in the UI: they are constructor arguments of
    // VisionCam and cannot change while streaming, so a control for them would only be
    // a control for "disconnect, edit, reconnect".
    private const float Fps = 20;
    private const bool Segmentation = true;
    private const int RoiWidth = 0;   // matches the model input → no resize, boxes are sensor pixels
    private const int RoiHeight = 0; // 0 = full sensor, otherwise the model input size (e.g. 640×480)
    private const double RoiXPercent = 50;  // 0 = left edge, 50 = centered, 100 = right edge
    private const double RoiYPercent = 50;  // 0 = top edge, 50 = centered, 100 = bottom edge
    private const string? PixelFormat = null;   // null = whatever the camera defaults to

    private VisionCam? _cam;

    private CancellationTokenSource? _pumpCts;
    private Task? _pumpTask;

    private WriteableBitmap? _bitmap;
    private int _bmpW;
    private int _bmpH;

    // Display-rate measurement, plus drop accounting straight from the camera.
    private readonly Stopwatch _fpsClock = Stopwatch.StartNew();
    private int _frames;
    private long _dropped;

    public MainWindow() => InitializeComponent();

    private void Connect_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            _cam = new VisionCam(Fps, Segmentation, RoiWidth, RoiHeight,
                                 RoiXPercent, RoiYPercent, PixelFormat);
            // Model load + DirectML init blocks for a second or two on the first connect.
            _cam.connect();

            StatusText.Text = "Streaming — " + _cam.Info;
            ConnectButton.IsEnabled = false;
            DisconnectButton.IsEnabled = true;
            StartPump();
        }
        catch (Exception ex)
        {
            _cam?.Dispose();
            _cam = null;
            StatusText.Text = "Connect failed: " + ex.Message;
        }
    }

    private async void Disconnect_Click(object sender, RoutedEventArgs e) => await ShutdownAsync();

    // --- pump --------------------------------------------------------------

    private void StartPump()
    {
        _frames = 0;
        _dropped = 0;
        _fpsClock.Restart();

        _pumpCts = new CancellationTokenSource();
        var ct = _pumpCts.Token;
        var cam = _cam!;

        _pumpTask = Task.Run(() =>
        {
            while (!ct.IsCancellationRequested)
            {
                Frame? frame;
                try
                {
                    // Blocking is fine here — this is not the UI thread. A timeout is a
                    // "no signal" state, not a failure, so the loop survives it.
                    frame = cam.get(timeoutMs: 500);
                }
                catch (Exception ex)
                {
                    Dispatcher.BeginInvoke(() => StatusText.Text = "Grab failed: " + ex.Message);
                    return;
                }

                if (ct.IsCancellationRequested) return;

                if (frame == null)
                {
                    Dispatcher.Invoke(() => NoSignalText.Visibility = Visibility.Visible);
                    continue;
                }

                // Synchronous Invoke gives the loop back-pressure for free — the next
                // get() only starts once WPF has finished drawing this frame.
                try
                {
                    Dispatcher.Invoke(() => Render(frame));
                }
                catch (OperationCanceledException)
                {
                    return; // dispatcher shut down mid-frame
                }
            }
        }, ct);
    }

    // --- rendering ---------------------------------------------------------

    private void Render(Frame frame)
    {
        NoSignalText.Visibility = Visibility.Collapsed;

        if (_bitmap == null || _bmpW != frame.Width || _bmpH != frame.Height)
        {
            // Gray8 maps one byte per pixel straight from Frame.Gray — no conversion.
            _bitmap = new WriteableBitmap(frame.Width, frame.Height, 96, 96, PixelFormats.Gray8, null);
            _bmpW = frame.Width;
            _bmpH = frame.Height;
            LiveImage.Source = _bitmap;
            Overlay.Width = frame.Width;
            Overlay.Height = frame.Height;
        }

        // WritePixels takes any rank-1 or rank-2 primitive array, so the [h, w] frame
        // goes in without flattening; stride is one byte per pixel.
        _bitmap.WritePixels(new Int32Rect(0, 0, frame.Width, frame.Height),
                            frame.Gray, frame.Width, 0);

        _dropped += frame.Skipped;
        _frames++;
        if (_fpsClock.ElapsedMilliseconds >= 500)
        {
            double fps = _frames * 1000.0 / _fpsClock.ElapsedMilliseconds;
            RateText.Text = $"{fps:F1} fps   {frame.Width}×{frame.Height}   dropped {_dropped}";
            _frames = 0;
            _fpsClock.Restart();
        }
    }

    // --- teardown ----------------------------------------------------------

    /// <summary>Stop the pump before touching the camera: cancel, wait for the in-flight
    /// get() to return, only then dispose. Reversing that order would leave get()
    /// blocking on a closed device.</summary>
    private async Task ShutdownAsync()
    {
        if (_pumpCts != null)
        {
            _pumpCts.Cancel();
            if (_pumpTask != null)
            {
                try { await _pumpTask; }
                catch (OperationCanceledException) { /* expected */ }
            }
            _pumpCts.Dispose();
            _pumpCts = null;
            _pumpTask = null;
        }

        if (_cam != null)
        {
            await Task.Run(() => _cam.Dispose());
            _cam = null;
        }

        ConnectButton.IsEnabled = true;
        DisconnectButton.IsEnabled = false;
        NoSignalText.Visibility = Visibility.Collapsed;
        StatusText.Text = "Disconnected.";
        RateText.Text = "";
        Overlay.Children.Clear();
    }

    private async void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        if (_cam == null && _pumpTask == null) return; // already clean; let it close

        e.Cancel = true;
        await ShutdownAsync();
        Close();
    }
}
