using System;
using System.IO;
using System.Xml.Linq;

namespace BaslerLiveView;

/// <summary>
/// App-level camera configuration loaded from <c>config\config.xml</c> next to
/// the executable. A missing file or field falls back to the defaults here, so
/// the app still runs without a config present.
/// </summary>
public sealed class CameraConfig
{
    /// <summary>Target acquisition frame rate (fps). The camera's AcquisitionFrameRate
    /// is pinned to this, and exposure is capped to fit inside the frame period.</summary>
    public double FrameRate { get; set; } = 85;

    /// <summary>Center-crop width/height (pixels) applied as a hardware ROI on the
    /// sensor. The app reads only this centered region, so frames arrive already
    /// cropped (less bandwidth, exact size for the 640×640 seg model). Set either
    /// to 0 to disable cropping and use the sensor's full frame.</summary>
    public int Width { get; set; } = 640;
    public int Height { get; set; } = 640;

    /// <summary>Where the crop window sits within the sensor, as a percentage of the
    /// available travel (0–100). 50 = centered. X: 0 = hard left, 100 = hard right.
    /// Y: 0 = top, 100 = bottom — so a value below 50 nudges the crop upward.</summary>
    public double CenterX { get; set; } = 50;
    public double CenterY { get; set; } = 50;

    /// <summary>ONNX segmentation model file name, resolved under the <c>Models\</c>
    /// folder beside the exe. YoloDotNet auto-detects the model version from the
    /// file, so any supported YOLO seg model (e.g. yolo26s-seg.onnx) works.</summary>
    public string SegModel { get; set; } = "yolo26s-seg.onnx";

    /// <summary>Default config location: <c>config\config.xml</c> beside the exe.</summary>
    public static string DefaultPath =>
        Path.Combine(AppContext.BaseDirectory, "config", "config.xml");

    /// <summary>Load config from <paramref name="path"/> (defaults to <see cref="DefaultPath"/>).
    /// Never throws — any missing/invalid input yields defaults.</summary>
    public static CameraConfig Load(string? path = null)
    {
        path ??= DefaultPath;
        var cfg = new CameraConfig();
        try
        {
            if (!File.Exists(path)) return cfg;

            var root = XDocument.Load(path).Root;
            if (root == null) return cfg;

            var cam = root.Element("Camera");
            if (cam != null)
            {
                var fps = (double?)cam.Element("FrameRate");
                if (fps is > 0) cfg.FrameRate = fps.Value;

                var w = (int?)cam.Element("Width");
                if (w is >= 0) cfg.Width = w.Value;
                var h = (int?)cam.Element("Height");
                if (h is >= 0) cfg.Height = h.Value;

                var cx = (double?)cam.Element("CenterX");
                if (cx is >= 0 and <= 100) cfg.CenterX = cx.Value;
                var cy = (double?)cam.Element("CenterY");
                if (cy is >= 0 and <= 100) cfg.CenterY = cy.Value;
            }

            var model = (string?)root.Element("Segmentation")?.Element("Model");
            if (!string.IsNullOrWhiteSpace(model)) cfg.SegModel = model.Trim();
        }
        catch
        {
            // Fall back to defaults on any parse error.
        }
        return cfg;
    }
}
