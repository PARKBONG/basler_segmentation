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
