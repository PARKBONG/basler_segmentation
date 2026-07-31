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

    /// <summary>Run segmentation on a <see cref="CropWidth"/>×<see cref="CropHeight"/>
    /// window instead of the whole frame. The live view always shows the full frame
    /// and just outlines this window.</summary>
    public bool CropEnabled { get; set; } = true;

    /// <summary>Crop size in pixels (training capture size, e.g. 640 or 960).
    /// 0 means the full sensor extent on that axis — same convention as
    /// finetuner/preprocess.py's crop.width/height.</summary>
    public int CropWidth { get; set; } = 640;
    public int CropHeight { get; set; } = 640;

    /// <summary>Crop window position per axis, 0–100% (0 = left/top, 100 = right/bottom,
    /// 50 = centered). Adjustable live from the toolbar sliders.</summary>
    public double CropCenterX { get; set; } = 50;
    public double CropCenterY { get; set; } = 50;

    /// <summary>Base folder the Record button writes under (full sensor resolution,
    /// uncropped). Each REC session creates its own <c>kimm_YYMMDD_HHMMSS/images</c>
    /// folder inside it. Relative paths are anchored at the repo root (see
    /// <see cref="FrameRecorder.ResolveDirectory"/>). The default puts sessions
    /// beside the other raw sources finetuner/preprocess.py reads.</summary>
    public string RecordDir { get; set; } = "datasets/raw";

    /// <summary>How many frames per second Record saves. Consecutive grabs are nearly
    /// identical, so the grab stream is sampled down to something worth labelling.
    /// 0 saves every grabbed frame.</summary>
    public double RecordFps { get; set; } = 2;

    /// <summary>Frames buffered while the encoder catches up; beyond this the newest
    /// frames are dropped instead of blocking the grab loop.</summary>
    public int RecordQueueCapacity { get; set; } = 120;

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

            var crop = root.Element("Crop");
            if (crop != null)
            {
                var enabled = (bool?)crop.Element("Enabled");
                if (enabled.HasValue) cfg.CropEnabled = enabled.Value;

                // 0 is meaningful here (= full extent on that axis), so accept it too.
                var w = (int?)crop.Element("Width");
                if (w is >= 0) cfg.CropWidth = w.Value;

                var h = (int?)crop.Element("Height");
                if (h is >= 0) cfg.CropHeight = h.Value;

                var cx = (double?)crop.Element("CenterX");
                if (cx is >= 0 and <= 100) cfg.CropCenterX = cx.Value;

                var cy = (double?)crop.Element("CenterY");
                if (cy is >= 0 and <= 100) cfg.CropCenterY = cy.Value;
            }

            var rec = root.Element("Recording");
            if (rec != null)
            {
                var dir = (string?)rec.Element("Directory");
                if (!string.IsNullOrWhiteSpace(dir)) cfg.RecordDir = dir.Trim();

                // 0 is meaningful here (= save every frame), so accept it too.
                var saveFps = (double?)rec.Element("Fps");
                if (saveFps is >= 0) cfg.RecordFps = saveFps.Value;

                var cap = (int?)rec.Element("QueueCapacity");
                if (cap is > 0) cfg.RecordQueueCapacity = cap.Value;
            }
        }
        catch
        {
            // Fall back to defaults on any parse error.
        }
        return cfg;
    }
}
