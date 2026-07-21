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

            var cam = XDocument.Load(path).Root?.Element("Camera");
            if (cam == null) return cfg;

            var fps = (double?)cam.Element("FrameRate");
            if (fps is > 0) cfg.FrameRate = fps.Value;
        }
        catch
        {
            // Fall back to defaults on any parse error.
        }
        return cfg;
    }
}
