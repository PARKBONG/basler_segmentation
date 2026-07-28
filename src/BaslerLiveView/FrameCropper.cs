using System;

namespace BaslerLiveView;

/// <summary>
/// Crops grabbed BGRA frames down to a fixed pixel size (e.g. 640×640) — the
/// region the segmentation model runs on, and the region finetuner/preprocess.py
/// will later cut the recorded training images down to.
///
/// The live view is NOT cropped: MainWindow shows the whole sensor frame and
/// outlines this window on top of it (see <see cref="TryGetWindow"/>), so the
/// operator can see what falls outside the framing while aiming it.
///
/// The crop window is positioned per axis as 0–100%:
///   0 = flush against the left/top edge, 100 = flush against the right/bottom
///   edge, 50 = centered. Percent (not pixels) keeps the setting meaningful when
///   the sensor resolution changes.
///
/// The output buffer is reused across frames, exactly like CameraService's grab
/// buffer: <see cref="Crop"/> is only ever called from the single grab thread,
/// and every consumer copies out of it before the next frame arrives.
/// </summary>
public sealed class FrameCropper
{
    private byte[] _buffer = Array.Empty<byte>();

    /// <summary>When false, <see cref="Crop"/> passes the frame through untouched.</summary>
    public bool Enabled { get; set; }

    /// <summary>Requested crop size in pixels. Clamped to the source frame.</summary>
    public int Width { get; set; } = 640;
    public int Height { get; set; } = 640;

    /// <summary>Crop window position, 0–100% (see the class remarks).</summary>
    public double CenterXPercent { get; set; } = 50;
    public double CenterYPercent { get; set; } = 50;

    /// <summary>
    /// Resolve the crop window in source-image pixels. The window is always filled
    /// in — the whole frame when cropping is off or clamps to the full size — so
    /// callers that need to place the cropped pixels back into the source frame can
    /// use it unconditionally. The return value says whether it is a genuine
    /// sub-region, i.e. whether there is anything worth outlining on screen.
    /// </summary>
    public bool TryGetWindow(int srcWidth, int srcHeight, out int x, out int y, out int width, out int height)
    {
        x = 0;
        y = 0;
        width = srcWidth;
        height = srcHeight;

        if (!Enabled)
            return false;

        // A crop larger than the sensor image is clamped instead of rejected, so
        // a 640 setting still does something sensible on a smaller camera.
        int w = Math.Clamp(Width, 1, srcWidth);
        int h = Math.Clamp(Height, 1, srcHeight);
        if (w == srcWidth && h == srcHeight)
            return false;

        // 0% → offset 0, 100% → the window's right/bottom edge touches the frame's.
        x = (int)Math.Round((srcWidth - w) * Math.Clamp(CenterXPercent, 0, 100) / 100.0);
        y = (int)Math.Round((srcHeight - h) * Math.Clamp(CenterYPercent, 0, 100) / 100.0);
        width = w;
        height = h;
        return true;
    }

    /// <summary>
    /// Crop <paramref name="src"/> and report the resulting size. Returns the
    /// source buffer unchanged when cropping is off or would be a no-op.
    /// </summary>
    public byte[] Crop(int srcWidth, int srcHeight, byte[] src, out int width, out int height)
    {
        if (!TryGetWindow(srcWidth, srcHeight, out int offsetX, out int offsetY, out width, out height))
            return src;

        int dstStride = width * 4;             // BGRA = 4 bytes/pixel
        int srcStride = srcWidth * 4;
        int needed = dstStride * height;
        if (_buffer.Length < needed)
            _buffer = new byte[needed];

        for (int y = 0; y < height; y++)
        {
            int srcOffset = (offsetY + y) * srcStride + offsetX * 4;
            Buffer.BlockCopy(src, srcOffset, _buffer, y * dstStride, dstStride);
        }

        return _buffer;
    }
}
