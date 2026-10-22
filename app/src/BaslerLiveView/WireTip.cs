using System;
using System.Collections.Generic;
using SkiaSharp;

namespace BaslerLiveView;

/// <summary>
/// Wire tip = midpoint of the bottom edge of the wire's oriented bounding box (OBB) —
/// the same definition as training/render.py, so the app and the Python tools report
/// the same point.
///
/// For a segmentation model the OBB is the minimum-area rotated rectangle around the
/// mask (what cv2.minAreaRect does in Python): take each mask row's leftmost and
/// rightmost pixel, build the convex hull of those, and try one rectangle per hull edge
/// (rotating calipers). The bottom edge is whichever of the two short sides lies lower
/// in the image. A ~600-row wire mask costs well under a millisecond.
/// </summary>
public static class WireTip
{
    /// <summary>Tip from a YoloDotNet bit-packed mask (box-relative, row-major, LSB-first),
    /// in image pixels. Null when the mask is empty.</summary>
    public static SKPoint? FromMask(byte[] packedMask, SKRectI box)
    {
        int w = box.Width, h = box.Height;
        if (packedMask.Length == 0 || w <= 0 || h <= 0)
            return null;

        // Row extents → pixel-edge corners (a pixel x spans [x, x+1]).
        var points = new List<SKPoint>(h * 4);
        for (int y = 0; y < h; y++)
        {
            int first = -1, last = -1;
            for (int x = 0, i = y * w; x < w; x++, i++)
            {
                if ((packedMask[i >> 3] & (1 << (i & 7))) == 0)
                    continue;
                if (first < 0)
                    first = x;
                last = x;
            }
            if (first < 0)
                continue;
            float top = box.Top + y, bottom = top + 1;
            points.Add(new SKPoint(box.Left + first, top));
            points.Add(new SKPoint(box.Left + first, bottom));
            points.Add(new SKPoint(box.Left + last + 1, top));
            points.Add(new SKPoint(box.Left + last + 1, bottom));
        }
        if (points.Count == 0)
            return null;

        return BottomMid(MinAreaRect(ConvexHull(points)));
    }

    /// <summary>Tip of an OBB model's rotated box (YoloDotNet: axis-aligned rect + angle in
    /// radians about its centre, same corner formula as its YOLO-format export).</summary>
    public static SKPoint FromObb(SKRectI rect, float angleRadians)
    {
        float cx = rect.MidX, cy = rect.MidY, hw = rect.Width / 2f, hh = rect.Height / 2f;
        float cos = MathF.Cos(angleRadians), sin = MathF.Sin(angleRadians);
        return BottomMid(new SKPoint[]
        {
            new(cx - hw * cos + hh * sin, cy - hw * sin - hh * cos),
            new(cx + hw * cos + hh * sin, cy + hw * sin - hh * cos),
            new(cx + hw * cos - hh * sin, cy + hw * sin + hh * cos),
            new(cx - hw * cos - hh * sin, cy - hw * sin + hh * cos),
        });
    }

    /// <summary>Midpoint of the lower of the two short edges of a rectangle (corners in order).</summary>
    public static SKPoint BottomMid(IReadOnlyList<SKPoint> c)
    {
        static float Len(SKPoint a, SKPoint b) => SKPoint.Distance(a, b);
        static SKPoint Mid(SKPoint a, SKPoint b) => new((a.X + b.X) / 2, (a.Y + b.Y) / 2);

        // Edge i runs corner i → i+1; edges 0/2 and 1/3 are opposite pairs.
        bool evenShort = Len(c[0], c[1]) + Len(c[2], c[3]) <= Len(c[1], c[2]) + Len(c[3], c[0]);
        SKPoint a = evenShort ? Mid(c[0], c[1]) : Mid(c[1], c[2]);
        SKPoint b = evenShort ? Mid(c[2], c[3]) : Mid(c[3], c[0]);
        return a.Y >= b.Y ? a : b;
    }

    // Andrew's monotone chain; returns the hull counter-clockwise (in y-down image space
    // the orientation does not matter for the calipers below).
    private static List<SKPoint> ConvexHull(List<SKPoint> pts)
    {
        pts.Sort((p, q) => p.X != q.X ? p.X.CompareTo(q.X) : p.Y.CompareTo(q.Y));
        static float Cross(SKPoint o, SKPoint a, SKPoint b) =>
            (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);

        var hull = new List<SKPoint>(pts.Count);
        for (int pass = 0; pass < 2; pass++)
        {
            int start = hull.Count;
            for (int k = 0; k < pts.Count; k++)
            {
                var p = pts[pass == 0 ? k : pts.Count - 1 - k];
                while (hull.Count >= start + 2 && Cross(hull[^2], hull[^1], p) <= 0)
                    hull.RemoveAt(hull.Count - 1);
                hull.Add(p);
            }
            hull.RemoveAt(hull.Count - 1);   // last point repeats as the next chain's first
        }
        return hull;
    }

    // Smallest-area rectangle with one side flush against a hull edge; corners in order.
    private static SKPoint[] MinAreaRect(List<SKPoint> hull)
    {
        if (hull.Count == 1)
            return new[] { hull[0], hull[0], hull[0], hull[0] };

        float bestArea = float.MaxValue;
        var best = new SKPoint[4];
        for (int i = 0; i < hull.Count; i++)
        {
            SKPoint e = hull[(i + 1) % hull.Count] - hull[i];
            float len = e.Length;
            if (len < 1e-6f)
                continue;
            var u = new SKPoint(e.X / len, e.Y / len);   // along the edge
            var v = new SKPoint(-u.Y, u.X);              // perpendicular

            float minU = float.MaxValue, maxU = float.MinValue, minV = float.MaxValue, maxV = float.MinValue;
            foreach (var p in hull)
            {
                float pu = p.X * u.X + p.Y * u.Y, pv = p.X * v.X + p.Y * v.Y;
                minU = Math.Min(minU, pu);
                maxU = Math.Max(maxU, pu);
                minV = Math.Min(minV, pv);
                maxV = Math.Max(maxV, pv);
            }

            float area = (maxU - minU) * (maxV - minV);
            if (area >= bestArea)
                continue;
            bestArea = area;
            SKPoint At(float a, float b) => new(u.X * a + v.X * b, u.Y * a + v.Y * b);
            best[0] = At(minU, minV);
            best[1] = At(maxU, minV);
            best[2] = At(maxU, maxV);
            best[3] = At(minU, maxV);
        }
        return best;
    }
}
