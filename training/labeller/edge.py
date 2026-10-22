"""
와이어 끝(아랫변) 경계 찾기 — "compute edge 1" (라벨러 버튼 이름. 다른 방식은 2, 3 … 으로).
OBB 의 윗변·옆변은 정답으로 두고, 끝 경계만 영상에서 찾는다.

OBB 를 와이어 축이 세로가 되게 펴서(affine — 열 = 두께 방향, 행 = 축 방향) 살짝 블러한 뒤,
가운데 열들(= 축과 나란한 얇은 띠들)마다 위에서부터 내려가며 밝기가 "와이어 밝기와 그 아래
최대 밝기의 중간(50%)" 을 처음 넘는 곳(서브픽셀)을 그 띠의 끝으로 삼는다 — 블러된 계단
엣지의 표준 위치. 그 점들에 2차 곡선 s = a·t² + b·t + c 를 튀는 점에 강하게(반복 재가중)
맞춘다. 결과 폴리곤 = OBB 윗변 + 양 옆변 + 끝 곡선 — seg 라벨로 그대로 쓸 수 있고 OBB 도 딸 수 있다.

고정 문턱(와이어보다 +N 밝아지면 끝)을 쓰지 않는 이유: 아크 빛이 와이어 끝을 감싸는 헤일로
때문에 와이어가 회색으로 서서히 밝아지다 하얘지므로, 고정 문턱은 헤일로 초입에서 너무 일찍
걸린다. 50% 는 헤일로(회색)를 지나 하얘지는 경계에 걸린다. 최대 그래디언트(Sobel)도 거의 같은
자리를 잡지만 띠마다 조금 더 들쭉날쭉하고 다른 강한 엣지로 튈 수 있어 "처음 넘는 곳" 을 쓴다.
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import median

from PIL import Image, ImageFilter

COLUMN_SPAN = (0.2, 0.8)   # 쓰는 열 — 두께의 가운데 60% (가장자리 띠는 휜 와이어 탓에 배경이 섞임)
COLUMN_STEP = 2            # 띠 간격(px)
BLUR = 2                   # 펼친 띠 영상의 가우시안 블러 반경(px) — 잡음·질감 억제
REF_PX = (8, 60)           # 와이어 밝기 기준 구간 (피벗에서 축 방향 px) — 튜브 출구 바로 아래는
                           # 항상 와이어. OBB 길이에 비례시키면 짧은 와이어에서 아크가 섞인다
MIN_STEP = 20              # 와이어 → 최대 밝기 차가 이보다 작으면(0~255) 그 띠엔 끝이 없다고 봄
CURVE_POINTS = 12          # 폴리곤에 넣을 끝 곡선 점 수
OK_INLIERS = 0.6           # 띠 중 이 비율 이상이 곡선에 맞고
OK_RMS = 3.0               # 잔차 RMS 가 이 px 이하면 "ok", 아니면 "low"


def obb_pose(corners: list) -> dict:
    """
    회전 사각형 꼭짓점 4개(인접 순서, 픽셀) → 와이어 자세 {pivot, angle, length, thickness}.

    긴 변 = 와이어 축(length), 짧은 변 = 두께. 축은 아래(y+)를 향하게 맞추고(수평이면
    오른쪽), 축의 위쪽 끝 두 꼭짓점 중 오른쪽(r = (uy, −ux) 쪽)이 피벗 — "우측 상단
    꼭짓점", 즉 와이어가 튜브에서 나오는 쪽 모서리입니다. 라벨러는 이 점을 축으로
    옮기고 돌립니다. index.html 의 poseFromPoints 와 같은 규칙.
    """
    p = [(float(x), float(y)) for x, y in corners]
    e01 = (p[1][0] - p[0][0], p[1][1] - p[0][1])
    e12 = (p[2][0] - p[1][0], p[2][1] - p[1][1])
    l01, l12 = math.hypot(*e01), math.hypot(*e12)
    (ax, ay), length, thickness = (e01, l01, l12) if l01 >= l12 else (e12, l12, l01)
    if length <= 0:
        raise ValueError("퇴화한 사각형 (길이 0)")
    ax, ay = ax / length, ay / length
    if ay < -1e-9 or (abs(ay) <= 1e-9 and ax < 0):
        ax, ay = -ax, -ay
    rx, ry = ay, -ax
    top = sorted(p, key=lambda q: q[0] * ax + q[1] * ay)[:2]
    pivot = max(top, key=lambda q: q[0] * rx + q[1] * ry)
    return {"pivot": list(pivot), "angle": math.atan2(ay, ax),
            "length": length, "thickness": thickness}


def _exit_distance(x0: float, y0: float, ux: float, uy: float, w: int, h: int) -> float:
    """(x0, y0) 에서 u 방향으로 이미지 밖으로 나가기까지의 거리."""
    if not (0 <= x0 <= w - 1 and 0 <= y0 <= h - 1):
        return 0.0
    d = math.inf
    if ux > 1e-9:
        d = min(d, (w - 1 - x0) / ux)
    elif ux < -1e-9:
        d = min(d, -x0 / ux)
    if uy > 1e-9:
        d = min(d, (h - 1 - y0) / uy)
    elif uy < -1e-9:
        d = min(d, -y0 / uy)
    return d


def _column_edge(p: bytes | list, y_ref0: int, y_ref1: int) -> float | None:
    """
    한 띠(블러된 밝기 프로필)의 끝 위치(축 방향 px, 서브픽셀) — 와이어 밝기와 그 아래
    최대 밝기의 중간을 처음 넘는 곳. 차이가 작으면(끝이 안 보임) None.
    """
    wire = median(p[y_ref0:y_ref1])
    peak = max(p[y_ref1:])
    if peak - wire < MIN_STEP:
        return None
    thr = wire + 0.5 * (peak - wire)
    for y in range(y_ref1, len(p)):
        if p[y] >= thr:
            prev = p[y - 1]
            return y - 1 + (thr - prev) / (p[y] - prev) if p[y] > prev else float(y)
    return None


def _solve3(m: list, v: list) -> list | None:
    """3×3 연립방정식 (가우스 소거, 부분 피벗)."""
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-12:
            return None
        a[c], a[piv] = a[piv], a[c]
        for r in range(3):
            if r != c:
                f = a[r][c] / a[c][c]
                for k in range(c, 4):
                    a[r][k] -= f * a[c][k]
    return [a[i][3] / a[i][i] for i in range(3)]


def _fit_quadratic(pts: list, t0: float) -> tuple[list, list] | None:
    """s = a·(t−t0)² + b·(t−t0) + c 를 반복 재가중으로 — (계수, 인라이어 여부)."""
    use = [True] * len(pts)
    coef = None
    for _ in range(4):
        sel = [p for p, u in zip(pts, use) if u]
        if len(sel) < 3:
            break
        m = [[0.0] * 3 for _ in range(3)]
        v = [0.0] * 3
        for t, s in sel:
            x = t - t0
            row = (x * x, x, 1.0)
            for i in range(3):
                v[i] += row[i] * s
                for j in range(3):
                    m[i][j] += row[i] * row[j]
        got = _solve3(m, v)
        if got is None:
            break
        coef = got
        res = [abs(s - (coef[0] * (t - t0) ** 2 + coef[1] * (t - t0) + coef[2])) for t, s in pts]
        cut = max(3.0, 3 * 1.4826 * median(r for r, u in zip(res, use) if u))
        new = [r <= cut for r in res]
        if new == use:
            break
        use = new
    return (coef, use) if coef else None


def find_tip(image_path: Path, points: list) -> dict:
    """
    이미지 + OBB 라벨(정규화 4점) → 끝 경계.

    반환: status ("ok" | "low" | "no_tip"), polygon (이미지 px — 윗변 2점 + 끝 곡선,
    옆변은 그 사이 직선), edges (띠별 끝 점, 이미지 px, inlier 여부), inliers/columns,
    rms(px), tip (곡선의 가장 먼 축 방향 거리 = 와이어 길이, px).
    """
    with Image.open(image_path) as im:
        gray = im.convert("L")
    w, h = gray.size
    pose = obb_pose([(x * w, y * h) for x, y in points])
    px, py = pose["pivot"]
    ux, uy = math.cos(pose["angle"]), math.sin(pose["angle"])
    rx, ry = uy, -ux
    thick, length = pose["thickness"], pose["length"]

    # 축 방향 탐색 범위 — 지금 OBB 바닥보다 더 내려가서(복사한 박스라 실제 끝이 밖일 수 있음)
    # 양 옆변이 이미지 밖으로 나가기 전까지.
    s_max = min(_exit_distance(px, py, ux, uy, w, h),
                _exit_distance(px - thick * rx, py - thick * ry, ux, uy, w, h), 3 * length)
    tw, sh = max(1, int(round(thick))), int(s_max)

    def to_img(t: float, s: float) -> list:
        """펼친 좌표(t = 왼쪽 변에서 두께 방향, s = 피벗에서 축 방향) → 이미지 px."""
        return [px + (t - thick) * rx + s * ux, py + (t - thick) * ry + s * uy]

    y_ref0, y_ref1 = REF_PX
    base = {"pose": pose, "polygon": None, "edges": [], "inliers": 0, "columns": 0,
            "rms": None, "tip": None}
    if sh <= y_ref1 + 10:
        return {**base, "status": "no_tip"}

    # 출력 (t, s) → 입력 (x, y) affine. 열 0 = 왼쪽 변, 열 tw = 오른쪽 변(피벗 쪽)
    strip = gray.transform((tw, sh), Image.AFFINE,
                           (rx, ux, px - thick * rx, ry, uy, py - thick * ry),
                           resample=Image.BILINEAR).filter(ImageFilter.GaussianBlur(BLUR))
    buf = strip.tobytes()
    cols = list(range(int(COLUMN_SPAN[0] * tw), int(COLUMN_SPAN[1] * tw) + 1, COLUMN_STEP))
    pts = []
    for t in cols:
        s = _column_edge(buf[t::tw], y_ref0, y_ref1)
        if s is not None:
            pts.append((t + 0.5, s))

    t0 = tw / 2
    fit = _fit_quadratic(pts, t0) if len(pts) >= 5 else None
    if fit is None:
        return {**base, "status": "no_tip", "columns": len(cols),
                "edges": [to_img(t, s) + [False] for t, s in pts]}
    (a, b, c), use = fit

    def curve(t: float) -> float:
        return min(max(a * (t - t0) ** 2 + b * (t - t0) + c, y_ref1), sh - 1)

    inl = [p for p, u in zip(pts, use) if u]
    rms = math.sqrt(sum((s - curve(t)) ** 2 for t, s in inl) / len(inl))
    ok = len(inl) >= OK_INLIERS * len(cols) and rms <= OK_RMS
    ts = [thick - thick * i / (CURVE_POINTS - 1) for i in range(CURVE_POINTS)]  # 오른쪽 → 왼쪽
    return {
        **base,
        "status": "ok" if ok else "low",
        "polygon": [to_img(0, 0), to_img(thick, 0)] + [to_img(t, curve(t)) for t in ts],
        "edges": [to_img(t, s) + [u] for (t, s), u in zip(pts, use)],
        "inliers": len(inl), "columns": len(cols), "rms": rms,
        "tip": max(curve(t) for t in ts),
    }
