"""
추론 결과 그림 — infer.py(사진·동영상)와 epoch_overlay.py(학습 중 격자)가 같이 쓴다.

Ultralytics 의 r.plot() 대신: 축 정렬 bbox 는 안 그리고, 마스크는 옅게 채우고 윤곽선
(OBB 모델은 회전 박스 윤곽), 클래스·점수는 그림 모서리에 한 줄씩.

와이어 끝점(tip) = OBB 아랫변의 중점. seg 모델은 마스크 윤곽의 최소 면적 회전 사각형
(cv2.minAreaRect)을 OBB 로 쓴다. 아랫변 = 두 짧은 변 중 화면 아래쪽(y 큰) 것.
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics.utils.plotting import colors

MASK_ALPHA = 0.25   # 마스크 채움 불투명도 (Ultralytics 기본 0.5 는 원본을 가림)


def bottom_mid(corners: np.ndarray) -> tuple:
    """회전 사각형 꼭짓점 4개(순서대로) → 아랫변(짧은 변 중 y 큰 것)의 중점 (x, y)."""
    c = np.asarray(corners, np.float64).reshape(4, 2)
    mids = (c + np.roll(c, -1, axis=0)) / 2               # 변 i = 꼭짓점 i → i+1
    lens = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    short = (0, 2) if lens[0] + lens[2] <= lens[1] + lens[3] else (1, 3)
    i = max(short, key=lambda k: mids[k, 1])
    return float(mids[i, 0]), float(mids[i, 1])


def detections(r) -> list:
    """점수 높은 순 [{name, conf, color, tip}] — tip 은 OBB 아랫변 중점 (없으면 None)."""
    out = []
    if r.obb is not None and len(r.obb):
        for pts, c, s in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.int().tolist(), r.obb.conf.tolist()):
            out.append({"name": r.names[c], "conf": s, "color": colors(c, True), "tip": bottom_mid(pts)})
    elif r.boxes is not None and len(r.boxes):
        polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        for poly, c, s in zip(polys, r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            tip = (bottom_mid(cv2.boxPoints(cv2.minAreaRect(poly.astype(np.float32))))
                   if poly is not None and len(poly) >= 3 else None)
            out.append({"name": r.names[c], "conf": s, "color": colors(c, True), "tip": tip})
    return sorted(out, key=lambda d: -d["conf"])


def shapes(r, alpha: float = MASK_ALPHA) -> np.ndarray:
    """원본(r.orig_img) 위에 마스크 / OBB 윤곽만 그린 그림."""
    im = r.orig_img.copy()
    lw = max(1, im.shape[1] // 600)
    if r.masks is not None and len(r.masks):
        fill = im.copy()
        for poly, c in zip(r.masks.xy, r.boxes.cls.int().tolist()):
            if len(poly) >= 3:
                cv2.fillPoly(fill, [poly.astype(np.int32)], colors(c, True))
        im = cv2.addWeighted(fill, alpha, im, 1 - alpha, 0)
        for poly, c in zip(r.masks.xy, r.boxes.cls.int().tolist()):
            if len(poly) >= 3:
                cv2.polylines(im, [poly.astype(np.int32)], True, colors(c, True), lw, cv2.LINE_AA)
    elif r.obb is not None and len(r.obb):
        for pts, c in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.int().tolist()):
            cv2.polylines(im, [pts.astype(np.int32)], True, colors(c, True), lw * 2, cv2.LINE_AA)
    return im


def labels(r) -> list:
    """[(글, BGR 색)] — 점수 높은 순. 검출이 없으면 빈 목록."""
    return [(f"{d['name']} {d['conf']:.2f}", d["color"]) for d in detections(r)]


def put_labels(im: np.ndarray, items: list, scale: float | None = None, top: bool = False) -> np.ndarray:
    """
    한 줄씩 — 색 네모 + 흰 글씨(검정 바탕). 색이 None 이면 네모 없이 회색 글씨.
    top=False: 왼쪽 아래부터 위로, top=True: 왼쪽 위부터 아래로. 비어 있으면 'no detection'.
    """
    items = items or [("no detection", None)]
    scale = scale or max(0.45, im.shape[1] / 1400)
    th_px = max(1, round(scale * 2))
    (_, h), base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, th_px)
    step = h + base + 6
    if top:
        ys = [6 + h + 4 + k * step for k in range(len(items))]
    else:
        ys = [im.shape[0] - 8 - base - k * step for k in range(len(items))][::-1]
    for (text, color), y in zip(items, ys):
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th_px)
        x = 6 if color is None else 6 + h + 6
        cv2.rectangle(im, (0, y - h - 4), (x + w + 6, y + base + 2), (0, 0, 0), -1)
        if color is not None:
            cv2.rectangle(im, (6, y - h), (6 + h, y), color, -1)
        cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255) if color is not None else (180, 180, 180), th_px, cv2.LINE_AA)
    return im


def draw_tip(im: np.ndarray, xy: tuple, color: tuple) -> None:
    """끝점 — 흰 테두리 원 + 가운데 색 점 (어두운·밝은 배경 모두에서 보이게)."""
    r = max(3, im.shape[1] // 200)
    p = (round(xy[0]), round(xy[1]))
    cv2.circle(im, p, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(im, p, r, color, -1, cv2.LINE_AA)


def render(r) -> np.ndarray:
    """shapes + 왼쪽 아래 라벨·점수."""
    return put_labels(shapes(r), labels(r))
