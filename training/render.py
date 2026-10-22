"""
추론 결과 그림 — infer.py(사진·동영상)와 epoch_overlay.py(학습 중 격자)가 같이 쓴다.

Ultralytics 의 r.plot() 대신: 축 정렬 bbox 는 안 그리고, 마스크는 옅게 채우고 윤곽선
(OBB 모델은 회전 박스 윤곽), 클래스·점수는 그림 위가 아니라 왼쪽 아래에 한 줄씩.
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics.utils.plotting import colors

MASK_ALPHA = 0.25   # 마스크 채움 불투명도 (Ultralytics 기본 0.5 는 원본을 가림)


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
    det = r.obb if r.obb is not None else r.boxes
    if det is None or not len(det):
        return []
    found = sorted(zip(det.conf.tolist(), det.cls.int().tolist()), reverse=True)
    return [(f"{r.names[c]} {s:.2f}", colors(c, True)) for s, c in found]


def put_labels(im: np.ndarray, items: list, scale: float | None = None) -> np.ndarray:
    """왼쪽 아래부터 위로 한 줄씩 — 클래스 색 네모 + 흰 글씨(검정 바탕). 검출이 없으면 'no detection'."""
    items = items or [("no detection", None)]
    scale = scale or max(0.45, im.shape[1] / 1400)
    th_px = max(1, round(scale * 2))
    (_, h), base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, th_px)
    step = h + base + 6
    y = im.shape[0] - 8 - base
    for text, color in reversed(items):
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th_px)
        x = 6 if color is None else 6 + h + 6
        cv2.rectangle(im, (0, y - h - 4), (x + w + 6, y + base + 2), (0, 0, 0), -1)
        if color is not None:
            cv2.rectangle(im, (6, y - h), (6 + h, y), color, -1)
        cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255) if color is not None else (160, 160, 160), th_px, cv2.LINE_AA)
        y -= step
    return im


def render(r) -> np.ndarray:
    """shapes + 왼쪽 아래 라벨·점수."""
    return put_labels(shapes(r), labels(r))
