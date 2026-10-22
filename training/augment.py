"""
Ultralytics 증강 키로 못 내는 광도 증강 — config 의 `blur` · `brightness` 블록을 albumentations
목록(Ultralytics `augmentations=` 인자)으로 바꾼다. 이 목록이 기본 변환(Blur/MedianBlur/
ToGray/CLAHE, 각 1%)을 대체한다. 학습 해상도(imgsz)로 줄인 뒤 RandomHSV 앞에서 적용된다.

  blur:       {p, sigma: [lo, hi]}   GaussianBlur, sigma 는 imgsz 기준 px
  brightness: {p, gain: [lo, hi]}    밝기 배율 — 로그 균등 (0.25~4 면 ±2 stop, 중앙값 1배).
                                     Ultralytics hsv_v 는 1±v 라 최대 0~2배까지만 된다.

epoch_overlay.py 가 같은 분포를 직접 뽑아 화면에 값을 적는다 (gain 은 log_uniform).
"""
from __future__ import annotations

import math
import random

import albumentations as A
import numpy as np


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


class Brightness(A.ImageOnlyTransform):
    """이미지 전체에 배율 하나를 곱한다 (0~255 로 자름). 배율은 [lo, hi] 로그 균등."""

    def __init__(self, gain: tuple = (0.25, 4.0), p: float = 1.0):
        super().__init__(p=p)
        self.gain = (float(gain[0]), float(gain[1]))

    def get_params(self) -> dict:
        return {"g": log_uniform(self.py_random, *self.gain)}

    def apply(self, img: np.ndarray, g: float = 1.0, **params) -> np.ndarray:
        return np.clip(img.astype(np.float32) * g, 0, 255).astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple:
        return ("gain",)


def albumentations_list(cfg: dict) -> None:
    """cfg(학습 인자 dict)의 blur · brightness 를 꺼내 cfg["augmentations"] 로."""
    blur = cfg.pop("blur", None)
    bright = cfg.pop("brightness", None)
    out = []
    if bright:
        out.append(Brightness(gain=bright.get("gain", [0.25, 4.0]), p=float(bright.get("p", 1.0))))
    if blur:
        lo, hi = blur.get("sigma", [1.0, 3.0])
        out.append(A.GaussianBlur(blur_limit=0, sigma_limit=(float(lo), float(hi)),
                                  p=float(blur.get("p", 0.2))))
    if out:
        cfg["augmentations"] = out
