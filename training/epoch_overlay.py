"""
매 val(epoch 끝)마다 고정된 입력 16장에 대한 현재 모델의 추론을 4×4 격자로 저장.

  <실행 폴더>/overlays/epoch_<N>.jpg   (TensorBoard 가 켜져 있으면 Images 탭 overlay 에도)

  행 = val 원본 한 장(val 전체에서 고르게 4장 — 세션이 섞이게)
  열 = [원본, 증강 1, 증강 2, 증강 3]  — 각 칸 위에 그 칸의 증강 값을 적음
  초록 선 = 정답 폴리곤 (증강 칸은 같은 변환을 적용해 옮긴 것), 그 위가 모델 추론(마스크·박스)

증강은 학습 config 와 같은 범위(degrees · translate · scale · shear · perspective · hsv_v ·
blur)에서 여기서 직접
뽑는다 — Ultralytics 의 학습 증강은 뽑힌 값을 밖으로 내주지 않아 화면에 적을 수 없기 때문.
입력 16장은 첫 epoch 에 한 번 만들고 고정 → epoch 이 지나며 같은 입력에서 추론이 어떻게
변하는지 본다. 블러는 학습에서 p(기본 10%)로만 들어가 무작위로는 거의 안 보이므로, 각 행의
마지막 증강 칸에는 항상 넣는다.

현재 가중치는 Ultralytics 가 epoch 마다 저장하는 last.pt(EMA)를 다시 읽어 예측 파이프라인을
그대로 탄다 — 전처리·후처리를 따로 재현하다 어긋나지 않게.

    model.add_callback("on_fit_epoch_end", epoch_overlay.callback(blur={"p": 0.1, "sigma": [0.3, 1.5]}))
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np

from common import IMG_EXTS

ROWS, AUGS = 4, 3   # 원본 4장 × (원본 + 증강 3) = 4×4
TILE = 360          # 격자 한 칸(px)
CONF = 0.25         # 이 점수 미만 예측은 그리지 않음
PAD = 114           # 증강으로 생긴 빈 곳 채움 (Ultralytics 와 같은 회색)
MASK_ALPHA = 0.25   # 추론 마스크 채움 불투명도 (Ultralytics 기본 0.5 는 원본을 가림)
MASK_BGR = (255, 42, 4)   # Ultralytics 0번 클래스 색(파랑)


def _val_images(data: dict, n: int) -> list:
    """data.yaml 의 val 폴더(들)에서 고르게 n 장 (정렬 후 균등 간격 — 세션이 섞이게)."""
    vals = data.get("val")
    vals = vals if isinstance(vals, list) else [vals]
    files = sorted(p for v in vals if v and Path(v).is_dir()
                   for p in Path(v).rglob("*") if p.suffix.lower() in IMG_EXTS)
    k = min(n, len(files))
    return [files[int(i * len(files) / k)] for i in range(k)]


def _gt_polygons(img_path: Path, w: int, h: int) -> list:
    """images/…/x.png 짝 라벨(labels/…/x.txt)의 폴리곤들 (px)."""
    parts = list(img_path.parts)
    if "images" not in parts:
        return []
    i = len(parts) - 1 - parts[::-1].index("images")
    lbl = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
    if not lbl.exists():
        return []
    polys = []
    for line in lbl.read_text(encoding="utf-8").splitlines():
        v = [float(x) for x in line.split()[1:]]
        if len(v) >= 6:
            polys.append(np.array([[v[j] * w, v[j + 1] * h] for j in range(0, len(v) - 1, 2)],
                                  np.float32))
    return polys


def _augment(img: np.ndarray, polys: list, hyp: dict, rng: random.Random,
             force_blur: bool) -> tuple:
    """학습과 같은 범위의 증강을 한 번 뽑아 적용 → (이미지, 옮긴 폴리곤, 값 문자열)."""
    h, w = img.shape[:2]
    # Ultralytics RandomPerspective 와 같은 합성: M = T·S·R·P·C (오른쪽부터, 중심 기준)
    deg = rng.uniform(-hyp["degrees"], hyp["degrees"])
    s = rng.uniform(1 - hyp["scale"], 1 + hyp["scale"])
    tx = rng.uniform(-hyp["translate"], hyp["translate"])
    ty = rng.uniform(-hyp["translate"], hyp["translate"])
    shx = rng.uniform(-hyp["shear"], hyp["shear"])
    shy = rng.uniform(-hyp["shear"], hyp["shear"])
    # perspective 는 학습 해상도(imgsz) px 기준 계수 — 입력이 더 크면 그만큼 줄여야 같은 휨
    px = rng.uniform(-hyp["perspective"], hyp["perspective"])
    py = rng.uniform(-hyp["perspective"], hyp["perspective"])
    k = hyp["imgsz"] / w
    C = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], np.float64)
    P = np.array([[1, 0, 0], [0, 1, 0], [px * k, py * k, 1]], np.float64)
    R = np.vstack([cv2.getRotationMatrix2D((0, 0), deg, s), [0, 0, 1]])
    S = np.array([[1, math.tan(math.radians(shx)), 0],
                  [math.tan(math.radians(shy)), 1, 0], [0, 0, 1]], np.float64)
    T = np.array([[1, 0, (0.5 + tx) * w], [0, 1, (0.5 + ty) * h], [0, 0, 1]], np.float64)
    m = T @ S @ R @ P @ C
    out = cv2.warpPerspective(img, m, (w, h), borderValue=(PAD, PAD, PAD))
    polys = [cv2.perspectiveTransform(p[None], m)[0] for p in polys]

    gain = rng.uniform(1 - hyp["hsv_v"], 1 + hyp["hsv_v"])
    out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    text = f"rot {deg:+.1f} s {s:.2f} t {tx:+.2f},{ty:+.2f} v x{gain:.2f}"
    if hyp["shear"] or hyp["perspective"]:
        text += f"\nsh {shx:+.1f},{shy:+.1f}  p {px * 1e4:+.1f},{py * 1e4:+.1f}e-4"

    blur = hyp.get("blur") or {}
    if blur and (force_blur or rng.random() < float(blur.get("p", 0))):
        lo, hi = blur.get("sigma", [0.3, 1.5])
        # sigma 는 학습 해상도(imgsz) 기준 px — 입력이 더 크면 그만큼 키워야 같은 흐림
        sigma = rng.uniform(float(lo), float(hi))
        out = cv2.GaussianBlur(out, (0, 0), sigma * w / hyp["imgsz"])
        text += f"  blur {sigma:.1f}"
    return out, polys, text


def _plot(r) -> np.ndarray:
    """추론 그림 — 박스·점수는 Ultralytics 그대로, 마스크만 옅게(MASK_ALPHA) + 윤곽선."""
    plot = r.plot(masks=False)
    if r.masks is not None and len(r.masks):
        polys = [p.astype(np.int32) for p in r.masks.xy if len(p) >= 3]
        fill = plot.copy()
        cv2.fillPoly(fill, polys, MASK_BGR)
        plot = cv2.addWeighted(fill, MASK_ALPHA, plot, 1 - MASK_ALPHA, 0)
        cv2.polylines(plot, polys, True, MASK_BGR, max(1, plot.shape[1] // 600), cv2.LINE_AA)
    return plot


def _tile(plot: np.ndarray, polys: list, text: str) -> np.ndarray:
    for p in polys:   # 정답 — 추론 그림 위에 초록 선
        cv2.polylines(plot, [p.astype(np.int32)], True, (0, 255, 0),
                      max(2, plot.shape[1] // 300), cv2.LINE_AA)
    t = cv2.resize(plot, (TILE, TILE))
    lines = text.split("\n")
    (tw, th), _ = cv2.getTextSize(max(lines, key=len), cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(t, (0, 0), (min(TILE, tw + 8), (th + 6) * len(lines) + 2), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(t, line, (4, (th + 6) * i + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return t


def callback(blur: dict | None = None, seed: int = 0):
    """Ultralytics on_fit_epoch_end 콜백. blur = 학습 config 의 blur 블록 (없으면 블러 안 함)."""
    state: dict = {}

    def build(trainer) -> None:
        a = trainer.args
        hyp = {"degrees": float(a.degrees), "translate": float(a.translate),
               "scale": float(a.scale), "shear": float(a.shear),
               "perspective": float(a.perspective), "hsv_v": float(a.hsv_v),
               "imgsz": int(a.imgsz), "blur": blur}
        rng = random.Random(seed)
        inputs = []                       # (이미지 BGR, 정답 폴리곤, 칸 위 문자열)
        for p in _val_images(trainer.data, ROWS):
            img = cv2.imread(str(p))
            polys = _gt_polygons(p, img.shape[1], img.shape[0])
            inputs.append((img, polys, f"orig  {p.stem[:28]}"))
            for k in range(AUGS):
                inputs.append(_augment(img, polys, hyp, rng, force_blur=(k == AUGS - 1)))
        state["inputs"] = inputs

    def on_fit_epoch_end(trainer) -> None:
        # 학습 루프의 마지막 epoch 에서 trainer.stop 이 켜지고, 그 뒤 final_eval 이 best.pt 로
        # 한 번 더 검증하며 이 콜백을 다시 부른다(epoch+1) → 그 호출은 best.pt 로 best.jpg.
        final = state.get("stopped", False)
        state["stopped"] = bool(trainer.stop)
        weights = Path(trainer.best if final else trainer.last)
        if not weights.exists():       # save=False 등 — 읽을 가중치가 없음
            return
        if "inputs" not in state:
            build(trainer)
        inputs = state["inputs"]
        if not inputs:
            return
        from ultralytics import YOLO

        model = YOLO(str(weights))
        results = model.predict([im for im, _, _ in inputs], imgsz=trainer.args.imgsz,
                                device=str(trainer.device), conf=CONF, verbose=False)
        tiles = [_tile(_plot(r), polys, text) for r, (_, polys, text) in zip(results, inputs)]
        cols = AUGS + 1
        rows = math.ceil(len(tiles) / cols)
        sheet = np.zeros((rows * TILE, cols * TILE, 3), np.uint8)
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t
        ep = trainer.epoch + 1
        label = "best.pt" if final else f"epoch {ep}"
        cv2.putText(sheet, label, (sheet.shape[1] - 130, sheet.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        out = Path(trainer.save_dir) / "overlays"
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / ("best.jpg" if final else f"epoch_{ep:03d}.jpg")), sheet)
        try:
            from ultralytics.utils.callbacks import tensorboard as tb
            writer = getattr(tb, "WRITER", None)
        except ImportError:
            writer = None
        if writer is not None:
            writer.add_image("overlay", sheet[..., ::-1].copy(), ep, dataformats="HWC")

    return on_fit_epoch_end
