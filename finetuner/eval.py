"""
학습된 checkpoint 를 지정한 split(train/val/test) 에 대해 평가.

Stage1/Stage2 를 명확히, 독립적으로 평가하기 위한 스크립트 (Box + Mask mAP 출력).
무엇을 평가할지는 전부 config(configs/eval.*.yaml — weights/data/split/imgsz/device)가
정합니다.

지표(mAP)에 더해, config 의 visualize 블록으로 **추론 결과를 이미지에 그려서**
저장합니다 (기본 켜짐). 원본 라벨이 있으면 같은 그림에 겹쳐 그리므로 숫자만으로는
안 보이는 실패 양상(위치는 맞는데 각도가 틀림 / 미검출 / 오검출)을 바로 볼 수 있습니다.

    finetuner/eval/<YYMMDD_HHMMSS>__<가중치>__<데이터셋>-<split>/
        summary.txt        무슨 pt · 무슨 데이터셋 · 무슨 split · 지표 (그림의 출처)
        <이미지이름>.png   초록 = 원본 라벨(GT), 빨강 = 추론 결과(conf 표시)

데이터 위생:
  · Stage1 은 공개 val 로 평가 (in-domain 격리 유지)      → eval.stage1.yaml
  · Stage2 선택 지표는 in-domain val                       → eval.stage2.yaml
  · in-domain test 는 "최종 stage2 모델"에 딱 한 번만      → eval.final.yaml
    이 수치로 하이퍼파라미터를 다시 만지면 그 순간 오염되어 논문에 쓸 수 없습니다.

사용법:
    python eval.py --config eval.stage1.yaml
    python eval.py --config eval.final.yaml     # 최종 1회만
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import IMG_EXTS, RUN_STAMP_RE, Stage, load_yaml, normalize_names, run_stamp

# 그림 색: 원본 라벨과 추론 결과를 한 장에서 구분하기 위한 규약 (summary.txt 에도 적습니다).
GT_COLOR = (40, 220, 80)      # 초록 = 정답(GT)
PRED_COLOR = (255, 60, 60)    # 빨강 = 추론


class Evaluator(Stage):
    """checkpoint 하나를 한 split 에 대해 평가하는 단계."""

    config_glob = "eval.*.yaml"
    label = "eval"

    @staticmethod
    def _fmt(obj, name) -> str:
        try:
            return f"{getattr(obj, name):.4f}"
        except Exception:
            return "n/a"

    # -- 시각화 ----------------------------------------------------------

    def split_images(self, data: Path, split: str) -> list:
        """
        data.yaml 의 split 항목이 가리키는 이미지들 (경로는 data.yaml 의 path 기준).

        split 값은 폴더 하나일 수도, 통합본처럼 폴더 목록일 수도 있습니다
        (train.py 의 merged_data_path 산출물). 둘 다 받습니다.
        """
        doc = load_yaml(data)
        entry = doc.get(split)
        if not entry:
            self.fail(f"{data} 에 {split} 항목이 없습니다 — 그릴 이미지를 못 찾습니다.")
        root = Path(str(doc.get("path") or data.parent))
        if not root.is_absolute():
            root = data.parent / root

        images: list = []
        for item in (entry if isinstance(entry, list) else [entry]):
            d = Path(str(item))
            d = d if d.is_absolute() else root / d
            if not d.is_dir():
                self.log(f"[주의] 이미지 폴더가 없습니다: {d}")
                continue
            images += sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXTS)
        return images

    @staticmethod
    def label_for(image: Path) -> Path | None:
        """이미지 짝 라벨 (.../images/<split>/x.jpg → .../labels/<split>/x.txt)."""
        parts = list(image.parts)
        for i in range(len(parts) - 2, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                return Path(*parts).with_suffix(".txt")
        return None

    @staticmethod
    def gt_polygons(label: Path, w: int, h: int) -> list:
        """라벨 파일 → [(cls, [(x,y), ...]), ...] (정규화 좌표를 픽셀로)."""
        polys = []
        for line in label.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) < 7:          # cls + 최소 3점 — obb 는 4점(9칸)
                continue
            pts = [(float(v[j]) * w, float(v[j + 1]) * h)
                   for j in range(1, len(v) - 1, 2)]
            polys.append((int(float(v[0])), pts))
        return polys

    @staticmethod
    def pred_polygons(result) -> list:
        """추론 결과 → [(cls, conf, [(x,y), ...]), ...]. OBB 면 회전 사각형, 아니면 xyxy."""
        obb = getattr(result, "obb", None)
        if obb is not None and len(obb):
            corners = obb.xyxyxyxy.cpu().numpy().reshape(len(obb), 4, 2)
            return [(int(c), float(f), [(float(x), float(y)) for x, y in quad])
                    for quad, c, f in zip(corners, obb.cls.tolist(), obb.conf.tolist())]
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            out = []
            for (x1, y1, x2, y2), c, f in zip(boxes.xyxy.cpu().numpy().tolist(),
                                              boxes.cls.tolist(), boxes.conf.tolist()):
                out.append((int(c), float(f),
                            [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
            return out
        return []

    @staticmethod
    def _font(size: int):
        """이미지 크기에 맞춘 글꼴 — 없으면 PIL 기본 글꼴(작지만 항상 있음)."""
        from PIL import ImageFont

        for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def visualize(self, model, weights: Path, data: Path, split: str, metrics: list) -> Path:
        """
        추론 결과를 이미지에 그려 finetuner/eval/<실행폴더>/ 에 저장.

        같은 이미지에 GT(초록)와 추론(빨강)을 겹쳐 그립니다 — 무엇이 어긋났는지
        보려면 둘을 따로 보는 것보다 겹쳐 보는 편이 훨씬 빠릅니다. 어떤 pt·데이터셋의
        그림인지는 폴더 이름과 summary.txt 양쪽에 남깁니다 (그림만 떼어 봐도 출처를
        알 수 있게).
        """
        from PIL import Image, ImageDraw

        vis = self.cfg.get("visualize") or {}
        conf = float(vis.get("conf", 0.25))
        count = int(vis.get("count", 20))

        images = self.split_images(data, split)
        if not images:
            self.log("[주의] 그릴 이미지가 없습니다 — 시각화를 건너뜁니다.")
            return self.resolve(vis.get("out", "eval"))
        total = len(images)
        if 0 < count < len(images):     # 앞쪽에 몰리지 않게 균등 간격으로 뽑는다
            step = len(images) / count
            images = [images[int(i * step)] for i in range(count)]

        # 폴더 이름에 무엇을 그린 그림인지 박아둔다:
        #   <실행시각>__<가중치이름>__<데이터셋>-<split>
        run_dir = weights.parent.parent          # .../<name>/<YYMMDD_HHMMSS>/weights/best.pt
        tag = (f"{run_dir.parent.name}-{run_dir.name}"
               if RUN_STAMP_RE.match(run_dir.name) else run_dir.name)
        out = (self.resolve(vis.get("out", "eval"))
               / f"{run_stamp()}__{tag}__{data.parent.name}-{split}")
        out.mkdir(parents=True, exist_ok=True)

        names = normalize_names(load_yaml(data)["names"])
        n_gt = n_pred = n_nolabel = 0

        results = model.predict(source=[str(p) for p in images],
                                imgsz=int(self.cfg.get("imgsz", 640)),
                                device=str(self.cfg.get("device", "0")),
                                conf=conf, verbose=False, stream=True)
        for path, result in zip(images, results):
            with Image.open(path) as im:
                canvas = im.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            width = max(2, canvas.width // 400)
            font = self._font(max(12, canvas.width // 40))
            gap = font.size + 2 if hasattr(font, "size") else 12

            # 글자는 도형 위/아래로 나눠 붙인다 — 같은 물체의 GT 와 추론은 거의 겹쳐
            # 있어서, 같은 자리에 쓰면 두 글자가 포개져 아무것도 못 읽습니다.
            label = self.label_for(path)
            if label is not None and label.exists():
                for cls, pts in self.gt_polygons(label, canvas.width, canvas.height):
                    draw.polygon(pts, outline=GT_COLOR, width=width)
                    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                    draw.text((x0, y0 - gap), names.get(cls, str(cls)),
                              fill=GT_COLOR, font=font)
                    n_gt += 1
            else:
                n_nolabel += 1

            for cls, score, pts in self.pred_polygons(result):
                draw.polygon(pts, outline=PRED_COLOR, width=width)
                x0, y1 = min(p[0] for p in pts), max(p[1] for p in pts)
                draw.text((x0, y1 + 2), f"{names.get(cls, str(cls))} {score:.2f}",
                          fill=PRED_COLOR, font=font)
                n_pred += 1

            canvas.save(out / f"{path.stem}.png")

        (out / "summary.txt").write_text("\n".join([
            "# 이 폴더의 그림이 무엇인지 (재현용)",
            f"weights : {weights}",
            f"data    : {data}",
            f"split   : {split}",
            f"imgsz   : {self.cfg.get('imgsz', 640)}   conf: {conf}",
            f"config  : {self.config_path}",
            "",
            "# 지표 (같은 weights/data/split 에 대한 val 결과)",
            *metrics,
            "",
            "# 그림 규약",
            "초록 = 원본 라벨(GT), 빨강 = 추론 결과(클래스 conf)",
            f"이미지 {len(images)}장 (split 전체 {total}장 중)"
            f" · GT {n_gt}개 · 추론 {n_pred}개"
            + (f" · 라벨 없는 이미지 {n_nolabel}장" if n_nolabel else ""),
        ]) + "\n", encoding="utf-8")

        self.log(f"시각화 {len(images)}장 → {out}")
        self.log("  초록 = 원본 라벨(GT), 빨강 = 추론 결과")
        return out

    # -- 실행 ------------------------------------------------------------

    def run(self) -> None:
        for key in ("weights", "data"):
            if not self.cfg.get(key):
                self.fail(f"{key} 가 없습니다 — 무엇을 평가할지 기본값으로 추측하지 않습니다.")
        split = str(self.cfg.get("split", "val"))
        if split not in ("train", "val", "test"):
            self.fail(f"split 은 train/val/test 중 하나여야 합니다: {split}")

        # runs/obb/<name>/weights/best.pt 처럼 타임스탬프를 빼고 적으면 최신 실행을 씁니다.
        weights = self.resolve_weights(self.cfg["weights"])
        if not weights.exists():
            raise FileNotFoundError(
                f"가중치가 없습니다: {weights}\n"
                f"학습 결과는 runs/obb/<name>/<YYMMDD_HHMMSS>/weights/best.pt 에 쌓입니다 "
                f"— 타임스탬프를 빼고 적으면 가장 최근 실행을 씁니다."
            )
        data = self.resolve(self.cfg["data"])
        if not data.exists():
            raise FileNotFoundError(f"data.yaml 이 없습니다: {data}")

        if split == "test":
            self.log("[주의] test 는 최종 무편향 평가용입니다. 이 결과를 보고 "
                     "하이퍼파라미터를 다시 조정하면 오염됩니다.")

        from ultralytics import YOLO   # 무거운 import 는 설정/경로 검증 뒤로

        model = YOLO(str(weights))
        m = model.val(data=str(data), split=split,
                      imgsz=int(self.cfg.get("imgsz", 640)),
                      device=str(self.cfg.get("device", "0")))

        lines = [f"Box  mAP50-95={self._fmt(m.box, 'map')}  mAP50={self._fmt(m.box, 'map50')}"]
        seg = getattr(m, "seg", None)
        if seg is not None:
            lines.append(f"Mask mAP50-95={self._fmt(seg, 'map')}  mAP50={self._fmt(seg, 'map50')}")

        print(f"\n[eval] weights={weights.name}  data={data.parent.name}  split={split}")
        for line in lines:
            print(f"  {line}")

        # 시각화는 기본 켜짐 — 지표만 보고 넘어가면 "어떻게 틀렸는지"를 놓칩니다.
        # 끄려면 config 에 visualize.enabled: false.
        if (self.cfg.get("visualize") or {}).get("enabled", True):
            self.visualize(model, weights, data, split, lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLO seg checkpoint 평가",
        epilog="configs/ 의 후보: " + Evaluator.config_candidates(),
    )
    ap.add_argument(
        "--config", metavar="YAML",
        help="평가 설정 yaml (후보가 여럿이라 사실상 필수). configs/ 안 파일명 또는 경로",
    )
    Evaluator(ap.parse_args().config).run()


if __name__ == "__main__":
    main()
