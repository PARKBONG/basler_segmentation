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
        <이미지이름>.png   초록 = 원본 라벨(GT), 빨강 = 추론 결과

같은 그림을 격자로 묶어 val 결과 폴더(runs/obb/val*)에도 val_batch*_overlay.jpg 로
남깁니다 — 정답과 추론을 다른 장에 그리는 Ultralytics 기본 val_batch*_labels/_pred.jpg
는 대신 지웁니다(같은 폴더에 규약이 다른 그림이 섞이면 판단이 흐려집니다).
단일 클래스이므로 어느 쪽 그림에도 클래스명·점수 글자는 넣지 않습니다. 다만
visualize.baseline 을 켜면(기본) 박스마다 아래 두 꼭짓점의 y 평균 높이에 가로
점선을 긋고 그 y 픽셀값을 적습니다 — wire 의 높이 오차를 눈으로 바로 재기 위한 것.

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

from common import IMG_EXTS, RUN_STAMP_RE, Stage, load_yaml, run_stamp

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

    # -- 아래변 기준선 ---------------------------------------------------

    @staticmethod
    def baseline_y(pts: list) -> float:
        """
        OBB 아래 두 꼭짓점의 y 평균 — wire 의 '높이'를 한 숫자로 읽기 위한 값.

        꼭짓점 순서를 믿지 않고 **y 가 가장 큰 두 점**을 고릅니다 (라벨러가 찍은
        순서·회전 방향과 무관하게 같은 값이 나오도록). 기울기가 45°를 넘으면 그 둘이
        아래 '긴 변'이 아니라 짧은 변이 되지만, 이 프로젝트의 wire 는 거의 수평이라
        실제로는 아래 긴 변의 두 끝점입니다.
        """
        ys = sorted(y for _x, y in pts)
        if not ys:
            return 0.0
        return (ys[-1] + ys[-2]) / 2 if len(ys) >= 2 else ys[-1]

    @staticmethod
    def _font(canvas_h: int):
        """이미지 높이에 비례하는 글꼴 — 없으면 PIL 기본 비트맵 글꼴."""
        from PIL import ImageFont

        size = max(12, int(canvas_h / 36))
        for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def draw_baseline(draw, canvas, y: float, color: tuple, font, right: bool) -> None:
        """
        y 높이에 이미지 가로 전체를 가로지르는 가는 점선 + 그 y 픽셀값을 적는다.

        점선(1px)으로 그려 박스 외곽선(굵은 실선)과 헷갈리지 않게 하고, GT 는 왼쪽
        끝·추론은 오른쪽 끝에 값을 적어 두 값이 비슷해도 글자가 겹치지 않게 합니다.
        y 는 **지금 그리는 이미지(=processed, 크롭·리사이즈된 뒤)** 기준 픽셀입니다.
        """
        yi = round(y)
        dash, gap = 14, 10
        x = 0
        while x < canvas.width:
            draw.line([(x, yi), (min(x + dash, canvas.width), yi)], fill=color, width=1)
            x += dash + gap

        text = f"y={y:.1f}"
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        tw, th = x1 - x0, y1 - y0
        tx = canvas.width - tw - 8 if right else 8
        ty = min(max(yi + 4, 0), canvas.height - th - 4)     # 선 아래, 화면 밖으로 안 나가게
        draw.rectangle([tx - 4, ty - 3, tx + tw + 4, ty + th + 3], fill=(0, 0, 0))
        draw.text((tx - x0, ty - y0), text, fill=color, font=font)

    @staticmethod
    def write_mosaic(sheets: list, run_dir: Path, cell: int = 480, cols: int = 4) -> list:
        """
        오버레이 그림들을 격자로 묶어 run 폴더에 저장 (val_batch*_overlay.jpg).

        Ultralytics 가 남기는 val_batch*_labels/_pred.jpg 는 정답과 추론을 **다른 장**에
        그려서 눈으로 대조해야 하는데, 같은 장에 겹쳐 놓으면 어긋난 각도·미검출이
        한눈에 보입니다. 그래서 같은 자리에 같은 모양(격자)으로 대체본을 남깁니다.
        """
        from PIL import Image

        written = []
        per = cols * cols
        for k in range(0, len(sheets), per):
            chunk = sheets[k:k + per]
            rows = -(-len(chunk) // cols)
            sheet = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
            for i, im in enumerate(chunk):
                thumb = im.copy()
                thumb.thumbnail((cell, cell))
                sheet.paste(thumb, ((i % cols) * cell + (cell - thumb.width) // 2,
                                    (i // cols) * cell + (cell - thumb.height) // 2))
            path = run_dir / f"val_batch{k // per}_overlay.jpg"
            sheet.save(path, quality=90)
            written.append(path)
        return written

    def visualize(self, model, weights: Path, data: Path, split: str,
                  metrics: list, run_dir: Path | None = None) -> Path:
        """
        추론 결과를 이미지에 그려 finetuner/eval/<실행폴더>/ 에 저장.

        같은 이미지에 GT(초록)와 추론(빨강)을 겹쳐 그립니다 — 무엇이 어긋났는지
        보려면 둘을 따로 보는 것보다 겹쳐 보는 편이 훨씬 빠릅니다. 어떤 pt·데이터셋의
        그림인지는 폴더 이름과 summary.txt 양쪽에 남깁니다 (그림만 떼어 봐도 출처를
        알 수 있게).

        클래스 이름·점수는 **적지 않습니다** — 단일 클래스라 글자가 정보를 더하지
        않고, 가느다란 물체 위에 겹쳐 도형만 가립니다. 색이 곧 의미입니다.

        visualize.baseline 이 켜져 있으면(기본) 박스마다 **아래 두 꼭짓점의 y 평균**
        높이에 가로 점선을 긋고 그 y 픽셀값을 적습니다. wire 의 높이가 관심사일 때
        GT 선과 추론 선의 간격이 곧 오차라서, 도형만 겹쳐 보는 것보다 몇 px 어긋났는지가
        바로 읽힙니다. GT 값은 왼쪽 끝, 추론 값은 오른쪽 끝에 적어 겹치지 않게 합니다.

        run_dir 을 주면(=val 결과 폴더) 같은 그림을 격자로 묶어 거기에도 남깁니다.
        """
        from PIL import Image, ImageDraw

        vis = self.cfg.get("visualize") or {}
        conf = float(vis.get("conf", 0.25))
        count = int(vis.get("count", 20))
        baseline = bool(vis.get("baseline", True))

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
        trained = weights.parent.parent          # .../<name>/<YYMMDD_HHMMSS>/weights/best.pt
        tag = (f"{trained.parent.name}-{trained.name}"
               if RUN_STAMP_RE.match(trained.name) else trained.name)
        out = (self.resolve(vis.get("out", "eval"))
               / f"{run_stamp()}__{tag}__{data.parent.name}-{split}")
        out.mkdir(parents=True, exist_ok=True)

        sheets: list = []                        # run 폴더 격자에 쓸 오버레이 원본
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
            font = self._font(canvas.height) if baseline else None

            label = self.label_for(path)
            if label is not None and label.exists():
                for _cls, pts in self.gt_polygons(label, canvas.width, canvas.height):
                    draw.polygon(pts, outline=GT_COLOR, width=width)
                    if baseline:
                        self.draw_baseline(draw, canvas, self.baseline_y(pts),
                                           GT_COLOR, font, right=False)
                    n_gt += 1
            else:
                n_nolabel += 1

            for _cls, _score, pts in self.pred_polygons(result):
                draw.polygon(pts, outline=PRED_COLOR, width=width)
                if baseline:
                    self.draw_baseline(draw, canvas, self.baseline_y(pts),
                                       PRED_COLOR, font, right=True)
                n_pred += 1

            canvas.save(out / f"{path.stem}.png")
            sheets.append(canvas)

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
            "초록 = 원본 라벨(GT), 빨강 = 추론 결과 (단일 클래스라 클래스명·점수는 적지 않음)",
            *(["가로 점선 = 박스 아래 두 꼭짓점의 y 평균 (왼쪽 값 = GT, 오른쪽 값 = 추론).",
               "  y 는 이 그림(= processed, 크롭·리사이즈 후) 기준 픽셀입니다."]
              if baseline else []),
            f"이미지 {len(images)}장 (split 전체 {total}장 중)"
            f" · GT {n_gt}개 · 추론 {n_pred}개"
            + (f" · 라벨 없는 이미지 {n_nolabel}장" if n_nolabel else ""),
        ]) + "\n", encoding="utf-8")

        self.log(f"시각화 {len(images)}장 → {out}")
        self.log("  초록 = 원본 라벨(GT), 빨강 = 추론 결과")

        if run_dir is not None and run_dir.is_dir():
            # 정답/추론을 따로 그린 Ultralytics 기본 그림은 지웁니다 — 같은 폴더에
            # 색·규약이 다른 그림이 섞여 있으면 어느 쪽을 보고 판단했는지 흐려집니다.
            stale = sorted(p for pat in ("val_batch*_labels.jpg", "val_batch*_pred.jpg")
                           for p in run_dir.glob(pat))
            for p in stale:
                p.unlink()
            sheet_paths = self.write_mosaic(sheets, run_dir)
            self.log(f"run 폴더에도 겹쳐그림 {len(sheet_paths)}장 → {run_dir}"
                     + (f" (기본 labels/pred 그림 {len(stale)}장 대체)" if stale else ""))
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
            # val 결과 폴더(runs/obb/val*)에도 같은 규약의 그림을 남긴다
            run_dir = getattr(m, "save_dir", None)
            self.visualize(model, weights, data, split, lines,
                           Path(run_dir) if run_dir else None)


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
