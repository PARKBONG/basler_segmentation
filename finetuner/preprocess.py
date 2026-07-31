"""
공개 + 로컬 데이터를 학습용 YOLO seg 데이터셋으로 전처리(크롭 · 리사이즈 · 클래스 통일).

폴더 규약:
- 원천은 datasets/raw/<이름>/ (kimm = 앱 REC 카메라 원본, rf_* = download.py 공개셋).
- 산출물은 소스별로 out/<이름>/images|labels/{train,val,test} 에 씁니다.
- out/data.yaml 하나가 디스크에 있는 소스 산출물 전체를 묶어 train.py 가 읽습니다
  (Ultralytics 는 각 split 에 폴더 목록을 지원). test 는 있을 때만 키를 넣습니다.

설계 원칙:
- 어떤 데이터셋을 굽는지는 download_config.yaml 의 stages 가 정합니다 — 소스 정의는
  sources 에 한 번만 적고, 스테이지별 use 가 포함 소스와 val/test 비율을 정합니다.
  후보(단일/stage1/stage2)가 여럿이라 --stage 로 반드시 명시합니다.
- 분할(train/val/test)은 소스 내부에서, oversample 전에 그룹(원본) 단위로 나눕니다.
  Roboflow 증강 사본(..._jpg.rf.<hash>)은 원본 단위로 묶여 같은 split 에만 들어갑니다
  (근중복이 train/val 에 갈라 들어가는 누수 방지).
- test 는 test_ratio 로 뗍니다 — 최종 모델에 딱 한 번 쓰는 평가용 (eval.py --split test).
- oversample 은 각 소스의 train 쪽에만 물리 복제로 적용 (val/test 누수 없음).
- 클래스 선택/이름통일은 class_map(이름 기반)이 담당 → source index 차이에 안전.
- 크롭은 이미지와 폴리곤 라벨을 함께 변환합니다. 창 밖으로 나간 인스턴스는 잘리고,
  남은 면적이 min_area 미만이면 인스턴스째 폐기, 살아남은 인스턴스가 하나도 없으면
  그 이미지는 데이터셋에서 제외됩니다(기존 동작과 동일: 대상 없는 이미지는 안 넣음).
- 크롭 인자는 소스마다 독립입니다(공유 기본값 없음). 창을 %로 고정하거나
  (width/height/center_x/center_y), auto_crop 으로 이미지마다 라벨에서 직접 잡습니다.
- --only 에 이름을 적으면 그 소스만 다시 굽습니다 (생략하면 스테이지 전체). 산출물이
  소스별 폴더라서 나머지 소스의 기존 산출물은 유지되고, data.yaml 만 매번 다시 묶입니다.
- 마음에 안 드는 이미지는 원천(raw)에서 이미지 파일만 지우면 됩니다 — 다음 실행에서
  짝 라벨(.txt)이 자동 삭제됩니다 (이미지가 하나도 없는 폴더는 경로 실수로 보고 보호).

증강은 여기서 하지 않습니다. Ultralytics 가 학습 중에 온라인 증강을 하므로
train_config.yaml 의 증강 하이퍼파라미터로 조절하고, 결과 확인은
`python train.py` 의 preview_aug 단계를 쓰세요 (offline 증강은 다양성이 오히려 줄고
온라인 증강과 이중으로 겹칩니다).

각 소스 폴더엔 data.yaml (names + train/val 경로) 이 있어야 함.
  · Roboflow export 는 기본 포함.
  · 로컬(kimm)은 최소 형식으로 하나 작성:  names: {0: wire}\n train: images

사용법:
    python preprocess.py --stage single    # 단일 스테이지
    python preprocess.py --stage stage1    # stage1 (공개 전용)
    python preprocess.py --stage stage2    # stage2 (in-domain)
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

import yaml

from common import IMG_EXTS, Stage, load_yaml, normalize_names, resolve


# ── 크롭 기하 (순수 함수 — selftest.py 가 직접 검증) ─────────────────────────

def crop_rect(src_w: int, src_h: int, crop: dict) -> tuple[int, int, int, int]:
    """
    크롭 창 (x, y, w, h) 을 픽셀로 계산.

    크기와 위치 모두 앱(FrameCropper.cs)과 같은 규약입니다:
    - width / height 가 0 이하(또는 없음)이면 그 축은 원본 전체를 씁니다.
      요청 크기가 원본보다 크면 원본 크기로 클램프합니다.
    - center_x / center_y 는 0~100%: 0 = 왼쪽/위 끝, 100 = 오른쪽/아래 끝, 50 = 중앙.
    """
    def size(want, src: int) -> int:
        want = int(want or 0)
        return src if want <= 0 else min(want, src)

    w = size(crop.get("width"), src_w)
    h = size(crop.get("height"), src_h)
    cx = min(max(float(crop.get("center_x", 50)), 0.0), 100.0)
    cy = min(max(float(crop.get("center_y", 50)), 0.0), 100.0)
    x = int(round((src_w - w) * cx / 100.0))
    y = int(round((src_h - h) * cy / 100.0))
    return x, y, w, h


def label_bbox(lines: list, src_w: int, src_h: int) -> tuple | None:
    """라벨 폴리곤 전부를 감싸는 픽셀 경계상자 (x0, y0, x1, y1). 폴리곤이 없으면 None."""
    xs: list = []
    ys: list = []
    for line in lines:
        coords = [float(v) for v in line.split()[1:]]
        if len(coords) < 6 or len(coords) % 2:
            continue                      # 폴리곤이 아닌 줄(bbox 등)은 무시
        xs.extend(coords[0::2])
        ys.extend(coords[1::2])
    if not xs:
        return None
    return (min(xs) * src_w, min(ys) * src_h, max(xs) * src_w, max(ys) * src_h)


def auto_crop_rect(src_w: int, src_h: int, bbox: tuple, auto: dict) -> tuple[int, int, int, int]:
    """
    라벨(wire) 위치에서 크롭 창을 직접 잡습니다 — center_x/center_y % 대신 쓰는 모드.

    축마다 따로: 크기 = 라벨 경계상자 변 × (1 + 2*margin), min_size 아래로는 내려가지
    않게 키우고 원본 크기로 클램프. 중심은 라벨 경계상자의 중심이며, 창이 이미지 밖으로
    나가면 안쪽으로 밀어 넣습니다.

    원본이 min_size 보다 작아 창을 더 키우지 못하는 경우가 있으므로(경고 대상),
    호출 쪽에서 결과 크기를 확인해야 합니다.
    """
    margin = max(float(auto.get("margin", 0.25)), 0.0)
    min_size = max(int(auto.get("min_size", 640) or 0), 0)
    x0, y0, x1, y1 = bbox

    out = []
    for lo, hi, src in ((x0, x1, src_w), (y0, y1, src_h)):
        size = min(max(int(round((hi - lo) * (1.0 + 2.0 * margin))), min_size, 1), src)
        start = int(round((lo + hi) / 2.0 - size / 2.0))
        out.append((min(max(start, 0), src - size), size))

    (x, w), (y, h) = out
    return x, y, w, h


def polygon_area(points: list) -> float:
    """신발끈 공식(부호 없는 면적)."""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def clip_polygon(points: list, x0: float, y0: float, x1: float, y1: float) -> list:
    """
    Sutherland–Hodgman 으로 폴리곤을 축 정렬 사각형에 클리핑.

    자르는 쪽(사각형)이 볼록하므로 오목한 입력에도 쓸 수 있지만, 오목 폴리곤이
    경계에서 두 조각으로 갈리면 조각을 잇는 변이 생깁니다. YOLO seg 라벨이
    인스턴스당 폴리곤 하나만 표현할 수 있어 어차피 조각을 나눠 담을 수 없으므로
    그대로 둡니다(와이어처럼 가늘고 긴 대상에서는 거의 문제가 되지 않음).
    """
    def inside(p, edge):
        if edge == 0: return p[0] >= x0
        if edge == 1: return p[0] <= x1
        if edge == 2: return p[1] >= y0
        return p[1] <= y1

    def intersect(p, q, edge):
        px, py = p
        qx, qy = q
        if edge in (0, 1):                       # 세로 경계 x = bound
            bound = x0 if edge == 0 else x1
            t = (bound - px) / (qx - px)         # inside 가 서로 다르면 qx != px
            return (bound, py + t * (qy - py))
        bound = y0 if edge == 2 else y1          # 가로 경계 y = bound
        t = (bound - py) / (qy - py)
        return (px + t * (qx - px), bound)

    out = list(points)
    for edge in range(4):
        if not out:
            return []
        buf, prev = [], out[-1]
        for cur in out:
            cur_in, prev_in = inside(cur, edge), inside(prev, edge)
            if cur_in:
                if not prev_in:
                    buf.append(intersect(prev, cur, edge))
                buf.append(cur)
            elif prev_in:
                buf.append(intersect(prev, cur, edge))
            prev = cur
        out = buf
    return out


def transform_label(line: str, src_w: int, src_h: int,
                    rect: tuple, min_area: float) -> str | None:
    """
    라벨 한 줄을 크롭 창 기준으로 변환. 살아남지 못하면 None.

    입력/출력 모두 YOLO seg 형식(`cls x1 y1 x2 y2 ...`, 0~1 정규화)이며,
    출력 좌표는 크롭 창 크기로 다시 정규화됩니다.
    """
    parts = line.split()
    coords = [float(v) for v in parts[1:]]
    if len(coords) < 6 or len(coords) % 2:
        return None                       # 폴리곤이 아님(bbox 등) → 크롭 시 폐기

    pts = [(coords[i] * src_w, coords[i + 1] * src_h) for i in range(0, len(coords), 2)]
    before = polygon_area(pts)
    if before <= 0:
        return None

    x, y, w, h = rect
    clipped = clip_polygon(pts, x, y, x + w, y + h)
    if len(clipped) < 3 or polygon_area(clipped) / before < min_area:
        return None

    out = [parts[0]]
    for px, py in clipped:
        out.append(f"{min(max((px - x) / w, 0.0), 1.0):.6f}")
        out.append(f"{min(max((py - y) / h, 0.0), 1.0):.6f}")
    return " ".join(out)


# ── 전처리 단계 ─────────────────────────────────────────────────────────────

class Preprocessor(Stage):
    """소스들을 소스별 산출물 + 통합 data.yaml 의 학습용 데이터셋으로 만드는 단계."""

    config_name = "download_config.yaml"   # download.py 와 같은 파일을 읽습니다
    label = "preprocess"

    def __init__(self, stage=None, config_path=None, only=None) -> None:
        super().__init__(config_path)
        stages = self.cfg.get("stages")
        if not isinstance(stages, dict) or not stages:
            raise SystemExit(
                f"[{self.label}] {self.config_path.name} 에 stages 블록이 없습니다."
            )
        if stage is None or stage not in stages:
            # config 후보가 여럿이던 때와 같은 원칙 — 기본 스테이지를 조용히 고르지 않음
            raise SystemExit(
                f"[{self.label}] 구울 스테이지를 명시하세요 — 기본값으로 조용히 넘어가지 "
                f"않습니다.\n"
                f"  python {self.label}.py --stage <이름>\n"
                f"  {self.config_path.name} 의 stages 후보: {', '.join(stages)}"
                + (f"\n  없는 스테이지: {stage}" if stage is not None else "")
            )
        self.stage_name = stage
        self.stage_cfg = stages[stage] or {}
        self.only = list(only or [])
        self.final_names = normalize_names(self.cfg.get("names"))
        self.final_ids = {v: k for k, v in self.final_names.items()}  # name → id
        self.out = self.resolve(self.stage_cfg["out"])
        self._dropped_empty = 0       # 크롭 후 인스턴스가 모두 사라져 제외된 이미지 수
        self._dropped_instances = 0   # 크롭 창 밖으로 나가 폐기된 인스턴스 수
        self._undersized = 0          # auto_crop 창이 min_size 에 못 미친 이미지 수
        self._undersized_min = 0      # 그 중 가장 작았던 변 (px)
        self.log(f"stage: {stage} → {self.out}")   # 무엇을 굽는지 항상 남긴다

    @classmethod
    def stage_candidates(cls) -> str:
        """download_config.yaml 의 stages 키들 — 오류 메시지/CLI 도움말용."""
        try:
            stages = load_yaml(resolve(cls.config_name)).get("stages") or {}
        except FileNotFoundError:
            return "(없음)"
        return ", ".join(stages) or "(없음)"

    # -- 소스 선택 -------------------------------------------------------

    def stage_sources(self) -> list:
        """이 스테이지(use)의 소스들 — use 값이 sources 정의를 얕게(필드 단위) 덮어씀."""
        by_name = {s["name"]: s for s in self.cfg.get("sources") or []}
        use = self.stage_cfg.get("use") or {}
        unknown = [n for n in use if n not in by_name]
        if unknown:
            raise SystemExit(
                f"[{self.label}] stage '{self.stage_name}' 의 use 에 없는 소스 이름: "
                f"{', '.join(unknown)}\n"
                f"  sources 에 있는 이름: {', '.join(by_name) or '(없음)'}"
            )
        return [{**by_name[n], **(use[n] or {})} for n in use]   # yaml 순서 유지

    def selected_sources(self) -> list:
        """--only 로 좁힌 소스만 (없으면 스테이지 전체). 이름이 틀리면 바로 알려줍니다."""
        sources = self.stage_sources()
        if not self.only:
            return sources

        by_name = {s["name"]: s for s in sources}
        unknown = [t for t in self.only if t not in by_name]
        if unknown:
            raise SystemExit(
                f"[{self.label}] --only 에 없는 소스 이름: {', '.join(unknown)}\n"
                f"  stage '{self.stage_name}' 의 소스: {', '.join(by_name) or '(없음)'}"
            )
        return [by_name[t] for t in self.only]

    # -- 소스 읽기 -------------------------------------------------------

    @staticmethod
    def find_split_image_dir(src_dir: Path, dy: dict, key: str):
        """data.yaml 의 split 경로(train/val/...)를 실제 이미지 폴더로 해석."""
        val = dy.get(key)
        if not val:
            return None
        p = Path(str(val))
        parts = [x for x in p.parts if x not in ("..", ".")]  # Roboflow '../train/images' 대응
        cand = src_dir / Path(*parts) if parts else src_dir
        if cand.exists():
            return cand
        base = src_dir / str(dy.get("path", "."))
        cand2 = (base / p).resolve()
        return cand2 if cand2.exists() else None

    @staticmethod
    def labels_dir_for(images_dir: Path) -> Path:
        """.../images/... → .../labels/... (YOLO 관례)."""
        parts = list(images_dir.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                break
        return Path(*parts)

    def prune_orphan_labels(self, img_dir: Path, lbl_dir: Path, name: str) -> None:
        """
        이미지가 지워진 라벨(.txt)을 원천에서 함께 지운다 — 데이터셋 항목 삭제는
        이미지 파일만 지우면 되도록 (짝 라벨은 다음 실행에서 여기서 정리).

        이미지가 하나도 없는데 라벨만 있으면 경로 실수일 가능성이 높으므로,
        지우지 않고 경고만 합니다 (라벨 폴더를 통째로 날리는 사고 방지).
        """
        if not lbl_dir.is_dir():
            return
        stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        orphans = sorted(p for p in lbl_dir.glob("*.txt") if p.stem not in stems)
        if not orphans:
            return
        if not stems:
            self.log(f"{name}: 이미지는 0장인데 라벨만 {len(orphans)}개 있습니다 — "
                     f"경로 실수 같아 지우지 않습니다 ({img_dir})")
            return
        for p in orphans:
            p.unlink()
        self.log(f"{name}: 이미지가 지워진 라벨 {len(orphans)}개 자동 삭제 "
                 f"(예: {orphans[0].name})")

    def remap_lines(self, text: str, src_names: dict, class_map: dict) -> list:
        """라벨의 class id 를 최종 id 로 remap. class_map 에 없는 클래스 줄은 제거."""
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                src_id = int(float(parts[0]))
            except (ValueError, IndexError):
                continue
            src_name = src_names.get(src_id)
            if src_name is None or src_name not in class_map:
                continue  # 우리가 원하는 클래스가 아니면 버림
            parts[0] = str(self.final_ids[class_map[src_name]])
            out.append(" ".join(parts))
        return out

    @staticmethod
    def crop_config(src: dict) -> dict:
        """소스별 크롭 설정. 공유 기본값 없이 소스마다 독립입니다."""
        return dict(src.get("crop") or {})

    @staticmethod
    def crop_summary(crop: dict) -> str:
        """로그 한 줄용 크롭 요약."""
        if not crop.get("enabled"):
            return "crop off"
        auto = crop.get("auto_crop") or {}
        if auto.get("enabled"):
            return (f"auto crop (margin {auto.get('margin', 0.25)}, "
                    f"min {auto.get('min_size', 640)}px)")
        return (f"crop {crop.get('width') or '원본'}×{crop.get('height') or '원본'} "
                f"@{crop.get('center_x', 50)}%,{crop.get('center_y', 50)}%")

    def collect_source(self, src: dict) -> list:
        """
        소스에서 (img_path, [최종 라벨 줄], rect) 목록 수집.

        크롭이 켜져 있으면 여기서 라벨까지 변환합니다 — 분할(split)보다 먼저 해야
        '크롭 후 대상이 사라진 이미지'가 train/val 개수에 섞이지 않습니다.
        """
        src_dir = self.resolve(src["path"])
        dy_path = src_dir / "data.yaml"
        if not dy_path.exists():
            raise FileNotFoundError(
                f"{src['name']}: data.yaml 이 없습니다 → {dy_path}\n"
                f"공개셋이면 python download.py 를 먼저 돌리세요.\n"
                f"내 카메라 원본(kimm)이면 두 줄짜리로 만들어 두면 됩니다:\n"
                f"    names: {{0: wire}}\n"
                f"    train: images"
            )
        with dy_path.open("r", encoding="utf-8") as f:
            dy = yaml.safe_load(f) or {}

        src_names = normalize_names(dy.get("names"))
        class_map = src.get("class_map", {})
        crop = self.crop_config(src)
        cropping = bool(crop.get("enabled"))
        auto = crop.get("auto_crop") or {}
        auto_on = cropping and bool(auto.get("enabled"))
        min_size = max(int(auto.get("min_size", 640) or 0), 0)
        min_area = float(crop.get("min_area", 0.10))
        if cropping:
            from PIL import Image   # 크롭을 쓸 때만 필요 (pillow 미설치여도 병합은 동작)

        items = []
        seen = set()  # 같은 이미지가 여러 split 에 중복 등록되는 것 방지
        for key in ("train", "val", "valid", "test"):
            img_dir = self.find_split_image_dir(src_dir, dy, key)
            if img_dir is None:
                continue
            lbl_dir = self.labels_dir_for(img_dir)
            self.prune_orphan_labels(img_dir, lbl_dir, src["name"])
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXTS or img.name in seen:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                lines = self.remap_lines(lbl.read_text(encoding="utf-8"),
                                         src_names, class_map)
                if not lines:
                    continue

                rect = None
                if cropping:
                    with Image.open(img) as im:
                        src_w, src_h = im.width, im.height
                    if auto_on:
                        # 창을 라벨에서 잡으므로 remap 된 라벨이 먼저 있어야 합니다.
                        bbox = label_bbox(lines, src_w, src_h)
                        if bbox is None:
                            self._dropped_empty += 1   # 폴리곤이 없으면 중심을 못 잡음
                            continue
                        rect = auto_crop_rect(src_w, src_h, bbox, auto)
                        got = min(rect[2], rect[3])
                        if got < min_size:             # 원본이 작아 더 못 키운 경우
                            self._undersized += 1
                            self._undersized_min = (min(self._undersized_min, got)
                                                    if self._undersized_min else got)
                    else:
                        rect = crop_rect(src_w, src_h, crop)
                    kept = [t for t in (transform_label(ln, src_w, src_h, rect, min_area)
                                        for ln in lines) if t]
                    self._dropped_instances += len(lines) - len(kept)
                    lines = kept
                    if not lines:
                        self._dropped_empty += 1
                        continue

                items.append((img, lines, rect))
                seen.add(img.name)
        return items

    # -- 쓰기 ------------------------------------------------------------

    @staticmethod
    def clear_dir(d: Path) -> None:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_image(src_img: Path, dst: Path, rect, resize) -> None:
        """크롭/리사이즈해서 저장. 둘 다 없으면 원본을 그대로 복사(재인코딩 없음)."""
        if rect is None and not resize:
            shutil.copyfile(src_img, dst)
            return

        from PIL import Image
        with Image.open(src_img) as im:
            if rect is not None:
                x, y, w, h = rect
                im = im.crop((x, y, x + w, y + h))
            if resize:
                size = (int(resize), int(resize)) if isinstance(resize, (int, float)) else tuple(resize)
                im = im.resize(size, Image.BICUBIC)
            if dst.suffix.lower() in (".jpg", ".jpeg"):
                im.save(dst, quality=95, subsampling=0)
            else:
                im.save(dst)

    def write_item(self, src: dict, item: tuple, split: str, reps: int,
                   dirs: dict, lbls: dict) -> None:
        """항목 하나를 split 폴더에 reps 개(oversample) 만큼 기록."""
        img, lines, rect = item
        resize = src.get("resize")
        text = "\n".join(lines) + "\n"
        first = None

        for r in range(reps):
            suffix = "" if r == 0 else f"_os{r}"
            stem = f"{src['name']}__{img.stem}{suffix}"
            dst = dirs[split] / (stem + img.suffix.lower())
            if first is None:
                self.write_image(img, dst, rect, resize)   # 인코딩은 한 번만
                first = dst
            else:
                shutil.copyfile(first, dst)
            (lbls[split] / (stem + ".txt")).write_text(text, encoding="utf-8")

    # -- 미리보기 --------------------------------------------------------

    def write_preview(self, count: int) -> None:
        """
        산출물에서 몇 장을 골라 라벨 폴리곤을 그려 저장 — 크롭·라벨 변환이 맞는지
        눈으로 확인하는 용도. 소스가 골고루 섞이도록 정렬 후 균등 간격으로 뽑습니다.
        """
        from PIL import Image, ImageDraw

        preview_dir = self.out / "_preview"
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for split in ("train", "val", "test"):
            images = []
            for sub in sorted(d for d in self.out.iterdir()
                              if d.is_dir() and d.name != "_preview"):
                images += sorted((sub / f"images/{split}").glob("*"))
            images = [p for p in images if p.suffix.lower() in IMG_EXTS]
            if not images:
                continue
            take = min(count, len(images))
            step = len(images) / take
            for i in range(take):
                path = images[int(i * step)]
                lbl = self.labels_dir_for(path.parent) / (path.stem + ".txt")
                with Image.open(path) as im:
                    canvas = im.convert("RGB")
                    draw = ImageDraw.Draw(canvas)
                    for line in lbl.read_text(encoding="utf-8").splitlines():
                        v = line.split()
                        if len(v) < 7:
                            continue
                        pts = [(float(v[j]) * canvas.width, float(v[j + 1]) * canvas.height)
                               for j in range(1, len(v) - 1, 2)]
                        draw.polygon(pts, outline=(255, 40, 40))
                    canvas.save(preview_dir / f"{split}__{path.stem}.png")
                written += 1

        self.log(f"미리보기 {written}장 → {preview_dir}")

    # -- 실행 ------------------------------------------------------------

    def write_data_yaml(self) -> tuple[list, list, list]:
        """디스크에 있는 소스 산출물 전체를 묶는 out/data.yaml 을 다시 쓴다."""
        def has_images(d: Path) -> bool:
            return d.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in d.iterdir())

        self.out.mkdir(parents=True, exist_ok=True)
        all_names = [s["name"] for s in self.stage_sources()]  # 이 스테이지의 소스만 묶음
        train = [f"{n}/images/train" for n in all_names
                 if has_images(self.out / n / "images/train")]
        val = [f"{n}/images/val" for n in all_names
               if has_images(self.out / n / "images/val")]
        test = [f"{n}/images/test" for n in all_names
                if has_images(self.out / n / "images/test")]

        stale = sorted(d.name for d in self.out.iterdir()
                       if d.is_dir() and d.name != "_preview" and d.name not in all_names)
        if stale:
            self.log(f"경고: 이 스테이지의 use 에 없는 산출물 폴더는 data.yaml 에서 "
                     f"제외했습니다: {', '.join(stale)} (안 쓰면 지우세요)")

        data_yaml = {
            "path": str(self.out.resolve()),
            "train": train,
            "val": val,
            "names": self.final_names,
        }
        if test:   # test 는 선택 사항 — 없으면 키 자체를 넣지 않음 (빈 리스트 금지)
            data_yaml["test"] = test
        with (self.out / "data.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)
        return train, val, test

    @staticmethod
    def split_group(stem: str) -> str:
        """
        train/val 분할의 그룹 키. Roboflow export 는 같은 원본의 증강 사본을
        `<원본>_jpg.rf.<hash>` 이름으로 여러 split 에 흩어 놓으므로, `.rf.` 앞부분으로
        묶어 근중복이 train/val 에 갈라 들어가는 누수를 막습니다. 그 외 파일은
        stem 자체가 그룹(= 이미지 단위 분할)입니다.

        download.py 가 붙이는 000000__ 인덱스 접두어는 rf 사본마다 달라 그룹을
        깨므로 rf 사본에서만 벗깁니다. 그 외 파일(예: 앱 캡처)은 접두어를 포함한
        stem 전체가 그룹이어야 합니다 — 캡처 파일명은 인덱스를 빼면 세션
        타임스탬프만 남아 세션 전체가 한 그룹으로 뭉쳐 버리기 때문입니다.
        """
        bare = re.sub(r"^\d+__", "", stem)
        return bare.split(".rf.")[0] if ".rf." in bare else stem

    def run(self) -> None:
        sources = self.selected_sources()
        all_names = [s["name"] for s in self.stage_sources()]
        if len(sources) != len(all_names):
            self.log(f"대상(--only): {', '.join(s['name'] for s in sources)} "
                     f"— 스테이지 전체 {len(all_names)}개 중. 이 소스만 다시 굽고, "
                     f"다른 소스의 기존 산출물은 유지됩니다.")

        n_train = n_val = n_test = 0
        for src in sources:
            out_dir = self.out / src["name"]
            dirs = {sp: out_dir / f"images/{sp}" for sp in ("train", "val", "test")}
            lbls = {sp: out_dir / f"labels/{sp}" for sp in ("train", "val", "test")}
            for d in (*dirs.values(), *lbls.values()):
                self.clear_dir(d)  # 이 소스의 산출물만 비움 (재실행 시 깨끗하게)

            items = self.collect_source(src)
            if not items:
                self.log(f"{src['name']}: 대상 이미지 0장 (경로/class_map/크롭 확인)")
                continue

            val_ratio = float(src.get("val_ratio", 0.0))
            test_ratio = float(src.get("test_ratio", 0.0))
            oversample = max(1, int(src.get("oversample", 1)))

            # 소스 내부에서 먼저 train/val/test 분할 → oversample 전에 나눠 누수 차단.
            # 시드는 소스별로 파생 — 소스를 추가/제거해도 다른 소스의 분할이 안 바뀜.
            # 그룹(원본) 단위로 val → test 순서로 채우고, 남는 그룹이 전부 train.
            rng = random.Random(f"{self.cfg.get('seed', 0)}:{src['name']}")
            groups: dict = {}
            for i, (img, _, _) in enumerate(items):
                groups.setdefault(self.split_group(img.stem), []).append(i)
            keys = sorted(groups)
            rng.shuffle(keys)

            split_of = {i: "train" for i in range(len(items))}
            ki = 0
            for sp, want in (("val", int(round(len(items) * val_ratio))),
                             ("test", int(round(len(items) * test_ratio)))):
                have = 0
                while have < want and ki < len(keys):
                    for i in groups[keys[ki]]:
                        split_of[i] = sp
                    have += len(groups[keys[ki]])
                    ki += 1

            counts = {"train": 0, "val": 0, "test": 0}
            for i, item in enumerate(items):
                split = split_of[i]
                reps = oversample if split == "train" else 1  # oversample 은 train 만
                self.write_item(src, item, split, reps, dirs, lbls)
                counts[split] += reps

            n_train += counts["train"]
            n_val += counts["val"]
            n_test += counts["test"]
            how = self.crop_summary(self.crop_config(src))
            self.log(f"{src['name']}: train +{counts['train']} (oversample x{oversample}), "
                     f"val +{counts['val']}, test +{counts['test']}  [{how}]")

        train_dirs, val_dirs, test_dirs = self.write_data_yaml()

        if self._dropped_instances or self._dropped_empty:
            self.log(f"크롭으로 폐기된 인스턴스 {self._dropped_instances}개 "
                     f"(그 중 대상이 하나도 안 남아 제외된 이미지 {self._dropped_empty}장). "
                     f"많으면 center_x/center_y 또는 min_area 를 조정하세요.")

        if self._undersized:
            self.log(f"경고: auto_crop 창이 min_size 에 못 미친 이미지 {self._undersized}장 "
                     f"(가장 작은 변 {self._undersized_min}px). 원본이 그보다 작아 더 키울 수 "
                     f"없었습니다 — 학습 해상도가 떨어집니다.")

        preview = self.cfg.get("preview") or {}
        if preview.get("enabled"):
            self.write_preview(int(preview.get("count", 12)))

        self.log(f"완료 → {self.out}")
        self.log(f"이번 실행: train {n_train}장, val {n_val}장, test {n_test}장")
        self.log(f"data.yaml 에 묶인 소스 폴더: train {len(train_dirs)}개, val {len(val_dirs)}개"
                 + (f", test {len(test_dirs)}개" if test_dirs else ""))
        self.log(f"data.yaml: {self.out / 'data.yaml'}")
        if not val_dirs:
            # 빈 val 로 학습에 들어가면 Ultralytics 가 데이터셋 빌드에서 죽습니다.
            # 산출물은 이미 다 썼으므로 여기서 멈춰도 이번 작업은 보존됩니다.
            raise SystemExit(
                f"[{self.label}] val 산출물이 하나도 없습니다 — 이 data.yaml 로 train.py 를 "
                f"돌리면 빈 검증셋으로 죽습니다.\n"
                f"  val_ratio > 0 인 소스를 처리해 val 을 만들어 두세요 "
                f"(단일/stage2 는 인도메인 kimm, stage1 은 공개셋에서 뗍니다).\n"
                f"  (--only 로 일부만 굽는 중이었다면 먼저 전체를 한 번 처리해야 합니다. "
                f"이번에 만든 train 산출물은 그대로 유지됩니다.)"
            )
        self.log("다음: python train.py --config <train_config*.yaml>")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="공개 + 로컬 데이터를 학습용 YOLO seg 데이터셋으로 전처리",
        epilog="download_config.yaml 의 스테이지 후보: " + Preprocessor.stage_candidates(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # required=True 대신 생성자가 실제 config 의 stages 를 읽어 후보와 함께 거절합니다.
    ap.add_argument(
        "--stage", metavar="NAME",
        help="구울 스테이지 (필수 — 기본값 없음). download_config.yaml 의 stages 키",
    )
    ap.add_argument(
        "--only", nargs="+", metavar="SOURCE",
        help="이 소스만 다시 굽기 (다른 소스의 기존 산출물은 유지)",
    )
    args = ap.parse_args()
    Preprocessor(args.stage, only=args.only).run()


if __name__ == "__main__":
    main()
