"""
공개 + 로컬 데이터를 학습용 YOLO seg 데이터셋으로 전처리(크롭 · 리사이즈 · 클래스 통일).

폴더 규약:
- 원천은 datasets/raw/<이름>/ (kimm = 앱 REC 카메라 원본, rf_* = download.py 공개셋).
- 산출물은 소스별로 out/<이름>/images|labels/{train,val,test} 에 씁니다.
- out/data.yaml 하나가 디스크에 있는 소스 산출물 전체를 묶어 train.py 가 읽습니다
  (Ultralytics 는 각 split 에 폴더 목록을 지원). test 는 있을 때만 키를 넣습니다.

설계 원칙:
- 전처리에는 스테이지가 없습니다 — configs/preprocess.yaml 하나가 소스 구성·크롭·비율
  등 전부를 소유하고, 소스마다 raw 를 processed/<이름>/ 으로 한 번만 굽습니다.
  스테이지(공개 warm-up → in-domain 적응)는 train/eval config 가 어느 data.yaml 을
  읽을지로 정합니다: 소스별 processed/<이름>/data.yaml + 통합 processed/data.yaml.
  config 후보가 하나뿐이라 --config 없이 자동으로 읽힙니다.
- 분할(train/val/test)은 소스 내부에서, oversample 전에 그룹(원본) 단위로 나눕니다.
  Roboflow 증강 사본(..._jpg.rf.<hash>)은 원본 단위로 묶여 같은 split 에만 들어갑니다
  (근중복이 train/val 에 갈라 들어가는 누수 방지). 앱 캡처(kimm)는 group_by: session 으로
  녹화 세션 폴더 통째로 나눕니다 — 연속 프레임끼리가 근중복이기 때문 (split_items).
- test 는 test_ratio 로 뗍니다 — 최종 모델에 딱 한 번 쓰는 평가용 (eval.py --split test).
- oversample 은 각 소스의 train 쪽에만 물리 복제로 적용 (val/test 누수 없음).
- 클래스 선택/이름통일은 class_map(이름 기반)이 담당 → source index 차이에 안전.
- task: obb 면 최종 라벨을 회전 사각형(OBB, 꼭지점 4개)으로 변환합니다 — 크롭·클리핑이
  끝난 폴리곤에 최종 픽셀 공간에서 최소 면적 회전 사각형을 피팅 (RMSE 최적화가 아니라
  기하 해법 — 결정론적이고 폴리곤 전체를 항상 덮음). 피팅 품질은 로그로 보고합니다.
- 크롭은 이미지와 폴리곤 라벨을 함께 변환합니다. 창 밖으로 나간 인스턴스는 잘리고,
  남은 면적이 min_area 미만이면 인스턴스째 폐기, 살아남은 인스턴스가 하나도 없으면
  그 이미지는 데이터셋에서 제외됩니다(기존 동작과 동일: 대상 없는 이미지는 안 넣음).
- 크롭 인자는 소스마다 독립입니다(공유 기본값 없음). 창을 %로 고정하거나
  (width/height/center_x/center_y), auto_crop 으로 이미지마다 라벨에서 직접 잡습니다.
- config 의 targets: 에 이름을 적으면 그 소스만 다시 굽습니다 (생략하면 전체). 산출물이
  소스별 폴더라서 나머지 소스의 기존 산출물은 유지되고, data.yaml 만 매번 다시 묶입니다.
- 마음에 안 드는 이미지는 원천(raw)에서 이미지 파일만 지우면 됩니다 — 다음 실행에서
  짝 라벨(.txt)이 자동 삭제됩니다 (이미지가 하나도 없는 폴더는 경로 실수로 보고 보호).
- 라벨에 지정 클래스(class_map)가 하나도 없는 데이터는 이미지+라벨 쌍째 원천에서
  자동 삭제됩니다 (예: rf_a 에 wire 없는 데이터). 단, 소스에 지정 클래스가 있는
  데이터가 한 쌍도 없으면 class_map 오타로 보고 지우지 않습니다.

증강은 여기서 하지 않습니다. Ultralytics 가 학습 중에 온라인 증강을 하므로
train.*.yaml 의 증강 하이퍼파라미터로 조절하고, 결과 확인은
`python train.py` 의 preview_aug 단계를 쓰세요 (offline 증강은 다양성이 오히려 줄고
온라인 증강과 이중으로 겹칩니다).

각 소스 폴더엔 data.yaml (names + train/val 경로) 이 있어야 함.
  · Roboflow export 는 기본 포함.
  · 로컬(kimm)은 최소 형식으로 하나 작성:  names: {0: wire}\n train: images
    (labeller/server.py 가 띄울 때 자동으로 만들어 줍니다.)
  · 소스 path 가 글롭(`raw/kimm_*` — 앱 REC 세션 폴더들)이면 폴더 전부를 한 소스로
    묶습니다. 폴더마다 data.yaml 을 두기 번거로우면 그 소스에 `names:` 를 적으면
    data.yaml 없는 폴더는 그것(train: images)으로 읽습니다.

사용법:
    python preprocess.py        # configs/preprocess.yaml 자동 (후보 1개)
"""
from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

import yaml

from common import IMG_EXTS, INDEX_RE, Stage, normalize_names, resolve


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


def convex_hull(points: list) -> list:
    """Andrew monotone chain — 반시계 방향 볼록 껍질. 중복점은 제거."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def min_area_rect(points: list) -> tuple[list, float]:
    """
    점들을 모두 덮는 최소 면적 회전 사각형 (rotating calipers).

    최적 사각형은 항상 볼록 껍질의 어느 한 변과 평행하므로, 껍질의 변마다
    그 방향으로 정렬한 경계 사각형을 재고 가장 작은 것을 고릅니다.
    반환: (꼭지점 4개 [(x, y) × 4, 인접 순서], 면적). 퇴화(일직선)면 면적 0.
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        p0 = hull[0]
        p1 = hull[-1]
        return [p0, p1, p1, p0], 0.0

    best = None
    n = len(hull)
    for i in range(n):
        x0, y0 = hull[i]
        ex, ey = hull[(i + 1) % n][0] - x0, hull[(i + 1) % n][1] - y0
        length = math.hypot(ex, ey)
        if length == 0:
            continue
        ux, uy = ex / length, ey / length          # 변 방향축 u, 법선축 v = (-uy, ux)
        us = [(px - x0) * ux + (py - y0) * uy for px, py in hull]
        vs = [(py - y0) * ux - (px - x0) * uy for px, py in hull]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best is None or area < best[0]:
            best = (area, x0, y0, ux, uy, min(us), max(us), min(vs), max(vs))

    area, x0, y0, ux, uy, u0, u1, v0, v1 = best

    def corner(u, v):
        return (x0 + u * ux - v * uy, y0 + u * uy + v * ux)

    return [corner(u0, v0), corner(u1, v0), corner(u1, v1), corner(u0, v1)], area


def polygon_to_obb(line: str, w: int, h: int) -> tuple[str, float] | None:
    """
    seg 라벨 한 줄(정규화 폴리곤)을 OBB 한 줄(꼭지점 4개, 정규화)로 변환.

    피팅은 최종 이미지의 **픽셀 공간**에서 합니다 — 정규화 공간은 가로세로
    비율이 뭉개져 각도가 왜곡되기 때문. 반환은 (obb 줄, 피팅 품질) 이며
    품질 = 폴리곤 면적 / OBB 면적 (OBB 가 폴리곤을 덮으므로 IoU 와 같음).
    폴리곤이 아니거나 퇴화면 None.
    """
    parts = line.split()
    coords = [float(v) for v in parts[1:]]
    if len(coords) < 6 or len(coords) % 2:
        return None
    pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
    area = polygon_area(pts)
    if area <= 0:
        return None
    corners, rect_area = min_area_rect(pts)
    if rect_area <= 0:
        return None

    out = [parts[0]]
    for x, y in corners:
        out.append(f"{min(max(x / w, 0.0), 1.0):.6f}")
        out.append(f"{min(max(y / h, 0.0), 1.0):.6f}")
    return " ".join(out), min(area / rect_area, 1.0)



class Cropper:
    """
    소스 하나의 crop 설정 해석 + 이미지별 창 계산·라벨 변환 + 크롭 통계.

    흩어져 있던 설정 파싱·인라인 크롭 로직·카운터를 묶은 것으로, 기하 계산 자체는
    모듈의 순수 함수(crop_rect · auto_crop_rect · transform_label)를 그대로 씁니다
    (selftest 가 직접 검증). 설정은 소스마다 독립입니다 — 공유 기본값 없음.
    """

    def __init__(self, crop: dict | None) -> None:
        self.cfg = dict(crop or {})
        self.enabled = bool(self.cfg.get("enabled"))
        self.auto = self.cfg.get("auto_crop") or {}
        self.auto_on = self.enabled and bool(self.auto.get("enabled"))
        self.min_size = max(int(self.auto.get("min_size", 640) or 0), 0)
        self.min_area = float(self.cfg.get("min_area", 0.10))
        self.dropped_empty = 0       # 대상이 하나도 안 남아 제외된 이미지 수
        self.dropped_instances = 0   # 창 밖으로 나가 폐기된 인스턴스 수
        self.undersized = 0          # auto_crop 창이 min_size 에 못 미친 이미지 수
        self.undersized_min = 0      # 그 중 가장 작았던 변 (px)

    def apply(self, img: Path, lines: list) -> tuple | None:
        """
        이미지 하나의 (rect, 변환된 라벨 줄) 계산. 크롭 후 남는 대상이 없으면 None
        (호출 쪽이 그 이미지를 데이터셋에서 제외). 크롭이 꺼져 있으면 (None, 원본 줄).
        """
        if not self.enabled:
            return None, lines

        from PIL import Image   # 크롭을 쓸 때만 필요 (pillow 미설치여도 병합은 동작)
        with Image.open(img) as im:
            src_w, src_h = im.width, im.height

        if self.auto_on:
            # 창을 라벨에서 잡으므로 remap 된 라벨이 먼저 있어야 합니다.
            bbox = label_bbox(lines, src_w, src_h)
            if bbox is None:
                self.dropped_empty += 1    # 폴리곤이 없으면 중심을 못 잡음
                return None
            rect = auto_crop_rect(src_w, src_h, bbox, self.auto)
            got = min(rect[2], rect[3])
            if got < self.min_size:        # 원본이 작아 더 못 키운 경우
                self.undersized += 1
                self.undersized_min = (min(self.undersized_min, got)
                                       if self.undersized_min else got)
        else:
            rect = crop_rect(src_w, src_h, self.cfg)

        kept = [t for t in (transform_label(ln, src_w, src_h, rect, self.min_area)
                            for ln in lines) if t]
        self.dropped_instances += len(lines) - len(kept)
        if not kept:
            self.dropped_empty += 1
            return None
        return rect, kept

    def summary(self) -> str:
        """로그 한 줄용 크롭 요약."""
        if not self.enabled:
            return "crop off"
        if self.auto_on:
            return (f"auto crop (margin {self.auto.get('margin', 0.25)}, "
                    f"min {self.min_size}px)")
        return (f"crop {self.cfg.get('width') or '원본'}×{self.cfg.get('height') or '원본'} "
                f"@{self.cfg.get('center_x', 50)}%,{self.cfg.get('center_y', 50)}%")


# ── 전처리 단계 ─────────────────────────────────────────────────────────────

class Preprocessor(Stage):
    """
    소스들을 소스별 산출물 + data.yaml 의 학습용 데이터셋으로 만드는 단계.

    config 는 configs/preprocess.yaml 하나뿐입니다 (후보 1개 → 인자 없이 자동).
    전처리에는 스테이지가 없습니다 — 소스마다 raw 를 processed/<이름>/ 으로 한 번
    굽고, 스테이지 구분은 train/eval config 가 어느 data.yaml 을 읽을지로 정합니다
    (소스별 data.yaml + 전체 통합 data.yaml 을 모두 씁니다). download 쪽과는
    datasets/raw/<이름>/ 경로 규약만 공유합니다.
    """

    config_glob = "preprocess.yaml"
    label = "preprocess"

    def __init__(self, config_path=None, task: str | None = None, out: str | None = None) -> None:
        super().__init__(config_path)

        # task/out 은 실행 인자로 덮어쓸 수 있음 — 같은 소스 정의로 seg 와 obb 산출물을
        # 따로 굽기 위해 (예: --task obb --out ../datasets/processed_obb). 소스 정의를
        # 복제한 두 번째 config 를 두지 않으려는 것.
        if task:
            self.cfg["task"] = task
        if out:
            self.cfg["out"] = out
        for key in ("out", "sources"):
            if not self.cfg.get(key):
                self.fail(f"{key} 가 없습니다 — 기본값으로 정하지 않습니다.")
        self.targets = list(self.cfg.get("targets") or [])
        self.task = str(self.cfg.get("task"))
        if self.task not in ("seg", "obb"):
            self.fail(f"task 는 seg 또는 obb 여야 합니다: {self.task}")
        self.final_names = normalize_names(self.cfg.get("names"))
        self.final_ids = {v: k for k, v in self.final_names.items()}  # name → id
        self.out = self.resolve(self.cfg["out"])
        self._dropped_empty = 0       # 크롭 후 인스턴스가 모두 사라져 제외된 이미지 수
        self._dropped_instances = 0   # 크롭 창 밖으로 나가 폐기된 인스턴스 수
        self._undersized = 0          # auto_crop 창이 min_size 에 못 미친 이미지 수
        self._undersized_min = 0      # 그 중 가장 작았던 변 (px)
        self._obb_count = 0           # obb 로 변환된 인스턴스 수
        self._obb_fit_sum = 0.0       # 피팅 품질(폴리곤/OBB 면적비) 누적
        self._obb_fit_min = 1.0
        self._obb_low = 0             # 품질 0.5 미만 (휘었거나 노이즈 라벨 의심)
        self._obb_dropped = 0         # 퇴화 등으로 변환 못 해 폐기한 라벨 수
        self.log(f"task: {self.task} → {self.out}")   # 무엇을 굽는지 항상 남긴다

    # -- 소스 선택 -------------------------------------------------------

    def all_sources(self) -> list:
        """config 의 sources 그대로 (yaml 순서 유지)."""
        return list(self.cfg.get("sources") or [])

    def selected_sources(self) -> list:
        """config 의 targets 로 좁힌 소스만 (없으면 전체). 이름이 틀리면 바로 알려줍니다."""
        sources = self.all_sources()
        if not self.targets:
            return sources

        by_name = {s["name"]: s for s in sources}
        unknown = [t for t in self.targets if t not in by_name]
        if unknown:
            self.fail(f"targets 에 없는 소스 이름: {', '.join(unknown)}\n"
                      f"  sources 의 이름: {', '.join(by_name) or '(없음)'}")
        return [by_name[t] for t in self.targets]

    # -- 소스 읽기 -------------------------------------------------------

    def source_dirs(self, src: dict) -> list:
        """
        소스 하나가 실제로 읽을 폴더들 (이름순).

        path 에 glob 문자(`*?[`)가 있으면 매칭되는 폴더 **전부를 한 소스로** 다룹니다.
        앱(BaslerLiveView)의 REC 는 녹화 세션마다 `raw/kimm_YYMMDD_HHMMSS[_메모]/`
        를 새로 만들기 때문에(FrameRecorder), 세션이 늘어도 sources/targets 를
        건드리지 않게 하려는 것입니다. 산출물 폴더 이름과 targets 는 그대로 소스
        `name` 이고, 글롭은 "그 소스의 원본이 어디 흩어져 있는지"만 바꿉니다.

        글롭은 경로의 **마지막 한 칸**에만 씁니다 (`raw/kimm_*` ○, `raw/*/images` ✗).
        `images/` 가 없는 항목은 세지 않습니다 — raw/ 에 같이 있는 zip 이나
        아직 비어 있는 세션 폴더가 조용히 끼지 않게.
        """
        raw = str(src["path"])
        if not any(c in raw for c in "*?["):
            return [self.resolve(raw)]

        pat = self.resolve(raw)
        if any(c in str(pat.parent) for c in "*?["):
            self.fail(f"{src['name']}: glob 은 경로 마지막 칸에만 쓸 수 있습니다: {raw}")
        dirs = sorted(p for p in pat.parent.glob(pat.name)
                      if p.is_dir() and (p / "images").is_dir())
        if not dirs:
            self.fail(f"{src['name']}: {raw} 에 맞는 폴더가 없습니다 "
                      f"(images/ 를 가진 폴더만 셉니다).")
        self.log(f"{src['name']}: {raw} → 폴더 {len(dirs)}개 "
                 f"({dirs[0].name} … {dirs[-1].name})")
        return dirs

    def read_dir_cfg(self, src: dict, src_dir: Path) -> tuple:
        """
        폴더 하나의 (data.yaml 내용, 원본 클래스 이름).

        data.yaml 이 없으면 소스에 선언된 `names` 로 대체합니다 — 글롭 소스는 세션
        폴더가 계속 늘어나 폴더마다 손으로 둘 수가 없으니, config 에 한 번만 적게
        합니다. 둘 다 없으면 중단합니다 (클래스 이름은 추측하지 않습니다).
        """
        dy_path = src_dir / "data.yaml"
        if dy_path.exists():
            with dy_path.open("r", encoding="utf-8") as f:
                dy = yaml.safe_load(f) or {}
            return dy, normalize_names(dy.get("names"))
        if src.get("names"):
            return {"train": "images"}, normalize_names(src["names"])
        raise FileNotFoundError(
            f"{src['name']}: data.yaml 이 없습니다 → {dy_path}\n"
            f"공개셋이면 python download.py 를 먼저 돌리세요.\n"
            f"내 카메라 원본(kimm)이면 두 줄짜리로 만들어 두면 됩니다 "
            f"(라벨러가 띄울 때 자동으로 만듭니다):\n"
            f"    names: {{0: wire}}\n"
            f"    train: images\n"
            f"폴더가 여러 개(glob)라 폴더마다 두기 번거로우면, preprocess.yaml 의 "
            f"그 소스에 한 번만 적어도 됩니다:\n"
            f"    names: {{0: wire}}"
        )

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
    def labels_dir_for(images_dir: Path, labels: str = "labels") -> Path:
        """
        .../images/... → .../labels/... (YOLO 관례). labels 로 다른 라벨 폴더를 고를 수
        있음 — 예: 라벨러가 저장한 seg 폴리곤은 labels_seg/ (소스의 `labels:` 인자).
        """
        parts = list(images_dir.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = labels
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

    def collect_source(self, src: dict) -> list:
        """
        소스에서 (img_path, [최종 라벨 줄], rect, 세션 폴더명) 목록 수집.

        크롭이 켜져 있으면 여기서 라벨까지 변환합니다 — 분할(split)보다 먼저 해야
        '크롭 후 대상이 사라진 이미지'가 train/val 개수에 섞이지 않습니다.

        소스가 글롭이면 폴더 여러 개를 이어서 읽습니다 (source_dirs 참고). 마지막 값은
        이미지가 나온 소스 폴더 이름 — group_by: session 분할의 그룹 키입니다.
        """
        class_map = src.get("class_map", {})
        cropper = Cropper(src.get("crop"))

        items = []
        # 이미지 파일명 → 그 이미지의 경로. 같은 이미지가 여러 split 에 중복 등록되는
        # 것을 막고(원래 목적), 글롭으로 묶은 폴더들 사이의 파일명 충돌도 잡습니다 —
        # 산출물 이름이 `<소스>__<파일명>` 이라 충돌하면 조용히 덮어써집니다.
        owner: dict = {}
        no_class = []  # 라벨은 있지만 지정 클래스가 없는 (img, lbl) 쌍 — 마지막에 일괄 삭제
        for src_dir in self.source_dirs(src):
            dy, src_names = self.read_dir_cfg(src, src_dir)
            for key in ("train", "val", "valid", "test"):
                img_dir = self.find_split_image_dir(src_dir, dy, key)
                if img_dir is None:
                    continue
                lbl_dir = self.labels_dir_for(img_dir, str(src.get("labels") or "labels"))
                self.prune_orphan_labels(img_dir, lbl_dir, src["name"])
                self.collect_dir(src, src_dir.name, img_dir, lbl_dir, src_names,
                                 class_map, cropper, items, owner, no_class)

        self.prune_no_class_pairs(src["name"], class_map, items, no_class)
        self.absorb_crop_stats(cropper)
        return items

    def collect_dir(self, src: dict, session: str, img_dir: Path, lbl_dir: Path,
                    src_names: dict, class_map: dict, cropper, items: list,
                    owner: dict, no_class: list) -> None:
        """collect_source 의 폴더 하나 분량 — 이미지별로 라벨 remap → 크롭 → OBB 변환."""
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            prev = owner.get(img.name)
            if prev is not None:
                if prev == img:
                    continue   # 같은 파일이 여러 split 경로에 걸린 경우 — 한 번만
                self.fail(f"{src['name']}: 파일명이 겹칩니다 — 산출물에서 서로를 "
                          f"덮어씁니다. 한쪽을 지우거나 이름을 바꾸세요.\n"
                          f"  {prev}\n  {img}")
            owner[img.name] = img
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            lines = self.remap_lines(lbl.read_text(encoding="utf-8"),
                                     src_names, class_map)
            if not lines:
                no_class.append((img, lbl))
                continue

            cropped = cropper.apply(img, lines)
            if cropped is None:
                continue
            rect, lines = cropped

            if self.task == "obb":
                # 소스 라벨이 폴리곤(seg)인지 이미 OBB인지는 파일만 보고 구분하기
                # 어렵고(둘 다 '클래스 + 좌표 8개'가 될 수 있음) 잘못 넘겨짚으면
                # 라벨이 조용히 망가지므로, config 의 type 으로만 판단합니다.
                stype = src.get("type")
                if stype == "seg":
                    lines = self.obb_lines(img, lines, rect, src.get("resize"))
                    if not lines:
                        continue          # 변환 가능한 인스턴스가 없으면 이미지째 제외
                elif stype == "obb":
                    pass  # 이미 OBB 라벨이므로 변환 불필요
                elif stype is None:
                    self.fail(f"{src['name']}: task: obb 로 구우려면 소스마다 "
                              f"type: seg 또는 type: obb 가 필요합니다 "
                              f"(원본 라벨이 폴리곤인지 OBB인지 추측하지 않습니다).")
                else:
                    self.fail(f"{src['name']}: type 은 seg 또는 obb 여야 합니다: {stype}")
            items.append((img, lines, rect, session))

    def obb_lines(self, img: Path, lines: list, rect, resize) -> list:
        """
        seg 폴리곤 줄들을 OBB 줄로 변환 — split 결정 전에 해야 '변환 불가로 빠진
        이미지'가 train/val 개수에 섞이지 않습니다. 피팅은 라벨이 실제로 해석될
        최종 이미지 크기(resize > 크롭 창 > 원본 순) 기준 픽셀 공간에서 합니다.
        """
        if resize:
            w, h = ((int(resize), int(resize))
                    if isinstance(resize, (int, float)) else tuple(resize))
        elif rect is not None:
            w, h = rect[2], rect[3]
        else:
            from PIL import Image
            with Image.open(img) as im:
                w, h = im.width, im.height

        out = []
        for line in lines:
            got = polygon_to_obb(line, w, h)
            if got is None:
                self._obb_dropped += 1
                continue
            obb, fit = got
            out.append(obb)
            self._obb_count += 1
            self._obb_fit_sum += fit
            self._obb_fit_min = min(self._obb_fit_min, fit)
            if fit < 0.5:
                self._obb_low += 1
        return out

    def absorb_crop_stats(self, cropper: Cropper) -> None:
        """소스별 Cropper 의 통계를 실행 전체(run 마지막 로그) 카운터에 합산."""
        self._dropped_empty += cropper.dropped_empty
        self._dropped_instances += cropper.dropped_instances
        if cropper.undersized:
            self._undersized += cropper.undersized
            self._undersized_min = (min(self._undersized_min, cropper.undersized_min)
                                    if self._undersized_min else cropper.undersized_min)

    def prune_no_class_pairs(self, name: str, class_map: dict,
                             items: list, pairs: list) -> None:
        """
        라벨에 지정 클래스(class_map)가 하나도 없는 이미지+라벨 쌍을 원천에서 지운다
        (예: rf_a 에서 wire 없이 다른 클래스만 있는 데이터는 이 파이프라인에 쓸모없음).

        소스에 지정 클래스가 있는 데이터가 한 쌍도 없으면 class_map 오타일 가능성이
        높으므로, 지우지 않고 경고만 합니다 (소스를 통째로 날리는 사고 방지).
        크롭 때문에 대상이 사라진 이미지는 여기 해당하지 않습니다 (원본 라벨 기준).
        """
        if not pairs:
            return
        if not items:
            self.log(f"{name}: 지정 클래스({', '.join(class_map) or '없음'})가 있는 데이터가 "
                     f"한 쌍도 없습니다 — class_map 오타 같아 {len(pairs)}쌍을 지우지 않습니다")
            return
        for img, lbl in pairs:
            img.unlink()
            lbl.unlink()
        self.log(f"{name}: 지정 클래스 없는 데이터 {len(pairs)}쌍 삭제 "
                 f"(예: {pairs[0][0].name})")

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
        img, lines, rect, _ = item
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
        눈으로 확인하는 용도. 소스별로 out/<이름>/preview/ 에 count 장씩,
        split 을 섞어 정렬 후 균등 간격으로 뽑습니다.
        """
        from PIL import Image, ImageDraw

        written = 0
        for sub in sorted(d for d in self.out.iterdir()
                          if d.is_dir() and (d / "images").is_dir()):
            preview_dir = sub / "preview"
            if preview_dir.exists():
                shutil.rmtree(preview_dir)
            preview_dir.mkdir(parents=True, exist_ok=True)

            images = []
            for split in ("train", "val", "test"):
                images += sorted(p for p in (sub / f"images/{split}").glob("*")
                                 if p.suffix.lower() in IMG_EXTS)
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
                    canvas.save(preview_dir / f"{path.parent.name}__{path.stem}.png")
                written += 1
            self.log(f"미리보기: {sub.name}/preview/ {take}장")

        self.log(f"미리보기 총 {written}장 (소스별 preview/ 폴더)")

    # -- 실행 ------------------------------------------------------------

    def write_data_yaml(self) -> tuple[list, list, list]:
        """
        학습용 data.yaml 을 다시 쓴다 — 두 종류:
        · out/<이름>/data.yaml   소스 하나짜리 (train.stage1 → rf_a, train.stage2 → kimm)
        · out/data.yaml          디스크의 소스 산출물 전체 통합본 (train.single)
        스테이지 구분은 여기가 아니라 train/eval config 가 어느 파일을 읽을지로 정합니다.
        """
        def has_images(d: Path) -> bool:
            return d.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in d.iterdir())

        self.out.mkdir(parents=True, exist_ok=True)
        all_names = [s["name"] for s in self.all_sources()]
        train = [f"{n}/images/train" for n in all_names
                 if has_images(self.out / n / "images/train")]
        val = [f"{n}/images/val" for n in all_names
               if has_images(self.out / n / "images/val")]
        test = [f"{n}/images/test" for n in all_names
                if has_images(self.out / n / "images/test")]

        stale = sorted(d.name for d in self.out.iterdir()
                       if d.is_dir() and d.name not in all_names)
        if stale:
            self.log(f"경고: sources 에 없는 산출물 폴더는 data.yaml 에서 "
                     f"제외했습니다: {', '.join(stale)} (안 쓰면 지우세요)")

        # 소스별 data.yaml — 그 소스 산출물이 있는 폴더에만 씁니다.
        for n in all_names:
            src_out = self.out / n
            per = {"path": str(src_out.resolve())}
            for sp in ("train", "val", "test"):
                if has_images(src_out / f"images/{sp}"):
                    per[sp] = f"images/{sp}"
            if len(per) == 1:      # 산출물이 없는 소스 (targets 로 안 구운 적 없는 경우 등)
                continue
            per["names"] = self.final_names
            with (src_out / "data.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(per, f, allow_unicode=True, sort_keys=False)
            # 전체 데이터 버전 (배포용 학습) — train 에 train+val+test 를 모두 넣고, val 은
            # 지표를 찍기 위한 자리라 val split 을 그대로 둠 (train 과 겹치니 지표는 낙관적,
            # early stopping 은 의미 없음 → 그 config 는 patience 를 끔).
            splits = [per[sp] for sp in ("train", "val", "test") if sp in per]
            every = {"path": per["path"], "train": splits,
                     "val": per.get("val", splits[0]), "names": self.final_names}
            with (src_out / "data.all.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(every, f, allow_unicode=True, sort_keys=False)

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
        (세션 단위로 나누고 싶으면 이 함수가 아니라 소스의 group_by: session.)
        """
        bare = INDEX_RE.sub("", stem)
        return bare.split(".rf.")[0] if ".rf." in bare else stem

    def split_items(self, src: dict, items: list) -> dict:
        """
        항목 index → "train" | "val" | "test". 그룹 단위로 val → test 를 채우고 남는
        그룹이 전부 train. 시드는 소스별로 파생 — 소스를 추가/제거해도 다른 소스의
        분할이 안 바뀝니다.

        group_by: file (기본) — split_group(파일 stem) 단위. 그룹이 작아서(1장, rf 는
          증강 사본 몇 장) 목표 장수에 닿을 때까지 채웁니다.
        group_by: session — 소스 폴더(글롭으로 묶인 세션 폴더 하나) 단위. 2fps 연속
          캡처는 이웃 프레임이 거의 같은 이미지라, 파일 단위로 나누면 val/test 가
          train 의 근중복으로 채워져 점수가 부풀려집니다. 세션은 크기가 제각각(수 장 ~
          수천 장)이라 넘칠 때까지 넣지 않고 목표 장수에 **가까워질 때만** 넣으며,
          train 몫으로 최소 한 세션은 남깁니다 (→ 실제 비율은 목표와 다를 수 있어
          세션 배정을 로그로 남깁니다). val_sessions / test_sessions 로 세션 이름을
          적으면 그 split 은 비율 대신 지정한 세션이 됩니다.
        """
        name = src["name"]
        by = src.get("group_by") or "file"
        if by not in ("file", "session"):
            self.fail(f"{name}: group_by 는 file 또는 session 이어야 합니다: {by}")
        fixed = {sp: [str(s) for s in (src.get(f"{sp}_sessions") or [])]
                 for sp in ("val", "test")}
        if by == "file" and (fixed["val"] or fixed["test"]):
            self.fail(f"{name}: val_sessions/test_sessions 는 group_by: session 일 때만 씁니다.")
        both = set(fixed["val"]) & set(fixed["test"])
        if both:
            self.fail(f"{name}: val_sessions 와 test_sessions 에 같은 세션: {', '.join(sorted(both))}")

        groups: dict = {}
        for i, (img, _, _, session) in enumerate(items):
            key = session if by == "session" else self.split_group(img.stem)
            groups.setdefault(key, []).append(i)
        keys = sorted(groups)
        random.Random(f"{self.cfg.get('seed', 0)}:{name}").shuffle(keys)

        want = {"val": int(round(len(items) * float(src.get("val_ratio", 0.0)))),
                "test": int(round(len(items) * float(src.get("test_ratio", 0.0))))}
        split_of = {i: "train" for i in range(len(items))}

        if by == "file":
            ki = 0
            for sp in ("val", "test"):
                have = 0
                while have < want[sp] and ki < len(keys):
                    for i in groups[keys[ki]]:
                        split_of[i] = sp
                    have += len(groups[keys[ki]])
                    ki += 1
            return split_of

        # ── group_by: session ──
        chosen: dict = {"val": [], "test": []}
        for sp in ("val", "test"):
            for sess in fixed[sp]:
                if sess in groups:
                    chosen[sp].append(sess)
                elif (self.resolve(src["path"]).parent / sess / "images").is_dir():
                    self.log(f"{name}: {sp}_sessions 의 {sess} 는 라벨된 이미지가 0장 — "
                             f"{sp} 에 들어갈 것이 없습니다")
                else:
                    self.fail(f"{name}: {sp}_sessions 에 이 소스의 세션이 아닌 이름: {sess}")
        pool = [k for k in keys if k not in fixed["val"] and k not in fixed["test"]]

        for sp in ("val", "test"):
            if fixed[sp] or want[sp] <= 0:
                continue
            have = 0
            for k in list(pool):
                if len(pool) <= 1:
                    break                          # train 몫으로 최소 한 세션
                n = len(groups[k])
                if abs(have + n - want[sp]) < abs(have - want[sp]):
                    chosen[sp].append(k)
                    pool.remove(k)
                    have += n
            if not chosen[sp] and len(pool) > 1:
                # 남은 세션이 전부 목표의 2배를 넘으면 위에서 하나도 안 뽑힘 → 가장 작은 것 하나
                k = min(pool, key=lambda k: (len(groups[k]), k))
                chosen[sp].append(k)
                pool.remove(k)
            if not chosen[sp]:
                self.log(f"{name}: 라벨된 세션이 {len(groups)}개뿐이라 {sp} 를 만들 수 없습니다 "
                         f"(train 에 최소 한 세션을 남김) — 세션을 더 라벨링하세요")

        for sp in ("val", "test"):
            for k in chosen[sp]:
                for i in groups[k]:
                    split_of[i] = sp
        n_train = sum(1 for v in split_of.values() if v == "train")
        if not n_train:
            self.fail(f"{name}: val_sessions/test_sessions 가 라벨된 세션 전부를 가져가 "
                      f"train 이 비었습니다.")

        self.log(f"{name}: 세션 단위 분할 — 라벨된 세션 {len(groups)}개, {len(items)}장")
        for sp in ("val", "test"):
            if not chosen[sp]:
                continue
            n = sum(len(groups[k]) for k in chosen[sp])
            how = "지정" if fixed[sp] else f"목표 {want[sp]}장"
            self.log(f"  {sp:<5} {len(chosen[sp])}세션 {n}장 ({n / len(items):.0%}, {how}): "
                     + ", ".join(f"{k}({len(groups[k])})" for k in sorted(chosen[sp])))
        self.log(f"  train {len(groups) - len(chosen['val']) - len(chosen['test'])}세션 "
                 f"{n_train}장 ({n_train / len(items):.0%})")
        return split_of

    def run(self) -> None:
        sources = self.selected_sources()
        all_names = [s["name"] for s in self.all_sources()]
        if len(sources) != len(all_names):
            self.log(f"대상(targets): {', '.join(s['name'] for s in sources)} "
                     f"— 전체 {len(all_names)}개 중. 이 소스만 다시 굽고, "
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

            oversample = max(1, int(src.get("oversample", 1)))
            # 소스 내부에서 먼저 train/val/test 분할 → oversample 전에 나눠 누수 차단.
            split_of = self.split_items(src, items)

            counts = {"train": 0, "val": 0, "test": 0}
            for i, item in enumerate(items):
                split = split_of[i]
                reps = oversample if split == "train" else 1  # oversample 은 train 만
                self.write_item(src, item, split, reps, dirs, lbls)
                counts[split] += reps

            n_train += counts["train"]
            n_val += counts["val"]
            n_test += counts["test"]
            how = Cropper(src.get("crop")).summary()
            self.log(f"{src['name']}: train +{counts['train']} (oversample x{oversample}), "
                     f"val +{counts['val']}, test +{counts['test']}  [{how}]")

        train_dirs, val_dirs, test_dirs = self.write_data_yaml()

        if self.task == "obb" and (self._obb_count or self._obb_dropped):
            self.log(f"seg→obb 변환 {self._obb_count}개: 피팅 품질(폴리곤/OBB 면적비) "
                     f"평균 {self._obb_fit_sum / max(self._obb_count, 1):.3f}, "
                     f"최소 {self._obb_fit_min:.3f}"
                     + (f", 0.5 미만 {self._obb_low}개 — 휘었거나 대각 노이즈 라벨인지 "
                        f"_preview 로 확인하세요" if self._obb_low else ""))
            if self._obb_dropped:
                self.log(f"obb 변환 불가(퇴화 폴리곤)로 폐기된 라벨 {self._obb_dropped}개")

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
            self.fail(
                f"val 산출물이 하나도 없습니다 — 이 data.yaml 로 train.py 를 "
                f"돌리면 빈 검증셋으로 죽습니다.\n"
                f"  val_ratio > 0 인 소스를 처리해 val 을 만들어 두세요.\n"
                f"  (targets 로 일부만 굽는 중이었다면 먼저 전체를 한 번 처리해야 합니다. "
                f"이번에 만든 train 산출물은 그대로 유지됩니다.)"
            )
        self.log("다음: python train.py --config <train.*.yaml>")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="공개 + 로컬 데이터를 학습용 YOLO seg 데이터셋으로 전처리",
        epilog="configs/ 의 후보: " + Preprocessor.config_candidates(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config", metavar="YAML",
        help="설정 yaml (후보가 preprocess.yaml 하나라 생략 시 자동). "
             "일부 소스만 다시 구우려면 yaml 의 targets 키를 쓰세요",
    )
    ap.add_argument("--task", choices=("seg", "obb"),
                    help="yaml 의 task 를 덮어씀 (같은 소스로 obb 산출물을 따로 구울 때)")
    ap.add_argument("--out", metavar="DIR",
                    help="yaml 의 out 을 덮어씀 (training/ 기준). 예: ../datasets/processed_obb")
    args = ap.parse_args()
    Preprocessor(args.config, task=args.task, out=args.out).run()


if __name__ == "__main__":
    main()
