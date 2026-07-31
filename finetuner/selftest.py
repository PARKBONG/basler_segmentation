"""
파이프라인 자체 검증 — GPU도 카메라도 공개 데이터셋도 없이 돌아갑니다.

검증 대상:
  1. 크롭 기하 (crop_rect)          — 0/50/100% 규약, 0=원본, 원본보다 큰 요청의 클램프
  2. 자동 크롭 (auto_crop_rect)     — 라벨 중심 정렬, margin, min_size 하한, 경계 밀어넣기
  3. 폴리곤 클리핑 (clip_polygon)   — 완전 포함 / 완전 배제 / 부분 걸침
  4. 라벨 변환 (transform_label)    — 재정규화 좌표, min_area 폐기, 비폴리곤 폐기
  5. Preprocessor 전체              — 합성 데이터셋으로 소스별 산출물·분할·oversample·크롭
                                      + val/test 누수 없음 (그룹 단위 3-way 분할)
                                      + class_map 필터 + 통합 data.yaml + --only 필터
                                      + 증분 재굽기 + auto_crop 산출물 + val 없으면 오류
  6. config 규약 (preprocess/train) — stage/config 명시 강제(조용한 fallback 금지),
                                      저장소 yaml 린트, stages/데이터/가중치 누락·오타 시
                                      즉시 오류

필요: pyyaml, pillow  (ultralytics/torch 는 필요 없음)

사용법:
    python selftest.py          # 종료 코드 0 이면 전부 통과
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image

from common import HERE
from preprocess import (Preprocessor, auto_crop_rect, clip_polygon, crop_rect, label_bbox,
                        polygon_area, transform_label)
from train import Trainer   # ultralytics 는 메서드 안에서 import 하므로 여기선 불필요

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def approx(a, b, tol=1e-6) -> bool:
    return abs(a - b) <= tol


def expect_raises(name: str, exc, fn, needle: str = "") -> None:
    """fn() 이 exc 를 던지고 메시지에 needle 이 들어있는지."""
    try:
        fn()
    except exc as e:
        check(name, needle in str(e), str(e).splitlines()[0])
    except Exception as e:                                    # noqa: BLE001
        check(name, False, f"다른 예외: {type(e).__name__}: {e}")
    else:
        check(name, False, "예외가 안 났음")


# ── 1. 크롭 기하 ────────────────────────────────────────────────────────────

def test_crop_rect() -> None:
    base = {"width": 640, "height": 640}
    for cx, cy, ex, ey in [(0, 0, 0, 0), (50, 50, 640, 220), (100, 100, 1280, 440)]:
        x, y, w, h = crop_rect(1920, 1080, {**base, "center_x": cx, "center_y": cy})
        check(f"crop_rect @{cx}%,{cy}%", (x, y, w, h) == (ex, ey, 640, 640), f"{(x, y, w, h)}")

    # 원본보다 큰 요청 → 원본 크기로 클램프
    check("crop_rect 오버사이즈 클램프",
          crop_rect(320, 240, {"width": 640, "height": 640}) == (0, 0, 320, 240))

    # 범위 밖 % → 0~100 으로 클램프
    check("crop_rect % 클램프",
          crop_rect(1920, 1080, {**base, "center_x": -50, "center_y": 500}) == (0, 440, 640, 640))

    # 0 이하 = 그 축은 원본 전체 (여기서는 좌우만 원본, 상하는 640 중앙) — 앱과 같은 규약
    check("crop_rect 0 은 원본 전체",
          crop_rect(1920, 1080, {"width": 0, "height": 640, "center_y": 50}) == (0, 220, 1920, 640),
          f"{crop_rect(1920, 1080, {'width': 0, 'height': 640, 'center_y': 50})}")
    check("crop_rect 음수도 원본 전체",
          crop_rect(1920, 1080, {"width": -1, "height": 640, "center_y": 50}) == (0, 220, 1920, 640))
    check("crop_rect width 누락도 원본 전체",
          crop_rect(1920, 1080, {"height": 640, "center_y": 50}) == (0, 220, 1920, 640))


# ── 2. 자동 크롭 ────────────────────────────────────────────────────────────

def test_auto_crop_rect() -> None:
    # 1000x1000 이미지, 라벨은 (0.40~0.60) → 픽셀 (400,400)-(600,600)
    lines = ["0 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60"]
    bbox = label_bbox(lines, 1000, 1000)
    check("label_bbox 픽셀 경계상자", bbox == (400.0, 400.0, 600.0, 600.0), f"{bbox}")

    # margin 0.25 → 200px * 1.5 = 300px, min_size 0 이면 그대로. 중심 500 → 시작 350
    check("auto: margin 적용 + 라벨 중심 정렬",
          auto_crop_rect(1000, 1000, bbox, {"margin": 0.25, "min_size": 0}) == (350, 350, 300, 300),
          f"{auto_crop_rect(1000, 1000, bbox, {'margin': 0.25, 'min_size': 0})}")

    # min_size 640 → 300px 대신 640px 로 키움 (중심은 그대로 500 → 시작 180)
    check("auto: min_size 하한",
          auto_crop_rect(1000, 1000, bbox, {"margin": 0.25, "min_size": 640}) == (180, 180, 640, 640),
          f"{auto_crop_rect(1000, 1000, bbox, {'margin': 0.25, 'min_size': 640})}")

    # 구석 라벨 → 창이 이미지 밖으로 못 나가게 안쪽으로 밀어 넣음
    corner = label_bbox(["0 0.01 0.01 0.05 0.01 0.05 0.05 0.01 0.05"], 1000, 1000)
    x, y, w, h = auto_crop_rect(1000, 1000, corner, {"margin": 0.25, "min_size": 640})
    check("auto: 이미지 경계 밀어넣기",
          (x, y, w, h) == (0, 0, 640, 640), f"{(x, y, w, h)}")

    # 원본이 min_size 보다 작으면 원본 크기로 클램프 (호출 쪽이 경고할 대상)
    small = auto_crop_rect(320, 240, (100.0, 100.0, 140.0, 140.0), {"margin": 0.25, "min_size": 640})
    check("auto: 원본보다 크게 못 키움", small == (0, 0, 320, 240), f"{small}")

    # 축마다 독립 — 가로로 긴 라벨은 가로만 넓어짐
    wide = label_bbox(["0 0.05 0.48 0.95 0.48 0.95 0.52 0.05 0.52"], 1000, 1000)
    x, y, w, h = auto_crop_rect(1000, 1000, wide, {"margin": 0.0, "min_size": 640})
    check("auto: 축마다 독립 크기", (w, h) == (900, 640), f"{(w, h)}")

    check("label_bbox: 폴리곤 없으면 None", label_bbox(["0 0.5 0.5 0.2 0.2"], 100, 100) is None)


# ── 3. 폴리곤 클리핑 ────────────────────────────────────────────────────────

def test_clip_polygon() -> None:
    square = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]

    inside = clip_polygon(square, 0, 0, 100, 100)
    check("clip: 완전 포함이면 면적 보존", approx(polygon_area(inside), 6400.0),
          f"area={polygon_area(inside)}")

    outside = clip_polygon(square, 200, 200, 300, 300)
    check("clip: 완전 배제면 빈 폴리곤", len(outside) == 0, f"{outside}")

    # 창이 사각형의 오른쪽 절반만 덮음 → 40x80 = 3200
    partial = clip_polygon(square, 50, 0, 200, 200)
    check("clip: 부분 걸침 면적", approx(polygon_area(partial), 3200.0),
          f"area={polygon_area(partial)}")

    # 빗변(대각선)을 실제로 가로지르는 경계 — 교점 계산이 맞는지 보는 케이스.
    # 삼각형 x+y<=100 을 x>=60 으로 자르면 (60,0),(100,0),(60,40) → 0.5*40*40 = 800
    tri = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]
    cut = clip_polygon(tri, 60, 0, 200, 200)
    check("clip: 빗변 교차", approx(polygon_area(cut), 800.0), f"area={polygon_area(cut)}")

    # 창이 도형 안에 완전히 들어가면 창 전체가 결과 (2500)
    contained = clip_polygon(tri, 0, 0, 50, 50)
    check("clip: 창이 도형 안에 포함", approx(polygon_area(contained), 2500.0),
          f"area={polygon_area(contained)}")


# ── 4. 라벨 변환 ────────────────────────────────────────────────────────────

def test_transform_label() -> None:
    # 1000x1000 원본의 (400,400)-(600,600) 사각형, 크롭 창 (200,200,600,600)
    line = "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"
    rect = (200, 200, 600, 600)
    out = transform_label(line, 1000, 1000, rect, 0.10)
    check("label: 변환 성공", out is not None)
    if out:
        v = [float(x) for x in out.split()[1:]]
        # 창 안 (400,400) → (200,200)/600 = 0.3333…, (600,600) → 0.6666…
        check("label: 좌표 재정규화",
              approx(min(v), 1 / 3, 1e-4) and approx(max(v), 2 / 3, 1e-4),
              f"min={min(v):.4f} max={max(v):.4f}")
        check("label: class id 보존", out.split()[0] == "0")

    # 창 밖 인스턴스 → 폐기
    far = "0 0.9 0.9 0.95 0.9 0.95 0.95 0.9 0.95"
    check("label: 창 밖이면 None", transform_label(far, 1000, 1000, (0, 0, 400, 400), 0.10) is None)

    # 살짝만 걸침 → min_area 로 폐기 (면적비 1/4 < 0.5)
    edge = "0 0.30 0.30 0.50 0.30 0.50 0.50 0.30 0.50"   # (300,300)-(500,500)
    rect2 = (0, 0, 400, 400)                              # 겹치는 부분 (300,300)-(400,400)
    check("label: min_area 미만이면 폐기",
          transform_label(edge, 1000, 1000, rect2, 0.50) is None)
    check("label: min_area 이상이면 유지",
          transform_label(edge, 1000, 1000, rect2, 0.20) is not None)

    # 폴리곤이 아닌 줄(bbox 4값) → 폐기
    check("label: 비폴리곤은 폐기",
          transform_label("0 0.5 0.5 0.2 0.2", 1000, 1000, rect, 0.10) is None)


# ── 5. Preprocessor 전체 ────────────────────────────────────────────────────

def make_source(root: Path, name: str, count: int, size: int,
                class_names: dict, lines_for, flat: bool = False) -> Path:
    """합성 소스 하나 생성. flat=True 면 kimm 규약(images/, labels/, train: images)."""
    src = root / name
    img_rel = "images" if flat else "images/train"
    lbl_rel = "labels" if flat else "labels/train"
    (src / img_rel).mkdir(parents=True, exist_ok=True)
    (src / lbl_rel).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (size, size), (60 + i, 90, 120)).save(
            src / img_rel / f"img{i:03d}.png")
        (src / lbl_rel / f"img{i:03d}.txt").write_text(
            "\n".join(lines_for(i)) + "\n", encoding="utf-8")
    with (src / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"names": class_names, "train": img_rel}, f)
    return src


def run_preprocess(root: Path, out: Path, crop: dict, preview: bool = False,
                   only: list | None = None, test_ratio: float = 0.0) -> dict:
    """합성 소스 2개로 Preprocessor 를 돌리고 그 인스턴스를 돌려준다(집계값 확인용)."""
    # 중앙에 붙은 사각형 하나 + (junk 클래스) 구석에 하나
    def local_lines(i):
        return ["0 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60"]

    def public_lines(i):
        return ["0 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60",
                "1 0.02 0.02 0.06 0.02 0.06 0.06 0.02 0.06"]   # junk → class_map 에 없음

    # kimm 은 실제 규약처럼 flat 레이아웃(images/ + train: images)으로 만든다
    make_source(root, "kimm", 10, 800, {0: "wire"}, local_lines, flat=True)
    make_source(root, "pub", 6, 800, {0: "cable", 1: "junk"}, public_lines)

    # kimm 의 val_ratio 는 sources 에서 0.0 으로 두고 stage use 가 0.2 로 덮어쓴다
    # → use 의 얕은 병합(필드 단위 덮어쓰기)이 실제로 동작하는지도 함께 검증
    cfg = {
        "names": {0: "wire"},
        "sources": [
            {"name": "kimm", "path": str(root / "kimm"),
             "class_map": {"wire": "wire"}, "val_ratio": 0.0,
             "oversample": 2, "crop": crop},
            {"name": "rf_a", "path": str(root / "pub"),
             "class_map": {"cable": "wire"}, "oversample": 1,
             "crop": crop},
        ],
        "stages": {
            "single": {
                "out": str(out),
                "use": {"kimm": {"val_ratio": 0.2, "test_ratio": test_ratio},
                        "rf_a": {"val_ratio": 0.0}},
            },
        },
        "seed": 0,
        "preview": {"enabled": preview, "count": 4},
    }
    cfg_path = root / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    stage = Preprocessor("single", cfg_path, only=only)
    stage.run()
    return stage


def test_pipeline_no_crop(root: Path) -> None:
    out = root / "out_nocrop"
    run_preprocess(root / "srcA", out, {"enabled": False})

    kimm_train = sorted((out / "kimm/images/train").glob("*.png"))
    kimm_val = sorted((out / "kimm/images/val").glob("*.png"))
    rf_train = sorted((out / "rf_a/images/train").glob("*.png"))
    rf_val = sorted((out / "rf_a/images/val").glob("*.png"))

    # kimm 10장 중 val 2장 → train 8장 × oversample 2 = 16, rf_a 6장 = 6
    check("pipeline: 소스별 train 장수",
          len(kimm_train) == 16 and len(rf_train) == 6,
          f"kimm={len(kimm_train)} rf_a={len(rf_train)}")
    check("pipeline: val 장수", len(kimm_val) == 2, f"{len(kimm_val)}")

    check("pipeline: val 은 로컬(kimm)에서만", not rf_val,
          ", ".join(p.name for p in rf_val))
    check("pipeline: oversample 사본은 val 에 없음",
          not any("_os" in p.name for p in kimm_val))

    # 같은 원본이 train 과 val 양쪽에 있으면 누수
    def origin(p): return p.name.split("_os")[0]
    check("pipeline: train/val 원본 겹침 없음",
          not ({origin(p) for p in kimm_train} & {origin(p) for p in kimm_val}))

    # class_map 에 없는 junk(class 1) 는 제거되어야 함
    pub_lbl = sorted((out / "rf_a/labels/train").glob("*.txt"))
    bad = [p.name for p in pub_lbl
           if any(l.split()[0] != "0" for l in p.read_text(encoding="utf-8").splitlines() if l.strip())]
    check("pipeline: class_map 밖 클래스 제거", not bad, str(bad[:3]))

    # 크롭이 없으면 원본 그대로 (재인코딩 없음) — kimm 은 flat 레이아웃(images/)
    src_bytes = (root / "srcA/kimm/images/img000.png").read_bytes()
    same = [p for p in kimm_train + kimm_val if p.read_bytes() == src_bytes]
    check("pipeline: 크롭 off 는 원본 바이트 그대로", len(same) >= 1, f"{len(same)}개 일치")

    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    check("pipeline: data.yaml 은 소스 폴더 목록",
          data["train"] == ["kimm/images/train", "rf_a/images/train"]
          and data["val"] == ["kimm/images/val"]
          and data["names"] == {0: "wire"}, str(data))
    check("pipeline: test_ratio 0 이면 data.yaml 에 test 키 없음", "test" not in data,
          str(data.get("test")))


def test_pipeline_crop(root: Path) -> None:
    out = root / "out_crop"
    crop = {"enabled": True, "width": 640, "height": 640,
            "center_x": 50, "center_y": 50, "min_area": 0.10}
    run_preprocess(root / "srcB", out, crop, preview=True)

    train_imgs = sorted((out / "kimm/images/train").glob("*.png"))
    rf_imgs = sorted((out / "rf_a/images/train").glob("*.png"))
    check("crop: 장수 유지(중앙 대상은 살아남음)",
          len(train_imgs) == 16 and len(rf_imgs) == 6,
          f"kimm={len(train_imgs)} rf_a={len(rf_imgs)}")

    with Image.open(train_imgs[0]) as im:
        check("crop: 출력 이미지 크기", (im.width, im.height) == (640, 640), f"{im.size}")

    # 800px 원본의 (0.4~0.6) 사각형 = (320,320)-(480,480). 중앙 640 크롭의 원점은 (80,80)
    # → (240,240)-(400,400) → /640 = 0.375 ~ 0.625
    line = (out / "kimm/labels/train" / (train_imgs[0].stem + ".txt")
            ).read_text(encoding="utf-8").splitlines()[0]
    v = [float(x) for x in line.split()[1:]]
    check("crop: 라벨이 크롭 좌표로 재정규화",
          approx(min(v), 0.375, 1e-3) and approx(max(v), 0.625, 1e-3),
          f"min={min(v):.4f} max={max(v):.4f}")
    check("crop: 좌표가 0~1 범위", all(0.0 <= x <= 1.0 for x in v))

    # 구석의 junk 는 class_map 에서 이미 빠지므로, 남은 건 wire 뿐
    check("crop: 클래스는 wire 뿐", all(l.split()[0] == "0" for l in
          (out / "kimm/labels/train" / (train_imgs[0].stem + ".txt")
           ).read_text(encoding="utf-8").splitlines() if l.strip()))

    previews = list((out / "_preview").glob("*.png"))
    check("crop: 미리보기 생성", len(previews) > 0, f"{len(previews)}장")


def test_pipeline_auto_crop(root: Path) -> None:
    out = root / "out_auto"
    crop = {"enabled": True, "min_area": 0.10,
            "auto_crop": {"enabled": True, "margin": 0.25, "min_size": 640}}
    stage = run_preprocess(root / "srcC", out, crop)

    train_imgs = sorted((out / "kimm/images/train").glob("*.png"))
    rf_imgs = sorted((out / "rf_a/images/train").glob("*.png"))
    check("auto: 장수 유지", len(train_imgs) == 16 and len(rf_imgs) == 6,
          f"kimm={len(train_imgs)} rf_a={len(rf_imgs)}")

    # 800px 원본의 (320,320)-(480,480) 라벨 → 160*1.5=240 이지만 min_size 640 으로 확대,
    # 중심 400 → 창 (80,80,640,640). 라벨은 (240,240)-(400,400) → /640 = 0.375~0.625
    with Image.open(train_imgs[0]) as im:
        check("auto: 출력 이미지 크기 = min_size", (im.width, im.height) == (640, 640), f"{im.size}")
    v = [float(x) for x in (out / "kimm/labels/train" / (train_imgs[0].stem + ".txt")
                            ).read_text(encoding="utf-8").splitlines()[0].split()[1:]]
    check("auto: 라벨이 창 좌표로 재정규화",
          approx(min(v), 0.375, 1e-3) and approx(max(v), 0.625, 1e-3),
          f"min={min(v):.4f} max={max(v):.4f}")
    check("auto: 원본이 충분히 크면 경고 없음", stage._undersized == 0, f"{stage._undersized}")

    # min_size 가 원본(800)보다 크면 원본 크기로 클램프되고 경고 대상이 됨
    out2 = root / "out_auto_small"
    crop2 = {"enabled": True, "min_area": 0.10,
             "auto_crop": {"enabled": True, "margin": 0.25, "min_size": 1024}}
    stage2 = run_preprocess(root / "srcD", out2, crop2)
    check("auto: min_size > 원본이면 경고 집계", stage2._undersized > 0, f"{stage2._undersized}")
    with Image.open(sorted((out2 / "kimm/images/train").glob("*.png"))[0]) as im:
        check("auto: 원본 크기로 클램프", (im.width, im.height) == (800, 800), f"{im.size}")


def test_pipeline_only(root: Path) -> None:
    out = root / "out_only"
    # val_ratio 0 소스만 처리하면 val 이 비는데, 그 data.yaml 로 학습하면 Ultralytics 가
    # 빈 검증셋 빌드에서 죽습니다 → preprocess 단계에서 SystemExit 로 미리 막아야 함.
    expect_raises("only: val 산출물 없으면 SystemExit",
                  SystemExit,
                  lambda: run_preprocess(root / "srcE", out, {"enabled": False},
                                         only=["rf_a"]),
                  "val 산출물")

    # 막더라도 이번에 만든 산출물과 data.yaml 은 보존되어야 함 (증분 워크플로 유지)
    train_imgs = sorted((out / "rf_a/images/train").glob("*.png"))
    check("only: 지정한 소스만 처리(산출물 보존)", len(train_imgs) == 6, f"{len(train_imgs)}")
    check("only: 빠진 소스는 산출물에 없음", not (out / "kimm").exists())

    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    check("only: data.yaml 은 있는 산출물만",
          data["train"] == ["rf_a/images/train"], str(data["train"]))

    # 없는 이름은 조용히 넘어가지 않고 바로 알려줘야 함
    try:
        run_preprocess(root / "srcF", root / "out_bad", {"enabled": False}, only=["nope"])
        check("only: 오타는 SystemExit", False, "예외가 안 났음")
    except SystemExit as e:
        check("only: 오타는 SystemExit", "nope" in str(e), str(e).splitlines()[0])


def test_pipeline_incremental(root: Path) -> None:
    """--only 재실행이 다른 소스의 산출물을 보존하고 data.yaml 을 다시 묶는지."""
    out = root / "out_incr"
    run_preprocess(root / "srcG", out, {"enabled": False})                    # 전체
    before = [p.name for p in sorted((out / "kimm/images/train").glob("*.png"))]

    run_preprocess(root / "srcG", out, {"enabled": False}, only=["rf_a"])     # rf_a 만
    after = [p.name for p in sorted((out / "kimm/images/train").glob("*.png"))]
    check("incremental: 다른 소스 산출물 보존", before == after and bool(before),
          f"before={len(before)} after={len(after)}")

    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    check("incremental: data.yaml 은 디스크 전체를 다시 묶음",
          data["train"] == ["kimm/images/train", "rf_a/images/train"]
          and data["val"] == ["kimm/images/val"], str(data))


def test_split_group() -> None:
    """Roboflow 증강 사본이 원본 단위로 묶여 근중복 누수를 막는지."""
    check("split_group: rf 사본은 원본 stem 으로",
          Preprocessor.split_group("frame12_jpg.rf.a1b2c3") == "frame12_jpg")
    check("split_group: 같은 원본의 사본은 같은 그룹",
          Preprocessor.split_group("frame12_jpg.rf.xxxx")
          == Preprocessor.split_group("frame12_jpg.rf.yyyy"))
    check("split_group: 일반 stem 은 그대로",
          Preprocessor.split_group("IMG_0042") == "IMG_0042")
    check("split_group: download 인덱스 접두어는 벗기고 그룹핑",
          Preprocessor.split_group("0003__frame12_jpg.rf.xxxx")
          == Preprocessor.split_group("0017__frame12_jpg.rf.yyyy"))
    check("split_group: 인덱스 접두어 + 일반 stem",
          Preprocessor.split_group("0042__cap_001") == "cap_001")


def test_pipeline_three_way(root: Path) -> None:
    """test_ratio 로 3-way 분할 — 그룹 무결성 + oversample 격리 + data.yaml test 키."""
    out = root / "out_3way"
    run_preprocess(root / "srcH", out, {"enabled": False}, test_ratio=0.2)

    tr = sorted((out / "kimm/images/train").glob("*.png"))
    va = sorted((out / "kimm/images/val").glob("*.png"))
    te = sorted((out / "kimm/images/test").glob("*.png"))
    # kimm 10장: val 2 (0.2), test 2 (0.2), train 6 × oversample 2 = 12
    check("3way: 장수 (train 6x2, val 2, test 2)",
          len(tr) == 12 and len(va) == 2 and len(te) == 2,
          f"train={len(tr)} val={len(va)} test={len(te)}")
    check("3way: test 에 oversample 사본 없음", not any("_os" in p.name for p in te))

    def origin(p): return p.name.split("_os")[0]
    otr, ova, ote = ({origin(p) for p in s} for s in (tr, va, te))
    check("3way: train/val/test 원본 겹침 없음",
          not (otr & ova) and not (otr & ote) and not (ova & ote))

    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    check("3way: data.yaml 에 test 목록", data.get("test") == ["kimm/images/test"],
          str(data.get("test")))


# ── 6. Trainer config 규약 ──────────────────────────────────────────────────

def write_train_cfg(root: Path, name: str, cfg: dict) -> Path:
    path = root / name
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def test_trainer_config_required(root: Path) -> None:
    """config 를 생략하면 기본값으로 넘어가지 않고 즉시 멈춰야 합니다."""
    expect_raises("trainer: config 생략은 SystemExit", SystemExit, Trainer, "명시")

    # 이름을 잘못 적어도 traceback 이 아니라 후보 안내
    expect_raises("trainer: 없는 config 는 SystemExit + 후보 안내", SystemExit,
                  lambda: Trainer("train_config.stage3.yaml"), "후보")
    check("trainer: config 후보 목록", "train_config.stage1.yaml" in Trainer.config_candidates(),
          Trainer.config_candidates())

    # 상대경로는 finetuner/ 기준으로 해석되고, 넘긴 파일이 그대로 쓰여야 함
    t = Trainer("train_config.stage1.yaml")
    check("trainer: 넘긴 config 를 그대로 읽음",
          t.config_path == HERE / "train_config.stage1.yaml", str(t.config_path))
    check("trainer: stage1 config 내용 확인",
          t.cfg["train"]["name"] == "stage1", str(t.cfg["train"].get("name")))


def test_trainer_stages_guard(root: Path) -> None:
    """stages 블록 누락 · 키 오타 · 전부 false 는 모두 오류."""
    base = {"train": {"model": "yolo26s-seg.pt", "data": "d/data.yaml"}}

    p = write_train_cfg(root, "t_nostages.yaml", base)
    expect_raises("stages: 블록 없으면 SystemExit", SystemExit, Trainer(p).run, "stages 블록")

    p = write_train_cfg(root, "t_typo.yaml", {**base, "stages": {"expor": True}})
    expect_raises("stages: 키 오타는 SystemExit", SystemExit, Trainer(p).run, "expor")

    p = write_train_cfg(root, "t_allfalse.yaml",
                        {**base, "stages": {"preview_aug": False, "train": False,
                                            "export": False}})
    expect_raises("stages: 전부 false 면 SystemExit", SystemExit, Trainer(p).run, "전부 false")

    # 유효한 stages 는 해당 단계만, 적힌 순서와 무관하게 정해진 순서로 호출
    called: list = []
    p = write_train_cfg(root, "t_ok.yaml",
                        {**base, "stages": {"export": True, "preview_aug": True,
                                            "train": False}})
    tr = Trainer(p)
    tr.preview_aug = lambda: called.append("preview_aug")   # type: ignore[method-assign]
    tr.train = lambda: called.append("train")               # type: ignore[method-assign]
    tr.export = lambda: called.append("export")             # type: ignore[method-assign]
    tr.run()
    check("stages: 켠 단계만 정해진 순서로 실행",
          called == ["preview_aug", "export"], str(called))


def test_trainer_data_and_weights(root: Path) -> None:
    """데이터셋·시작 가중치도 기본값으로 추측하지 않습니다."""
    p = write_train_cfg(root, "t_nodata.yaml",
                        {"stages": {"train": True}, "train": {"model": "yolo26s-seg.pt"}})
    expect_raises("data: train.data 없으면 SystemExit", SystemExit,
                  Trainer(p).data_path, "train.data")

    p = write_train_cfg(root, "t_nomodel.yaml",
                        {"stages": {"train": True}, "train": {"data": "d/data.yaml"}})
    expect_raises("model: train.model 없으면 SystemExit", SystemExit,
                  Trainer(p).start_weights, "train.model")

    # 경로 구분자가 없으면 Ultralytics 가 받아올 이름 → 존재 확인 없이 그대로
    p = write_train_cfg(root, "t_name.yaml",
                        {"stages": {"train": True}, "train": {"data": "d/data.yaml",
                                                              "model": "yolo26s-seg.pt"}})
    check("model: 모델 이름은 그대로 통과",
          Trainer(p).start_weights() == "yolo26s-seg.pt")

    # 경로 형태인데 파일이 없으면 오류 (COCO 로 조용히 되돌아가지 않음)
    p = write_train_cfg(root, "t_badckpt.yaml",
                        {"stages": {"train": True},
                         "train": {"data": "d/data.yaml",
                                   "model": "runs/segment/stage1/weights/best.pt"}})
    expect_raises("model: 없는 checkpoint 는 FileNotFoundError", FileNotFoundError,
                  Trainer(p).start_weights, "시작 가중치가 없습니다")

    # 있으면 절대경로로 해석
    ckpt = root / "weights/best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"not-a-real-checkpoint")
    p = write_train_cfg(root, "t_goodckpt.yaml",
                        {"stages": {"train": True},
                         "train": {"data": "d/data.yaml", "model": str(ckpt)}})
    check("model: 있는 checkpoint 는 절대경로로", Trainer(p).start_weights() == str(ckpt))


def test_preprocess_stage_required() -> None:
    """preprocess 는 stage 후보가 여럿(단일/stage1/stage2) → 생략·오타는 오류."""
    expect_raises("preprocess: stage 생략은 SystemExit", SystemExit, Preprocessor, "명시")
    expect_raises("preprocess: 없는 stage 는 SystemExit + 후보", SystemExit,
                  lambda: Preprocessor("stage9"), "stage1")
    check("preprocess: stage 후보 목록",
          "stage1" in Preprocessor.stage_candidates(), Preprocessor.stage_candidates())


def test_real_download_config() -> None:
    """저장소의 download_config.yaml 이 규약을 지키는지 (통합 config 린트)."""
    stage_names = list((yaml.safe_load(
        (HERE / Preprocessor.config_name).read_text(encoding="utf-8")) or {}
    ).get("stages") or {})
    required = {"single", "stage1", "stage2"}
    check("download_config: 단일/stage1/stage2 존재", required <= set(stage_names),
          ", ".join(stage_names))

    outs = {}
    for name in stage_names:
        st = Preprocessor(name)
        srcs = st.stage_sources()   # use 의 이름 오타면 여기서 SystemExit
        check(f"stage {name}: names/use/out",
              bool(st.cfg.get("names")) and bool(srcs) and bool(st.stage_cfg.get("out")))
        # val 이 없으면 run() 이 SystemExit 로 막으므로, config 부터 걸러낸다
        check(f"stage {name}: val_ratio > 0 소스 존재",
              any(float(s.get("val_ratio", 0)) > 0 for s in srcs))
        outs[name] = str(st.stage_cfg.get("out"))
    check("download_config: 단일/stage1/stage2 out 이 전부 다름",
          len({outs[n] for n in required if n in outs}) == len(required & set(stage_names)),
          str(outs))

    # download.py 쪽 규약 — 내려받을 소스는 workspace/project 가 있어야 함
    all_sources = yaml.safe_load(
        (HERE / Preprocessor.config_name).read_text(encoding="utf-8")).get("sources") or []
    rf = [s for s in all_sources if s.get("roboflow")]
    check("download_config: roboflow 소스 존재 + workspace/project",
          bool(rf) and all(s["roboflow"].get("workspace") and s["roboflow"].get("project")
                           for s in rf),
          ", ".join(s["name"] for s in rf) or "(없음)")


def test_real_train_configs() -> None:
    """저장소에 있는 train_config*.yaml 이 모두 규약을 지키는지 (config 린트)."""
    names = sorted(p.name for p in HERE.glob(Trainer.config_glob))
    check("configs: train_config*.yaml 이 존재", bool(names), ", ".join(names))

    for name in names:
        t = Trainer(name)
        stages = t.cfg.get("stages")
        ok = isinstance(stages, dict) and bool(stages)
        check(f"config {name}: stages 블록", ok, str(stages))
        if not ok:
            continue
        check(f"config {name}: stages 키 유효",
              not [k for k in stages if k not in Trainer.STAGE_KEYS], str(list(stages)))
        tr = t.cfg.get("train") or {}
        check(f"config {name}: train.model 명시", bool(tr.get("model")), str(tr.get("model")))
        check(f"config {name}: train.data 명시", bool(tr.get("data")), str(tr.get("data")))
        if stages.get("export"):
            check(f"config {name}: export 켰으면 export 블록도", bool(t.cfg.get("export")))


def main() -> int:
    test_crop_rect()
    test_auto_crop_rect()
    test_clip_polygon()
    test_transform_label()
    test_split_group()

    root = Path(tempfile.mkdtemp(prefix="finetuner_selftest_"))
    try:
        test_pipeline_no_crop(root)
        test_pipeline_crop(root)
        test_pipeline_auto_crop(root)
        test_pipeline_only(root)
        test_pipeline_incremental(root)
        test_pipeline_three_way(root)
        test_preprocess_stage_required()
        test_real_download_config()
        test_trainer_config_required(root)
        test_trainer_stages_guard(root)
        test_trainer_data_and_weights(root)
        test_real_train_configs()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\nALL PASS" if not FAILURES else f"\n{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    return len(FAILURES)


if __name__ == "__main__":
    sys.exit(main())
