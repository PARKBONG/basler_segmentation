"""
파이프라인 자체 검증 — GPU도 카메라도 공개 데이터셋도 없이 돌아갑니다.

검증 대상:
  1. 크롭 기하 (crop_rect)          — 0/50/100% 규약, 원본보다 큰 요청의 클램프
  2. 폴리곤 클리핑 (clip_polygon)   — 완전 포함 / 완전 배제 / 부분 걸침
  3. 라벨 변환 (transform_label)    — 재정규화 좌표, min_area 폐기, 비폴리곤 폐기
  4. Preprocessor 전체              — 합성 데이터셋으로 병합·분할·oversample·크롭
                                      + val 누수 없음 + class_map 필터 + data.yaml

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

from preprocess import Preprocessor, clip_polygon, crop_rect, polygon_area, transform_label

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def approx(a, b, tol=1e-6) -> bool:
    return abs(a - b) <= tol


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


# ── 2. 폴리곤 클리핑 ────────────────────────────────────────────────────────

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


# ── 3. 라벨 변환 ────────────────────────────────────────────────────────────

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


# ── 4. Preprocessor 전체 ────────────────────────────────────────────────────

def make_source(root: Path, name: str, count: int, size: int,
                class_names: dict, lines_for) -> Path:
    """합성 소스 하나 생성: images/train, labels/train, data.yaml."""
    src = root / name
    (src / "images/train").mkdir(parents=True, exist_ok=True)
    (src / "labels/train").mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (size, size), (60 + i, 90, 120)).save(
            src / f"images/train/img{i:03d}.png")
        (src / f"labels/train/img{i:03d}.txt").write_text(
            "\n".join(lines_for(i)) + "\n", encoding="utf-8")
    with (src / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"names": class_names, "train": "images/train"}, f)
    return src


def run_preprocess(root: Path, out: Path, crop: dict, preview: bool = False) -> dict:
    """합성 소스 2개로 Preprocessor 를 돌리고 config 를 돌려준다."""
    # 중앙에 붙은 사각형 하나 + (junk 클래스) 구석에 하나
    def local_lines(i):
        return ["0 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60"]

    def public_lines(i):
        return ["0 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60",
                "1 0.02 0.02 0.06 0.02 0.06 0.06 0.02 0.06"]   # junk → class_map 에 없음

    make_source(root, "raw", 10, 800, {0: "wire"}, local_lines)
    make_source(root, "pub", 6, 800, {0: "cable", 1: "junk"}, public_lines)

    cfg = {
        "names": {0: "wire"},
        "defaults": {"crop": {"enabled": False}},
        "sources": [
            {"name": "raw", "path": str(root / "raw"),
             "class_map": {"wire": "wire"}, "val_ratio": 0.2, "oversample": 2,
             "crop": crop},
            {"name": "rf_a", "path": str(root / "pub"),
             "class_map": {"cable": "wire"}, "val_ratio": 0.0, "oversample": 1,
             "crop": crop},
        ],
        "out": str(out),
        "seed": 0,
        "preview": {"enabled": preview, "count": 4},
    }
    cfg_path = root / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    Preprocessor(cfg_path).run()
    return cfg


def test_pipeline_no_crop(root: Path) -> None:
    out = root / "out_nocrop"
    run_preprocess(root / "srcA", out, {"enabled": False})

    train_imgs = sorted((out / "images/train").glob("*.png"))
    val_imgs = sorted((out / "images/val").glob("*.png"))

    # raw 10장 중 val 2장 → train 8장 × oversample 2 = 16, pub 6장 = 6 → 합 22
    check("pipeline: train 장수", len(train_imgs) == 22, f"{len(train_imgs)}")
    check("pipeline: val 장수", len(val_imgs) == 2, f"{len(val_imgs)}")

    check("pipeline: val 은 로컬에서만",
          all(p.name.startswith("raw__") for p in val_imgs),
          ", ".join(p.name for p in val_imgs))
    check("pipeline: oversample 사본은 val 에 없음",
          not any("_os" in p.name for p in val_imgs))

    # 같은 원본이 train 과 val 양쪽에 있으면 누수
    def origin(p): return p.name.split("_os")[0]
    check("pipeline: train/val 원본 겹침 없음",
          not ({origin(p) for p in train_imgs} & {origin(p) for p in val_imgs}))

    # class_map 에 없는 junk(class 1) 는 제거되어야 함
    pub_lbl = sorted((out / "labels/train").glob("rf_a__*.txt"))
    bad = [p.name for p in pub_lbl
           if any(l.split()[0] != "0" for l in p.read_text(encoding="utf-8").splitlines() if l.strip())]
    check("pipeline: class_map 밖 클래스 제거", not bad, str(bad[:3]))

    # 크롭이 없으면 원본 그대로 (재인코딩 없음)
    src_bytes = (root / "srcA/raw/images/train/img000.png").read_bytes()
    same = [p for p in train_imgs if p.read_bytes() == src_bytes]
    check("pipeline: 크롭 off 는 원본 바이트 그대로", len(same) >= 1, f"{len(same)}개 일치")

    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    check("pipeline: data.yaml 내용",
          data["train"] == "images/train" and data["val"] == "images/val"
          and data["names"] == {0: "wire"}, str(data))


def test_pipeline_crop(root: Path) -> None:
    out = root / "out_crop"
    crop = {"enabled": True, "width": 640, "height": 640,
            "center_x": 50, "center_y": 50, "min_area": 0.10}
    run_preprocess(root / "srcB", out, crop, preview=True)

    train_imgs = sorted((out / "images/train").glob("*.png"))
    check("crop: 장수 유지(중앙 대상은 살아남음)", len(train_imgs) == 22, f"{len(train_imgs)}")

    with Image.open(train_imgs[0]) as im:
        check("crop: 출력 이미지 크기", (im.width, im.height) == (640, 640), f"{im.size}")

    # 800px 원본의 (0.4~0.6) 사각형 = (320,320)-(480,480). 중앙 640 크롭의 원점은 (80,80)
    # → (240,240)-(400,400) → /640 = 0.375 ~ 0.625
    line = (out / "labels/train" / (train_imgs[0].stem + ".txt")
            ).read_text(encoding="utf-8").splitlines()[0]
    v = [float(x) for x in line.split()[1:]]
    check("crop: 라벨이 크롭 좌표로 재정규화",
          approx(min(v), 0.375, 1e-3) and approx(max(v), 0.625, 1e-3),
          f"min={min(v):.4f} max={max(v):.4f}")
    check("crop: 좌표가 0~1 범위", all(0.0 <= x <= 1.0 for x in v))

    # 구석의 junk 는 class_map 에서 이미 빠지므로, 남은 건 wire 뿐
    check("crop: 클래스는 wire 뿐", all(l.split()[0] == "0" for l in
          (out / "labels/train" / (train_imgs[0].stem + ".txt")
           ).read_text(encoding="utf-8").splitlines() if l.strip()))

    previews = list((out / "_preview").glob("*.png"))
    check("crop: 미리보기 생성", len(previews) > 0, f"{len(previews)}장")


def main() -> int:
    test_crop_rect()
    test_clip_polygon()
    test_transform_label()

    root = Path(tempfile.mkdtemp(prefix="finetuner_selftest_"))
    try:
        test_pipeline_no_crop(root)
        test_pipeline_crop(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\nALL PASS" if not FAILURES else f"\n{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    return len(FAILURES)


if __name__ == "__main__":
    sys.exit(main())
