"""
공개(Roboflow) + 로컬 데이터를 하나의 YOLO seg 데이터셋으로 병합.

설계 원칙:
- 로컬(인도메인)은 소스 내부에서 먼저 train/val 분할 → 검증(val)은 로컬에서만 나옴.
- 공개셋은 val_ratio: 0.0 로 두어 train 전용 (검증 오염 방지).
- oversample 은 각 소스의 train 쪽에만 물리 복제로 적용 (val 누수 없음).
- 클래스 선택/이름통일은 class_map(이름 기반)이 담당 → source index 차이에 안전.

각 소스 폴더엔 data.yaml (names + train/val 경로) 이 있어야 함.
  · Roboflow export 는 기본 포함.
  · 로컬은 최소 형식으로 하나 작성:  names: {0: wire}\n train: images/train

사용법:
    python merge.py
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "merge_config.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (HERE / p)


def normalize_names(names) -> dict:
    """names(list 또는 dict) → {idx: name} 로 정규화."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    raise ValueError("names 를 찾을 수 없습니다.")


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


def labels_dir_for(images_dir: Path) -> Path:
    """.../images/... → .../labels/... (YOLO 관례)."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts)


def remap_lines(text: str, src_names: dict, class_map: dict, final_ids: dict) -> list:
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
        parts[0] = str(final_ids[class_map[src_name]])
        out.append(" ".join(parts))
    return out


def collect_source(src: dict, final_ids: dict) -> list:
    """소스에서 (img_path, [remapped lines]) 목록 수집 (대상 클래스가 있는 것만)."""
    src_dir = resolve(src["path"])
    dy_path = src_dir / "data.yaml"
    if not dy_path.exists():
        raise FileNotFoundError(f"{src['name']}: data.yaml 이 없습니다 → {dy_path}")
    dy = load_yaml(dy_path)
    src_names = normalize_names(dy.get("names"))
    class_map = src.get("class_map", {})

    items = []
    seen = set()  # 같은 이미지가 여러 split 에 중복 등록되는 것 방지
    for key in ("train", "val", "valid", "test"):
        img_dir = find_split_image_dir(src_dir, dy, key)
        if img_dir is None:
            continue
        lbl_dir = labels_dir_for(img_dir)
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS or img.name in seen:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            lines = remap_lines(lbl.read_text(encoding="utf-8"),
                                src_names, class_map, final_ids)
            if not lines:
                continue
            items.append((img, lines))
            seen.add(img.name)
    return items


def clear_dir(d: Path) -> None:
    if d.exists():
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
    d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = load_yaml(CONFIG_PATH)
    rng = random.Random(cfg.get("seed", 0))

    final_names = normalize_names(cfg.get("names"))
    final_ids = {v: k for k, v in final_names.items()}  # name → id

    out = resolve(cfg["out"])
    dirs = {sp: out / f"images/{sp}" for sp in ("train", "val")}
    lbls = {sp: out / f"labels/{sp}" for sp in ("train", "val")}
    for d in (*dirs.values(), *lbls.values()):
        clear_dir(d)  # 재실행 시 깨끗하게

    n_train = n_val = 0
    for src in cfg.get("sources", []):
        items = collect_source(src, final_ids)
        if not items:
            print(f"[merge] {src['name']}: 대상 이미지 0장 (경로/class_map 확인)")
            continue

        val_ratio = float(src.get("val_ratio", 0.0))
        oversample = max(1, int(src.get("oversample", 1)))

        # 소스 내부에서 먼저 train/val 분할 → oversample 전에 나눠 val 누수 차단
        idx = list(range(len(items)))
        rng.shuffle(idx)
        n_val_src = int(round(len(items) * val_ratio))
        val_idx = set(idx[:n_val_src])

        src_train = 0
        for i, (img, lines) in enumerate(items):
            split = "val" if i in val_idx else "train"
            reps = 1 if split == "val" else oversample  # oversample 은 train 만
            for r in range(reps):
                suffix = "" if r == 0 else f"_os{r}"
                stem = f"{src['name']}__{img.stem}{suffix}"
                shutil.copyfile(img, dirs[split] / (stem + img.suffix.lower()))
                (lbls[split] / (stem + ".txt")).write_text(
                    "\n".join(lines) + "\n", encoding="utf-8")
            if split == "train":
                src_train += reps
        n_train += src_train
        n_val += n_val_src
        print(f"[merge] {src['name']}: train +{src_train} "
              f"(oversample x{oversample}), val +{n_val_src}")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": final_names,
    }
    with (out / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n[merge] 완료 → {out}")
    print(f"[merge] 합계: train {n_train}장, val {n_val}장 (val = 로컬 인도메인)")
    print(f"[merge] data.yaml: {out / 'data.yaml'}")
    print("다음: python train.py")


if __name__ == "__main__":
    main()
