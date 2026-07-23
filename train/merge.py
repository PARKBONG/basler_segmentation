"""
공개(Stage1) + 로컬 in-domain(Stage2) 데이터를 2단계 파인튜닝용 데이터셋 2벌로 분리 생성.

설계(2단계 순차 파인튜닝):
- Stage1 (out.stage1): role=public 소스만. warm-up 전용.
    · in-domain 을 전혀 포함하지 않음 → in-domain val/test 완전 격리(데이터 위생).
    · 공개셋 내부를 public_val_ratio 로 train/val 분할 (학습 메커니즘/모니터링용).
- Stage2 (out.stage2): role=local(in-domain) 소스만. 도메인 적응 + 검증 + 최종 테스트.
    · train/val/test 3분할. val=best.pt 선택, test=최종 1회 평가(data.yaml 의 test: 키).
    · ★ 그룹(실험) 단위 분할: images/<실험>/ 하위 폴더를 그룹으로 보고 통째로 배정.
      영상 프레임이 train/val/test 로 쪼개지는 상관-프레임 누수(leakage)를 원천 차단.
      (하위 폴더가 없으면 각 이미지가 자기 자신 그룹 = 사실상 랜덤 분할)
    · 기본은 자동 분할(ratio + seed). val_list/test_list 지정 시 우선(수동 선별, 실험명 기준).
- 클래스 선택/이름통일은 class_map(이름 기반) → source index 차이에 안전. class_map 에 없는 클래스 줄은 제외.

각 소스 폴더엔 data.yaml (names + train/val 경로) 이 있어야 함.
  · Roboflow export 는 기본 포함.
  · 로컬은 최소 형식으로 하나 작성:  names: {0: wire}\n  train: images
    (images/ 아래에 실험별 하위폴더 exp001/, exp002/ ... 를 두는 것을 권장)

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
    """.../images/... → .../labels/... (YOLO 관례). 라벨 루트를 반환."""
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
    """소스에서 (src_name, group, img_path, [remapped lines]) 수집 (대상 클래스가 있는 것만).

    group = images/ 아래 첫 하위폴더명(실험). 하위폴더가 없으면 파일 stem(각자 그룹).
    라벨은 images↔labels 관례로 같은 하위경로에서 찾음.
    """
    src_dir = resolve(src["path"])
    dy_path = src_dir / "data.yaml"
    if not dy_path.exists():
        raise FileNotFoundError(f"{src['name']}: data.yaml 이 없습니다 → {dy_path}")
    dy = load_yaml(dy_path)
    src_names = normalize_names(dy.get("names"))
    class_map = src.get("class_map", {})

    items = []
    seen = set()  # 같은 물리 파일이 여러 split 키에 중복 등록되는 것 방지
    for key in ("train", "val", "valid", "test"):
        img_dir = find_split_image_dir(src_dir, dy, key)
        if img_dir is None:
            continue
        lbl_root = labels_dir_for(img_dir)
        for img in sorted(img_dir.rglob("*")):   # 하위폴더(실험)까지 재귀 탐색
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            ap = str(img.resolve())
            if ap in seen:
                continue
            rel = img.relative_to(img_dir)
            group = rel.parts[0] if len(rel.parts) > 1 else img.stem
            lbl = lbl_root / rel.parent / (img.stem + ".txt")
            if not lbl.exists():
                continue
            lines = remap_lines(lbl.read_text(encoding="utf-8"),
                                src_names, class_map, final_ids)
            if not lines:
                continue
            items.append((src["name"], group, img, lines))
            seen.add(ap)
    return items


def clear_dir(d: Path) -> None:
    if d.exists():
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
    d.mkdir(parents=True, exist_ok=True)


def write_bucket(items: list, images_dir: Path, labels_dir: Path) -> int:
    """(src_name, group, img, lines) 목록을 images/labels 로 복사. 실험/소스 접두로 충돌 방지."""
    for src_name, group, img, lines in items:
        key = img.stem if group == img.stem else f"{group}__{img.stem}"
        stem = f"{src_name}__{key}"
        shutil.copyfile(img, images_dir / (stem + img.suffix.lower()))
        (labels_dir / (stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(items)


def write_dataset(out: Path, buckets: dict, names: dict, with_test: bool = False) -> None:
    """buckets({split: [items]}) 를 YOLO seg 데이터셋으로 기록 + data.yaml 생성."""
    splits = ["train", "val"] + (["test"] if with_test else [])
    for sp in splits:
        clear_dir(out / f"images/{sp}")
        clear_dir(out / f"labels/{sp}")
    for sp in splits:
        write_bucket(buckets.get(sp, []), out / f"images/{sp}", out / f"labels/{sp}")

    data_yaml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val"}
    if with_test:
        data_yaml["test"] = "images/test"
    data_yaml["names"] = names
    with (out / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)


def load_name_set(path_str, base: Path):
    """텍스트 파일(한 줄에 하나) → 이름 집합(실험명). 확장자 무시. 없으면 None."""
    if not path_str:
        return None
    p = base / path_str if not Path(path_str).is_absolute() else Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"선별 목록 파일이 없습니다: {p}")
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.add(Path(line).stem)
    return out


def _groups_of(items: list) -> dict:
    """items → {group: [items]}."""
    g = {}
    for t in items:
        g.setdefault(t[1], []).append(t)
    return g


def split_public(triples: list, val_ratio: float, rng: random.Random):
    """공개셋 → (train, val) 2분할. val 은 stage1 학습 메커니즘용 (그룹 무관, 이미지 단위)."""
    idx = list(range(len(triples)))
    rng.shuffle(idx)
    n_val = int(round(len(triples) * val_ratio))
    val_idx = set(idx[:n_val])
    train = [t for i, t in enumerate(triples) if i not in val_idx]
    val = [t for i, t in enumerate(triples) if i in val_idx]
    return train, val


def split_local(triples: list, split_cfg: dict, rng: random.Random) -> dict:
    """in-domain → {train, val, test} 3분할. ★ 그룹(실험) 통째로 배정 → 프레임 누수 차단.

    수동 목록(val_list/test_list, 실험명 기준)이 있으면 우선.
    """
    groups = _groups_of(triples)
    val_names = load_name_set(split_cfg.get("val_list"), HERE)
    test_names = load_name_set(split_cfg.get("test_list"), HERE)
    buckets = {"train": [], "val": [], "test": []}

    if val_names is not None or test_names is not None:
        # 수동 선별 우선: 실험명이 목록에 있으면 해당 split, 나머지는 train.
        vl, tl = (val_names or set()), (test_names or set())
        for g, items in groups.items():
            dest = "test" if g in tl else ("val" if g in vl else "train")
            buckets[dest].extend(items)
        return buckets

    # 자동: 그룹을 셔플 후, 목표 이미지 비율에 도달할 때까지 그룹 통째로 배정.
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    test_ratio = float(split_cfg.get("test_ratio", 0.0))
    total = len(triples)
    n_test_target = round(total * test_ratio)
    n_val_target = round(total * val_ratio)

    gnames = list(groups.keys())
    rng.shuffle(gnames)
    c_test = c_val = 0
    for g in gnames:
        items = groups[g]
        if c_test < n_test_target:
            buckets["test"].extend(items); c_test += len(items)
        elif c_val < n_val_target:
            buckets["val"].extend(items); c_val += len(items)
        else:
            buckets["train"].extend(items)
    return buckets


def main() -> None:
    cfg = load_yaml(CONFIG_PATH)
    rng = random.Random(cfg.get("seed", 0))

    final_names = normalize_names(cfg.get("names"))
    final_ids = {v: k for k, v in final_names.items()}  # name → id

    sources = cfg.get("sources", [])
    public = [s for s in sources if s.get("role") == "public"]
    local = [s for s in sources if s.get("role") == "local"]
    if not public and not local:
        raise SystemExit("[merge] 소스에 role: public/local 이 지정되어야 합니다.")

    out_cfg = cfg.get("out", {})
    stage1_out = resolve(out_cfg.get("stage1", "../datasets/stage1"))
    stage2_out = resolve(out_cfg.get("stage2", "../datasets/stage2"))

    # ── Stage 1: 공개 데이터 (warm-up 전용) ──────────────────────────────
    pub_items = []
    for s in public:
        pub_items += collect_source(s, final_ids)
    if pub_items:
        train, val = split_public(pub_items, float(cfg.get("public_val_ratio", 0.1)), rng)
        write_dataset(stage1_out, {"train": train, "val": val}, final_names)
        print(f"[merge] stage1(공개): train {len(train)}, val {len(val)}")
        if not val:
            print("[merge][경고] stage1 val 0장 — public_val_ratio 를 올리세요 "
                  "(Ultralytics 는 검증셋이 필요합니다).")
    else:
        print("[merge] stage1: 대상 이미지 0장 (role: public / class_map / 경로 확인)")

    # ── Stage 2: 로컬 in-domain (그룹 단위 3분할) ────────────────────────
    loc_items = []
    for s in local:
        loc_items += collect_source(s, final_ids)
    if loc_items:
        buckets = split_local(loc_items, cfg.get("local_split", {}), rng)
        write_dataset(stage2_out, buckets, final_names, with_test=True)
        for sp in ("train", "val", "test"):
            n_img = len(buckets[sp])
            n_grp = len(_groups_of(buckets[sp]))
            print(f"[merge] stage2 {sp:5s}: {n_img:5d}장 / {n_grp} 실험")
        if not buckets["val"]:
            print("[merge][경고] stage2 val 0장 — local_split.val_ratio 또는 val_list 확인.")
        if not buckets["test"]:
            print("[merge][경고] stage2 test 0장 — test_ratio/test_list 를 설정하세요.")
    else:
        print("[merge] stage2: 대상 이미지 0장 (role: local / class_map / 경로 확인)")

    print("\n[merge] 완료.")
    print(f"  stage1 → {stage1_out / 'data.yaml'}  (in-domain 미포함)")
    print(f"  stage2 → {stage2_out / 'data.yaml'}  (val/test = in-domain, 실험 단위 분리)")
    print("다음: python train.py --config train_config.stage1.yaml")


if __name__ == "__main__":
    main()
