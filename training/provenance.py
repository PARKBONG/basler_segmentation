"""
모델 출처 기록 — 어떤 코드 · 설정 · 데이터로 만든 모델인지.

  학습 시작 (train.py)  →  <실행 폴더>/run_info.yaml
                           + 커밋 안 된 수정이 있으면 <실행 폴더>/code.patch (git diff HEAD)
  export   (train.py)  →  .onnx 메타데이터에 train_commit · train_dirty · train_run · train_config
  보관     (archive.py) →  ../weights/<name>/<시각>/ 에 run_info.yaml 째로 복사

Ultralytics 도 .pt 안에 git 정보를 넣지만(ckpt["git"]) 커밋 안 된 수정은 모르고, .venv 가
저장소 안에 있을 때만 우리 저장소를 잡는다 — 그래서 여기서 직접 남긴다. run_info.yaml 이
없는 예전 실행은 그 ckpt["git"] 으로 대신한다 (from_checkpoint).
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from common import HERE, IMG_EXTS

REPO = HERE.parent
RUN_INFO = "run_info.yaml"
PATCH = "code.patch"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              encoding="utf-8", check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_state() -> dict:
    """현재 커밋 · 브랜치 · 커밋 안 된 파일(추적 중인 수정 + 새 파일)."""
    changed = [line[3:] for line in _git("status", "--porcelain").splitlines() if line.strip()]
    return {
        "commit": _git("rev-parse", "HEAD").strip() or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip() or None,
        "subject": _git("log", "-1", "--format=%s").strip() or None,
        "dirty": bool(changed),
        "uncommitted": changed,
    }


def _count_images(data_yaml: Path) -> dict:
    """data.yaml 의 split 별 이미지 수 (폴더 목록이면 합)."""
    try:
        d = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    root = Path(d.get("path") or data_yaml.parent)
    out = {}
    for split in ("train", "val", "test"):
        dirs = d.get(split)
        if not dirs:
            continue
        dirs = dirs if isinstance(dirs, list) else [dirs]
        out[split] = sum(1 for s in dirs if (root / s).is_dir()
                         for p in (root / s).rglob("*") if p.suffix.lower() in IMG_EXTS)
    return out


def write_run_info(save_dir: Path, config_path: Path, cfg: dict, data_yaml: Path) -> dict:
    """학습 시작 시 실행 폴더에 run_info.yaml (+ dirty 면 code.patch)."""
    save_dir = Path(save_dir)
    git = git_state()
    info = {
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": save_dir.relative_to(HERE).as_posix() if save_dir.is_relative_to(HERE) else str(save_dir),
        "config": config_path.name,
        "git": git,
        "data": {"yaml": str(data_yaml), "images": _count_images(data_yaml)},
        "config_resolved": cfg,   # base 상속까지 합친 config 전체 (blur · brightness 등 포함)
    }
    if git["dirty"]:
        patch = _git("diff", "HEAD", "--binary")
        if patch:
            (save_dir / PATCH).write_text(patch, encoding="utf-8")
            info["git"]["patch"] = PATCH
    (save_dir / RUN_INFO).write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
                                     encoding="utf-8")
    return info


def from_checkpoint(weights: Path) -> dict:
    """run_info.yaml 이 없는 예전 실행 — Ultralytics 가 .pt 에 넣은 git 정보로 대신."""
    import torch

    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    g = ck.get("git") or {}
    return {
        "started": None,
        "saved": ck.get("date"),
        "run": None,
        "config": None,
        "git": {"commit": g.get("commit"), "branch": g.get("branch"), "subject": g.get("message"),
                "dirty": None, "uncommitted": None,
                "note": "run_info.yaml 이전 실행 — Ultralytics ckpt['git'] (커밋 안 된 수정 여부 모름)"},
    }


def read(weights: Path) -> dict:
    """<실행 폴더>/weights/x.pt → 그 실행의 run_info (없으면 .pt 의 git 정보)."""
    weights = Path(weights)
    run_dir = weights.parent.parent
    p = run_dir / RUN_INFO
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    info = from_checkpoint(weights)
    info["run"] = run_dir.relative_to(HERE).as_posix() if run_dir.is_relative_to(HERE) else str(run_dir)
    return info


def stamp_onnx(onnx_path: Path, info: dict) -> None:
    """.onnx 메타데이터에 출처를 덧붙인다 (Ultralytics 메타데이터는 그대로 둠)."""
    import onnx

    git = info.get("git") or {}
    extra = {
        "train_commit": git.get("commit") or "unknown",
        "train_dirty": {True: "true", False: "false"}.get(git.get("dirty"), "unknown"),
        "train_run": info.get("run") or "unknown",
        "train_config": info.get("config") or "unknown",
    }
    model = onnx.load(str(onnx_path))
    keep = [p for p in model.metadata_props if p.key not in extra]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    for k, v in extra.items():
        model.metadata_props.add(key=k, value=str(v))
    onnx.save(model, str(onnx_path))
