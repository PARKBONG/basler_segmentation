"""
공개(Roboflow Universe) 데이터셋을 SDK로 다운로드.

merge_config.yaml 의 각 소스 중 roboflow: 블록이 있는 것만 받아서
해당 소스의 path 폴더에 저장합니다.

- 공개 데이터셋도 다운로드에는 본인 무료 API 키가 필요.
- API 키는 보안상 yaml 이 아니라 환경변수 ROBOFLOW_API_KEY 에서 읽습니다.

사용법:
    pip install roboflow
    # Windows PowerShell:  $env:ROBOFLOW_API_KEY="xxxx"
    # bash:                export ROBOFLOW_API_KEY=xxxx
    python download.py
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "merge_config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (HERE / p)


def main() -> None:
    cfg = load_config()
    targets = [s for s in cfg.get("sources", []) if s.get("roboflow")]
    if not targets:
        print("[download] roboflow: 블록이 있는 소스가 없습니다. 받을 게 없어요.")
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "환경변수 ROBOFLOW_API_KEY 가 필요합니다.\n"
            "  PowerShell:  $env:ROBOFLOW_API_KEY=\"xxxx\"\n"
            "  bash:        export ROBOFLOW_API_KEY=xxxx"
        )

    from roboflow import Roboflow  # 로컬에 없으면: pip install roboflow

    rf = Roboflow(api_key=api_key)
    for s in targets:
        rb = s["roboflow"]
        if not rb.get("workspace") or not rb.get("project"):
            print(f"[download] {s['name']}: workspace/project 가 비어있어 건너뜀")
            continue
        dest = resolve(s["path"])
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[download] {s['name']} → {dest}")
        project = rf.workspace(rb["workspace"]).project(rb["project"])
        project.version(int(rb.get("version", 1))).download(
            rb.get("format", "yolov11"), location=str(dest), overwrite=True
        )

    print("[download] 완료. 다음: python merge.py")


if __name__ == "__main__":
    main()
