"""
파이프라인 공통 기반 — 각 단계가 공유하는 config 로드 · 경로 해석 · 로그.

각 단계는 자기 이름의 yaml 하나만 읽습니다 (명령행 인자 없음):

    download.py    → download_config.yaml     공개셋 획득
    preprocess.py  → preprocess_config.yaml   병합 + 크롭 (+ 라벨 변환)
    upload.py      → upload_config.yaml       로컬 폴더를 Roboflow 로 되돌려 보냄
    train.py       → train_config.yaml        학습 + ONNX export

경로는 모두 이 폴더(finetuner/) 기준 상대경로이거나 절대경로입니다.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve(path_str) -> Path:
    """finetuner/ 폴더 기준으로 상대경로를 절대경로로 변환."""
    p = Path(path_str)
    return p if p.is_absolute() else (HERE / p)


def roboflow_api_key() -> str:
    """Roboflow API 키 — 보안상 yaml 이 아니라 환경변수에서만 읽습니다."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "환경변수 ROBOFLOW_API_KEY 가 필요합니다.\n"
            '  PowerShell:  $env:ROBOFLOW_API_KEY="xxxx"\n'
            "  bash:        export ROBOFLOW_API_KEY=xxxx"
        )
    return key


def count_images(root: Path) -> int:
    """폴더 아래 이미지 파일 수 (업로드 전 규모 확인용)."""
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def normalize_names(names) -> dict:
    """names(list 또는 dict) → {idx: name} 로 정규화."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    raise ValueError("names 를 찾을 수 없습니다.")


class Stage:
    """
    파이프라인 한 단계.

    서브클래스는 `config_name` 과 `run()` 만 정의하면 됩니다. 단계끼리 import 해서
    한 프로세스 안에서 이어 붙이는 것도 가능합니다:

        Downloader().run(); Preprocessor().run(); Trainer().run()
    """

    config_name: str = ""
    label: str = "stage"

    def __init__(self, config_path=None) -> None:
        if not self.config_name:
            raise NotImplementedError(f"{type(self).__name__}: config_name 을 정의하세요.")
        self.config_path = Path(config_path) if config_path else HERE / self.config_name
        self.cfg = load_yaml(self.config_path)

    @staticmethod
    def resolve(path_str) -> Path:
        return resolve(path_str)

    def log(self, msg: str) -> None:
        print(f"[{self.label}] {msg}")

    def run(self) -> None:
        raise NotImplementedError
