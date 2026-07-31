"""
파이프라인 공통 기반 — 각 단계가 공유하는 config 로드 · 경로 해석 · 로그.

download 와 preprocess 는 통합 설정 download_config.yaml 을 자동으로 읽습니다
(config 인자 없음). 다만 후보가 여럿인 선택은 **반드시 명시**해야 합니다 —
기본값으로 조용히 넘어가면 의도한 것과 다른 데이터/설정으로 학습해도 알 수
없기 때문입니다. preprocess 는 스테이지(단일/stage1/stage2)를, train 은 config
파일을 명시합니다:

    download.py                          공개셋 획득 (download_config.yaml 자동)
    preprocess.py --stage stage1         데이터셋 굽기 (같은 yaml 의 stages 중 하나)
    train.py --config train_config.stage1.yaml   학습 + ONNX export

어느 쪽이든 읽은 config 경로는 항상 로그 첫 줄에 찍습니다.
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

        Downloader().run(); Preprocessor("single").run(); Trainer("train_config.yaml").run()

    `explicit_config = True` 인 단계는 config 를 생략할 수 없습니다 — 후보가 여러 개라
    기본값을 고르는 순간 "무엇으로 학습했는지" 를 잃기 때문입니다.
    (preprocess 는 config 가 하나가 됐지만 같은 이유로 스테이지를 생략할 수 없습니다.)
    """

    config_name: str = ""      # 단일 config 단계의 기본 파일명 / 오류 메시지의 예시
    config_glob: str = ""      # explicit_config 일 때 후보를 찾을 패턴
    explicit_config: bool = False
    label: str = "stage"       # 로그 접두사 겸 스크립트 이름(<label>.py)

    def __init__(self, config_path=None) -> None:
        if not self.config_name:
            raise NotImplementedError(f"{type(self).__name__}: config_name 을 정의하세요.")

        if config_path is None:
            if self.explicit_config:
                raise SystemExit(
                    f"[{self.label}] 읽을 config 를 명시하세요 — 기본값으로 조용히 넘어가지 "
                    f"않습니다.\n"
                    f"  python {self.label}.py --config <yaml>\n"
                    f"  finetuner/ 의 후보: {self.config_candidates()}"
                )
            config_path = self.config_name

        self.config_path = resolve(config_path)
        if self.explicit_config and not self.config_path.exists():
            # 이름을 잘못 적었을 때 traceback 대신 후보를 보여준다
            raise SystemExit(
                f"[{self.label}] config 파일이 없습니다: {self.config_path}\n"
                f"  finetuner/ 의 후보: {self.config_candidates()}"
            )
        self.cfg = load_yaml(self.config_path)
        self.log(f"config: {self.config_path}")   # 무엇을 읽었는지 항상 남긴다

    @classmethod
    def config_candidates(cls) -> str:
        """finetuner/ 에서 이 단계가 쓸 수 있는 config 파일 이름들."""
        found = sorted(p.name for p in HERE.glob(cls.config_glob or cls.config_name))
        return ", ".join(found) or "(없음)"

    @staticmethod
    def resolve(path_str) -> Path:
        return resolve(path_str)

    def log(self, msg: str) -> None:
        print(f"[{self.label}] {msg}")

    def run(self) -> None:
        raise NotImplementedError
