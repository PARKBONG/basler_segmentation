"""
파이프라인 공통 기반 — 각 단계가 공유하는 config 로드 · 경로 해석 · 로그.

설정 yaml 은 모두 configs/ 아래에 있고, 모든 스크립트의 인자는 --config 하나입니다.
각 스크립트는 자기 패턴(config_glob)으로 configs/ 를 찾아서:
  · 후보가 하나뿐이면 --config 없이 그 파일을 자동으로 읽습니다 (고를 게 없어 모호하지 않음).
  · 후보가 여럿이면 --config 를 **반드시 명시**해야 합니다 — 기본값으로 조용히
    넘어가면 의도한 것과 다른 데이터/설정으로 학습해도 알 수 없기 때문입니다.

    download.py                                  configs/download.yaml 자동 (후보 1개)
    preprocess.py --config preprocess.stage1.yaml    데이터셋 굽기
    train.py --config train.stage1.yaml              학습 + ONNX export
    eval.py --config eval.stage2.yaml                체크포인트 평가

어느 쪽이든 읽은 config 경로는 항상 로그 첫 줄에 찍습니다.
--config 인자는 configs/ 안 파일명으로도, 경로 그대로(finetuner/ 기준)로도 됩니다.
yaml **내용물**의 경로(../datasets/… 등)는 configs/ 가 아니라 finetuner/ 기준입니다.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"

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

    서브클래스는 `config_glob` 과 `run()` 만 정의하면 됩니다. 단계끼리 import 해서
    한 프로세스 안에서 이어 붙이는 것도 가능합니다:

        Downloader().run(); Preprocessor("preprocess.single.yaml").run()
        Trainer("train.single.yaml").run()

    config 인자를 생략하면: configs/ 의 후보가 하나뿐일 때만 자동으로 그 파일을
    읽고, 여럿이면 후보 목록과 함께 거절합니다 — 기본값을 고르는 순간
    "무엇으로 학습했는지" 를 잃기 때문입니다.
    """

    config_glob: str = ""      # configs/ 에서 이 단계의 config 를 찾는 패턴
    label: str = "stage"       # 로그 접두사 겸 스크립트 이름(<label>.py)

    def __init__(self, config_path=None) -> None:
        if not self.config_glob:
            raise NotImplementedError(f"{type(self).__name__}: config_glob 을 정의하세요.")

        if config_path is None:
            found = self.candidate_paths()
            if len(found) == 1:
                config_path = found[0]   # 후보가 하나면 자동 — 고를 게 없어 모호하지 않다
            else:
                raise SystemExit(
                    f"[{self.label}] 읽을 config 를 명시하세요 — 후보가 "
                    f"{'없어' if not found else '여럿이라'} 기본값으로 조용히 넘어가지 "
                    f"않습니다.\n"
                    f"  python {self.label}.py --config <yaml>\n"
                    f"  configs/ 의 후보: {self.config_candidates()}"
                )

        # 경로 그대로(finetuner/ 기준) 먼저, 없으면 configs/<이름> 으로도 찾는다 —
        # `--config train.single.yaml` 처럼 파일명만 적어도 되게 하기 위함.
        self.config_path = resolve(config_path)
        if not self.config_path.exists():
            alt = CONFIG_DIR / str(config_path)
            if alt.exists():
                self.config_path = alt
        if not self.config_path.exists():
            # 이름을 잘못 적었을 때 traceback 대신 후보를 보여준다
            raise SystemExit(
                f"[{self.label}] config 파일이 없습니다: {self.config_path}\n"
                f"  configs/ 의 후보: {self.config_candidates()}"
            )
        self.cfg = load_yaml(self.config_path)
        self.log(f"config: {self.config_path}")   # 무엇을 읽었는지 항상 남긴다

    @classmethod
    def candidate_paths(cls) -> list:
        """configs/ 에서 이 단계가 쓸 수 있는 config 파일들."""
        return sorted(CONFIG_DIR.glob(cls.config_glob))

    @classmethod
    def config_candidates(cls) -> str:
        found = [p.name for p in cls.candidate_paths()]
        return ", ".join(found) or "(없음)"

    @staticmethod
    def resolve(path_str) -> Path:
        return resolve(path_str)

    def log(self, msg: str) -> None:
        print(f"[{self.label}] {msg}")

    def run(self) -> None:
        raise NotImplementedError
