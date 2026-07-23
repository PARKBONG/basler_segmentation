"""
파인튜닝 산출물(best.pt) → ONNX 변환 → 앱(BaslerLiveView) Models/ 배치.

설정은 전부 export_config.yaml 에서 읽습니다 (명령행 인자 없음).

사용법(학습 완료 후):
    python export.py
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "export_config.yaml"
TRAIN_CONFIG_PATH = HERE / "train_config.yaml"


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    """train/ 폴더 기준으로 상대경로를 절대경로로 변환."""
    p = Path(path_str)
    return p if p.is_absolute() else (HERE / p)


def best_weights_from_train(train_cfg: dict) -> Path:
    """train_config.yaml 의 project/name 에서 best.pt 경로를 유도."""
    project = train_cfg.get("project", "runs/segment")
    name = train_cfg.get("name", "yolo26s-seg-finetune")
    return resolve(project) / name / "weights" / "best.pt"


def main() -> None:
    cfg = load_yaml(CONFIG_PATH)
    # 가중치(best.pt)와 imgsz 는 train_config.yaml 에서 가져옴
    train_cfg = load_yaml(TRAIN_CONFIG_PATH).get("train", {})

    weights = best_weights_from_train(train_cfg)
    if not weights.exists():
        raise FileNotFoundError(
            f"가중치를 찾을 수 없습니다: {weights}\n"
            f"학습을 먼저 끝내거나 train_config.yaml 의 project/name 을 확인하세요."
        )
    print("[export] 가중치:", weights)

    model = YOLO(str(weights))
    exported = Path(model.export(
        format=cfg.get("format", "onnx"),
        imgsz=train_cfg.get("imgsz", 640),   # train_config.yaml 과 항상 일치
        opset=cfg.get("opset", 12),
        simplify=cfg.get("simplify", True),
        half=cfg.get("half", False),
        dynamic=cfg.get("dynamic", False),
    ))
    print("[export] 생성:", exported)

    deploy_to = (cfg.get("deploy_to") or "").strip()
    if deploy_to:
        base = resolve(deploy_to)
        base.parent.mkdir(parents=True, exist_ok=True)
        # 기존 파일은 절대 건드리지 않고, 새 파일에 타임스탬프를 붙여 생성
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = base.with_name(f"{base.stem}.{stamp}{base.suffix}")
        shutil.copyfile(exported, dest)   # runs/ 의 원본도 그대로 유지(복사)
        print("[export] 앱에 배치 완료(신규):", dest)
    else:
        print("[export] deploy_to 비어있음 → 복사 생략")


if __name__ == "__main__":
    main()
