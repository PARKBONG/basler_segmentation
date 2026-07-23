"""
yolo26s-seg 파인튜닝 스크립트.

코드만 여기(로컬)에서 작성하고, 실제 학습은 RTX 5090 머신에서 돌립니다.
설정은 전부 train_config.yaml 에서 읽습니다 (명령행 인자 없음).
데이터셋은 merge.py 가 만든 datasets/merged/data.yaml 을 사용합니다.

파이프라인:  download.py → merge.py → train.py → export.py

사용법(5090 머신):
    pip install -r requirements.txt
    python train.py
"""
from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "train_config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    """train/ 폴더 기준으로 상대경로를 절대경로로 변환."""
    p = Path(path_str)
    return p if p.is_absolute() else (HERE / p)


def main() -> None:
    cfg = load_config()
    train_cfg = dict(cfg.get("train", {}))
    model_weights = train_cfg.pop("model", "yolo26s-seg.pt")

    # 데이터셋 경로를 절대경로로 고정 (Ultralytics 의 상대경로 해석 이슈 회피)
    data_path = resolve(train_cfg.get("data", "../datasets/merged/data.yaml"))
    if not data_path.exists():
        raise FileNotFoundError(
            f"데이터셋 정의가 없습니다: {data_path}\n"
            f"먼저 python merge.py 로 데이터셋을 병합하세요 "
            f"(공개셋은 python download.py 선행)."
        )
    train_cfg["data"] = str(data_path)

    model = YOLO(model_weights)
    # train_cfg 의 나머지 키는 모두 Ultralytics train() 인자와 1:1 대응
    results = model.train(**train_cfg)

    best = Path(results.save_dir) / "weights" / "best.pt"
    print("\n[완료] best.pt:", best)
    print("다음 단계: python export.py  (ONNX 변환 → 앱 Models/ 배치)")


if __name__ == "__main__":
    main()
