"""
yolo26s-seg 2단계 파인튜닝 — 단일 stage 실행 (독립 실행 + 수동 checkpoint 지정).

코드만 여기(로컬)에서 작성하고, 실제 학습은 RTX 5090 머신에서 돌립니다.
설정은 --config 로 지정한 yaml 에서 전부 읽습니다.

파이프라인:
    python download.py                                    # (선택) 공개셋 다운로드
    python merge.py                                        # stage1/stage2 데이터 생성
    python train.py --config train_config.stage1.yaml     # Stage1 (공개 warm-up)
    #   → 평가:  python eval.py --weights runs/segment/stage1/weights/best.pt \
    #                           --data ../datasets/stage1/data.yaml --split val
    #   → train_config.stage2.yaml 의 model: 에 stage1 checkpoint 를 직접 지정
    python train.py --config train_config.stage2.yaml     # Stage2 (in-domain 적응)
    python export.py                                       # ONNX → 앱 Models/

사용법(5090 머신):
    pip install -r requirements.txt
    python train.py --config train_config.stage1.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def resolve(path_str: str) -> Path:
    """train/ 폴더 기준으로 상대경로를 절대경로로 변환."""
    p = Path(path_str)
    return p if p.is_absolute() else (HERE / p)


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO seg 단일 stage 파인튜닝")
    ap.add_argument("--config", default="train_config.stage1.yaml",
                    help="stage 설정 yaml (train/ 기준 상대경로 가능)")
    args = ap.parse_args()

    config_path = resolve(args.config)
    cfg = load_config(config_path)
    train_cfg = dict(cfg.get("train", {}))
    print(f"[train] config: {config_path.name}")

    # 시작 가중치: 실제 파일이면 절대경로로 고정, 아니면 이름 그대로(자동 다운로드).
    #   - "yolo26s-seg.pt"           → 파일 없음 → 이름 전달 → Ultralytics 자동 다운로드
    #   - "runs/segment/stage1/.../best.pt" → 파일 있음 → 절대경로로 전달 (수동 체이닝)
    model_str = train_cfg.pop("model", "yolo26s-seg.pt")
    model_path = resolve(model_str)
    if model_path.exists():
        model_arg = str(model_path)
        print(f"[train] 시작 가중치(파일): {model_arg}")
    else:
        model_arg = model_str
        print(f"[train] 시작 가중치(이름/자동다운로드): {model_arg}")

    # 데이터셋 경로를 절대경로로 고정 (Ultralytics 의 상대경로 해석 이슈 회피)
    data_path = resolve(train_cfg.get("data", "../datasets/stage1/data.yaml"))
    if not data_path.exists():
        raise FileNotFoundError(
            f"데이터셋 정의가 없습니다: {data_path}\n"
            f"먼저 python merge.py 로 데이터셋을 생성하세요 (공개셋은 python download.py 선행)."
        )
    train_cfg["data"] = str(data_path)

    model = YOLO(model_arg)
    # train_cfg 의 나머지 키는 모두 Ultralytics train() 인자와 1:1 대응
    results = model.train(**train_cfg)

    best = Path(results.save_dir) / "weights" / "best.pt"
    print("\n[완료] best.pt:", best)
    print(f"평가:  python eval.py --weights {best} --data {data_path} --split val")


if __name__ == "__main__":
    main()
