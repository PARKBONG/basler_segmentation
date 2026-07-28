"""
학습된 checkpoint 를 지정한 split(train/val/test) 에 대해 평가.

Stage1/Stage2 를 명확히, 독립적으로 평가하기 위한 스크립트 (Box + Mask mAP 출력).

데이터 위생:
  · Stage1 은 공개 val 로 평가 (in-domain 격리 유지).
  · Stage2 선택 지표는 in-domain val.
  · in-domain test 는 "최종 stage2 모델"에 딱 한 번만 사용하세요. 이 수치로
    하이퍼파라미터를 다시 만지면 그 순간 오염되어 논문에 쓸 수 없습니다.

사용법:
    # Stage1 (공개 val 기준)
    python eval.py --weights runs/segment/stage1/weights/best.pt \
                   --data ../datasets/stage1/data.yaml --split val

    # Stage2 최종 (in-domain test — 딱 한 번)
    python eval.py --weights runs/segment/stage2/weights/best.pt \
                   --data ../datasets/stage2/data.yaml --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (HERE / p)


def _fmt(obj, name) -> str:
    try:
        return f"{getattr(obj, name):.4f}"
    except Exception:
        return "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO seg checkpoint 평가")
    ap.add_argument("--weights", required=True, help="평가할 .pt 가중치")
    ap.add_argument("--data", default="../datasets/stage2/data.yaml",
                    help="data.yaml (기본: stage2)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="평가 split (기본: val). test 는 최종 1회만")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    weights = resolve(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"가중치가 없습니다: {weights}")
    data = resolve(args.data)
    if not data.exists():
        raise FileNotFoundError(f"data.yaml 이 없습니다: {data}")

    if args.split == "test":
        print("[eval][주의] test 는 최종 무편향 평가용입니다. 이 결과를 보고 "
              "하이퍼파라미터를 다시 조정하면 오염됩니다.")

    model = YOLO(str(weights))
    m = model.val(data=str(data), split=args.split, imgsz=args.imgsz, device=args.device)

    print(f"\n[eval] weights={weights.name}  data={data.parent.name}  split={args.split}")
    print(f"  Box  mAP50-95={_fmt(m.box, 'map')}  mAP50={_fmt(m.box, 'map50')}")
    seg = getattr(m, "seg", None)
    if seg is not None:
        print(f"  Mask mAP50-95={_fmt(seg, 'map')}  mAP50={_fmt(seg, 'map50')}")


if __name__ == "__main__":
    main()
