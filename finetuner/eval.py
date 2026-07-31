"""
학습된 checkpoint 를 지정한 split(train/val/test) 에 대해 평가.

Stage1/Stage2 를 명확히, 독립적으로 평가하기 위한 스크립트 (Box + Mask mAP 출력).
무엇을 평가할지는 전부 config(configs/eval.*.yaml — weights/data/split/imgsz/device)가
정합니다.

데이터 위생:
  · Stage1 은 공개 val 로 평가 (in-domain 격리 유지)      → eval.stage1.yaml
  · Stage2 선택 지표는 in-domain val                       → eval.stage2.yaml
  · in-domain test 는 "최종 stage2 모델"에 딱 한 번만      → eval.final.yaml
    이 수치로 하이퍼파라미터를 다시 만지면 그 순간 오염되어 논문에 쓸 수 없습니다.

사용법:
    python eval.py --config eval.stage1.yaml
    python eval.py --config eval.final.yaml     # 최종 1회만
"""
from __future__ import annotations

import argparse

from common import Stage


class Evaluator(Stage):
    """checkpoint 하나를 한 split 에 대해 평가하는 단계."""

    config_glob = "eval.*.yaml"
    label = "eval"

    @staticmethod
    def _fmt(obj, name) -> str:
        try:
            return f"{getattr(obj, name):.4f}"
        except Exception:
            return "n/a"

    def run(self) -> None:
        for key in ("weights", "data"):
            if not self.cfg.get(key):
                raise SystemExit(
                    f"[{self.label}] {self.config_path.name}: {key} 가 없습니다 — "
                    f"무엇을 평가할지 기본값으로 추측하지 않습니다."
                )
        split = str(self.cfg.get("split", "val"))
        if split not in ("train", "val", "test"):
            raise SystemExit(
                f"[{self.label}] {self.config_path.name}: split 은 train/val/test 중 "
                f"하나여야 합니다: {split}"
            )

        weights = self.resolve(self.cfg["weights"])
        if not weights.exists():
            raise FileNotFoundError(f"가중치가 없습니다: {weights}")
        data = self.resolve(self.cfg["data"])
        if not data.exists():
            raise FileNotFoundError(f"data.yaml 이 없습니다: {data}")

        if split == "test":
            self.log("[주의] test 는 최종 무편향 평가용입니다. 이 결과를 보고 "
                     "하이퍼파라미터를 다시 조정하면 오염됩니다.")

        from ultralytics import YOLO   # 무거운 import 는 설정/경로 검증 뒤로

        model = YOLO(str(weights))
        m = model.val(data=str(data), split=split,
                      imgsz=int(self.cfg.get("imgsz", 640)),
                      device=str(self.cfg.get("device", "0")))

        print(f"\n[eval] weights={weights.name}  data={data.parent.name}  split={split}")
        print(f"  Box  mAP50-95={self._fmt(m.box, 'map')}  mAP50={self._fmt(m.box, 'map50')}")
        seg = getattr(m, "seg", None)
        if seg is not None:
            print(f"  Mask mAP50-95={self._fmt(seg, 'map')}  mAP50={self._fmt(seg, 'map50')}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLO seg checkpoint 평가",
        epilog="configs/ 의 후보: " + Evaluator.config_candidates(),
    )
    ap.add_argument(
        "--config", metavar="YAML",
        help="평가 설정 yaml (후보가 여럿이라 사실상 필수). configs/ 안 파일명 또는 경로",
    )
    Evaluator(ap.parse_args().config).run()


if __name__ == "__main__":
    main()
