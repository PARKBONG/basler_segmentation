"""
yolo26s-seg 파인튜닝 + ONNX export.

코드만 여기(로컬)에서 작성하고, 실제 학습은 RTX 5090 머신에서 돌립니다.
설정은 전부 train_config.yaml 에서 읽습니다 (명령행 인자 없음).
데이터셋은 preprocess.py 가 만든 datasets/processed/data.yaml 을 사용합니다.

파이프라인:  download.py → preprocess.py → train.py

train_config.yaml 의 stages 로 무엇을 돌릴지 고릅니다:
    preview_aug : 증강이 실제로 어떻게 먹는지 눈으로 확인 (학습 전 점검용)
    train       : 파인튜닝
    export      : best.pt → ONNX → 앱 Models/ 배치
재학습 없이 export 만 다시 하려면 train: false, export: true 로 두면 됩니다.

사용법(5090 머신):
    pip install -r requirements.txt
    python train.py
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from common import Stage


class Trainer(Stage):
    """파인튜닝과 ONNX export 를 담당하는 단계."""

    config_name = "train_config.yaml"
    label = "train"

    def __init__(self, config_path=None) -> None:
        super().__init__(config_path)
        self._save_dir: Path | None = None   # 이번 실행에서 학습한 결과 폴더

    # -- 경로 ------------------------------------------------------------

    def data_path(self) -> Path:
        """데이터셋 정의를 절대경로로 (Ultralytics 의 상대경로 해석 이슈 회피)."""
        raw = self.cfg.get("train", {}).get("data", "../datasets/processed/data.yaml")
        path = self.resolve(raw)
        if not path.exists():
            raise FileNotFoundError(
                f"데이터셋 정의가 없습니다: {path}\n"
                f"먼저 python preprocess.py 로 데이터셋을 만드세요 "
                f"(공개셋은 python download.py 선행)."
            )
        return path

    def best_weights(self) -> Path:
        """best.pt 위치. 이번 실행에서 학습했다면 그 결과 폴더를, 아니면 config 로 유도."""
        if self._save_dir is not None:
            return self._save_dir / "weights" / "best.pt"
        train_cfg = self.cfg.get("train", {})
        project = self.resolve(train_cfg.get("project", "runs/segment"))
        name = train_cfg.get("name", "yolo26s-seg-finetune")
        return project / name / "weights" / "best.pt"

    # -- 단계 ------------------------------------------------------------

    def preview_aug(self) -> Path:
        """
        증강 결과 미리보기.

        증강 파이프라인을 따로 재현하지 않고, 실제 학습을 1 epoch(그것도 fraction 만큼만)
        돌려서 Ultralytics 가 남기는 train_batch*.jpg 를 얻습니다. 학습이 실제로 보는
        배치를 그대로 그린 그림이라 재현 코드가 어긋날 여지가 없고, 마스크까지 함께
        그려집니다. 결과는 runs/preview/ 로 나가므로 실제 학습 결과를 건드리지 않습니다.
        """
        from ultralytics import YOLO

        pv = self.cfg.get("preview") or {}
        cfg = dict(self.cfg.get("train", {}))
        weights = cfg.pop("model", "yolo26s-seg.pt")
        cfg["data"] = str(self.data_path())
        cfg.update(
            epochs=1,
            fraction=float(pv.get("fraction", 0.1)),
            project=str(self.resolve(pv.get("project", "runs/preview"))),
            name=pv.get("name", "aug"),
            exist_ok=True,
            plots=True,
            val=False,
            save=False,
            resume=False,
        )

        results = YOLO(weights).train(**cfg)
        save_dir = Path(results.save_dir)
        batches = sorted(save_dir.glob("train_batch*.jpg"))
        self.log(f"증강 미리보기 {len(batches)}장 → {save_dir}")
        self.log("train_batch*.jpg = 학습이 실제로 먹는 배치(mosaic·hsv·flip·scale 반영).")
        return save_dir

    def train(self) -> Path:
        """파인튜닝. best.pt 경로를 돌려준다."""
        from ultralytics import YOLO

        cfg = dict(self.cfg.get("train", {}))
        weights = cfg.pop("model", "yolo26s-seg.pt")
        cfg["data"] = str(self.data_path())

        # cfg 의 나머지 키는 모두 Ultralytics train() 인자와 1:1 대응 (증강 포함)
        results = YOLO(weights).train(**cfg)

        self._save_dir = Path(results.save_dir)
        best = self.best_weights()
        self.log(f"학습 완료 → {best}")
        return best

    def export(self) -> Path:
        """best.pt → ONNX. deploy_to 가 있으면 앱 Models/ 로도 복사한다."""
        from ultralytics import YOLO

        ex = dict(self.cfg.get("export", {}))
        deploy_to = (ex.pop("deploy_to", "") or "").strip()

        weights = self.best_weights()
        if not weights.exists():
            raise FileNotFoundError(
                f"가중치를 찾을 수 없습니다: {weights}\n"
                f"학습을 먼저 끝내거나 train_config.yaml 의 train.project/name 을 확인하세요."
            )
        self.log(f"가중치: {weights}")

        exported = Path(YOLO(str(weights)).export(
            format=ex.get("format", "onnx"),
            imgsz=self.cfg.get("train", {}).get("imgsz", 640),  # 학습 설정과 항상 일치
            opset=ex.get("opset", 12),
            simplify=ex.get("simplify", True),
            half=ex.get("half", False),
            dynamic=ex.get("dynamic", False),
        ))
        self.log(f"생성: {exported}")

        if deploy_to:
            base = self.resolve(deploy_to)
            base.parent.mkdir(parents=True, exist_ok=True)
            # 기존 파일은 절대 건드리지 않고, 새 파일에 타임스탬프를 붙여 생성
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = base.with_name(f"{base.stem}.{stamp}{base.suffix}")
            shutil.copyfile(exported, dest)   # runs/ 의 원본도 그대로 유지(복사)
            self.log(f"앱에 배치 완료(신규): {dest}")
        else:
            self.log("export.deploy_to 비어있음 → 복사 생략")

        return exported

    # -- 실행 ------------------------------------------------------------

    def run(self) -> None:
        stages = self.cfg.get("stages") or {}
        if stages.get("preview_aug"):
            self.preview_aug()
        if stages.get("train", True):
            self.train()
        if stages.get("export", True):
            self.export()


if __name__ == "__main__":
    Trainer().run()
