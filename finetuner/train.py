"""
yolo26s-obb 파인튜닝 + ONNX export.

코드만 여기(로컬)에서 작성하고, 실제 학습은 RTX 5090 머신에서 돌립니다.
데이터셋은 preprocess.py 산출물의 data.yaml 을 사용합니다.

파이프라인:  download.py → preprocess.py → train.py

**config 는 반드시 --config 로 명시합니다.** 후보가 여럿(단일/stage1/stage2)이라
기본값을 고르면 의도와 다른 설정으로 학습해도 알아채지 못하기 때문입니다. 생략하면
바로 오류이며, 읽은 config 경로는 항상 로그 첫 줄에 찍힙니다.

config 의 stages 로 무엇을 돌릴지 고릅니다 (블록 자체가 필수 — 빠지면 오류):
    preview_aug : 증강이 실제로 어떻게 먹는지 눈으로 확인 (학습 전 점검용)
    train       : 파인튜닝
    export      : best.pt → ONNX → 앱 Models/ 배치
재학습 없이 export 만 다시 하려면 train: false, export: true 로 두면 됩니다.

사용법(5090 머신):
    pip install -r requirements.txt
    python train.py --config train.single.yaml     # 단일 스테이지
    python train.py --config train.stage1.yaml     # 공개 warm-up
    python train.py --config train.stage2.yaml     # in-domain 적응
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from common import IMG_EXTS, Stage, load_yaml


class Trainer(Stage):
    """
    파인튜닝과 ONNX export 를 담당하는 단계.

    스테이지마다 다른 yaml 을 쓰므로(`Trainer("train.stage1.yaml")`) 후보가 여럿이면
    기본값을 고르지 않습니다 — 조용히 엉뚱한 설정으로 학습하는 것을 막기 위함입니다.
    """

    config_glob = "train.*.yaml"
    label = "train"

    def __init__(self, config_path=None) -> None:
        super().__init__(config_path)
        self._save_dir: Path | None = None   # 이번 실행에서 학습한 결과 폴더

    # -- 경로 ------------------------------------------------------------

    def data_path(self) -> Path:
        """
        데이터셋 정의를 절대경로로 (Ultralytics 의 상대경로 해석 이슈 회피).

        두 가지로 적을 수 있습니다:
            data: ../datasets/processed/rf_a/data.yaml   경로 그대로 (한 소스)
            data: rf_a, rf_b                             소스 **이름** 목록 → 그 자리에서 통합
        이름으로 적으면 datasets/processed/<이름>/ 규약으로 코드가 경로를 만듭니다.
        """
        raw = self.cfg.get("train", {}).get("data")
        if not raw:
            self.fail("train.data 가 없습니다 — 어떤 데이터셋으로 학습할지 기본값으로 "
                      "추측하지 않습니다.")
        # 경로 구분자나 .yaml 이 없으면 소스 이름 목록으로 본다 (yaml 은 쉼표 나열을
        # 리스트가 아니라 문자열로 읽으므로 둘 다 받는다).
        names = raw if isinstance(raw, list) else str(raw).split(",")
        names = [str(n).strip() for n in names if str(n).strip()]
        if names and not any(s in n for n in names for s in ("/", "\\", ".yaml")):
            return self.merged_data_path(names)
        path = self.resolve(raw)
        if not path.exists():
            raise FileNotFoundError(
                f"데이터셋 정의가 없습니다: {path}\n"
                f"먼저 python preprocess.py --config <yaml> 으로 "
                f"데이터셋을 만드세요 (공개셋은 python download.py 선행)."
            )
        return path

    def merged_data_path(self, names: list) -> Path:
        """
        소스 이름 목록(`data: rf_a, rf_b`)을 묶은 data.yaml 을 쓰고 그 경로를 준다.

        모양은 preprocess.py 의 통합본과 같습니다 — split 마다 폴더 목록을 주면
        Ultralytics 가 알아서 합쳐 읽습니다. 소스 조합이 바뀌면 파일도 새로 써집니다.
        """
        root = self.resolve("../datasets/processed").resolve()
        missing = [n for n in names if not (root / n / "data.yaml").exists()]
        if missing:
            have = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
            self.fail(f"산출물이 없는 소스: {', '.join(missing)}\n"
                      f"  {root} 의 소스: {', '.join(have) or '(없음)'}\n"
                      f"  먼저 python preprocess.py 로 구우세요.")

        def has_images(d: Path) -> bool:
            return d.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in d.iterdir())

        merged = {"path": str(root)}
        for split in ("train", "val", "test"):
            dirs = [f"{n}/images/{split}" for n in names
                    if has_images(root / n / "images" / split)]
            if dirs:                       # 비었거나 없는 split 은 키 자체를 넣지 않는다
                merged[split] = dirs
        if "train" not in merged:
            self.fail(f"train 이미지가 있는 소스가 없습니다: {', '.join(names)}")
        # 클래스 정의는 preprocess 가 소스마다 같은 names 로 굽습니다 — 첫 소스 것을 씁니다.
        merged["names"] = load_yaml(root / names[0] / "data.yaml")["names"]

        out = root / f"data.{'+'.join(names)}.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)
        self.log(f"소스 {len(names)}개 통합 → {out}")
        return out

    def start_weights(self) -> str:
        """
        시작 가중치. 기본값 없음 — stage2 는 stage1 checkpoint 를 가리켜야 하는데,
        키를 빠뜨렸을 때 조용히 COCO 에서 다시 시작하면 warm-start 가 사라집니다.

        경로 구분자가 있으면 체크포인트 파일로 보고 finetuner/ 기준으로 해석하고 존재를
        확인합니다. 구분자가 없으면(`yolo26s-obb.pt`) Ultralytics 가 받아올 모델 이름이라
        그대로 넘깁니다.
        """
        model = str(self.cfg.get("train", {}).get("model") or "").strip()
        if not model:
            self.fail("train.model 이 없습니다 — 시작 가중치를 기본값으로 추측하지 "
                      "않습니다 (예: yolo26s-obb.pt 또는 runs/obb/stage1/weights/best.pt).")
        if "/" not in model and "\\" not in model:
            return model                      # Ultralytics 가 이름으로 해석/다운로드
        path = self.resolve(model)
        if not path.exists():
            raise FileNotFoundError(
                f"시작 가중치가 없습니다: {path}\n"
                f"{self.config_path.name} 의 train.model 을 확인하세요 "
                f"(stage2 는 stage1 을 먼저 학습해야 합니다)."
            )
        return str(path)

    def best_weights(self) -> Path:
        """best.pt 위치. 이번 실행에서 학습했다면 그 결과 폴더를, 아니면 config 로 유도."""
        if self._save_dir is not None:
            return self._save_dir / "weights" / "best.pt"
        train_cfg = self.cfg.get("train", {})
        project = self.resolve(train_cfg.get("project", "runs/obb"))
        name = train_cfg.get("name", "yolo26s-obb-finetune")
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
        pv = self.cfg.get("preview") or {}
        cfg = dict(self.cfg.get("train", {}))
        weights = self.start_weights()      # 설정 검증을 무거운 import 앞에 둔다
        cfg.pop("model", None)
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

        from ultralytics import YOLO

        results = YOLO(weights).train(**cfg)
        save_dir = Path(results.save_dir)
        batches = sorted(save_dir.glob("train_batch*.jpg"))
        self.log(f"증강 미리보기 {len(batches)}장 → {save_dir}")
        self.log("train_batch*.jpg = 학습이 실제로 먹는 배치(mosaic·hsv·flip·scale 반영).")
        return save_dir

    def train(self) -> Path:
        """파인튜닝. best.pt 경로를 돌려준다."""
        cfg = dict(self.cfg.get("train", {}))
        weights = self.start_weights()      # 설정 검증을 무거운 import 앞에 둔다
        cfg.pop("model", None)
        cfg["data"] = str(self.data_path())
        self.log(f"시작 가중치: {weights}")
        self.log(f"데이터셋: {cfg['data']}")

        from ultralytics import YOLO

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
                f"학습을 먼저 끝내거나 train.*.yaml 의 train.project/name 을 확인하세요."
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

    STAGE_KEYS = ("preview_aug", "train", "export")

    def run(self) -> None:
        """
        stages 블록은 필수이고 키 오타도 오류입니다 — `expor: true` 를 조용히 무시하면
        export 를 돌린 줄 알고 끝나 버립니다.
        """
        stages = self.cfg.get("stages")
        if not isinstance(stages, dict) or not stages:
            self.fail("stages 블록이 필요합니다 — 무엇을 돌릴지 기본값으로 정하지 않습니다.\n"
                      "  stages:\n"
                      "    preview_aug: false\n"
                      "    train: true\n"
                      "    export: true")
        unknown = [k for k in stages if k not in self.STAGE_KEYS]
        if unknown:
            self.fail(f"stages 에 모르는 키: {', '.join(unknown)}\n"
                      f"  쓸 수 있는 키: {', '.join(self.STAGE_KEYS)}")
        if not any(bool(stages.get(k)) for k in self.STAGE_KEYS):
            self.fail("stages 가 전부 false 입니다 — 할 일이 없습니다.")

        self.log("stages: " + ", ".join(k for k in self.STAGE_KEYS if stages.get(k)))
        if stages.get("preview_aug"):
            self.preview_aug()
        if stages.get("train"):
            self.train()
        if stages.get("export"):
            self.export()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="yolo26s-obb 파인튜닝 + ONNX export",
        epilog="configs/ 의 후보: " + Trainer.config_candidates(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config", metavar="YAML",
        help="학습 설정 yaml (후보가 여럿이라 사실상 필수). configs/ 안 파일명 또는 경로",
    )
    Trainer(ap.parse_args().config).run()


if __name__ == "__main__":
    main()
