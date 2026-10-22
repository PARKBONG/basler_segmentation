"""
yolo26s-seg / yolo26s-obb 파인튜닝 + ONNX export.

데이터셋은 preprocess.py 산출물의 data.yaml 을 사용합니다.

파이프라인:  (라벨러) → preprocess.py → train.py → (eval.py · infer.py)

**config 는 반드시 --config 로 명시합니다.** 후보가 여럿(single/seg_all/obb_all)이라
기본값을 고르면 의도와 다른 설정으로 학습해도 알아채지 못하기 때문입니다. 생략하면
바로 오류이며, 읽은 config 경로는 항상 로그 첫 줄에 찍힙니다.

config 의 stages 로 무엇을 돌릴지 고릅니다 (블록 자체가 필수 — 빠지면 오류):
    preview_aug : 증강이 실제로 어떻게 먹는지 눈으로 확인 (학습 전 점검용)
    train       : 파인튜닝
    export      : best.pt → ONNX → 앱 Models/ 배치
재학습 없이 export 만 다시 하려면 train: false, export: true 로 두면 됩니다.

사용법 (training/ 에서, uv sync --extra train --extra export 후):
    uv run train.py --config train.single.yaml     # 측정용 (val/test 분리, early stop)
    uv run train.py --config train.seg_all.yaml    # 배포용 seg (전체 데이터)
    uv run train.py --config train.obb_all.yaml    # 배포용 obb (전체 데이터)
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from common import HERE, IMG_EXTS, Stage, latest_run, load_yaml, run_stamp


class Trainer(Stage):
    """
    파인튜닝과 ONNX export 를 담당하는 단계.

    실행마다 다른 yaml 을 쓰므로(`Trainer("train.single.yaml")`) 후보가 여럿이면
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
            data: ../datasets/processed_seg/kimm/data.yaml   경로 그대로 (한 소스)
            data: kimm, other                                 소스 **이름** 목록 → 그 자리에서 통합
        이름으로 적으면 datasets/processed_seg/<이름>/ (OBB 모델이면 processed_obb/<이름>/)
        규약으로 코드가 경로를 만듭니다.
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
                f"먼저 python preprocess.py 로 데이터셋을 만드세요 "
                f"(OBB 는 --task obb --out ../datasets/processed_obb)."
            )
        return path

    def merged_data_path(self, names: list) -> Path:
        """
        소스 이름 목록(`data: kimm, other`)을 묶은 data.yaml 을 쓰고 그 경로를 준다.

        모양은 preprocess.py 의 통합본과 같습니다 — split 마다 폴더 목록을 주면
        Ultralytics 가 알아서 합쳐 읽습니다. 소스 조합이 바뀌면 파일도 새로 써집니다.
        """
        # 전처리 산출물은 task 별로 따로 구워진다 — OBB 모델이면 processed_obb, 아니면 processed_seg
        model = str((self.cfg.get("train") or {}).get("model", ""))
        root = self.resolve("../datasets/processed_obb" if "obb" in model.lower()
                            else "../datasets/processed_seg").resolve()
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
        시작 가중치. 기본값 없음 — 이전 실행의 checkpoint 에서 이어 학습할 때 키를
        빠뜨리면 조용히 COCO 에서 다시 시작해 warm-start 가 사라지기 때문.

        경로 구분자가 있으면 체크포인트 파일로 보고 training/ 기준으로 해석하고 존재를
        확인합니다. 구분자가 없으면(`yolo26s-seg.pt`) Ultralytics 가 받아올 모델 이름이라
        그대로 넘깁니다.
        """
        model = str(self.cfg.get("train", {}).get("model") or "").strip()
        if not model:
            self.fail("train.model 이 없습니다 — 시작 가중치를 기본값으로 추측하지 "
                      "않습니다 (예: yolo26s-seg.pt 또는 runs/segment/<name>/weights/best.pt).")
        if "/" not in model and "\\" not in model:
            return model                      # Ultralytics 가 이름으로 해석/다운로드
        # 타임스탬프를 뺀 runs/<task>/<name>/weights/best.pt 로 적으면 그 name 의 최신 실행
        path = self.resolve_weights(model)
        if not path.exists():
            raise FileNotFoundError(
                f"시작 가중치가 없습니다: {path}\n"
                f"{self.config_path.name} 의 train.model 을 확인하세요 "
                f"(이어 학습할 실행을 먼저 끝내야 합니다)."
            )
        return str(path)

    def run_root(self) -> Path:
        """
        이 config 의 결과가 쌓이는 폴더: `<project>/<name>/` (그 아래 실행별 타임스탬프).

        project 는 **절대경로로 바꿔서** Ultralytics 에 넘깁니다. 상대경로로 주면
        Ultralytics 가 자기 설정의 runs_dir 아래에 다시 붙여(`runs/obb/runs/obb/…`)
        경로가 중첩됩니다.
        """
        train_cfg = self.cfg.get("train", {})
        return (self.resolve(train_cfg.get("project", "runs/obb"))
                / str(train_cfg.get("name", "yolo26s-obb-finetune")))

    def best_weights(self) -> Path:
        """best.pt 위치. 이번 실행에서 학습했다면 그 결과 폴더를, 아니면 가장 최근 실행."""
        if self._save_dir is not None:
            return self._save_dir / "weights" / "best.pt"
        root = self.run_root()
        latest = latest_run(root)
        # 실행 폴더가 하나도 없으면 없는 경로를 그대로 — 호출부가 안내 메시지를 냅니다.
        return (latest or root) / "weights" / "best.pt"

    # -- 단계 ------------------------------------------------------------

    @staticmethod
    def shorten_plot_names() -> None:
        """
        Ultralytics 의 train_batch*/val_batch* 그림에 찍히는 파일명을 짧게
        (`kimm__000039__cap_261014_174102` → `000039_174102`, 번호_세션시각).
        그대로는 이름이 칸보다 길어 옆 칸으로 넘친다 — 글꼴 크기와 40자 제한이 Ultralytics
        안에 고정돼 있어, 그림 함수에 넘기는 이름만 바꾼다 (학습·데이터에는 영향 없음).
        """
        import importlib
        import re

        pat = re.compile(r"(\d+)__cap_\d{6}_(\d{6})")

        def short(p) -> str:
            stem = Path(str(p)).stem
            m = pat.search(stem)
            return f"{m.group(1)}_{m.group(2)}" if m else stem[-16:]

        for name in ("ultralytics.models.yolo.detect.train", "ultralytics.models.yolo.detect.val",
                     "ultralytics.models.yolo.obb.val"):
            mod = importlib.import_module(name)
            orig = mod.plot_images
            if getattr(orig, "_short_names", False):
                continue

            def wrapped(*args, _orig=orig, **kw):
                if kw.get("paths") is not None:
                    kw["paths"] = [short(p) for p in kw["paths"]]
                return _orig(*args, **kw)

            wrapped._short_names = True
            mod.plot_images = wrapped

    @staticmethod
    def apply_photometric(cfg: dict) -> None:
        """
        config 의 `blur` · `brightness` → Ultralytics `augmentations` (augment.py).
        Ultralytics 에는 블러 키가 없고(기본 Blur 는 p=0.01), hsv_v 는 최대 0~2배라 따로 둔다.
        """
        import augment

        augment.albumentations_list(cfg)

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
        self.apply_photometric(cfg)
        self.shorten_plot_names()

        from ultralytics import YOLO

        results = YOLO(weights).train(**cfg)
        save_dir = Path(results.save_dir)
        batches = sorted(save_dir.glob("train_batch*.jpg"))
        self.log(f"증강 미리보기 {len(batches)}장 → {save_dir}")
        self.log("train_batch*.jpg = 학습이 실제로 먹는 배치(mosaic·hsv·flip·scale 반영).")
        return save_dir

    def train_kwargs(self) -> dict:
        """
        Ultralytics train() 에 넘길 인자. config 의 train 블록에서 model 만 빼고,
        데이터셋 경로와 결과 폴더를 확정한 것입니다 (무거운 import 없이 검증 가능).
        """
        cfg = dict(self.cfg.get("train", {}))
        cfg.pop("model", None)
        cfg["data"] = str(self.data_path())
        # 결과는 <project>/<name>/<YYMMDD_HHMMSS>/ 로 — 실행마다 폴더가 하나씩 쌓이므로
        # 이전 결과를 덮어쓰지도(exist_ok), stage12/stage13 으로 늘어나지도 않습니다.
        cfg["project"] = str(self.run_root())
        cfg["name"] = run_stamp()
        return cfg

    @staticmethod
    def tensorboard_layout(trainer) -> None:
        """
        TensorBoard CUSTOM SCALARS 탭 — Ultralytics 는 train/x 와 val/x 를 따로 그려서 과적합을
        보려면 겹친 그래프가 필요하다. 태그는 정규식 (Ultralytics 가 남기는 이름 그대로).
        """
        import re

        from ultralytics.utils.callbacks import tensorboard as tb

        writer = getattr(tb, "WRITER", None)
        if writer is None:
            return

        def m(name: str) -> str:
            return rf"^metrics/{re.escape(name)}$"

        def both(loss: str) -> list:
            return ["Multiline", [rf"^train/{loss}$", rf"^val/{loss}$"]]

        writer.add_custom_scalars({
            "핵심": {
                "seg_loss — train vs val": both("seg_loss"),
                "mAP50-95 — Mask vs Box (best.pt·early stop 기준은 둘의 합)":
                    ["Multiline", [m("mAP50-95(M)"), m("mAP50-95(B)")]],
                "Mask — P · R · mAP50": ["Multiline", [m("precision(M)"), m("recall(M)"), m("mAP50(M)")]],
            },
            # l1_loss(YOLO26) / dfl_loss(그 이전) 중 남는 쪽만 그려진다. sem_loss 는 val 에서 항상 0
            "손실 train vs val": {k: both(k) for k in ("box_loss", "cls_loss", "l1_loss", "dfl_loss",
                                                     "sem_loss")},
            "학습률": {"lr (pg 짝수 = 헤드 ×3)": ["Multiline", [r"^lr/pg\d+$"]]},
        })

    def record_run(self, trainer, cfg: dict) -> None:
        """
        학습 시작 시 실행 폴더에 run_info.yaml (커밋 · 커밋 안 된 수정 · config · 데이터 장수),
        수정이 있으면 code.patch 도 (provenance.py). Ultralytics 가 실행 폴더를 정한 뒤라
        콜백에서 한다 — 미리 만들면 Ultralytics 가 이름 뒤에 2 를 붙여 새 폴더를 만든다.
        """
        import provenance

        try:
            info = provenance.write_run_info(Path(trainer.save_dir), self.config_path, self.cfg,
                                             Path(cfg["data"]))
        except Exception as e:   # 기록 실패로 학습을 죽이지 않는다
            self.log(f"run_info 기록 실패 (학습은 계속): {e}")
            return
        git = info["git"]
        self.log(f"코드: {(git['commit'] or '?')[:7]} ({git['branch']})"
                 + (f" + 커밋 안 된 수정 {len(git['uncommitted'])}개 → code.patch" if git["dirty"] else ""))

    def train(self) -> Path:
        """파인튜닝. best.pt 경로를 돌려준다."""
        weights = self.start_weights()      # 설정 검증을 무거운 import 앞에 둔다
        cfg = self.train_kwargs()
        self.log(f"시작 가중치: {weights}")
        self.log(f"데이터셋: {cfg['data']}")
        self.log(f"결과 폴더: {Path(cfg['project']) / cfg['name']}")
        self.apply_photometric(cfg)
        self.shorten_plot_names()

        from ultralytics import YOLO, settings

        # Ultralytics 전역 설정에 기대지 않게 여기서 고정:
        # · TensorBoard 로그를 결과 폴더에 같이 남김 (보기: tensorboard --logdir runs)
        # · 사전학습/AMP 점검용 가중치 다운로드와 기본 runs 를 training/ 아래로 — 안 그러면
        #   설정 파일이 처음 만들어진 곳(저장소 루트 등)에 weights/·runs/ 가 생긴다
        want = {"tensorboard": True, "weights_dir": str(HERE / "weights"), "runs_dir": str(HERE / "runs")}
        if any(settings.get(k) != v for k, v in want.items()):
            settings.update(want)

        model = YOLO(weights)
        model.add_callback("on_pretrain_routine_start", lambda tr: self.record_run(tr, cfg))
        model.add_callback("on_train_start", self.tensorboard_layout)
        if (self.cfg.get("overlay") or {}).get("enabled"):
            # 매 val 마다 원본 4장 + 증강 12장 추론 격자 → <실행 폴더>/overlays/, TensorBoard
            import epoch_overlay
            t = self.cfg.get("train", {})
            model.add_callback("on_fit_epoch_end", epoch_overlay.callback(
                blur=t.get("blur"), brightness=t.get("brightness")))
        # cfg 의 나머지 키는 모두 Ultralytics train() 인자와 1:1 대응 (증강 포함)
        results = model.train(**cfg)

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
                f"학습을 먼저 끝내거나 train.*.yaml 의 train.project/name 을 확인하세요.\n"
                f"(export 만 돌리면 {self.run_root()} 의 가장 최근 실행에서 찾습니다.)"
            )
        self.log(f"가중치: {weights}")

        exported = Path(YOLO(str(weights)).export(
            format=ex.get("format", "onnx"),
            imgsz=self.cfg.get("train", {}).get("imgsz", 640),  # 학습 설정과 항상 일치
            opset=ex.get("opset", 12),
            simplify=ex.get("simplify", True),
            half=ex.get("half", False),
            dynamic=ex.get("dynamic", False),
            nms=ex.get("nms", False),   # False = YOLO26 end-to-end 헤드 (앱 YoloDotNet 이 읽는 형식)
        ))
        self.log(f"생성: {exported}")

        # 출처(커밋 · dirty · 실행 폴더 · config)를 .onnx 메타데이터에 — 앱에 들어간 파일만
        # 봐도 어떤 학습에서 나왔는지 알 수 있게
        import provenance

        info = provenance.read(weights)
        provenance.stamp_onnx(exported, info)
        commit = ((info.get("git") or {}).get("commit") or "unknown")[:7]
        self.log(f"출처 기록: {info.get('run')} @ {commit}")

        if deploy_to:
            # 앱은 고정 이름(deploy_to)을 읽는다 (config.xml <Model>) — 그 이름을 새 모델로
            # 덮어쓰고, 되돌릴 수 있게 타임스탬프 사본을 history/ 에 남긴다 (앱 빌드가
            # history/ 는 복사하지 않음). 처음 덮어쓰는 원래 파일(배포 이력이 없는 것)은
            # history/<이름>.orig 로 한 번 백업.
            current = self.resolve(deploy_to)
            hist_dir = current.parent / "history"
            hist_dir.mkdir(parents=True, exist_ok=True)
            history = list(hist_dir.glob(f"{current.stem}.2*{current.suffix}"))
            if current.exists() and not history:
                orig = hist_dir / f"{current.stem}.orig{current.suffix}"
                shutil.copyfile(current, orig)
                self.log(f"원래 모델 백업: {orig}")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = hist_dir / f"{current.stem}.{stamp}{current.suffix}"
            shutil.copyfile(exported, dest)      # 이력 (runs/ 의 원본도 그대로 유지)
            shutil.copyfile(exported, current)   # 앱이 읽는 현재 모델
            self.log(f"앱에 배치 완료: {current}  (이력 {dest.name})")
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
