"""
공개(Roboflow Universe) 데이터셋을 SDK로 다운로드.

configs/download.yaml 의 sources 를 해당 path 폴더에
받아옵니다. preprocess.py 도 같은 파일의 같은 path 를 읽으므로 따로 맞출 게 없습니다
(preprocess 는 폴더가 없으면 "download 를 먼저 돌리라"고 알려줍니다).

Roboflow export 의 train/valid/test 분할은 받은 직후 kimm 과 같은
images/ + labels/ 한 덩어리로 합칩니다 — 분할은 preprocess.py 가
val_ratio/test_ratio 로 다시 하므로 원천(raw)에는 분할을 두지 않습니다.

- 공개 데이터셋도 다운로드에는 본인 무료 API 키가 필요.
- API 키는 보안상 yaml 이 아니라 환경변수 ROBOFLOW_API_KEY 에서 읽습니다.

사용법:
    pip install roboflow
    # Windows PowerShell:  $env:ROBOFLOW_API_KEY="xxxx"
    # bash:                export ROBOFLOW_API_KEY=xxxx
    python download.py
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml

from common import INDEX_RE, Stage, load_yaml, normalize_names, roboflow_api_key


class Downloader(Stage):
    """configs/download.yaml 의 공개 데이터셋을 내려받는 단계."""

    config_glob = "download.yaml"
    label = "download"

    @staticmethod
    def pick_version(project, want) -> int:
        """version 설정값을 실제 버전 번호로. latest/빈값이면 가장 높은 버전."""
        if want not in (None, "", "latest"):
            return int(want)

        # 버전 식별자는 SDK 버전에 따라 "ws/proj/3" 또는 3 으로 나옵니다. 뒤 숫자만 취함.
        nums = []
        for v in project.versions():
            m = re.search(r"(\d+)$", str(getattr(v, "version", "") or getattr(v, "id", "")))
            if m:
                nums.append(int(m.group(1)))
        if not nums:
            raise SystemExit("버전 목록을 읽지 못했습니다. version 에 숫자를 직접 적어주세요.")
        return max(nums)

    def report_classes(self, dest) -> None:
        """받은 data.yaml 의 클래스명을 찍어준다 (preprocess class_map 작성용)."""
        for name in ("data.yaml", "data.yml"):
            p = dest / name
            if p.exists():
                names = normalize_names(load_yaml(p).get("names"))
                self.log(f"  클래스: {list(names.values())}  → sources 의 class_map 에 사용")
                return
        self.log("  data.yaml 을 찾지 못했습니다. 폴더를 직접 확인하세요.")

    def index_files(self, dest: Path) -> None:
        """
        images/·labels/ 의 파일명 맨 앞에 000000__ 식 6자리 인덱스를 붙인다
        (이미지·라벨 짝 유지 — 앱 FrameRecorder 의 캡처 파일명과 같은 규약).

        이름순으로 매기므로 같은 내용이면 재실행해도 번호가 같습니다. 이미 인덱스가
        붙어 있으면 벗기고 다시 매겨 중복 접두어가 생기지 않습니다. preprocess 의
        split_group 이 이 접두어를 벗기고 그룹핑하므로 누수 방지와 충돌하지 않습니다.
        """
        img_out = dest / "images"
        lbl_out = dest / "labels"
        pairs = []
        for img in img_out.iterdir():
            bare = INDEX_RE.sub("", img.stem)
            pairs.append((bare, img))
        for i, (bare, img) in enumerate(sorted(pairs)):
            lbl = lbl_out / (img.stem + ".txt")
            stem = f"{i:06d}__{bare}"
            img.rename(img_out / (stem + img.suffix))
            if lbl.exists():
                lbl.rename(lbl_out / (stem + ".txt"))

    def merge_splits(self, dest: Path) -> None:
        """
        Roboflow 가 나눠 준 train/valid/test 를 kimm 처럼 images/ + labels/ 로 합친다.

        분할은 preprocess.py 가 val_ratio/test_ratio 로 다시 하므로 원천(raw)에는
        분할을 두지 않습니다. data.yaml 은 kimm 과 같은 규약(train: images)으로
        다시 써서 preprocess 가 그대로 읽게 합니다.
        """
        img_out = dest / "images"
        lbl_out = dest / "labels"
        img_out.mkdir(exist_ok=True)
        lbl_out.mkdir(exist_ok=True)

        moved = 0
        for split in ("train", "valid", "val", "test"):
            split_dir = dest / split
            img_dir = split_dir / "images"
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                stem = img.stem
                if (img_out / img.name).exists():   # split 간 이름 충돌 시 접두어
                    stem = f"{split}__{stem}"
                lbl = split_dir / "labels" / (img.stem + ".txt")
                img.rename(img_out / (stem + img.suffix))
                if lbl.exists():
                    lbl.rename(lbl_out / (stem + ".txt"))
                moved += 1
            shutil.rmtree(split_dir)

        self.index_files(dest)

        # data.yaml 재작성: names 등은 유지, split 경로만 한 덩어리로.
        dy_path = next((dest / n for n in ("data.yaml", "data.yml")
                        if (dest / n).exists()), dest / "data.yaml")
        dy = load_yaml(dy_path) if dy_path.exists() else {}
        for key in ("val", "valid", "test", "path"):
            dy.pop(key, None)
        dy["train"] = "images"
        with dy_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(dy, f, allow_unicode=True, sort_keys=False)

        self.log(f"  분할 합침: {moved}장 → images/ (train/val/test 분할은 preprocess 담당)")

    def download_one(self, rf, source: dict) -> bool:
        """소스 하나를 내려받는다. 설정이 비어 있으면 건너뛰고 False."""
        rb = source.get("roboflow", {})
        if not rb.get("workspace") or not rb.get("project"):
            self.log(f"{source['name']}: workspace/project 가 비어있어 건너뜀")
            return False

        dest = self.resolve(source["path"])
        if dest.exists():   # 이전 실행의 합쳐진 images/ 와 섞이지 않게 통째로 초기화
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        project = rf.workspace(rb["workspace"]).project(rb["project"])
        version = self.pick_version(project, rb.get("version"))
        self.log(f"{source['name']} v{version} → {dest}")

        project.version(version).download(
            rb.get("format", "yolov11"), location=str(dest), overwrite=True
        )
        self.merge_splits(dest)
        self.report_classes(dest)
        return True

    def run(self) -> None:
        sources = [s for s in self.cfg.get("sources", []) if s.get("roboflow")]
        if not sources:
            self.log("roboflow: 블록이 있는 소스가 없습니다. 받을 게 없어요.")
            return

        from roboflow import Roboflow  # 로컬에 없으면: pip install roboflow

        rf = Roboflow(api_key=roboflow_api_key())
        done = sum(self.download_one(rf, s) for s in sources)

        self.log(f"완료: {done}/{len(sources)} 소스. 다음: python preprocess.py --config <yaml>")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Roboflow 공개 데이터셋 다운로드 (환경변수 ROBOFLOW_API_KEY 필요)",
        epilog="configs/ 의 후보: " + Downloader.config_candidates()
               + " — 하나뿐이라 --config 생략 시 자동 선택",
    )
    ap.add_argument("--config", metavar="YAML",
                    help="설정 yaml (생략 시 후보가 하나면 자동). configs/ 안 파일명 또는 경로")
    Downloader(ap.parse_args().config).run()


if __name__ == "__main__":
    main()
