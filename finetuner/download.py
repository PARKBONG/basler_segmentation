"""
공개(Roboflow Universe) 데이터셋을 SDK로 다운로드.

download_config.yaml 의 각 소스를 해당 path 폴더에 받아옵니다.
받아온 폴더는 preprocess_config.yaml 의 sources 에서 같은 path 로 참조합니다
(preprocess 는 폴더가 없으면 "download 를 먼저 돌리라"고 알려줍니다).

- 공개 데이터셋도 다운로드에는 본인 무료 API 키가 필요.
- API 키는 보안상 yaml 이 아니라 환경변수 ROBOFLOW_API_KEY 에서 읽습니다.

사용법:
    pip install roboflow
    # Windows PowerShell:  $env:ROBOFLOW_API_KEY="xxxx"
    # bash:                export ROBOFLOW_API_KEY=xxxx
    python download.py
"""
from __future__ import annotations

import re

from common import Stage, load_yaml, normalize_names, roboflow_api_key


class Downloader(Stage):
    """download_config.yaml 의 공개 데이터셋을 내려받는 단계."""

    config_name = "download_config.yaml"
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
                self.log(f"  클래스: {list(names.values())}  → preprocess_config class_map 에 사용")
                return
        self.log("  data.yaml 을 찾지 못했습니다. 폴더를 직접 확인하세요.")

    def download_one(self, rf, source: dict) -> bool:
        """소스 하나를 내려받는다. 설정이 비어 있으면 건너뛰고 False."""
        rb = source.get("roboflow", {})
        if not rb.get("workspace") or not rb.get("project"):
            self.log(f"{source['name']}: workspace/project 가 비어있어 건너뜀")
            return False

        dest = self.resolve(source["path"])
        dest.mkdir(parents=True, exist_ok=True)

        project = rf.workspace(rb["workspace"]).project(rb["project"])
        version = self.pick_version(project, rb.get("version"))
        self.log(f"{source['name']} v{version} → {dest}")

        project.version(version).download(
            rb.get("format", "yolov11"), location=str(dest), overwrite=True
        )
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

        self.log(f"완료: {done}/{len(sources)} 소스. 다음: python preprocess.py")


if __name__ == "__main__":
    Downloader().run()
