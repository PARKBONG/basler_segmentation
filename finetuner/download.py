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

import os

from common import Stage


class Downloader(Stage):
    """download_config.yaml 의 공개 데이터셋을 내려받는 단계."""

    config_name = "download_config.yaml"
    label = "download"

    @staticmethod
    def api_key() -> str:
        key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        if not key:
            raise SystemExit(
                "환경변수 ROBOFLOW_API_KEY 가 필요합니다.\n"
                '  PowerShell:  $env:ROBOFLOW_API_KEY="xxxx"\n'
                "  bash:        export ROBOFLOW_API_KEY=xxxx"
            )
        return key

    def download_one(self, rf, source: dict) -> bool:
        """소스 하나를 내려받는다. 설정이 비어 있으면 건너뛰고 False."""
        rb = source.get("roboflow", {})
        if not rb.get("workspace") or not rb.get("project"):
            self.log(f"{source['name']}: workspace/project 가 비어있어 건너뜀")
            return False

        dest = self.resolve(source["path"])
        dest.mkdir(parents=True, exist_ok=True)
        self.log(f"{source['name']} → {dest}")

        project = rf.workspace(rb["workspace"]).project(rb["project"])
        project.version(int(rb.get("version", 1))).download(
            rb.get("format", "yolov11"), location=str(dest), overwrite=True
        )
        return True

    def run(self) -> None:
        sources = [s for s in self.cfg.get("sources", []) if s.get("roboflow")]
        if not sources:
            self.log("roboflow: 블록이 있는 소스가 없습니다. 받을 게 없어요.")
            return

        from roboflow import Roboflow  # 로컬에 없으면: pip install roboflow

        rf = Roboflow(api_key=self.api_key())
        done = sum(self.download_one(rf, s) for s in sources)

        self.log(f"완료: {done}/{len(sources)} 소스. 다음: python preprocess.py")


if __name__ == "__main__":
    Downloader().run()
