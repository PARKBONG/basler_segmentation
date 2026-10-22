"""pytest 공통 — 테스트들이 합성 데이터를 만들 임시 폴더(root)."""
from pathlib import Path

import pytest


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """테스트마다 새 임시 폴더 (pytest 가 정리)."""
    return tmp_path
