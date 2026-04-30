"""
Pytest config cho verify suite của document_processing.

Cho phép chỉ định thư mục JSON cần verify qua:
  - env var: JSON_DIR=/path/to/processed_json_v4 pytest document_processing/tests
  - CLI flag: pytest document_processing/tests --json-dir=/path/to/processed_json_v4

Default: <repo_root>/3GPP_JSON_DOC/processed_json_v4 (output mặc định của pipeline mới).
"""
import json
import os
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_DIR = REPO_ROOT / "3GPP_JSON_DOC" / "processed_json_v4"


def pytest_addoption(parser):
    # Thêm option --json-dir để override thư mục JSON
    parser.addoption(
        "--json-dir",
        action="store",
        default=None,
        help="Thư mục chứa file JSON đã chunk (mặc định: 3GPP_JSON_DOC/processed_json_v4)",
    )


@pytest.fixture(scope="session")
def json_dir(request) -> Path:
    # Resolve theo thứ tự ưu tiên: --json-dir > $JSON_DIR > default
    cli = request.config.getoption("--json-dir")
    env = os.environ.get("JSON_DIR")
    target = Path(cli or env or DEFAULT_JSON_DIR)
    if not target.is_dir():
        pytest.skip(f"JSON dir không tồn tại: {target}")
    return target


@pytest.fixture(scope="session")
def json_files(json_dir: Path) -> List[Path]:
    # Liệt kê toàn bộ JSON output để các test parametrize/loop
    files = sorted(json_dir.glob("*.json"))
    if not files:
        pytest.skip(f"Không có file JSON nào trong {json_dir}")
    return files


@pytest.fixture(scope="session")
def all_docs(json_files: List[Path]) -> List[dict]:
    # Load 1 lần dùng chung — tránh đọc lại trong từng test
    docs = []
    for p in json_files:
        with p.open(encoding="utf-8") as f:
            docs.append({"_path": p, **json.load(f)})
    return docs
