import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip benchmarks unless ``-m`` names them; they report, never assert."""
    if "benchmark" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="a benchmark: select it with -m benchmark")
    for item in items:
        if item.get_closest_marker("benchmark"):
            item.add_marker(skip)
