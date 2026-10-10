import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip benchmarks unless ``-m`` names them; they report, never assert."""
    expression = config.getoption("markexpr") or ""
    if "mlx_hardware" not in expression or "not mlx_hardware" in expression:
        hardware_skip = pytest.mark.skip(
            reason="select explicitly with -m mlx_hardware"
        )
        for item in items:
            if item.get_closest_marker("mlx_hardware"):
                item.add_marker(hardware_skip)
    if "benchmark" in expression:
        return
    skip = pytest.mark.skip(reason="a benchmark: select it with -m benchmark")
    for item in items:
        if item.get_closest_marker("benchmark"):
            item.add_marker(skip)
