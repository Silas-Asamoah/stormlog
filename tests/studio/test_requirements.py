"""Studio's schema tests need jsonschema 4.18+, for referencing (F-6)."""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JSONSCHEMA = re.compile(r"jsonschema\s*>=\s*([0-9][0-9.]*)")
# Every place that pins jsonschema. Each must hold a pin; how many it
# holds is free, so a new extra or a reordered file does not fail this.
PINS = (
    "pyproject.toml",
    "requirements-ci-base.txt",
    "requirements-test.txt",
    "requirements-dev.txt",
)


def _release(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3])


@pytest.mark.parametrize("name", PINS)
def test_every_jsonschema_pin_allows_only_4_18_or_later(name: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    floors = JSONSCHEMA.findall(text)
    assert floors, f"{name} pins no jsonschema"
    for floor in floors:
        assert _release(floor) >= (4, 18), f"{name}: jsonschema>={floor}"


def test_the_installed_jsonschema_ships_referencing() -> None:
    assert _release(version("jsonschema")) >= (4, 18)
    import referencing

    assert referencing.Registry is not None
