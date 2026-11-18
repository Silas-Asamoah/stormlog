"""Fixtures every Studio test file can use (design §9, §10.3 C5).

``studio_artifacts`` builds each standard artifact once per session, on
first use; tests that change a file take a copy with ``fresh_artifact``.
``large_studio_artifact`` is the 500,000-request file: it takes about 12 s
and 740 MiB to write, so only tests marked ``studio_large`` get it, and
those run only when selected with ``-m studio_large`` or with
``STORMLOG_STUDIO_LARGE=1`` in the environment.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.studio.artifacts import STANDARD_ARTIFACTS, large_artifact

LARGE_MARKER = "studio_large"
LARGE_ENV = "STORMLOG_STUDIO_LARGE"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{LARGE_MARKER}: needs the 500,000-request artifact; skipped unless "
        f"selected with -m {LARGE_MARKER} or {LARGE_ENV}=1",
    )


def large_opted_in(config: pytest.Config) -> bool:
    """Whether this run asked for the large artifact."""
    expression = config.getoption("markexpr") or ""
    named = LARGE_MARKER in expression and f"not {LARGE_MARKER}" not in expression
    return named or os.environ.get(LARGE_ENV) == "1"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if large_opted_in(config):
        return
    skip = pytest.mark.skip(
        reason=f"select with -m {LARGE_MARKER} or set {LARGE_ENV}=1"
    )
    for item in items:
        if item.get_closest_marker(LARGE_MARKER):
            item.add_marker(skip)


class StandardArtifacts:
    """The standard artifacts of one session, each built on first use.

    The files are shared by every test of the session: read them, never
    change them. ``copy`` gives a private one.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._built: dict[str, Path] = {}

    def __getitem__(self, name: str) -> Path:
        path = self._built.get(name)
        if path is None:
            build = STANDARD_ARTIFACTS[name]
            target = self.directory / name
            target.mkdir()
            path = self._built[name] = build(target)
        return path

    def copy(self, name: str, directory: Path) -> Path:
        """A private copy of ``name``'s artifact in ``directory``."""
        source = self[name]
        return Path(shutil.copy2(source, Path(directory) / source.name))


@pytest.fixture(scope="session")
def studio_artifacts(tmp_path_factory: pytest.TempPathFactory) -> StandardArtifacts:
    return StandardArtifacts(tmp_path_factory.mktemp("studio-artifacts"))


@pytest.fixture
def fresh_artifact(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> Callable[[str], Path]:
    """A function giving a test its own copy of a standard artifact."""

    def copy(name: str) -> Path:
        return studio_artifacts.copy(name, tmp_path)

    return copy


@pytest.fixture(scope="session")
def large_studio_artifact(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The 500,000-request, 15-gap artifact, written once per session."""
    if not large_opted_in(request.config):
        pytest.skip(f"select with -m {LARGE_MARKER} or set {LARGE_ENV}=1")
    return large_artifact(tmp_path_factory.mktemp("studio-large"))
