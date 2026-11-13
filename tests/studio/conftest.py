"""Fixtures every Studio test file can use (design §9, §10.3 C5).

``studio_artifacts`` builds each standard artifact once per session, on
first use; tests that change a file take a copy with ``fresh_artifact``.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.studio.artifacts import STANDARD_ARTIFACTS


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
