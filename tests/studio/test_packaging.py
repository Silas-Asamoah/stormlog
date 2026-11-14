"""The base wheel ships Studio's server and static files (decision 9).

Builds a wheel from the source tree, installs it into an empty directory
and, in a fresh interpreter that sees only that directory, reads
``static/manifest.json`` through ``importlib.resources`` and every file it
lists, then loads ``StaticFiles`` from the installed package.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_INPUTS = ("pyproject.toml", "setup.py", "README.md", "LICENSE")

PROBE = """
import json, sys
from importlib import resources
import stormlog
assert stormlog.__file__.startswith(sys.argv[1]), stormlog.__file__
static = resources.files("stormlog.studio").joinpath("static")
manifest = json.loads(static.joinpath("manifest.json").read_text(encoding="utf-8"))
sizes = {}
for name in manifest["files"]:
    sizes[name] = len(static.joinpath(*name.split("/")).read_bytes())
from stormlog.studio.static_files import StaticFiles
from stormlog.studio import cli, httpd, security
served = sorted(StaticFiles().files)
print(json.dumps({"manifest": manifest, "sizes": sizes, "served": served}))
"""

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pip") is None, reason="needs pip to build a wheel"
)


def _copy_sources(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for name in BUILD_INPUTS:
        source = REPO_ROOT / name
        if source.exists():
            shutil.copy2(source, target / name)
    shutil.copytree(
        REPO_ROOT / "stormlog",
        target / "stormlog",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return target


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = _copy_sources(tmp_path_factory.mktemp("source"))
    return _build(source, tmp_path_factory.mktemp("dist"))


def _isolation() -> list[str]:
    """Build offline when the build backend is already installed.

    An isolated build downloads setuptools and setuptools_scm, which needs
    the network; an environment that has them builds without it.
    """
    present = all(
        importlib.util.find_spec(name) is not None
        for name in ("setuptools", "setuptools_scm")
    )
    return ["--no-build-isolation"] if present else []


def _build(source: Path, dist: Path) -> Path:
    env = {**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0"}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--quiet",
            *_isolation(),
            "--wheel-dir",
            str(dist),
            str(source),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=280,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    (built,) = dist.glob("stormlog-*.whl")
    return built


def test_the_wheel_lists_the_static_files(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("stormlog/studio/static/manifest.json"))
    for name in manifest["files"]:
        assert f"stormlog/studio/static/{name}" in names
    for module in ("__init__", "__main__", "cli", "httpd", "security", "static_files"):
        assert f"stormlog/studio/{module}.py" in names


def test_vendored_files_ship_too(tmp_path: Path) -> None:
    # The frontend vendors Preact and htm under static/vendor/ (decision
    # 14); the "studio/static/*" pattern also matches that directory.
    source = _copy_sources(tmp_path / "source")
    static = source / "stormlog" / "studio" / "static"
    (static / "vendor").mkdir()
    (static / "vendor" / "lib.js").write_text("export {};\n")
    built = _build(source, tmp_path / "dist")
    with zipfile.ZipFile(built) as archive:
        assert "stormlog/studio/static/vendor/lib.js" in archive.namelist()


def test_the_installed_wheel_serves_its_manifest(wheel: Path, tmp_path: Path) -> None:
    target = tmp_path / "site"
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--quiet",
            "--target",
            str(target),
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=280,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    env = {**os.environ, "PYTHONPATH": str(target)}
    probe = subprocess.run(
        [sys.executable, "-c", PROBE, str(target)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert probe.returncode == 0, probe.stderr[-4000:]
    report = json.loads(probe.stdout)
    assert "index.html" in report["manifest"]["files"]
    assert all(size > 0 for size in report["sizes"].values())
    assert report["served"] == sorted(report["manifest"]["files"])
