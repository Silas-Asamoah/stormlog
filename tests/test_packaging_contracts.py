import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _optional_dependencies() -> dict[str, list[str]]:
    content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section_match = re.search(
        r"(?ms)^\[project\.optional-dependencies\]\n(.*?)(?=^\[)",
        content,
    )
    assert section_match is not None, "Missing [project.optional-dependencies] section"

    section = section_match.group(1)
    extras: dict[str, list[str]] = {}
    for match in re.finditer(r"(?ms)^([A-Za-z0-9_-]+)\s*=\s*\[(.*?)^\]", section):
        name = match.group(1)
        values = re.findall(r'"([^"]+)"', match.group(2))
        extras[name] = values
    return extras


def test_all_extra_covers_every_runtime_extra() -> None:
    extras = _optional_dependencies()
    expected = (
        set(extras["viz"])
        | set(extras["torch"])
        | set(extras["tf"])
        | set(extras["tui"])
    )

    assert expected.issubset(set(extras["all"])), (
        "The all extra must cover every user-facing runtime extra "
        "(viz, torch, tf, tui)."
    )


def test_ci_uses_built_wheel_for_cli_smoke() -> None:
    content = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = content.index("artifact-cli-smoke:")
    end = content.index("examples-smoke:", start)
    job_block = content[start:end]

    assert "actions/download-artifact@v4" in job_block
    assert "python3 -m venv .venv-wheel-smoke" in job_block
    assert 'pip install "$WHEEL_PATH"' in job_block
    assert "torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu" in job_block
    assert "gpumemprof info" in job_block
    assert "examples.cli.quickstart" not in job_block
    assert "pip install -e ." not in job_block
