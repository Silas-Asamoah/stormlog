import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

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


def _requirement_names(requirements: list[str]) -> set[str]:
    return {Requirement(requirement).name.lower() for requirement in requirements}


def test_all_extra_covers_every_runtime_extra() -> None:
    extras = _optional_dependencies()
    expected_names = _requirement_names(
        extras["viz"]
        + extras["torch"]
        + extras["tf"]
        + extras["infer-tokenizers"]
        + extras["tui"]
        + extras["jax"]
        + extras["mlx"]
        + extras["wandb"]
    )

    assert expected_names.issubset(_requirement_names(extras["all"])), (
        "The all extra must cover every user-facing runtime extra "
        "(viz, torch, tf, infer-tokenizers, tui, jax, mlx, wandb)."
    )


def test_all_extra_uses_protobuf_compatible_tensorflow() -> None:
    extras = _optional_dependencies()
    requirements = {
        requirement.name.lower(): requirement
        for requirement in map(Requirement, extras["all"])
    }

    assert any(
        specifier.operator in {">", ">="}
        and Version(specifier.version) >= Version("2.21.0")
        for specifier in requirements["tensorflow"].specifier
    )
    assert any(
        specifier.operator in {">", ">="}
        and Version(specifier.version) >= Version("6.31.1")
        for specifier in requirements["protobuf"].specifier
    )


def test_mlx_extra_is_platform_gated_and_optional() -> None:
    extras = _optional_dependencies()
    for extra in ("mlx", "all"):
        requirement = next(
            r for r in map(Requirement, extras[extra]) if r.name == "mlx"
        )
        assert requirement.marker is not None
        for platform, machine, expected in (
            ("darwin", "arm64", True),
            ("darwin", "x86_64", False),
            ("linux", "arm64", False),
            ("linux", "x86_64", False),
        ):
            assert (
                requirement.marker.evaluate(
                    {"sys_platform": platform, "platform_machine": machine}
                )
                is expected
            )
        assert Version("0.32.3") in requirement.specifier
        assert Version("0.32.2") not in requirement.specifier
    content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # PR 1 does not ship a CLI entry point before the implementation exists.
    assert "mlxmemprof =" not in content
    base = re.search(r"(?ms)^dependencies = \[(.*?)^\]", content)
    assert base is not None
    assert not re.search(r'"mlx(?:[><=;"\s])', base.group(1))
