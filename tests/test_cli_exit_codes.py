"""Exit-code contract coverage for the gpumemprof dispatcher."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import stormlog.cli as gpumemprof_cli
from stormlog.exit_codes import ExitCode


def _run_main_with(
    monkeypatch: pytest.MonkeyPatch, handler: object, *argv: str
) -> int | str | None:
    monkeypatch.setattr(gpumemprof_cli.sys, "argv", ["gpumemprof", *argv])
    monkeypatch.setattr(gpumemprof_cli, "cmd_info", handler)
    with pytest.raises(SystemExit) as excinfo:
        gpumemprof_cli.main()
    return excinfo.value.code


def test_main_interrupt_exits_interrupted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(_args: argparse.Namespace) -> None:
        raise KeyboardInterrupt

    assert _run_main_with(monkeypatch, interrupted, "info") == ExitCode.INTERRUPTED
    assert "cancelled" in capsys.readouterr().out


def test_main_unexpected_exception_exits_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_args: argparse.Namespace) -> None:
        raise RuntimeError("collector exploded")

    assert _run_main_with(monkeypatch, broken, "info") == ExitCode.ERROR
    assert "Error: collector exploded" in capsys.readouterr().out


def test_main_usage_error_exits_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gpumemprof_cli.sys, "argv", ["gpumemprof", "diagnose", "--duration", "abc"]
    )
    with pytest.raises(SystemExit) as excinfo:
        gpumemprof_cli.main()
    assert excinfo.value.code == ExitCode.USAGE


@pytest.mark.parametrize(
    ("resolver", "config_factory", "ensure_name"),
    [
        (
            "_resolve_wandb_config_or_exit",
            "wandb_config_from_namespace",
            "ensure_wandb_available",
        ),
        (
            "_resolve_mlflow_config_or_exit",
            "mlflow_config_from_namespace",
            "ensure_mlflow_available",
        ),
    ],
)
def test_missing_integration_extra_exits_usage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    resolver: str,
    config_factory: str,
    ensure_name: str,
) -> None:
    monkeypatch.setattr(
        gpumemprof_cli,
        config_factory,
        lambda _args: SimpleNamespace(enabled=True),
    )

    def missing_extra(_config: object) -> None:
        raise ImportError("install stormlog[extra]")

    monkeypatch.setattr(gpumemprof_cli, ensure_name, missing_extra)

    with pytest.raises(SystemExit) as excinfo:
        getattr(gpumemprof_cli, resolver)(argparse.Namespace())

    assert excinfo.value.code == ExitCode.USAGE
    assert "install stormlog[extra]" in capsys.readouterr().err
