"""Tests for JAX CLI."""

import json
from typing import Any
from unittest import mock

import pytest

from stormlog.jax.cli import main
from tests.jax_test_helpers import fake_jax_runtime, jax_mark  # noqa: F401

pytestmark = pytest.mark.usefixtures("fake_jax_runtime")


@jax_mark
@mock.patch("sys.argv", ["jaxmemprof"])
def test_main_no_args_prints_help(capsys: Any) -> None:
    """Verify main() prints help when no command is provided."""
    assert main() == 0
    captured = capsys.readouterr()
    assert "Available commands" in captured.out


@jax_mark
@mock.patch("sys.argv", ["jaxmemprof", "info"])
def test_cmd_info(capsys: Any) -> None:
    """Verify cmd_info produces expected output."""
    assert main() == 0
    captured = capsys.readouterr()
    assert "JAX Stormlog - System Information" in captured.out
    assert "Platform:" in captured.out
    assert "Runtime Backend:" in captured.out


@jax_mark
@mock.patch("sys.argv", ["jaxmemprof", "info"])
@mock.patch("stormlog.jax.cli.cmd_info", side_effect=KeyboardInterrupt)
def test_main_interrupt_outside_capture_loop_exits_interrupted(
    _cmd_info: Any, capsys: Any
) -> None:
    """Ctrl+C outside monitor/track exits 130 with a message, not a traceback."""
    assert main() == 130
    assert "cancelled" in capsys.readouterr().err


@jax_mark
@mock.patch("sys.argv", ["jaxmemprof", "--verbose", "info"])
def test_cmd_info_verbose(capsys: Any) -> None:
    """Verify cmd_info verbose mode works without crashing."""
    assert main() == 0
    captured = capsys.readouterr()
    assert "JAX Stormlog - System Information" in captured.out


@jax_mark
@mock.patch("stormlog.jax.cli.cmd_monitor", return_value=0)
@mock.patch("sys.argv", ["jaxmemprof", "monitor", "--device", "cpu", "--duration", "0"])
def test_monitor_accepts_named_device_selector(mock_monitor: Any) -> None:
    assert main() == 0
    assert mock_monitor.call_args.args[0].device == "cpu"


@jax_mark
def test_cmd_monitor_args() -> None:
    """Verify monitor command arguments."""
    import argparse

    from stormlog.jax.cli import cmd_monitor

    args = argparse.Namespace(
        interval=1.0,
        duration=0.1,  # Short duration to exit quickly
        threshold=1000,
        device=0,
        output=None,
    )

    assert cmd_monitor(args) == 0


@jax_mark
def test_cmd_monitor_output(tmp_path: Any) -> None:
    """Verify monitor command writes output file."""
    import argparse

    from stormlog.jax.cli import cmd_monitor

    out_file = tmp_path / "monitor.json"
    args = argparse.Namespace(
        interval=0.1, duration=0.2, threshold=1000, device=0, output=str(out_file)
    )

    assert cmd_monitor(args) == 0
    assert out_file.exists()

    with open(out_file) as f:
        data = json.load(f)

    assert "peak_memory" in data
    assert "memory_usage" in data


@jax_mark
@mock.patch("stormlog.wandb_integration.wandb_config_from_namespace")
def test_cmd_track_args(mock_wandb_config: Any, tmp_path: Any) -> None:
    """Verify track command handles arguments."""
    import argparse

    from stormlog.jax.cli import cmd_track

    class MockConfig:
        enabled = False

    mock_wandb_config.return_value = MockConfig()

    out_file = tmp_path / "track.json"
    args = argparse.Namespace(
        interval=0.1,
        threshold=4000,
        device=0,
        profile=False,
        job_id="job123",
        rank=0,
        local_rank=0,
        world_size=1,
        output=str(out_file),
        telemetry_sink_dir=str(tmp_path / "telemetry"),
        telemetry_flush_seconds=2.0,
        telemetry_rollover_mb=64,
        telemetry_retention_files=8,
        telemetry_retention_total_mb=512,
    )

    # We mock time.sleep to exit the loop after 1 iteration by raising KeyboardInterrupt
    with mock.patch("time.sleep", side_effect=KeyboardInterrupt):
        assert cmd_track(args) == 0

    assert out_file.exists()

    with open(out_file) as f:
        data = json.load(f)

    assert "peak_memory" in data
    assert "events" in data


@jax_mark
@mock.patch("sys.argv", ["jaxmemprof", "diagnose", "--duration", "0"])
@mock.patch("stormlog.jax.cli._load_run_diagnose")
def test_cmd_diagnose(mock_loader: Any, tmp_path: Any) -> None:
    """Verify diagnose command creates valid output."""
    mock_run_diagnose = mock_loader.return_value
    mock_run_diagnose.return_value = (tmp_path / "dummy", 0)

    assert main() == 0
    mock_run_diagnose.assert_called_once()
