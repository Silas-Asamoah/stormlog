from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("textual")

from textual.widgets import Button

from stormlog.tui import app as appmod


def test_build_system_markdown_uses_system_info_fallback_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _raise_system_info() -> dict[str, object]:
        raise RuntimeError("system info unavailable")

    def _fake_build_system_markdown(
        *,
        system_info: dict[str, object],
        gpu_info: dict[str, object],
        tf_system_info: dict[str, object],
        tf_gpu_info: dict[str, object],
    ) -> str:
        captured["system_info"] = system_info
        captured["gpu_info"] = gpu_info
        captured["tf_system_info"] = tf_system_info
        captured["tf_gpu_info"] = tf_gpu_info
        return "ok"

    monkeypatch.setattr(appmod, "get_system_info", _raise_system_info)
    monkeypatch.setattr(appmod, "_safe_get_gpu_info", lambda: {})
    monkeypatch.setattr(appmod, "_safe_get_tf_system_info", lambda: {})
    monkeypatch.setattr(appmod, "_safe_get_tf_gpu_info", lambda: {})
    monkeypatch.setattr(
        appmod.tui_builders, "build_system_markdown", _fake_build_system_markdown
    )

    assert appmod._build_system_markdown() == "ok"
    assert captured["system_info"] == {}


@pytest.mark.parametrize(
    ("button_id", "method_name", "args", "kwargs", "is_async"),
    [
        ("btn-refresh-overview", "action_refresh_overview", (), {}, False),
        ("btn-log-system", "run_cli_command", ("gpumemprof info",), {}, True),
        (
            "btn-log-pytorch",
            "run_cli_command",
            ("gpumemprof monitor --duration 30 --interval 0.5",),
            {},
            True,
        ),
        (
            "btn-log-tensorflow",
            "run_cli_command",
            ("tfmemprof monitor --duration 30 --interval 0.5",),
            {},
            True,
        ),
        (
            "btn-log-diagnose",
            "run_cli_command",
            ("gpumemprof diagnose --duration 0 --output artifacts/tui_diagnose",),
            {},
            True,
        ),
        ("btn-run-pytorch", "run_pytorch_sample", (), {}, True),
        ("btn-run-tf", "run_tensorflow_sample", (), {}, True),
        (
            "btn-run-oom-scenario",
            "run_cli_command",
            (
                "python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated",
            ),
            {},
            True,
        ),
        (
            "btn-run-cap-matrix",
            "run_cli_command",
            (
                "python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated --skip-tui",
            ),
            {},
            True,
        ),
        ("btn-cli-run", "run_cli_command", ("custom command",), {}, True),
        ("btn-cli-cancel", "cancel_cli_command", (), {}, True),
        ("btn-start-tracking", "start_live_tracking", (), {}, True),
        ("btn-stop-tracking", "stop_live_tracking", (), {}, False),
        ("btn-toggle-watchdog", "toggle_auto_cleanup", (), {}, False),
        ("btn-force-cleanup", "force_cleanup", (), {}, False),
        (
            "btn-force-cleanup-aggressive",
            "force_cleanup",
            (),
            {"aggressive": True},
            False,
        ),
        ("btn-export-csv", "export_tracker_events", ("csv",), {}, True),
        ("btn-export-json", "export_tracker_events", ("json",), {}, True),
        ("btn-apply-thresholds", "apply_thresholds", (), {}, False),
        ("btn-clear-monitor-log", "clear_monitor_log", (), {}, False),
        ("btn-refresh-visual", "refresh_visualizations", (), {}, True),
        ("btn-visual-png", "generate_visual_plot", ("png",), {}, True),
        ("btn-visual-html", "generate_visual_plot", ("html",), {}, True),
        ("btn-diag-load-live", "load_diagnostics_live", (), {}, True),
        ("btn-diag-load-artifacts", "load_diagnostics_artifacts", (), {}, True),
        ("btn-diag-refresh", "refresh_diagnostics", (), {}, True),
        ("btn-diag-apply-session", "apply_diagnostics_session_selection", (), {}, True),
        ("btn-diag-apply-filter", "apply_diagnostics_rank_filter", (), {}, False),
        ("btn-diag-reset-filter", "reset_diagnostics_rank_filter", (), {}, False),
        ("btn-refresh-pt-profiles", "refresh_pytorch_profiles", (), {}, True),
        ("btn-clear-pt-profiles", "clear_pytorch_profiles", (), {}, True),
        ("btn-refresh-tf-profiles", "refresh_tensorflow_profiles", (), {}, True),
        ("btn-clear-tf-profiles", "clear_tensorflow_profiles", (), {}, True),
    ],
)
def test_button_dispatch_preserves_action_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
    button_id: str,
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    is_async: bool,
) -> None:
    app = appmod.GPUMemoryProfilerTUI()
    monkeypatch.setattr(
        app, "cli_command_input", Mock(value="custom command"), raising=False
    )
    handler = AsyncMock() if is_async else Mock()
    monkeypatch.setattr(app, method_name, handler)

    asyncio.run(app.on_button_pressed(Button.Pressed(Button(id=button_id))))

    handler.assert_called_once_with(*args, **kwargs)
    if is_async:
        handler.assert_awaited_once_with(*args, **kwargs)


@pytest.mark.parametrize("button_id", [None, "btn-unrecognized"])
def test_button_dispatch_ignores_unknown_buttons(button_id: str | None) -> None:
    app = appmod.GPUMemoryProfilerTUI()

    asyncio.run(app.on_button_pressed(Button.Pressed(Button(id=button_id))))
