import asyncio
from typing import Any

import pytest

pytest.importorskip("textual")

from textual.widgets import RichLog, TabbedContent, TabPane

from gpumemprof.tui.app import GPUMemoryProfilerTUI

pytestmark = pytest.mark.tui_pilot


def _log_text(log: RichLog) -> str:
    return "\n".join(line.text for line in log.lines)


def _tab_labels(app: GPUMemoryProfilerTUI) -> list[str]:
    labels: list[str] = []

    for widget in app.query("*"):
        if widget.__class__.__name__ == "ContentTab":
            label = getattr(widget, "label", "")
            if hasattr(label, "plain"):
                label = label.plain
            labels.append(str(label))

    if labels:
        return labels

    for pane in app.query(TabPane):
        title = getattr(pane, "title", None)
        if title is None:
            title = getattr(pane, "_title", "")
        if hasattr(title, "plain"):
            title = title.plain  # type: ignore[union-attr, unused-ignore]
        labels.append(str(title))
    return labels


class _StubTrackerSession:
    def __init__(self) -> None:
        self.backend = "gpu"
        self.is_active = False
        self.start_calls = 0
        self.stop_calls = 0
        self.threshold_calls: list[tuple[float, float]] = []

    def start(self) -> None:
        self.start_calls += 1
        self.is_active = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_active = False

    def set_thresholds(self, warning: float, critical: float) -> None:
        self.threshold_calls.append((warning, critical))

    def get_thresholds(self) -> dict[str, float]:
        return {
            "memory_warning_percent": 80.0,
            "memory_critical_percent": 95.0,
        }

    def get_device_label(self) -> str:
        return "stub-gpu"

    def set_auto_cleanup(self, enabled: bool) -> None:
        _ = enabled

    def get_statistics(self) -> dict[str, Any]:
        return {}

    def get_cleanup_stats(self) -> dict[str, Any]:
        return {}

    def pull_events(self) -> list[Any]:
        return []

    def get_memory_timeline(self, interval: float = 1.0) -> dict[str, list[Any]]:
        _ = interval
        return {}


class _StubCLIRunner:
    def __init__(self) -> None:
        self.is_running = False
        self.commands: list[str] = []
        self.cancel_calls = 0

    async def run(self, command: str, callback: Any) -> int:
        self.is_running = True
        self.commands.append(command)
        await callback("stdout", "hello")
        await callback("stderr", "warn")
        self.is_running = False
        return 17

    async def cancel(self) -> bool:
        self.cancel_calls += 1
        self.is_running = False
        return True


def test_tab_rendering_and_key_bindings_r_f_g_t(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        app = GPUMemoryProfilerTUI()
        async with app.run_test(headless=True, size=(140, 44)) as pilot:
            await pilot.pause()

            assert _tab_labels(app) == [
                "Overview",
                "PyTorch",
                "TensorFlow",
                "Monitoring",
                "Visualizations",
                "CLI & Actions",
            ]

            messages: list[tuple[str, str]] = []
            refresh_calls = {"count": 0}

            def fake_log_message(title: str, content: str) -> None:
                messages.append((title, content))

            def fake_refresh_content() -> None:
                refresh_calls["count"] += 1

            monkeypatch.setattr(app, "log_message", fake_log_message)
            monkeypatch.setattr(
                app.overview_panel, "refresh_content", fake_refresh_content
            )

            await pilot.press("f")
            await pilot.pause()
            assert app.focused is app.command_log

            await pilot.press("r")
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            await pilot.press("t")
            await pilot.pause()

            assert refresh_calls["count"] == 1
            assert ("Overview", "System overview refreshed.") in messages
            assert (
                "gpumemprof info",
                "Run: gpumemprof info\nRun: gpumemprof monitor --duration 30",
            ) in messages
            assert (
                "tfmemprof info",
                "Run: tfmemprof info\nRun: tfmemprof monitor --duration 30",
            ) in messages

    asyncio.run(scenario())


def test_monitoring_buttons_start_stop_apply_thresholds_clear_log() -> None:
    async def scenario() -> None:
        app = GPUMemoryProfilerTUI()
        async with app.run_test(headless=True, size=(140, 44)) as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab-4"
            await pilot.pause()

            session = _StubTrackerSession()
            app.tracker_session = session  # type: ignore[assignment, unused-ignore]
            app._get_or_create_tracker_session = lambda: session  # type: ignore[assignment, method-assign, return-value, unused-ignore]

            await pilot.click("#btn-start-tracking")
            await pilot.pause()

            app.warning_input.value = "70"
            app.critical_input.value = "90"
            await pilot.click("#btn-apply-thresholds")
            await pilot.pause()

            app.monitor_log.write("transient line")
            await pilot.click("#btn-clear-monitor-log")
            await pilot.pause()

            await pilot.click("#btn-stop-tracking")
            await pilot.pause()

            monitor_text = _log_text(app.monitor_log)
            assert session.start_calls == 1
            assert session.stop_calls == 1
            assert (70.0, 90.0) in session.threshold_calls
            assert "Cleared monitoring log." in monitor_text
            assert "Live tracking stopped." in monitor_text

    asyncio.run(scenario())


def test_cli_runner_run_stream_output_and_cancel() -> None:
    async def scenario() -> None:
        app = GPUMemoryProfilerTUI()
        async with app.run_test(headless=True, size=(140, 44)) as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab-6"
            await pilot.pause()

            runner = _StubCLIRunner()
            app.cli_runner = runner  # type: ignore[assignment, unused-ignore]

            app.cli_command_input.value = "gpumemprof info"
            await pilot.click("#btn-cli-run")
            await pilot.pause()

            command_text = _log_text(app.command_log)
            assert runner.commands == ["gpumemprof info"]
            assert "gpumemprof info" in command_text
            assert "stdout hello" in command_text
            assert "stderr warn" in command_text
            assert "Command finished with exit code 17." in command_text

            runner.is_running = True
            await pilot.click("#btn-cli-cancel")
            await pilot.pause()

            assert runner.cancel_calls == 1
            assert "Command was cancelled." in _log_text(app.command_log)

    asyncio.run(scenario())


def test_cli_action_buttons_cover_diagnose_oom_and_matrix() -> None:
    async def scenario() -> None:
        app = GPUMemoryProfilerTUI()
        async with app.run_test(headless=True, size=(140, 44)) as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab-6"
            await pilot.pause()

            runner = _StubCLIRunner()
            app.cli_runner = runner  # type: ignore[assignment, unused-ignore]

            await pilot.click("#btn-log-diagnose")
            await pilot.pause()
            await pilot.click("#btn-run-oom-scenario")
            await pilot.pause()
            await pilot.click("#btn-run-cap-matrix")
            await pilot.pause()

            assert runner.commands == [
                "gpumemprof diagnose --duration 0 --output artifacts/tui_diagnose",
                "python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated",
                "python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated --skip-tui",
            ]

    asyncio.run(scenario())
