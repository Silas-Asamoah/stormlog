"""Interactive Textual TUI for Stormlog."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

# Suppress TensorFlow oneDNN warnings
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

logger = logging.getLogger(__name__)

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    RichLog,
    Rule,
    TabbedContent,
    TabPane,
)

from gpumemprof.utils import format_bytes, get_gpu_info, get_system_info
from tfmemprof.utils import get_gpu_info as get_tf_gpu_info
from tfmemprof.utils import get_system_info as get_tf_system_info

from . import builders as tui_builders
from .commands import CLICommandRunner
from .monitor import TrackerEventView, TrackerSession, TrackerUnavailableError
from .profiles import (
    clear_pytorch_profiles,
    clear_tensorflow_profiles,
    fetch_pytorch_profiles,
    fetch_tensorflow_profiles,
)
from .styles import TUI_APP_CSS
from .widgets import (
    AlertHistoryTable,
    AnomalySummaryTable,
    AsciiWelcome,
    DistributedRankTable,
    DistributedTimelineCanvas,
    GPUStatsTable,
    KeyValueTable,
    MarkdownPanel,
    ProfileResultsTable,
    TimelineCanvas,
)
from .workloads import (
    format_cpu_summary,
    format_pytorch_summary,
    format_tensorflow_results,
    run_cpu_sample_workload,
    run_pytorch_sample_workload,
    run_tensorflow_sample_workload,
)

try:
    import torch as _torch

    torch: Any = _torch
except ImportError as e:
    raise ImportError(
        "torch is required for the TUI application. Install it with: pip install torch"
    ) from e

try:
    import tensorflow as _tf

    # Suppress TensorFlow INFO and WARNING messages
    _tf.get_logger().setLevel("ERROR")
    # Also suppress oneDNN warnings via environment
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    tf: Optional[Any] = _tf
except ImportError:
    tf = None

try:
    from pyfiglet import Figlet as _Figlet

    Figlet: Optional[Any] = _Figlet
except ImportError:
    Figlet = None

try:
    from gpumemprof import GPUMemoryProfiler as _GPUMemoryProfiler

    GPUMemoryProfiler: Optional[Any] = _GPUMemoryProfiler
except ImportError as e:
    raise ImportError(
        "GPUMemoryProfiler is required for the TUI application. "
        "Ensure gpumemprof is properly installed."
    ) from e

try:
    from gpumemprof.cpu_profiler import CPUMemoryProfiler as _CPUMemoryProfiler

    CPUMemoryProfiler: Optional[Any] = _CPUMemoryProfiler
except ImportError as e:
    raise ImportError(
        "CPUMemoryProfiler is required for the TUI application. "
        "Ensure gpumemprof is properly installed."
    ) from e

try:
    from tfmemprof.profiler import TFMemoryProfiler as _TFMemoryProfiler

    TFMemoryProfiler: Optional[Any] = _TFMemoryProfiler
except ImportError:
    TFMemoryProfiler = None


WELCOME_MESSAGES = [
    "Stormlog",
    "Live Monitoring & Watchdogs",
    "CLI · Docs · Examples",
]


def _safe_get_gpu_info() -> dict[str, Any]:
    try:
        return get_gpu_info()
    except Exception as exc:
        logger.debug("_safe_get_gpu_info failed: %s", exc)
        return {}


def _safe_get_system_info() -> dict[str, Any]:
    try:
        return get_system_info()
    except Exception as exc:
        logger.debug("_safe_get_system_info failed: %s", exc)
        return {}


def _safe_get_tf_system_info() -> dict[str, Any]:
    try:
        return get_tf_system_info()
    except Exception as exc:
        logger.debug("_safe_get_tf_system_info failed: %s", exc)
        return {}


def _safe_get_tf_gpu_info() -> dict[str, Any]:
    try:
        return get_tf_gpu_info()
    except Exception as exc:
        logger.debug("_safe_get_tf_gpu_info failed: %s", exc)
        return {}


def _build_welcome_info() -> str:
    return tui_builders.build_welcome_info()


def _build_system_markdown() -> str:
    return tui_builders.build_system_markdown(
        system_info=_safe_get_system_info(),
        gpu_info=_safe_get_gpu_info(),
        tf_system_info=_safe_get_tf_system_info(),
        tf_gpu_info=_safe_get_tf_gpu_info(),
    )


def _pytorch_stats_provider() -> list[dict]:
    return tui_builders.build_pytorch_stats_rows(_safe_get_gpu_info())


def _tensorflow_stats_provider() -> list[dict]:
    return tui_builders.build_tensorflow_stats_rows(_safe_get_tf_gpu_info())


def _build_framework_markdown(framework: str) -> str:
    return tui_builders.build_framework_markdown(framework)


def _build_cli_markdown() -> str:
    return tui_builders.build_cli_markdown()


def _build_visual_markdown() -> str:
    return tui_builders.build_visual_markdown()


def _build_diagnostics_markdown() -> str:
    return tui_builders.build_diagnostics_markdown()


class GPUMemoryProfilerTUI(App):
    """Main Textual application."""

    tracker_session: TrackerSession | None
    cli_runner: CLICommandRunner
    monitor_auto_cleanup: bool
    _last_monitor_stats: dict[str, Any]
    _last_timeline: dict[str, list[Any]]
    recent_alerts: List[dict[str, Any]]

    CSS = TUI_APP_CSS

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_overview", "Refresh Overview"),
        ("f", "focus_log", "Focus Log"),
        ("g", "log_gpumemprof_help", "gpumemprof info"),
        ("t", "log_tfmemprof_help", "tfmemprof info"),
    ]

    def compose(self) -> ComposeResult:
        self.overview_panel = MarkdownPanel(_build_system_markdown, id="overview")
        self.welcome_panel = AsciiWelcome(
            WELCOME_MESSAGES,
            figlet_cls=Figlet,
            logger=logger,
            id="overview-welcome",
        )
        self.welcome_info = Markdown(_build_welcome_info(), id="welcome-info")
        self.pytorch_panel = MarkdownPanel(
            lambda: _build_framework_markdown("pytorch"), id="pytorch"
        )
        self.tensorflow_panel = MarkdownPanel(
            lambda: _build_framework_markdown("tensorflow"), id="tensorflow"
        )
        self.cli_panel = MarkdownPanel(_build_cli_markdown, id="cli-docs")
        self.visual_panel = MarkdownPanel(_build_visual_markdown, id="visual-docs")
        self.diagnostics_panel = MarkdownPanel(
            _build_diagnostics_markdown, id="diagnostics-docs"
        )
        self.command_log = RichLog(highlight=True, markup=True, id="command-log")
        self.loader = LoadingIndicator(id="cli-loader")
        self.loader.display = False
        self.cli_command_input = Input(
            placeholder="gpumemprof info", id="cli-command-input"
        )
        self.monitor_status = Markdown("", id="monitor-status")
        self.monitor_stats_table = KeyValueTable(zebra_stripes=True, id="monitor-stats")
        self.monitor_log = RichLog(highlight=True, markup=True, id="monitor-log")
        self.watchdog_button = Button(
            "Auto Cleanup: OFF", id="btn-toggle-watchdog", variant="warning"
        )
        self.timeline_stats_table = KeyValueTable(
            zebra_stripes=True, id="timeline-stats"
        )
        self.timeline_canvas = TimelineCanvas(id="timeline-canvas")
        self.visual_log = RichLog(highlight=True, markup=True, id="visual-log")
        self.pytorch_profile_table = ProfileResultsTable(id="pytorch-profile-table")
        self.tensorflow_profile_table = ProfileResultsTable(
            id="tensorflow-profile-table"
        )
        self.alert_history_table = AlertHistoryTable(id="monitor-alerts-table")
        self.warning_input = Input(value="80", placeholder="80", id="input-warning")
        self.critical_input = Input(value="95", placeholder="95", id="input-critical")
        self.diagnostics_path_input = Input(
            placeholder="artifacts/run_rank0.json,artifacts/run_rank1.json",
            id="diagnostics-path-input",
        )
        self.diagnostics_rank_filter_input = Input(
            value="all",
            placeholder="all",
            id="diagnostics-rank-filter",
        )
        self.diagnostics_rank_table = DistributedRankTable(id="diagnostics-rank-table")
        self.diagnostics_timeline_canvas = DistributedTimelineCanvas(
            id="diagnostics-timeline-canvas"
        )
        self.diagnostics_anomaly_table = AnomalySummaryTable(
            id="diagnostics-anomaly-table"
        )
        self.diagnostics_log = RichLog(
            highlight=True, markup=True, id="diagnostics-log"
        )

        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Overview"):
                yield VerticalScroll(
                    self.welcome_panel,
                    self.welcome_info,
                    self.overview_panel,
                )

            with TabPane("PyTorch"):
                yield VerticalScroll(
                    self.pytorch_panel,
                    Horizontal(
                        Button(
                            "Refresh Profiles",
                            id="btn-refresh-pt-profiles",
                            variant="primary",
                        ),
                        Button(
                            "Clear Profiles",
                            id="btn-clear-pt-profiles",
                            variant="warning",
                        ),
                        id="pytorch-profile-controls",
                    ),
                    GPUStatsTable("pytorch", _pytorch_stats_provider),
                    self.pytorch_profile_table,
                )

            with TabPane("TensorFlow"):
                yield VerticalScroll(
                    self.tensorflow_panel,
                    Horizontal(
                        Button(
                            "Refresh Profiles",
                            id="btn-refresh-tf-profiles",
                            variant="primary",
                        ),
                        Button(
                            "Clear Profiles",
                            id="btn-clear-tf-profiles",
                            variant="warning",
                        ),
                        id="tensorflow-profile-controls",
                    ),
                    GPUStatsTable("tensorflow", _tensorflow_stats_provider),
                    self.tensorflow_profile_table,
                )

            with TabPane("Monitoring"):
                yield VerticalScroll(
                    self.monitor_status,
                    Horizontal(
                        Button(
                            "Start Live Tracking",
                            id="btn-start-tracking",
                            variant="primary",
                        ),
                        Button(
                            "Stop Tracking",
                            id="btn-stop-tracking",
                            variant="warning",
                        ),
                        self.watchdog_button,
                        Button(
                            "Apply Thresholds",
                            id="btn-apply-thresholds",
                            variant="primary",
                        ),
                        id="monitor-controls-row1",
                    ),
                    Horizontal(
                        Button(
                            "Force Cleanup",
                            id="btn-force-cleanup",
                            variant="success",
                        ),
                        Button(
                            "Aggressive Cleanup",
                            id="btn-force-cleanup-aggressive",
                            variant="error",
                        ),
                        Button(
                            "Export CSV",
                            id="btn-export-csv",
                            variant="success",
                        ),
                        Button(
                            "Export JSON",
                            id="btn-export-json",
                            variant="success",
                        ),
                        id="monitor-controls-row2",
                    ),
                    Horizontal(
                        Button(
                            "Clear Monitor Log",
                            id="btn-clear-monitor-log",
                        ),
                        id="monitor-controls-row3",
                    ),
                    Horizontal(
                        Label("Warning %"),
                        self.warning_input,
                        Label("Critical %"),
                        self.critical_input,
                        id="monitor-thresholds",
                    ),
                    self.monitor_stats_table,
                    self.alert_history_table,
                    self.monitor_log,
                )

            with TabPane("Visualizations"):
                yield VerticalScroll(
                    self.visual_panel,
                    Horizontal(
                        Button(
                            "Refresh Timeline",
                            id="btn-refresh-visual",
                            variant="primary",
                        ),
                        Button(
                            "Generate PNG Plot",
                            id="btn-visual-png",
                            variant="success",
                        ),
                        Button(
                            "Generate HTML Plot",
                            id="btn-visual-html",
                            variant="success",
                        ),
                        id="visual-buttons",
                    ),
                    self.timeline_stats_table,
                    self.timeline_canvas,
                    self.visual_log,
                )

            with TabPane("Diagnostics"):
                yield VerticalScroll(
                    self.diagnostics_panel,
                    Horizontal(
                        Button(
                            "Load Live",
                            id="btn-diag-load-live",
                            variant="primary",
                        ),
                        Button(
                            "Load Artifacts",
                            id="btn-diag-load-artifacts",
                            variant="success",
                        ),
                        Button("Refresh", id="btn-diag-refresh", variant="primary"),
                        id="diagnostics-controls-row1",
                    ),
                    Horizontal(
                        self.diagnostics_path_input,
                        self.diagnostics_rank_filter_input,
                        Button(
                            "Apply Filter",
                            id="btn-diag-apply-filter",
                            variant="primary",
                        ),
                        Button(
                            "Reset Filter",
                            id="btn-diag-reset-filter",
                            variant="warning",
                        ),
                        id="diagnostics-controls-row2",
                    ),
                    self.diagnostics_rank_table,
                    self.diagnostics_timeline_canvas,
                    self.diagnostics_anomaly_table,
                    self.diagnostics_log,
                )

            with TabPane("CLI & Actions"):
                yield VerticalScroll(
                    self.cli_panel,
                    Rule(),
                    Horizontal(
                        Button(
                            "gpumemprof info", id="btn-log-system", variant="primary"
                        ),
                        Button(
                            "gpumemprof monitor",
                            id="btn-log-pytorch",
                            variant="success",
                        ),
                        Button(
                            "tfmemprof monitor",
                            id="btn-log-tensorflow",
                            variant="success",
                        ),
                        Button(
                            "gpumemprof diagnose",
                            id="btn-log-diagnose",
                            variant="warning",
                        ),
                        id="cli-buttons-row1",
                    ),
                    Horizontal(
                        Button(
                            "PyTorch Sample", id="btn-run-pytorch", variant="primary"
                        ),
                        Button("TensorFlow Sample", id="btn-run-tf", variant="primary"),
                        Button(
                            "OOM Scenario", id="btn-run-oom-scenario", variant="warning"
                        ),
                        Button(
                            "Capability Matrix",
                            id="btn-run-cap-matrix",
                            variant="success",
                        ),
                        id="cli-buttons-row2",
                    ),
                    Horizontal(
                        self.cli_command_input,
                        Button("Run Command", id="btn-cli-run", variant="primary"),
                        Button(
                            "Cancel Command", id="btn-cli-cancel", variant="warning"
                        ),
                        id="cli-runner",
                    ),
                    self.loader,
                    self.command_log,
                )
        yield Footer()

    async def action_quit(self) -> None:
        self.exit()

    def action_refresh_overview(self) -> None:
        self.overview_panel.refresh_content()
        self.log_message("Overview", "System overview refreshed.")

    def action_focus_log(self) -> None:
        self.set_focus(self.command_log)

    def action_log_gpumemprof_help(self) -> None:
        self.log_message(
            "gpumemprof info",
            "Run: gpumemprof info\nRun: gpumemprof monitor --duration 30",
        )

    def action_log_tfmemprof_help(self) -> None:
        self.log_message(
            "tfmemprof info",
            "Run: tfmemprof info\nRun: tfmemprof monitor --duration 30",
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "btn-refresh-overview":
            self.action_refresh_overview()
        elif button_id == "btn-log-system":
            await self.run_cli_command("gpumemprof info")
        elif button_id == "btn-log-pytorch":
            await self.run_cli_command(
                "gpumemprof monitor --duration 30 --interval 0.5"
            )
        elif button_id == "btn-log-tensorflow":
            await self.run_cli_command("tfmemprof monitor --duration 30 --interval 0.5")
        elif button_id == "btn-log-diagnose":
            await self.run_cli_command(
                "gpumemprof diagnose --duration 0 --output artifacts/tui_diagnose"
            )
        elif button_id == "btn-run-pytorch":
            await self.run_pytorch_sample()
        elif button_id == "btn-run-tf":
            await self.run_tensorflow_sample()
        elif button_id == "btn-run-oom-scenario":
            await self.run_cli_command(
                "python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated"
            )
        elif button_id == "btn-run-cap-matrix":
            await self.run_cli_command(
                "python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated --skip-tui"
            )
        elif button_id == "btn-cli-run":
            await self.run_cli_command(self.cli_command_input.value)
        elif button_id == "btn-cli-cancel":
            await self.cancel_cli_command()
        elif button_id == "btn-start-tracking":
            await self.start_live_tracking()
        elif button_id == "btn-stop-tracking":
            self.stop_live_tracking()
        elif button_id == "btn-toggle-watchdog":
            self.toggle_auto_cleanup()
        elif button_id == "btn-force-cleanup":
            self.force_cleanup()
        elif button_id == "btn-force-cleanup-aggressive":
            self.force_cleanup(aggressive=True)
        elif button_id == "btn-export-csv":
            await self.export_tracker_events("csv")
        elif button_id == "btn-export-json":
            await self.export_tracker_events("json")
        elif button_id == "btn-apply-thresholds":
            self.apply_thresholds()
        elif button_id == "btn-clear-monitor-log":
            self.clear_monitor_log()
        elif button_id == "btn-refresh-visual":
            await self.refresh_visualizations()
        elif button_id == "btn-visual-png":
            await self.generate_visual_plot("png")
        elif button_id == "btn-visual-html":
            await self.generate_visual_plot("html")
        elif button_id == "btn-diag-load-live":
            await self.load_diagnostics_live()
        elif button_id == "btn-diag-load-artifacts":
            await self.load_diagnostics_artifacts()
        elif button_id == "btn-diag-refresh":
            await self.refresh_diagnostics()
        elif button_id == "btn-diag-apply-filter":
            self.apply_diagnostics_rank_filter()
        elif button_id == "btn-diag-reset-filter":
            self.reset_diagnostics_rank_filter()
        elif button_id == "btn-refresh-pt-profiles":
            await self.refresh_pytorch_profiles()
        elif button_id == "btn-clear-pt-profiles":
            await self.clear_pytorch_profiles()
        elif button_id == "btn-refresh-tf-profiles":
            await self.refresh_tensorflow_profiles()
        elif button_id == "btn-clear-tf-profiles":
            await self.clear_tensorflow_profiles()

    async def run_pytorch_sample(self) -> None:
        if GPUMemoryProfiler is None or torch is None:
            self.log_message(
                "PyTorch Sample", "PyTorch profiler is unavailable in this environment."
            )
            return
        if not torch.cuda.is_available():
            if CPUMemoryProfiler is None:
                self.log_message(
                    "PyTorch Sample", "CPU profiler is unavailable; install psutil."
                )
                return
            await self._execute_task(
                "PyTorch Sample (CPU)",
                self._cpu_sample_workload,
                self._format_cpu_summary,
            )
            return
        await self._execute_task(
            "PyTorch Sample",
            self._pytorch_sample_workload,
            self._format_pytorch_summary,
        )
        await self.refresh_pytorch_profiles()

    async def run_tensorflow_sample(self) -> None:
        if TFMemoryProfiler is None or tf is None:
            self.log_message(
                "TensorFlow Sample",
                "TensorFlow profiler is unavailable. Install tensorflow and tfmemprof: "
                "pip install tensorflow tfmemprof",
            )
            return
        await self._execute_task(
            "TensorFlow Sample",
            self._tensorflow_sample_workload,
            self._format_tensorflow_results,
        )
        await self.refresh_tensorflow_profiles()

    async def start_live_tracking(self) -> None:
        session = self._get_or_create_tracker_session()
        if not session:
            return
        if session.is_active:
            self.log_monitor_message("Tracker", "Live tracking already running.")
            return
        try:
            session.start()
        except TrackerUnavailableError as exc:
            self.log_monitor_message("Tracker", str(exc))
            return
        self.log_monitor_message("Tracker", "Live tracking started.")
        self._sync_threshold_inputs()
        self._update_monitor_status()

    def stop_live_tracking(self) -> None:
        session = self.tracker_session
        if not session or not session.is_active:
            self.log_monitor_message("Tracker", "Tracker is not running.")
            return
        session.stop()
        self.log_monitor_message("Tracker", "Live tracking stopped.")
        self._update_monitor_status()

    def toggle_auto_cleanup(self) -> None:
        self.monitor_auto_cleanup = not getattr(self, "monitor_auto_cleanup", False)
        session = self.tracker_session
        if session:
            session.set_auto_cleanup(self.monitor_auto_cleanup)
        state = "enabled" if self.monitor_auto_cleanup else "disabled"
        self.log_monitor_message("Watchdog", f"Auto cleanup {state}.")
        self._update_watchdog_button_label()
        self._update_monitor_status()

    def force_cleanup(self, aggressive: bool = False) -> None:
        session = self.tracker_session
        if not session or not session.is_active:
            self.log_monitor_message(
                "Watchdog", "Start tracking before requesting cleanup."
            )
            return
        if not session.force_cleanup(aggressive=aggressive):
            self.log_monitor_message(
                "Watchdog",
                "Watchdog controls are unavailable in this environment.",
            )
            return
        label = "aggressive" if aggressive else "standard"
        self.log_monitor_message("Watchdog", f"Requested {label} cleanup.")

    def clear_monitor_log(self) -> None:
        self.monitor_log.clear()
        self.log_monitor_message("Monitor", "Cleared monitoring log.")

    async def run_cli_command(self, command: str) -> None:
        command = (command or "").strip()
        if not command:
            self.log_message("CLI Runner", "Enter a command to run.")
            return
        if self.cli_runner.is_running:
            self.log_message("CLI Runner", "A command is already running.")
            return

        self.cli_command_input.value = command
        self.command_log.write(f"[bold green]$ {command}[/bold green]\n")
        self._set_loader(True)
        try:
            exit_code = await self.cli_runner.run(command, self._handle_cli_output)
            self.log_message(
                "CLI Runner", f"Command finished with exit code {exit_code}."
            )
        except Exception as exc:
            self.log_message("CLI Runner", f"Error running command: {exc}")
        finally:
            self._set_loader(False)

    async def cancel_cli_command(self) -> None:
        if not self.cli_runner.is_running:
            self.log_message("CLI Runner", "No running command to cancel.")
            return
        await self.cli_runner.cancel()
        self._set_loader(False)
        self.log_message("CLI Runner", "Command was cancelled.")

    async def _handle_cli_output(self, stream: str, line: str) -> None:
        color = "cyan" if stream == "stdout" else "yellow"
        self.command_log.write(f"[{color}]{stream}[/] {line}\n")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is self.cli_command_input:
            await self.run_cli_command(event.value)

    async def export_tracker_events(self, format: str) -> None:
        session = self.tracker_session
        if not session or not session.is_active:
            self.log_monitor_message(
                "Export", "Start tracking before exporting events."
            )
            return

        exports_dir = Path.cwd() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = exports_dir / f"tracker_events_{timestamp}.{format}"
        active_session = session

        def _export() -> bool:
            return bool(active_session.export_events(str(file_path), format=format))

        success = await asyncio.to_thread(_export)
        if not success:
            self.log_monitor_message(
                "Export", "No tracker events available to export yet."
            )
            return

        self.log_monitor_message(
            "Export",
            f"Saved tracker events to {file_path}",
        )

    def apply_thresholds(self) -> None:
        session = self.tracker_session
        if not session or session.backend != "gpu":
            self.log_monitor_message(
                "Thresholds", "Thresholds are only available when using a GPU tracker."
            )
            return

        warning_text = (
            self.warning_input.value or self.warning_input.placeholder or ""
        ).strip()
        critical_text = (
            self.critical_input.value or self.critical_input.placeholder or ""
        ).strip()

        try:
            warning = float(warning_text)
            critical = float(critical_text)
        except ValueError:
            self.log_monitor_message(
                "Thresholds", "Enter numeric warning and critical percentages."
            )
            return
        if warning >= critical:
            self.log_monitor_message(
                "Thresholds", "Warning threshold must be less than critical threshold."
            )
            return

        session.set_thresholds(warning, critical)
        self.log_monitor_message(
            "Thresholds",
            f"Updated warning={warning:.0f}% critical={critical:.0f}%.",
        )

    async def refresh_pytorch_profiles(self) -> None:
        rows = await asyncio.to_thread(fetch_pytorch_profiles)
        self.pytorch_profile_table.update_rows(rows)
        msg = (
            "Loaded PyTorch profile results."
            if rows
            else "No PyTorch profiles captured yet."
        )
        self.log_message("PyTorch Profiles", msg)

    async def clear_pytorch_profiles(self) -> None:
        success = await asyncio.to_thread(clear_pytorch_profiles)
        message = (
            "Cleared PyTorch profile results."
            if success
            else "No PyTorch profiles to clear."
        )
        self.log_message("PyTorch Profiles", message)
        await self.refresh_pytorch_profiles()

    async def refresh_tensorflow_profiles(self) -> None:
        rows = await asyncio.to_thread(fetch_tensorflow_profiles)
        self.tensorflow_profile_table.update_rows(rows)
        msg = (
            "Loaded TensorFlow profile summaries."
            if rows
            else "No TensorFlow profiles captured yet."
        )
        self.log_message("TensorFlow Profiles", msg)

    async def clear_tensorflow_profiles(self) -> None:
        success = await asyncio.to_thread(clear_tensorflow_profiles)
        message = (
            "Cleared TensorFlow profiles."
            if success
            else "No TensorFlow profiles to clear."
        )
        self.log_message("TensorFlow Profiles", message)
        await self.refresh_tensorflow_profiles()

    def refresh_monitoring_panel(self) -> None:
        session = self.tracker_session
        stats: dict[str, Any] = {}
        cleanup_stats: dict[str, Any] = {}

        if session:
            stats = session.get_statistics() or {}
            cleanup_stats = session.get_cleanup_stats() or {}
            if session.is_active:
                events = session.pull_events()
                if events:
                    self._append_monitor_events(events)

        if stats:
            self._last_monitor_stats = stats
        elif self._last_monitor_stats:
            stats = self._last_monitor_stats

        self._update_monitor_stats(stats, cleanup_stats)
        self._update_monitor_status()

    def _update_monitor_stats(
        self,
        stats: dict[str, Any],
        cleanup_stats: dict[str, Any],
    ) -> None:
        table = self.monitor_stats_table
        table.clear()
        session = self.tracker_session
        status_label = "Active" if session and session.is_active else "Idle"
        device_label = session.get_device_label() if session else "-"

        if not stats:
            table.add_row("Status", status_label)
            table.add_row("Device", device_label or "-")
            table.add_row("Current Allocated", "-")
            table.add_row("Peak Memory", "-")
            table.add_row("Alerts", "-")
            cleanup_count = cleanup_stats.get("cleanup_count", 0)
            table.add_row("Cleanups", str(cleanup_count))
            return

        cleanup_count = cleanup_stats.get("cleanup_count", 0)
        utilization = stats.get("memory_utilization_percent", 0.0)
        duration = stats.get("tracking_duration_seconds", 0.0)

        table.add_row("Status", status_label)
        table.add_row("Device", device_label or "-")
        table.add_row(
            "Current Allocated",
            self._format_bytes_metric(stats.get("current_memory_allocated")),
        )
        table.add_row(
            "Current Reserved",
            self._format_bytes_metric(stats.get("current_memory_reserved")),
        )
        table.add_row(
            "Peak Memory",
            self._format_bytes_metric(stats.get("peak_memory")),
        )
        table.add_row("Utilization", f"{utilization:.1f}%")
        table.add_row(
            "Alloc/sec",
            f"{stats.get('allocations_per_second', 0.0):.2f}",
        )
        table.add_row("Alert Count", str(stats.get("alert_count", 0)))
        table.add_row("Total Events", str(stats.get("total_events", 0)))
        table.add_row("Duration (s)", f"{duration:.1f}")
        table.add_row("Cleanups", str(cleanup_count))

    def _format_bytes_metric(self, value: Any) -> str:
        if value is None:
            return "-"
        try:
            return format_bytes(int(value))
        except (TypeError, ValueError):
            return "-"

    def _append_monitor_events(self, events: list[TrackerEventView]) -> None:
        for event in events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            color = self._event_color(event.event_type)
            summary = event.message or "No context provided."
            self.monitor_log.write(
                f"[{timestamp}] [{color}]{event.event_type.upper()}[/{color}] {summary}\n"
                f"Allocated: {event.allocated} | Reserved: {event.reserved} | Δ: {event.change}\n"
            )
        self._capture_alerts(events)
        self.alert_history_table.update_rows(self.recent_alerts)

    def _capture_alerts(self, events: list[TrackerEventView]) -> None:
        alert_types = {"warning", "critical", "error"}
        for event in events:
            if event.event_type in alert_types:
                self.recent_alerts.append(
                    {
                        "timestamp": event.timestamp,
                        "type": event.event_type,
                        "message": event.message or "",
                    }
                )
        self.recent_alerts = self.recent_alerts[-50:]

    def _event_color(self, event_type: str) -> str:
        return {
            "warning": "yellow",
            "critical": "red",
            "error": "red",
            "cleanup": "cyan",
            "peak": "magenta",
        }.get(event_type, "green")

    def _get_or_create_tracker_session(self) -> TrackerSession | None:
        if self.tracker_session is None:
            try:
                self.tracker_session = TrackerSession(
                    auto_cleanup=self.monitor_auto_cleanup
                )
            except TrackerUnavailableError as exc:
                self.log_monitor_message("Tracker", str(exc))
                return None
        else:
            self.tracker_session.set_auto_cleanup(self.monitor_auto_cleanup)
        return self.tracker_session

    def _update_monitor_status(self) -> None:
        session = self.tracker_session
        cleanup_state = "enabled" if self.monitor_auto_cleanup else "disabled"

        if session and session.is_active:
            device_label = session.get_device_label() or "current CUDA device"
            message = (
                f"Live tracking on **{device_label}**.\n"
                f"Auto cleanup is {cleanup_state}."
            )
        else:
            message = (
                "Tracker idle. Start a session to stream GPU allocation events.\n"
                f"Auto cleanup is currently {cleanup_state}."
            )

        self.monitor_status.update(message)

    def _update_watchdog_button_label(self) -> None:
        label = "Auto Cleanup: ON" if self.monitor_auto_cleanup else "Auto Cleanup: OFF"
        variant = "success" if self.monitor_auto_cleanup else "warning"
        self.watchdog_button.label = label
        self.watchdog_button.variant = variant
        self._sync_threshold_inputs()

    def _sync_threshold_inputs(self) -> None:
        session = self.tracker_session
        if not session:
            return
        thresholds = session.get_thresholds()
        warning = thresholds.get("memory_warning_percent")
        critical = thresholds.get("memory_critical_percent")
        if warning is not None:
            self.warning_input.value = f"{warning:.0f}"
        if critical is not None:
            self.critical_input.value = f"{critical:.0f}"

    async def refresh_visualizations(self) -> None:
        timeline = self._collect_timeline_data()
        if not timeline:
            self.timeline_canvas.render_placeholder(
                "No timeline samples found. Start live tracking and try again."
            )
            self._clear_timeline_stats_table()
            self.log_visual_message("Visualizations", "No timeline data yet.")
            return

        self._last_timeline = timeline
        self._update_timeline_view(timeline)
        self.log_visual_message("Visualizations", "Timeline refreshed.")

    async def generate_visual_plot(self, format: str) -> None:
        timeline = self._last_timeline or self._collect_timeline_data()
        if not timeline:
            self.log_visual_message(
                "Visualizations", "Need timeline samples before exporting plots."
            )
            return

        self.log_visual_message(
            "Visualizations", f"Generating {format.upper()} timeline plot..."
        )
        try:
            file_path = await asyncio.to_thread(
                self._save_timeline_plot, timeline, format
            )
        except ImportError as exc:
            self.log_visual_message("Visualizations", f"Error: {exc}")
            return
        except Exception as exc:
            self.log_visual_message("Visualizations", f"Export failed: {exc}")
            return

        self.log_visual_message(
            "Visualizations", f"Saved timeline plot to: {file_path}"
        )

    async def load_diagnostics_live(self) -> None:
        self.log_diagnostics_message(
            "Diagnostics",
            "Live diagnostics loading is enabled. Press Refresh to update rank views.",
        )

    async def load_diagnostics_artifacts(self) -> None:
        paths_value = (self.diagnostics_path_input.value or "").strip()
        if not paths_value:
            self.log_diagnostics_message(
                "Diagnostics",
                "Enter one or more artifact paths (comma-separated) first.",
            )
            return

        self.log_diagnostics_message(
            "Diagnostics",
            "Artifact diagnostics source configured. Press Refresh to load data.",
        )

    async def refresh_diagnostics(self) -> None:
        self.log_diagnostics_message(
            "Diagnostics",
            "Diagnostics tab is ready. Data binding and rank analytics load next.",
        )

    def apply_diagnostics_rank_filter(self) -> None:
        text = (self.diagnostics_rank_filter_input.value or "all").strip() or "all"
        self.log_diagnostics_message("Diagnostics", f"Applied rank filter: {text}")

    def reset_diagnostics_rank_filter(self) -> None:
        self.diagnostics_rank_filter_input.value = "all"
        self.log_diagnostics_message("Diagnostics", "Reset rank filter to: all")

    def _collect_timeline_data(self, interval: float = 1.0) -> dict[str, Any]:
        session = self.tracker_session
        if session:
            timeline = session.get_memory_timeline(interval=interval)
            if timeline and timeline.get("timestamps"):
                return timeline
        return self._last_timeline or {}

    def _update_timeline_view(self, timeline: dict) -> None:
        if not timeline or not timeline.get("timestamps"):
            self.timeline_canvas.render_placeholder(
                "Timeline is empty. Start tracking to capture samples."
            )
            self._clear_timeline_stats_table()
            return

        self.timeline_canvas.render_timeline(timeline)
        self._update_timeline_stats_table(timeline)

    def _update_timeline_stats_table(self, timeline: dict) -> None:
        table = self.timeline_stats_table
        table.clear()
        timestamps = timeline.get("timestamps") or []
        allocated = timeline.get("allocated") or []
        reserved = timeline.get("reserved") or []

        sample_count = len(allocated)
        if not sample_count or not timestamps:
            self._clear_timeline_stats_table()
            return

        duration = (
            max(0.0, timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        )
        alloc_max = max(allocated) if allocated else 0
        reserv_max = max(reserved) if reserved else 0
        alloc_latest = allocated[-1] if allocated else 0
        reserv_latest = reserved[-1] if reserved else 0

        table.add_row("Samples", str(sample_count))
        table.add_row("Duration (s)", f"{duration:.1f}")
        table.add_row("Allocated Max", format_bytes(int(alloc_max)))
        table.add_row("Reserved Max", format_bytes(int(reserv_max)))
        table.add_row("Allocated Latest", format_bytes(int(alloc_latest)))
        table.add_row("Reserved Latest", format_bytes(int(reserv_latest)))

    def _clear_timeline_stats_table(self) -> None:
        table = self.timeline_stats_table
        table.clear()
        table.add_row("Samples", "0")
        table.add_row("Duration (s)", "-")
        table.add_row("Allocated Max", "-")
        table.add_row("Reserved Max", "-")
        table.add_row("Allocated Latest", "-")
        table.add_row("Reserved Latest", "-")

    def _save_timeline_plot(self, timeline: dict, format: str) -> str:
        timestamps = timeline.get("timestamps") or []
        allocated = timeline.get("allocated") or []
        reserved = timeline.get("reserved") or []

        if not timestamps or not allocated:
            raise ValueError("Timeline data is empty.")

        start = timestamps[0]
        rel_times = [t - start for t in timestamps]
        allocated_gb = [val / (1024**3) for val in allocated]
        reserved_gb = [val / (1024**3) for val in reserved] if reserved else []

        plots_dir = Path.cwd() / "visualizations"
        plots_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "png":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(rel_times, allocated_gb, label="Allocated (GB)", color="tab:blue")
            if reserved_gb:
                ax.plot(rel_times, reserved_gb, label="Reserved (GB)", color="tab:red")
            ax.set_title("GPU Memory Timeline")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Memory (GB)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()

            file_path = plots_dir / f"timeline_{stamp}.png"
            fig.savefig(file_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return str(file_path)

        if format == "html":
            try:
                import plotly.graph_objects as go
            except ImportError as exc:
                raise ImportError(
                    "Plotly is required for HTML output. Install stormlog[viz]."
                ) from exc

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=rel_times,
                    y=allocated_gb,
                    mode="lines",
                    name="Allocated (GB)",
                )
            )
            if reserved_gb:
                fig.add_trace(
                    go.Scatter(
                        x=rel_times,
                        y=reserved_gb,
                        mode="lines",
                        name="Reserved (GB)",
                    )
                )

            fig.update_layout(
                title="GPU Memory Timeline",
                xaxis_title="Time (s)",
                yaxis_title="Memory (GB)",
                hovermode="x unified",
            )

            file_path = plots_dir / f"timeline_{stamp}.html"
            fig.write_html(file_path)
            return str(file_path)

        raise ValueError(f"Unsupported format: {format}")

    async def _execute_task(
        self,
        title: str,
        func: Callable[[], Any],
        formatter: Optional[Callable[[Any], str]],
    ) -> None:
        formatter = formatter or (lambda value: str(value))
        self._set_loader(True)
        self.log_message(title, "Running sample workload...")
        try:
            result = await asyncio.to_thread(func)
            self.log_message(title, formatter(result))
        except Exception as exc:
            self.log_message(title, f"Error: {exc}")
        finally:
            self._set_loader(False)

    def _set_loader(self, visible: bool) -> None:
        self.loader.display = visible

    @staticmethod
    def _pytorch_sample_workload() -> dict[str, Any]:
        if GPUMemoryProfiler is None or torch is None:
            raise RuntimeError("PyTorch profiler is unavailable.")
        return run_pytorch_sample_workload(GPUMemoryProfiler, torch)

    @staticmethod
    def _tensorflow_sample_workload() -> Any:
        if TFMemoryProfiler is None or tf is None:
            raise RuntimeError(
                "TensorFlow profiler is unavailable. Install tensorflow and tfmemprof: "
                "pip install tensorflow tfmemprof"
            )
        return run_tensorflow_sample_workload(TFMemoryProfiler, tf)

    @staticmethod
    def _cpu_sample_workload() -> dict[str, Any]:
        if CPUMemoryProfiler is None:
            raise RuntimeError("CPUMemoryProfiler is unavailable.")
        return run_cpu_sample_workload(CPUMemoryProfiler)

    @staticmethod
    def _format_pytorch_summary(summary: dict) -> str:
        return format_pytorch_summary(summary)

    @staticmethod
    def _format_tensorflow_results(results: Any) -> str:
        return format_tensorflow_results(results)

    @staticmethod
    def _format_cpu_summary(summary: dict) -> str:
        return format_cpu_summary(summary)

    def log_monitor_message(self, title: str, content: str) -> None:
        self.monitor_log.write(f"[bold]{title}[/bold]\n{content}\n")

    def log_visual_message(self, title: str, content: str) -> None:
        self.visual_log.write(f"[bold]{title}[/bold]\n{content}\n")

    def log_diagnostics_message(self, title: str, content: str) -> None:
        self.diagnostics_log.write(f"[bold]{title}[/bold]\n{content}\n")

    def log_message(self, title: str, content: str) -> None:
        self.command_log.write(f"[bold]{title}[/bold]\n{content}\n")

    async def on_mount(self) -> None:
        self.tracker_session = None
        self.cli_runner = CLICommandRunner()
        self.monitor_auto_cleanup = False
        self._last_monitor_stats = {}
        self._last_timeline = {}
        self.recent_alerts = []
        self._diagnostics_source = "none"
        self.set_interval(1.0, self.refresh_monitoring_panel)
        self._update_watchdog_button_label()
        self._update_monitor_status()
        self.timeline_canvas.render_placeholder(
            "No timeline data yet. Start live tracking and refresh."
        )
        self._clear_timeline_stats_table()
        self.diagnostics_rank_table.update_rows([])
        self.diagnostics_timeline_canvas.render_placeholder(
            "No distributed timelines yet. Load live or artifact data."
        )
        self.diagnostics_anomaly_table.update_rows([])

        # Initial log entry
        await asyncio.sleep(0)
        self.log_message(
            "Welcome",
            "Use the tabs or press [b]r[/b] to refresh the overview. "
            "Buttons in the CLI tab will log summaries here.",
        )
        self.log_diagnostics_message(
            "Diagnostics",
            "Use Load Live or Load Artifacts, then Refresh to build rank-level diagnostics.",
        )
        await self.refresh_pytorch_profiles()
        await self.refresh_tensorflow_profiles()


def run_app() -> None:
    """Entry-point to launch the Textual application."""
    GPUMemoryProfilerTUI().run()


__all__ = ["run_app", "GPUMemoryProfilerTUI"]
