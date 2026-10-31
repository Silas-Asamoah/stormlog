"""Collision-safe MLX capture bundles and shared verdict envelopes."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from stormlog.diagnose_report import (
    DIAGNOSE_MANIFEST_SCHEMA_VERSION,
    build_diagnose_report,
    validate_output_directory,
)
from stormlog.exit_codes import ExitCode
from stormlog.report import Artifact, Evidence, Finding, build_report, write_report
from stormlog.session import (
    create_session_summary,
    session_summary_to_dict,
    update_session_summary,
)
from stormlog.system_info import get_system_info
from stormlog.telemetry_sink import TelemetrySinkConfig

from .models import TrackingResult
from .runtime import MLXRuntime, MLXRuntimeError, Runtime
from .sampling import positive_interval
from .tracker import MemoryTracker
from .utils import get_device_info

# One telemetry record is about 2 KB; a smaller buffer drops every record.
MIN_BUFFER_BYTES = 64 * 1024


@dataclass(frozen=True)
class CaptureOptions:
    interval: float = 0.1
    max_history: int = 1000
    alert_threshold_mb: float | None = None
    rollover_max_events: int = 10000
    retention_max_files: int = 8
    max_buffer_bytes: int = 4 * 1024**2
    oom_max_dumps: int = 5
    oom_max_total_mb: int = 256
    job_id: str | None = None
    rank: int | None = None
    local_rank: int | None = None
    world_size: int | None = None

    def __post_init__(self) -> None:
        positive_interval(self.interval, "interval")
        if self.alert_threshold_mb is not None:
            positive_interval(self.alert_threshold_mb, "alert_threshold_mb")
        for name in (
            "max_history",
            "rollover_max_events",
            "retention_max_files",
            "max_buffer_bytes",
            "oom_max_dumps",
            "oom_max_total_mb",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_buffer_bytes < MIN_BUFFER_BYTES:
            raise ValueError(f"max_buffer_bytes must be at least {MIN_BUFFER_BYTES}")


@dataclass
class Outcome:
    exit_code: int = int(ExitCode.OK)
    error: str | None = None
    workload: dict[str, Any] = field(default_factory=dict)
    oom_dump: str | None = None
    artifact_errors: list[str] = field(default_factory=list)

    @property
    def session_status(self) -> str:
        if self.exit_code == ExitCode.INTERRUPTED:
            return "interrupted"
        return "completed" if self.exit_code == ExitCode.OK else "incomplete"


def create_artifact_dir(output: str | Path | None, command: str) -> Path:
    validate_output_directory(str(output) if output is not None else None)
    base = Path(output).resolve() if output is not None else Path.cwd()
    if output is not None and not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for suffix in range(10000):
        directory = base / f"mlx-{command}-{stamp}-{suffix}"
        try:
            directory.mkdir(parents=True, exist_ok=False)
            return directory
        except FileExistsError:
            continue
    raise OSError("Cannot create a unique MLX artifact directory")


class CaptureBundle:
    """One capture; native discovery is deferred until ``start``."""

    def __init__(
        self,
        output: str | Path | None,
        command: str,
        options: CaptureOptions,
        *,
        identity: Mapping[str, Any] | None = None,
        command_line: str | None = None,
    ) -> None:
        self.directory = create_artifact_dir(output, command)
        self.command = command
        self.options = options
        # Caller-declared, already redacted identity; recorded verbatim.
        self.identity = dict(identity) if identity is not None else None
        self.command_line = command_line or f"mlxmemprof {command}"
        self.tracker: MemoryTracker | None = None
        self.session = create_session_summary(source=f"stormlog.mlx.{command}")
        self.environment: dict[str, Any] = {
            "system": get_system_info(),
            "framework": "mlx",
            "memory_scope": "process_allocator",
        }
        if self.identity is not None:
            self.environment["capture_identity"] = self.identity

    def start(self, runtime: Runtime | None = None) -> None:
        adapter = runtime if runtime is not None else MLXRuntime()
        self.environment["device"] = get_device_info(adapter)
        self.tracker = MemoryTracker(
            runtime=adapter,
            sampling_interval=self.options.interval,
            max_history=self.options.max_history,
            alert_threshold_mb=self.options.alert_threshold_mb,
            job_id=self.options.job_id,
            rank=self.options.rank,
            local_rank=self.options.local_rank,
            world_size=self.options.world_size,
            telemetry_sink_config=TelemetrySinkConfig(
                self.directory / "telemetry",
                max_buffer_bytes=self.options.max_buffer_bytes,
                rollover_max_events=self.options.rollover_max_events,
                retention_max_files=self.options.retention_max_files,
            ),
            enable_oom_flight_recorder=True,
            oom_dump_dir=str(self.directory / "oom"),
            oom_max_dumps=self.options.oom_max_dumps,
            oom_max_total_mb=self.options.oom_max_total_mb,
        )
        self.environment["memory_capabilities"] = self.tracker.collector.capabilities
        self.tracker.start_tracking()

    def finish(self, outcome: Outcome) -> Outcome:
        """Best effort finalization; never turn a workload failure into success."""
        result = self._stop(outcome)
        if result is not None:
            self.session = result.session_summary
        else:
            self.session = update_session_summary(
                self.session, status=outcome.session_status, ended_at_ns=time.time_ns()
            )
        summary = self._summary(result, outcome)
        payloads = {
            "environment.json": self.environment,
            "diagnostic_summary.json": summary,
        }
        if result is not None:
            payloads["tracking.json"] = result.to_dict()
        files = self._write_payloads(payloads, outcome)
        if (self.directory / "telemetry" / "manifest.json").exists():
            files.append("telemetry/manifest.json")
        self._write_envelope(files, summary, outcome)
        return outcome

    def _stop(self, outcome: Outcome) -> TrackingResult | None:
        if self.tracker is None:
            return None
        try:
            status = (
                None if outcome.exit_code == ExitCode.OK else outcome.session_status
            )
            result = self.tracker.stop_tracking(status=status)
            if (
                result is not None
                and result.session_summary.status != "completed"
                and outcome.exit_code == ExitCode.OK
            ):
                outcome.exit_code = int(ExitCode.ERROR)
                outcome.error = "Collection or persistence did not complete"
            return result
        except BaseException as exc:
            self._artifact_error(outcome, "tracker cleanup", exc)
            self.tracker._stop_best_effort(outcome.session_status)
            return self.tracker.get_results() if self.tracker.session_summary else None

    def _summary(
        self, result: TrackingResult | None, outcome: Outcome
    ) -> dict[str, Any]:
        alerts = result.total_alerts if result is not None else 0
        if outcome.exit_code == ExitCode.OK and alerts:
            outcome.exit_code = int(ExitCode.FINDINGS)
        return {
            "schema_version": 1,
            "framework": "mlx",
            "backend": "metal",
            "memory_scope": "process_allocator",
            "session_status": self.session.status,
            "allocated_bytes": (
                result.samples[-1].active_bytes
                if result is not None and result.samples
                else None
            ),
            "peak_bytes": result.sampled_peak_bytes if result is not None else None,
            "reserved_bytes": None,
            "total_bytes": None,
            "utilization_ratio": None,
            "fragmentation_ratio": None,
            "risk_flags": {
                "oom_occurred": bool(outcome.workload.get("oom_occurred")),
                "active_threshold": bool(alerts),
            },
            "num_ooms": int(bool(outcome.workload.get("oom_occurred"))),
            "threshold_alerts": alerts,
            "workload": outcome.workload,
            "error": outcome.error,
            "suggestions": [
                "Review application evaluation, warmup and retained roots when investigating active growth."
            ],
        }

    def _write_payloads(self, payloads: dict[str, Any], outcome: Outcome) -> list[str]:
        files = []
        for name, payload in payloads.items():
            try:
                (self.directory / name).write_text(
                    json.dumps(payload, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                files.append(name)
            except Exception as exc:
                self._artifact_error(outcome, name, exc)
        return files

    @staticmethod
    def _artifact_error(outcome: Outcome, context: str, exc: BaseException) -> None:
        outcome.artifact_errors.append(f"{context}: {type(exc).__name__}: {exc}")
        if outcome.exit_code == ExitCode.OK or (
            outcome.exit_code == ExitCode.FINDINGS
            and outcome.workload.get("status") != "failed"
        ):
            outcome.exit_code = int(ExitCode.ERROR)
            outcome.error = "Artifact finalization failed"

    def _write_envelope(
        self, files: list[str], summary: dict[str, Any], outcome: Outcome
    ) -> None:
        if outcome.artifact_errors:
            self.session = update_session_summary(
                self.session, status=outcome.session_status
            )
        original_code = outcome.exit_code
        self._save_report(files, summary, outcome)
        if outcome.artifact_errors:
            self.session = update_session_summary(
                self.session, status=outcome.session_status
            )
        manifest = {
            "schema_version": DIAGNOSE_MANIFEST_SCHEMA_VERSION,
            "version": DIAGNOSE_MANIFEST_SCHEMA_VERSION,
            "command_line": self.command_line,
            "files": [*files, "manifest.json"]
            + (["report.json"] if (self.directory / "report.json").exists() else []),
            "exit_code": outcome.exit_code,
            "risk_detected": bool(any(summary["risk_flags"].values())),
            "session_id": self.session.session_id,
            "session_status": self.session.status,
            "session": session_summary_to_dict(self.session),
            "framework": "mlx",
            "error": outcome.error,
        }
        self._write_payloads({"manifest.json": manifest}, outcome)
        if outcome.exit_code != original_code:
            self._save_report(files, summary, outcome)

    def _save_report(
        self, files: list[str], summary: dict[str, Any], outcome: Outcome
    ) -> None:
        try:
            write_report(
                self.directory / "report.json", self._report(files, summary, outcome)
            )
        except Exception as exc:
            self._artifact_error(outcome, "report.json", exc)
            with suppress(OSError):
                (self.directory / "report.json").unlink(missing_ok=True)

    def _report(
        self, files: list[str], summary: dict[str, Any], outcome: Outcome
    ) -> dict[str, Any]:
        if self.command == "diagnose":
            report = build_diagnose_report(
                tool_name="mlxmemprof",
                summary=summary,
                exit_code=outcome.exit_code,
                session_id=self.session.session_id,
                files=files,
                error=outcome.error,
            )
        else:
            findings = (
                _capture_findings(summary) if "diagnostic_summary.json" in files else []
            )
            report = build_report(
                report_kind="mlx_capture",
                tool_name="mlxmemprof",
                command=self.command,
                exit_code=outcome.exit_code,
                summary=outcome.error or "Captured this process's MLX allocator",
                session_id=self.session.session_id,
                findings=findings,
                metrics={"sampled_peak_bytes": summary["peak_bytes"]},
                artifacts=[Artifact("capture", name) for name in files],
            )
        report["payload"] = {
            **report.get("payload", {}),
            "framework": "mlx",
            "memory_scope": "process_allocator",
            "workload": outcome.workload,
            "error": outcome.error,
            "artifact_errors": outcome.artifact_errors,
            "oom_dump": outcome.oom_dump,
            "final_lazy_work": "application_responsibility",
            "capture_options": self.options.__dict__,
        }
        if self.identity is not None:
            report["payload"]["capture_identity"] = self.identity
        return report


def _capture_findings(summary: dict[str, Any]) -> list[Finding]:
    findings = []
    for flag, raised in summary["risk_flags"].items():
        if raised:
            findings.append(
                Finding(
                    id=f"mlx.{flag}",
                    kind=flag,
                    severity="critical" if flag == "oom_occurred" else "warning",
                    title=(
                        "MLX allocation failed"
                        if flag == "oom_occurred"
                        else "Active allocator threshold was reached"
                    ),
                    evidence=[
                        Evidence(
                            kind="diagnose_summary",
                            path="diagnostic_summary.json",
                            pointer=f"/risk_flags/{flag}",
                        )
                    ],
                )
            )
    return findings


def capture_failure(exc: BaseException) -> Outcome:
    code = ExitCode.ERROR
    if isinstance(exc, KeyboardInterrupt):
        code = ExitCode.INTERRUPTED
    elif isinstance(exc, MLXRuntimeError):
        code = ExitCode.USAGE
    return Outcome(int(code), f"{type(exc).__name__}: {exc}")
