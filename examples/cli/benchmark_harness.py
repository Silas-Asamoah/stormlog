"""Operability harness for always-on monitoring qualification."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

import psutil

from stormlog.cpu_profiler import CPUMemoryTracker
from stormlog.telemetry_sink import TelemetrySinkConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_VERSION = "v0.4"
DEFAULT_BUDGETS_PATH = REPO_ROOT / "docs" / "benchmarks" / "v0.4_operating_budget.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "latest_v0.4.json"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "benchmarks" / "operability"
DEFAULT_PROFILE = "pr"
DEFAULT_MODE = "all"
DEFAULT_GATE_MODE = "budget"
DEFAULT_ITERATIONS = 5_000
DEFAULT_ALLOCATION_KB = 512
DEFAULT_OVERHEAD_TRIAL_COUNT = 5
DEFAULT_OVERHEAD_TRIAL_QUANTILE = 0.25
# The soak reads RSS at this many evenly spaced checkpoints. The first
# checkpoint is the warmup boundary: it becomes the RSS baseline so allocator,
# flush-thread, and first-segment setup are excluded from growth metrics.
SOAK_RSS_CHECKPOINT_COUNT = 50
REFERENCE_INTERVAL_SECONDS = 0.1
DEFAULT_RETENTION_VALIDATION = {
    "flush_every_events": 50,
    "flush_every_seconds": 2.0,
    "rollover_max_bytes": 8 * 1024,
    "rollover_max_events": 200,
    "retention_max_files": 2,
    "retention_max_total_bytes": 24 * 1024,
    "sample_limit": 2_048,
}
PROFILE_EQUIVALENT_HOURS = {
    "pr": 6.0,
    "nightly": 24.0,
}
DEFAULT_RETENTION_LIMITS = {
    "flush_every_events": 50,
    "flush_every_seconds": 2.0,
    "rollover_max_bytes": 64 * 1024 * 1024,
    "rollover_max_events": 10_000,
    "retention_max_files": 8,
    "retention_max_total_bytes": 512 * 1024 * 1024,
}


class SinkOverrides(TypedDict, total=False):
    flush_every_events: int
    flush_every_seconds: float
    rollover_max_bytes: int
    rollover_max_events: int
    retention_max_files: int
    retention_max_total_bytes: int


def _default_runtime_baseline_path() -> Path:
    return REPO_ROOT / "docs" / "benchmarks" / "v0.4_baseline.json"


def _default_runtime_tolerances_path() -> Path:
    return REPO_ROOT / "docs" / "benchmarks" / "v0.4_tolerances.json"


def _directory_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _process_rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def _pct_overhead(baseline: float, measured: float) -> float:
    if baseline <= 0:
        return 0.0 if measured <= 0 else float("inf")
    return ((measured - baseline) / baseline) * 100.0


def _format_metric_value(metric_key: str, value: float) -> str:
    if metric_key.endswith("_pct"):
        return f"{value:.2f}%"
    return f"{value:.0f}"


def _run_workload(
    iterations: int,
    allocation_kb: int,
    *,
    on_iteration: Optional[Callable[[int], None]] = None,
) -> int:
    """Run a deterministic allocation workload with optional sampling hooks."""
    checksum = 0
    block_bytes = max(1, allocation_kb) * 1024

    for step in range(max(1, iterations)):
        payload = bytearray(block_bytes)
        marker = (step * 17) % 251
        for offset in range(0, len(payload), 4096):
            payload[offset] = marker
            checksum += payload[offset]
        checksum ^= len(payload)
        if on_iteration is not None:
            on_iteration(step)

    return checksum


def _sample_points(iterations: int, sample_count: int) -> list[int]:
    total_steps = max(1, iterations)
    total_samples = max(1, sample_count)
    return [
        min(total_steps - 1, math.floor(index * total_steps / total_samples))
        for index in range(total_samples)
    ]


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    default_interval: float
    factory: Callable[[Path, float, Optional[SinkOverrides]], "RuntimeSession"]


class RuntimeSession:
    """Synthetic tracking session used by the operability harness."""

    def start(self) -> None:
        raise NotImplementedError

    def emit_sample(self, index: int) -> None:
        raise NotImplementedError

    def finish(self) -> dict[str, Any]:
        raise NotImplementedError


def _finalize_scenario_summary(
    scenario_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    summary_path = scenario_dir / "summary.json"
    artifact_size_bytes = 0

    while True:
        summary["artifact_size_bytes"] = artifact_size_bytes
        _write_json(summary_path, summary)
        final_size = _directory_size_bytes(scenario_dir)
        if final_size == artifact_size_bytes:
            return summary
        artifact_size_bytes = final_size


def _telemetry_sink_config(
    artifact_dir: Path,
    sink_overrides: Optional[SinkOverrides],
) -> TelemetrySinkConfig:
    merged: SinkOverrides = {
        "flush_every_events": int(DEFAULT_RETENTION_LIMITS["flush_every_events"]),
        "flush_every_seconds": float(DEFAULT_RETENTION_LIMITS["flush_every_seconds"]),
        "rollover_max_bytes": int(DEFAULT_RETENTION_LIMITS["rollover_max_bytes"]),
        "rollover_max_events": int(DEFAULT_RETENTION_LIMITS["rollover_max_events"]),
        "retention_max_files": int(DEFAULT_RETENTION_LIMITS["retention_max_files"]),
        "retention_max_total_bytes": int(
            DEFAULT_RETENTION_LIMITS["retention_max_total_bytes"]
        ),
    }
    if sink_overrides:
        merged.update(sink_overrides)
    return TelemetrySinkConfig(
        root_dir=artifact_dir / "sink",
        flush_every_events=merged["flush_every_events"],
        flush_every_seconds=merged["flush_every_seconds"],
        rollover_max_bytes=merged["rollover_max_bytes"],
        rollover_max_events=merged["rollover_max_events"],
        retention_max_files=merged["retention_max_files"],
        retention_max_total_bytes=merged["retention_max_total_bytes"],
    )


class CPURuntimeSession(RuntimeSession):
    """Synthetic `gpumemprof track` CPU-fallback session."""

    def __init__(
        self,
        artifact_dir: Path,
        sampling_interval: float,
        sink_overrides: Optional[SinkOverrides] = None,
    ) -> None:
        sink_config = _telemetry_sink_config(artifact_dir, sink_overrides)
        self.artifact_dir = artifact_dir
        self.tracker = CPUMemoryTracker(
            sampling_interval=sampling_interval,
            enable_alerts=False,
            telemetry_sink_config=sink_config,
        )

    def start(self) -> None:
        self.tracker.is_tracking = True
        self.tracker._stop_event.clear()
        self.tracker.stats["tracking_start_time"] = time.time()
        self.tracker._session_summary = None
        self.tracker._open_session()
        self.tracker._add_event("start", 0, "operability harness start")

    def emit_sample(self, index: int) -> None:
        rss = self.tracker._current_rss()
        self.tracker.stats["peak_memory"] = max(self.tracker.stats["peak_memory"], rss)
        self.tracker.stats["total_events"] = int(self.tracker.stats["total_events"]) + 1
        self.tracker._add_event("sample", 0, f"operability sample {index}")

    def finish(self) -> dict[str, Any]:
        self.tracker.stop_tracking()
        output_path = self.artifact_dir / "events.json"
        self.tracker.export_events(str(output_path), format="json")
        events = self.tracker.get_events()
        return {
            "stats": self.tracker.get_statistics(),
            "event_count": len(events),
            "collector_failure_event_count": 0,
            "output_path": str(output_path),
        }


class TensorFlowRuntimeSession(RuntimeSession):
    """Synthetic `tfmemprof track --device /CPU:0` session."""

    def __init__(
        self,
        artifact_dir: Path,
        sampling_interval: float,
        sink_overrides: Optional[SinkOverrides] = None,
    ) -> None:
        tf_module = importlib.import_module("stormlog.tensorflow.tracker")
        tracker_cls = getattr(tf_module, "MemoryTracker")
        sink_config = _telemetry_sink_config(artifact_dir, sink_overrides)
        self.artifact_dir = artifact_dir
        self.tracker = tracker_cls(
            sampling_interval=sampling_interval,
            device="/CPU:0",
            enable_logging=False,
            telemetry_sink_config=sink_config,
        )

    def start(self) -> None:
        self.tracker.tracking = True
        self.tracker._stop_event.clear()
        self.tracker._session_start_time = time.time()
        self.tracker._session_end_time = None
        self.tracker._session_summary = None
        with self.tracker._lock:
            self.tracker._reset_history()
        self.tracker._last_successful_memory_mb = None
        self.tracker._set_collector_health(
            status="healthy",
            telemetry_partial=False,
        )
        self.tracker._ensure_session_summary()
        self.tracker._append_event(
            timestamp=self.tracker._session_start_time,
            memory_mb=self.tracker._status_memory_value(),
            event_type="start",
            context="operability harness start",
        )

    def emit_sample(self, index: int) -> None:
        _ = index
        self.tracker._run_tracking_iteration()

    def finish(self) -> dict[str, Any]:
        result = self.tracker.stop_tracking()
        output_path = self.artifact_dir / "track.json"
        payload = {
            "peak_memory": result.peak_memory,
            "average_memory": result.average_memory,
            "duration": result.duration,
            "memory_usage": result.memory_usage,
            "timestamps": result.timestamps,
            "alerts": result.alerts_triggered,
            "events": result.events,
            "history_window_limit": result.history_window_limit,
            "history_retained_samples": result.history_retained_samples,
            "history_dropped_samples": result.history_dropped_samples,
            "history_retained_events": result.history_retained_events,
            "history_dropped_events": result.history_dropped_events,
            "history_retained_alerts": result.history_retained_alerts,
            "history_dropped_alerts": result.history_dropped_alerts,
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        stats = self.tracker.get_statistics()
        return {
            "stats": stats,
            "event_count": result.history_retained_events
            + result.history_dropped_events,
            "collector_failure_event_count": int(
                stats.get("collector_failure_event_count", 0)
            ),
            "output_path": str(output_path),
        }


_RUNTIME_SPECS: dict[str, RuntimeSpec] = {
    "gpumemprof_cpu": RuntimeSpec(
        name="gpumemprof_cpu",
        default_interval=0.1,
        factory=lambda artifact_dir, interval, sink_overrides: CPURuntimeSession(
            artifact_dir,
            interval,
            sink_overrides,
        ),
    ),
    "tfmemprof_cpu": RuntimeSpec(
        name="tfmemprof_cpu",
        default_interval=1.0,
        factory=lambda artifact_dir, interval, sink_overrides: TensorFlowRuntimeSession(
            artifact_dir,
            interval,
            sink_overrides,
        ),
    ),
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run_unprofiled_scenario(
    scenario_dir: Path,
    *,
    iterations: int,
    allocation_kb: int,
) -> dict[str, Any]:
    _prepare_directory(scenario_dir)
    gc.collect()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    checksum = _run_workload(iterations=iterations, allocation_kb=allocation_kb)
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    summary = {
        "name": "unprofiled",
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "checksum": checksum,
        "artifact_dir": str(scenario_dir),
    }
    return _finalize_scenario_summary(scenario_dir, summary)


def _run_tracked_scenario(
    spec: RuntimeSpec,
    scenario_dir: Path,
    *,
    iterations: int,
    allocation_kb: int,
    sample_count: int,
    sink_overrides: Optional[SinkOverrides] = None,
) -> dict[str, Any]:
    _prepare_directory(scenario_dir)
    gc.collect()
    session = spec.factory(scenario_dir, spec.default_interval, sink_overrides)
    session_started = False
    finish_attempted = False
    session_report: dict[str, Any] | None = None
    try:
        session.start()
        session_started = True
        schedule = _sample_points(iterations, sample_count)
        schedule_index = 0
        emitted = 0

        def _on_iteration(step: int) -> None:
            nonlocal schedule_index, emitted
            while schedule_index < len(schedule) and schedule[schedule_index] == step:
                session.emit_sample(emitted)
                emitted += 1
                schedule_index += 1

        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        checksum = _run_workload(
            iterations=iterations,
            allocation_kb=allocation_kb,
            on_iteration=_on_iteration,
        )
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start

        while emitted < sample_count:
            session.emit_sample(emitted)
            emitted += 1

        finish_attempted = True
        session_report = session.finish()
    except Exception:
        if session_started and not finish_attempted:
            with suppress(Exception):
                session.finish()
        raise

    assert session_report is not None
    stats = dict(session_report["stats"])
    summary = {
        "name": spec.name,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "checksum": checksum,
        "sample_count": sample_count,
        "emitted_samples": emitted,
        "event_count": int(session_report["event_count"]),
        "collector_failure_event_count": int(
            session_report["collector_failure_event_count"]
        ),
        "stats": stats,
        "artifact_dir": str(scenario_dir),
        "output_path": str(session_report["output_path"]),
    }
    return _finalize_scenario_summary(scenario_dir, summary)


def _build_overhead_metrics(
    unprofiled: Mapping[str, Any],
    tracked: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "runtime_overhead_pct": _pct_overhead(
            float(unprofiled["wall_seconds"]),
            float(tracked["wall_seconds"]),
        ),
        "cpu_overhead_pct": _pct_overhead(
            float(unprofiled["cpu_seconds"]),
            float(tracked["cpu_seconds"]),
        ),
        "artifact_growth_bytes": float(
            max(
                0,
                int(tracked["artifact_size_bytes"])
                - int(unprofiled["artifact_size_bytes"]),
            )
        ),
    }


def _select_overhead_trial_index(trials: Sequence[Mapping[str, Any]]) -> int:
    if not trials:
        raise ValueError("at least one overhead trial is required")
    ordered = sorted(
        enumerate(trials),
        key=lambda item: (
            float(item[1]["metrics"]["runtime_overhead_pct"]),
            float(item[1]["metrics"]["cpu_overhead_pct"]),
            item[0],
        ),
    )
    selected_rank = int(
        math.floor((len(ordered) - 1) * DEFAULT_OVERHEAD_TRIAL_QUANTILE)
    )
    return ordered[selected_rank][0]


def _promote_overhead_scenario(
    source_dir: Path,
    target_dir: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))

    promoted = dict(summary)
    promoted["artifact_dir"] = str(target_dir)

    output_path = promoted.get("output_path")
    if output_path:
        promoted["output_path"] = str(target_dir / Path(str(output_path)).name)

    return _finalize_scenario_summary(target_dir, promoted)


def _overhead_trial_root(
    spec: RuntimeSpec,
    runtime_dir: Path,
    scratch_root: Optional[Path],
) -> Path:
    """Return the directory that holds the overhead trials while they run.

    Trials time the tracked workload with the sink's flush and fsync calls on
    the workload's critical path, so the trial directory's storage latency goes
    straight into ``runtime_overhead_pct``. A RAM-backed scratch root (for
    example ``/dev/shm`` on Linux) keeps shared-runner disk latency out of the
    measurement; the selected trial is still promoted into ``runtime_dir``.
    """
    if scratch_root is None:
        return runtime_dir / ".overhead_trials"
    return scratch_root / spec.name / "overhead_trials"


def _run_overhead_report(
    spec: RuntimeSpec,
    runtime_dir: Path,
    *,
    iterations: int,
    allocation_kb: int,
    scratch_root: Optional[Path] = None,
) -> dict[str, Any]:
    trial_root = _overhead_trial_root(spec, runtime_dir, scratch_root)
    if trial_root.exists():
        shutil.rmtree(trial_root)

    sample_count = _overhead_sample_count(spec, iterations)
    trial_reports: list[dict[str, Any]] = []
    try:
        for trial_number in range(1, DEFAULT_OVERHEAD_TRIAL_COUNT + 1):
            trial_dir = trial_root / f"trial_{trial_number}"
            unprofiled = _run_unprofiled_scenario(
                trial_dir / "unprofiled",
                iterations=iterations,
                allocation_kb=allocation_kb,
            )
            tracked = _run_tracked_scenario(
                spec,
                trial_dir / "tracked_default",
                iterations=iterations,
                allocation_kb=allocation_kb,
                sample_count=sample_count,
            )
            trial_reports.append(
                {
                    "trial_number": trial_number,
                    "scenarios": {
                        "unprofiled": unprofiled,
                        "tracked_default": tracked,
                    },
                    "metrics": _build_overhead_metrics(unprofiled, tracked),
                }
            )

        selected_index = _select_overhead_trial_index(trial_reports)
        selected_trial = trial_reports[selected_index]
        overhead_dir = runtime_dir / "overhead"
        _prepare_directory(overhead_dir)

        promoted_unprofiled = _promote_overhead_scenario(
            trial_root / f"trial_{selected_trial['trial_number']}" / "unprofiled",
            overhead_dir / "unprofiled",
            selected_trial["scenarios"]["unprofiled"],
        )
        promoted_tracked = _promote_overhead_scenario(
            trial_root / f"trial_{selected_trial['trial_number']}" / "tracked_default",
            overhead_dir / "tracked_default",
            selected_trial["scenarios"]["tracked_default"],
        )
    finally:
        if trial_root.exists():
            shutil.rmtree(trial_root, ignore_errors=True)
        if scratch_root is not None:
            with suppress(OSError):
                trial_root.parent.rmdir()

    return {
        "scenarios": {
            "unprofiled": promoted_unprofiled,
            "tracked_default": promoted_tracked,
        },
        "metrics": _build_overhead_metrics(promoted_unprofiled, promoted_tracked),
        "trial_count": DEFAULT_OVERHEAD_TRIAL_COUNT,
        "trial_quantile": DEFAULT_OVERHEAD_TRIAL_QUANTILE,
        "selected_trial": int(selected_trial["trial_number"]),
        "trial_metrics": [dict(trial["metrics"]) for trial in trial_reports],
        "scratch_root": str(scratch_root) if scratch_root is not None else None,
    }


def _soak_rss_checkpoints(sample_count: int) -> list[int]:
    """Return the sample indices after which the soak reads process RSS.

    The first index is the warmup boundary and supplies the RSS baseline; the
    last index is always the final sample so the measured window ends with the
    soak loop itself.
    """
    stride = max(sample_count // SOAK_RSS_CHECKPOINT_COUNT, 1)
    checkpoints = list(range(stride - 1, sample_count, stride))
    if checkpoints[-1] != sample_count - 1:
        checkpoints.append(sample_count - 1)
    return checkpoints


def _soak_rss_metrics(
    rss_points: Sequence[int],
    *,
    checkpoints: Sequence[int],
    sample_count: int,
    interval_seconds: float,
    rss_after_finish: int,
) -> dict[str, float | int]:
    """Derive soak RSS metrics from the in-loop checkpoint readings only.

    Session finalization (sink close, rollup load-back, history export) is a
    one-shot transient whose residual RSS depends on allocator behaviour, not
    on steady-state growth, so it is reported separately and never gated.
    """
    baseline_rss = int(rss_points[0])
    final_rss = int(rss_points[-1])
    final_delta = final_rss - baseline_rss
    warmup_sample_count = int(checkpoints[0]) + 1
    measured_sample_count = sample_count - warmup_sample_count
    measured_seconds = measured_sample_count * interval_seconds
    return {
        "rss_warmup_sample_count": warmup_sample_count,
        "rss_measured_sample_count": measured_sample_count,
        "rss_measured_equivalent_seconds": measured_seconds,
        "rss_checkpoint_count": len(rss_points),
        "rss_baseline_bytes": baseline_rss,
        "rss_final_bytes": final_rss,
        "rss_delta_bytes": final_delta,
        "rss_growth_per_24h_equiv": (
            (final_delta / measured_seconds) * 86400.0 if measured_seconds else 0.0
        ),
        "max_rss_delta_bytes": float(max(value - baseline_rss for value in rss_points)),
        "rss_after_finish_bytes": rss_after_finish,
        "finalization_rss_delta_bytes": rss_after_finish - final_rss,
    }


def _run_soak_scenario(
    spec: RuntimeSpec,
    scenario_dir: Path,
    *,
    profile: str,
) -> dict[str, Any]:
    _prepare_directory(scenario_dir)
    gc.collect()
    session = spec.factory(scenario_dir, spec.default_interval, None)
    session_started = False
    finish_attempted = False
    session_report: dict[str, Any] | None = None
    try:
        session.start()
        session_started = True
        sample_count = max(
            1,
            int(
                round(
                    (PROFILE_EQUIVALENT_HOURS[profile] * 3600.0) / spec.default_interval
                )
            ),
        )
        equivalent_seconds = sample_count * spec.default_interval
        checkpoints = _soak_rss_checkpoints(sample_count)
        pending_checkpoints = set(checkpoints)
        rss_points: list[int] = []

        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        for index in range(sample_count):
            session.emit_sample(index)
            if index in pending_checkpoints:
                rss_points.append(_process_rss_bytes())
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start

        finish_attempted = True
        session_report = session.finish()
    except Exception:
        if session_started and not finish_attempted:
            with suppress(Exception):
                session.finish()
        raise

    assert session_report is not None
    rss_after_finish = _process_rss_bytes()
    stats = dict(session_report["stats"])
    summary: dict[str, Any] = {
        "name": f"{spec.name}_soak",
        "profile": profile,
        "sample_count": sample_count,
        "equivalent_seconds": equivalent_seconds,
        "equivalent_hours": PROFILE_EQUIVALENT_HOURS[profile],
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "collector_failure_event_count": int(
            session_report["collector_failure_event_count"]
        ),
        **_soak_rss_metrics(
            rss_points,
            checkpoints=checkpoints,
            sample_count=sample_count,
            interval_seconds=spec.default_interval,
            rss_after_finish=rss_after_finish,
        ),
        "stats": stats,
        "artifact_dir": str(scenario_dir),
        "output_path": str(session_report["output_path"]),
    }
    return _finalize_scenario_summary(scenario_dir, summary)


def _run_retention_validation(
    spec: RuntimeSpec,
    scenario_dir: Path,
) -> dict[str, Any]:
    sample_limit = int(DEFAULT_RETENTION_VALIDATION["sample_limit"])
    overrides: SinkOverrides = {
        "flush_every_events": int(DEFAULT_RETENTION_VALIDATION["flush_every_events"]),
        "flush_every_seconds": float(
            DEFAULT_RETENTION_VALIDATION["flush_every_seconds"]
        ),
        "rollover_max_bytes": int(DEFAULT_RETENTION_VALIDATION["rollover_max_bytes"]),
        "rollover_max_events": int(DEFAULT_RETENTION_VALIDATION["rollover_max_events"]),
        "retention_max_files": int(DEFAULT_RETENTION_VALIDATION["retention_max_files"]),
        "retention_max_total_bytes": int(
            DEFAULT_RETENTION_VALIDATION["retention_max_total_bytes"]
        ),
    }
    _prepare_directory(scenario_dir)
    gc.collect()
    session = spec.factory(scenario_dir, spec.default_interval, overrides)
    session_started = False
    finish_attempted = False
    session_report: dict[str, Any] | None = None
    try:
        session.start()
        session_started = True

        last_stats: dict[str, Any] = {}
        for index in range(sample_limit):
            session.emit_sample(index)
            if index % 25 == 0:
                # All session implementations surface up-to-date sink stats through
                # tracker statistics, so we can inspect progress without finalizing.
                if hasattr(session, "tracker"):
                    last_stats = dict(getattr(session, "tracker").get_statistics())
                    if (
                        int(last_stats.get("rollover_count", 0)) > 0
                        and int(last_stats.get("pruned_segment_count", 0)) > 0
                    ):
                        break

        finish_attempted = True
        session_report = session.finish()
    except Exception:
        if session_started and not finish_attempted:
            with suppress(Exception):
                session.finish()
        raise

    assert session_report is not None
    stats = dict(session_report["stats"])
    checks = {
        "rollover_observed": int(stats.get("rollover_count", 0)) > 0,
        "pruning_observed": int(stats.get("pruned_segment_count", 0)) > 0,
        "retained_files_bounded": int(stats.get("final_retained_files", 0))
        <= int(overrides["retention_max_files"]),
        "retained_bytes_bounded": int(stats.get("final_retained_bytes", 0))
        <= int(overrides["retention_max_total_bytes"]),
    }
    result = {
        "sample_limit": sample_limit,
        "stats": stats,
        "checks": checks,
        "passed": all(checks.values()),
        "artifact_dir": str(scenario_dir),
        "output_path": str(session_report["output_path"]),
    }
    return _finalize_scenario_summary(scenario_dir, result)


def _runtime_metric_key(runtime_name: str, metric_name: str) -> str:
    return f"{runtime_name}.{metric_name}"


def _flatten_metrics(
    runtime_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for runtime_name, runtime_report in runtime_reports.items():
        if runtime_report.get("status") != "ok":
            continue
        overhead = runtime_report.get("overhead")
        if isinstance(overhead, Mapping):
            runtime_metrics = overhead.get("metrics", {})
            if isinstance(runtime_metrics, Mapping):
                for metric_name, value in runtime_metrics.items():
                    metrics[_runtime_metric_key(runtime_name, metric_name)] = float(
                        value
                    )
        soak = runtime_report.get("soak")
        if isinstance(soak, Mapping):
            for metric_name in (
                "rss_growth_per_24h_equiv",
                "max_rss_delta_bytes",
                "final_retained_bytes",
                "final_retained_files",
                "collector_failure_event_count",
                "history_dropped_events",
                "history_dropped_samples",
                "history_dropped_alerts",
                "rollover_count",
                "pruned_segment_count",
                "pruned_bytes",
            ):
                if metric_name in soak:
                    metrics[_runtime_metric_key(runtime_name, metric_name)] = float(
                        soak[metric_name]
                    )
    return metrics


def _normalize_comparison_config(
    raw_config: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    required = {
        "profile",
        "mode",
        "iterations",
        "allocation_kb",
        "profile_equivalent_hours",
        "runtimes",
        "retention_validation",
    }
    missing = sorted(required.difference(raw_config))
    if missing:
        raise ValueError(f"{label} missing config keys: {', '.join(missing)}")

    raw_runtimes = raw_config["runtimes"]
    if not isinstance(raw_runtimes, Mapping):
        raise ValueError(f"{label} runtimes must be a mapping")
    runtimes: dict[str, dict[str, int | float]] = {}
    for runtime_name, raw_runtime in raw_runtimes.items():
        if not isinstance(runtime_name, str) or not isinstance(raw_runtime, Mapping):
            raise ValueError(f"{label} contains invalid runtime config")
        runtimes[runtime_name] = {
            "default_interval": float(raw_runtime["default_interval"]),
            "overhead_sample_count": int(raw_runtime["overhead_sample_count"]),
            "soak_sample_count": int(raw_runtime["soak_sample_count"]),
        }

    raw_retention = raw_config["retention_validation"]
    if not isinstance(raw_retention, Mapping):
        raise ValueError(f"{label} retention_validation must be a mapping")
    retention_validation = {
        "flush_every_events": int(raw_retention["flush_every_events"]),
        "flush_every_seconds": float(raw_retention["flush_every_seconds"]),
        "rollover_max_bytes": int(raw_retention["rollover_max_bytes"]),
        "rollover_max_events": int(raw_retention["rollover_max_events"]),
        "retention_max_files": int(raw_retention["retention_max_files"]),
        "retention_max_total_bytes": int(raw_retention["retention_max_total_bytes"]),
        "sample_limit": int(raw_retention["sample_limit"]),
    }
    return {
        "profile": str(raw_config["profile"]),
        "mode": str(raw_config["mode"]),
        "iterations": int(raw_config["iterations"]),
        "allocation_kb": int(raw_config["allocation_kb"]),
        "profile_equivalent_hours": float(raw_config["profile_equivalent_hours"]),
        "runtimes": dict(sorted(runtimes.items())),
        "retention_validation": retention_validation,
    }


def _metric_values_from_mapping(
    raw_values: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, float]:
    if not raw_values:
        raise ValueError(f"{label} must contain at least one metric")
    return {str(key): float(value) for key, value in raw_values.items()}


def load_budget_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in {None, REPORT_VERSION}:
        raise ValueError(
            f"Budget file version must be {REPORT_VERSION}, "
            f"found {payload.get('version')!r}"
        )
    budgets_obj = payload.get("budgets", payload)
    if not isinstance(budgets_obj, Mapping):
        raise ValueError("Budget file missing budgets mapping")
    return _metric_values_from_mapping(budgets_obj, label="Budget file")


def load_regression_baseline(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != REPORT_VERSION:
        raise ValueError(
            f"Baseline file version must be {REPORT_VERSION}, "
            f"found {payload.get('version')!r}"
        )
    raw_config = payload.get("config")
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Baseline file missing config mapping")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("Baseline file missing metrics mapping")
    return {
        "version": REPORT_VERSION,
        "config": _normalize_comparison_config(raw_config, label="Baseline file"),
        "metrics": _metric_values_from_mapping(raw_metrics, label="Baseline file"),
    }


def load_regression_tolerances(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != REPORT_VERSION:
        raise ValueError(
            f"Tolerance file version must be {REPORT_VERSION}, "
            f"found {payload.get('version')!r}"
        )
    tolerances_obj = payload.get("tolerances", payload)
    if not isinstance(tolerances_obj, Mapping):
        raise ValueError("Tolerance file missing tolerances mapping")
    return _metric_values_from_mapping(tolerances_obj, label="Tolerance file")


def evaluate_budgets(
    metrics: Mapping[str, float],
    budgets: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    missing = sorted(set(metrics).difference(budgets))
    if missing:
        raise ValueError(f"Budget file missing metric keys: {', '.join(missing)}")
    checks: dict[str, dict[str, Any]] = {}
    for metric_key, value in metrics.items():
        max_allowed = float(budgets[metric_key])
        checks[metric_key] = {
            "value": float(value),
            "max_allowed": max_allowed,
            "passed": float(value) <= max_allowed,
        }
    return checks


def evaluate_regressions(
    metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    tolerances: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    missing_baseline = sorted(set(metrics).difference(baseline_metrics))
    missing_tolerances = sorted(set(metrics).difference(tolerances))
    if missing_baseline:
        raise ValueError(
            "Baseline file missing metric keys: " + ", ".join(missing_baseline)
        )
    if missing_tolerances:
        raise ValueError(
            "Tolerance file missing metric keys: " + ", ".join(missing_tolerances)
        )
    checks: dict[str, dict[str, Any]] = {}
    for metric_key, value in metrics.items():
        baseline_value = float(baseline_metrics[metric_key])
        max_regression = float(tolerances[metric_key])
        delta = float(value) - baseline_value
        checks[metric_key] = {
            "current_value": float(value),
            "baseline_value": baseline_value,
            "delta": delta,
            "max_regression": max_regression,
            "passed": delta <= max_regression,
        }
    return checks


def validate_regression_config(
    current_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
) -> None:
    normalized_current = _normalize_comparison_config(
        current_config, label="Current benchmark config"
    )
    normalized_baseline = _normalize_comparison_config(
        baseline_config, label="Baseline file"
    )
    if normalized_current != normalized_baseline:
        raise ValueError(
            "Baseline config mismatch: "
            f"current={normalized_current!r}, baseline={normalized_baseline!r}"
        )


def format_regression_summary(report: Mapping[str, Any]) -> list[str]:
    checks = report.get("regression_checks", {})
    if not isinstance(checks, Mapping):
        return []
    lines: list[str] = []
    for metric_key in sorted(checks):
        raw_check = checks.get(metric_key)
        if not isinstance(raw_check, Mapping):
            continue
        status = "PASS" if bool(raw_check["passed"]) else "FAIL"
        lines.append(
            f"{metric_key}: current={_format_metric_value(metric_key, float(raw_check['current_value']))} "
            f"baseline={_format_metric_value(metric_key, float(raw_check['baseline_value']))} "
            f"delta={_format_metric_value(metric_key, float(raw_check['delta']))} "
            f"allowed={_format_metric_value(metric_key, float(raw_check['max_regression']))} "
            f"[{status}]"
        )
    return lines


def format_budget_summary(report: Mapping[str, Any]) -> list[str]:
    checks = report.get("budget_checks", {})
    if not isinstance(checks, Mapping):
        return []
    lines: list[str] = []
    for metric_key in sorted(checks):
        raw_check = checks.get(metric_key)
        if not isinstance(raw_check, Mapping):
            continue
        status = "PASS" if bool(raw_check["passed"]) else "FAIL"
        lines.append(
            f"{metric_key}: value={_format_metric_value(metric_key, float(raw_check['value']))} "
            f"max={_format_metric_value(metric_key, float(raw_check['max_allowed']))} "
            f"[{status}]"
        )
    return lines


def _extract_runtime_diagnostics(runtime_report: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    soak = runtime_report.get("soak")
    if isinstance(soak, Mapping):
        for key in (
            "collector_health_status",
            "collector_failure_event_count",
            "rollover_count",
            "pruned_segment_count",
            "pruned_bytes",
            "final_retained_files",
            "final_retained_bytes",
            "history_retained_events",
            "history_dropped_events",
            "history_retained_samples",
            "history_dropped_samples",
            "history_retained_alerts",
            "history_dropped_alerts",
        ):
            if key in soak:
                diagnostics[key] = soak[key]
        return diagnostics

    overhead = runtime_report.get("overhead")
    if not isinstance(overhead, Mapping):
        return diagnostics

    scenarios = overhead.get("scenarios", {})
    if not isinstance(scenarios, Mapping):
        return diagnostics

    tracked_default = scenarios.get("tracked_default", {})
    if not isinstance(tracked_default, Mapping):
        return diagnostics

    stats = tracked_default.get("stats", {})
    if not isinstance(stats, Mapping):
        return diagnostics

    for key in (
        "collector_health_status",
        "rollover_count",
        "pruned_segment_count",
        "pruned_bytes",
        "final_retained_files",
        "final_retained_bytes",
        "history_retained_events",
        "history_dropped_events",
        "history_retained_samples",
        "history_dropped_samples",
        "history_retained_alerts",
        "history_dropped_alerts",
    ):
        if key in stats:
            diagnostics[key] = stats[key]
    diagnostics.setdefault(
        "collector_failure_event_count",
        int(tracked_default.get("collector_failure_event_count", 0)),
    )
    return diagnostics


def _runtime_context_suffix(runtime_report: Mapping[str, Any]) -> str:
    diagnostics = _extract_runtime_diagnostics(runtime_report)
    if not diagnostics:
        return ""
    parts: list[str] = []
    if "collector_health_status" in diagnostics:
        parts.append(f"collector={diagnostics['collector_health_status']}")
    if "collector_failure_event_count" in diagnostics:
        parts.append(
            "collector_failures=" f"{int(diagnostics['collector_failure_event_count'])}"
        )
    for key in (
        "rollover_count",
        "pruned_segment_count",
        "pruned_bytes",
        "final_retained_files",
        "final_retained_bytes",
        "history_retained_events",
        "history_dropped_events",
        "history_retained_samples",
        "history_dropped_samples",
        "history_retained_alerts",
        "history_dropped_alerts",
    ):
        if key in diagnostics:
            parts.append(f"{key}={diagnostics[key]}")
    if not parts:
        return ""
    return "; " + ", ".join(parts)


def _failure_diagnostics(report: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    runtimes = report.get("runtimes", {})
    runtime_reports: dict[str, Mapping[str, Any]] = {}
    if isinstance(runtimes, Mapping):
        for runtime_name, runtime_report in runtimes.items():
            if not isinstance(runtime_report, Mapping):
                continue
            runtime_reports[str(runtime_name)] = runtime_report
            if runtime_report.get("status") != "ok":
                failures.append(
                    f"{runtime_name}: unavailable - {runtime_report.get('reason', 'unknown')}"
                )
                continue
            soak = runtime_report.get("soak")
            if isinstance(soak, Mapping):
                if int(soak.get("collector_failure_event_count", 0)) > 0:
                    failures.append(
                        f"{runtime_name}: collector failures observed "
                        f"({int(soak['collector_failure_event_count'])})"
                        f"{_runtime_context_suffix(runtime_report)}"
                    )
                retention = soak.get("retention_validation", {})
                if isinstance(retention, Mapping):
                    for check_name, passed in retention.get("checks", {}).items():
                        if not bool(passed):
                            failures.append(
                                f"{runtime_name}: retention validation failed for {check_name}"
                                f"{_runtime_context_suffix(runtime_report)}"
                            )
                collector_health_status = str(
                    soak.get("collector_health_status", "healthy")
                )
                if collector_health_status != "healthy":
                    failures.append(
                        f"{runtime_name}: collector health is {collector_health_status}"
                        f"{_runtime_context_suffix(runtime_report)}"
                    )

    for section_name in ("budget_checks", "regression_checks"):
        checks = report.get(section_name, {})
        if not isinstance(checks, Mapping):
            continue
        for metric_key, raw_check in checks.items():
            if not isinstance(raw_check, Mapping) or bool(
                raw_check.get("passed", True)
            ):
                continue
            runtime_name = None
            if "." in str(metric_key):
                runtime_name, _ = str(metric_key).split(".", 1)
            runtime_report = (
                runtime_reports.get(runtime_name) if runtime_name is not None else None
            )
            context_suffix = (
                _runtime_context_suffix(runtime_report)
                if runtime_report is not None
                else ""
            )
            if section_name == "budget_checks":
                failures.append(
                    f"{metric_key}: {_format_metric_value(metric_key, float(raw_check['value']))} "
                    f"> {_format_metric_value(metric_key, float(raw_check['max_allowed']))}"
                    f"{context_suffix}"
                )
            else:
                failures.append(
                    f"{metric_key}: delta {_format_metric_value(metric_key, float(raw_check['delta']))} "
                    f"> {_format_metric_value(metric_key, float(raw_check['max_regression']))}"
                    f"{context_suffix}"
                )
    return failures


def _overhead_sample_count(spec: RuntimeSpec, iterations: int) -> int:
    samples = int(
        round(iterations * (REFERENCE_INTERVAL_SECONDS / spec.default_interval))
    )
    return max(1, min(max(1, iterations), samples))


def _runtime_config(
    profile: str, iterations: int, runtime_names: list[str]
) -> dict[str, Any]:
    runtimes: dict[str, dict[str, int | float]] = {}
    for runtime_name in runtime_names:
        spec = _RUNTIME_SPECS[runtime_name]
        soak_sample_count = max(
            1,
            int(
                round(
                    (PROFILE_EQUIVALENT_HOURS[profile] * 3600.0) / spec.default_interval
                )
            ),
        )
        runtimes[runtime_name] = {
            "default_interval": spec.default_interval,
            "overhead_sample_count": _overhead_sample_count(spec, iterations),
            "soak_sample_count": soak_sample_count,
        }
    return runtimes


def _run_runtime_report(
    spec: RuntimeSpec,
    runtime_dir: Path,
    *,
    profile: str,
    mode: str,
    iterations: int,
    allocation_kb: int,
    overhead_scratch_root: Optional[Path] = None,
) -> dict[str, Any]:
    runtime_report: dict[str, Any] = {
        "status": "ok",
        "default_interval": spec.default_interval,
    }
    if mode in {"overhead", "all"}:
        runtime_report["overhead"] = _run_overhead_report(
            spec,
            runtime_dir,
            iterations=iterations,
            allocation_kb=allocation_kb,
            scratch_root=overhead_scratch_root,
        )
    if mode in {"soak", "all"}:
        soak = _run_soak_scenario(
            spec,
            runtime_dir / "soak" / "default",
            profile=profile,
        )
        retention_validation = _run_retention_validation(
            spec,
            runtime_dir / "soak" / "retention_validation",
        )
        runtime_report["soak"] = {
            "sample_count": soak["sample_count"],
            "equivalent_seconds": soak["equivalent_seconds"],
            "equivalent_hours": soak["equivalent_hours"],
            "wall_seconds": soak["wall_seconds"],
            "cpu_seconds": soak["cpu_seconds"],
            "artifact_size_bytes": soak["artifact_size_bytes"],
            "rss_growth_per_24h_equiv": soak["rss_growth_per_24h_equiv"],
            "max_rss_delta_bytes": soak["max_rss_delta_bytes"],
            "rss_warmup_sample_count": soak["rss_warmup_sample_count"],
            "rss_measured_sample_count": soak["rss_measured_sample_count"],
            "finalization_rss_delta_bytes": soak["finalization_rss_delta_bytes"],
            "collector_failure_event_count": soak["collector_failure_event_count"],
            "history_dropped_events": int(
                soak["stats"].get("history_dropped_events", 0)
            ),
            "history_dropped_samples": int(
                soak["stats"].get("history_dropped_samples", 0)
            ),
            "history_dropped_alerts": int(
                soak["stats"].get("history_dropped_alerts", 0)
            ),
            "rollover_count": int(soak["stats"].get("rollover_count", 0)),
            "pruned_segment_count": int(soak["stats"].get("pruned_segment_count", 0)),
            "pruned_bytes": int(soak["stats"].get("pruned_bytes", 0)),
            "final_retained_files": int(soak["stats"].get("final_retained_files", 0)),
            "final_retained_bytes": int(soak["stats"].get("final_retained_bytes", 0)),
            "history_retained_events": int(
                soak["stats"].get("history_retained_events", 0)
            ),
            "history_retained_samples": int(
                soak["stats"].get("history_retained_samples", 0)
            ),
            "history_retained_alerts": int(
                soak["stats"].get("history_retained_alerts", 0)
            ),
            "collector_health_status": soak["stats"].get(
                "collector_health_status", "healthy"
            ),
            "retention_validation": retention_validation,
        }
    return runtime_report


def run_benchmark_harness(
    *,
    profile: str,
    mode: str,
    gate_mode: str,
    budgets_path: Optional[Path],
    baseline_path: Optional[Path],
    tolerances_path: Optional[Path],
    artifact_root: Path,
    output_path: Path,
    iterations: int = DEFAULT_ITERATIONS,
    allocation_kb: int = DEFAULT_ALLOCATION_KB,
    runtime_names: Optional[list[str]] = None,
    overhead_scratch_root: Optional[Path] = None,
) -> dict[str, Any]:
    if profile not in PROFILE_EQUIVALENT_HOURS:
        raise ValueError(f"Unsupported profile: {profile}")
    if mode not in {"overhead", "soak", "all"}:
        raise ValueError(f"Unsupported mode: {mode}")

    selected_runtime_names = runtime_names or list(_RUNTIME_SPECS.keys())
    missing_runtimes = [
        name for name in selected_runtime_names if name not in _RUNTIME_SPECS
    ]
    if missing_runtimes:
        raise ValueError(f"Unknown runtimes: {', '.join(sorted(missing_runtimes))}")

    artifact_root.mkdir(parents=True, exist_ok=True)
    if overhead_scratch_root is not None:
        overhead_scratch_root.mkdir(parents=True, exist_ok=True)
    runtime_reports: dict[str, dict[str, Any]] = {}
    for runtime_name in selected_runtime_names:
        runtime_dir = artifact_root / runtime_name
        spec = _RUNTIME_SPECS[runtime_name]
        try:
            runtime_reports[runtime_name] = _run_runtime_report(
                spec,
                runtime_dir,
                profile=profile,
                mode=mode,
                iterations=iterations,
                allocation_kb=allocation_kb,
                overhead_scratch_root=overhead_scratch_root,
            )
        except Exception as exc:
            runtime_reports[runtime_name] = {
                "status": "unavailable",
                "reason": str(exc),
            }

    config = {
        "profile": profile,
        "mode": mode,
        "iterations": iterations,
        "allocation_kb": allocation_kb,
        "profile_equivalent_hours": PROFILE_EQUIVALENT_HOURS[profile],
        "runtimes": _runtime_config(profile, iterations, selected_runtime_names),
        "retention_validation": dict(DEFAULT_RETENTION_VALIDATION),
        "overhead_scratch_root": (
            str(overhead_scratch_root) if overhead_scratch_root is not None else None
        ),
    }
    metrics = _flatten_metrics(runtime_reports)
    report: dict[str, Any] = {
        "version": REPORT_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": profile,
        "mode": mode,
        "gate_mode": gate_mode,
        "config": config,
        "runtimes": runtime_reports,
        "metrics": metrics,
    }

    if gate_mode == "budget":
        if budgets_path is None:
            raise ValueError("Budget gate mode requires a budgets path")
        budgets = load_budget_thresholds(budgets_path)
        budget_checks = evaluate_budgets(metrics, budgets)
        report["config"]["budgets_path"] = str(budgets_path)
        report["budgets"] = budgets
        report["budget_checks"] = budget_checks
        report["passed"] = all(
            bool(check["passed"]) for check in budget_checks.values()
        )
    elif gate_mode == "regression":
        if baseline_path is None or tolerances_path is None:
            raise ValueError(
                "Regression gate mode requires baseline and tolerance paths"
            )
        baseline = load_regression_baseline(baseline_path)
        validate_regression_config(config, baseline["config"])
        tolerances = load_regression_tolerances(tolerances_path)
        regression_checks = evaluate_regressions(
            metrics,
            baseline["metrics"],
            tolerances,
        )
        report["config"]["baseline_path"] = str(baseline_path)
        report["config"]["tolerances_path"] = str(tolerances_path)
        report["baseline"] = baseline
        report["tolerances"] = tolerances
        report["regression_checks"] = regression_checks
        report["passed"] = all(
            bool(check["passed"]) for check in regression_checks.values()
        )
    else:
        raise ValueError(f"Unsupported gate mode: {gate_mode}")

    failures = _failure_diagnostics(report)
    report["failure_diagnostics"] = failures
    report["passed"] = bool(report["passed"]) and not failures

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qualify always-on monitoring overhead and soak budgets.",
    )
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_EQUIVALENT_HOURS), default=DEFAULT_PROFILE
    )
    parser.add_argument(
        "--mode", choices=["overhead", "soak", "all"], default=DEFAULT_MODE
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--allocation-kb", type=int, default=DEFAULT_ALLOCATION_KB)
    parser.add_argument(
        "--gate-mode",
        choices=["budget", "regression"],
        default=DEFAULT_GATE_MODE,
        help="Gate mode to evaluate when --check is used.",
    )
    parser.add_argument("--budgets", type=Path, default=DEFAULT_BUDGETS_PATH)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--tolerances", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--overhead-scratch-root",
        type=Path,
        default=None,
        help=(
            "Directory that holds overhead trials while they run, ideally on a "
            "RAM-backed filesystem such as /dev/shm so storage latency does not "
            "enter runtime_overhead_pct. Selected trials are promoted into "
            "--artifact-root."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return a non-zero exit code when any gate fails.",
    )
    args = parser.parse_args(argv)

    if (
        args.gate_mode == "regression"
        and args.profile != DEFAULT_PROFILE
        and (args.baseline is None or args.tolerances is None)
    ):
        raise ValueError(
            "Regression defaults are only checked in for the pr profile; "
            "pass --baseline and --tolerances explicitly for other profiles."
        )

    baseline_path = args.baseline or _default_runtime_baseline_path()
    tolerances_path = args.tolerances or _default_runtime_tolerances_path()
    report = run_benchmark_harness(
        profile=args.profile,
        mode=args.mode,
        iterations=args.iterations,
        allocation_kb=args.allocation_kb,
        gate_mode=args.gate_mode,
        budgets_path=args.budgets if args.gate_mode == "budget" else None,
        baseline_path=baseline_path if args.gate_mode == "regression" else None,
        tolerances_path=tolerances_path if args.gate_mode == "regression" else None,
        artifact_root=args.artifact_root,
        output_path=args.output,
        overhead_scratch_root=args.overhead_scratch_root,
    )

    print(f"Operability report written to: {args.output}")
    for line in format_regression_summary(report):
        print(line)
    for line in format_budget_summary(report):
        print(line)
    if report.get("failure_diagnostics"):
        print("Failures:")
        for failure in report["failure_diagnostics"]:
            print(f"- {failure}")
    print(f"Overall status: {'PASS' if report['passed'] else 'FAIL'}")

    if args.check and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
