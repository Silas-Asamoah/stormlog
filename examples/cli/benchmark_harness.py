"""Benchmark harness for profiling overhead and artifact-size budget checks."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from gpumemprof.cpu_profiler import CPUMemoryTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGETS_PATH = REPO_ROOT / "docs" / "benchmarks" / "v0.2_budgets.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "latest.json"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "benchmarks" / "scenarios"

_BUDGET_KEY_BY_METRIC: Dict[str, str] = {
    "runtime_overhead_pct": "runtime_overhead_pct_max",
    "cpu_overhead_pct": "cpu_overhead_pct_max",
    "sampling_impact_pct": "sampling_impact_pct_max",
    "artifact_growth_bytes": "artifact_growth_bytes_max",
}


@dataclass
class ScenarioResult:
    """Single benchmark scenario output."""

    name: str
    wall_seconds: float
    cpu_seconds: float
    checksum: int
    event_count: int
    peak_memory_bytes: int
    artifact_size_bytes: int
    artifact_dir: str


def _run_workload(iterations: int, allocation_kb: int) -> int:
    """Run a deterministic CPU workload with allocation churn."""
    checksum = 0
    block_bytes = max(1, allocation_kb) * 1024

    for step in range(max(1, iterations)):
        payload = bytearray(block_bytes)
        marker = (step * 17) % 251
        for offset in range(0, len(payload), 4096):
            payload[offset] = marker
            checksum += payload[offset]
        checksum ^= len(payload)

    return checksum


def _directory_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _run_scenario(
    name: str,
    *,
    iterations: int,
    allocation_kb: int,
    artifact_root: Path,
    sampling_interval: Optional[float],
) -> ScenarioResult:
    scenario_dir = artifact_root / name
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    tracker: Optional[CPUMemoryTracker] = None
    if sampling_interval is not None:
        tracker = CPUMemoryTracker(
            sampling_interval=sampling_interval,
            enable_alerts=False,
        )
        tracker.start_tracking()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    checksum = _run_workload(iterations=iterations, allocation_kb=allocation_kb)
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start

    event_count = 0
    peak_memory_bytes = 0

    if tracker is not None:
        tracker.stop_tracking()
        events = tracker.get_events()
        stats = tracker.get_statistics()
        event_count = len(events)
        peak_memory_bytes = int(stats.get("peak_memory", 0))
        tracker.export_events(str(scenario_dir / "events.json"), format="json")

    summary_path = scenario_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "name": name,
                "sampling_interval": sampling_interval,
                "iterations": iterations,
                "allocation_kb": allocation_kb,
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "checksum": checksum,
                "event_count": event_count,
                "peak_memory_bytes": peak_memory_bytes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return ScenarioResult(
        name=name,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        checksum=checksum,
        event_count=event_count,
        peak_memory_bytes=peak_memory_bytes,
        artifact_size_bytes=_directory_size_bytes(scenario_dir),
        artifact_dir=str(scenario_dir),
    )


def _pct_overhead(baseline: float, measured: float) -> float:
    if baseline <= 0:
        return 0.0 if measured <= 0 else float("inf")
    return max(0.0, ((measured - baseline) / baseline) * 100.0)


def load_budget_thresholds(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    budgets_obj = payload.get("budgets", payload)

    missing_keys = [
        key for key in _BUDGET_KEY_BY_METRIC.values() if key not in budgets_obj
    ]
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Budget file missing keys: {missing}")

    return {key: float(value) for key, value in budgets_obj.items()}


def evaluate_budgets(
    metrics: Dict[str, float],
    budgets: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for metric_key, budget_key in _BUDGET_KEY_BY_METRIC.items():
        value = float(metrics[metric_key])
        max_allowed = float(budgets[budget_key])
        checks[metric_key] = {
            "value": value,
            "max_allowed": max_allowed,
            "passed": value <= max_allowed,
        }
    return checks


def run_benchmark_harness(
    *,
    iterations: int,
    allocation_kb: int,
    default_interval: float,
    lowfreq_interval: float,
    budgets_path: Path,
    artifact_root: Path,
    output_path: Path,
) -> Dict[str, Any]:
    budgets = load_budget_thresholds(budgets_path)
    artifact_root.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "unprofiled": _run_scenario(
            "unprofiled",
            iterations=iterations,
            allocation_kb=allocation_kb,
            artifact_root=artifact_root,
            sampling_interval=None,
        ),
        "tracked_default": _run_scenario(
            "tracked_default",
            iterations=iterations,
            allocation_kb=allocation_kb,
            artifact_root=artifact_root,
            sampling_interval=default_interval,
        ),
        "tracked_lowfreq": _run_scenario(
            "tracked_lowfreq",
            iterations=iterations,
            allocation_kb=allocation_kb,
            artifact_root=artifact_root,
            sampling_interval=lowfreq_interval,
        ),
    }

    metrics = {
        "runtime_overhead_pct": _pct_overhead(
            scenarios["unprofiled"].wall_seconds,
            scenarios["tracked_default"].wall_seconds,
        ),
        "cpu_overhead_pct": _pct_overhead(
            scenarios["unprofiled"].cpu_seconds,
            scenarios["tracked_default"].cpu_seconds,
        ),
        "sampling_impact_pct": _pct_overhead(
            scenarios["tracked_lowfreq"].wall_seconds,
            scenarios["tracked_default"].wall_seconds,
        ),
        "artifact_growth_bytes": float(
            max(
                0,
                scenarios["tracked_default"].artifact_size_bytes
                - scenarios["unprofiled"].artifact_size_bytes,
            )
        ),
    }

    budget_checks = evaluate_budgets(metrics, budgets)
    passed = all(bool(check["passed"]) for check in budget_checks.values())

    report: Dict[str, Any] = {
        "version": "v0.2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "iterations": iterations,
            "allocation_kb": allocation_kb,
            "default_interval": default_interval,
            "lowfreq_interval": lowfreq_interval,
            "budgets_path": str(budgets_path),
            "artifact_root": str(artifact_root),
        },
        "budgets": budgets,
        "scenarios": {name: asdict(result) for name, result in scenarios.items()},
        "metrics": metrics,
        "budget_checks": budget_checks,
        "passed": passed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark profiler overhead and enforce v0.2 budgets.",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--allocation-kb", type=int, default=512)
    parser.add_argument("--default-interval", type=float, default=0.1)
    parser.add_argument("--lowfreq-interval", type=float, default=0.5)
    parser.add_argument("--budgets", type=Path, default=DEFAULT_BUDGETS_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return a non-zero exit code when any budget is violated.",
    )
    args = parser.parse_args(argv)

    report = run_benchmark_harness(
        iterations=args.iterations,
        allocation_kb=args.allocation_kb,
        default_interval=args.default_interval,
        lowfreq_interval=args.lowfreq_interval,
        budgets_path=args.budgets,
        artifact_root=args.artifact_root,
        output_path=args.output,
    )

    print(f"Benchmark report written to: {args.output}")
    print(f"Runtime overhead: {report['metrics']['runtime_overhead_pct']:.2f}%")
    print(f"CPU overhead: {report['metrics']['cpu_overhead_pct']:.2f}%")
    print(f"Sampling impact: {report['metrics']['sampling_impact_pct']:.2f}%")
    print(f"Artifact growth: {report['metrics']['artifact_growth_bytes']:.0f} bytes")
    print(f"Budget status: {'PASS' if report['passed'] else 'FAIL'}")

    if args.check and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
