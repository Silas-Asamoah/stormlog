"""CPU telemetry scenario for launch QA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from examples.common.formatting import print_header, print_kv, print_section
from gpumemprof.cpu_profiler import CPUMemoryProfiler, CPUMemoryTracker
from gpumemprof.telemetry import validate_telemetry_record

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "examples" / "scenarios" / "cpu_telemetry"
)


def _run_cpu_workload(duration_s: float) -> int:
    """Allocate and mutate memory to generate tracker events."""
    chunks: list[bytearray] = []
    start = time.time()
    checksum = 0

    while (time.time() - start) < duration_s:
        payload = bytearray(1024 * 1024)
        for idx in range(0, len(payload), 4096):
            payload[idx] = (idx // 4096) % 251
            checksum += payload[idx]
        chunks.append(payload)
        if len(chunks) > 8:
            chunks.pop(0)
        time.sleep(0.05)

    return checksum


def run_scenario(
    *,
    output_dir: Path,
    tracker_interval: float,
    workload_duration_s: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cpu_tracker_events.csv"
    json_path = output_dir / "cpu_tracker_events.json"
    profile_summary_path = output_dir / "cpu_profiler_summary.json"

    print_header("CPU Telemetry Scenario")
    print_kv("Output directory", output_dir)
    print_kv("Tracker interval (s)", tracker_interval)
    print_kv("Workload duration (s)", workload_duration_s)

    profiler = CPUMemoryProfiler()
    tracker = CPUMemoryTracker(sampling_interval=tracker_interval)

    print_section("Running workload")
    tracker.start_tracking()
    with profiler.profile_context("cpu_workload_context"):
        profile_result = profiler.profile_function(
            _run_cpu_workload,
            workload_duration_s,
        )
    tracker.stop_tracking()

    tracker.export_events(str(csv_path), format="csv")
    tracker.export_events(str(json_path), format="json")

    events = json.loads(json_path.read_text(encoding="utf-8"))
    for event in events:
        validate_telemetry_record(event)

    summary = {
        "profile_name": profile_result.name,
        "profile_duration_s": profile_result.duration,
        "profile_peak_rss_bytes": profile_result.peak_rss,
        "profile_memory_diff_bytes": profile_result.memory_diff(),
        "tracker_stats": tracker.get_statistics(),
        "exported_event_count": len(events),
        "exports": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
    profile_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print_section("Scenario Summary")
    print_kv("Profiler duration (s)", f"{profile_result.duration:.3f}")
    print_kv("Profiler memory diff (bytes)", profile_result.memory_diff())
    print_kv("Exported events", len(events))
    print_kv("JSON telemetry", json_path)
    print_kv("CSV telemetry", csv_path)
    print_kv("Summary report", profile_summary_path)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CPU telemetry scenario for launch QA.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--tracker-interval", type=float, default=0.1)
    parser.add_argument("--workload-duration", type=float, default=1.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.tracker_interval <= 0:
        raise SystemExit("--tracker-interval must be > 0")
    if args.workload_duration <= 0:
        raise SystemExit("--workload-duration must be > 0")

    run_scenario(
        output_dir=args.output_dir,
        tracker_interval=args.tracker_interval,
        workload_duration_s=args.workload_duration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
