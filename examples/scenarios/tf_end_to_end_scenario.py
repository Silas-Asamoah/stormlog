"""TensorFlow telemetry and diagnose end-to-end scenario for launch QA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from examples.common.formatting import print_header, print_kv, print_section
from gpumemprof.telemetry import validate_telemetry_record

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "examples" / "scenarios" / "tf_end_to_end"
)


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def _tfmemprof_cmd(*args: str) -> list[str]:
    """Run tfmemprof via the current Python interpreter for env consistency."""
    return [sys.executable, "-m", "tfmemprof.cli", *args]


def _run_track_with_sigint(cmd: list[str], interrupt_after_s: float) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(interrupt_after_s)
    proc.send_signal(signal.SIGINT)
    return proc.wait(timeout=45)


def run_scenario(
    *,
    output_dir: Path,
    monitor_duration_s: float,
    monitor_interval_s: float,
    track_interrupt_after_s: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor_path = output_dir / "tf_monitor.json"
    track_path = output_dir / "tf_track.json"
    diagnose_dir = output_dir / "tf_diagnose"
    summary_path = output_dir / "tf_e2e_summary.json"

    print_header("TensorFlow End-to-End Scenario")
    print_kv("Output directory", output_dir)

    if importlib.util.find_spec("tensorflow") is None:
        summary: dict[str, object] = {
            "status": "SKIP",
            "reason": "TensorFlow is not installed.",
            "monitor_output": str(monitor_path),
            "track_output": str(track_path),
            "diagnose_output": str(diagnose_dir),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print_kv("Status", summary["status"])
        print_kv("Reason", summary["reason"])
        print_kv("Summary report", summary_path)
        return summary

    print_section("Monitor")
    monitor_cmd = _tfmemprof_cmd(
        "monitor",
        "--interval",
        str(monitor_interval_s),
        "--duration",
        str(monitor_duration_s),
        "--output",
        str(monitor_path),
    )
    monitor_result = _run_command(monitor_cmd)
    if monitor_result.returncode != 0:
        raise RuntimeError(
            f"Monitor command failed ({monitor_result.returncode}):\n{monitor_result.stderr}"
        )

    print_section("Track")
    track_cmd = _tfmemprof_cmd(
        "track",
        "--interval",
        str(monitor_interval_s),
        "--threshold",
        "4096",
        "--output",
        str(track_path),
    )
    track_exit_code = _run_track_with_sigint(track_cmd, track_interrupt_after_s)
    if track_exit_code != 0 and not track_path.exists():
        raise RuntimeError(f"Track command failed ({track_exit_code})")

    payload = json.loads(track_path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    for event in events:
        validate_telemetry_record(event)

    print_section("Analyze")
    analyze_cmd = _tfmemprof_cmd(
        "analyze",
        "--input",
        str(monitor_path),
        "--detect-leaks",
        "--optimize",
    )
    analyze_result = _run_command(analyze_cmd)
    if analyze_result.returncode != 0:
        raise RuntimeError(
            f"Analyze command failed ({analyze_result.returncode}):\n{analyze_result.stderr}"
        )

    print_section("Diagnose")
    diagnose_cmd = _tfmemprof_cmd(
        "diagnose",
        "--duration",
        "0",
        "--output",
        str(diagnose_dir),
    )
    diagnose_result = _run_command(diagnose_cmd)
    if diagnose_result.returncode != 0:
        raise RuntimeError(
            f"Diagnose command failed ({diagnose_result.returncode}):\n{diagnose_result.stderr}"
        )
    manifest_path = diagnose_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Expected diagnose manifest not found: {manifest_path}")

    summary = {
        "status": "PASS",
        "monitor_output": str(monitor_path),
        "track_output": str(track_path),
        "diagnose_output": str(diagnose_dir),
        "track_event_count": len(events),
        "track_first_collector": events[0]["collector"] if events else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_section("Scenario Summary")
    print_kv("Status", summary["status"])
    print_kv("Track event count", summary["track_event_count"])
    print_kv("Track first collector", summary["track_first_collector"])
    print_kv("Monitor JSON", monitor_path)
    print_kv("Track JSON", track_path)
    print_kv("Diagnose directory", diagnose_dir)
    print_kv("Summary report", summary_path)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TensorFlow telemetry and diagnose scenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--monitor-duration", type=float, default=2.0)
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--track-interrupt-after", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.monitor_duration <= 0:
        raise SystemExit("--monitor-duration must be > 0")
    if args.monitor_interval <= 0:
        raise SystemExit("--monitor-interval must be > 0")
    if args.track_interrupt_after <= 0:
        raise SystemExit("--track-interrupt-after must be > 0")

    run_scenario(
        output_dir=args.output_dir,
        monitor_duration_s=args.monitor_duration,
        monitor_interval_s=args.monitor_interval,
        track_interrupt_after_s=args.track_interrupt_after,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
