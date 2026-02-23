"""MPS telemetry scenario for launch QA."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from examples.common.formatting import print_header, print_kv, print_section
from gpumemprof.telemetry import validate_telemetry_record
from gpumemprof.utils import get_system_info

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "examples" / "scenarios" / "mps_telemetry"
)


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def run_scenario(
    *,
    output_dir: Path,
    duration_s: float,
    interval_s: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_dir / "mps_track_events.json"
    analysis_path = output_dir / "mps_track_analysis.txt"
    summary_path = output_dir / "mps_scenario_summary.json"

    print_header("MPS Telemetry Scenario")
    print_kv("Output directory", output_dir)

    backend = str(get_system_info().get("detected_backend", "cpu"))
    if backend != "mps":
        summary: dict[str, object] = {
            "status": "SKIP",
            "reason": f"MPS backend unavailable (detected_backend={backend})",
            "exports": {},
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print_kv("Status", summary["status"])
        print_kv("Reason", summary["reason"])
        print_kv("Summary report", summary_path)
        return summary

    print_section("Track")
    track_cmd = [
        "gpumemprof",
        "track",
        "--duration",
        str(duration_s),
        "--interval",
        str(interval_s),
        "--format",
        "json",
        "--output",
        str(telemetry_path),
    ]
    track_result = _run_command(track_cmd)
    if track_result.returncode != 0:
        raise RuntimeError(
            f"Track command failed ({track_result.returncode}):\n{track_result.stderr}"
        )

    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    for record in payload:
        validate_telemetry_record(record)
    if payload and payload[0]["collector"] != "gpumemprof.mps_tracker":
        raise RuntimeError(
            f"Unexpected collector for MPS run: {payload[0]['collector']}"
        )

    print_section("Analyze")
    analyze_cmd = [
        "gpumemprof",
        "analyze",
        str(telemetry_path),
        "--format",
        "txt",
        "--output",
        str(analysis_path),
    ]
    analyze_result = _run_command(analyze_cmd)
    if analyze_result.returncode != 0:
        raise RuntimeError(
            f"Analyze command failed ({analyze_result.returncode}):\n{analyze_result.stderr}"
        )

    summary = {
        "status": "PASS",
        "detected_backend": backend,
        "event_count": len(payload),
        "collector": payload[0]["collector"] if payload else None,
        "exports": {
            "events_json": str(telemetry_path),
            "analysis_txt": str(analysis_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_section("Scenario Summary")
    print_kv("Status", summary["status"])
    print_kv("Event count", summary["event_count"])
    print_kv("Collector", summary["collector"])
    print_kv("Telemetry JSON", telemetry_path)
    print_kv("Analysis TXT", analysis_path)
    print_kv("Summary report", summary_path)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MPS telemetry scenario for launch QA.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration <= 0:
        raise SystemExit("--duration must be > 0")
    if args.interval <= 0:
        raise SystemExit("--interval must be > 0")
    run_scenario(
        output_dir=args.output_dir,
        duration_s=args.duration,
        interval_s=args.interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
