"""OOM flight-recorder scenario for launch QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from examples.common.formatting import print_header, print_kv, print_section
from gpumemprof.oom_flight_recorder import classify_oom_exception
from gpumemprof.utils import get_system_info

if TYPE_CHECKING:
    import torch

    from gpumemprof.tracker import MemoryTracker


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "examples" / "scenarios" / "oom_flight_recorder"
)


def _tracker_device_for_backend(backend: str) -> str | int:
    if backend == "mps":
        return "mps"
    return 0


def _simulate_oom(tracker: MemoryTracker) -> tuple[str, str]:
    try:
        with tracker.capture_oom(
            context="examples.scenarios.oom_flight_recorder.simulated",
            metadata={"scenario_mode": "simulated"},
        ):
            raise RuntimeError("simulated out of memory for demo")
    except RuntimeError as exc:
        return type(exc).__name__, str(exc)
    raise RuntimeError("Simulated OOM flow did not raise as expected")


def _stress_attempt_oom(
    tracker: MemoryTracker,
    *,
    backend: str,
    max_total_mb: int,
    step_mb: int,
) -> tuple[str, str, bool]:
    import torch

    tensors: list[torch.Tensor] = []
    target_device = torch.device("mps" if backend == "mps" else "cuda")
    allocated_mb = 0
    caught_oom = False
    exc_name = ""
    exc_message = ""

    try:
        with tracker.capture_oom(
            context="examples.scenarios.oom_flight_recorder.stress",
            metadata={
                "scenario_mode": "stress",
                "max_total_mb": max_total_mb,
                "step_mb": step_mb,
            },
        ):
            while allocated_mb < max_total_mb:
                elements = int(step_mb * 1024 * 1024 / 4)
                block = torch.randn(elements, device=target_device)
                tensors.append(block)
                allocated_mb += step_mb
    except Exception as exc:  # noqa: BLE001
        classified = classify_oom_exception(exc)
        caught_oom = bool(classified.is_oom)
        exc_name = type(exc).__name__
        exc_message = str(exc)
    finally:
        tensors.clear()
        if backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
        elif backend == "mps":
            import torch.mps as torch_mps

            if hasattr(torch_mps, "empty_cache"):
                torch_mps.empty_cache()

    return exc_name, exc_message, caught_oom


def run_scenario(
    *,
    output_dir: Path,
    mode: str,
    sampling_interval: float,
    stress_max_mb: int,
    stress_step_mb: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "oom_scenario_summary.json"

    print_header("OOM Flight Recorder Scenario")
    print_kv("Mode", mode)
    print_kv("Output directory", output_dir)

    backend = str(get_system_info().get("detected_backend", "cpu"))
    if backend not in {"cuda", "rocm", "mps"}:
        summary: dict[str, object] = {
            "status": "SKIP",
            "reason": f"GPU tracker backend unavailable (detected_backend={backend})",
            "mode": mode,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print_kv("Status", summary["status"])
        print_kv("Reason", summary["reason"])
        print_kv("Summary report", summary_path)
        return summary

    try:
        from gpumemprof.tracker import MemoryTracker
    except ImportError as exc:
        summary = {
            "status": "SKIP",
            "reason": f"GPU tracker dependencies unavailable: {exc}",
            "mode": mode,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print_kv("Status", summary["status"])
        print_kv("Reason", summary["reason"])
        print_kv("Summary report", summary_path)
        return summary

    tracker = MemoryTracker(
        device=_tracker_device_for_backend(backend),
        sampling_interval=sampling_interval,
        enable_oom_flight_recorder=True,
        oom_dump_dir=str(output_dir / "oom_dumps"),
        oom_max_dumps=3,
        oom_max_total_mb=128,
    )

    tracker.start_tracking()
    exc_name = ""
    exc_message = ""
    stress_got_oom = False
    try:
        print_section("Trigger OOM flow")
        if mode == "simulated":
            exc_name, exc_message = _simulate_oom(tracker)
            stress_got_oom = True
        else:
            exc_name, exc_message, stress_got_oom = _stress_attempt_oom(
                tracker,
                backend=backend,
                max_total_mb=stress_max_mb,
                step_mb=stress_step_mb,
            )
    finally:
        tracker.stop_tracking()

    dump_path = tracker.last_oom_dump_path
    dump_exists = bool(dump_path and Path(dump_path).exists())

    if dump_exists and (mode == "simulated" or stress_got_oom):
        status = "PASS"
        reason = None
    elif mode == "stress" and not stress_got_oom:
        status = "SKIP"
        reason = "Stress mode did not trigger OOM before configured cap."
    else:
        status = "FAIL"
        reason = f"OOM dump missing (dump_path={dump_path}, dump_exists={dump_exists})"

    summary = {
        "status": status,
        "reason": reason,
        "mode": mode,
        "detected_backend": backend,
        "exception_type": exc_name,
        "exception_message": exc_message,
        "oom_dump_path": dump_path,
        "oom_dump_exists": dump_exists,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_section("Scenario Summary")
    print_kv("Status", status)
    print_kv("Backend", backend)
    print_kv("Exception type", exc_name or "N/A")
    print_kv("OOM dump", dump_path or "N/A")
    if reason:
        print_kv("Reason", reason)
    print_kv("Summary report", summary_path)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OOM flight-recorder coverage scenarios.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulated", "stress"],
        default="simulated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--sampling-interval", type=float, default=0.1)
    parser.add_argument("--stress-max-mb", type=int, default=1024)
    parser.add_argument("--stress-step-mb", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sampling_interval <= 0:
        raise SystemExit("--sampling-interval must be > 0")
    if args.stress_max_mb <= 0:
        raise SystemExit("--stress-max-mb must be > 0")
    if args.stress_step_mb <= 0:
        raise SystemExit("--stress-step-mb must be > 0")

    run_scenario(
        output_dir=args.output_dir,
        mode=args.mode,
        sampling_interval=args.sampling_interval,
        stress_max_mb=args.stress_max_mb,
        stress_step_mb=args.stress_step_mb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
