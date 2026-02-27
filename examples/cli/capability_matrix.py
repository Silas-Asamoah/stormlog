"""Launch QA capability matrix orchestrator."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

from examples.common import print_header, print_kv, print_section
from examples.common.capability_matrix_utils import (
    CheckResult,
    result_to_dict,
    run_command,
    summarize_results,
    timed_result,
    write_report,
)
from gpumemprof.utils import get_system_info

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "examples" / "capability_matrix"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _target_flags(target: str, backend: str) -> tuple[bool, bool]:
    if target == "cpu":
        return True, False
    if target == "mps":
        return False, True
    if target == "both":
        return True, True
    # auto
    return True, backend == "mps"


def _load_scenario_runner(module_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, "run_scenario")


def _run_optional_scenario(module_name: str, **kwargs: object) -> Dict[str, object]:
    try:
        runner = _load_scenario_runner(module_name)
    except ImportError as exc:
        return {
            "status": "SKIP",
            "reason": f"{module_name} unavailable: {exc}",
        }
    result: Dict[str, object] = runner(**kwargs)
    return result


def _run_gpumemprof_diagnose(check_dir: Path) -> Dict[str, object]:
    out_dir = check_dir / "gpumemprof_diagnose"
    cmd = [
        "gpumemprof",
        "diagnose",
        "--duration",
        "0",
        "--output",
        str(out_dir),
    ]
    result = run_command(cmd)
    combined_output = f"{result.stdout}\n{result.stderr}"
    if result.returncode == 0:
        status = "PASS"
        reason = None
    elif "requires PyTorch" in combined_output:
        status = "SKIP"
        reason = "gpumemprof diagnose requires PyTorch in this environment."
    else:
        status = "FAIL"
        reason = None
    return {
        "status": status,
        "reason": reason,
        "returncode": result.returncode,
        "artifact_dir": str(out_dir),
        "manifest_exists": (out_dir / "manifest.json").exists(),
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def _run_benchmark_check(check_dir: Path, mode: str) -> Dict[str, object]:
    output = check_dir / "benchmark_report.json"
    artifact_root = check_dir / "benchmark_scenarios"
    cmd = [
        "python",
        "-m",
        "examples.cli.benchmark_harness",
        "--check",
        "--budgets",
        str(REPO_ROOT / "docs" / "benchmarks" / "v0.2_budgets.json"),
        "--output",
        str(output),
        "--artifact-root",
        str(artifact_root),
    ]
    if mode == "smoke":
        cmd.extend(["--iterations", "200"])
    result = run_command(cmd)
    status = "PASS" if result.returncode == 0 else "FAIL"
    return {
        "status": status,
        "returncode": result.returncode,
        "output": str(output),
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def _run_tui_smoke() -> Dict[str, object]:
    executable = shutil.which("stormlog")
    if executable is None:
        return {"status": "SKIP", "reason": "stormlog entrypoint not found"}

    try:
        import pexpect  # type: ignore[import-untyped, unused-ignore]
    except ModuleNotFoundError:
        return {"status": "SKIP", "reason": "pexpect is not installed"}

    env = os.environ.copy()
    env.setdefault("TERM", "xterm-256color")
    child = pexpect.spawn(executable, env=env, encoding="utf-8", timeout=25)
    try:
        child.expect(["Overview", "PyTorch"])
        child.send("q")
        child.expect(pexpect.EOF, timeout=10)
        return {
            "status": "PASS",
            "exitstatus": child.exitstatus,
            "signalstatus": child.signalstatus,
        }
    except Exception as exc:  # noqa: BLE001
        tail = (child.before or "")[-1200:]
        return {"status": "FAIL", "error": str(exc), "output_tail": tail}
    finally:
        if child.isalive():
            child.terminate(force=True)
        child.close()


def _run_full_extra_examples(check_dir: Path) -> List[CheckResult]:
    commands = [
        ("quickstart", ["python", "-m", "examples.cli.quickstart"]),
        ("pytorch_demo", ["python", "-m", "examples.basic.pytorch_demo"]),
        ("tensorflow_demo", ["python", "-m", "examples.basic.tensorflow_demo"]),
        ("advanced_tracking_demo", ["python", "-m", "examples.advanced.tracking_demo"]),
    ]
    results: List[CheckResult] = []
    for name, cmd in commands:
        started = time.perf_counter()
        proc = run_command(cmd)
        duration = time.perf_counter() - started
        status = "PASS" if proc.returncode == 0 else "FAIL"
        results.append(
            CheckResult(
                name=f"full_example:{name}",
                status=status,
                duration_s=duration,
                details={
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-1000:],
                    "stderr_tail": proc.stderr[-1000:],
                    "artifacts_dir": str(check_dir),
                },
            )
        )
    return results


def run_matrix(
    *,
    mode: str,
    target: str,
    oom_mode: str,
    artifacts_root: Path,
    skip_tui: bool,
) -> Dict[str, object]:
    system_info = get_system_info()
    backend = str(system_info.get("detected_backend", "cpu"))
    include_cpu, include_mps = _target_flags(target, backend)

    run_dir = artifacts_root / _timestamp_slug()
    run_dir.mkdir(parents=True, exist_ok=True)

    results: List[CheckResult] = []

    if include_cpu:
        results.append(
            timed_result(
                "scenario:cpu_telemetry",
                lambda: _load_scenario_runner(
                    "examples.scenarios.cpu_telemetry_scenario"
                )(
                    output_dir=run_dir / "cpu_telemetry",
                    tracker_interval=0.1,
                    workload_duration_s=1.5,
                ),
            )
        )
    else:
        results.append(
            CheckResult(
                name="scenario:cpu_telemetry",
                status="SKIP",
                duration_s=0.0,
                details={"reason": "CPU target disabled by --target"},
            )
        )

    if include_mps:
        results.append(
            timed_result(
                "scenario:mps_telemetry",
                lambda: _load_scenario_runner(
                    "examples.scenarios.mps_telemetry_scenario"
                )(
                    output_dir=run_dir / "mps_telemetry",
                    duration_s=2.0,
                    interval_s=0.5,
                ),
            )
        )
    else:
        results.append(
            CheckResult(
                name="scenario:mps_telemetry",
                status="SKIP",
                duration_s=0.0,
                details={"reason": "MPS target disabled by --target"},
            )
        )

    results.append(
        timed_result(
            "scenario:oom_flight_recorder",
            lambda: _run_optional_scenario(
                "examples.scenarios.oom_flight_recorder_scenario",
                output_dir=run_dir / "oom_flight_recorder",
                mode=oom_mode,
                sampling_interval=0.1,
                stress_max_mb=1024,
                stress_step_mb=64,
            ),
        )
    )

    results.append(
        timed_result(
            "scenario:tf_end_to_end",
            lambda: _run_optional_scenario(
                "examples.scenarios.tf_end_to_end_scenario",
                output_dir=run_dir / "tf_end_to_end",
                monitor_duration_s=2.0,
                monitor_interval_s=0.5,
                track_interrupt_after_s=8.0,
            ),
        )
    )

    results.append(
        timed_result(
            "cli:gpumemprof_diagnose",
            lambda: _run_gpumemprof_diagnose(run_dir),
        )
    )

    results.append(
        timed_result(
            "cli:benchmark_harness",
            lambda: _run_benchmark_check(run_dir, mode),
        )
    )

    if skip_tui:
        results.append(
            CheckResult(
                name="tui:pty_smoke",
                status="SKIP",
                duration_s=0.0,
                details={"reason": "--skip-tui specified"},
            )
        )
    else:
        results.append(timed_result("tui:pty_smoke", _run_tui_smoke))

    if mode == "full":
        results.extend(_run_full_extra_examples(run_dir))

    summary = summarize_results(results)
    report: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "target": target,
        "oom_mode": oom_mode,
        "detected_backend": backend,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "run_dir": str(run_dir),
        "summary": summary,
        "checks": [result_to_dict(result) for result in results],
    }
    write_report(run_dir / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run launch readiness capability checks for Stormlog.",
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--target", choices=["auto", "cpu", "mps", "both"], default="auto"
    )
    parser.add_argument(
        "--oom-mode", choices=["simulated", "stress"], default="simulated"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help=f"Report root (default: {DEFAULT_ARTIFACTS_ROOT})",
    )
    parser.add_argument("--skip-tui", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print_header("Capability Matrix")
    print_kv("Mode", args.mode)
    print_kv("Target", args.target)
    print_kv("OOM mode", args.oom_mode)
    print_kv("Artifacts root", args.artifacts_dir)
    print_kv("Skip TUI", args.skip_tui)

    report = run_matrix(
        mode=args.mode,
        target=args.target,
        oom_mode=args.oom_mode,
        artifacts_root=args.artifacts_dir,
        skip_tui=args.skip_tui,
    )

    print_section("Summary")
    summary = cast(Dict[str, int], report["summary"])
    print_kv("PASS", summary["PASS"])
    print_kv("SKIP", summary["SKIP"])
    print_kv("FAIL", summary["FAIL"])
    print_kv("Report", Path(cast(str, report["run_dir"])) / "report.json")

    return 1 if summary["FAIL"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
