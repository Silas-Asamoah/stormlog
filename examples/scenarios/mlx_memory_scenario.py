"""Reproducible fake contracts or required native Metal qualification.

Fake mode validates integration contracts only. Hardware mode fails if the
native runtime is unavailable; it never substitutes simulated performance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from stormlog import __version__
from stormlog.mlx import MemoryTracker, MLXMemoryProfiler
from stormlog.mlx.analyzer import analyze
from stormlog.mlx.runtime import MLXRuntime
from stormlog.system_info import get_system_info


class ScenarioRuntime:
    """Deterministic synthetic counters for CPU-only contract qualification."""

    version = "0.32.3-fake"
    device_info = {"device_name": "Synthetic contract runtime"}
    device_info_error = None

    def __init__(self) -> None:
        self.active = 0
        self.cache = 0
        self.peak = 0

    def supports(self, name: str) -> bool:
        return name in {
            "get_active_memory",
            "get_cache_memory",
            "get_peak_memory",
            "get_memory_limit",
            "reset_peak_memory",
        }

    def read_bytes(self, name: str) -> int:
        return {
            "get_active_memory": self.active,
            "get_cache_memory": self.cache,
            "get_peak_memory": self.peak,
            "get_memory_limit": 1024**2,
        }[name]

    def evaluate(self, *roots: Any) -> None:
        self.active = 4096
        self.cache = 1024
        self.peak = max(self.peak, self.active)

    def synchronize(self, streams: tuple[Any, ...]) -> None:
        pass

    def reset_peak(self) -> None:
        self.peak = 0

    def default_stream(self) -> str:
        return "fake"


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    files = [*sorted((root / "stormlog/mlx").glob("*.py")), Path(__file__).resolve()]
    return {
        "revision": revision,
        "stormlog_version": __version__,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }


def _capture_command(args: list[str], hardware: bool) -> list[str]:
    if hardware:
        return [sys.executable, "-m", "stormlog.mlx.cli", *args]
    code = "from examples.scenarios.mlx_memory_scenario import ScenarioRuntime\nimport stormlog.mlx.artifacts as a\na.MLXRuntime = ScenarioRuntime\nfrom stormlog.mlx.cli import main\nimport sys\nraise SystemExit(main(sys.argv[1:]))"
    return [sys.executable, "-c", code, *args]


def _failure_roundtrips(output: Path, hardware: bool) -> list[dict[str, Any]]:
    cases = [
        ("script_exit", "raise SystemExit(7)", 1),
        ("unexpected", "raise RuntimeError('qualification failure')", 1),
        (
            "oom_signature",
            "raise RuntimeError('[malloc] Unable to allocate 4096 bytes.')",
            3,
        ),
        ("interrupt", "import os, signal; os.kill(os.getpid(), signal.SIGINT)", 130),
    ]
    records = []
    for name, body, expected in cases:
        script = output / (name + ".py")
        prefix = (
            "import mlx.core as mx\nx = mx.ones((128,128))\nmx.eval(x)\n"
            if hardware
            else ""
        )
        script.write_text(prefix + body + "\n")
        command = _capture_command(
            ["run", "--json", "--output", str(output / name), "--", str(script)],
            hardware,
        )
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        assert proc.returncode == expected, proc.stdout + proc.stderr
        payload = json.loads(proc.stdout)
        directory = Path(payload["artifact_dir"])
        report = json.loads((directory / "report.json").read_text())
        assert report["verdict"]["exit_code"] == expected
        analysis = analyze(directory)
        assert analysis["session"]["status"] in ("incomplete", "interrupted")
        records.append(
            {
                "case": name,
                "exit_code": expected,
                "session_status": analysis["session"]["status"],
                "capture": str(directory),
                "oom_is_simulated": name == "oom_signature",
            }
        )
    return records


def _roundtrip(output: Path, hardware: bool) -> dict[str, Any]:
    from stormlog import query

    script = output / "workload.py"
    body = "import time\n"
    if hardware:
        body += "import mlx.core as mx\nx = mx.ones((128, 128))\nmx.eval(x)\n"
        body += "for _ in range(20):\n    y = x @ x.T\n    mx.eval(y)\n    time.sleep(.005)\n"
    else:
        body += "time.sleep(.08)\n"
    script.write_text(body)
    args = [
        "run",
        "--json",
        "--interval",
        ".002",
        "--max-history",
        "3",
        "--output",
        str(output / "capture"),
        "--",
        str(script),
    ]
    command = _capture_command(args, hardware)
    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=30, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    directory = Path(json.loads(proc.stdout)["artifact_dir"])
    analysis = analyze(directory)
    tracking = json.loads((directory / "tracking.json").read_text())
    events = query.open([directory / "telemetry"]).query_events()
    assert len(events) == tracking["total_events"]
    assert tracking["total_samples"] > len(tracking["samples"]) == 3
    assert analysis["session"]["status"] == "completed"
    offline = subprocess.run(
        [
            sys.executable,
            "-m",
            "stormlog.mlx.cli",
            "analyze",
            str(directory),
            "--json",
            "--output",
            str(output / "analysis"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert offline.returncode in (0, 3), offline.stdout + offline.stderr
    return {
        "capture": str(directory),
        "total_samples": tracking["total_samples"],
        "retained_samples": len(tracking["samples"]),
        "query_events": len(events),
        "analysis_exit_code": offline.returncode,
        "session_id": analysis["session"]["session_id"],
    }


def _snapshot(profiler: MLXMemoryProfiler, name: str, mx: Any) -> dict[str, Any]:
    mx.synchronize(mx.gpu)
    snapshot = profiler.capture_snapshot(name)
    assert snapshot.active_bytes == mx.get_active_memory()
    assert snapshot.cache_bytes == mx.get_cache_memory()
    return dict(snapshot.to_dict())


def _native_boundaries(output: Path, runtime: MLXRuntime) -> dict[str, Any]:
    import mlx.core as mx

    profiler = MLXMemoryProfiler(runtime=runtime)
    mx.synchronize(mx.gpu)
    gc.collect()
    mx.clear_cache()  # explicit harness mutation, not collector behavior
    idle = _snapshot(profiler, "idle", mx)
    x = mx.ones((128, 128))
    mx.eval(x)
    allocated = _snapshot(profiler, "evaluated_input", mx)
    lazy = x @ x.T
    pending = _snapshot(profiler, "lazy_output", mx)
    assert pending["active_bytes"] == allocated["active_bytes"]
    result = profiler.profile_function(lambda output=lazy: output, name="lazy_matmul")
    assert result is lazy and result[0, 0].item() == 128
    retained = _snapshot(profiler, "evaluated_output", mx)
    assert retained["active_bytes"] > allocated["active_bytes"]
    del x, lazy, result
    gc.collect()
    released = _snapshot(profiler, "released", mx)
    assert released["active_bytes"] < retained["active_bytes"]
    mx.clear_cache()
    cleared = _snapshot(profiler, "cache_cleared", mx)
    assert cleared["cache_bytes"] == 0
    profiler.export(str(output / "profiles.json"))
    return {
        "snapshots": [idle, allocated, pending, retained, released, cleared],
        "lazy_completion_verified": profiler.get_results()[0].completion_verified,
        "cleanup_policy": "explicit harness GC and cache clear only at named boundaries",
    }


def _measure_mode(
    mode: str, repetitions: int, shape: int, runtime: MLXRuntime
) -> dict[str, Any]:
    import resource

    import mlx.core as mx

    x = mx.ones((shape, shape), dtype=mx.float32)
    mx.eval(x)
    for _ in range(20):
        y = x @ x.T
        mx.eval(y)
    del y
    mx.synchronize(mx.gpu)
    mx.reset_peak_memory()  # owned by this standalone qualification harness
    baseline = mx.get_active_memory()
    profiler = MLXMemoryProfiler(
        runtime=runtime, max_history=3, sampling_interval=0.002
    )
    tracker = MemoryTracker(runtime=runtime, sampling_interval=0.002, max_history=3)
    if mode == "passive":
        tracker.start_tracking()
    durations = []
    try:
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            if mode == "bounded":
                y = profiler.profile_function(lambda: x @ x.T, name="matmul")
            else:
                mx.synchronize(mx.gpu)
                y = x @ x.T
                mx.eval(y)
                mx.synchronize(mx.gpu)
            durations.append(time.perf_counter_ns() - start)
            del y
    finally:
        if mode == "passive":
            tracker.stop_tracking()
    ordered = sorted(durations)
    return {
        "mode": mode,
        "repetitions": repetitions,
        "mean_ns": statistics.mean(durations),
        "p90_ns": ordered[math.ceil(0.9 * repetitions) - 1],
        "durations_ns": durations,
        "sample_count": (
            tracker.get_results().total_samples
            if mode == "passive"
            else sum(p.valid_sample_count for p in profiler.get_results())
        ),
        "native_peak_bytes": max(baseline, mx.get_peak_memory()),
        "native_peak_scope": "mode window after explicit harness reset; includes evaluated input baseline",
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_scope": "process lifetime high-water on macOS; cumulative across modes",
        "boundary_rss_bytes": psutil.Process().memory_info().rss,
    }


def _aggregate_windows(mode: str, windows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [w for w in windows if w["mode"] == mode]
    durations = [d for w in selected for d in w["durations_ns"]]
    return {
        "mode": mode,
        "repetitions": len(durations),
        "mean_ns": statistics.mean(durations),
        "p90_ns": sorted(durations)[math.ceil(0.9 * len(durations)) - 1],
        "sample_count": sum(w["sample_count"] for w in selected),
        "native_peak_bytes": max(w["native_peak_bytes"] for w in selected),
        "peak_rss_bytes": max(w["peak_rss_bytes"] for w in selected),
        "rss_scope": "process lifetime high-water, cumulative; per-window boundary RSS is recorded separately",
    }


def run_scenario(
    *, output_dir: Path, hardware: bool = False, repetitions: int = 30, shape: int = 128
) -> dict[str, Any]:
    if repetitions < 2 or shape < 1 or shape > 1024:
        raise ValueError("Require >=2 repetitions and shape in 1..1024")
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "status": "PASS",
        "hardware": hardware,
        "system": get_system_info(),
        "source": _source_identity(),
    }
    if hardware:
        runtime = MLXRuntime()
        result.update(
            mlx_version=runtime.version,
            device_info=runtime.device_info,
            boundaries=_native_boundaries(output, runtime),
        )
        orders = [
            ("off", "passive", "bounded"),
            ("passive", "bounded", "off"),
            ("bounded", "off", "passive"),
        ]
        windows = [
            {**_measure_mode(mode, repetitions, shape, runtime), "round": round_index}
            for round_index, order in enumerate(orders)
            for mode in order
        ]
        modes = [_aggregate_windows(mode, windows) for mode in orders[0]]
        result["windows"] = windows
        result["measurements"] = modes
        result["passive_mean_overhead_ratio"] = (
            modes[1]["mean_ns"] / modes[0]["mean_ns"] - 1
        )
        result["policy"] = {
            "shape": [shape, shape],
            "dtype": "float32",
            "warmups_per_window": 20,
            "stream": "GPU default",
            "compile": "eager",
            "cache": "warm cache, no clears within timing modes",
            "timing": "host wall elapsed per completed operation; bounded instrumentation intentionally changes scheduling",
            "mode_order": orders,
            "performance_limit": "three counterbalanced rounds in one process; timings are workload-specific, not a universal overhead budget",
        }
    else:
        profiler = MLXMemoryProfiler(runtime=ScenarioRuntime())
        profiler.profile_function(lambda: object(), name="synthetic")
        profiler.export(str(output / "profiles.json"))
        result["policy"] = {
            "measurement": "synthetic contract checks; no hardware/performance claim"
        }
    result["roundtrip"] = _roundtrip(output, hardware)
    result["failure_roundtrips"] = _failure_roundtrips(output, hardware)
    (output / "qualification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Require usable Metal; unavailable runtime fails",
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--shape", type=int, default=128)
    args = parser.parse_args()
    try:
        result = run_scenario(
            output_dir=args.output,
            hardware=args.hardware,
            repetitions=args.repetitions,
            shape=args.shape,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "hardware": args.hardware,
                    "output": str(args.output),
                }
            )
        )
        return 0
    except Exception as exc:
        print(f"MLX qualification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
