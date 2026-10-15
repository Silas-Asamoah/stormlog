"""Protocol v2: profile a window of W measured requests and measure its cost.

Every mode serves the same pinned workload (200 warmup, then 2,200 measured
requests offered every 0.1 s). Capture starts when the measured load starts and
stops W * 0.1 s later, while the remaining requests keep arriving. Resources are
sampled from before capture starts until the stop is acknowledged and the
server has been idle for a few seconds, so export and flush costs are included.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psutil

from scripts.native_probes.workloads import vllm_open_loop as v1

MODES = (
    "off",
    "kineto",
    "cupti-min",
    "cupti-full",
    "nsys",
    "cupti-kernel",
    "cupti-discard",
)
CUPTI_ACTIVITIES = {
    "cupti-min": "runtime,driver,kernel",
    "cupti-full": "driver,runtime,kernel,memcpy,memset,synchronization",
    # Ablations: device activity only, and the min set with records discarded.
    "cupti-kernel": "kernel",
    "cupti-discard": "runtime,driver,kernel",
}
# Stack-free Kineto, matching the public-engine preflights in v1, with the
# per-op extras off so the comparison is about trace buffering and export.
KINETO_CONFIG: dict[str, Any] = {
    "profiler": "torch",
    "torch_profiler_with_stack": False,
    "torch_profiler_record_shapes": False,
    "torch_profiler_with_memory": False,
    "torch_profiler_with_flops": False,
    "torch_profiler_dump_cuda_time_total": False,
}
CUPTI_MAX_BYTES = 64 * 1024**3
CUPTI_BUFFER_BYTES = 8 * 1024**2
SETTLE_SECONDS = 5.0
SAMPLE_SECONDS = 0.1
GPU_SAMPLE_EVERY = 10
INHERITED_CAPTURE_KEYS = ("CUDA_INJECTION64_PATH",)


def mode_environment(
    mode: str,
    base: Mapping[str, str],
    *,
    capture_dir: Path,
    helper: Path | None,
    buffer_bytes: int = CUPTI_BUFFER_BYTES,
    consumer_delay_ms: int = 0,
    device_buffer_bytes: int = 0,
    device_buffer_pool_limit: int = 0,
    flush_period_ms: int = 0,
) -> dict[str, str]:
    """Return the server environment for one mode, never inheriting capture."""
    environment = {
        key: value
        for key, value in base.items()
        if key not in INHERITED_CAPTURE_KEYS and not key.startswith("STORMLOG_CUPTI_")
    }
    if mode in CUPTI_ACTIVITIES:
        if helper is None:
            raise ValueError(f"{mode} requires --helper")
        environment.update(
            {
                "CUDA_INJECTION64_PATH": str(helper),
                "STORMLOG_CUPTI_OUTPUT_DIR": str(capture_dir),
                "STORMLOG_CUPTI_ACTIVITIES": CUPTI_ACTIVITIES[mode],
                "STORMLOG_CUPTI_MAX_BYTES": str(CUPTI_MAX_BYTES),
                "STORMLOG_CUPTI_BUFFER_BYTES": str(buffer_bytes),
                "STORMLOG_CUPTI_CONSUMER_DELAY_MS": str(consumer_delay_ms),
                "STORMLOG_CUPTI_DEFER_START": "1",
                "STORMLOG_CUPTI_DEVICE_BUFFER_BYTES": str(device_buffer_bytes),
                "STORMLOG_CUPTI_DEVICE_BUFFER_POOL_LIMIT": str(
                    device_buffer_pool_limit
                ),
                "STORMLOG_CUPTI_FLUSH_PERIOD_MS": str(flush_period_ms),
                "STORMLOG_CUPTI_DISCARD_RECORDS": (
                    "1" if mode == "cupti-discard" else "0"
                ),
            }
        )
    if mode == "nsys":
        environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return environment


def server_argv(
    mode: str,
    output: Path,
    port: int,
    *,
    cuda_graphs: bool = False,
    kineto_overrides: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return the pinned vLLM command for one mode."""
    command = v1._server_argv("off", output, port)
    if cuda_graphs:
        command.remove("--enforce-eager")
    if mode == "kineto":
        config = {
            **KINETO_CONFIG,
            **(kineto_overrides or {}),
            "torch_profiler_dir": str(output / "kineto"),
        }
        command += ["--profiler-config", json.dumps(config, sort_keys=True)]
    elif mode == "nsys":
        command += ["--profiler-config", json.dumps({"profiler": "cuda"})]
        command = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--trace-fork-before-exec=true",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            f"--output={output / 'nsys' / 'trace'}",
            *command,
        ]
    return command


def window_indices(rows: Sequence[Mapping[str, Any]], window: int) -> list[int]:
    """Return positions of measured requests offered inside the capture window."""
    return [
        position
        for position, row in enumerate(rows)
        if int(str(row["request_id"]).rsplit("-", 1)[1]) < window
    ]


def latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize offered and successful requests; quantiles over successes."""
    successful = [row for row in rows if row.get("status") == "ok"]
    latencies = [float(row["e2e_latency_ms"]) for row in successful]
    ttft = [float(row["ttft_ms"]) for row in successful if row.get("ttft_ms")]
    return {
        "offered": len(rows),
        "successful": len(successful),
        "e2e_p50_ms": v1._percentile(latencies, 0.5),
        "e2e_p95_ms": v1._percentile(latencies, 0.95),
        "e2e_p99_ms": v1._percentile(latencies, 0.99),
        "ttft_p50_ms": v1._percentile(ttft, 0.5),
        "ttft_p95_ms": v1._percentile(ttft, 0.95),
    }


def _phase_at(phases: Mapping[str, int | None], at_ns: int) -> str:
    """Name the run phase a sample belongs to from the recorded boundaries."""
    for name, boundary in (
        ("settle", phases.get("stop_acknowledged_ns")),
        ("stopping", phases.get("stop_requested_ns")),
        ("window", phases.get("capture_start_requested_ns")),
    ):
        if boundary is not None and at_ns >= boundary:
            return name
    return "baseline"


def memory_summary(
    samples: Sequence[Mapping[str, Any]], phases: Mapping[str, int | None]
) -> dict[str, Any]:
    """Peak and median target-tree RSS per phase, and growth over baseline."""
    by_phase: dict[str, list[int]] = {}
    gpu_by_phase: dict[str, list[int]] = {}
    incomplete = 0
    for sample in samples:
        phase = _phase_at(phases, sample["at_ns"])
        rss = sample.get("target_rss_bytes")
        if rss is None:
            incomplete += 1
            continue
        by_phase.setdefault(phase, []).append(rss)
        if sample.get("gpu_bytes") is not None:
            gpu_by_phase.setdefault(phase, []).append(sample["gpu_bytes"])
    baseline = by_phase.get("baseline")
    base = statistics.median(baseline) if baseline else None
    captured = [
        value
        for phase in ("window", "stopping", "settle")
        for value in by_phase.get(phase, [])
    ]
    summary: dict[str, Any] = {
        "target_rss_baseline_median_bytes": base,
        "target_rss_capture_peak_bytes": max(captured) if captured else None,
        "target_rss_added_peak_bytes": (
            max(captured) - base if captured and base is not None else None
        ),
        "incomplete_samples": incomplete,
    }
    for phase, values in by_phase.items():
        summary[f"target_rss_{phase}_peak_bytes"] = max(values)
        summary[f"target_rss_{phase}_median_bytes"] = statistics.median(values)
        summary[f"samples_{phase}"] = len(values)
    for phase, values in gpu_by_phase.items():
        summary[f"gpu_{phase}_peak_bytes"] = max(values)
    return summary


class Sampler(threading.Thread):
    """Sample target-tree and client RSS, target CPU time and GPU memory."""

    def __init__(self, server_pid: int, path: Path, rss_cap_bytes: int) -> None:
        super().__init__(daemon=True)
        self.server_pid = server_pid
        self.path = path
        self.rss_cap_bytes = rss_cap_bytes
        self.cap_exceeded_at_ns: int | None = None
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []

    def run(self) -> None:
        client = psutil.Process(os.getpid())
        index = 0
        with self.path.open("x", encoding="utf-8") as target:
            while not self.stop_event.is_set():
                row: dict[str, Any] = {"at_ns": time.time_ns()}
                try:
                    server = psutil.Process(self.server_pid)
                    tree = [server, *server.children(recursive=True)]
                    rss = 0
                    cpu = 0.0
                    pids = []
                    for process in tree:
                        info = process.memory_info()
                        times = process.cpu_times()
                        rss += info.rss
                        cpu += times.user + times.system
                        pids.append(process.pid)
                    row.update(
                        target_rss_bytes=rss, target_cpu_seconds=cpu, target_pids=pids
                    )
                    if rss > self.rss_cap_bytes and self.cap_exceeded_at_ns is None:
                        # Protect the host; the trial records this as its outcome.
                        self.cap_exceeded_at_ns = row["at_ns"]
                        os.killpg(self.server_pid, signal.SIGKILL)
                except (psutil.Error, OSError) as exc:
                    row.update(target_rss_bytes=None, error=f"{type(exc).__name__}")
                row["client_rss_bytes"] = client.memory_info().rss
                if index % GPU_SAMPLE_EVERY == 0:
                    gpu, error = v1._query_gpu_memory()
                    row["gpu_bytes"] = sum(gpu.values()) if gpu else None
                    row["gpu_by_pid_bytes"] = gpu
                    row["gpu_error"] = error
                target.write(json.dumps(row, sort_keys=True) + "\n")
                self.samples.append(row)
                index += 1
                self.stop_event.wait(SAMPLE_SECONDS)


def _cupti_control(directory: Path, message: bytes, timeout: float) -> dict[str, Any]:
    """Send one control message to every injected process and collect acks."""
    sockets = sorted(directory.rglob("stop.sock"))
    acks: dict[str, str | None] = {}
    errors: list[str] = []
    for path in sockets:
        key = path.parent.name
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as control:
                control.settimeout(timeout)
                control.connect(str(path))
                control.sendall(message)
                acks[key] = control.recv(4).decode("ascii", errors="replace")
        except OSError as exc:
            acks[key] = None
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    if not sockets:
        errors.append("no injected process exposed a control socket")
    ok = bool(sockets) and not errors and all(ack == "OK\n" for ack in acks.values())
    return {"acks": acks, "errors": errors, "ok": ok}


def _http_control(endpoint: str, action: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(endpoint + f"/{action}", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"http_status": response.status, "ok": response.status < 300}
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def capture_controls(
    mode: str, endpoint: str, capture_dir: Path, stop_timeout: float
) -> tuple[Callable[[], dict[str, Any]], Callable[[], dict[str, Any]]]:
    """Return start and stop callables for a mode; `off` does nothing."""
    if mode in CUPTI_ACTIVITIES:
        return (
            lambda: _cupti_control(capture_dir, b"STRT", 30),
            lambda: _cupti_control(capture_dir, b"STOP", stop_timeout),
        )
    if mode in {"kineto", "nsys"}:
        return (
            lambda: _http_control(endpoint, "start_profile", 120),
            lambda: _http_control(endpoint, "stop_profile", stop_timeout),
        )
    return (lambda: {"ok": True}, lambda: {"ok": True})


def _artifact_summary(mode: str, output: Path, capture_dir: Path) -> dict[str, Any]:
    roots = {"kineto": output / "kineto", "nsys": output / "nsys"}
    summary: dict[str, Any] = {}
    root = capture_dir if mode in CUPTI_ACTIVITIES else roots.get(mode)
    if root is not None and root.exists():
        files = [path for path in root.rglob("*") if path.is_file()]
        summary["files"] = {
            str(path.relative_to(output)): path.stat().st_size for path in files
        }
        summary["bytes"] = sum(summary["files"].values())
    if mode in CUPTI_ACTIVITIES:
        statuses = []
        for path in sorted(capture_dir.rglob("cupti_status.json")):
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        summary["cupti_status"] = statuses
        summary["delivered_records"] = sum(s["delivered_records"] for s in statuses)
        summary["dropped_records"] = sum(
            s["cupti_dropped_records"] + s["local_dropped_records"] for s in statuses
        )
        summary["buffer_peak_outstanding_bytes"] = max(
            (s.get("activity_buffer_peak_outstanding_bytes", 0) for s in statuses),
            default=None,
        )
        summary["buffer_completed_cpu_seconds"] = (
            sum(s.get("buffer_completed_cpu_ns", 0) for s in statuses) / 1e9
        )
        summary["kernels_without_timestamps"] = sum(
            s.get("kernels_without_timestamps", 0) for s in statuses
        )
        summary["all_finalized"] = bool(statuses) and all(
            s.get("finalized") for s in statuses
        )
    return summary


def _installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(args: argparse.Namespace) -> int:
    """Serve one trial in one mode and write result.json, even on failure."""
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    capture_dir = output / "cupti"
    for sub in ("cupti", "nsys", "kineto"):
        (output / sub).mkdir(mode=0o700)
    environment = mode_environment(
        args.mode,
        os.environ,
        capture_dir=capture_dir,
        helper=args.helper,
        buffer_bytes=args.buffer_bytes,
        consumer_delay_ms=args.consumer_delay_ms,
        device_buffer_bytes=args.device_buffer_bytes,
        device_buffer_pool_limit=args.device_buffer_pool_limit,
        flush_period_ms=args.flush_period_ms,
    )
    argv = server_argv(
        args.mode,
        output,
        args.port,
        cuda_graphs=args.cuda_graphs,
        kineto_overrides=json.loads(args.kineto_overrides),
    )
    endpoint = f"http://127.0.0.1:{args.port}"
    phases: dict[str, int | None] = {}
    result: dict[str, Any] = {
        "protocol": "native-probe-v2",
        "mode": args.mode,
        "window_requests": args.window,
        "cuda_graphs": args.cuda_graphs,
        "trial_id": output.name,
        "vllm_version": _installed("vllm"),
        "torch_version": _installed("torch"),
        "server_argv": argv,
        "capture_environment": {
            key: value
            for key, value in environment.items()
            if key.startswith("STORMLOG_CUPTI_") or key in INHERITED_CAPTURE_KEYS
        },
        "errors": [],
    }
    start_capture, stop_capture = capture_controls(
        args.mode, endpoint, capture_dir, args.stop_timeout
    )
    rows: list[dict[str, Any]] = []
    sampler: Sampler | None = None
    with (
        (output / "server-stdout.log").open("xb") as stdout,
        (output / "server-stderr.log").open("xb") as stderr,
    ):
        server = subprocess.Popen(
            argv, stdout=stdout, stderr=stderr, env=environment, start_new_session=True
        )
        try:
            v1._wait_ready(endpoint, server)
            phases["ready_ns"] = time.time_ns()

            def sender(request_id: str) -> dict[str, Any]:
                return v1._request(endpoint, request_id, 60)

            warmup = asyncio.run(v1.offer_requests(args.warmup, "warmup", sender))
            result["warmup"] = latency_summary(warmup)
            sampler = Sampler(server.pid, output / "samples.jsonl", args.rss_cap_bytes)
            sampler.start()
            time.sleep(2.0)
            stop_result: dict[str, Any] = {}

            def window_stop() -> None:
                time.sleep(args.window * v1.INTERVAL_SECONDS)
                phases["stop_requested_ns"] = time.time_ns()
                stop_result.update(stop_capture())
                phases["stop_acknowledged_ns"] = time.time_ns()

            phases["capture_start_requested_ns"] = time.time_ns()
            result["start"] = start_capture()
            phases["capture_started_ns"] = time.time_ns()
            stopper = threading.Thread(target=window_stop, daemon=True)
            stopper.start()
            rows = asyncio.run(v1.offer_requests(args.measured, "measured", sender))
            phases["measured_done_ns"] = time.time_ns()
            stopper.join(timeout=args.stop_timeout + 30)
            result["stop"] = stop_result or {
                "ok": False,
                "error": "stop never returned",
            }
            if stopper.is_alive():
                result["errors"].append("stop control still blocked")
            time.sleep(SETTLE_SECONDS)
        except Exception as exc:  # retained as a failed trial, never dropped
            result["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            if sampler is not None:
                sampler.stop_event.set()
                sampler.join()
                if sampler.cap_exceeded_at_ns is not None:
                    phases["rss_cap_exceeded_ns"] = sampler.cap_exceeded_at_ns
                    result["errors"].append("rss_cap_exceeded")
            phases["shutdown_requested_ns"] = time.time_ns()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
                result["errors"].append("server needed SIGKILL")
            result["server_return_code"] = server.returncode
    v1._write_jsonl(output / "requests.jsonl", rows)
    result["phases_ns"] = phases
    start_ns, stop_ns = phases.get("stop_requested_ns"), phases.get(
        "stop_acknowledged_ns"
    )
    result["stop_seconds"] = (
        (stop_ns - start_ns) / 1e9 if start_ns and stop_ns else None
    )
    inside_positions = set(window_indices(rows, args.window))
    inside = [row for position, row in enumerate(rows) if position in inside_positions]
    after = [
        row for position, row in enumerate(rows) if position not in inside_positions
    ]
    result["latency_window"] = latency_summary(inside)
    result["latency_after_window"] = latency_summary(after)
    result["latency_all"] = latency_summary(rows)
    samples = sampler.samples if sampler else []
    result["memory"] = memory_summary(samples, phases)
    cpu = [
        s["target_cpu_seconds"]
        for s in samples
        if s.get("target_cpu_seconds") is not None
        and phases.get("capture_start_requested_ns", 0) <= s["at_ns"]
    ]
    result["target_cpu_seconds_capture_to_end"] = cpu[-1] - cpu[0] if cpu else None
    try:
        result["artifacts"] = _artifact_summary(args.mode, output, capture_dir)
    except (OSError, ValueError, KeyError) as exc:
        result["errors"].append(f"artifact summary: {type(exc).__name__}: {exc}")
    for key in ("start", "stop"):
        if not result.get(key, {}).get("ok"):
            result["errors"].append(f"capture {key} failed")
    result["ok"] = not result["errors"]
    (output / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({"trial": output.name, "ok": result["ok"]}))
    return 0 if result["ok"] else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--stop-timeout", type=float, default=600.0)
    parser.add_argument("--warmup", type=int, default=v1.WARMUP_REQUESTS)
    parser.add_argument("--measured", type=int, default=v1.MEASURED_REQUESTS)
    parser.add_argument("--buffer-bytes", type=int, default=CUPTI_BUFFER_BYTES)
    parser.add_argument("--consumer-delay-ms", type=int, default=0)
    parser.add_argument("--device-buffer-bytes", type=int, default=0)
    parser.add_argument("--device-buffer-pool-limit", type=int, default=0)
    parser.add_argument("--flush-period-ms", type=int, default=0)
    parser.add_argument("--kineto-overrides", default="{}")
    parser.add_argument("--rss-cap-bytes", type=int, default=80 * 1024**3)
    args = parser.parse_args(argv)
    if not 0 < args.window <= args.measured:
        parser.error("--window must be between 1 and --measured")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
