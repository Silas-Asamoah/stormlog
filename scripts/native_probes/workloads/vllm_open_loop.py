"""Pinned, open-loop vLLM workload for native-probe research trials."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import socket
import stat
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Callable, Sequence

import psutil

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "c89bee90d9f811437d9735454613c35b4a3c4dc8"
PROMPT = (
    "Explain in one paragraph why a GPU kernel launch can precede its execution. "
    "Include one example."
)
RANGE_ID = "stormlog-native-probe-measured"
INTERVAL_SECONDS = 0.1
WARMUP_REQUESTS = 200
MEASURED_REQUESTS = 2200
MAX_IN_FLIGHT = 32
SLO_DEADLINE_MS = 2000
SLO_OFFER_INTERVAL_SECONDS = MEASURED_REQUESTS * INTERVAL_SECONDS
CUPTI_TRACE_ENCODING = "stormlog-zlib-frames-v1"
CUPTI_TRACE_MAGIC = b"SLCPTZ1\n"
CUPTI_FRAME_HEADER = struct.Struct("<III")
CUPTI_MAX_FRAME_BYTES = 16 * 1024 * 1024
CUPTI_STOP_TIMEOUT_SECONDS = 300


def request_body() -> dict[str, Any]:
    """Return one exact request shared by every profiler mode."""
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def schedule(count: int, *, interval_seconds: float = INTERVAL_SECONDS) -> list[float]:
    """Return fixed offsets; completion times never shift later arrivals."""
    if count < 1 or interval_seconds <= 0:
        raise ValueError("count and interval_seconds must be positive")
    return [index * interval_seconds for index in range(count)]


def _request(endpoint: str, request_id: str, timeout: float) -> dict[str, Any]:
    started = time.time_ns()
    first_chunk: int | None = None
    usage: dict[str, Any] | None = None
    completed = False
    http_status: int | None = None
    try:
        request = urllib.request.Request(
            endpoint + "/v1/chat/completions",
            data=json.dumps(request_body(), sort_keys=True).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": request_id,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = response.status
            for raw in response:
                line = raw.strip()
                if line == b"data: [DONE]":
                    completed = True
                    continue
                if not line.startswith(b"data: "):
                    continue
                payload = json.loads(line[6:])
                if payload.get("error"):
                    raise ValueError(f"stream error: {payload['error']}")
                if payload.get("choices") and first_chunk is None:
                    first_chunk = time.time_ns()
                if isinstance(payload.get("usage"), dict):
                    usage = payload["usage"]
        status = (
            "ok"
            if first_chunk is not None
            and completed
            and http_status is not None
            and 200 <= http_status < 300
            else "error"
        )
        error = (
            None
            if status == "ok"
            else (
                "stream ended before [DONE]"
                if not completed
                else "stream contained no choices"
            )
        )
    except (OSError, ValueError) as exc:
        status = "timeout" if isinstance(exc, TimeoutError) else "error"
        error = f"{type(exc).__name__}: {exc}"
    ended = time.time_ns()
    return {
        "request_id": request_id,
        "status": status,
        "started_at_ns": started,
        "ended_at_ns": ended,
        "e2e_latency_ms": (ended - started) / 1_000_000,
        "ttft_ms": (first_chunk - started) / 1_000_000 if first_chunk else None,
        "first_chunk_at_ns": first_chunk,
        "usage": usage,
        "http_status": http_status,
        "stream_done": completed,
        "error": error,
    }


async def offer_requests(
    count: int,
    phase: str,
    sender: Callable[[str], dict[str, Any]],
    *,
    interval_seconds: float = INTERVAL_SECONDS,
    max_in_flight: int = MAX_IN_FLIGHT,
) -> list[dict[str, Any]]:
    """Offer requests on schedule and account for overload without queueing it."""
    offsets = schedule(count, interval_seconds=interval_seconds)
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(max_in_flight)
    started = loop.time()

    async def one(index: int, offset: float) -> dict[str, Any]:
        await asyncio.sleep(max(0.0, started + offset - loop.time()))
        request_id = f"stormlog-118-{phase}-{index:04d}"
        offered_ns = time.time_ns()
        if semaphore.locked():
            return {
                "request_id": request_id,
                "status": "client_rejected",
                "offered_at_ns": offered_ns,
                "scheduled_offset_ms": round(offset * 1000, 3),
            }
        await semaphore.acquire()
        try:
            try:
                result = await loop.run_in_executor(None, sender, request_id)
            except Exception as exc:
                result = {
                    "request_id": request_id,
                    "status": "error",
                    "started_at_ns": offered_ns,
                    "ended_at_ns": time.time_ns(),
                    "first_chunk_at_ns": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        finally:
            semaphore.release()
        if "ended_at_ns" in result:
            result["e2e_latency_ms"] = (result["ended_at_ns"] - offered_ns) / 1_000_000
            result["dispatch_delay_ms"] = (
                result["started_at_ns"] - offered_ns
            ) / 1_000_000
            first_chunk = result.get("first_chunk_at_ns")
            result["ttft_ms"] = (
                (first_chunk - offered_ns) / 1_000_000
                if first_chunk is not None
                else None
            )
        return {
            **result,
            "offered_at_ns": offered_ns,
            "scheduled_offset_ms": round(offset * 1000, 3),
        }

    return await asyncio.gather(
        *(one(index, offset) for index, offset in enumerate(offsets))
    )


def _control(endpoint: str, action: str) -> None:
    request = urllib.request.Request(endpoint + f"/{action}", data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status >= 300:
            raise RuntimeError(f"{action} returned HTTP {response.status}")


def _wait_ready(endpoint: str, server: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup: {server.returncode}")
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.HTTPError):
            pass
        time.sleep(0.5)
    raise TimeoutError("vLLM readiness timed out")


def _server_argv(mode: str, output: Path, port: int) -> list[str]:
    binary = str(Path(sys.executable).with_name("vllm"))
    command = [
        binary,
        "serve",
        MODEL,
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "half",
        "--kv-cache-dtype",
        "auto",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "32",
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        "0.8",
        "--seed",
        "118",
        "--no-enable-prefix-caching",
        "--enforce-eager",
    ]
    profiler: dict[str, Any] | None = None
    if mode == "public-engine":
        profiler = {
            "profiler": "torch",
            "torch_profiler_dir": str(output / "public-engine"),
        }
    elif mode == "proton":
        profiler = {
            "profiler": "proton",
            "proton_profiler_dir": str(output / "proton"),
            "proton_data": "trace",
            "proton_output_format": "chrome_trace",
            "proton_hook": "triton",
            "proton_graph_attribution": False,
        }
    elif mode == "trusted":
        profiler = {"profiler": "cuda"}
    if profiler:
        command.extend(("--profiler-config", json.dumps(profiler, sort_keys=True)))
    if mode == "trusted":
        command = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--trace-fork-before-exec=true",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=repeat",
            "--force-overwrite=false",
            f"--output={output / 'nsys' / 'vendor-trace'}",
            *command,
        ]
    return command


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (index - lower)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, sort_keys=True) + "\n")


def _sample_resources(
    server_pid: int, output: Path, stop: threading.Event, samples: list[dict[str, Any]]
) -> None:
    """Retain each measured-window process RSS sample and sampling failure."""
    with output.open("x", encoding="utf-8") as target:
        while not stop.is_set():
            row: dict[str, Any] = {"at_ns": time.time_ns(), "processes": []}
            try:
                server = psutil.Process(server_pid)
                processes = [server, *server.children(recursive=True)]
                for process in [psutil.Process(os.getpid()), *processes]:
                    role = "helper_agent" if process.pid == os.getpid() else "target"
                    try:
                        row["processes"].append(
                            {
                                "pid": process.pid,
                                "role": role,
                                "rss_bytes": process.memory_info().rss,
                            }
                        )
                    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
                        row["processes"].append(
                            {
                                "pid": process.pid,
                                "role": role,
                                "rss_bytes": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            target.write(json.dumps(row, sort_keys=True) + "\n")
            target.flush()
            samples.append(row)
            stop.wait(0.1)


def _memory_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    combined: list[int] = []
    target_sets: list[set[int]] = []
    for sample in samples:
        processes = sample.get("processes", [])
        if sample.get("error") or any(
            row.get("rss_bytes") is None for row in processes
        ):
            continue
        if not {"target", "helper_agent"}.issubset(
            {row.get("role") for row in processes}
        ):
            continue
        targets = {
            row["pid"]
            for row in processes
            if row.get("role") == "target"
            and isinstance(row.get("pid"), int)
            and not isinstance(row.get("pid"), bool)
        }
        if not targets or len(targets) != sum(
            row.get("role") == "target" for row in processes
        ):
            continue
        target_sets.append(targets)
        combined.append(sum(row["rss_bytes"] for row in processes))
    complete = (
        bool(samples)
        and len(combined) == len(samples)
        and all(targets == target_sets[0] for targets in target_sets)
    )
    return {
        "host_rss_measured_peak_bytes": max(combined) if complete else None,
        "host_rss_measured_median_bytes": (
            statistics.median(combined) if complete else None
        ),
        "host_rss_samples": len(samples),
        "host_rss_valid_samples": len(combined),
        "gpu_memory_bytes": None,
        "pinned_host_memory_bytes": None,
        "collector_buffer_bytes": None,
    }


def _gpu_process_pids() -> tuple[set[int] | None, str | None]:
    """Query live CUDA context owners before the target shutdown begins."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"GPU process query failed: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return None, f"GPU process query exited {result.returncode}: {result.stderr}"
    try:
        return {
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        }, None
    except ValueError as exc:
        return None, f"GPU process query contained an invalid pid: {exc}"


def _stop_cupti_helpers(
    directory: Path,
    *,
    timeout: float = CUPTI_STOP_TIMEOUT_SECONDS,
    expected_controls: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Finalize every live injected process before stopping the vLLM server."""
    sockets = sorted(directory.rglob("stop.sock"))
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    if not sockets:
        errors.append("no live CUPTI stop socket")
    for path in sockets:
        row: dict[str, Any] = {
            "socket": str(path.relative_to(directory)),
            "request_at_ns": time.time_ns(),
            "ack_at_ns": None,
        }
        descriptor = -1
        try:
            socket_status = path.lstat()
            directory_status = path.parent.stat()
            if (
                not stat.S_ISSOCK(socket_status.st_mode)
                or socket_status.st_uid != os.geteuid()
                or stat.S_IMODE(socket_status.st_mode) != 0o600
                or directory_status.st_uid != os.geteuid()
                or stat.S_IMODE(directory_status.st_mode) != 0o700
            ):
                raise ValueError("stop socket or process directory is not private")
            address = str(path)
            if len(os.fsencode(address)) >= 100 and Path("/proc/self/fd").is_dir():
                descriptor = os.open(
                    path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                address = f"/proc/self/fd/{descriptor}/stop.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as control:
                control.settimeout(timeout)
                control.connect(address)
                control.sendall(b"STOP")
                row["ack"] = control.recv(4).decode("ascii", errors="replace")
                row["ack_at_ns"] = time.time_ns()
            status_path = path.parent / "cupti_status.json"
            if status_path.is_file():
                row["status"] = json.loads(status_path.read_text(encoding="utf-8"))
            if row.get("ack") != "OK\n" or not isinstance(row.get("status"), dict):
                errors.append(f"{row['socket']}: stop acknowledgment or status missing")
            elif row["status"].get("finalized") is not True:
                errors.append(f"{row['socket']}: CUPTI helper did not finalize")
            elif len(path.parent.name.split("-")) < 3 or path.parent.name.split("-")[
                1
            ] != str(row["status"].get("pid")):
                errors.append(f"{row['socket']}: process identity mismatch")
            if isinstance(row.get("status"), dict) and expected_controls is not None:
                for key, expected in expected_controls.items():
                    if row["status"].get(key) != expected:
                        errors.append(
                            f"{row['socket']}: observed {key} differs from configured value"
                        )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{row['socket']}: {type(exc).__name__}: {exc}")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            reports.append(row)
    return {"reports": reports, "errors": errors, "complete": not errors}


def _inspect_cupti_trace(path: Path) -> dict[str, Any]:
    """Require a terminating frame and verify every compressed JSON record."""
    digest = hashlib.sha256()
    records = 0
    decoded_bytes = 0
    frames = 0
    with path.open("rb") as source:
        if source.read(len(CUPTI_TRACE_MAGIC)) != CUPTI_TRACE_MAGIC:
            raise ValueError("CUPTI trace magic does not match the pinned codec")
        while True:
            header = source.read(CUPTI_FRAME_HEADER.size)
            if len(header) != CUPTI_FRAME_HEADER.size:
                raise ValueError("CUPTI trace end frame is missing")
            encoded_size, raw_size, frame_records = CUPTI_FRAME_HEADER.unpack(header)
            if (encoded_size, raw_size, frame_records) == (0, 0, 0):
                if source.read(1):
                    raise ValueError("CUPTI trace has trailing bytes")
                break
            if not (0 < encoded_size <= CUPTI_MAX_FRAME_BYTES):
                raise ValueError("CUPTI encoded frame size is invalid")
            if not (0 < raw_size <= CUPTI_MAX_FRAME_BYTES) or frame_records <= 0:
                raise ValueError("CUPTI decoded frame size or count is invalid")
            encoded = source.read(encoded_size)
            if len(encoded) != encoded_size:
                raise ValueError("CUPTI compressed frame is truncated")
            decoder = zlib.decompressobj()
            try:
                raw = decoder.decompress(encoded, raw_size + 1)
                if len(raw) > raw_size or decoder.unconsumed_tail:
                    raise ValueError("CUPTI decoded frame exceeds declared size")
                raw += decoder.flush()
            except zlib.error as error:
                raise ValueError(
                    f"CUPTI compressed frame is corrupt: {error}"
                ) from error
            if (
                len(raw) != raw_size
                or not decoder.eof
                or decoder.unused_data
                or decoder.unconsumed_tail
            ):
                raise ValueError("CUPTI compressed frame does not decode exactly")
            if not raw.endswith(b"\n"):
                raise ValueError("CUPTI decoded frame lacks record terminator")
            lines = raw.splitlines()
            if len(lines) != frame_records:
                raise ValueError("CUPTI frame record count does not match")
            for line in lines:
                try:
                    if not isinstance(json.loads(line), dict):
                        raise ValueError("activity record is not an object")
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"CUPTI record is malformed: {error}") from error
            digest.update(raw)
            records += frame_records
            decoded_bytes += raw_size
            frames += 1
    return {
        "encoding": CUPTI_TRACE_ENCODING,
        "encoded_bytes": path.stat().st_size,
        "decoded_bytes": decoded_bytes,
        "decoded_sha256": digest.hexdigest(),
        "records": records,
        "frames": frames,
    }


def _cupti_capture_status(
    output: Path,
    samples: list[dict[str, Any]],
    required_gpu_pids: set[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a finalized capture for every observed target process."""
    observed = sorted(
        {
            row["pid"]
            for sample in samples
            for row in sample.get("processes", [])
            if row.get("role") == "target"
        }
    )
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted((output / "cupti").rglob("cupti_status.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("status is not an object")
            report["status_path"] = str(path.relative_to(output))
            reports.append(report)
        except (OSError, ValueError) as exc:
            errors.append(f"{path.relative_to(output)}: {type(exc).__name__}: {exc}")
    reported = [row.get("pid") for row in reports]
    valid_reported = [
        pid
        for pid in reported
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
    ]
    if not reports:
        errors.append("no CUPTI status report")
    if len(valid_reported) != len(set(valid_reported)):
        errors.append("duplicate CUPTI process status")
    if len(valid_reported) != len(reported):
        errors.append("CUPTI process status has an invalid pid")
    required = set(observed) if required_gpu_pids is None else required_gpu_pids
    missing = sorted(required - set(valid_reported))
    if missing:
        errors.append(f"observed target processes without CUPTI status: {missing}")
    if not observed:
        errors.append("no target process observed during measurement")
    if required_gpu_pids is not None and not required_gpu_pids:
        errors.append("no GPU process identified before capture stop")
    if required_gpu_pids is not None and not required_gpu_pids.issubset(observed):
        errors.append("GPU process was not observed in target process samples")
    unexpected = sorted(set(valid_reported) - set(observed))
    if unexpected:
        errors.append(f"CUPTI status for unobserved target processes: {unexpected}")
    dropped = 0
    delivered = 0
    trace_paths: set[Path] = set()
    trace_validation: list[dict[str, Any]] = []
    for report in reports:
        if report.get("finalized") is not True or report.get("initialization_error"):
            errors.append(f"CUPTI process {report.get('pid')} did not finalize cleanly")
        for key in (
            "delivered_records",
            "cupti_dropped_records",
            "local_dropped_records",
            "bytes_dropped",
        ):
            value = report.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"CUPTI process {report.get('pid')} has invalid {key}")
        if all(
            isinstance(report.get(key), int) and not isinstance(report.get(key), bool)
            for key in (
                "delivered_records",
                "cupti_dropped_records",
                "local_dropped_records",
            )
        ):
            delivered += report["delivered_records"]
            dropped += report["cupti_dropped_records"] + report["local_dropped_records"]
        encoding = report.get("trace_encoding")
        if encoding == CUPTI_TRACE_ENCODING:
            trace = Path(report["status_path"]).parent / "activity.sclz"
        elif encoding is None:
            trace = Path(report["status_path"]).parent / "activity.ndjson"
        else:
            errors.append(
                f"CUPTI process {report.get('pid')} has unknown trace encoding"
            )
            continue
        if trace in trace_paths:
            errors.append(f"CUPTI processes share an activity trace: {trace}")
        trace_paths.add(trace)
        if not (output / trace).is_file() or (output / trace).stat().st_size == 0:
            errors.append(f"CUPTI process {report.get('pid')} has no activity trace")
        else:
            try:
                if encoding == CUPTI_TRACE_ENCODING:
                    inspected = _inspect_cupti_trace(output / trace)
                    trace_validation.append({"pid": report.get("pid"), **inspected})
                    records = inspected["records"]
                    if inspected["encoded_bytes"] != report.get("bytes_written"):
                        errors.append(
                            f"CUPTI process {report.get('pid')} encoded bytes differ from status"
                        )
                    if inspected["decoded_bytes"] != report.get(
                        "uncompressed_bytes_delivered"
                    ):
                        errors.append(
                            f"CUPTI process {report.get('pid')} decoded bytes differ from status"
                        )
                    maximum = report.get("maximum_output_bytes")
                    if (
                        not isinstance(maximum, int)
                        or inspected["encoded_bytes"] > maximum
                    ):
                        errors.append(
                            f"CUPTI process {report.get('pid')} exceeded encoded byte bound"
                        )
                else:
                    with (output / trace).open(encoding="utf-8") as source:
                        records = 0
                        for line in source:
                            if not isinstance(json.loads(line), dict):
                                raise ValueError("activity record is not an object")
                            records += 1
                if records != report.get("delivered_records"):
                    errors.append(
                        f"CUPTI process {report.get('pid')} trace record count "
                        f"{records} differs from delivered_records"
                    )
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(
                    f"CUPTI process {report.get('pid')} has malformed activity trace: "
                    f"{type(exc).__name__}: {exc}"
                )
        if report.get("bytes_dropped"):
            errors.append(f"CUPTI process {report.get('pid')} dropped output bytes")
    complete = not errors and dropped == 0 and delivered > 0
    if dropped:
        errors.append(f"CUPTI reported {dropped} dropped records")
    if delivered == 0:
        errors.append("CUPTI delivered no activity records")
    capture = {
        "observed_target_pids": observed,
        "required_gpu_pids": sorted(required),
        "reported_pids": reported,
        "status_reports": reports,
        "trace_validation": trace_validation,
        "errors": errors,
        "complete": complete,
    }
    loss = {
        "vendor_activity": {
            "status": "reported" if complete or dropped else "unknown",
            "delivered_records": delivered,
            "lost_records": dropped if reports else None,
            "expected_records": delivered + dropped if reports else None,
            "reason": "; ".join(errors) if errors else None,
        }
    }
    return capture, loss


def _slo_goodput(
    rows: list[dict[str, Any]],
    seconds: float,
    deadline_ms: float | None,
    *,
    expected_requests: int | None = None,
) -> float | None:
    """Count successful requests within a declared SLO, or retain unknown."""
    if deadline_ms is None or seconds <= 0:
        return None
    if not math.isfinite(deadline_ms) or deadline_ms <= 0:
        raise ValueError("SLO deadline must be finite and positive")
    if expected_requests is not None:
        if len(rows) != expected_requests:
            return None
        expected_ids = {
            f"stormlog-118-measured-{index:04d}" for index in range(expected_requests)
        }
        if any(not isinstance(row.get("request_id"), str) for row in rows):
            return None
        if {row["request_id"] for row in rows} != expected_ids:
            return None
        for row in rows:
            index = int(row["request_id"].rsplit("-", 1)[1])
            if (
                row.get("scheduled_offset_ms")
                != round(index * INTERVAL_SECONDS * 1000, 3)
                or type(row.get("offered_at_ns")) is not int
                or row.get("status")
                not in {"ok", "error", "timeout", "client_rejected"}
            ):
                return None
            if row["status"] == "ok" and (
                row.get("stream_done") is not True
                or not isinstance(row.get("http_status"), int)
                or not 200 <= row["http_status"] < 300
                or not isinstance(row.get("ended_at_ns"), int)
                or not isinstance(row.get("e2e_latency_ms"), (int, float))
                or isinstance(row.get("e2e_latency_ms"), bool)
                or not math.isfinite(row["e2e_latency_ms"])
                or row["e2e_latency_ms"] < 0
                or abs(
                    row["e2e_latency_ms"]
                    - (row["ended_at_ns"] - row["offered_at_ns"]) / 1_000_000
                )
                > 0.001
            ):
                return None
            if row["status"] in {"error", "timeout"} and not isinstance(
                row.get("ended_at_ns"), int
            ):
                return None
    return (
        float(
            sum(
                row.get("status") == "ok"
                and isinstance(row.get("e2e_latency_ms"), (int, float))
                and not isinstance(row.get("e2e_latency_ms"), bool)
                and math.isfinite(row["e2e_latency_ms"])
                and row["e2e_latency_ms"] <= deadline_ms
                for row in rows
            )
        )
        / seconds
    )


def _result(
    rows: list[dict[str, Any]],
    started_ns: int,
    ended_ns: int,
    *,
    slo_deadline_ms: float | None = SLO_DEADLINE_MS,
) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "ok"]
    latencies = [float(row["e2e_latency_ms"]) for row in successful]
    ttft = [float(row["ttft_ms"]) for row in successful if row["ttft_ms"] is not None]
    seconds = (ended_ns - started_ns) / 1_000_000_000
    return {
        "artifact_kind": "workload_result",
        "metrics": {
            "offered_requests": len(rows),
            "successful_requests": len(successful),
            "failed_requests": sum(row["status"] == "error" for row in rows),
            "timed_out_requests": sum(row["status"] == "timeout" for row in rows),
            "client_rejected_requests": sum(
                row["status"] == "client_rejected" for row in rows
            ),
            "e2e_p50_ms": _percentile(latencies, 0.5),
            "e2e_p95_ms": _percentile(latencies, 0.95),
            "e2e_p99_ms": _percentile(latencies, 0.99),
            "ttft_p95_ms": _percentile(ttft, 0.95),
            "successful_requests_per_second": len(successful) / seconds,
            "slo_goodput_requests_per_second": _slo_goodput(
                rows,
                SLO_OFFER_INTERVAL_SECONDS,
                slo_deadline_ms,
                expected_requests=MEASURED_REQUESTS,
            ),
            "slo_deadline_ms": slo_deadline_ms,
            "slo_offer_interval_seconds": SLO_OFFER_INTERVAL_SECONDS,
        },
        "measurement_window": {
            "range_id": RANGE_ID,
            "marker": "client-open-loop",
            "warmup_iterations": WARMUP_REQUESTS,
            "measured_iterations": len(rows),
            "host_started_ns": started_ns,
            "host_finished_ns": ended_ns,
            "clock": "CLOCK_REALTIME",
            "flush_completed": True,
        },
        "ground_truth": {
            "scheduled_requests": MEASURED_REQUESTS,
            "interval_ms": INTERVAL_SECONDS * 1000,
            "max_in_flight": MAX_IN_FLIGHT,
        },
    }


def run(mode: str, output: Path, port: int) -> int:
    """Start one fresh server, capture one matched population, and stop it."""
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        installed_version: str | None = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    (output / "versions.json").write_text(
        json.dumps(
            {"vllm": installed_version, "python": sys.version.split()[0]},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if installed_version != "0.30.0":
        raise RuntimeError(f"pinned vLLM 0.30.0 required; found {installed_version}")
    if mode == "direct-cupti":
        (output / "cupti").mkdir(mode=0o700)
    if mode == "trusted":
        (output / "nsys").mkdir(mode=0o700)
    (output / "request.json").write_text(
        json.dumps(request_body(), sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "schedule.json").write_text(
        json.dumps(
            {
                "seed": 118,
                "warmup_offsets_seconds": schedule(WARMUP_REQUESTS),
                "measured_offsets_seconds": schedule(MEASURED_REQUESTS),
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    endpoint = f"http://127.0.0.1:{port}"
    argv = _server_argv(mode, output, port)
    (output / "server-command.json").write_text(
        json.dumps(argv) + "\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    if mode == "direct-cupti":
        environment["STORMLOG_CUPTI_OUTPUT_DIR"] = str(output / "cupti")
    if mode == "trusted":
        environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    (output / "server-environment.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in environment.items()
                if key.startswith(("STORMLOG_", "CUDA_INJECTION"))
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    profiled_modes = {"public-engine", "proton", "trusted"}
    stop_completed = mode not in profiled_modes
    cupti_stop: dict[str, Any] | None = None
    gpu_pids: set[int] | None = None
    gpu_pid_error: str | None = None
    with (
        (output / "server-stdout.log").open("xb") as stdout,
        (output / "server-stderr.log").open("xb") as stderr,
    ):
        server = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=environment)
        try:
            _wait_ready(endpoint, server)

            def sender(request_id: str) -> dict[str, Any]:
                return _request(endpoint, request_id, 60)

            warmup = asyncio.run(offer_requests(WARMUP_REQUESTS, "warmup", sender))
            _write_jsonl(output / "warmup.jsonl", warmup)
            if mode in profiled_modes:
                _control(endpoint, "start_profile")
            samples: list[dict[str, Any]] = []
            sampling_stop = threading.Event()
            sampler = threading.Thread(
                target=_sample_resources,
                args=(
                    server.pid,
                    output / "resource-samples.jsonl",
                    sampling_stop,
                    samples,
                ),
            )
            started_ns = time.time_ns()
            sampler.start()
            try:
                rows = asyncio.run(
                    offer_requests(MEASURED_REQUESTS, "measured", sender)
                )
            finally:
                ended_ns = time.time_ns()
                sampling_stop.set()
                sampler.join()
            _write_jsonl(output / "requests.jsonl", rows)
            if mode in profiled_modes:
                _control(endpoint, "stop_profile")
                stop_completed = True
            if mode == "direct-cupti":
                gpu_pids, gpu_pid_error = _gpu_process_pids()
                controls = {
                    "activity_buffer_bytes": environment.get(
                        "STORMLOG_CUPTI_BUFFER_BYTES"
                    ),
                    "consumer_delay_ms": environment.get(
                        "STORMLOG_CUPTI_CONSUMER_DELAY_MS"
                    ),
                    "maximum_output_bytes": environment.get("STORMLOG_CUPTI_MAX_BYTES"),
                }
                expected_controls = (
                    {
                        key: int(value)
                        for key, value in controls.items()
                        if value is not None
                    }
                    if all(
                        value is not None and value.isdecimal()
                        for value in controls.values()
                    )
                    else None
                )
                cupti_stop = _stop_cupti_helpers(
                    output / "cupti", expected_controls=expected_controls
                )
                if expected_controls is None:
                    cupti_stop["errors"].append(
                        "CUPTI control environment is incomplete"
                    )
                    cupti_stop["complete"] = False
                cupti_stop["gpu_process_pids"] = (
                    sorted(gpu_pids) if gpu_pids is not None else None
                )
                cupti_stop["gpu_process_error"] = gpu_pid_error
                (output / "cupti-stop.json").write_text(
                    json.dumps(cupti_stop, sort_keys=True) + "\n", encoding="utf-8"
                )
            server_exited_early = server.poll() is not None
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            (output / "server-exit.json").write_text(
                json.dumps(
                    {
                        "return_code": server.returncode,
                        "profile_stop_completed": stop_completed,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    if not stop_completed:
        raise RuntimeError("profiler stop did not complete")
    if mode in {"public-engine", "proton"}:
        trace_dir = output / mode
        if not trace_dir.is_dir() or not any(
            path.is_file() for path in trace_dir.rglob("*")
        ):
            raise RuntimeError(f"{mode} produced no profiler output")
    if mode == "trusted" and not any((output / "nsys").glob("*.nsys-rep")):
        raise RuntimeError("Nsight Systems report is missing")
    result = _result(rows, started_ns, ended_ns)
    result["metrics"]["warmup_offered_requests"] = len(warmup)
    result["metrics"]["warmup_successful_requests"] = sum(
        row["status"] == "ok" for row in warmup
    )
    result["metrics"].update(_memory_metrics(samples))
    result["metrics"]["server_exited_early_count"] = int(server_exited_early)
    if mode == "direct-cupti":
        capture, loss = _cupti_capture_status(
            output, samples, required_gpu_pids=gpu_pids
        )
        if cupti_stop is None or not cupti_stop["complete"] or gpu_pid_error:
            capture["errors"].append("CUPTI stop control or GPU process query failed")
            capture["complete"] = False
            loss["vendor_activity"]["status"] = "unknown"
            loss["vendor_activity"]["reason"] = "; ".join(capture["errors"])
        (output / "cupti-capture-status.json").write_text(
            json.dumps(capture, sort_keys=True) + "\n", encoding="utf-8"
        )
        result["loss"] = loss
        result["measurement_window"]["flush_completed"] = capture["complete"]
    print(json.dumps(result, sort_keys=True))
    return (
        1
        if server_exited_early or (mode == "direct-cupti" and not capture["complete"])
        else 0
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("off", "public-engine", "proton", "direct-cupti", "trusted"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    return run(args.mode, args.output, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
