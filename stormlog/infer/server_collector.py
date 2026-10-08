"""Optional on-host process and NVML memory collection for inference runs."""

from __future__ import annotations

import ctypes
import json
import math
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import psutil

from .errors import InferUsageError
from .host_clock import host_boot_id
from .telemetry import ServerIdentity, TelemetrySample, validate_group_membership

STOP_DURATION_ELAPSED = "duration_elapsed"
STOP_REQUESTED = "stop_requested"
STOP_SERVER_PROCESS_ENDED = "server_process_ended"
STOP_GPU_IDENTITY_CHANGED = "gpu_identity_changed"

_PROCESS_ENDED_DETAIL = "server process ended or its PID was reused"
_NVML_SUCCESS = 0
_NVML_ERROR_INSUFFICIENT_SIZE = 7


class _NvmlMemoryV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlProcessInfo(ctypes.Structure):
    """``nvmlProcessInfo_t``; the v2 and v3 process queries share this layout."""

    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


def _configure_nvml_library(lib: ctypes.CDLL) -> None:
    lib.nvmlInit_v2.restype = ctypes.c_int
    lib.nvmlShutdown.restype = ctypes.c_int
    lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    lib.nvmlDeviceGetUUID.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    lib.nvmlDeviceGetUUID.restype = ctypes.c_int
    lib.nvmlDeviceGetMemoryInfo_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NvmlMemoryV2),
    ]
    lib.nvmlDeviceGetMemoryInfo_v2.restype = ctypes.c_int


# NVML_ERROR_INVALID_ARGUMENT and NVML_ERROR_NOT_FOUND: the host has no such device.
_NVML_NO_SUCH_DEVICE = frozenset({2, 6})


def _lookup_nvml_handle(
    lib: ctypes.CDLL, device_index: int, expected_uuid: str | None
) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    if expected_uuid is None:
        code = lib.nvmlDeviceGetHandleByIndex_v2(device_index, ctypes.byref(handle))
    else:
        lib.nvmlDeviceGetHandleByUUID.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
        code = lib.nvmlDeviceGetHandleByUUID(
            expected_uuid.encode(), ctypes.byref(handle)
        )
    if code in _NVML_NO_SUCH_DEVICE:
        named = (
            f"--device-uuid {expected_uuid}"
            if expected_uuid is not None
            else f"--device-index {device_index}"
        )
        raise InferUsageError(f"{named}: no such GPU on this host (NVML code {code})")
    if code != 0:
        raise RuntimeError(f"NVML device lookup failed (code {code})")
    return handle


def _mig_parent_handle(lib: ctypes.CDLL, handle: ctypes.c_void_p) -> ctypes.c_void_p:
    lib.nvmlDeviceGetDeviceHandleFromMigDeviceHandle.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.nvmlDeviceGetDeviceHandleFromMigDeviceHandle.restype = ctypes.c_int
    parent = ctypes.c_void_p()
    code = lib.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(
        handle, ctypes.byref(parent)
    )
    if code != 0:
        raise RuntimeError(f"NVML MIG parent lookup failed (code {code})")
    return parent


def _running_compute_pids(function: Any, handle: ctypes.c_void_p) -> set[int] | None:
    """Call an ``nvmlDeviceGetComputeRunningProcesses`` variant for its PIDs."""
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(_NvmlProcessInfo),
    ]
    function.restype = ctypes.c_int
    count = ctypes.c_uint(0)
    code = function(handle, ctypes.byref(count), None)
    if code == _NVML_SUCCESS:
        return set()
    if code != _NVML_ERROR_INSUFFICIENT_SIZE:
        return None
    # Leave room for processes that start between the size query and the read.
    capacity = count.value + 8
    infos = (_NvmlProcessInfo * capacity)()
    count = ctypes.c_uint(capacity)
    if function(handle, ctypes.byref(count), infos) != _NVML_SUCCESS:
        return None
    return {int(infos[index].pid) for index in range(count.value)}


class GpuMemorySource(Protocol):
    device_uuid: str
    gpu_instance_id: str | None

    def read(self) -> GpuMemoryReading:
        """Return memory values and their availability state."""

    def close(self) -> None: ...


@dataclass(frozen=True)
class GpuMemoryReading:
    used_bytes: int | None
    reserved_bytes: int | None
    state: str
    detail: str | None = None


@dataclass(frozen=True)
class CollectionResult:
    """How many polls a collection wrote and why it stopped."""

    polls: int
    stop_reason: str
    detail: str | None = None
    warnings: tuple[str, ...] = ()


class NvmlUnavailableError(RuntimeError):
    """The NVML library cannot be loaded on this host."""


class NvmlMemorySource:
    """Read NVML v2 memory counters from a verified GPU or MIG handle."""

    def __init__(self, *, device_index: int = 0, expected_uuid: str | None = None):
        try:
            self._lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as exc:
            raise NvmlUnavailableError("NVML is unavailable on this host") from exc
        lib = self._lib
        _configure_nvml_library(lib)
        self._closed = False
        if lib.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        try:
            self._handle = _lookup_nvml_handle(lib, device_index, expected_uuid)
            self._handle_uuid = self._uuid(self._handle)
            if expected_uuid and self._handle_uuid != expected_uuid:
                raise RuntimeError("NVML device UUID differs from requested UUID")
            self.gpu_instance_id = (
                self._handle_uuid if self._handle_uuid.startswith("MIG-") else None
            )
            self.device_uuid = self._handle_uuid
            self._parent_handle = (
                _mig_parent_handle(lib, self._handle) if self.gpu_instance_id else None
            )
            if self._parent_handle:
                self.device_uuid = self._uuid(self._parent_handle)
        except Exception:
            self.close()
            raise

    def _uuid(self, handle: ctypes.c_void_p) -> str:
        buffer = ctypes.create_string_buffer(128)
        code = self._lib.nvmlDeviceGetUUID(handle, buffer, len(buffer))
        if code != 0:
            raise RuntimeError(f"NVML UUID lookup failed (code {code})")
        return buffer.value.decode()

    def read(self) -> GpuMemoryReading:
        issue = self._identity_issue()
        if issue is not None:
            return issue
        info = _NvmlMemoryV2()
        info.version = ctypes.sizeof(_NvmlMemoryV2) | (2 << 24)
        code = self._lib.nvmlDeviceGetMemoryInfo_v2(self._handle, ctypes.byref(info))
        if code != 0:
            return GpuMemoryReading(
                None, None, "missing", f"NVML memory read unavailable (code {code})"
            )
        return GpuMemoryReading(int(info.used), int(info.reserved), "valid")

    def _identity_issue(self) -> GpuMemoryReading | None:
        """Only a different UUID invalidates; an unreadable UUID is missing data."""
        try:
            current_uuid = self._uuid(self._handle)
            parent_uuid = (
                self._uuid(self._parent_handle) if self._parent_handle else None
            )
        except RuntimeError as exc:
            return GpuMemoryReading(
                None, None, "missing", f"device identity unreadable: {exc}"
            )
        if current_uuid != self._handle_uuid:
            return GpuMemoryReading(None, None, "invalid", "device UUID changed")
        if parent_uuid is not None and parent_uuid != self.device_uuid:
            return GpuMemoryReading(None, None, "invalid", "parent device UUID changed")
        return None

    def compute_pids(self) -> set[int] | None:
        """Return PIDs NVML reports on this GPU, or None when NVML cannot say."""
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
        ):
            function = getattr(self._lib, name, None)
            if function is not None:
                return _running_compute_pids(function, self._handle)
        return None

    def close(self) -> None:
        if not self._closed:
            self._lib.nvmlShutdown()
            self._closed = True


def collect_server_telemetry(
    *,
    run_id: str,
    pid: int,
    output_path: str | Path,
    interval_seconds: float = 0.1,
    duration_seconds: float | None = None,
    device_index: int = 0,
    device_uuid: str | None = None,
    no_gpu: bool = False,
    replica_id: str | None = None,
    rank: int | None = None,
    group_id: str | None = None,
    world_size: int | None = None,
    gpu_source: GpuMemorySource | None = None,
    stop_event: threading.Event | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> CollectionResult:
    """Sample a live server process until the duration, a stop, or a change.

    Collection stops when ``stop_event`` is set or the caller is interrupted,
    when the duration elapses, when the process ends, or when the GPU identity
    changes. The result says which, so callers can tell a clean stop from one
    that leaves later case windows unobserved. ``group_id``, ``rank`` and
    ``world_size`` declare this process as one member of a server group, such as
    one tensor-parallel worker.
    """
    try:
        _validate_collection_options(run_id, pid, no_gpu, device_uuid, gpu_source)
        _validate_timing(interval_seconds, duration_seconds)
        validate_group_membership(group_id, rank, world_size)
        process = _server_process(pid)
    except ValueError as exc:
        # Each check is about the caller's options or the process it named.
        raise InferUsageError(str(exc)) from exc
    source, own_source = _gpu_source(gpu_source, no_gpu, device_index, device_uuid)
    try:
        warnings = _gpu_process_warnings(process, source)
        for message in warnings:
            if on_warning is not None:
                on_warning(message)
        identity = _server_identity(
            process, source, replica_id, (group_id, rank, world_size)
        )
        polls, stop_reason, detail = _collect_loop(
            run_id,
            process,
            identity,
            source,
            Path(output_path),
            interval_seconds,
            duration_seconds,
            stop_event or threading.Event(),
        )
        return CollectionResult(polls, stop_reason, detail, tuple(warnings))
    finally:
        if own_source and source:
            source.close()


def _server_process(pid: int) -> psutil.Process:
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess as exc:
        raise ValueError(f"no process with pid {pid}") from exc
    if not process.is_running():
        raise ValueError("server process is not running")
    return process


def _validate_collection_options(
    run_id: str,
    pid: int,
    no_gpu: bool,
    device_uuid: str | None,
    gpu_source: GpuMemorySource | None,
) -> None:
    if not run_id or pid <= 0:
        raise ValueError("run_id and a positive pid are required")
    if no_gpu and (device_uuid or gpu_source):
        raise ValueError("--no-gpu cannot be combined with a GPU source")


def _validate_timing(interval_seconds: float, duration_seconds: float | None) -> None:
    # NaN compares false with everything, so check finiteness explicitly.
    if not math.isfinite(interval_seconds) or interval_seconds < 0.01:
        raise ValueError("interval must be a finite number of seconds >= 0.01")
    if duration_seconds is not None and (
        not math.isfinite(duration_seconds) or duration_seconds <= 0
    ):
        raise ValueError("duration must be a finite, positive number of seconds")


def _gpu_source(
    gpu_source: GpuMemorySource | None,
    no_gpu: bool,
    device_index: int,
    device_uuid: str | None,
) -> tuple[GpuMemorySource | None, bool]:
    """Return the GPU source and whether this collection owns (closes) it."""
    if gpu_source is not None or no_gpu:
        return gpu_source, False
    return NvmlMemorySource(device_index=device_index, expected_uuid=device_uuid), True


def _server_identity(
    process: psutil.Process,
    source: GpuMemorySource | None,
    replica_id: str | None,
    group: tuple[str | None, int | None, int | None],
) -> ServerIdentity:
    group_id, rank, world_size = group
    return ServerIdentity(
        host=socket.gethostname(),
        pid=process.pid,
        process_start_ns=int(process.create_time() * 1_000_000_000),
        device_uuid=source.device_uuid if source else None,
        gpu_instance_id=source.gpu_instance_id if source else None,
        replica_id=replica_id,
        rank=rank,
        boot_id=host_boot_id(),
        group_id=group_id,
        world_size=world_size,
    )


def _gpu_process_warnings(
    process: psutil.Process, source: GpuMemorySource | None
) -> list[str]:
    """Warn when NVML does not show the server PID on the sampled GPU.

    NVML numbers GPUs in its own order, and CUDA_VISIBLE_DEVICES renumbers them
    for the server, so an index can name a different GPU than the server uses.
    """
    list_compute_pids = getattr(source, "compute_pids", None)
    if source is None or list_compute_pids is None:
        return []
    return describe_gpu_process_match(
        process.pid,
        source.device_uuid,
        list_compute_pids(),
        _descendant_pids(process),
    )


def describe_gpu_process_match(
    pid: int,
    device_uuid: str,
    gpu_pids: set[int] | None,
    descendant_pids: set[int],
) -> list[str]:
    """Explain how the server PID relates to the processes NVML sees on a GPU."""
    if gpu_pids is None:
        return [
            f"could not list compute processes on {device_uuid}; "
            f"cannot confirm that PID {pid} uses this GPU"
        ]
    if pid in gpu_pids:
        return []
    workers = sorted(descendant_pids & gpu_pids)
    if workers:
        listed = ", ".join(str(worker) for worker in workers)
        return [
            f"PID {pid} has no compute context on {device_uuid}, but its child "
            f"process(es) {listed} do. GPU memory is device-wide, but RSS is "
            f"measured for PID {pid} only; pass --pid {workers[0]} to measure "
            "the GPU worker instead"
        ]
    seen = ", ".join(str(item) for item in sorted(gpu_pids)) or "none"
    return [
        f"PID {pid} has no compute context on {device_uuid} (NVML reports PIDs: "
        f"{seen}). If the server uses a different GPU, pass --device-uuid. "
        "Inside a container NVML may report host PIDs, so this check can "
        "be wrong there"
    ]


def _descendant_pids(process: psutil.Process) -> set[int]:
    try:
        return {child.pid for child in process.children(recursive=True)}
    except psutil.Error:
        return set()


def _collect_loop(
    run_id: str,
    process: psutil.Process,
    identity: ServerIdentity,
    source: GpuMemorySource | None,
    path: Path,
    interval_seconds: float,
    duration_seconds: float | None,
    stop_event: threading.Event,
) -> tuple[int, str, str | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = None if duration_seconds is None else started + duration_seconds
    next_poll = started
    interval_ms = round(interval_seconds * 1000)
    polls = 0
    with path.open("w", encoding="utf-8") as handle:
        try:
            while True:
                waited = _wait_for_poll(stop_event, next_poll, deadline)
                if waited is not None:
                    return polls, waited, None
                samples, stop = _poll_records(
                    run_id, process, identity, source, interval_ms
                )
                # One write per poll keeps each poll's lines together on disk.
                handle.write(
                    "".join(
                        json.dumps(sample.to_record(), sort_keys=True) + "\n"
                        for sample in samples
                    )
                )
                handle.flush()
                polls += 1
                if stop is not None:
                    return polls, stop[0], stop[1]
                next_poll = next_poll_time(
                    next_poll, interval_seconds, time.monotonic()
                )
        except KeyboardInterrupt:
            return polls, STOP_REQUESTED, "interrupted"


def _wait_for_poll(
    stop_event: threading.Event, next_poll: float, deadline: float | None
) -> str | None:
    """Sleep until the next poll; return a stop reason if collection should end."""
    wake = next_poll if deadline is None else min(next_poll, deadline)
    if stop_event.wait(max(0.0, wake - time.monotonic())):
        return STOP_REQUESTED
    if deadline is not None and time.monotonic() >= deadline:
        return STOP_DURATION_ELAPSED
    return None


def next_poll_time(previous: float, interval: float, now: float) -> float:
    """Keep a steady cadence, but skip missed polls instead of bursting."""
    scheduled = previous + interval
    return scheduled if scheduled > now else now + interval


def _poll_records(
    run_id: str,
    process: psutil.Process,
    identity: ServerIdentity,
    source: GpuMemorySource | None,
    interval_ms: int,
) -> tuple[list[TelemetrySample], tuple[str, str | None] | None]:
    observed_at_ns = time.time_ns()
    process_sample = _process_sample(
        run_id, process, identity, observed_at_ns, interval_ms
    )
    samples = [process_sample]
    if source:
        samples.extend(
            _gpu_samples(
                run_id,
                identity,
                source,
                process_sample.state == "invalid",
                observed_at_ns,
                interval_ms,
            )
        )
    return samples, _stop_reason(samples)


def _stop_reason(samples: list[TelemetrySample]) -> tuple[str, str | None] | None:
    process_sample = samples[0]
    if process_sample.state == "invalid":
        return STOP_SERVER_PROCESS_ENDED, process_sample.detail
    changed = next((s for s in samples[1:] if s.state == "invalid"), None)
    if changed is not None:
        return STOP_GPU_IDENTITY_CHANGED, changed.detail
    return None


def _process_sample(
    run_id: str,
    process: psutil.Process,
    identity: ServerIdentity,
    observed_at_ns: int,
    interval_ms: int,
) -> TelemetrySample:
    state, rss, detail = read_process_rss(process)
    return TelemetrySample(
        run_id=run_id,
        identity=identity,
        observed_at_ns=observed_at_ns,
        metric="process_rss_bytes",
        value_bytes=rss,
        state=state,
        source="psutil",
        interval_ms=interval_ms,
        detail=detail,
    )


def read_process_rss(process: psutil.Process) -> tuple[str, int | None, str | None]:
    """Return ``(state, rss, detail)``; only a gone or replaced process is invalid.

    ``is_running`` compares the process creation time with the original, so it
    also detects a PID that the OS reused for another process. A zombie has
    ended too: on Linux it still counts as running, with an RSS of 0, until
    its parent reaps it.
    """
    try:
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return "invalid", None, _PROCESS_ENDED_DETAIL
        return "valid", int(process.memory_info().rss), None
    except psutil.NoSuchProcess:  # includes ZombieProcess
        return "invalid", None, _PROCESS_ENDED_DETAIL
    except psutil.AccessDenied:
        return "missing", None, "access denied reading server process memory"
    except psutil.Error as exc:
        return "missing", None, f"server process memory unavailable: {exc}"


def _gpu_samples(
    run_id: str,
    identity: ServerIdentity,
    source: GpuMemorySource,
    process_ended: bool,
    observed_at_ns: int,
    interval_ms: int,
) -> list[TelemetrySample]:
    reading = (
        GpuMemoryReading(None, None, "invalid", _PROCESS_ENDED_DETAIL)
        if process_ended
        else source.read()
    )
    prefix = "instance" if identity.gpu_instance_id else "device"
    return [
        TelemetrySample(
            run_id=run_id,
            identity=identity,
            observed_at_ns=observed_at_ns,
            metric=f"{prefix}_memory_{name}_bytes",
            value_bytes=value,
            state=reading.state,
            source="nvml-v2",
            interval_ms=interval_ms,
            detail=reading.detail,
        )
        for name, value in (
            ("used", reading.used_bytes),
            ("reserved", reading.reserved_bytes),
        )
    ]
