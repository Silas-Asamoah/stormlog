"""Optional on-host process and NVML memory collection for inference runs."""

from __future__ import annotations

import ctypes
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import psutil

from .host_clock import host_boot_id
from .telemetry import ServerIdentity, TelemetrySample


class _NvmlMemoryV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
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


class NvmlMemorySource:
    """Read NVML v2 memory counters from a verified GPU or MIG handle."""

    def __init__(self, *, device_index: int = 0, expected_uuid: str | None = None):
        try:
            self._lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as exc:
            raise RuntimeError("NVML is unavailable on this host") from exc
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
        try:
            current_uuid = self._uuid(self._handle)
            if current_uuid != self._handle_uuid:
                return GpuMemoryReading(None, None, "invalid", "device UUID changed")
            if (
                self._parent_handle
                and self._uuid(self._parent_handle) != self.device_uuid
            ):
                return GpuMemoryReading(
                    None, None, "invalid", "parent device UUID changed"
                )
            info = _NvmlMemoryV2()
            info.version = ctypes.sizeof(_NvmlMemoryV2) | (2 << 24)
            code = self._lib.nvmlDeviceGetMemoryInfo_v2(
                self._handle, ctypes.byref(info)
            )
            if code != 0:
                return GpuMemoryReading(
                    None, None, "missing", f"NVML memory read unavailable (code {code})"
                )
            return GpuMemoryReading(int(info.used), int(info.reserved), "valid")
        except RuntimeError as exc:
            return GpuMemoryReading(
                None, None, "invalid", f"device identity unavailable: {exc}"
            )

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
    gpu_source: GpuMemorySource | None = None,
) -> int:
    """Sample a live server process; stop on process or GPU identity change."""
    _validate_collection_options(
        run_id, pid, interval_seconds, duration_seconds, no_gpu, device_uuid, gpu_source
    )
    process = psutil.Process(pid)
    if not process.is_running():
        raise ValueError("server process is not running")
    start_ns = int(process.create_time() * 1_000_000_000)
    own_source = gpu_source is None and not no_gpu
    source = gpu_source or (
        NvmlMemorySource(device_index=device_index, expected_uuid=device_uuid)
        if not no_gpu
        else None
    )
    try:
        identity = ServerIdentity(
            host=socket.gethostname(),
            pid=pid,
            process_start_ns=start_ns,
            device_uuid=source.device_uuid if source else None,
            gpu_instance_id=source.gpu_instance_id if source else None,
            replica_id=replica_id,
            rank=rank,
            boot_id=host_boot_id(),
        )
        return _collect_loop(
            run_id,
            process,
            identity,
            source,
            output_path,
            interval_seconds,
            duration_seconds,
        )
    finally:
        if own_source and source:
            source.close()


def _validate_collection_options(
    run_id: str,
    pid: int,
    interval_seconds: float,
    duration_seconds: float | None,
    no_gpu: bool,
    device_uuid: str | None,
    gpu_source: GpuMemorySource | None,
) -> None:
    if not run_id or pid <= 0 or interval_seconds < 0.01:
        raise ValueError("run_id, positive pid, and interval >= 0.01s are required")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration must be positive")
    if no_gpu and (device_uuid or gpu_source):
        raise ValueError("--no-gpu cannot be combined with a GPU source")


def _collect_loop(
    run_id: str,
    process: psutil.Process,
    identity: ServerIdentity,
    source: GpuMemorySource | None,
    output_path: str | Path,
    interval_seconds: float,
    duration_seconds: float | None,
) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration_seconds if duration_seconds else None
    next_poll = time.monotonic()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(max(0, next_poll - time.monotonic()))
            next_poll += interval_seconds
            samples, stop = _poll_records(
                run_id, process, identity, source, round(interval_seconds * 1000)
            )
            for sample in samples:
                handle.write(json.dumps(sample.to_record(), sort_keys=True) + "\n")
            handle.flush()
            count += 1
            if stop:
                break
    return count


def _poll_records(
    run_id: str,
    process: psutil.Process,
    identity: ServerIdentity,
    source: GpuMemorySource | None,
    interval_ms: int,
) -> tuple[list[TelemetrySample], bool]:
    observed_at_ns = time.time_ns()
    process_sample, same_process = _process_sample(
        run_id, process, identity, observed_at_ns, interval_ms
    )
    samples = [process_sample]
    gpu_invalid = False
    if source:
        gpu_samples, gpu_invalid = _gpu_samples(
            run_id, identity, source, same_process, observed_at_ns, interval_ms
        )
        samples.extend(gpu_samples)
    return samples, not same_process or gpu_invalid


def _process_sample(
    run_id: str,
    process: psutil.Process,
    identity: ServerIdentity,
    observed_at_ns: int,
    interval_ms: int,
) -> tuple[TelemetrySample, bool]:
    try:
        same_process = (
            process.is_running()
            and int(process.create_time() * 1_000_000_000) == identity.process_start_ns
        )
        rss = int(process.memory_info().rss) if same_process else None
    except psutil.Error:
        same_process, rss = False, None
    return (
        TelemetrySample(
            run_id=run_id,
            identity=identity,
            observed_at_ns=observed_at_ns,
            metric="process_rss_bytes",
            value_bytes=rss,
            state="valid" if rss is not None else "invalid",
            source="psutil",
            interval_ms=interval_ms,
            detail=None if rss is not None else "server process ended or restarted",
        ),
        same_process,
    )


def _gpu_samples(
    run_id: str,
    identity: ServerIdentity,
    source: GpuMemorySource,
    same_process: bool,
    observed_at_ns: int,
    interval_ms: int,
) -> tuple[list[TelemetrySample], bool]:
    reading = (
        source.read()
        if same_process
        else GpuMemoryReading(
            None, None, "invalid", "server process ended or restarted"
        )
    )
    prefix = "instance" if identity.gpu_instance_id else "device"
    samples = [
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
    return samples, reading.state == "invalid"
