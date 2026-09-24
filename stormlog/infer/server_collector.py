"""Optional on-host process and NVML memory collection for inference runs."""

from __future__ import annotations

import ctypes
import json
import socket
import time
from pathlib import Path
from typing import Protocol

import psutil

from .telemetry import ServerIdentity, TelemetrySample


class _NvmlMemoryV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class GpuMemorySource(Protocol):
    device_uuid: str
    gpu_instance_id: str | None

    def read(self) -> tuple[int | None, int | None, str | None]:
        """Return allocated, reserved, and an error if identity or read fails."""

    def close(self) -> None: ...


class NvmlMemorySource:
    """Read NVML v2 memory counters from a verified GPU or MIG handle."""

    def __init__(self, *, device_index: int = 0, expected_uuid: str | None = None):
        try:
            self._lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as exc:
            raise RuntimeError("NVML is unavailable on this host") from exc
        lib = self._lib
        lib.nvmlInit_v2.restype = ctypes.c_int
        lib.nvmlShutdown.restype = ctypes.c_int
        lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        lib.nvmlDeviceGetUUID.restype = ctypes.c_int
        lib.nvmlDeviceGetMemoryInfo_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_NvmlMemoryV2),
        ]
        lib.nvmlDeviceGetMemoryInfo_v2.restype = ctypes.c_int
        self._closed = False
        if lib.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        try:
            self._handle = ctypes.c_void_p()
            if expected_uuid is None:
                code = lib.nvmlDeviceGetHandleByIndex_v2(
                    device_index, ctypes.byref(self._handle)
                )
            else:
                lib.nvmlDeviceGetHandleByUUID.argtypes = [
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                lib.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
                code = lib.nvmlDeviceGetHandleByUUID(
                    expected_uuid.encode(), ctypes.byref(self._handle)
                )
            if code != 0:
                raise RuntimeError(f"NVML device lookup failed (code {code})")
            self._handle_uuid = self._uuid(self._handle)
            if expected_uuid and self._handle_uuid != expected_uuid:
                raise RuntimeError("NVML device UUID differs from requested UUID")
            self.gpu_instance_id = (
                self._handle_uuid if self._handle_uuid.startswith("MIG-") else None
            )
            self.device_uuid = self._handle_uuid
            if self.gpu_instance_id:
                lib.nvmlDeviceGetDeviceHandleFromMigDeviceHandle.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                lib.nvmlDeviceGetDeviceHandleFromMigDeviceHandle.restype = ctypes.c_int
                parent = ctypes.c_void_p()
                code = lib.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(
                    self._handle, ctypes.byref(parent)
                )
                if code != 0:
                    raise RuntimeError(f"NVML MIG parent lookup failed (code {code})")
                self.device_uuid = self._uuid(parent)
        except Exception:
            self.close()
            raise

    def _uuid(self, handle: ctypes.c_void_p) -> str:
        buffer = ctypes.create_string_buffer(128)
        code = self._lib.nvmlDeviceGetUUID(handle, buffer, len(buffer))
        if code != 0:
            raise RuntimeError(f"NVML UUID lookup failed (code {code})")
        return buffer.value.decode()

    def read(self) -> tuple[int | None, int | None, str | None]:
        try:
            current_uuid = self._uuid(self._handle)
            if current_uuid != self._handle_uuid:
                return None, None, "device UUID changed"
            info = _NvmlMemoryV2()
            info.version = ctypes.sizeof(_NvmlMemoryV2) | (2 << 24)
            code = self._lib.nvmlDeviceGetMemoryInfo_v2(
                self._handle, ctypes.byref(info)
            )
            if code != 0:
                return None, None, f"NVML memory read unavailable (code {code})"
            return int(info.used), int(info.reserved), None
        except RuntimeError as exc:
            return None, None, str(exc)

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
    if not run_id or pid <= 0 or interval_seconds < 0.01:
        raise ValueError("run_id, positive pid, and interval >= 0.01s are required")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration must be positive")
    if no_gpu and (device_uuid or gpu_source):
        raise ValueError("--no-gpu cannot be combined with a GPU source")
    process = psutil.Process(pid)
    if not process.is_running():
        raise ValueError("server process is not running")
    start_ns = int(process.create_time() * 1_000_000_000)
    own_source = gpu_source is None and not no_gpu
    source = gpu_source or (NvmlMemorySource(device_index=device_index, expected_uuid=device_uuid) if not no_gpu else None)
    try:
        identity = ServerIdentity(
            host=socket.gethostname(),
            pid=pid,
            process_start_ns=start_ns,
            device_uuid=source.device_uuid if source else None,
            gpu_instance_id=source.gpu_instance_id if source else None,
            replica_id=replica_id,
            rank=rank,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        interval_ms = round(interval_seconds * 1000)
        deadline = time.monotonic() + duration_seconds if duration_seconds else None
        next_poll = time.monotonic()
        with path.open("w", encoding="utf-8") as handle:
            while deadline is None or time.monotonic() < deadline:
                time.sleep(max(0, next_poll - time.monotonic()))
                next_poll += interval_seconds
                observed_at_ns = time.time_ns()
                try:
                    same_process = process.is_running() and int(
                        process.create_time() * 1_000_000_000
                    ) == start_ns
                    rss = int(process.memory_info().rss) if same_process else None
                except psutil.Error:
                    same_process, rss = False, None
                samples = [
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
                    )
                ]
                if source:
                    used, reserved, error = source.read() if same_process else (None, None, "server process ended or restarted")
                    state = "valid" if error is None else (
                        "invalid" if "UUID changed" in error or not same_process else "missing"
                    )
                    prefix = "instance" if identity.gpu_instance_id else "device"
                    for metric, value in ((f"{prefix}_memory_used_bytes", used), (f"{prefix}_memory_reserved_bytes", reserved)):
                        samples.append(
                            TelemetrySample(
                                run_id=run_id,
                                identity=identity,
                                observed_at_ns=observed_at_ns,
                                metric=metric,
                                value_bytes=value if state == "valid" else None,
                                state=state,
                                source="nvml-v2",
                                interval_ms=interval_ms,
                                detail=error,
                            )
                        )
                for sample in samples:
                    handle.write(json.dumps(sample.to_record(), sort_keys=True) + "\n")
                handle.flush()
                count += 1
                if not same_process or (source and state == "invalid"):
                    break
        return count
    finally:
        if own_source and source:
            source.close()
