"""The GPUs a vLLM server uses, and their settings, from NVML.

The server's GPUs are the devices on which NVML lists one of the server's
processes as a compute process: CUDA_VISIBLE_DEVICES renumbers devices for
the server, so an index alone cannot name them. Each device gives its
settings (driver, power limits, application clocks, persistence, ECC, MIG,
compute mode), which identify the hardware a run used, and a reading of
what changes during a run (SM clock, temperature, clock event reasons),
which a later description compares as drift.

Each field is either a value or the NVML error that kept it unread; an
unreadable field is never filled in.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .server_collector import NvmlUnavailableError, running_compute_pids

NVML = "nvml"

_NVML_SUCCESS = 0
_NAME_BUFFER = 96
_VERSION_BUFFER = 80
_UUID_BUFFER = 96
_CLOCK_GRAPHICS, _CLOCK_SM, _CLOCK_MEM = 0, 1, 2
_TEMPERATURE_GPU = 0
_COMPUTE_MODES = {
    0: "default",
    1: "exclusive_thread",
    2: "prohibited",
    3: "exclusive_process",
}
# nvmlClocksEventReasons bits (formerly "throttle reasons").
CLOCK_EVENT_REASONS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}

SETTINGS = (
    "name",
    "power_limit_w",
    "enforced_power_limit_w",
    "application_clocks_mhz",
    "persistence_mode",
    "ecc_enabled",
    "mig_enabled",
    "compute_mode",
)
SERIES = ("sm_clock_mhz", "temperature_c", "clock_event_reasons")


class NvmlError(RuntimeError):
    """An NVML call that did not succeed, with its return code.

    ``code`` is None when this NVML library does not have the function.
    """

    def __init__(self, function: str, code: int | None) -> None:
        detail = "is missing" if code is None else f"returned {code}"
        super().__init__(f"{function} {detail}")
        self.function = function
        self.code = code


class GpuReader(Protocol):
    """What a GPU description reads; NVML on a real host."""

    def driver_version(self) -> str: ...

    def cuda_driver_version(self) -> int: ...

    def device_count(self) -> int: ...

    def uuid(self, index: int) -> str: ...

    def compute_pids(self, index: int) -> set[int] | None: ...

    def read(self, index: int, name: str) -> Any:
        """One field of ``SETTINGS`` or ``SERIES``; raises NvmlError."""

    def close(self) -> None: ...


@dataclass(frozen=True)
class GpuDevice:
    """One device, its settings and one reading of what drifts."""

    index: int
    uuid: str
    server_pids: tuple[int, ...]
    settings: Mapping[str, Any]
    series: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "server_pids": list(self.server_pids),
            "settings": dict(self.settings),
            "series": dict(self.series),
        }


@dataclass(frozen=True)
class GpuDescription:
    """The host's driver and the devices the server uses."""

    driver_version: Any
    cuda_driver_version: Any
    device_count: int
    devices: tuple[GpuDevice, ...]
    issues: tuple[str, ...] = field(default=())

    @property
    def server_uuids(self) -> list[str]:
        return sorted(device.uuid for device in self.devices if device.server_pids)

    def to_record(self) -> dict[str, Any]:
        return {
            "source": NVML,
            "driver_version": self.driver_version,
            "cuda_driver_version": self.cuda_driver_version,
            "device_count": self.device_count,
            "server_uuids": self.server_uuids,
            "devices": [device.to_record() for device in self.devices],
            "issues": list(self.issues),
        }


def describe_gpus(reader: GpuReader, server_pids: Iterable[int]) -> GpuDescription:
    """Every device NVML lists, marked with the server processes on it."""
    pids = set(server_pids)
    devices = []
    unlisted = 0
    for index in range(reader.device_count()):
        on_device = reader.compute_pids(index)
        if on_device is None:
            unlisted += 1
        devices.append(
            GpuDevice(
                index=index,
                uuid=reader.uuid(index),
                server_pids=tuple(sorted(pids & (on_device or set()))),
                settings=_fields(reader, index, SETTINGS),
                series=_fields(reader, index, SERIES),
            )
        )
    description = GpuDescription(
        driver_version=_value(reader.driver_version),
        cuda_driver_version=_value(lambda: _cuda_version(reader.cuda_driver_version())),
        device_count=len(devices),
        devices=tuple(devices),
    )
    return replace(description, issues=tuple(_issues(description, unlisted)))


def read_series(reader: GpuReader, uuids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """A fresh reading of what drifts, for the devices with these UUIDs."""
    wanted = set(uuids)
    return {
        reader.uuid(index): _fields(reader, index, SERIES)
        for index in range(reader.device_count())
        if reader.uuid(index) in wanted
    }


def clock_event_reasons(mask: int) -> list[str]:
    """The names of the bits set in an NVML clock event reason mask."""
    names = [name for bit, name in CLOCK_EVENT_REASONS.items() if mask & bit]
    unknown = mask & ~sum(CLOCK_EVENT_REASONS)
    return names + ([f"unknown_0x{unknown:x}"] if unknown else [])


def _fields(reader: GpuReader, index: int, names: Iterable[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in names:
        try:
            values[name] = reader.read(index, name)
        except NvmlError as exc:
            values[name] = _unavailable(exc)
    return values


def _value(read: Callable[[], Any]) -> Any:
    try:
        return read()
    except NvmlError as exc:
        return _unavailable(exc)


def _unavailable(error: NvmlError) -> dict[str, str]:
    if error.code is None:
        return {"unavailable": f"{error.function} missing from this NVML"}
    return {"unavailable": f"NVML code {error.code}"}


def _cuda_version(raw: int) -> str:
    """NVML's 12080 as "12.8"."""
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def _issues(description: GpuDescription, unlisted: int) -> list[str]:
    issues = []
    if unlisted:
        issues.append(f"compute processes unreadable on {unlisted} device(s)")
    if not description.server_uuids:
        issues.append(
            "no device lists a server process; inside a container NVML may "
            "report host PIDs"
        )
    return issues


# ------------------------------------------------------------------- NVML


class NvmlGpuReader:
    """``GpuReader`` over the NVML library, through ctypes."""

    def __init__(self) -> None:
        try:
            self._lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as exc:
            raise NvmlUnavailableError("NVML is unavailable on this host") from exc
        self._closed = False
        self._call("nvmlInit_v2")
        self._handles: dict[int, ctypes.c_void_p] = {}

    def driver_version(self) -> str:
        buffer = ctypes.create_string_buffer(_VERSION_BUFFER)
        self._call("nvmlSystemGetDriverVersion", buffer, ctypes.c_uint(len(buffer)))
        return buffer.value.decode()

    def cuda_driver_version(self) -> int:
        version = ctypes.c_int()
        self._call("nvmlSystemGetCudaDriverVersion_v2", ctypes.byref(version))
        return int(version.value)

    def device_count(self) -> int:
        count = ctypes.c_uint()
        self._call("nvmlDeviceGetCount_v2", ctypes.byref(count))
        return int(count.value)

    def uuid(self, index: int) -> str:
        buffer = ctypes.create_string_buffer(_UUID_BUFFER)
        self._call(
            "nvmlDeviceGetUUID", self._handle(index), buffer, ctypes.c_uint(len(buffer))
        )
        return buffer.value.decode()

    def compute_pids(self, index: int) -> set[int] | None:
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
        ):
            function = getattr(self._lib, name, None)
            if function is not None:
                return running_compute_pids(function, self._handle(index))
        return None

    def read(self, index: int, name: str) -> Any:
        return _READERS[name](self, self._handle(index))

    def close(self) -> None:
        if not self._closed:
            self._lib.nvmlShutdown()
            self._closed = True

    def _handle(self, index: int) -> ctypes.c_void_p:
        if index not in self._handles:
            handle = ctypes.c_void_p()
            self._call(
                "nvmlDeviceGetHandleByIndex_v2",
                ctypes.c_uint(index),
                ctypes.byref(handle),
            )
            self._handles[index] = handle
        return self._handles[index]

    def _call(self, function: str, *arguments: Any) -> None:
        call = getattr(self._lib, function, None)
        if call is None:
            raise NvmlError(function, None)
        # A CDLL function returns a C int unless told otherwise: NVML's code.
        code = int(call(*arguments))
        if code != _NVML_SUCCESS:
            raise NvmlError(function, code)

    def _uint(self, function: str, *arguments: Any) -> int:
        value = ctypes.c_uint()
        self._call(function, *arguments, ctypes.byref(value))
        return int(value.value)

    def _pair(self, function: str, handle: ctypes.c_void_p) -> tuple[int, int]:
        current, pending = ctypes.c_uint(), ctypes.c_uint()
        self._call(function, handle, ctypes.byref(current), ctypes.byref(pending))
        return int(current.value), int(pending.value)


def _name(reader: NvmlGpuReader, handle: ctypes.c_void_p) -> str:
    buffer = ctypes.create_string_buffer(_NAME_BUFFER)
    reader._call("nvmlDeviceGetName", handle, buffer, ctypes.c_uint(len(buffer)))
    return buffer.value.decode()


def _application_clocks(
    reader: NvmlGpuReader, handle: ctypes.c_void_p
) -> dict[str, int]:
    function = "nvmlDeviceGetApplicationsClock"
    return {
        "graphics": reader._uint(function, handle, ctypes.c_int(_CLOCK_GRAPHICS)),
        "sm": reader._uint(function, handle, ctypes.c_int(_CLOCK_SM)),
        "memory": reader._uint(function, handle, ctypes.c_int(_CLOCK_MEM)),
    }


def _event_reasons(reader: NvmlGpuReader, handle: ctypes.c_void_p) -> list[str]:
    mask = ctypes.c_ulonglong()
    try:
        reader._call(
            "nvmlDeviceGetCurrentClocksEventReasons", handle, ctypes.byref(mask)
        )
    except NvmlError:
        reader._call(
            "nvmlDeviceGetCurrentClocksThrottleReasons", handle, ctypes.byref(mask)
        )
    return clock_event_reasons(int(mask.value))


_READERS: dict[str, Callable[[NvmlGpuReader, ctypes.c_void_p], Any]] = {
    "name": _name,
    "power_limit_w": lambda r, h: r._uint("nvmlDeviceGetPowerManagementLimit", h)
    / 1000,
    "enforced_power_limit_w": lambda r, h: r._uint("nvmlDeviceGetEnforcedPowerLimit", h)
    / 1000,
    "application_clocks_mhz": _application_clocks,
    "persistence_mode": lambda r, h: r._uint("nvmlDeviceGetPersistenceMode", h) == 1,
    "ecc_enabled": lambda r, h: r._pair("nvmlDeviceGetEccMode", h)[0] == 1,
    "mig_enabled": lambda r, h: r._pair("nvmlDeviceGetMigMode", h)[0] == 1,
    "compute_mode": lambda r, h: _COMPUTE_MODES.get(
        r._uint("nvmlDeviceGetComputeMode", h), "unknown"
    ),
    "sm_clock_mhz": lambda r, h: r._uint(
        "nvmlDeviceGetClockInfo", h, ctypes.c_int(_CLOCK_SM)
    ),
    "temperature_c": lambda r, h: r._uint(
        "nvmlDeviceGetTemperature", h, ctypes.c_int(_TEMPERATURE_GPU)
    ),
    "clock_event_reasons": _event_reasons,
}


__all__ = [
    "CLOCK_EVENT_REASONS",
    "NVML",
    "SERIES",
    "SETTINGS",
    "GpuDescription",
    "GpuDevice",
    "GpuReader",
    "NvmlError",
    "NvmlGpuReader",
    "clock_event_reasons",
    "describe_gpus",
    "read_series",
]
