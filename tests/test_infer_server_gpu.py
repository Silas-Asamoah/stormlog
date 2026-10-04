"""Describing the GPUs a vLLM server uses, through NVML."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.server_gpu import (
    NvmlError,
    clock_event_reasons,
    describe_gpus,
    read_series,
)

NOT_SUPPORTED = 3


class FakeReader:
    """Two A30s; the server's worker runs on the second one."""

    def __init__(self) -> None:
        self.devices: list[dict[str, Any]] = [
            {"uuid": "GPU-aaaa", "pids": {900}, "sm_clock_mhz": 210},
            {"uuid": "GPU-bbbb", "pids": {102, 950}, "sm_clock_mhz": 1440},
        ]
        self.closed = False

    def driver_version(self) -> str:
        return "580.82.07"

    def cuda_driver_version(self) -> int:
        return 13000

    def device_count(self) -> int:
        return len(self.devices)

    def uuid(self, index: int) -> str:
        return str(self.devices[index]["uuid"])

    def compute_pids(self, index: int) -> set[int] | None:
        pids = self.devices[index]["pids"]
        return None if pids is None else set(pids)

    def read(self, index: int, name: str) -> Any:
        device = self.devices[index]
        if name in device:
            return device[name]
        values: dict[str, Any] = {
            "name": "NVIDIA A30",
            "power_limit_w": 165.0,
            "enforced_power_limit_w": 165.0,
            "application_clocks_mhz": {"graphics": 1440, "sm": 1440, "memory": 1215},
            "persistence_mode": True,
            "ecc_enabled": True,
            "compute_mode": "default",
            "temperature_c": 41,
            "clock_event_reasons": ["gpu_idle"],
        }
        if name not in values:
            raise NvmlError("nvmlDeviceGetMigMode", NOT_SUPPORTED)
        return values[name]

    def close(self) -> None:
        self.closed = True


def test_the_servers_gpus_are_the_devices_listing_its_processes() -> None:
    description = describe_gpus(FakeReader(), server_pids={100, 101, 102})

    assert description.server_uuids == ["GPU-bbbb"]
    assert description.device_count == 2
    used = description.devices[1]
    assert used.server_pids == (102,)
    assert description.devices[0].server_pids == ()
    assert description.driver_version == "580.82.07"
    assert description.cuda_driver_version == "13.0"
    assert description.issues == ()


def test_settings_and_drifting_fields_are_kept_apart() -> None:
    record = describe_gpus(FakeReader(), server_pids={102}).to_record()
    device = record["devices"][1]

    assert device["settings"]["name"] == "NVIDIA A30"
    assert device["settings"]["application_clocks_mhz"]["sm"] == 1440
    assert device["series"] == {
        "sm_clock_mhz": 1440,
        "temperature_c": 41,
        "clock_event_reasons": ["gpu_idle"],
    }
    assert record["source"] == "nvml"


def test_an_unreadable_field_says_why_and_is_never_filled_in() -> None:
    device = describe_gpus(FakeReader(), server_pids={102}).devices[1]
    assert device.settings["mig_enabled"] == {"unavailable": "NVML code 3"}


def test_no_server_process_on_any_device_is_an_issue() -> None:
    reader = FakeReader()
    reader.devices[1]["pids"] = None
    description = describe_gpus(reader, server_pids={102})
    assert description.server_uuids == []
    assert "compute processes unreadable on 1 device(s)" in description.issues
    assert any("host PIDs" in issue for issue in description.issues)


def test_a_later_reading_covers_only_the_servers_devices() -> None:
    reader = FakeReader()
    reader.devices[1]["sm_clock_mhz"] = 1200
    assert read_series(reader, ["GPU-bbbb"]) == {
        "GPU-bbbb": {
            "sm_clock_mhz": 1200,
            "temperature_c": 41,
            "clock_event_reasons": ["gpu_idle"],
        }
    }


@pytest.mark.parametrize(
    ("mask", "names"),
    [
        (0, []),
        (0x1, ["gpu_idle"]),
        (0x4 | 0x40, ["sw_power_cap", "hw_thermal_slowdown"]),
        (0x1000, ["unknown_0x1000"]),
    ],
)
def test_clock_event_reasons_are_named_bit_by_bit(mask: int, names: list[str]) -> None:
    assert clock_event_reasons(mask) == names


class _FakeNvml:
    """The NVML functions the reader calls, writing through ctypes pointers."""

    def __init__(self) -> None:
        self.shutdown = False

    @staticmethod
    def _set(pointer: Any, value: Any) -> int:
        pointer._obj.value = value
        return 0

    def nvmlInit_v2(self) -> int:  # noqa: N802
        return 0

    def nvmlShutdown(self) -> int:  # noqa: N802
        self.shutdown = True
        return 0

    def nvmlSystemGetDriverVersion(
        self, buffer: Any, _length: Any
    ) -> int:  # noqa: N802
        buffer.value = b"580.82.07"
        return 0

    def nvmlSystemGetCudaDriverVersion_v2(self, version: Any) -> int:  # noqa: N802
        return self._set(version, 12080)

    def nvmlDeviceGetCount_v2(self, count: Any) -> int:  # noqa: N802
        return self._set(count, 1)

    def nvmlDeviceGetHandleByIndex_v2(
        self, index: Any, handle: Any
    ) -> int:  # noqa: N802
        return self._set(handle, 0x1000 + index.value)

    def nvmlDeviceGetUUID(
        self, _handle: Any, buffer: Any, _length: Any
    ) -> int:  # noqa: N802
        buffer.value = b"GPU-real"
        return 0

    def nvmlDeviceGetName(
        self, _handle: Any, buffer: Any, _length: Any
    ) -> int:  # noqa: N802
        buffer.value = b"NVIDIA A30"
        return 0

    def nvmlDeviceGetPowerManagementLimit(
        self, _h: Any, limit: Any
    ) -> int:  # noqa: N802
        return self._set(limit, 165000)

    def nvmlDeviceGetEnforcedPowerLimit(self, _h: Any, limit: Any) -> int:  # noqa: N802
        return self._set(limit, 150000)

    def nvmlDeviceGetApplicationsClock(  # noqa: N802
        self, _h: Any, clock: Any, mhz: Any
    ) -> int:
        return self._set(mhz, {0: 1440, 1: 1440, 2: 1215}[clock.value])

    def nvmlDeviceGetPersistenceMode(self, _h: Any, mode: Any) -> int:  # noqa: N802
        return self._set(mode, 1)

    def nvmlDeviceGetEccMode(
        self, _h: Any, current: Any, pending: Any
    ) -> int:  # noqa: N802
        self._set(pending, 1)
        return self._set(current, 1)

    def nvmlDeviceGetMigMode(
        self, _h: Any, _current: Any, _pending: Any
    ) -> int:  # noqa: N802
        return NOT_SUPPORTED

    def nvmlDeviceGetComputeMode(self, _h: Any, mode: Any) -> int:  # noqa: N802
        return self._set(mode, 3)

    def nvmlDeviceGetClockInfo(
        self, _h: Any, _clock: Any, mhz: Any
    ) -> int:  # noqa: N802
        return self._set(mhz, 1410)

    def nvmlDeviceGetTemperature(
        self, _h: Any, _sensor: Any, celsius: Any
    ) -> int:  # noqa: N802
        return self._set(celsius, 52)

    # An older driver: only the throttle-reason name exists.
    def nvmlDeviceGetCurrentClocksThrottleReasons(  # noqa: N802
        self, _h: Any, mask: Any
    ) -> int:
        return self._set(mask, 0x4)


def test_the_nvml_reader_reads_through_ctypes(monkeypatch: pytest.MonkeyPatch) -> None:
    from stormlog.infer import server_gpu

    fake = _FakeNvml()
    monkeypatch.setattr(server_gpu.ctypes, "CDLL", lambda _name: fake)
    reader = server_gpu.NvmlGpuReader()
    description = describe_gpus(reader, server_pids=set())
    reader.close()

    assert fake.shutdown
    assert (description.driver_version, description.cuda_driver_version) == (
        "580.82.07",
        "12.8",
    )
    (device,) = description.devices
    assert device.uuid == "GPU-real"
    assert device.settings == {
        "name": "NVIDIA A30",
        "power_limit_w": 165.0,
        "enforced_power_limit_w": 150.0,
        "application_clocks_mhz": {"graphics": 1440, "sm": 1440, "memory": 1215},
        "persistence_mode": True,
        "ecc_enabled": True,
        "mig_enabled": {"unavailable": "NVML code 3"},
        "compute_mode": "exclusive_process",
    }
    assert device.series == {
        "sm_clock_mhz": 1410,
        "temperature_c": 52,
        "clock_event_reasons": ["sw_power_cap"],
    }
    # No compute-process query in this fake: NVML cannot say who runs there.
    assert "compute processes unreadable on 1 device(s)" in description.issues
