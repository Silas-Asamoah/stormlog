"""Scoped inference telemetry contract and identity safeguards."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import psutil
import pytest
from jsonschema import Draft202012Validator

from stormlog.infer.server_collector import (
    GpuMemoryReading,
    NvmlMemorySource,
    collect_server_telemetry,
    read_process_rss,
)
from stormlog.infer.telemetry import ServerIdentity, TelemetrySample, load_telemetry

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs/schemas/inference_telemetry_v1.schema.json"
    ).read_text()
)


def _identity(**changes: Any) -> ServerIdentity:
    values: dict[str, Any] = {
        "host": "server-a",
        "pid": 321,
        "process_start_ns": 1_000_000_000,
        "device_uuid": "GPU-a",
        "replica_id": "replica-a",
        **changes,
    }
    return ServerIdentity(**values)


def _sample(**changes: Any) -> TelemetrySample:
    values: dict[str, Any] = {
        "run_id": "run-a",
        "identity": _identity(),
        "observed_at_ns": 2_000_000_000,
        "metric": "device_memory_used_bytes",
        "value_bytes": 100,
        "state": "valid",
        "source": "nvml-v2",
        "interval_ms": 100,
        **changes,
    }
    return TelemetrySample(**values)


class _FakeGpu:
    """A GPU source that reports fixed values and can act on each read."""

    device_uuid = "GPU-live"
    gpu_instance_id: str | None = None

    def __init__(
        self,
        reading: GpuMemoryReading | None = None,
        on_read: Callable[[int], None] | None = None,
    ) -> None:
        self.reading = reading or GpuMemoryReading(256, 64, "valid")
        self.on_read = on_read
        self.reads = 0

    def read(self) -> GpuMemoryReading:
        self.reads += 1
        if self.on_read is not None:
            self.on_read(self.reads)
        return self.reading

    def close(self) -> None:
        pass


def _assert_schema_valid(samples: list[TelemetrySample]) -> None:
    validator = Draft202012Validator(SCHEMA)
    for sample in samples:
        validator.validate(sample.to_record())


def test_telemetry_has_explicit_scope_owner_and_provenance(tmp_path: Path) -> None:
    device = _sample()
    process = _sample(metric="process_rss_bytes", source="psutil")
    instance = _sample(
        identity=_identity(device_uuid="MIG-a", gpu_instance_id="MIG-a"),
        metric="instance_memory_used_bytes",
    )
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        "".join(
            json.dumps(sample.to_record()) + "\n"
            for sample in (device, process, instance)
        )
    )
    assert [sample.scope for sample in load_telemetry(path)] == [
        "gpu_device",
        "server_process",
        "gpu_instance",
    ]
    assert device.to_record()["counter_owner"] == "gpu_device"
    assert device.to_record()["provenance"] == "observed"
    Draft202012Validator.check_schema(SCHEMA)
    _assert_schema_valid([device, process, instance])


def test_unavailable_counter_is_null_and_scope_cannot_be_forged() -> None:
    missing = _sample(value_bytes=None, state="missing")
    assert missing.to_record()["value_bytes"] is None
    with pytest.raises(ValueError, match="null value_bytes"):
        _sample(value_bytes=0, state="missing")
    forged = _sample().to_record()
    forged["scope"] = "server_process"
    with pytest.raises(ValueError, match="scope"):
        TelemetrySample.from_record(forged)
    with pytest.raises(ValueError, match="non-negative value_bytes"):
        _sample(value_bytes=True)
    with pytest.raises(ValueError, match="positive pid"):
        _identity(pid=True)
    with pytest.raises(ValueError, match="non-empty strings"):
        _identity(boot_id="")
    forged = _sample().to_record()
    forged["schema_version"] = True
    with pytest.raises(ValueError, match="unsupported telemetry"):
        TelemetrySample.from_record(forged)


def test_instance_metric_requires_instance_identity() -> None:
    with pytest.raises(ValueError, match="gpu_instance_id"):
        _sample(metric="instance_memory_used_bytes")


def test_allocator_and_engine_cache_are_separate_reported_counters() -> None:
    allocator = _sample(
        metric="allocator_allocated_bytes",
        source="pytorch",
        provenance="reported",
    )
    cache = _sample(
        metric="engine_cache_occupied_bytes",
        source="vllm",
        provenance="reported",
    )
    assert allocator.scope == cache.scope == "server_process"
    assert allocator.to_record()["counter_owner"] == "allocator"
    assert cache.to_record()["counter_owner"] == "engine_cache"
    assert TelemetrySample.from_record(cache.to_record()) == cache
    _assert_schema_valid([allocator, cache])


def test_on_host_collector_writes_process_and_gpu_with_same_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "server.jsonl"
    count = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.025,
        gpu_source=_FakeGpu(),
    )
    samples = load_telemetry(path)
    assert len(samples) == count * 3
    assert {sample.metric for sample in samples} == {
        "process_rss_bytes",
        "device_memory_used_bytes",
        "device_memory_reserved_bytes",
    }
    assert {sample.identity.process_start_ns for sample in samples} == {
        samples[0].identity.process_start_ns
    }
    assert {sample.identity.device_uuid for sample in samples} == {"GPU-live"}
    _assert_schema_valid(samples)


def test_on_host_collector_emits_missing_nvml_counter(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.015,
        gpu_source=_FakeGpu(GpuMemoryReading(None, None, "missing", "unavailable")),
    )
    gpu_samples = [s for s in load_telemetry(path) if s.scope == "gpu_device"]
    assert gpu_samples
    assert {s.state for s in gpu_samples} == {"missing"}
    assert all(s.value_bytes is None for s in gpu_samples)


def test_on_host_collector_stops_after_gpu_identity_failure(tmp_path: Path) -> None:
    path = tmp_path / "changed.jsonl"
    count = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.1,
        gpu_source=_FakeGpu(
            GpuMemoryReading(None, None, "invalid", "device UUID changed")
        ),
    )
    assert count == 1
    samples = load_telemetry(path)
    assert [s.state for s in samples] == ["valid", "invalid", "invalid"]
    _assert_schema_valid(samples)


def test_access_denied_is_missing_and_collection_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny(process: psutil.Process) -> None:
        raise psutil.AccessDenied(process.pid)

    monkeypatch.setattr(psutil.Process, "memory_info", deny)
    path = tmp_path / "denied.jsonl"
    count = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.06,
        gpu_source=_FakeGpu(),
    )
    assert count >= 2
    samples = load_telemetry(path)
    rss = [s for s in samples if s.metric == "process_rss_bytes"]
    assert {s.state for s in rss} == {"missing"}
    assert {s.detail for s in rss} == {"access denied reading server process memory"}
    assert {s.state for s in samples if s.scope == "gpu_device"} == {"valid"}
    _assert_schema_valid(samples)


class _Process:
    pid = 7

    def __init__(self, *, running: bool = True, error: Exception | None = None):
        self.running = running
        self.error = error

    def is_running(self) -> bool:
        return self.running

    def memory_info(self) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(rss=4096)


@pytest.mark.parametrize(
    ("process", "expected_state"),
    [
        (_Process(), "valid"),
        (_Process(running=False), "invalid"),
        (_Process(error=psutil.NoSuchProcess(7)), "invalid"),
        (_Process(error=psutil.ZombieProcess(7)), "invalid"),
        (_Process(error=psutil.AccessDenied(7)), "missing"),
        (_Process(error=psutil.Error()), "missing"),
    ],
)
def test_only_a_gone_process_invalidates_rss(
    process: _Process, expected_state: str
) -> None:
    state, rss, _detail = read_process_rss(process)
    assert state == expected_state
    assert (rss is not None) == (expected_state == "valid")


class _FakeNvmlLibrary:
    def __init__(self, uuids: dict[str, str], *, uuid_code: int = 0) -> None:
        self.uuids = uuids
        self.uuid_code = uuid_code
        self.memory_code = 0

    def nvmlDeviceGetUUID(  # noqa: N802
        self, handle: str, buffer: Any, _length: int
    ) -> int:
        if self.uuid_code:
            return self.uuid_code
        buffer.value = self.uuids[handle].encode()
        return 0

    def nvmlDeviceGetMemoryInfo_v2(  # noqa: N802
        self, _handle: str, info_ref: Any
    ) -> int:
        if self.memory_code:
            return self.memory_code
        info_ref._obj.used, info_ref._obj.reserved = 300, 40
        return 0


def _nvml_source(
    library: _FakeNvmlLibrary, *, parent_uuid: str | None = None
) -> NvmlMemorySource:
    source = object.__new__(NvmlMemorySource)
    # Fill in what __init__ would take from a real NVML library.
    state: Any = source
    state._lib = library
    state._closed = True
    state._handle = "gpu"
    state._handle_uuid = "MIG-a" if parent_uuid else "GPU-a"
    state._parent_handle = "parent" if parent_uuid else None
    state.device_uuid = parent_uuid or "GPU-a"
    state.gpu_instance_id = "MIG-a" if parent_uuid else None
    return source


def test_nvml_read_reports_values_for_the_same_gpu() -> None:
    reading = _nvml_source(_FakeNvmlLibrary({"gpu": "GPU-a"})).read()
    assert reading == GpuMemoryReading(300, 40, "valid")


def test_unreadable_gpu_uuid_is_missing_not_invalid() -> None:
    library = _FakeNvmlLibrary({"gpu": "GPU-a"}, uuid_code=999)
    reading = _nvml_source(library).read()
    assert reading.state == "missing"
    assert "device identity unreadable" in (reading.detail or "")


def test_a_different_gpu_uuid_invalidates() -> None:
    reading = _nvml_source(_FakeNvmlLibrary({"gpu": "GPU-b"})).read()
    assert (reading.state, reading.detail) == ("invalid", "device UUID changed")
    library = _FakeNvmlLibrary({"gpu": "MIG-a", "parent": "GPU-other"})
    reading = _nvml_source(library, parent_uuid="GPU-a").read()
    assert (reading.state, reading.detail) == ("invalid", "parent device UUID changed")


def test_failed_nvml_memory_read_is_missing() -> None:
    library = _FakeNvmlLibrary({"gpu": "GPU-a"})
    library.memory_code = 3
    reading = _nvml_source(library).read()
    assert reading.state == "missing"
    assert reading.used_bytes is None
