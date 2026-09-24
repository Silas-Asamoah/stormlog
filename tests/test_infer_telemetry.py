"""Scoped inference telemetry contract and identity safeguards."""

from __future__ import annotations

import json
import os

import pytest

from stormlog.infer.server_collector import collect_server_telemetry
from stormlog.infer.telemetry import ServerIdentity, TelemetrySample, load_telemetry


def _identity(**changes: object) -> ServerIdentity:
    values = {
        "host": "server-a",
        "pid": 321,
        "process_start_ns": 1_000_000_000,
        "device_uuid": "GPU-a",
        "replica_id": "replica-a",
        **changes,
    }
    return ServerIdentity(**values)


def _sample(**changes: object) -> TelemetrySample:
    values = {
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


def test_telemetry_has_explicit_scope_owner_and_provenance(tmp_path) -> None:
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


def test_unavailable_counter_is_null_and_scope_cannot_be_forged() -> None:
    missing = _sample(value_bytes=None, state="missing")
    assert missing.to_record()["value_bytes"] is None
    with pytest.raises(ValueError, match="null value_bytes"):
        _sample(value_bytes=0, state="missing")
    forged = _sample().to_record()
    forged["scope"] = "server_process"
    with pytest.raises(ValueError, match="scope"):
        TelemetrySample.from_record(forged)


def test_instance_metric_requires_instance_identity() -> None:
    with pytest.raises(ValueError, match="gpu_instance_id"):
        _sample(metric="instance_memory_used_bytes")


def test_on_host_collector_writes_process_and_gpu_with_same_identity(tmp_path) -> None:
    class FakeGpu:
        device_uuid = "GPU-live"
        gpu_instance_id = None

        def read(self):
            return 256, 64, None

        def close(self):
            pass

    path = tmp_path / "server.jsonl"
    count = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.025,
        gpu_source=FakeGpu(),
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


def test_on_host_collector_emits_missing_nvml_counter(tmp_path) -> None:
    class MissingGpu:
        device_uuid = "GPU-live"
        gpu_instance_id = None

        def read(self):
            return None, None, "NVML unavailable"

        def close(self):
            pass

    path = tmp_path / "missing.jsonl"
    collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.015,
        gpu_source=MissingGpu(),
    )
    gpu_samples = [s for s in load_telemetry(path) if s.scope == "gpu_device"]
    assert gpu_samples
    assert {s.state for s in gpu_samples} == {"missing"}
    assert all(s.value_bytes is None for s in gpu_samples)
