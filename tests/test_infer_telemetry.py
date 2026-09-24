"""Scoped inference telemetry contract and identity safeguards."""

from __future__ import annotations

import json

import pytest

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
        "".join(json.dumps(sample.to_record()) + "\n" for sample in (device, process, instance))
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
