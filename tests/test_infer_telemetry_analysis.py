"""Do not turn unrelated server measurements into request-owned memory."""

from __future__ import annotations

import json

import pytest

from stormlog.infer.analysis import analyze_inference_events
from stormlog.infer.telemetry import ServerIdentity, TelemetrySample


def _profile(tmp_path, *, host: str = "client", run_id: str = "run-1"):
    path = tmp_path / "profile.jsonl"
    records = [
        {
            "schema_version": 2,
            "event_type": "infer.artifact",
            "context": {"run_id": run_id, "host": host},
        },
        {
            "schema_version": 1,
            "event_type": "infer.system_sample",
            "timestamp_ns": 150,
            "device_used_bytes": 999,
            "observation_scope": "client_local",
        },
        {
            "schema_version": 1,
            "event_type": "infer.request",
            "phase": "measured",
            "status": "ok",
            "case_id": "case-a",
            "started_at_ns": 100,
            "ended_at_ns": 300,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _server_sample(
    *,
    host: str = "server-a",
    pid: int = 42,
    start_ns: int = 10,
    uuid: str = "GPU-A",
    instance: str | None = None,
    metric: str = "device_memory_used_bytes",
    state: str = "valid",
    value: int | None = 256,
    run_id: str = "run-1",
) -> TelemetrySample:
    return TelemetrySample(
        run_id=run_id,
        identity=ServerIdentity(
            host=host,
            pid=pid,
            process_start_ns=start_ns,
            device_uuid=uuid,
            gpu_instance_id=instance,
            replica_id=f"replica-{host}",
        ),
        observed_at_ns=150,
        metric=metric,
        value_bytes=value,
        state=state,
        source="nvml-v2",
        interval_ms=100,
    )


def _telemetry(tmp_path, name: str, *samples: TelemetrySample):
    path = tmp_path / name
    path.write_text("".join(json.dumps(s.to_record()) + "\n" for s in samples))
    return path


def test_endpoint_only_memory_is_explicitly_client_local(tmp_path) -> None:
    report = analyze_inference_events(_profile(tmp_path))
    memory = report["cases"]["case-a"]["memory"]
    assert memory["observation_scope"] == "client_local"
    assert memory["peak_device_used_bytes"] == 999
    assert memory["server_observations"] == {}
    assert report["telemetry"]["server_join"]["status"] == "not_configured"


def test_direct_remote_join_requires_clock_alignment(tmp_path) -> None:
    profile = _profile(tmp_path)
    telemetry = _telemetry(tmp_path, "server.jsonl", _server_sample())
    unaligned = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    assert unaligned["telemetry"]["server_join"]["reason"] == "clock_alignment_required"
    assert unaligned["cases"]["case-a"]["memory"]["server_observations"] == {}
    report = analyze_inference_events(
        profile,
        server_telemetry_paths=[telemetry],
        direct_server=True,
        clock_offset_ns=0,
        clock_uncertainty_ns=5,
    )
    join = report["telemetry"]["server_join"]
    assert join["status"] == "joined"
    assert join["route_evidence"] == "operator_declared_direct"
    memory = report["cases"]["case-a"]["memory"]
    assert memory["peak_device_used_bytes"] == 999
    assert memory["server_observations"]["device_memory_used_bytes"] == {
        "observation_scope": "gpu_device",
        "counter_owner": "gpu_device",
        "source": "nvml-v2",
        "highest_observed_bytes": 256,
        "valid_samples": 1,
        "missing_samples": 0,
        "stale_samples": 0,
        "invalid_samples": 0,
        "interval_ms": 100,
    }


def test_two_servers_both_index_zero_are_not_merged(tmp_path) -> None:
    profile = _profile(tmp_path)
    a = _telemetry(tmp_path, "a.jsonl", _server_sample(uuid="GPU-A"))
    b = _telemetry(
        tmp_path,
        "b.jsonl",
        _server_sample(host="server-b", pid=43, uuid="GPU-B"),
    )
    report = analyze_inference_events(
        profile,
        server_telemetry_paths=[a, b],
        direct_server=True,
        clock_offset_ns=0,
        clock_uncertainty_ns=0,
    )
    assert report["telemetry"]["server_join"]["reason"] == "multiple_server_identities"
    assert {
        t["identity"]["device_uuid"] for t in report["telemetry"]["server_targets"]
    } == {
        "GPU-A",
        "GPU-B",
    }


@pytest.mark.parametrize("change", [{"pid": 43}, {"start_ns": 11}, {"uuid": "GPU-B"}])
def test_process_restart_or_gpu_change_cannot_silently_join(tmp_path, change) -> None:
    profile = _profile(tmp_path)
    telemetry = _telemetry(
        tmp_path, "changed.jsonl", _server_sample(), _server_sample(**change)
    )
    report = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    assert report["telemetry"]["server_join"]["reason"] == "multiple_server_identities"


def test_missing_counter_is_null_and_instance_scope_remains_distinct(tmp_path) -> None:
    profile = _profile(tmp_path, host="server-a")
    telemetry = _telemetry(
        tmp_path,
        "mig.jsonl",
        _server_sample(
            uuid="GPU-A",
            instance="MIG-A",
            metric="instance_memory_used_bytes",
        ),
        _server_sample(
            uuid="GPU-A",
            instance="MIG-A",
            metric="instance_memory_reserved_bytes",
            state="missing",
            value=None,
        ),
    )
    report = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    memory = report["cases"]["case-a"]["memory"]["server_observations"]
    assert memory["instance_memory_used_bytes"]["observation_scope"] == "gpu_instance"
    assert memory["instance_memory_reserved_bytes"]["highest_observed_bytes"] is None
    assert memory["instance_memory_reserved_bytes"]["missing_samples"] == 1


def test_mismatched_run_id_is_rejected(tmp_path) -> None:
    profile = _profile(tmp_path)
    telemetry = _telemetry(
        tmp_path, "wrong-run.jsonl", _server_sample(run_id="other-run")
    )
    with pytest.raises(ValueError, match="run_id"):
        analyze_inference_events(profile, server_telemetry_paths=[telemetry])
