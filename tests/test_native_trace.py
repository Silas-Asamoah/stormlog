from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest

from stormlog import query
from stormlog.collector_health import CollectorHealthState
from stormlog.native_trace import (
    NativeHelperMessage,
    NativeTraceRecord,
    native_trace_preflight,
)
from stormlog.native_trace_store import (
    BoundedTraceWriter,
    NativeTraceArtifact,
    NativeTraceIdentity,
    NativeTraceManifest,
    register_native_trace_attachment,
    write_native_trace_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "docs" / "schemas" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _manifest(
    *,
    health: CollectorHealthState | None = None,
    writer: BoundedTraceWriter | None = None,
) -> NativeTraceManifest:
    artifacts: tuple[NativeTraceArtifact, ...] = ()
    loss = None
    if writer is not None:
        artifacts = (
            writer.finalize(
                content_type="application/x-ndjson",
                sensitive_fields=("symbols", "addresses"),
            ),
        )
        loss = writer.loss(flush_outcome="complete")
    from stormlog.native_trace_store import NativeTraceLoss

    return NativeTraceManifest(
        capture_id="capture-1",
        backend="cupti_activity",
        identity=NativeTraceIdentity(
            run_id="run-1",
            session_id="session-1",
            job_id="job-1",
            rank=0,
            pid=123,
            device_id="GPU-0",
        ),
        helper_executable="stormlog-cupti-helper",
        helper_version="0.1.0",
        started_ns=100,
        ended_ns=400,
        clock_domains=("host/monotonic_ns", "gpu-0/device_ns"),
        requested_activities=("runtime", "kernel"),
        enabled_activities=("runtime", "kernel"),
        max_bytes=4096,
        privilege="same-process",
        target_selector="pid:123",
        health=health or CollectorHealthState(),
        loss=loss or NativeTraceLoss(flush_outcome="complete"),
        artifacts=artifacts,
        metadata={"evidence": "synthetic-contract-test"},
    )


def test_preflight_reports_unsupported_backends_on_macos() -> None:
    results = native_trace_preflight(system="Darwin", architecture="arm64")

    assert [result.backend for result in results] == [
        "cupti_activity",
        "rocprofiler",
    ]
    assert {result.status for result in results} == {"unsupported"}
    assert all(result.library_path is None for result in results)


def test_preflight_finds_library_without_loading_it(tmp_path: Path) -> None:
    library = tmp_path / "libcupti.so"
    library.write_bytes(b"not-a-real-library")

    results = native_trace_preflight(
        system="Linux",
        architecture="x86_64",
        library_search_paths=(tmp_path,),
    )

    cupti = results[0]
    assert cupti.status == "available"
    assert cupti.library_path == str(library)
    assert "unverified" in cupti.reason


def test_helper_protocol_rejects_unknown_version_and_message() -> None:
    with pytest.raises(ValueError, match="protocol version"):
        NativeHelperMessage.from_dict(
            {
                "protocol_version": 2,
                "message_type": "hello",
                "request_id": "request-1",
                "payload": {},
            }
        )
    with pytest.raises(ValueError, match="message type"):
        NativeHelperMessage.from_dict(
            {
                "protocol_version": 1,
                "message_type": "execute-shell",
                "request_id": "request-1",
                "payload": {},
            }
        )


def test_helper_protocol_schema_accepts_strict_message() -> None:
    message = NativeHelperMessage(
        message_type="capabilities",
        request_id="request-1",
        payload={"backends": []},
    )

    jsonschema.Draft202012Validator(
        _schema("native_helper_protocol_v1.schema.json")
    ).validate(message.to_dict())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cpu_start_ns": 10}, "CPU interval"),
        ({"device_start_ns": 20, "device_end_ns": 19}, "device interval"),
        ({"uncertainty": ""}, "uncertainty"),
    ],
)
def test_trace_record_rejects_ambiguous_time_evidence(
    kwargs: dict[str, Any], message: str
) -> None:
    values: dict[str, Any] = {
        "record_id": "record-1",
        "activity_kind": "kernel",
        "clock_domain": "gpu/device_ns",
        "provenance": "synthetic",
        "uncertainty": "not hardware evidence",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        NativeTraceRecord(**values)


def test_synthetic_fixture_preserves_overlap_graph_and_clock_domains() -> None:
    fixture = REPO_ROOT / "tests" / "fixtures" / "native_trace"
    records = [
        json.loads(line)
        for line in (fixture / "overlap_graph_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    validator = jsonschema.Draft202012Validator(
        _schema("native_trace_record_v1.schema.json")
    )
    for record in records:
        validator.validate(record)

    kernels = [record for record in records if record["activity_kind"] == "kernel"]
    assert kernels[0]["device_start_ns"] < kernels[1]["device_end_ns"]
    assert kernels[1]["device_start_ns"] < kernels[0]["device_end_ns"]
    assert {record["stream_id"] for record in kernels[:2]} == {
        "stream-1",
        "stream-2",
    }
    assert {record["metadata"]["replay"] for record in kernels} == {1, 2}
    assert records[0]["cpu_start_ns"] is not None
    assert records[0]["device_start_ns"] is None
    assert all("not hardware evidence" in record["uncertainty"] for record in records)


@pytest.mark.parametrize(
    "relative_path",
    ["../trace.bin", "/tmp/trace.bin", "C:\\trace.bin", "traces\\trace.bin"],
)
def test_bounded_writer_rejects_unsafe_paths(
    tmp_path: Path, relative_path: str
) -> None:
    with pytest.raises(ValueError, match="path"):
        BoundedTraceWriter(tmp_path, relative_path, max_bytes=16)


def test_bounded_writer_drops_complete_record_and_preserves_permissions(
    tmp_path: Path,
) -> None:
    writer = BoundedTraceWriter(tmp_path, "raw/trace.ndjson", max_bytes=5)

    assert writer.write(b"one\n") is True
    assert writer.write(b"toolong\n") is False
    artifact = writer.finalize(content_type="application/x-ndjson")
    loss = writer.loss(flush_outcome="partial")

    trace_path = tmp_path / artifact.path
    assert trace_path.read_bytes() == b"one\n"
    assert stat.S_IMODE(trace_path.stat().st_mode) == 0o600
    assert loss.delivered_records == 1
    assert loss.dropped_records == 1
    assert loss.bytes_dropped == len(b"toolong\n")
    assert loss.truncated is True


def test_bounded_writer_preserves_failed_capture_as_partial(tmp_path: Path) -> None:
    writer = BoundedTraceWriter(tmp_path, "raw/trace.bin", max_bytes=16)
    writer.write(b"partial")

    artifact = writer.preserve_partial()

    assert artifact.path == "raw/trace.bin.partial"
    assert (tmp_path / artifact.path).read_bytes() == b"partial"


def test_bounded_writer_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        BoundedTraceWriter(tmp_path, "escape/trace.bin", max_bytes=16)


def test_manifest_schema_and_attachment_registration(tmp_path: Path) -> None:
    writer = BoundedTraceWriter(tmp_path, "raw/trace.ndjson", max_bytes=4096)
    fixture = REPO_ROOT / "tests" / "fixtures" / "native_trace"
    writer.write((fixture / "overlap_graph_records.jsonl").read_bytes())
    manifest = _manifest(writer=writer)

    manifest_path = write_native_trace_manifest(tmp_path, manifest)
    sidecar_path = register_native_trace_attachment(tmp_path, manifest_path, manifest)

    jsonschema.Draft202012Validator(
        _schema("native_trace_manifest_v1.schema.json")
    ).validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    jsonschema.Draft202012Validator(
        _schema("stormlog_attachments_v1.schema.json")
    ).validate(json.loads(sidecar_path.read_text(encoding="utf-8")))
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(sidecar_path.stat().st_mode) == 0o600

    attachments = query.open([tmp_path]).list_attachments()
    assert len(attachments) == 1
    assert attachments[0].session_id == "session-1"
    assert attachments[0].kind == "native_trace_manifest"


def test_attachment_registration_is_idempotent_but_rejects_conflict(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest_path = write_native_trace_manifest(tmp_path, manifest)
    sidecar = register_native_trace_attachment(tmp_path, manifest_path, manifest)
    register_native_trace_attachment(tmp_path, manifest_path, manifest)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert len(payload["attachments"]) == 1

    conflicting = _manifest(
        health=CollectorHealthState(
            status="unhealthy",
            telemetry_partial=True,
            partial_fields=("trace",),
            last_error="helper crashed",
            consecutive_failures=1,
        )
    )
    with pytest.raises(ValueError, match="different evidence"):
        register_native_trace_attachment(tmp_path, manifest_path, conflicting)


def test_manifest_rejects_healthy_truncated_evidence(tmp_path: Path) -> None:
    writer = BoundedTraceWriter(tmp_path, "trace.bin", max_bytes=1)
    writer.write(b"too-large")
    from stormlog.native_trace_store import NativeTraceLoss

    with pytest.raises(ValueError, match="cannot be healthy"):
        NativeTraceManifest(
            capture_id="capture-loss",
            backend="cupti_activity",
            identity=NativeTraceIdentity(session_id="session-1", pid=123),
            helper_executable="helper",
            helper_version="0.1",
            started_ns=1,
            ended_ns=2,
            clock_domains=("gpu/device_ns",),
            requested_activities=("kernel",),
            enabled_activities=("kernel",),
            max_bytes=1,
            privilege="same-process",
            target_selector="pid:123",
            health=CollectorHealthState(),
            loss=NativeTraceLoss(
                dropped_records=1,
                bytes_dropped=9,
                truncated=True,
                flush_outcome="partial",
            ),
        )


def test_manifest_rejects_unrequested_enabled_activity() -> None:
    values = _manifest().__dict__.copy()
    values["enabled_activities"] = ("kernel", "memcpy")

    with pytest.raises(ValueError, match="must be requested"):
        NativeTraceManifest(**values)
