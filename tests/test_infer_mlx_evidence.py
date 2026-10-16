"""Bounded, portable evidence validation and conflict contracts."""

import copy
import json
import shutil
import time

import pytest

from stormlog.infer.errors import InferInputError
from stormlog.infer.mlx_evidence import (
    MANIFEST_NAME,
    collect_files,
    load_all_evidence,
    load_evidence,
    validate_manifest,
    write_manifest,
)
from stormlog.infer.mlx_server import capture_mlx_server
from tests.mlx_fakes import FakeCore, make_runtime


@pytest.fixture
def evidence(tmp_path):
    with capture_mlx_server(
        engine="mlx-lm",
        run_id="evidence-run",
        output=tmp_path / "native",
        runtime=make_runtime(FakeCore()),
        endpoint_host="127.0.0.1",
        endpoint_port=8000,
    ) as capture:
        time.sleep(0.02)
    return capture.directory


def rewrite(root, edit):
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    edit(manifest)
    write_manifest(root, manifest)


def test_copied_evidence_deduplicates_by_identity_and_content(evidence, tmp_path):
    copied = tmp_path / "copy"
    shutil.copytree(evidence, copied)
    original = load_evidence(evidence)
    assert original.events and all(e.schema_version == 4 for e in original.events)
    assert len(load_all_evidence([evidence, copied])) == 1
    rewrite(copied, lambda m: m["limitations"].append("conflicting metadata"))
    with pytest.raises(InferInputError, match="conflict"):
        load_all_evidence([evidence, copied])


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("format", "another.format"),
    ],
)
def test_unknown_versions_fail(evidence, field, value):
    payload = json.loads((evidence / MANIFEST_NAME).read_text())
    payload[field] = value
    with pytest.raises(InferInputError):
        validate_manifest(payload)


@pytest.mark.parametrize(
    "path",
    ["../outside", "/absolute", "telemetry/../x", "telemetry\\x", "telemetry//x"],
)
def test_unsafe_paths_fail(evidence, path):
    payload = json.loads((evidence / MANIFEST_NAME).read_text())
    payload["files"][0]["path"] = path
    with pytest.raises(InferInputError):
        validate_manifest(payload)


def test_digest_corruption_and_unlisted_native_file_fail(evidence):
    segment = next((evidence / "telemetry").glob("segment-*.jsonl"))
    data = segment.read_bytes()
    segment.write_bytes(data + b" ")
    with pytest.raises(InferInputError, match="digest"):
        load_evidence(evidence)
    segment.write_bytes(data)
    (evidence / "telemetry" / "unlisted.json").write_text("{}")
    with pytest.raises(InferInputError, match="not listed"):
        load_evidence(evidence)


def test_symbolic_link_is_rejected_even_with_matching_digest(evidence, tmp_path):
    segment = next((evidence / "telemetry").glob("segment-*.jsonl"))
    target = tmp_path / "external.jsonl"
    shutil.move(segment, target)
    segment.symlink_to(target)
    with pytest.raises(InferInputError, match="symbolic link"):
        load_evidence(evidence)


@pytest.mark.parametrize(
    "field,value", [("pid", 1), ("job_id", "other-run"), ("schema_version", 2)]
)
def test_native_identity_and_wire_family_checked_before_decoding(
    evidence, field, value
):
    segment = next((evidence / "telemetry").glob("segment-*.jsonl"))
    rows = [json.loads(line) for line in segment.read_text().splitlines()]
    rows[0][field] = value
    segment.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rewrite(evidence, lambda m: m.update(files=collect_files(evidence, stats=False)))
    with pytest.raises(InferInputError, match="native MLX telemetry"):
        load_evidence(evidence)


def test_missing_samples_are_valid_and_explicit(evidence):
    rewrite(evidence, lambda m: m.update(files=[]))
    shutil.rmtree(evidence / "telemetry")
    item = load_evidence(evidence)
    assert item.events == ()
    assert "no native telemetry" in " ".join(item.warnings)


def test_missing_epoch_cannot_carry_verified_routing_proof(evidence):
    manifest = json.loads((evidence / MANIFEST_NAME).read_text())
    process, capture = manifest["process"], manifest["capture"]
    proof = dict(
        method="psutil_process_listening_socket",
        pid=process["pid"],
        start_time_ns=process["start_time_ns"],
        address="127.0.0.1",
        port=8000,
        observed_at_ns=capture["started_at_ns"],
    )
    manifest["endpoint"]["routing_proof"] = proof
    validate_manifest(manifest)
    broken = copy.deepcopy(manifest)
    broken["process"]["start_time_ns"] = None
    broken["endpoint"]["routing_proof"]["start_time_ns"] = None
    with pytest.raises(InferInputError):
        validate_manifest(broken)
    broken = copy.deepcopy(manifest)
    broken["endpoint"]["routing_proof"]["address"] = "::1"
    with pytest.raises(InferInputError):
        validate_manifest(broken)


def test_duplicate_json_fields_rejected(evidence):
    path = evidence / MANIFEST_NAME
    path.write_text(
        path.read_text().replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
        )
    )
    with pytest.raises(InferInputError, match="Duplicate JSON"):
        load_evidence(evidence)


def test_published_schema_matches_capture_and_stats(evidence, monkeypatch):
    from pathlib import Path

    import jsonschema

    from stormlog.infer.omlx_stats import OmlxStatsSampler, normalize_stats

    schema = json.loads(
        Path("docs/schemas/inference_mlx_evidence_v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(load_evidence(evidence).manifest, schema)
    sampler = OmlxStatsSampler(
        url="http://127.0.0.1:8000/admin/api/stats",
        api_key=None,
        interval=1,
        timeout=1,
        session_id="s",
        run_id="evidence-run",
        server_version=None,
    )
    monkeypatch.setattr(
        sampler, "_fetch_metrics", lambda _: normalize_stats({}, sampler.identity)
    )
    record = sampler.observe("c1", "measured", "interval")
    jsonschema.validate(record, schema)
    record["api_key"] = "never-allowed"
    from stormlog.infer.omlx_stats import validate_stats_record

    with pytest.raises(ValueError, match="Unknown"):
        validate_stats_record(record)
