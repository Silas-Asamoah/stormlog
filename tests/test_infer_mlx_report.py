"""Associations, process epochs and uncertainty-aware case placement."""

import copy
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.correlation_events import ArtifactIdentityEvent, CorrelationContext
from stormlog.infer.host_clock import wall_clock_domain
from stormlog.infer.mlx_evidence import (
    LoadedEvidence,
    load_evidence,
    write_manifest,
)
from stormlog.infer.mlx_report import mlx_evidence_report
from stormlog.infer.mlx_server import capture_mlx_server
from tests.mlx_fakes import FakeCore, make_runtime


@pytest.fixture
def native(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "stormlog.infer.mlx_server.host_boot_id", lambda: "contract-boot"
    )
    with capture_mlx_server(
        engine="mlx-lm",
        run_id="join-run",
        output=tmp_path / "native",
        runtime=make_runtime(FakeCore()),
        endpoint_host="127.0.0.1",
        endpoint_port=8000,
    ) as capture:
        time.sleep(0.02)
    return capture.directory


def client(
    item: LoadedEvidence, *, run_id: str = "join-run", boot: str | None = None
) -> list[dict[str, Any]]:
    process = item.manifest["process"]
    context = CorrelationContext(
        run_id=run_id,
        session_id="client-session",
        producer_id="client",
        source="test",
        host=process["host"],
        clock_domain=wall_clock_domain(
            process["host"], process["boot_id"] if boot is None else boot
        ),
        clock_kind="wall",
        collection_mode="active",
        provenance="observed",
    )
    capture = item.manifest["capture"]
    return [
        ArtifactIdentityEvent(
            context=context,
            event_id="artifact",
            artifact_kind="inference_jsonl",
            created_at_ns=capture["started_at_ns"],
        ).to_record(),
        {
            "event_type": "infer.request",
            "endpoint": "http://127.0.0.1:8000/v1/completions",
            "schema_version": 1,
            "phase": "measured",
            "case_id": "c1",
            "status": "ok",
            "started_at_ns": capture["started_at_ns"],
            "ended_at_ns": capture["ended_at_ns"],
        },
        {
            "event_type": "infer.phase_window",
            "case_id": "c1",
            "phase": "measured",
            "started_at_ns": capture["started_at_ns"],
            "drained_at_ns": capture["ended_at_ns"],
        },
    ]


def test_run_id_is_declaration_and_memory_remains_process_scoped(
    native: Path,
) -> None:
    item = load_evidence(native)
    report = mlx_evidence_report(client(item), [native])
    process = report["processes"][0]
    assert process["association"]["state"] == "declared_direct"
    assert process["memory"]["metrics"]["active_allocator_bytes"]["sampled_max"] == 100
    assert process["cases"][0]["available"]
    assert process["memory"]["device_total_bytes"] is None
    assert report["request_execution_attribution"] == "unavailable"
    assert report["aggregation"].startswith("none")


def test_wrong_run_endpoint_and_missing_identity_are_unjoined(
    native: Path,
) -> None:
    item = load_evidence(native)
    for records in [
        client(item, run_id="other"),
        [],
        [r for r in client(item) if r["event_type"] != "infer.request"],
    ]:
        process = mlx_evidence_report(records, [native])["processes"][0]
        assert process["association"]["state"] == "unjoined"
        assert not any(c["available"] for c in process["cases"])


def test_verified_requires_observed_launcher_epoch_same_host_boot(
    native: Path,
) -> None:
    item = load_evidence(native)
    manifest = copy.deepcopy(item.manifest)
    manifest["producer"]["mode"] = "mlx-serve"
    p, c = manifest["process"], manifest["capture"]
    manifest["endpoint"]["routing_proof"] = dict(
        method="psutil_process_listening_socket",
        pid=p["pid"],
        start_time_ns=p["start_time_ns"],
        address="127.0.0.1",
        port=8000,
        observed_at_ns=c["started_at_ns"],
    )
    write_manifest(native, manifest)
    item = load_evidence(native)
    assert (
        mlx_evidence_report(client(item), [native])["association_counts"]["verified"]
        == 1
    )
    different = mlx_evidence_report(client(item, boot="other-boot"), [native])[
        "processes"
    ][0]
    assert different["association"]["state"] == "declared_direct"
    assert not different["clock"]["available"]


def test_uncertainty_excludes_boundary_samples_and_marks_missing_clock(
    native: Path,
) -> None:
    item = load_evidence(native)
    process = mlx_evidence_report(
        client(item), [native], clock_uncertainty_ns=10**12
    )["processes"][0]
    case = process["cases"][0]
    assert not case["available"]
    assert case["coverage"]["uncertain_boundary_samples"] > 0
    records = client(item, boot="other-boot")
    process = mlx_evidence_report(
        records, [native], clock_offset_ns=0, clock_uncertainty_ns=0
    )["processes"][0]
    assert process["clock"]["available"]
    assert process["association"]["state"] == "declared_direct"


def test_overlapping_physical_epochs_are_ambiguous(
    native: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stormlog.infer.mlx_report as module

    item = load_evidence(native)
    manifest = copy.deepcopy(item.manifest)
    manifest["process"]["pid"] += 1
    manifest["process"]["session_id"] = "other-process"
    other = replace(item, manifest=manifest)
    monkeypatch.setattr(module, "load_all_evidence", lambda _: [item, other])
    report = mlx_evidence_report(client(item), [native])
    assert report["association_counts"]["ambiguous"] == 2
    assert all(not p["cases"][0]["available"] for p in report["processes"])
    assert all(p["memory"]["samples"] > 0 for p in report["processes"])


def test_cli_offline_join_blocks_all_framework_and_server_imports(
    native: Path, tmp_path: Path
) -> None:
    item = load_evidence(native)
    path = tmp_path / "client.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in client(item)))
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'tensorflow', 'jax', 'omlx', 'mlx_lm', 'mlx_audio'}:
            raise AssertionError('Offline import attempted: ' + fullname)
sys.meta_path.insert(0, Block())
from stormlog.infer.cli import main
raise SystemExit(main(['analyze', sys.argv[1], '--mlx-server-evidence', sys.argv[2], '--format', 'json']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(native)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["telemetry"]["mlx"]["association_counts"]["declared_direct"] == 1


@pytest.mark.parametrize(
    ("host", "match"),
    [
        ("localhost", "loopback_alias"),
        ("LOCALHOST", "loopback_alias"),
        ("localhost.", "loopback_alias"),
        ("127.0.0.1", "literal"),
        ("[::ffff:127.0.0.1]", "literal"),
    ],
)
def test_declared_loopback_aliases_match_without_dns(
    native: Path, host: str, match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    def no_dns(*args: object, **kwargs: object) -> None:
        pytest.fail("offline association must not resolve DNS")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    item = load_evidence(native)
    records = client(item)
    records[1]["endpoint"] = f"http://{host}:8000/v1/completions"
    state = mlx_evidence_report(records, [native])["processes"][0]["association"]
    assert state["state"] == "declared_direct"
    assert state["endpoint_match"] == match
    assert state["request_ownership"] == "unavailable"


@pytest.mark.parametrize(
    "origin",
    [
        "https://localhost:8000",
        "http://localhost:8001",
        "http://0.0.0.0:8000",
        "http://example.test:8000",
        "http://localhost..:8000",
        # A server bound to 127.0.0.1 accepts neither: they are other sockets.
        "http://127.0.0.2:8000",
        "http://[::1]:8000",
    ],
)
def test_declared_endpoint_aliases_preserve_port_scheme_and_host_boundaries(
    native: Path, origin: str
) -> None:
    records = client(load_evidence(native))
    records[1]["endpoint"] = origin + "/v1/completions"
    state = mlx_evidence_report(records, [native])["processes"][0]["association"]
    assert state["state"] == "unjoined"
    assert state["endpoint_match"] is None


def test_loopback_alias_does_not_turn_listener_proof_into_routing_proof(
    native: Path,
) -> None:
    item = load_evidence(native)
    manifest = copy.deepcopy(item.manifest)
    manifest["producer"]["mode"] = "mlx-serve"
    process, capture = manifest["process"], manifest["capture"]
    manifest["endpoint"]["routing_proof"] = dict(
        method="psutil_process_listening_socket",
        pid=process["pid"],
        start_time_ns=process["start_time_ns"],
        address="127.0.0.1",
        port=8000,
        observed_at_ns=capture["started_at_ns"],
    )
    write_manifest(native, manifest)
    records = client(load_evidence(native))
    records[1]["endpoint"] = "http://localhost:8000/v1/completions"
    state = mlx_evidence_report(records, [native])["processes"][0]["association"]
    assert state["state"] == "declared_direct"
    assert state["endpoint_match"] == "loopback_alias"


def test_localhost_aliases_only_the_addresses_it_resolves_to(native: Path) -> None:
    # A server bound to 127.0.0.2 is loopback, but localhost never reaches it.
    item = load_evidence(native)
    manifest = copy.deepcopy(item.manifest)
    manifest["endpoint"]["host"] = "127.0.0.2"
    write_manifest(native, manifest)
    records = client(load_evidence(native))
    records[1]["endpoint"] = "http://localhost:8000/v1/completions"
    state = mlx_evidence_report(records, [native])["processes"][0]["association"]
    assert state["state"] == "unjoined"
    assert state["endpoint_match"] is None


def test_overlapping_alias_endpoints_remain_ambiguous(
    native: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stormlog.infer.mlx_report as module

    item = load_evidence(native)
    manifest = copy.deepcopy(item.manifest)
    manifest["endpoint"]["host"] = "localhost"
    manifest["process"]["pid"] += 1
    manifest["process"]["session_id"] = "other-process"
    other = replace(item, manifest=manifest)
    monkeypatch.setattr(module, "load_all_evidence", lambda _: [item, other])
    assert (
        mlx_evidence_report(client(item), [native])["association_counts"]["ambiguous"]
        == 2
    )


@pytest.mark.parametrize(
    ("host", "states"),
    [
        ("127.0.0.1", ["declared_direct", "unjoined"]),
        ("[::1]", ["unjoined", "declared_direct"]),
        # localhost can resolve to either listener, so neither is proven.
        ("localhost", ["ambiguous", "ambiguous"]),
    ],
)
def test_dual_stack_captures_join_only_the_listener_the_client_named(
    native: Path, host: str, states: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import stormlog.infer.mlx_report as module

    item = load_evidence(native)
    manifest = copy.deepcopy(item.manifest)
    manifest["endpoint"]["host"] = "::1"
    manifest["process"]["pid"] += 1
    manifest["process"]["session_id"] = "other-process"
    other = replace(item, manifest=manifest)
    monkeypatch.setattr(module, "load_all_evidence", lambda _: [item, other])
    records = client(item)
    records[1]["endpoint"] = f"http://{host}:8000/v1/completions"
    report = mlx_evidence_report(records, [native])
    assert [p["association"]["state"] for p in report["processes"]] == states
