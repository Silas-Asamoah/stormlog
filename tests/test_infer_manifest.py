"""The run manifest: before, after and declared server descriptions."""

from __future__ import annotations

import contextlib
import copy
import email.utils
import io
import json
import time
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.analysis import analyze_inference_events, format_analysis_text
from stormlog.infer.cli import main as infer_main
from stormlog.infer.describe_server import (
    DESCRIPTION_FORMAT,
    description_digest,
    write_description,
)
from stormlog.infer.errors import InferInputError
from stormlog.infer.manifest import (
    after_refusals,
    attach_manifest,
    compare_descriptions,
    description_mismatches,
    description_record,
    load_declarations,
)
from tests.infer_workload_helpers import run_profile_with_fake_client


def _description(**changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "format": DESCRIPTION_FORMAT,
        "version": 1,
        "observed_at_ns": time.time_ns(),
        "run_id": None,
        "host": {"hostname": "gpu-box", "boot_id": "boot-1"},
        "server": {
            "pid": 100,
            "start_ticks": 500,
            "launch": {"model": "Qwen/Qwen2.5-0.5B", "revision": None},
        },
        "gpus": {
            "driver_version": "580.82.07",
            "cuda_driver_version": "13.0",
            "server_uuids": ["GPU-aaaa"],
            "devices": [
                {
                    "uuid": "GPU-aaaa",
                    "server_pids": [102],
                    "settings": {"name": "NVIDIA A30", "power_limit_w": 165.0},
                    "series": {"sm_clock_mhz": 1410, "temperature_c": 41},
                }
            ],
        },
        "model": {"weights_digest": "w" * 64, "resolved_snapshot": "a" * 40},
        "runtime": {"python": "3.12.3", "packages": {"torch": "2.9.0"}},
        "log": None,
    }
    for path, value in changes.items():
        target: Any = document
        *parents, leaf = path.split("__")
        for name in parents:
            target = target[name]
        target[leaf] = value
    document["sha256"] = description_digest(document)
    return document


def _write(path: Path, document: dict[str, Any]) -> Path:
    write_description(document, path)
    return path


def _profile(tmp_path: Path, **changes: Any) -> Path:
    run_profile_with_fake_client(tmp_path, latency_seconds=0.0, **changes)
    return tmp_path / "infer.jsonl"


def _later(document: dict[str, Any], **changes: Any) -> dict[str, Any]:
    later = copy.deepcopy(document)
    later.pop("sha256")
    later["observed_at_ns"] = time.time_ns() + 1_000_000_000
    for path, value in changes.items():
        target: Any = later
        *parents, leaf = path.split("__")
        for name in parents:
            target = target[name] if not name.isdigit() else target[int(name)]
        target[leaf] = value
    later["sha256"] = description_digest(later)
    return later


def test_a_profile_records_its_before_description_and_declarations(
    tmp_path: Path,
) -> None:
    before = _description()
    declarations = {
        "format": "stormlog.infer.declared",
        "version": 1,
        "fields": {"engine.version": "0.30.0", "host.purpose": "baseline"},
    }
    artifact = _profile(tmp_path, server_description=before, declarations=declarations)
    records = [json.loads(line) for line in artifact.read_text().splitlines()]
    manifests = [r for r in records if r.get("event_type") == "infer.manifest"]

    assert [m["role"] for m in manifests] == ["before", "declared"]
    assert manifests[0]["sha256"] == before["sha256"]
    assert manifests[0]["gpu_uuids"] == ["GPU-aaaa"]
    assert manifests[1]["provenance"] == "declared"
    report = analyze_inference_events(artifact)
    assert report["manifest"]["declared"] == declarations["fields"]
    assert report["manifest"]["before"][0]["sha256"] == before["sha256"]
    assert "Server manifest: before 1, after 0, declared yes" in format_analysis_text(
        report
    )


def test_an_after_description_of_the_same_lifetime_is_attached(
    tmp_path: Path,
) -> None:
    before = _description()
    artifact = _profile(tmp_path, server_description=before)
    after = _later(before, gpus__devices__0__series__sm_clock_mhz=1395)
    record = attach_manifest(artifact, _write(tmp_path / "after.json", after))

    assert record["role"] == "after"
    report = analyze_inference_events(artifact)
    manifest = report["manifest"]
    assert manifest["identity_changes"] == []
    assert manifest["protocol_failure"] is None
    assert manifest["drift"]["GPU-aaaa"]["sm_clock_mhz"] == {
        "before": 1410,
        "after": 1395,
    }
    assert "drift on GPU-aaaa: sm_clock_mhz 1410 -> 1395" in format_analysis_text(
        report
    )


def test_a_changed_identity_is_a_protocol_failure(tmp_path: Path) -> None:
    before = _description()
    artifact = _profile(tmp_path, server_description=before)
    after = _later(before, gpus__devices__0__settings__power_limit_w=150.0)
    attach_manifest(artifact, _write(tmp_path / "after.json", after))

    manifest = analyze_inference_events(artifact)["manifest"]
    assert manifest["protocol_failure"] == "identity_changed"
    assert manifest["identity_changes"] == [
        {"field": "gpu.GPU-aaaa.power_limit_w", "before": 165.0, "after": 150.0}
    ]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"run_id": "another-run"}, "it describes run another-run"),
        ({"host__boot_id": "boot-2"}, "another host or boot"),
        ({"server__start_ticks": 999}, "the server was restarted"),
        ({"observed_at_ns": 1}, "before the last measured phase ended"),
    ],
)
def test_an_after_description_that_cannot_be_compared_is_refused(
    tmp_path: Path, changes: dict[str, Any], reason: str
) -> None:
    before = _description()
    artifact = _profile(tmp_path, server_description=before)
    after = _later(before, **changes)
    with pytest.raises(InferInputError, match=reason):
        attach_manifest(artifact, _write(tmp_path / "after.json", after))


def test_attaching_needs_a_before_description_and_attaches_once(
    tmp_path: Path,
) -> None:
    without = _profile(tmp_path / "a")
    after = _later(_description())
    with pytest.raises(InferInputError, match="no before description"):
        attach_manifest(without, _write(tmp_path / "after.json", after))

    artifact = _profile(tmp_path / "b", server_description=_description())
    path = _write(tmp_path / "after.json", _later(_description()))
    attach_manifest(artifact, path)
    with pytest.raises(InferInputError, match="already attached"):
        attach_manifest(artifact, path)


@pytest.mark.parametrize(
    "missing",
    [{"log": None}, {"runtime": None}, {"gpus__devices__0__settings": {}}],
    ids=["no_server_log", "no_python", "no_gpu_settings"],
)
def test_an_after_with_less_evidence_is_unverified_not_a_change(
    missing: dict[str, Any],
) -> None:
    # The before had the server log, Python and NVML; the after did not.
    before = _description(
        log={"attention_backend": "FLASH_ATTN", "kv_cache_size_tokens": 890960}
    )
    after = _later(before, **missing)
    compared = compare_descriptions(before, after)
    assert compared["identity_changes"] == []
    assert compared["identity_unverified"]


def test_less_evidence_after_the_run_is_no_protocol_failure(tmp_path: Path) -> None:
    before = _description(
        log={"attention_backend": "FLASH_ATTN", "kv_cache_size_tokens": 890960}
    )
    artifact = _profile(tmp_path, server_description=before)
    attach_manifest(artifact, _write(tmp_path / "after.json", _later(before, log=None)))
    manifest = analyze_inference_events(artifact)["manifest"]
    assert manifest["protocol_failure"] is None
    assert [item["field"] for item in manifest["identity_unverified"]] == [
        "log.attention_backend",
        "log.kv_cache_size_tokens",
    ]


def test_the_before_description_cannot_be_attached_as_the_after(
    tmp_path: Path,
) -> None:
    # Without a measured window (an interrupted run, or a server clock ahead
    # of the client's) only the document's own identity gives it away.
    before = _description()
    artifact = _profile(tmp_path, server_description=before)
    lines = artifact.read_text().splitlines()
    kept = [line for line in lines if '"infer.phase_window"' not in line]
    artifact.write_text("\n".join(kept) + "\n")
    with pytest.raises(InferInputError, match="it is the before description"):
        attach_manifest(artifact, _write(tmp_path / "after.json", before))


SECOND = 1_000_000_000


def _skewed_run(description: dict[str, Any]) -> list[dict[str, Any]]:
    """A run whose measured phase took 10 s, by a client clock far behind."""
    return [
        description_record(description, role="before", session_id="s", run_id="r"),
        {
            "event_type": "infer.phase_window",
            "phase": "measured",
            "started_at_ns": 5 * SECOND,
            "drained_at_ns": 15 * SECOND,
        },
    ]


def test_an_after_must_be_taken_after_the_before() -> None:
    before = _description(observed_at_ns=1_000 * SECOND)
    earlier = _later(before, observed_at_ns=999 * SECOND)
    refusals = after_refusals(_skewed_run(before), earlier, run_id="r")
    assert refusals == ["it was taken before the before description"]


def test_an_after_closer_to_the_before_than_the_run_lasted_is_refused() -> None:
    # The server's clock is far ahead, so by the client's clock any after is
    # late enough. Each clock's own interval still shows the run did not fit.
    before = _description(observed_at_ns=1_000 * SECOND)
    soon = _later(before, observed_at_ns=1_002 * SECOND)
    refusals = after_refusals(_skewed_run(before), soon, run_id="r")
    assert refusals == [
        "it was taken 2.0 s after the before description, but the run's "
        "measured phases took 10.0 s"
    ]
    late = _later(before, observed_at_ns=1_011 * SECOND)
    assert after_refusals(_skewed_run(before), late, run_id="r") == []


def test_an_after_taken_before_the_server_answered_the_after_probe_is_refused() -> None:
    # Taken mid-run: 12 s after the before description covers the 10 s run,
    # and the client's clock is behind, so only the server's own clock,
    # stamped on the after probe's answers, shows it came too early.
    before = _description(observed_at_ns=1_000 * SECOND)
    after_probe = {
        "event_type": "infer.server_probe",
        "phase": "after",
        "answers": {"/version": {"status": "ok", "date": _http_date(1_020)}},
    }
    records = [*_skewed_run(before), after_probe]
    midrun = _later(before, observed_at_ns=1_012 * SECOND)
    assert after_refusals(records, midrun, run_id="r") == [
        "it was taken before the server answered the after probe, by the "
        "server's own clock"
    ]
    after = _later(before, observed_at_ns=1_020 * SECOND + SECOND // 2)
    assert after_refusals(records, after, run_id="r") == []


def _http_date(seconds: int) -> str:
    return email.utils.formatdate(seconds, usegmt=True)


def test_other_weights_after_the_run_are_an_identity_change() -> None:
    before = _description()
    after = _later(before, model__weights_digest="v" * 64)
    changes = compare_descriptions(before, after)["identity_changes"]
    assert [change["field"] for change in changes] == ["model.weights_digest"]


def test_settings_that_drift_are_not_identity() -> None:
    before = _description()
    after = _later(
        before,
        gpus__devices__0__series__temperature_c=70,
        observed_at_ns=time.time_ns() + 10,
    )
    compared = compare_descriptions(before, after)
    assert compared["identity_changes"] == []
    assert compared["drift"]["GPU-aaaa"]["temperature_c"]["after"] == 70


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"format": "other", "version": 1, "fields": {}}',
        '{"format": "stormlog.infer.declared", "version": 2, "fields": {}}',
        '{"format": "stormlog.infer.declared", "version": 1, "fields": []}',
    ],
)
def test_a_declarations_file_must_be_the_versioned_format(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "declared.json"
    path.write_text(content)
    with pytest.raises(InferInputError, match="--declare"):
        load_declarations(path)


def _cli(*argv: str) -> tuple[int, str]:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = infer_main(list(argv))
    return code, stderr.getvalue()


def test_the_cli_attaches_or_refuses_with_exit_5(tmp_path: Path) -> None:
    before = _description()
    artifact = _profile(tmp_path, server_description=before)
    stale = _write(tmp_path / "stale.json", _later(before, observed_at_ns=1))
    code, err = _cli("attach-manifest", str(artifact), str(stale))
    assert code == ExitCode.INVALID_INPUT
    assert "before the last measured phase ended" in err

    fresh = _write(tmp_path / "after.json", _later(before))
    code, _err = _cli("attach-manifest", str(artifact), str(fresh), "--role", "after")
    assert code == ExitCode.OK


def test_profile_refuses_a_description_of_another_run(tmp_path: Path) -> None:
    other = _write(tmp_path / "before.json", _description(run_id="R1"))
    code, err = _cli(
        "profile",
        "--endpoint",
        "http://127.0.0.1:1/v1/chat/completions",
        "--model",
        "m",
        "--system-sampler",
        "none",
        "--tokenizer",
        "none",
        "--server-probe",
        "none",
        "--run-id",
        "R2",
        "--describe-server",
        str(other),
        "--output",
        str(tmp_path / "infer.jsonl"),
    )
    assert code == ExitCode.INVALID_INPUT
    assert "it describes run R1, not R2" in err
    assert not (tmp_path / "infer.jsonl").exists()


def test_a_description_that_does_not_match_the_probed_server_is_a_protocol_failure(
    tmp_path: Path,
) -> None:
    before = _description(runtime={"python": "3.12.3", "packages": {"vllm": "0.29.0"}})
    artifact = _profile(tmp_path, server_description=before)
    probe = {
        "event_type": "infer.server_probe",
        "phase": "before",
        "answers": {
            "/version": {"status": "ok", "body": {"version": "0.30.0"}},
            "/v1/models": {
                "status": "ok",
                "body": {"data": [{"id": "other-model", "root": "Other/Model"}]},
            },
        },
    }
    with artifact.open("a") as handle:
        handle.write(json.dumps(probe) + "\n")
    manifest = analyze_inference_events(artifact)["manifest"]
    assert manifest["protocol_failure"] == "description_mismatch"
    assert {item["field"] for item in manifest["description_mismatches"]} == {
        "model",
        "engine.version",
    }


def test_a_served_model_name_is_matched_by_the_models_root() -> None:
    # vLLM lists a --served-model-name deployment under its alias, with the
    # model it loaded as root: the same server, never a mismatch.
    probe = {
        "event_type": "infer.server_probe",
        "phase": "before",
        "answers": {
            "/v1/models": {
                "status": "ok",
                "body": {"data": [{"id": "my-alias", "root": "Qwen/Qwen2.5-0.5B"}]},
            }
        },
    }
    assert description_mismatches(_description(), [probe]) == []


@pytest.mark.parametrize(
    ("origin", "shares", "mismatch"),
    [
        ("http://127.0.0.1:8013", True, False),
        ("http://127.0.0.1:8014", True, True),
        ("http://localhost:8014", True, True),
        # Another network namespace may map the port, such as a container's.
        ("http://127.0.0.1:18013", False, False),
        # Through a remote address a proxy may map it too.
        ("http://10.0.0.5:18013", True, False),
    ],
)
def test_a_twin_server_on_another_port_fails_the_cross_check(
    origin: str, shares: bool, mismatch: bool
) -> None:
    # Same model, version and driver: only the port it listens on tells a
    # twin server on another GPU from the one the run measured.
    description = _description()
    description["server"].update(listen_ports=[8013], shares_network_namespace=shares)
    probe = {
        "event_type": "infer.server_probe",
        "phase": "before",
        "origin": origin,
        "answers": {},
    }
    found = description_mismatches(description, [probe])
    expected = [{"field": "server.port", "description": [8013], "server": [8014]}]
    assert (found == expected) if mismatch else (found == [])


def test_profile_refuses_a_description_it_cannot_read_before_sending(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "before.json"
    broken.write_text(json.dumps({**_description(), "sha256": "0" * 64}))
    code, err = _cli(
        "profile",
        "--endpoint",
        "http://127.0.0.1:1/v1/chat/completions",
        "--model",
        "m",
        "--system-sampler",
        "none",
        "--tokenizer",
        "none",
        "--server-probe",
        "none",
        "--describe-server",
        str(broken),
        "--output",
        str(tmp_path / "infer.jsonl"),
    )
    assert code == ExitCode.INVALID_INPUT
    assert "sha256 does not match" in err
    assert not (tmp_path / "infer.jsonl").exists()
