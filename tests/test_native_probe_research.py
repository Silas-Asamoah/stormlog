"""Tests for the repository-local native probe experiment framework."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Sequence

import jsonschema
import pytest

from research.native_probes.analysis import analyze_trials, paired_perturbations
from research.native_probes.mode_commands import (
    Workload,
    expected_artifacts,
    microbenchmark_command,
    process_roles,
)
from research.native_probes.models import (
    ArtifactExpectation,
    CommandSpec,
    ExperimentMode,
    TrialSpec,
    WorkloadId,
)
from research.native_probes.normalization import (
    classify_overlap,
    normalize_loss,
    normalize_trial,
    validate_measurement_window,
)
from research.native_probes.planning import build_plan, counterbalanced_order, trial_id
from research.native_probes.preflight import collect_environment, write_manifest
from research.native_probes.references import _safe_extract
from research.native_probes.runner import run_trial
from research.native_probes.validation import (
    build_unvalidated_matrix,
    validate_matrix_promotions,
)
from research.native_probes.workloads.cuda_microbench import w2_contract

REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMAS = REPOSITORY / "research/native_probes/schemas"
MATRICES = REPOSITORY / "research/native_probes/matrices"


def test_preflight_is_explicit_about_unavailable_accelerators() -> None:
    manifest = collect_environment("test-control", REPOSITORY)

    _validate(manifest, "environment.schema.json")
    assert manifest["source"]["revision"]
    assert manifest["mode_qualifications"]["off"]["status"] == "pass"
    assert manifest["platform"]["python_executable"]
    if manifest["platform"]["system"] == "Darwin":
        assert (
            manifest["mode_qualifications"]["ebpf-semantic-linux"]["status"]
            == "unsupported"
        )


def test_preflight_treats_vendor_tools_as_alternatives() -> None:
    paths = {"nvidia-smi": "/bin/nvidia-smi", "nsys": "/bin/nsys"}

    def probe(argv: Sequence[str]) -> tuple[bool, str]:
        command = tuple(argv)
        if "--query-gpu=name,compute_cap,driver_version" in command:
            return True, "A100, 8.0, 555.1"
        return True, "tool 1.0"

    manifest = collect_environment(
        "nvidia", REPOSITORY, command_probe=probe, system="Linux", tool_paths=paths
    )

    assert manifest["mode_qualifications"]["trusted-nvidia"]["status"] == "untested"
    assert manifest["mode_qualifications"]["trusted-amd"]["status"] == "unsupported"
    assert manifest["accelerators"]["nvidia"]["cuda_toolkit_detected"] is False


def test_preflight_does_not_treat_nvcc_as_a_gpu() -> None:
    manifest = collect_environment(
        "toolkit-only",
        REPOSITORY,
        command_probe=lambda argv: (True, "12.8"),
        system="Linux",
        tool_paths={"nvcc": "/bin/nvcc"},
    )

    assert manifest["accelerators"]["nvidia"]["cuda_toolkit_detected"] is True
    assert manifest["accelerators"]["nvidia"]["gpu_detected"] is False
    assert (
        manifest["mode_qualifications"]["direct-cupti-nvidia"]["status"]
        == "unsupported"
    )


def test_source_capability_matrix_is_complete_and_not_mislabeled() -> None:
    matrix = json.loads((MATRICES / "source_backed.json").read_text(encoding="utf-8"))

    _validate(matrix, "capability_matrix.schema.json")
    assert len(matrix["candidates"]) >= 7
    for candidate in matrix["candidates"]:
        assert candidate["evidence_sources"]
        statuses = {claim["status"] for claim in candidate["claims"].values()}
        assert "STORMLOG_VALIDATED" not in statuses


def test_validated_matrix_does_not_promote_unrun_experiments() -> None:
    matrix = json.loads(
        (MATRICES / "stormlog_validated.json").read_text(encoding="utf-8")
    )

    _validate(matrix, "capability_matrix.schema.json")
    assert matrix["matrix_kind"] == "stormlog_validated"
    for candidate in matrix["candidates"]:
        for claim in candidate["claims"].values():
            assert claim["status"] == "UNKNOWN"
            assert claim["detail"].startswith("UNTESTED - ")
            for evidence in claim["evidence"]:
                assert (REPOSITORY / evidence).is_file()


def test_validated_matrix_initialization_resets_source_confidence() -> None:
    source = json.loads((MATRICES / "source_backed.json").read_text(encoding="utf-8"))

    result = build_unvalidated_matrix(source, "evidence/environment.json")

    statuses = {
        claim["status"]
        for candidate in result["candidates"]
        for claim in candidate["claims"].values()
    }
    assert statuses == {"UNKNOWN"}
    assert source["candidates"][0]["claims"]["signal"]["status"] != "UNKNOWN"


def test_manifest_writer_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    checksum = write_manifest(output, {"value": 1})

    assert len(checksum) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(FileExistsError):
        write_manifest(output, {"value": 2})


def test_reference_extraction_rejects_traversal_and_extracts_regular_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "reference"
    destination.mkdir()

    _safe_extract(
        _tar_bytes("native/cupti/CMakeLists.txt", b"project(test)"), destination
    )

    extracted = destination / "native/cupti/CMakeLists.txt"
    assert extracted.read_bytes() == b"project(test)"
    with pytest.raises(ValueError, match="unsafe archive member"):
        _safe_extract(_tar_bytes("../escape", b"unsafe"), destination)
    assert not (tmp_path / "escape").exists()


def test_counterbalanced_order_is_reproducible_and_rotates() -> None:
    modes = [
        ExperimentMode.OFF,
        ExperimentMode.PUBLIC_PYTORCH,
        ExperimentMode.DIRECT_CUPTI,
    ]

    first = counterbalanced_order(modes, repetitions=6, seed=118)
    second = counterbalanced_order(modes, repetitions=6, seed=118)

    assert first == second
    assert all(set(order) == set(modes) for order in first)
    assert len({tuple(order) for order in first}) > 1


def test_trial_id_is_stable_and_rejects_unsafe_identity() -> None:
    assert trial_id("nvidia-a100", "w1-eager", ExperimentMode.OFF, 0) == trial_id(
        "nvidia-a100", "w1-eager", ExperimentMode.OFF, 0
    )
    with pytest.raises(ValueError, match="configuration_id"):
        trial_id("../escape", "w1-eager", ExperimentMode.OFF, 0)


def test_analysis_retains_failures_unknowns_and_individual_values() -> None:
    trials = [
        _trial("trial-1", "pass", 10.0, None),
        _trial("trial-2", "fail", 14.0, 1.0),
        _trial("trial-3", "pass", 12.0, 0.0),
    ]

    result = analyze_trials(trials, bootstrap_samples=200)
    group = result["groups"]["nvidia-a100:w1-eager:off"]

    assert group["trial_count"] == 3
    assert group["failure_count"] == 1
    assert group["status_counts"]["fail"] == 1
    assert group["metrics"]["latency_ms"]["individual_values"] == [10.0, 12.0, 14.0]
    assert group["metrics"]["latency_ms"]["median"] == 12.0
    assert group["unknown_metric_counts"] == {"loss_rate": 1}


def test_command_spec_requires_argv_and_positive_timeout() -> None:
    command = CommandSpec.from_mapping(
        {"argv": ["python", "workload.py"], "timeout_seconds": 30}
    )

    assert command.argv == ("python", "workload.py")
    with pytest.raises(ValueError, match="positive"):
        CommandSpec.from_mapping({"argv": ["python"], "timeout_seconds": 0})


def test_runner_retains_metrics_logs_and_checksums(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "artifact_kind": "workload_result",
            "metrics": {"latency_ms": 2.5},
            "measurement_window": _window(),
            "ground_truth": {},
        }
    )
    command = CommandSpec((sys.executable, "-c", f"print({payload!r})"), {}, 5.0)
    spec = TrialSpec(
        trial_id="control-w1-off-r00",
        configuration_id="control",
        workload_id=WorkloadId.W1_EAGER,
        mode=ExperimentMode.OFF,
        repetition=0,
        command=command,
        expected_artifacts=expected_artifacts(ExperimentMode.OFF),
        process_roles=process_roles(ExperimentMode.OFF),
    )

    result = run_trial(spec, tmp_path)

    _validate(result, "trial.schema.json")
    assert result["status"] == "pass"
    assert result["metrics"]["latency_ms"] == 2.5
    assert result["artifacts"][0]["sha256"]
    stdout_log = tmp_path / "control" / "trials" / spec.trial_id / "logs" / "stdout.log"
    assert (
        result["artifacts"][0]["sha256"]
        == hashlib.sha256(stdout_log.read_bytes()).hexdigest()
    )
    assert result["resources"]["target"]["status"] == "observed"


def test_runner_rejects_secret_environment_keys(tmp_path: Path) -> None:
    command = CommandSpec((sys.executable, "-c", "pass"), {"API_TOKEN": "x"}, 5.0)
    spec = TrialSpec(
        trial_id="secret",
        configuration_id="control",
        workload_id=WorkloadId.W1_EAGER,
        mode=ExperimentMode.OFF,
        repetition=0,
        command=command,
    )

    with pytest.raises(ValueError, match="secret-like"):
        run_trial(spec, tmp_path)


def test_runner_downgrades_missing_result_and_malformed_trace(tmp_path: Path) -> None:
    malformed_result = TrialSpec(
        trial_id="bad-result",
        configuration_id="control",
        workload_id=WorkloadId.W1_EAGER,
        mode=ExperimentMode.OFF,
        repetition=0,
        command=CommandSpec((sys.executable, "-c", "print('not-json')"), {}, 5.0),
        expected_artifacts=expected_artifacts(ExperimentMode.OFF),
        process_roles=process_roles(ExperimentMode.OFF),
    )
    assert run_trial(malformed_result, tmp_path)["status"] == "partial"

    trial_id = "bad-trace"
    trace_path = tmp_path / "control" / "trials" / trial_id / "trace.json"
    payload = {
        "artifact_kind": "workload_result",
        "metrics": {"latency_ms": 1.0},
        "measurement_window": _window(),
        "ground_truth": {},
    }
    code = (
        "from pathlib import Path; import json; "
        f"Path({str(trace_path)!r}).write_text('{{broken'); "
        f"print({json.dumps(payload)!r})"
    )
    malformed_trace = TrialSpec(
        trial_id=trial_id,
        configuration_id="control",
        workload_id=WorkloadId.W1_EAGER,
        mode=ExperimentMode.PUBLIC_PYTORCH,
        repetition=0,
        command=CommandSpec((sys.executable, "-c", code), {}, 5.0),
        expected_artifacts=(
            ArtifactExpectation(
                "trace", "raw_trace", "trace.json", "target", "chrome-trace-json"
            ),
        ),
        process_roles=process_roles(ExperimentMode.OFF),
    )
    result = run_trial(malformed_trace, tmp_path)
    assert result["status"] == "partial"
    assert result["artifacts"][0]["status"] == "malformed"


def test_runner_marks_missing_required_profiler_artifact_partial(
    tmp_path: Path,
) -> None:
    payload = json.dumps(
        {
            "artifact_kind": "workload_result",
            "metrics": {},
            "measurement_window": _window(),
            "ground_truth": {},
        }
    )
    spec = TrialSpec(
        trial_id="missing",
        configuration_id="control",
        workload_id=WorkloadId.W1_EAGER,
        mode=ExperimentMode.PUBLIC_PYTORCH,
        repetition=0,
        command=CommandSpec((sys.executable, "-c", f"print({payload!r})"), {}, 5.0),
        expected_artifacts=expected_artifacts(ExperimentMode.PUBLIC_PYTORCH),
        process_roles=process_roles(ExperimentMode.PUBLIC_PYTORCH),
    )

    result = run_trial(spec, tmp_path)

    assert result["status"] == "partial"
    assert (
        next(
            row for row in result["artifacts"] if row["artifact_id"] == "pytorch-trace"
        )["status"]
        == "missing"
    )


def test_runner_timeout_retains_partial_evidence(tmp_path: Path) -> None:
    spec = TrialSpec(
        trial_id="timeout",
        configuration_id="control",
        workload_id=WorkloadId.W1_EAGER,
        mode=ExperimentMode.OFF,
        repetition=0,
        command=CommandSpec(
            (sys.executable, "-c", "import time; time.sleep(10)"), {}, 0.1
        ),
        expected_artifacts=expected_artifacts(ExperimentMode.OFF),
        process_roles=process_roles(ExperimentMode.OFF),
    )

    result = run_trial(spec, tmp_path)

    assert result["status"] == "timeout"
    assert all(row["status"] == "present" for row in result["artifacts"])


def test_paired_perturbation_requires_matched_nonzero_baseline() -> None:
    baseline = _paired_trial("off-0", "off", 0, 10.0)
    profiled = _paired_trial("public-0", "public-pytorch", 0, 12.0)
    unmatched = _paired_trial("public-1", "public-pytorch", 1, 13.0)

    comparisons = paired_perturbations([baseline, profiled, unmatched], "latency_ms")

    assert comparisons[0]["percent_delta"] == 20.0
    assert comparisons[1]["status"] == "unknown"
    assert comparisons[1]["reason"] == "matched off trial missing"


def test_mode_commands_preserve_identical_workload_parameters(tmp_path: Path) -> None:
    workload = Workload(WorkloadId.W2_OVERLAP, warmup=10, iterations=50, seed=7)
    off = microbenchmark_command(ExperimentMode.OFF, workload, tmp_path)
    public = microbenchmark_command(ExperimentMode.PUBLIC_PYTORCH, workload, tmp_path)

    assert public.argv[: len(off.argv)] == off.argv
    assert public.argv[-2] == "--torch-trace"
    with pytest.raises(ValueError, match="session-managed"):
        microbenchmark_command(ExperimentMode.PROTON, workload, tmp_path)


def test_workload_ids_match_persisted_trial_schema() -> None:
    schema = json.loads((SCHEMAS / "trial.schema.json").read_text(encoding="utf-8"))

    assert set(schema["properties"]["workload_id"]["enum"]) == {
        row.value for row in WorkloadId
    }


def test_w2_contract_separates_design_from_observation() -> None:
    overlap = w2_contract(
        serialized=False, stream_ids=[11, 12], operations_per_stream=8, elements=256
    )
    serialized = w2_contract(
        serialized=True, stream_ids=[11, 12], operations_per_stream=8, elements=256
    )

    assert overlap["overlap_eligible_by_design"] is True
    assert overlap["overlap_observed_in_trusted_trace"] is None
    assert overlap["ordering_edges"] == []
    assert serialized["overlap_eligible_by_design"] is False
    assert serialized["ordering_edges"] == [[11, 12]]


def test_overlap_requires_demonstrated_trusted_ground_truth() -> None:
    result = classify_overlap(
        {"overlap_observed_in_trusted_trace": False},
        {"concurrent_interval_ns": 100},
    )

    assert result["status"] == "partial"
    assert result["eligible"] is False
    assert result["overlap_preserved_by_candidate"] is None


def test_loss_domains_remain_separate_and_unknown() -> None:
    loss = normalize_loss(
        {
            "vendor_activity": {
                "status": "reported",
                "lost_records": 0,
                "expected_records": 4,
            }
        }
    )

    assert loss["complete"] is False
    assert loss["domains"]["vendor_activity"]["lost_records"] == 0
    assert loss["domains"]["bpf_transport"]["lost_records"] is None


def test_measurement_window_rejects_partial_boundaries() -> None:
    assert validate_measurement_window(_window()) == []
    assert "collector flush did not complete" in validate_measurement_window(
        {**_window(), "flush_completed": False}
    )
    assert "measurement window measured_iterations must be an integer >= 1" in (
        validate_measurement_window({**_window(), "measured_iterations": 0})
    )


def test_plan_is_counterbalanced_and_schema_valid(tmp_path: Path) -> None:
    plan = build_plan(
        configuration_id="nvidia-test",
        vendor="nvidia",
        workloads=[WorkloadId.W1_EAGER],
        modes=[ExperimentMode.OFF, ExperimentMode.PUBLIC_PYTORCH],
        repetitions=5,
        seed=118,
        environment_artifact="environment.json",
        artifact_root=tmp_path,
    )

    _validate(plan, "plan.schema.json")
    assert len(plan["trials"]) == 10
    assert {row["mode"] for row in plan["trials"]} == {"off", "public-pytorch"}


def test_normalization_downgrades_missing_measurement_window() -> None:
    trial = {
        "trial_id": "t",
        "configuration_id": "c",
        "workload_id": "w1-eager",
        "mode": "off",
        "repetition": 0,
        "status": "pass",
        "metrics": {},
        "artifacts": [],
        "limitations": [],
    }

    normalized = normalize_trial(trial)

    _validate(normalized, "normalized.schema.json")
    assert normalized["status"] == "partial"


def test_matrix_promotion_requires_all_evidence_roles(tmp_path: Path) -> None:
    matrix = json.loads((MATRICES / "stormlog_validated.json").read_text())
    candidate = matrix["candidates"][0]
    claim = next(iter(candidate["claims"]))
    with pytest.raises(ValueError, match="five unique evidence roles"):
        validate_matrix_promotions(
            matrix,
            [{"candidate_id": candidate["id"], "claim_id": claim, "evidence": []}],
            tmp_path,
        )


def test_matrix_promotion_rejects_unlinked_checksum_bundle(tmp_path: Path) -> None:
    matrix = json.loads((MATRICES / "stormlog_validated.json").read_text())
    candidate = matrix["candidates"][0]
    claim = next(iter(candidate["claims"]))
    evidence = []
    for role in ("environment", "command", "raw_artifact", "trial", "analysis"):
        path = tmp_path / f"{role}.json"
        path.write_text("{}", encoding="utf-8")
        evidence.append(
            {
                "role": role,
                "path": path.name,
                "sha256": hashlib.sha256(b"{}").hexdigest(),
            }
        )
    with pytest.raises(ValueError):
        validate_matrix_promotions(
            matrix,
            [
                {
                    "candidate_id": candidate["id"],
                    "claim_id": claim,
                    "detail": "hardware run",
                    "evidence": evidence,
                }
            ],
            tmp_path,
        )


def test_matrix_promotion_preserves_verified_role_hash_provenance(
    tmp_path: Path,
) -> None:
    matrix = json.loads((MATRICES / "stormlog_validated.json").read_text())
    candidate = matrix["candidates"][0]
    claim = next(iter(candidate["claims"]))
    documents = _promotion_fixture(tmp_path, candidate["id"], claim)

    promoted = validate_matrix_promotions(matrix, [documents], tmp_path)

    promoted_claim = promoted["candidates"][0]["claims"][claim]
    assert promoted_claim["status"] == "STORMLOG_VALIDATED"
    assert {row["role"] for row in promoted_claim["evidence_roles"]} == {
        "environment",
        "command",
        "raw_artifact",
        "trial",
        "analysis",
    }
    assert all(len(row["sha256"]) == 64 for row in promoted_claim["evidence_roles"])


def test_matrix_promotion_rejects_cross_file_workload_mismatch(tmp_path: Path) -> None:
    matrix = json.loads((MATRICES / "stormlog_validated.json").read_text())
    candidate = matrix["candidates"][0]
    claim = next(iter(candidate["claims"]))
    promotion = _promotion_fixture(tmp_path, candidate["id"], claim)
    trial_path = tmp_path / "trial.json"
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    trial["workload_id"] = "w2-overlap"
    trial_path.write_text(json.dumps(trial), encoding="utf-8")
    promotion["workload_id"] = "w2-overlap"
    promotion["evidence"] = _evidence_rows(tmp_path)

    with pytest.raises(ValueError, match="mismatch for workload_id"):
        validate_matrix_promotions(matrix, [promotion], tmp_path)


def test_matrix_promotion_rejects_tampered_digest_and_symlink(tmp_path: Path) -> None:
    matrix = json.loads((MATRICES / "stormlog_validated.json").read_text())
    candidate = matrix["candidates"][0]
    claim = next(iter(candidate["claims"]))

    (tmp_path / "tampered").mkdir()
    tampered = _promotion_fixture(tmp_path / "tampered", candidate["id"], claim)
    (tmp_path / "tampered" / "raw.trace").write_bytes(b"changed after hashing")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_matrix_promotions(matrix, [tampered], tmp_path / "tampered")

    (tmp_path / "linked").mkdir()
    linked = _promotion_fixture(tmp_path / "linked", candidate["id"], claim)
    root = tmp_path / "linked"
    (root / "environment-link.json").symlink_to(root / "environment.json")
    linked["evidence"] = _evidence_rows(root)
    linked["evidence"][0]["path"] = "environment-link.json"
    with pytest.raises(ValueError, match="symlink"):
        validate_matrix_promotions(matrix, [linked], root)


def test_analysis_groups_by_workload_identity() -> None:
    result = analyze_trials(
        [
            {**_trial("w1", "pass", 10.0, None), "workload_id": "w1-eager"},
            {**_trial("w2", "pass", 90.0, None), "workload_id": "w2-overlap"},
        ],
        bootstrap_samples=100,
    )

    assert set(result["groups"]) == {
        "nvidia-a100:w1-eager:off",
        "nvidia-a100:w2-overlap:off",
    }


def test_pairing_rejects_duplicate_and_nonfinite_trials() -> None:
    baseline = _paired_trial("off-0", "off", 0, 10.0)
    with pytest.raises(ValueError, match="duplicate"):
        paired_perturbations([baseline, baseline], "latency_ms")
    with pytest.raises(ValueError, match="finite"):
        paired_perturbations(
            [baseline, _paired_trial("public-0", "public-pytorch", 0, float("nan"))],
            "latency_ms",
        )


def _window() -> dict[str, object]:
    return {
        "range_id": "range",
        "marker": "nvtx-range",
        "warmup_iterations": 1,
        "measured_iterations": 2,
        "host_started_ns": 1,
        "host_finished_ns": 2,
        "clock": "CLOCK_REALTIME",
        "flush_completed": True,
    }


def _validate(instance: object, schema_name: str) -> None:
    schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(instance)


def _tar_bytes(name: str, contents: bytes) -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
        member = tarfile.TarInfo(name)
        member.size = len(contents)
        output.addfile(member, io.BytesIO(contents))
    return archive.getvalue()


def _trial(
    trial: str, status: str, latency_ms: float, loss_rate: float | None
) -> dict[str, object]:
    return {
        "trial_id": trial,
        "configuration_id": "nvidia-a100",
        "workload_id": "w1-eager",
        "mode": "off",
        "status": status,
        "metrics": {"latency_ms": latency_ms, "loss_rate": loss_rate},
    }


def _paired_trial(
    trial: str, mode: str, repetition: int, latency_ms: float
) -> dict[str, object]:
    return {
        "trial_id": trial,
        "configuration_id": "nvidia-a100",
        "workload_id": "w1-eager",
        "mode": mode,
        "repetition": repetition,
        "metrics": {"latency_ms": latency_ms},
    }


def _promotion_fixture(
    tmp_path: Path, candidate_id: str, claim_id: str
) -> dict[str, object]:
    revision = "a" * 40
    raw_path = tmp_path / "raw.trace"
    raw_path.write_bytes(b"usable raw trace")
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    trial_id = "fixture-trial-1"
    documents = {
        "environment": {
            "artifact_kind": "native_probe_environment",
            "source": {"revision": revision},
        },
        "command": {
            "artifact_kind": "native_probe_plan",
            "configuration_id": "fixture-config",
            "environment_artifact": "environment.json",
            "revision": revision,
            "trials": [
                {
                    "trial_id": trial_id,
                    "configuration_id": "fixture-config",
                    "workload_id": "w1-eager",
                    "mode": "off",
                    "repetition": 0,
                    "command": {
                        "argv": ["fixture"],
                        "environment": {},
                        "timeout_seconds": 1,
                    },
                }
            ],
        },
        "trial": {
            "artifact_kind": "native_probe_trial",
            "trial_id": trial_id,
            "configuration_id": "fixture-config",
            "workload_id": "w1-eager",
            "mode": "off",
            "repetition": 0,
            "revision": revision,
            "command": {
                "argv": ["fixture"],
                "environment": {},
                "timeout_seconds": 1,
            },
            "status": "pass",
            "return_code": 0,
            "measurement_window": _window(),
            "artifacts": [
                {
                    "path": str(raw_path),
                    "artifact_id": "trace-1",
                    "sha256": raw_hash,
                    "status": "present",
                    "required": True,
                    "kind": "cuda_trace",
                    "producer": "fixture",
                }
            ],
        },
        "analysis": {
            "artifact_kind": "native_probe_analysis",
            "groups": {"fixture": {"trial_ids": [trial_id]}},
            "claim_evidence": {
                claim_id: {
                    "status": "pass",
                    "basis": "reviewed fixture outcome",
                    "criterion": "one verified trace interval meets the claim criterion",
                    "observed": "the linked trace records the required interval",
                    "trial_ids": [trial_id],
                    "artifact_ids": ["trace-1"],
                }
            },
        },
    }
    for name, document in documents.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")
    evidence = _evidence_rows(tmp_path)
    return {
        "candidate_id": candidate_id,
        "claim_id": claim_id,
        "detail": "verified fixture evidence",
        "trial_id": trial_id,
        "configuration_id": "fixture-config",
        "workload_id": "w1-eager",
        "mode": "off",
        "repetition": 0,
        "revision": revision,
        "evidence": evidence,
    }


def _evidence_rows(tmp_path: Path) -> list[dict[str, str]]:
    names = {
        "environment": "environment.json",
        "command": "command.json",
        "raw_artifact": "raw.trace",
        "trial": "trial.json",
        "analysis": "analysis.json",
    }
    return [
        {
            "role": role,
            "path": filename,
            "sha256": hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest(),
        }
        for role, filename in names.items()
    ]
