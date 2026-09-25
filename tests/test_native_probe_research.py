"""Tests for the repository-local native probe experiment framework."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

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
    group = result["groups"]["nvidia-a100:off"]

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
    with pytest.raises(ValueError, match="missing evidence roles"):
        validate_matrix_promotions(
            matrix,
            [{"candidate_id": candidate["id"], "claim_id": claim, "evidence": []}],
            tmp_path,
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
