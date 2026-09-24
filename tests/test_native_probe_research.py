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
from research.native_probes.mode_commands import Workload, microbenchmark_command
from research.native_probes.models import CommandSpec, ExperimentMode, TrialSpec
from research.native_probes.planning import counterbalanced_order, trial_id
from research.native_probes.preflight import collect_environment, write_manifest
from research.native_probes.references import _safe_extract
from research.native_probes.runner import run_trial
from research.native_probes.validation import build_unvalidated_matrix

REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMAS = REPOSITORY / "research/native_probes/schemas"
MATRICES = REPOSITORY / "research/native_probes/matrices"


def test_preflight_is_explicit_about_unavailable_accelerators() -> None:
    manifest = collect_environment("test-control", REPOSITORY)

    _validate(manifest, "environment.schema.json")
    assert manifest["source"]["revision"]
    assert manifest["modes"]["off"]["status"] == "pass"
    assert manifest["platform"]["python_executable"]
    if manifest["platform"]["system"] == "Darwin":
        assert manifest["modes"]["ebpf-semantic"]["status"] == "unsupported"


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
        {"artifact_kind": "workload_result", "metrics": {"latency_ms": 2.5}}
    )
    command = CommandSpec((sys.executable, "-c", f"print({payload!r})"), {}, 5.0)
    spec = TrialSpec(
        trial_id="control-w1-off-r00",
        configuration_id="control",
        workload_id="w1-eager",
        mode=ExperimentMode.OFF,
        repetition=0,
        command=command,
    )

    result = run_trial(spec, tmp_path)

    _validate(result, "trial.schema.json")
    assert result["status"] == "pass"
    assert result["metrics"]["latency_ms"] == 2.5
    assert result["artifacts"][0]["sha256"]


def test_runner_rejects_secret_environment_keys(tmp_path: Path) -> None:
    command = CommandSpec((sys.executable, "-c", "pass"), {"API_TOKEN": "x"}, 5.0)
    spec = TrialSpec(
        trial_id="secret",
        configuration_id="control",
        workload_id="w1-eager",
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
    workload = Workload("w2-overlap", warmup=10, iterations=50, seed=7)
    off = microbenchmark_command(ExperimentMode.OFF, workload, tmp_path)
    public = microbenchmark_command(ExperimentMode.PUBLIC_PYTORCH, workload, tmp_path)

    assert public.argv[: len(off.argv)] == off.argv
    assert public.argv[-2] == "--torch-trace"
    with pytest.raises(ValueError, match="session-managed"):
        microbenchmark_command(ExperimentMode.PROTON, workload, tmp_path)


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
