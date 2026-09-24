"""Tests for the repository-local native probe experiment framework."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from research.native_probes.analysis import analyze_trials
from research.native_probes.models import CommandSpec, ExperimentMode
from research.native_probes.planning import counterbalanced_order, trial_id
from research.native_probes.preflight import collect_environment, write_manifest

REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMAS = REPOSITORY / "research/native_probes/schemas"


def test_preflight_is_explicit_about_unavailable_accelerators() -> None:
    manifest = collect_environment("test-control", REPOSITORY)

    _validate(manifest, "environment.schema.json")
    assert manifest["source"]["revision"]
    assert manifest["modes"]["off"]["status"] == "pass"
    assert manifest["platform"]["python_executable"]
    if manifest["platform"]["system"] == "Darwin":
        assert manifest["modes"]["ebpf-semantic"]["status"] == "unsupported"


def test_manifest_writer_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    checksum = write_manifest(output, {"value": 1})

    assert len(checksum) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(FileExistsError):
        write_manifest(output, {"value": 2})


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


def _validate(instance: object, schema_name: str) -> None:
    schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(instance)


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
