"""Whether two runs measured the same thing."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.compatibility import (
    REQUIRED_V1,
    RunField,
    classify,
    compatible,
    run_fields,
)
from stormlog.infer.config_classes import (
    IDENTITY,
    LABEL,
    LAUNCH,
    OBSERVATION,
    UNCLASSIFIED,
    config_class,
    field_class,
)
from stormlog.infer.server_privacy import redacted


def _field(value: Any, provenance: str = "observed") -> RunField:
    return RunField(value, "test", provenance)


def _run(**overrides: Any) -> dict[str, RunField]:
    fields = {
        "model.weights_digest": _field("w" * 64),
        "engine.version": _field("0.30.0", "reported"),
        "gpu.name": _field("NVIDIA A30"),
        "gpu.driver_version": _field("580.82.07"),
        "gpu.uuids": _field(["GPU-aaaa"]),
        "workload.spec_digest": _field("s" * 64),
        "workload.realization_digest": _field("r1"),
        "vllm_config/scheduler_config/max_num_seqs": _field(256, "reported"),
        "vllm_config/parallel_config/master_port": _field(29501, "reported"),
        "vllm_config/model_config/hf_token": _field(
            redacted("/model_config/hf_token"), "reported"
        ),
        "observer.vllm_metrics": _field(None),
        "experiment.arm": _field("control"),
    }
    for name, value in overrides.items():
        fields[name.replace("__", "/")] = value
    return fields


def test_identical_runs_are_compatible() -> None:
    result = compatible(_run(), _run())
    assert result.status == "compatible"
    assert result.blocking == () and result.unverified == ()


def test_an_identity_difference_is_incompatible_unless_allowed() -> None:
    other = _run(vllm_config__scheduler_config__max_num_seqs=_field(64, "reported"))
    result = compatible(_run(), other)
    assert result.status == "incompatible"
    (blocking,) = result.blocking
    assert blocking.name == "vllm_config/scheduler_config/max_num_seqs"
    assert (blocking.a, blocking.b, blocking.reason) == (256, 64, "differs")

    by_alias = compatible(_run(), other, allowed=["engine.max_num_seqs"])
    by_pointer = compatible(_run(), other, allowed=["/scheduler_config"])
    assert by_alias.status == by_pointer.status == "compatible"
    assert [item.name for item in by_pointer.allowed] == [blocking.name]


def test_launch_differences_are_covariates() -> None:
    other = _run(
        vllm_config__parallel_config__master_port=_field(29600, "reported"),
        **{"gpu.uuids": _field(["GPU-bbbb"])},
        **{"workload.realization_digest": _field("r2")},
    )
    result = compatible(_run(), other)
    assert result.status == "compatible"
    assert {item.name for item in result.covariates} == {
        "vllm_config/parallel_config/master_port",
        "gpu.uuids",
        "workload.realization_digest",
    }


def test_a_different_workload_spec_is_incompatible() -> None:
    other = _run(**{"workload.spec_digest": _field("t" * 64)})
    assert compatible(_run(), other).status == "incompatible"


def test_an_unclassified_configuration_leaf_blocks() -> None:
    first = _run(vllm_config__new_section__knob=_field(1, "reported"))
    second = _run(vllm_config__new_section__knob=_field(2, "reported"))
    result = compatible(first, second)
    assert result.status == "incompatible"
    assert [(item.name, item.reason) for item in result.blocking] == [
        ("vllm_config/new_section/knob", "unclassified")
    ]


def test_observers_may_differ_only_when_measuring_their_cost() -> None:
    other = _run(**{"observer.vllm_metrics": _field("http://127.0.0.1:8000/metrics")})
    assert compatible(_run(), _run()).status == "compatible"
    first = _run(**{"observer.vllm_metrics": _field("off")})
    assert compatible(first, other, mode="config").status == "incompatible"
    for mode in ("overhead", "incremental"):
        result = compatible(first, other, mode=mode)
        assert result.status == "compatible"
        assert [item.name for item in result.observation] == ["observer.vllm_metrics"]


def test_labels_are_ignored() -> None:
    other = _run(**{"experiment.arm": _field("candidate")})
    assert compatible(_run(), other).status == "compatible"


def test_a_required_field_unknown_on_either_or_both_sides_is_unverified() -> None:
    missing = _run()
    del missing["engine.version"]
    one = compatible(_run(), missing)
    both = compatible(missing, missing)
    assert one.status == both.status == "unverified"
    assert [item.name for item in both.unverified] == ["engine.version"]


def test_two_redacted_values_are_never_equal() -> None:
    result = compatible(_run(), _run())
    assert "vllm_config/model_config/hf_token" in result.unknown
    required = (*REQUIRED_V1, "vllm_config/model_config/hf_token")
    assert compatible(_run(), _run(), required=required).status == "unverified"


def test_inferred_and_declared_values_never_verify_a_required_field() -> None:
    inferred = _run(**{"model.weights_digest": _field("w" * 64, "inferred")})
    declared = _run(**{"model.weights_digest": _field("w" * 64, "declared")})
    assert compatible(inferred, inferred).status == "unverified"
    assert compatible(declared, declared).status == "unverified"


def test_inferred_values_that_disagree_are_shown() -> None:
    first = _run(**{"model.weights_digest": _field("a" * 64, "inferred")})
    second = _run(**{"model.weights_digest": _field("b" * 64, "inferred")})
    result = compatible(first, second)
    assert result.status == "unverified"
    reasons = {item.name: item.reason for item in result.unverified}
    assert reasons["model.weights_digest"] == "differs_unverified"


def test_without_a_vllm_config_a_run_is_unverified() -> None:
    bare = {
        name: item
        for name, item in _run().items()
        if not name.startswith("vllm_config/")
    }
    result = compatible(_run(), bare)
    assert result.status == "unverified"
    assert "vllm_config" in [item.name for item in result.unverified]


@pytest.mark.parametrize(
    ("pointer", "kind"),
    [
        ("/scheduler_config/max_num_seqs", IDENTITY),
        ("/parallel_config/master_port", LAUNCH),
        ("/parallel_config/tensor_parallel_size", IDENTITY),
        ("/compilation_config/cache_dir", LAUNCH),
        ("/compilation_config/cudagraph_mode", IDENTITY),
        ("/observability_config/otlp_traces_endpoint", OBSERVATION),
        ("/model_config/served_model_name", LABEL),
        ("/model_config/served_model_name/0", LABEL),
        ("/instance_id", LAUNCH),
        ("/brand_new_config/x", UNCLASSIFIED),
    ],
)
def test_configuration_leaves_take_their_longest_pointers_class(
    pointer: str, kind: str
) -> None:
    assert config_class(pointer) == kind


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("model.configured", LABEL),
        ("model.configured_revision", IDENTITY),
        ("gpu.uuids", LAUNCH),
        ("gpu.power_limit_w", IDENTITY),
        ("environ.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", OBSERVATION),
        ("environ.CUDA_VISIBLE_DEVICES", LAUNCH),
        ("environ.NCCL_P2P_DISABLE", IDENTITY),
        ("host.nproc", LAUNCH),
        ("something.else", UNCLASSIFIED),
    ],
)
def test_canonical_fields_take_their_own_or_prefix_class(name: str, kind: str) -> None:
    assert field_class(name) == kind


def test_vllm_environment_variables_are_classified_by_name() -> None:
    assert classify("vllm_env/VLLM_PORT") == LAUNCH
    assert classify("vllm_env/VLLM_ATTENTION_BACKEND") == IDENTITY
    assert classify("vllm_env/VLLM_SERVER_DEV_MODE") == OBSERVATION


def _records(
    *, max_num_seqs: int = 256, evidence: str = "inferred"
) -> list[dict[str, Any]]:
    description = {
        "host": {"hostname": "box", "nproc": 32},
        "server": {
            "environ": {"CUDA_VISIBLE_DEVICES": "0", "VLLM_PORT": "8000"},
            "start_method": {"configured": "spawn"},
        },
        "gpus": {
            "driver_version": "580.82.07",
            "devices": [
                {
                    "uuid": "GPU-aaaa",
                    "server_pids": [102],
                    "settings": {"name": "NVIDIA A30", "power_limit_w": 165.0},
                },
                {"uuid": "GPU-idle", "server_pids": [], "settings": {"name": "Other"}},
            ],
        },
        "model": {"weights_digest": "w" * 64, "identity_evidence": evidence},
        "runtime": {"python": "3.12.3", "packages": {"torch": "2.9.0"}},
    }
    probe = {
        "event_type": "infer.server_probe",
        "phase": "before",
        "answers": {
            "/version": {"body": {"version": "0.30.0"}},
            "/server_info?config_format=json": {
                "body": {
                    "vllm_config": {"scheduler_config": {"max_num_seqs": max_num_seqs}},
                    "vllm_env": {"VLLM_PORT": 8000},
                    "system_env": {"packages": {"torch": "9.9.9"}},
                }
            },
        },
    }
    return [
        {"event_type": "infer.session", "config": {"system_sampler": "none"}},
        {"event_type": "infer.manifest", "role": "before", "description": description},
        probe,
        {
            "event_type": "infer.workload",
            "seed": 0,
            "workload_digest": "r",
            "spec_digest": "s",
            "measurement": {"timeout_seconds": 60.0},
        },
        {
            "event_type": "infer.manifest",
            "role": "declared",
            "fields": {"host.purpose": "baseline", "engine.version": "9.9"},
        },
    ]


def test_a_runs_fields_come_from_its_artifact() -> None:
    fields = run_fields(_records())

    assert fields["gpu.name"].value == "NVIDIA A30"
    assert fields["gpu.uuids"].value == ["GPU-aaaa"]
    assert fields["engine.version"].provenance == "reported"
    assert fields["vllm_config/scheduler_config/max_num_seqs"].value == 256
    # Observed on the host outranks what the server reported.
    assert fields["runtime.torch"].value == "2.9.0"
    # A declaration fills a gap, never an observed or reported field.
    assert fields["engine.version"].value == "0.30.0"
    assert fields["host.purpose"].provenance == "declared"
    # A digest not bound to the launch is inferred, so it is not yet evidence.
    assert fields["model.weights_digest"].provenance == "inferred"
    assert not fields["model.weights_digest"].known
    assert fields["workload.timeout_seconds"].value == 60.0


def test_runs_from_artifacts_compare_by_their_configuration() -> None:
    same = compatible(run_fields(_records()), run_fields(_records()))
    other = compatible(run_fields(_records()), run_fields(_records(max_num_seqs=64)))
    assert same.status == "unverified"
    assert [item.name for item in same.unverified] == ["model.weights_digest"]
    assert other.status == "incompatible"
    verified = run_fields(_records(evidence="pinned_commit_verified"))
    assert verified["model.weights_digest"].known
