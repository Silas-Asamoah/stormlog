"""Whether two runs measured the same thing."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.compatibility import RunField, classify, compatible, run_fields
from stormlog.infer.config_classes import (
    IDENTITY,
    LABEL,
    LAUNCH,
    OBSERVATION,
    UNCLASSIFIED,
    config_class,
    field_class,
)
from stormlog.infer.manifest import model_identity_record
from stormlog.infer.server_privacy import redacted


def _field(value: Any, provenance: str = "observed") -> RunField:
    return RunField(value, "test", provenance)


def _leaf(value: Any) -> RunField:
    """A /server_info leaf: null there is a setting, not missing evidence."""
    return RunField(value, "/server_info", "reported", null_is_value=True)


def _run(**overrides: Any) -> dict[str, RunField]:
    fields = {
        "model.weights_digest": _field("w" * 64),
        "engine.version": _field("0.30.0", "reported"),
        "gpu.name": _field("NVIDIA A30"),
        "gpu.driver_version": _field("580.82.07"),
        "gpu.uuids": _field(["GPU-aaaa"]),
        "workload.spec_digest": _field("s" * 64),
        "workload.realization_digest": _field("r1"),
        "scope.vllm_config": _field(True, "reported"),
        "scope.environ": _field(True),
        "vllm_config/scheduler_config/max_num_seqs": _leaf(256),
        "vllm_config/parallel_config/master_port": _leaf(29501),
        "vllm_config/model_config/quantization": _leaf(None),
        "vllm_config/model_config/hf_token": _leaf(redacted("/model_config/hf_token")),
        "environ.CUDA_VISIBLE_DEVICES": _field("0"),
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
    hidden = redacted("/load_config/model_loader_extra_config")
    run = _run(vllm_config__load_config__model_loader_extra_config=_leaf(hidden))
    result = compatible(run, run)
    assert result.status == "unverified"
    (item,) = result.unverified
    assert (item.name, item.reason) == (
        "vllm_config/load_config/model_loader_extra_config",
        "unknown",
    )


def test_a_token_is_a_credential_not_configuration() -> None:
    # Which account downloaded the weights says nothing about what ran.
    assert compatible(_run(), _run()).status == "compatible"
    other = _run(vllm_config__model_config__hf_token=_leaf(None))
    assert compatible(_run(), other).status == "compatible"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("vllm_config/model_config/quantization", "fp8"),
        ("vllm_config/model_config/dtype", "float16"),
    ],
)
def test_null_is_a_setting_and_differs_from_a_value(name: str, value: Any) -> None:
    first = _run(**{"vllm_config/model_config/dtype": _leaf("bfloat16")})
    second = _run(**{"vllm_config/model_config/dtype": _leaf("bfloat16")})
    second[name] = _leaf(value)
    result = compatible(first, second)
    assert result.status == "incompatible"
    assert [item.name for item in result.blocking] == [name]


def test_a_leaf_on_one_side_only_is_a_difference() -> None:
    # A section null in one run and set in the other: speculative decoding on.
    first = _run(vllm_config__speculative_config=_leaf(None))
    second = _run(
        vllm_config__speculative_config__method=_leaf("ngram"),
        vllm_config__speculative_config__num_speculative_tokens=_leaf(5),
    )
    result = compatible(first, second)
    assert result.status == "incompatible"
    reasons = {item.name: item.reason for item in result.blocking}
    assert reasons == {
        "vllm_config/speculative_config": "only_in_a",
        "vllm_config/speculative_config/method": "only_in_b",
        "vllm_config/speculative_config/num_speculative_tokens": "only_in_b",
    }


def test_an_environment_variable_set_on_one_side_only_is_a_difference() -> None:
    second = _run(**{"environ.CUDA_LAUNCH_BLOCKING": _field("1")})
    result = compatible(_run(), second)
    assert result.status == "incompatible"
    assert [(i.name, i.reason) for i in result.blocking] == [
        ("environ.CUDA_LAUNCH_BLOCKING", "only_in_b")
    ]


def test_a_leaf_missing_where_nothing_was_read_is_unknown_not_a_difference() -> None:
    unread = {k: v for k, v in _run().items() if k != "scope.environ"}
    second = _run(**{"environ.CUDA_LAUNCH_BLOCKING": _field("1")})
    result = compatible(unread, second)
    # Unknown on one side: an identity field that cannot be verified.
    assert result.status == "unverified"
    assert [(i.name, i.reason) for i in result.unverified] == [
        ("environ.CUDA_LAUNCH_BLOCKING", "unknown")
    ]


def test_a_one_sided_value_that_is_not_evidence_leaves_the_field_unverified() -> None:
    # Declared, not reported: the other run's configuration lacks it, but a
    # declaration cannot show the server ran with it.
    second = _run(**{"vllm_config/new_section/knob": _field(1, "declared")})
    result = compatible(_run(), second)
    assert result.status == "unverified"
    assert [(i.name, i.reason) for i in result.unverified] == [
        ("vllm_config/new_section/knob", "differs_unverified")
    ]


def test_an_unknown_launch_field_is_noted_without_blocking() -> None:
    second = _run(**{"host.nproc": _field(32)})
    result = compatible(_run(), second)
    assert result.status == "compatible"
    assert "host.nproc" in result.unknown


def test_a_declared_configuration_name_does_not_stand_for_server_info() -> None:
    bare = {
        name: item
        for name, item in _run().items()
        if not name.startswith(("vllm_config/", "scope.vllm_config"))
    }
    bare["vllm_config/scheduler_config/max_num_seqs"] = _field(256, "declared")
    result = compatible(_run(), bare)
    assert result.status == "unverified"
    assert "vllm_config" in [item.name for item in result.unverified]


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
        if not name.startswith(("vllm_config/", "scope.vllm_config"))
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
    # Performance settings block; their cache directories only place a run.
    assert classify("environ.LD_PRELOAD") == IDENTITY
    assert classify("environ.OMP_NUM_THREADS") == IDENTITY
    assert classify("environ.TRITON_CACHE_DIR") == LAUNCH
    assert classify("environ.TORCHINDUCTOR_CACHE_DIR") == LAUNCH


def _records(
    *, max_num_seqs: int = 256, evidence: str = "inferred"
) -> list[dict[str, Any]]:
    description = {
        "host": {"hostname": "box", "nproc": 32, "boot_id": "boot-1"},
        "server": {
            "pid": 100,
            "start_ticks": 500,
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
        "model": {
            "weights_digest": "w" * 64,
            "identity_evidence": evidence,
            "resolved_snapshot": "c" * 40,
            "chat_template_digest": "t" * 64,
        },
        "runtime": {"python": "3.12.3", "packages": {"torch": "2.9.0"}},
    }
    probe = {
        "event_type": "infer.server_probe",
        "phase": "before",
        "answers": {
            "/version": {"body": {"version": "0.30.0"}},
            "/server_info?config_format=json": {
                "body": {
                    "vllm_config": {
                        "scheduler_config": {"max_num_seqs": max_num_seqs},
                        "model_config": {"quantization": None},
                    },
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


def test_the_before_probe_that_answered_server_info_gives_the_configuration() -> None:
    # A runner probes /server_info once before measuring, and the workload
    # probes only the basic routes, so no collector runs beside it: the
    # configuration comes from whichever before probe has it.
    records = _records()
    full = next(r for r in records if r.get("event_type") == "infer.server_probe")
    basic = {
        "event_type": "infer.server_probe",
        "phase": "before",
        "answers": {"/version": {"body": {"version": "0.30.0"}}},
    }
    records.remove(full)
    records.insert(1, basic)
    records.append({**full, "origin": "runner"})
    fields = run_fields(records)
    assert fields["scope.vllm_config"].value is True
    assert fields["vllm_config/scheduler_config/max_num_seqs"].value == 256


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
    # A null setting is a value, and the scopes the run read are marked.
    assert fields["vllm_config/model_config/quantization"].known
    assert fields["scope.vllm_config"].known and fields["scope.environ"].known


def _bound(
    *, pid: int = 100, start_ticks: int = 500, evidence: str = "pinned_commit_verified"
) -> dict[str, Any]:
    """The runner's record: weights verified before it launched this server."""
    return model_identity_record(
        {
            "weights_digest": "w" * 64,
            "resolved_snapshot": "c" * 40,
            "identity_evidence": evidence,
        },
        session_id="s",
        run_id="r",
        server={"pid": pid, "start_ticks": start_ticks},
        boot_id="boot-1",
    )


def test_runs_from_artifacts_compare_by_their_configuration() -> None:
    same = compatible(run_fields(_records()), run_fields(_records()))
    other = compatible(run_fields(_records()), run_fields(_records(max_num_seqs=64)))
    assert same.status == "unverified"
    assert [item.name for item in same.unverified] == [
        "model.resolved_snapshot",
        "model.weights_digest",
        "server.chat_template_digest",
    ]
    assert other.status == "incompatible"
    bound = [*_records(), _bound()]
    assert compatible(run_fields(bound), run_fields(bound)).status == "compatible"


def test_a_description_alone_never_verifies_its_weights() -> None:
    # describe-server never claims verified evidence; an edited description
    # that does is still a digest taken after launch.
    edited = run_fields(_records(evidence="pinned_commit_verified"))
    assert not edited["model.weights_digest"].known


@pytest.mark.parametrize(
    "record",
    [
        _bound(pid=101),
        _bound(start_ticks=999),
        _bound(evidence="inferred"),
    ],
    ids=["another_pid", "a_restarted_server", "not_verified"],
)
def test_the_runners_record_verifies_only_the_server_it_launched(
    record: dict[str, Any]
) -> None:
    fields = run_fields([*_records(), record])
    assert not fields["model.weights_digest"].known


def test_weights_the_runner_verified_but_the_description_disagrees_with_are_unknown() -> (
    None
):
    records = _records()
    records[1]["description"]["model"]["weights_digest"] = "x" * 64
    fields = run_fields([*records, _bound()])
    assert not fields["model.weights_digest"].known
