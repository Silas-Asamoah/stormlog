"""Compact transport preserves semantic events and fails closed on bad references."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from stormlog.infer.correlation_codec import (
    CorrelationRecordEncoder,
    expand_inference_records,
)
from stormlog.infer.correlation_events import (
    CorrelationContext,
    EntityRef,
    IterationEvent,
    load_inference_artifact,
)
from stormlog.infer.events import JsonlEventWriter


def _event():
    return IterationEvent(
        context=CorrelationContext(
            run_id="run",
            session_id="session",
            producer_id="engine",
            source="test",
            clock_domain="host/mono",
            clock_kind="monotonic",
            collection_mode="imported",
            provenance="observed",
        ),
        event_id="iteration",
        iteration_ref=EntityRef("engine", "1"),
        start_ns=10,
        end_ns=20,
    )


def test_shared_context_roundtrip_and_schema(tmp_path):
    event = _event()
    other = replace(event, event_id="second")
    path = tmp_path / "events.jsonl"
    with JsonlEventWriter(path) as writer:
        writer.append(event.to_record())
        writer.append(other.to_record())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["event_type"] == "infer.context"
    assert rows[1]["context_id"] == rows[2]["context_id"]
    assert load_inference_artifact(path) == [event, other]
    schema = json.loads(
        Path("docs/schemas/inference_correlation_v4.schema.json").read_text()
    )
    validator = Draft202012Validator(schema)
    for row in rows:
        validator.validate(row)
    expanded = expand_inference_records(rows)
    assert expand_inference_records(expanded) == expanded
    expanded[0]["context"]["run_id"] = "changed"
    assert expanded[1]["context"]["run_id"] == "run"


def test_failed_serialization_does_not_emit_or_register(tmp_path):
    event = _event()
    path = tmp_path / "events.jsonl"
    with JsonlEventWriter(path) as writer:
        with pytest.raises(TypeError):
            writer.append(replace(event, metadata={"bad": object()}).to_record())
        assert path.read_bytes() == b""
        writer.append(event.to_record())
    assert load_inference_artifact(path) == [event]
    assert len(path.read_text().splitlines()) == 2


def test_unknown_context_physical_line(tmp_path):
    rows = CorrelationRecordEncoder().encode(_event().to_record())
    path = tmp_path / "events.jsonl"
    path.write_text("\n\n" + json.dumps(rows[1]) + "\n")
    with pytest.raises(ValueError, match="line 3: unknown context_id"):
        load_inference_artifact(path)


def test_aliases_and_conflicts():
    rows = CorrelationRecordEncoder().encode(_event().to_record())
    alias = dict(rows[0], context_id="alias")
    event_alias = dict(rows[1], context_id="alias")
    expanded = expand_inference_records([*rows, alias, event_alias])
    assert expanded[0] == expanded[1]
    conflicting = dict(rows[0], context=dict(rows[0]["context"], host="different"))
    with pytest.raises(ValueError, match="conflicting context"):
        expand_inference_records([*rows, conflicting])


def test_hash_collision(monkeypatch):
    monkeypatch.setattr(
        "stormlog.infer.correlation_codec._generated_id", lambda _: "same"
    )
    encoder = CorrelationRecordEncoder()
    event = _event()
    encoder.encode(event.to_record())
    with pytest.raises(ValueError, match="conflicting context"):
        encoder.encode(
            replace(event, context=replace(event.context, host="other")).to_record()
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("producer_id", "other"),
        ("host", "other"),
        ("pid", 2),
        ("device_uuid", "GPU-other"),
        ("rank", 2),
        ("clock_domain", "other"),
        ("source_version", "other"),
        ("provenance", "reported"),
        ("run_id", "other"),
        ("session_id", "other"),
    ],
)
def test_full_context_identity(field, value):
    event = _event()
    encoder = CorrelationRecordEncoder()
    encoder.encode(event.to_record())
    changed = replace(event, context=replace(event.context, **{field: value}))
    assert len(encoder.encode(changed.to_record())) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"context_id": ""},
        {"context_id": None},
        {"context_id": 2},
        {"schema_version": True},
        {"schema_version": 3},
        {"schema_version": 5},
        {"event_type": "infer.unknown"},
        {"context": {}},
    ],
)
def test_invalid_transport(changes):
    rows = CorrelationRecordEncoder().encode(_event().to_record())
    rows[1].update(changes)
    with pytest.raises((ValueError, TypeError)):
        expand_inference_records(rows)


def test_missing_id_and_definition_fields():
    rows = CorrelationRecordEncoder().encode(_event().to_record())
    del rows[1]["context_id"]
    with pytest.raises(ValueError, match="context_id"):
        expand_inference_records(rows)
    rows[0]["event_id"] = "forbidden"
    with pytest.raises(ValueError, match="exactly"):
        expand_inference_records(rows[:1])


def test_optional_nulls_and_append_alias():
    from stormlog.infer.correlation_codec import CorrelationRecordDecoder

    event = _event()
    rows = CorrelationRecordEncoder().encode(event.to_record())
    definition = rows[0]
    definition["context"] = {
        k: v for k, v in definition["context"].items() if v is not None
    }
    definition["context_id"] = "first"
    decoder = CorrelationRecordDecoder()
    decoder.decode(definition)
    decoder.decode(dict(definition, context_id="second"))
    encoder = CorrelationRecordEncoder(decoder.registry)
    [encoded] = encoder.encode(event.to_record())
    assert encoded["context_id"] == "first"


def test_mixed_order_and_registry_lifetime(tmp_path):
    event = _event()
    embedded = event.to_record()
    legacy = {"schema_version": 1, "event_type": "infer.request", "custom": 42}
    compact = CorrelationRecordEncoder().encode(embedded)
    rows = [legacy, embedded, *compact]
    assert expand_inference_records(rows) == [legacy, embedded, embedded]
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("\n".join(map(json.dumps, compact)))
    b.write_text(json.dumps(compact[1]))
    assert load_inference_artifact(a) == [event]
    with pytest.raises(ValueError, match="unknown context_id"):
        load_inference_artifact(b)


def test_writer_reentry_and_closed(tmp_path):
    event = _event()
    writer = JsonlEventWriter(tmp_path / "events")
    for _ in range(2):
        with writer:
            writer.append(event.to_record())
        assert load_inference_artifact(writer.path) == [event]
        assert len(writer.path.read_text().splitlines()) == 2
    with pytest.raises(RuntimeError, match="not open"):
        writer.append(event.to_record())


def test_input_immutability_and_old_parser():
    import copy

    from stormlog.infer.correlation_events import parse_inference_record

    embedded = _event().to_record()
    before = copy.deepcopy(embedded)
    rows = CorrelationRecordEncoder().encode(embedded)
    raw_before = copy.deepcopy(rows)
    assert expand_inference_records(rows) == [embedded]
    assert embedded == before
    assert rows == raw_before
    for row in rows:
        with pytest.raises(ValueError, match="v4 requires"):
            parse_inference_record(row)


@pytest.mark.parametrize(
    "intervals", [None, [], [[-1, 1]], [[0, 8], [3, 2]], [[0, 11]]]
)
def test_invalid_busy_intervals(intervals):
    from stormlog.infer.correlation_events import ActivityReferenceEvent

    event = ActivityReferenceEvent(
        context=_event().context,
        event_id="activity",
        activity_ref=EntityRef("engine", "a"),
        activity_kind="kernel",
        activity_domain="gpu",
        attribution_status="unresolved",
        start_ns=10,
        end_ns=20,
        metadata={"intervals": [[0, 2], [5, 3]]},
    )
    rows = CorrelationRecordEncoder().encode(event.to_record())
    rows[1]["metadata"] = {"intervals": intervals}
    with pytest.raises(ValueError, match="intervals"):
        expand_inference_records(rows)


def _all_events():
    from stormlog.infer.correlation_events import (
        ActivityReferenceEvent,
        ArtifactIdentityEvent,
        CapabilityEvent,
        ClockAlignmentEvent,
        MembershipEvent,
        RequestEvent,
        StageEvent,
    )

    base = {"context": _event().context, "event_id": "event"}
    ref = EntityRef("engine", "1")
    activity = ActivityReferenceEvent(
        **base,
        activity_ref=ref,
        activity_kind="kernel",
        activity_domain="gpu",
        attribution_status="unresolved",
        start_ns=10,
        end_ns=20,
    )
    return [
        _event(),
        ArtifactIdentityEvent(**base, artifact_kind="inference_jsonl", created_at_ns=0),
        CapabilityEvent(**base, component="test", available=False),
        ClockAlignmentEvent(
            **base,
            from_clock_domain="a",
            to_clock_domain="b",
            offset_ns=-1,
            uncertainty_ns=0,
        ),
        MembershipEvent(**base, request_ref=ref, iteration_ref=ref, role="decode"),
        RequestEvent(**base, request_ref=ref),
        StageEvent(**base, stage_ref=ref, name="stage", iteration_ref=ref),
        activity,
        replace(activity, metadata={"intervals": [[0, 2], [5, 3]]}),
    ]


@pytest.mark.parametrize(
    "event", _all_events(), ids=lambda e: e.EVENT_TYPE + str(e.schema_version)
)
def test_all_event_types_schema_and_typed_roundtrip(event, tmp_path):
    rows = CorrelationRecordEncoder().encode(event.to_record())
    schema = json.loads(
        Path("docs/schemas/inference_correlation_v4.schema.json").read_text()
    )
    validator = Draft202012Validator(schema)
    for row in rows:
        validator.validate(row)
    assert expand_inference_records(rows) == [event.to_record()]
    path = tmp_path / "artifact"
    path.write_text("\n".join(map(json.dumps, rows)))
    assert load_inference_artifact(path) == [event]
    for invalid in (
        dict(rows[1], context=rows[0]["context"]),
        {k: v for k, v in rows[1].items() if k != "context_id"},
    ):
        assert not validator.is_valid(invalid)


def test_compact_mixed_and_alias_accounting():
    from stormlog.infer.correlation_accounting import (
        account_gpu_time,
        resolve_inference_events,
    )
    from stormlog.infer.correlation_events import parse_inference_record

    activities = _all_events()[-2:]
    activities = [
        replace(
            a,
            context=replace(a.context, device_uuid="GPU-test"),
            event_id=str(i),
            activity_ref=EntityRef("engine", str(i)),
        )
        for i, a in enumerate(activities)
    ]
    encoder = CorrelationRecordEncoder()
    rows = [r for e in activities for r in encoder.encode(e.to_record())]
    alias = dict(rows[0], context_id="alias")
    duplicate = dict(rows[1], context_id="alias")
    mixed = [activities[0].to_record(), *rows, alias, duplicate]
    restored = [parse_inference_record(r) for r in expand_inference_records(mixed)]
    original_graph = resolve_inference_events(activities)
    assert resolve_inference_events(restored) == original_graph
    accounting = account_gpu_time(original_graph)
    assert account_gpu_time(resolve_inference_events(restored)) == accounting
    assert next(iter(accounting.device_totals.values())).busy_ns == 10


def test_embedded_null_intervals_stays_legacy_but_compact_rejects_it():
    from stormlog.infer.correlation_events import parse_inference_record

    embedded = _all_events()[-1].to_record()
    embedded["metadata"]["intervals"] = None
    original = parse_inference_record(embedded)
    assert CorrelationRecordEncoder().encode(embedded) == [embedded]
    assert expand_inference_records([embedded]) == [embedded]
    assert original.to_record() == embedded


def test_writer_retains_unrelated_rows_and_v1_fields(tmp_path):
    rows = [
        {"schema_version": 1, "event_type": "infer.request", "extra": {"value": 42}},
        {"event_type": "custom", "payload": [1, 2]},
    ]
    with JsonlEventWriter(tmp_path / "events") as writer:
        for row in rows:
            writer.append(row)
    assert [json.loads(line) for line in writer.path.read_text().splitlines()] == rows


def test_typed_legacy_failure_keeps_physical_lines_after_definitions(tmp_path):
    definition = CorrelationRecordEncoder().encode(_event().to_record())[0]
    path = tmp_path / "events"
    path.write_text(
        json.dumps(definition)
        + "\n\n"
        + json.dumps({"schema_version": 2, "event_type": "infer.iteration"})
    )
    with pytest.raises(ValueError, match="line 3: context must be an object"):
        load_inference_artifact(path)


@pytest.mark.parametrize("first", ["correlation_codec", "correlation_events"])
def test_modules_import_independently(first):
    import subprocess
    import sys

    other = (
        "correlation_events" if first == "correlation_codec" else "correlation_codec"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import stormlog.infer.{first}; import stormlog.infer.{other}",
        ],
        check=True,
    )


@pytest.mark.parametrize("detail", ["launch", "kernel"])
def test_offline_comparison_scenario_uses_production_import(tmp_path, detail):
    import argparse

    from examples.scenarios.context_compaction_scenario import compare
    from tests.test_infer_trace_kineto import _trace_document

    trace = tmp_path / "fixture.json"
    trace.write_text(json.dumps(_trace_document()))
    output = tmp_path / "results"
    results = compare(
        argparse.Namespace(
            trace=trace, output_dir=output, detail=detail, device_uuid=["0=GPU-FIXTURE"]
        )
    )
    assert all(results["invariants"].values())
    assert results["saved_bytes"] > 0
    assert (
        results["compact"]["physical_lines"]
        == results["semantic_events"] + results["context_definitions"]
    )
    assert results["device_clock_totals"]
    with pytest.raises(FileExistsError):
        compare(
            argparse.Namespace(
                trace=trace, output_dir=output, detail=detail, device_uuid=[]
            )
        )


def test_forward_context_reference_is_not_buffered(tmp_path):
    rows = CorrelationRecordEncoder().encode(_event().to_record())
    path = tmp_path / "events"
    path.write_text(json.dumps(rows[1]) + "\n" + json.dumps(rows[0]) + "\n")
    with pytest.raises(
        ValueError, match="line 1: unknown context_id: " + rows[0]["context_id"]
    ):
        load_inference_artifact(path)


@pytest.mark.parametrize("context", [{}, {"run_id": None}, {"unexpected": "value"}])
def test_invalid_definition_context_agrees_with_schema(context):
    rows = CorrelationRecordEncoder().encode(_event().to_record())
    rows[0]["context"] = context
    schema = json.loads(
        Path("docs/schemas/inference_correlation_v4.schema.json").read_text()
    )
    assert not Draft202012Validator(schema).is_valid(rows[0])
    with pytest.raises((TypeError, ValueError)):
        expand_inference_records(rows)
