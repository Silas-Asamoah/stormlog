"""SLO policies: the declaration, its parsers and the artifact record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.errors import InferInputError, InferUsageError
from stormlog.infer.slo import (
    CRITERIA,
    Criterion,
    SloInterval,
    SloSpec,
    load_slo,
    parse_slo_flags,
    slo_from_artifact,
    slo_from_document,
    slo_record,
)


def _document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "format": "stormlog.infer.slo",
        "version": 1,
        "name": "chat_interactive",
        "criteria": [
            {"metric": "ttft", "boundary": "client", "max_ms": 500},
            {
                "metric": "ttft",
                "boundary": "server",
                "max_ms": 400,
                "attainment_target": 0.99,
            },
        ],
        "attainment_target": 0.98,
        "population": "offered",
        "unknown_policy": "bounds",
        "interval": {"kind": "measured_window"},
    }
    document.update(overrides)
    return document


def _write(path: Path, document: Any) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_criteria_keys_name_their_boundary() -> None:
    for key, definition in CRITERIA.items():
        assert key == f"{definition.boundary}.{definition.metric}"
    assert "client.itl" not in CRITERIA
    aggregate_only = {
        key for key, definition in CRITERIA.items() if definition.per_request is None
    }
    assert aggregate_only == {"server.itl", "server.tpot"}


def test_a_document_round_trips_and_keeps_client_and_server_apart() -> None:
    spec = slo_from_document(_document())

    assert spec.name == "chat_interactive"
    assert [criterion.key for criterion in spec.criteria] == [
        "client.ttft",
        "server.ttft",
    ]
    assert spec.criterion("server.ttft") == Criterion(
        metric="ttft", boundary="server", max_ms=400.0, attainment_target=0.99
    )
    assert spec.attainment_target == 0.98
    assert slo_from_document(spec.to_record()) == spec


def test_the_digest_ignores_integral_float_spelling() -> None:
    whole = slo_from_document(_document())
    spelled = _document()
    spelled["criteria"][0]["max_ms"] = 500.0
    assert slo_from_document(spelled).digest() == whole.digest()
    changed = _document()
    changed["criteria"][0]["max_ms"] = 501
    assert slo_from_document(changed).digest() != whole.digest()


def test_a_sliding_interval_carries_seconds() -> None:
    spec = slo_from_document(_document(interval={"kind": "sliding", "seconds": 60}))
    assert spec.interval == SloInterval(kind="sliding", seconds=60.0)
    assert spec.to_record()["interval"] == {"kind": "sliding", "seconds": 60.0}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"format": "other"}, "format must be"),
        ({"version": 2}, "version must be 1"),
        ({"version": True}, "version must be 1"),
        ({"name": "Chat"}, "name must start"),
        ({"name": "x" * 65}, "name must start"),
        ({"criteria": []}, "at least one criterion"),
        ({"criteria": {}}, "criteria must be a list"),
        ({"population": "sent"}, "population must be"),
        ({"unknown_policy": "missed"}, "unknown_policy must be"),
        ({"attainment_target": 0}, "attainment_target must be"),
        ({"attainment_target": 1.5}, "attainment_target must be"),
        ({"extra": 1}, "unknown fields: extra"),
        ({"interval": {"kind": "sliding"}}, "needs seconds > 0"),
        ({"interval": {"kind": "measured_window", "seconds": 5}}, "takes no seconds"),
        ({"interval": {"kind": "hourly"}}, "interval.kind must be"),
    ],
)
def test_invalid_documents_name_the_rule_they_break(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        slo_from_document(_document(**overrides))


@pytest.mark.parametrize(
    ("criterion", "message"),
    [
        (
            {"metric": "itl", "boundary": "client", "max_ms": 50},
            "chunks are not tokens",
        ),
        ({"metric": "ttft", "boundary": "edge", "max_ms": 50}, "unknown criterion"),
        ({"metric": "ttft", "boundary": "client", "max_ms": 0}, "positive number"),
        ({"metric": "ttft", "boundary": "client", "max_ms": "50"}, "positive number"),
        ({"metric": "ttft", "boundary": "client", "max_ms": True}, "positive number"),
        (
            {"metric": "ttft", "boundary": "client", "max_ms": float("inf")},
            "positive number",
        ),
        ({"metric": "ttft", "boundary": "client", "max_ms": 5, "p": 9}, "unknown"),
        ({"metric": "ttft", "max_ms": 50}, "needs a boundary"),
    ],
)
def test_invalid_criteria_are_refused(criterion: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        slo_from_document(_document(criteria=[criterion]))


def test_a_repeated_criterion_is_refused() -> None:
    criterion = {"metric": "ttft", "boundary": "client", "max_ms": 50}
    with pytest.raises(ValueError, match="client.ttft is given more than once"):
        slo_from_document(_document(criteria=[criterion, criterion]))


def test_load_slo_reads_a_file(tmp_path: Path) -> None:
    spec = load_slo(_write(tmp_path / "slo.json", _document()))
    assert spec == slo_from_document(_document())


@pytest.mark.parametrize("content", ["{not json", "[]"])
def test_load_slo_refuses_unreadable_or_invalid_files(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "slo.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InferInputError, match="SLO policy"):
        load_slo(path)


def test_load_slo_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InferInputError, match="SLO policy"):
        load_slo(tmp_path / "absent.json")


def test_flags_default_to_the_client_boundary() -> None:
    spec = parse_slo_flags(["ttft:500", "server.ttft:400", "client.tpot:50.5"])

    assert spec == SloSpec(
        name="cli",
        criteria=(
            Criterion(metric="ttft", boundary="client", max_ms=500.0),
            Criterion(metric="ttft", boundary="server", max_ms=400.0),
            Criterion(metric="tpot", boundary="client", max_ms=50.5),
        ),
    )
    assert spec.interval == SloInterval()
    assert spec.attainment_target is None


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["ttft"], "expected KEY:MS"),
        ([":500"], "expected KEY:MS"),
        (["ttft:fast"], "is not a number"),
        (["ttft:-1"], "positive number"),
        (["itl:50"], "chunks are not tokens"),
        (["server.ttfb:50"], "unknown criterion server.ttfb"),
        (["ttft:500", "client.ttft:400"], "given more than once"),
        ([], "at least one criterion"),
    ],
)
def test_bad_flags_are_usage_errors(flags: list[str], message: str) -> None:
    with pytest.raises(InferUsageError, match=message):
        parse_slo_flags(flags)


def test_a_bad_flag_name_is_a_usage_error() -> None:
    with pytest.raises(InferUsageError, match="name must start"):
        parse_slo_flags(["ttft:500"], name="Bad Name")


def test_the_artifact_record_round_trips() -> None:
    spec = slo_from_document(_document())
    record = slo_record(spec, session_id="session-1", source="file")

    assert record["event_type"] == "infer.slo"
    assert record["digest"] == spec.digest()
    assert slo_from_artifact([{"event_type": "infer.session"}, record]) == spec


def test_an_artifact_without_a_policy_has_none() -> None:
    assert slo_from_artifact([{"event_type": "infer.request"}]) is None


def test_an_artifact_with_two_policies_is_invalid() -> None:
    record = slo_record(
        slo_from_document(_document()), session_id="session-1", source="flags"
    )
    with pytest.raises(InferInputError, match="2 infer.slo records"):
        slo_from_artifact([record, record])


def test_an_artifact_with_an_invalid_policy_is_invalid() -> None:
    record = {"event_type": "infer.slo", "slo": _document(name="Bad")}
    with pytest.raises(InferInputError, match="infer.slo record: name must start"):
        slo_from_artifact([record])
