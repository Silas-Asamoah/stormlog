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
    CriterionValue,
    SloInterval,
    SloSpec,
    client_values,
    evaluate_criteria,
    evaluate_request,
    evaluate_span,
    load_slo,
    parse_slo_flags,
    request_span,
    server_values,
    slo_attained,
    slo_from_artifact,
    slo_from_document,
    slo_record,
    span_attributes_by_request,
)
from stormlog.infer.vllm_spans import VllmSpanRecord


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


# --- judging requests and spans ------------------------------------------------


def _request(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event_type": "infer.request",
        "status": "ok",
        "ttft_ms": 100.0,
        "e2e_latency_ms": 1100.0,
        "dispatch_lag_ms": 5.0,
        "intended_at_ns": 1_000_000_000,
        "ended_at_ns": 2_105_000_000,
        "output_tokens": 11,
        "output_token_source": "server_usage",
    }
    record.update(overrides)
    return record


def _span(**seconds: float) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "gen_ai.latency.time_to_first_token": 0.08,
        "gen_ai.latency.e2e": 1.0,
        "gen_ai.latency.time_in_queue": 0.01,
    }
    names = {
        "ttft": "gen_ai.latency.time_to_first_token",
        "e2e": "gen_ai.latency.e2e",
        "queue": "gen_ai.latency.time_in_queue",
    }
    for short, value in seconds.items():
        attributes[names[short]] = value
    return attributes


def test_client_values_follow_the_documented_formulas() -> None:
    values = client_values(_request())

    assert values["client.ttft"] == CriterionValue(100.0)
    assert values["client.ttft_from_intended"] == CriterionValue(105.0)
    assert values["client.e2e"] == CriterionValue(1100.0)
    assert values["client.e2e_from_intended"] == CriterionValue(1105.0)
    # (1100 - 100) / (11 - 1)
    assert values["client.tpot"] == CriterionValue(100.0)


def test_server_values_convert_span_seconds_to_milliseconds() -> None:
    values = server_values(_span(ttft=0.25))
    assert values["server.ttft"] == CriterionValue(250.0)
    assert values["server.e2e"] == CriterionValue(1000.0)
    assert values["server.queue"] == CriterionValue(10.0)


def test_a_request_inside_every_limit_is_met_and_limits_are_inclusive() -> None:
    spec = parse_slo_flags(["ttft:100", "e2e:1100", "server.ttft:80"])
    outcome = evaluate_request(_request(), spec, span=_span())

    assert outcome.outcome == "met"
    assert outcome.met is True
    assert {key: item.outcome for key, item in outcome.criteria.items()} == {
        "client.ttft": "pass",
        "client.e2e": "pass",
        "server.ttft": "pass",
    }
    assert slo_attained(_request(), spec, span=_span()) is True


def test_one_failing_criterion_misses_even_beside_an_unknown_one() -> None:
    spec = parse_slo_flags(["ttft:50", "server.ttft:400"])
    outcome = evaluate_request(_request(), spec)  # no span: server.ttft unknown

    assert outcome.outcome == "missed"
    assert outcome.criteria["client.ttft"].outcome == "fail"
    assert outcome.criteria["server.ttft"].outcome == "unknown"
    assert outcome.criteria["server.ttft"].reason == "no_joined_span"


@pytest.mark.parametrize(
    "status",
    [
        "dropped",
        "unreachable",
        "delivery_unknown",
        "rejected",
        "error",
        "timeout",
        "cancelled",
    ],
)
def test_a_request_that_did_not_succeed_is_missed(status: str) -> None:
    spec = parse_slo_flags(["e2e:5000"])
    outcome = evaluate_request(_request(status=status), spec)

    assert outcome.outcome == "missed"
    assert outcome.met is False
    # The criteria are still judged on what was recorded, for diagnosis.
    assert outcome.criteria["client.e2e"].outcome == "pass"


def test_a_criterion_that_cannot_be_judged_makes_the_request_unknown() -> None:
    spec = parse_slo_flags(["ttft:500"])
    outcome = evaluate_request(_request(ttft_ms=None), spec)  # a non-streaming request

    assert outcome.outcome == "unknown"
    assert outcome.met is None
    assert outcome.criteria["client.ttft"].reason == "no_client_ttft"
    assert slo_attained(_request(ttft_ms=None), spec) is None


def test_server_criteria_never_read_client_values() -> None:
    spec = parse_slo_flags(["ttft:500", "server.ttft:400"])
    slow_server = _span(ttft=0.6)

    outcome = evaluate_request(_request(ttft_ms=100.0), spec, span=slow_server)

    assert outcome.criteria["client.ttft"].outcome == "pass"
    assert outcome.criteria["server.ttft"].outcome == "fail"
    assert outcome.criteria["server.ttft"].value_ms == pytest.approx(600.0)
    assert outcome.outcome == "missed"


def test_tpot_needs_server_reported_output_tokens() -> None:
    spec = parse_slo_flags(["tpot:500"])
    locally_counted = _request(output_token_source="tiktoken", output_token_exact=True)

    outcome = evaluate_request(locally_counted, spec)

    assert outcome.outcome == "unknown"
    assert outcome.criteria["client.tpot"].reason == "output_tokens_not_server_reported"


def test_tpot_with_one_output_token_is_not_applicable_and_passes() -> None:
    spec = parse_slo_flags(["tpot:1"])
    outcome = evaluate_request(_request(output_tokens=1), spec)

    assert outcome.criteria["client.tpot"].outcome == "not_applicable"
    assert outcome.outcome == "met"


def test_aggregate_only_criteria_are_unknown_per_request() -> None:
    spec = parse_slo_flags(["server.itl:50"])
    outcome = evaluate_request(_request(), spec, span=_span())

    assert outcome.criteria["server.itl"].reason == "aggregate_only"
    assert outcome.outcome == "unknown"


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"gen_ai.latency.time_to_first_token": "0.1"},
        {"gen_ai.latency.time_to_first_token": True},
        {"gen_ai.latency.time_to_first_token": float("nan")},
    ],
)
def test_a_span_without_a_usable_attribute_leaves_the_criterion_unknown(
    attributes: dict[str, Any],
) -> None:
    spec = parse_slo_flags(["server.ttft:400"])
    outcome = evaluate_request(_request(), spec, span=attributes)
    assert outcome.criteria["server.ttft"].outcome == "unknown"
    assert outcome.outcome == "unknown"


def test_a_span_is_judged_on_server_criteria_without_claiming_success() -> None:
    spec = parse_slo_flags(["ttft:500", "server.ttft:400", "server.queue:50"])

    within = evaluate_span(_span(), spec)
    beyond = evaluate_span(_span(ttft=0.5), spec)

    # vLLM emits identical spans for STOP, LENGTH, ABORT, ERROR and IGNORED
    # finishes, so a span never shows that the request succeeded.
    assert within.service_success == "unverified"
    assert within.outcome == "unknown"  # the client criterion is unknown on a span
    assert within.criteria["client.ttft"].reason == "client_boundary"
    assert within.criteria["server.ttft"].outcome == "pass"
    assert beyond.outcome == "criteria_missed"


def test_a_server_only_policy_on_a_span_can_meet_its_criteria() -> None:
    spec = parse_slo_flags(["server.ttft:400", "server.e2e:1000"])
    outcome = evaluate_span(_span(), spec)
    assert outcome.outcome == "criteria_met"
    assert set(outcome.criteria) == {"server.ttft", "server.e2e"}


def test_evaluate_criteria_accepts_plain_numbers() -> None:
    spec = parse_slo_flags(["ttft:100", "server.e2e:900"])
    outcome = evaluate_criteria({"client.ttft": 100.0, "server.e2e": None}, spec)

    assert outcome.outcome == "unknown"
    assert outcome.criteria["client.ttft"].outcome == "pass"
    assert outcome.criteria["server.e2e"].reason == "no_value"
    infinite = evaluate_criteria({"client.ttft": float("inf")}, spec)
    assert infinite.criteria["client.ttft"].reason == "non_finite_value"


# --- spans joined to requests ------------------------------------------------


def _span_record(
    x_request_id: str, span_id: str, ttft_seconds: float
) -> dict[str, Any]:
    return VllmSpanRecord(
        session_id="s1",
        run_id="run-1",
        source="otlp_http_receiver",
        name="llm_request",
        clock_domain="gpu-box/unix_epoch_ns",
        trace_id="a" * 32,
        span_id=span_id,
        start_unix_ns=0,
        end_unix_ns=1,
        attributes={
            "gen_ai.request.id": f"chatcmpl-{x_request_id}",
            "gen_ai.latency.time_to_first_token": ttft_seconds,
            "gen_ai.latency.e2e": 1.0,
            "gen_ai.latency.time_in_queue": 0.01,
        },
        request_id=x_request_id,
    ).to_record()


def _measured(index: int) -> dict[str, Any]:
    return _request(
        phase="measured",
        case_id="c1",
        session_id="s1",
        x_request_id=f"stormlog-run-{index}",
    )


def test_joined_spans_judge_server_criteria_and_quarantine_conflicts() -> None:
    first, second = _measured(0), _measured(1)
    conflicting = _span_record("stormlog-run-1", "2" * 16, 0.2)
    conflicting["attributes"]["gen_ai.latency.time_to_first_token"] = 0.9
    records = [
        {"event_type": "infer.session", "session_id": "s1"},
        first,
        second,
        _span_record("stormlog-run-0", "1" * 16, 0.2),
        _span_record("stormlog-run-1", "2" * 16, 0.2),
        conflicting,
    ]
    spans = span_attributes_by_request(records)
    spec = parse_slo_flags(["server.ttft:400"])

    assert spans.quarantined == {"stormlog-run-1": "conflicting_spans"}
    span, reason = request_span(first, spans)
    assert evaluate_request(first, spec, span=span, missing_span_reason=reason).met
    span, reason = request_span(second, spans)
    outcome = evaluate_request(second, spec, span=span, missing_span_reason=reason)
    assert outcome.outcome == "unknown"
    assert outcome.criteria["server.ttft"].reason == "conflicting_spans"


def test_a_request_without_a_span_says_so() -> None:
    records = [{"event_type": "infer.session", "session_id": "s1"}, _measured(0)]
    span, reason = request_span(_measured(0), span_attributes_by_request(records))
    assert (span, reason) == (None, "no_joined_span")


def test_an_unreadable_span_file_is_invalid_input(tmp_path: Path) -> None:
    records = [{"event_type": "infer.session", "session_id": "s1"}, _measured(0)]
    with pytest.raises(InferInputError, match="vLLM spans"):
        span_attributes_by_request(records, [tmp_path / "absent.json"])
