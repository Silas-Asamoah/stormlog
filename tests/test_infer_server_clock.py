"""Server telemetry joins use the shared clock-alignment model."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.correlation_accounting import covering_alignments
from stormlog.infer.correlation_events import ClockAlignmentEvent, CorrelationContext
from stormlog.infer.server_clock import (
    AMBIGUOUS,
    CLI_ALIGNMENT_ID,
    UNCOVERED,
    Placement,
    ServerClock,
    build_server_clock,
    cli_alignment,
    client_clock_domain,
)

SERVER = "server-a/boot-s/unix_epoch_ns"
CLIENT = "client/boot-c/unix_epoch_ns"


def _alignment(
    event_id: str,
    *,
    offset_ns: int = 0,
    uncertainty_ns: int = 0,
    valid_from_ns: int | None = None,
    valid_to_ns: int | None = None,
    from_clock_domain: str = SERVER,
    run_id: str = "run-1",
) -> ClockAlignmentEvent:
    return ClockAlignmentEvent(
        context=CorrelationContext(
            run_id=run_id,
            session_id="session-probe",
            producer_id="clock-probe",
            source="clock-probe",
            clock_domain=CLIENT,
            clock_kind="wall",
            collection_mode="passive",
            provenance="observed",
        ),
        event_id=event_id,
        from_clock_domain=from_clock_domain,
        to_clock_domain=CLIENT,
        offset_ns=offset_ns,
        uncertainty_ns=uncertainty_ns,
        valid_from_ns=valid_from_ns,
        valid_to_ns=valid_to_ns,
    )


def test_covering_alignments_returns_every_match() -> None:
    before = _alignment("before", valid_to_ns=100)
    after = _alignment("after", valid_from_ns=100)
    unbounded = _alignment("unbounded")

    def cover(timestamp_ns: int, *alignments: ClockAlignmentEvent) -> list[str]:
        return [
            item.event_id
            for item in covering_alignments(
                timestamp_ns,
                from_clock_domain=SERVER,
                to_clock_domain=CLIENT,
                alignments=alignments,
            )
        ]

    assert cover(50, before, after) == ["before"]
    # Validity windows are half-open, so the boundary belongs to the next one.
    assert cover(100, before, after) == ["after"]
    assert cover(50, before, unbounded) == ["before", "unbounded"]
    assert cover(50, _alignment("other", from_clock_domain="elsewhere")) == []


def _clock(**overrides: Any) -> ServerClock | str:
    options: dict[str, Any] = {
        "run_id": "run-1",
        "server_domain": SERVER,
        "client_domain": CLIENT,
        "recorded": (),
        "offset_ns": None,
        "uncertainty_ns": None,
        **overrides,
    }
    return build_server_clock(**options)


def _place(clock: ServerClock | str, timestamp_ns: int) -> Placement:
    assert isinstance(clock, ServerClock)
    placement = clock.align(timestamp_ns)
    assert not isinstance(placement, str)
    return placement


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({}, "clock_alignment_required"),
        ({"offset_ns": 5}, "clock_uncertainty_required"),
        ({"uncertainty_ns": 5}, "clock_offset_required"),
        (
            {"uncertainty_ns": 5, "recorded": (_alignment("probe"),)},
            "clock_offset_required",
        ),
        (
            {
                "server_domain": "host/unix_epoch_ns",
                "client_domain": "host/unix_epoch_ns",
            },
            "clock_domain_unverified",
        ),
    ],
)
def test_missing_clock_evidence_is_a_reason(
    overrides: dict[str, Any], reason: str
) -> None:
    assert _clock(**overrides) == reason


def test_one_host_boot_is_one_clock_with_optional_uncertainty() -> None:
    clock = _clock(server_domain=CLIENT, uncertainty_ns=40)
    assert isinstance(clock, ServerClock)
    assert clock.evidence == "same_host"
    placed, record = _place(clock, 1_000)
    assert (placed.value_ns, placed.uncertainty_ns, record) == (1_000, 40, None)
    shifted = _clock(server_domain=CLIENT, offset_ns=5, uncertainty_ns=1)
    assert shifted == "clock_offset_on_shared_clock"


def test_operator_flags_replace_records_for_the_same_clocks() -> None:
    recorded = (_alignment("probe-1", offset_ns=1_000),)
    clock = _clock(recorded=recorded, offset_ns=7, uncertainty_ns=2)
    assert isinstance(clock, ServerClock)
    assert (clock.evidence, clock.overridden) == ("operator_supplied", ("probe-1",))
    placed, record = _place(clock, 100)
    assert (placed.value_ns, placed.uncertainty_ns) == (107, 2)
    assert record is not None and record.event_id == CLI_ALIGNMENT_ID


def test_records_are_used_per_validity_window() -> None:
    recorded = (
        _alignment("early", offset_ns=10, uncertainty_ns=1, valid_to_ns=100),
        _alignment("late", offset_ns=20, uncertainty_ns=3, valid_from_ns=100),
    )
    clock = _clock(recorded=recorded)
    assert isinstance(clock, ServerClock)
    assert clock.evidence == "artifact_record"
    early, early_record = _place(clock, 50)
    late, late_record = _place(clock, 150)
    assert early_record is not None and late_record is not None
    assert (early.value_ns, early.uncertainty_ns, early_record.event_id) == (
        60,
        1,
        "early",
    )
    assert (late.value_ns, late.uncertainty_ns, late_record.event_id) == (
        170,
        3,
        "late",
    )


def test_uncovered_and_ambiguous_timestamps_are_named() -> None:
    bounded = _clock(recorded=(_alignment("bounded", valid_to_ns=100),))
    assert isinstance(bounded, ServerClock)
    assert bounded.align(150) == UNCOVERED
    overlapping = _clock(recorded=(_alignment("a"), _alignment("b")))
    assert isinstance(overlapping, ServerClock)
    assert overlapping.align(150) == AMBIGUOUS


@pytest.mark.parametrize(
    ("offset_ns", "uncertainty_ns", "message"),
    [
        (1.5, 0, "offset_ns must be an integer"),
        (True, 0, "offset_ns must be an integer"),
        (0, -1, "uncertainty_ns must be a non-negative integer"),
    ],
)
def test_operator_values_are_validated_like_alignment_records(
    offset_ns: Any, uncertainty_ns: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _alignment("record", offset_ns=offset_ns, uncertainty_ns=uncertainty_ns)
    with pytest.raises(ValueError, match=message):
        cli_alignment("run-1", SERVER, CLIENT, offset_ns, uncertainty_ns)
    with pytest.raises(ValueError, match=message):
        _clock(offset_ns=offset_ns, uncertainty_ns=uncertainty_ns)


@pytest.mark.parametrize(
    ("artifact", "domain"),
    [
        ({"context": {"clock_domain": CLIENT}}, CLIENT),
        (
            {
                "context": {"host": "client", "clock_domain": "client/unix_epoch_ns"},
                "metadata": {"boot_id": "boot-c"},
            },
            CLIENT,
        ),
        ({"context": {"host": "client"}, "metadata": {}}, "client/unix_epoch_ns"),
        ({"context": {"clock_domain": "client/unix_epoch_ns"}}, "client/unix_epoch_ns"),
        ({"context": {}}, None),
        ({}, None),
    ],
)
def test_client_domain_names_the_boot_when_the_artifact_knows_it(
    artifact: dict[str, Any], domain: str | None
) -> None:
    assert client_clock_domain(artifact) == domain


def test_records_from_another_run_are_ignored() -> None:
    foreign = _alignment("copied", run_id="other-run", offset_ns=1_000)
    assert _clock(recorded=(foreign,)) == "clock_alignment_from_another_run"
    own = _alignment("probe", offset_ns=5)
    clock = _clock(recorded=(foreign, own))
    assert isinstance(clock, ServerClock)
    assert (clock.alignments, clock.ignored) == ((own,), ("copied",))
    placed, record = _place(clock, 100)
    assert (placed.value_ns, record) == (105, own)


def test_equal_names_without_boot_ids_join_only_with_explicit_flags() -> None:
    bootless = "win-box/unix_epoch_ns"
    same = {"server_domain": bootless, "client_domain": bootless}
    assert _clock(**same) == "clock_domain_unverified"
    assert _clock(**same, uncertainty_ns=5) == "clock_offset_required"
    clock = _clock(**same, offset_ns=20, uncertainty_ns=5)
    assert isinstance(clock, ServerClock)
    assert (clock.evidence, clock.server_domain) == (
        "operator_supplied",
        "win-box/unix_epoch_ns#server",
    )
    placed, _record = _place(clock, 100)
    assert (placed.value_ns, placed.uncertainty_ns) == (120, 5)


def test_compact_alignments_expand_generator_before_filtering():
    from stormlog.infer.correlation_codec import CorrelationRecordEncoder
    from stormlog.infer.server_clock import artifact_alignments

    own, foreign = _alignment("own"), _alignment("foreign", run_id="other")
    encoder = CorrelationRecordEncoder()
    rows = [
        row
        for event in (own, own, foreign)
        for row in encoder.encode(event.to_record())
    ]
    assert artifact_alignments(iter(rows), "run-1") == (own, foreign)
    with pytest.raises(ValueError, match="unknown context_id"):
        artifact_alignments(iter(rows[1:]), "run-1")
