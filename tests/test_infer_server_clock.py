"""Server telemetry joins use the shared clock-alignment model."""

from __future__ import annotations

from stormlog.infer.correlation_accounting import covering_alignments
from stormlog.infer.correlation_events import ClockAlignmentEvent, CorrelationContext

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
) -> ClockAlignmentEvent:
    return ClockAlignmentEvent(
        context=CorrelationContext(
            run_id="run-1",
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
