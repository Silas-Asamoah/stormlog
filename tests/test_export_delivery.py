"""Settling batches exactly once, and the breaker's transitions."""

from typing import Any

import pytest

from stormlog._export.delivery import (
    BREAKER_CLOSED,
    BREAKER_OPEN,
    FIRST_FAILURE,
    FIRST_SUCCESS,
    MAX_TRANSITIONS,
    REFUSED_AFTER_AMBIGUOUS,
    REJECTED_AFTER_AMBIGUOUS,
    SHUTDOWN,
    SHUTDOWN_IN_FLIGHT,
    BatchHistory,
    Breaker,
    DeliveryLedger,
    RetryPolicy,
    Settlement,
)
from stormlog._export.otlp_encoding import ExportResult
from stormlog._export.otlp_http import (
    AMBIGUOUS,
    CONFIRMED,
    CONNECT_REFUSED,
    HTTP_4XX,
    NOT_SENT,
    REFUSED,
    RESET_AFTER_SEND,
    THROTTLED,
    TIMEOUT_AFTER_SEND,
    UNREADABLE_RESPONSE,
    Transmission,
)

RESET = Transmission(AMBIGUOUS, RESET_AFTER_SEND, retryable=True)
LOST_ACK = Transmission(AMBIGUOUS, TIMEOUT_AFTER_SEND, retryable=True)
BAD_REQUEST = Transmission(REFUSED, HTTP_4XX, 400)
REFUSED_CONNECT = Transmission(NOT_SENT, CONNECT_REFUSED, retryable=True)


def _ok(rejected: int = 0) -> Transmission:
    return Transmission(CONFIRMED, None, 200, result=ExportResult(rejected))


def _settle(*transmissions: Transmission, spans: int = 10) -> Settlement:
    history = BatchHistory(spans)
    for transmission in transmissions:
        history.record(transmission)
    return history.settle()


def _bounds(settlement: Settlement, unique: int, raw: int) -> bool:
    """The collector-side bounds #221 checks against what was received."""
    unknown = sum(n for _, n in settlement.unknown)
    return (
        settlement.exported <= unique <= settlement.exported + unknown
        and raw - unique <= settlement.extra_copies
    )


# Astra's five R1 histories, n = 10: what the collector really stored, as
# (unique, raw), and what the exporter can know.
@pytest.mark.parametrize(
    ("history", "expected", "copies", "received"),
    [
        (
            [RESET, BAD_REQUEST],
            Settlement(unknown=((REFUSED_AFTER_AMBIGUOUS, 10),)),
            0,
            (10, 10),
        ),
        (
            [RESET, _ok(rejected=5)],
            Settlement(exported=5, unknown=((REJECTED_AFTER_AMBIGUOUS, 5),)),
            10,
            (10, 15),
        ),
        ([LOST_ACK] * 4 + [_ok()], Settlement(exported=10), 40, (10, 50)),
        (
            [LOST_ACK] * 5,
            Settlement(unknown=((TIMEOUT_AFTER_SEND, 10),)),
            40,
            (10, 50),
        ),
        (
            [Transmission(AMBIGUOUS, UNREADABLE_RESPONSE, 200)],
            Settlement(unknown=((UNREADABLE_RESPONSE, 10),)),
            0,
            (5, 5),
        ),
    ],
)
def test_astras_r1_histories(
    history: list[Transmission],
    expected: Settlement,
    copies: int,
    received: tuple[int, int],
) -> None:
    settlement = _settle(*history)
    assert settlement == Settlement(
        exported=expected.exported,
        rejected=expected.rejected,
        unknown=expected.unknown,
        extra_copies=copies,
    )
    assert settlement.spans == 10
    assert _bounds(settlement, *received)


def test_the_other_settlements() -> None:
    assert _settle(_ok(rejected=3)) == Settlement(exported=7, rejected=3)
    assert _settle(_ok()) == Settlement(exported=10)
    assert _settle(REFUSED_CONNECT, BAD_REQUEST) == Settlement(
        refused=((HTTP_4XX, 10),)
    )
    throttled = Transmission(REFUSED, THROTTLED, 429, retryable=True)
    assert _settle(throttled, REFUSED_CONNECT) == Settlement(refused=((THROTTLED, 10),))
    assert _settle(REFUSED_CONNECT, REFUSED_CONNECT) == Settlement(
        dropped=((CONNECT_REFUSED, 10),)
    )
    # A refusal after an ambiguous transmission says nothing about the first.
    assert _settle(LOST_ACK, throttled).unknown == ((REFUSED_AFTER_AMBIGUOUS, 10),)
    # A connection failure after an ambiguous one keeps the ambiguous reason.
    assert _settle(LOST_ACK, REFUSED_CONNECT).unknown == ((TIMEOUT_AFTER_SEND, 10),)


@pytest.mark.parametrize(
    ("history", "sending", "expected"),
    [
        ([], False, Settlement(dropped=((SHUTDOWN, 10),))),
        ([REFUSED_CONNECT], False, Settlement(dropped=((SHUTDOWN, 10),))),
        ([], True, Settlement(unknown=((SHUTDOWN_IN_FLIGHT, 10),))),
        (
            [LOST_ACK],
            True,
            Settlement(unknown=((SHUTDOWN_IN_FLIGHT, 10),), extra_copies=10),
        ),
        ([LOST_ACK], False, Settlement(unknown=((TIMEOUT_AFTER_SEND, 10),))),
        ([BAD_REQUEST], False, Settlement(refused=((HTTP_4XX, 10),))),
    ],
)
def test_settling_at_the_freeze(
    history: list[Transmission], sending: bool, expected: Settlement
) -> None:
    batch = BatchHistory(10)
    for transmission in history:
        batch.record(transmission)
    assert batch.settle_at_freeze(sending=sending) == expected


def _identity(ledger: DeliveryLedger, offered: int, queued: int = 0) -> bool:
    snap: dict[str, Any] = ledger.snapshot()
    return bool(
        offered
        == (
            snap["exported"]
            + snap["rejected"]
            + sum(snap["refused"].values())
            + sum(snap["dropped"].values())
            + sum(snap["unknown"].values())
            + snap["in_flight"]
            + queued
        )
    )


def test_the_ledger_identity_holds_at_every_step() -> None:
    ledger = DeliveryLedger()
    assert ledger.take(12) and _identity(ledger, 12)
    ledger.drop_pending("encode_error", 2)
    batch = BatchHistory(10)
    assert ledger.begin(batch) and _identity(ledger, 12)
    assert ledger.attempting()
    assert ledger.record(batch, _ok(rejected=1))
    assert ledger.finish(batch) == Settlement(exported=9, rejected=1)
    snap = ledger.snapshot()
    assert (snap["exported"], snap["rejected"], snap["in_flight"]) == (9, 1, 0)
    assert snap["dropped"] == {"encode_error": 2}
    assert _identity(ledger, 12)


def test_the_freeze_settles_the_batch_in_flight_once() -> None:
    ledger = DeliveryLedger()
    ledger.take(15)
    batch = BatchHistory(10)
    ledger.begin(batch)
    ledger.attempting()
    # Five spans still building, three still queued, and a send in progress.
    ledger.freeze(drained=3, sending=lambda: True)
    snap: dict[str, Any] = ledger.snapshot()
    assert snap["unknown"] == {SHUTDOWN_IN_FLIGHT: 10}
    assert snap["dropped"] == {SHUTDOWN: 8}
    assert snap["in_flight"] == 0 and snap["frozen"]
    assert _identity(ledger, 18)
    # The worker's answer arrives after the freeze: late, and nothing changes.
    assert not ledger.record(batch, _ok())
    assert ledger.finish(batch) is None
    assert not ledger.take(1) and not ledger.attempting()
    after: dict[str, Any] = ledger.snapshot()
    assert after["late_results"] == {CONFIRMED: 1}
    assert after["unknown"] == snap["unknown"] and after["exported"] == 0
    # A second freeze is a no-op.
    ledger.freeze(drained=5, sending=lambda: True)
    assert ledger.snapshot()["dropped"] == {SHUTDOWN: 8}


def test_a_batch_between_attempts_settles_by_its_history() -> None:
    ledger = DeliveryLedger()
    ledger.take(10)
    batch = BatchHistory(10)
    ledger.begin(batch)
    ledger.attempting()
    ledger.record(batch, LOST_ACK)
    asked: list[bool] = []

    def sending() -> bool:
        asked.append(True)
        return True

    ledger.freeze(drained=0, sending=sending)
    # Not attempting at the freeze, so the sink is not even asked.
    assert asked == []
    assert ledger.snapshot()["unknown"] == {TIMEOUT_AFTER_SEND: 10}


def test_retry_delays_grow_to_the_cap_and_the_budget_ends_a_batch() -> None:
    policy = RetryPolicy()
    ceilings = [policy.delay(n, rng=lambda low, high: high) for n in range(1, 8)]
    assert ceilings == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
    assert policy.delay(3, rng=lambda low, high: low) == 0.0
    batch = BatchHistory(1, started_at=100.0)
    assert policy.remaining(batch, now=110.0) == 20.0
    assert policy.remaining(batch, now=131.0) == 0.0
    for _ in range(5):
        batch.record(LOST_ACK)
    assert policy.remaining(batch, now=101.0) == 0.0


def test_the_breaker_opens_after_three_failed_batches_and_closes_on_success() -> None:
    clock = iter(range(1, 1000))
    breaker = Breaker(clock_ns=lambda: next(clock))
    for _ in range(3):
        batch = BatchHistory(1)
        batch.record(REFUSED_CONNECT)
        breaker.attempted(REFUSED_CONNECT)
        breaker.settled(batch)
    assert breaker.snapshot()["up"] is False
    breaker.attempted(_ok())
    snap: dict[str, Any] = breaker.snapshot()
    assert snap["up"] is True
    events = [(t["event"], t["reason"]) for t in snap["transitions"]]
    assert events == [
        (FIRST_FAILURE, CONNECT_REFUSED),
        (BREAKER_OPEN, NOT_SENT),
        (FIRST_SUCCESS, None),
        (BREAKER_CLOSED, None),
    ]
    assert snap["last_success_ns"] is not None
    # A confirmed batch resets the count; two failures do not reopen it.
    for _ in range(2):
        batch = BatchHistory(1)
        batch.record(REFUSED_CONNECT)
        breaker.attempted(REFUSED_CONNECT)
        breaker.settled(batch)
    assert breaker.up


def test_the_breaker_keeps_the_latest_transitions() -> None:
    breaker = Breaker()
    for _ in range(MAX_TRANSITIONS):
        breaker.attempted(REFUSED_CONNECT)
        breaker.attempted(_ok())
    snap: dict[str, Any] = breaker.snapshot()
    assert len(snap["transitions"]) == MAX_TRANSITIONS
    assert snap["transitions_dropped"] == MAX_TRANSITIONS
