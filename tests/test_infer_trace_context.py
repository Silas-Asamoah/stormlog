"""W3C trace context and sampling by trace ID."""

import itertools

import pytest

from stormlog.infer.trace_context import (
    FOLLOW_SAMPLING,
    OFF,
    PRESERVE_ENGINE,
    TraceIds,
    derived_ids,
    keeps,
    new_trace_ids,
    parse_traceparent,
    traceparent,
)


def test_new_ids_are_valid_hex_and_never_all_zero() -> None:
    draws = iter([bytes(16), bytes(16), b"\x01" * 16, bytes(8), b"\x02" * 8])
    ids = new_trace_ids(random_bytes=lambda size: next(draws))
    assert ids.trace_id == "01" * 16 and ids.span_id == "02" * 8
    for _ in range(100):
        fresh = new_trace_ids()
        assert len(fresh.trace_id) == 32 and len(fresh.span_id) == 16
        assert parse_traceparent(traceparent(fresh, PRESERVE_ENGINE)) is not None


def test_the_sampling_threshold_is_on_the_lowest_56_bits() -> None:
    low = "f" * 18 + "00000000000000"
    high = "0" * 18 + "ffffffffffffff"
    assert not keeps(low, 0.5) and keeps(high, 0.5)
    assert keeps(low, 1.0) and not keeps(high, 0.0)
    half = "0" * 18 + "80000000000000"  # exactly the 0.5 threshold
    assert keeps(half, 0.5) and not keeps("0" * 18 + "7fffffffffffff", 0.5)


def test_a_trace_kept_at_one_ratio_is_kept_at_every_higher_one() -> None:
    ids = [new_trace_ids().trace_id for _ in range(500)]
    ratios = [0.0, 0.01, 0.1, 0.25, 0.5, 0.9, 1.0]
    for trace_id in ids:
        kept = [keeps(trace_id, ratio) for ratio in ratios]
        assert kept == sorted(kept)
    share = sum(keeps(trace_id, 0.25) for trace_id in ids) / len(ids)
    assert 0.15 < share < 0.35


def test_preserve_engine_always_sends_sampled() -> None:
    ids = TraceIds("ab" * 16, "cd" * 8, sampled=False)
    assert traceparent(ids, PRESERVE_ENGINE).endswith("-01")
    assert traceparent(ids, FOLLOW_SAMPLING).endswith("-00")
    assert traceparent(TraceIds("ab" * 16, "cd" * 8), FOLLOW_SAMPLING).endswith("-01")
    with pytest.raises(ValueError):
        traceparent(ids, OFF)


@pytest.mark.parametrize(
    "value",
    [
        "01-" + "ab" * 16 + "-" + "cd" * 8 + "-01",
        "00-" + "0" * 32 + "-" + "cd" * 8 + "-01",
        "00-" + "ab" * 16 + "-" + "0" * 16 + "-01",
        "00-" + "AB" * 16 + "-" + "cd" * 8 + "-01",
        "00-" + "ab" * 15 + "-" + "cd" * 8 + "-01",
    ],
)
def test_invalid_traceparents_are_refused(value: str) -> None:
    assert parse_traceparent(value) is None


def test_derived_ids_are_stable_and_distinct() -> None:
    first = derived_ids("run", "session", "phase", "c1", "measured")
    assert first == derived_ids("run", "session", "phase", "c1", "measured")
    others = {
        derived_ids("run", "session", "phase", case, phase)
        for case, phase in itertools.product(("c1", "c2"), ("warmup", "measured"))
    }
    assert len(others) == 4
    # Joining parts must not let two different splits collide.
    assert derived_ids("ab", "c") != derived_ids("a", "bc")
