"""The analyzer's columnar summaries give the record path's floats (F-2)."""

from __future__ import annotations

import json
import math
import random
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from stormlog.infer.latency_report import (
    CLIENT_METRICS,
    latency_summary,
    streaming_summary,
)
from stormlog.infer.report_stats import FLOAT_INT_LIMIT, int_value, is_number
from stormlog.infer.request_columns import (
    ColumnRows,
    int_sum,
    numbers_at,
    sorted_floats,
)
from stormlog.infer.sample_report import request_window
from stormlog.infer.slo import client_values
from stormlog.infer.token_report import successful_throughput, token_coverage
from tests.test_infer_latency_report import _streamed

SECOND = 1_000_000_000
NAN, INF = float("nan"), float("inf")
ODD: tuple[Any, ...] = (
    NAN, INF, -INF, -0.0, 0.0, 0, 2**53 + 1, 2**64, True, False, None, "7", [], {},
    5e-324, 1e308, 7.5, 10**400,
)  # fmt: skip
#: Every field where an int past a float's range once failed a path, or
#: made the record and column paths disagree (Fable's P2-1 and P2-2).
HUGE_FIELDS = (
    "output_tokens", "total_tokens", "prompt_tokens", "e2e_latency_ms", "ttft_ms",
    "started_at_ns", "ended_at_ns", "intended_at_ns", "dispatch_lag_ms",
    "request_index",
)  # fmt: skip


@dataclass(frozen=True)
class Pool:
    values: Sequence[float]
    starts: Sequence[int]
    counts: Sequence[int]


class RecordColumns:
    """The columns of plain records, as the contract defines them.

    ``shuffled`` lays the gap pool out in reverse row order, so the
    columnar path cannot rely on a back-to-back pool.
    """

    def __init__(
        self, records: Sequence[Mapping[str, Any]], *, shuffled: bool = False
    ) -> None:
        self.records = records
        self.shuffled = shuffled

    def __len__(self) -> int:
        return len(self.records)

    def column(self, name: str) -> Sequence[Any]:
        return [r.get(name) for r in self.records]

    def client_values(self, metric: str) -> Sequence[float]:
        values = [client_values(r)[metric].value_ms for r in self.records]
        return array("d", (math.nan if v is None else v for v in values))

    def gap_counts(self) -> Sequence[int]:
        return [len(self._gaps(r)) for r in self.records]

    def gap_pool(self) -> Pool:
        order = range(len(self.records))
        values: list[float] = []
        starts = [0] * len(self.records)
        for i in reversed(order) if self.shuffled else order:
            starts[i] = len(values)
            values.extend(self._gaps(self.records[i]))
        return Pool(array("d", values), starts, self.gap_counts())

    @staticmethod
    def _gaps(record: Mapping[str, Any]) -> list[float]:
        gaps = record.get("chunk_interarrival_ms")
        if not isinstance(gaps, list):
            return []
        return [float(g) if is_number(g) or _float_like(g) else NAN for g in gaps]


def _float_like(value: Any) -> bool:
    """What the index stores as a float: NaN and infinities too, but not an
    int past a float's range, which it stores as NaN like a string."""
    return isinstance(value, float) or (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -FLOAT_INT_LIMIT < value < FLOAT_INT_LIMIT
    )


def _odd(rng: random.Random, usual: Any) -> Any:
    return rng.choice(ODD) if rng.random() < 0.3 else usual


def _record(rng: random.Random, index: int) -> dict[str, Any]:
    started = index * SECOND
    record: dict[str, Any] = {
        "event_type": "infer.request",
        "status": rng.choice(["ok"] * 6 + ["timeout", "error", "cancelled", None]),
        "stream": _odd(rng, True),
        "started_at_ns": _odd(rng, started),
        "ended_at_ns": _odd(rng, started + SECOND // 2),
        "intended_at_ns": _odd(rng, started),
        "ttft_ms": _odd(rng, rng.choice([0.0, -0.0, 1.0, 2.5])),
        "e2e_latency_ms": _odd(rng, rng.choice([0.0, -0.0, 3.0, 9.5])),
        "chunk_interarrival_ms": _odd(
            rng, [_odd(rng, rng.choice([0.0, -0.0, 4.0])) for _ in range(3)]
        ),
        "output_tokens": _odd(rng, rng.randint(0, 9)),
        "output_token_source": _odd(rng, "server_usage"),
        "output_token_exact": _odd(rng, True),
        "prompt_tokens": _odd(rng, 4),
        "prompt_token_source": _odd(rng, "estimated"),
        "total_tokens": _odd(rng, 13),
    }
    for key in list(record)[1:]:
        if rng.random() < 0.05:
            del record[key]
    return record


def _by_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    return {
        "latency": latency_summary(records),
        "streaming": streaming_summary(records),
        "tokens": token_coverage(records),
        "throughput": successful_throughput(records, None),
        "window": request_window(ok),
    }


def _by_columns(columns: ColumnRows) -> dict[str, Any]:
    return {
        "latency": latency_summary((), columns=columns),
        "streaming": streaming_summary((), columns=columns),
        "tokens": token_coverage((), columns=columns),
        "throughput": successful_throughput((), None, columns=columns),
        "window": request_window((), columns=columns.subset(columns.successful())),
    }


def _text(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("shuffled", [False, True])
def test_columns_give_the_records_floats(seed: int, shuffled: bool) -> None:
    rng = random.Random(seed)
    records = [_record(rng, i) for i in range(rng.randint(0, 300))]
    table = RecordColumns(records, shuffled=shuffled)

    def lacks(i: int, name: str) -> bool:
        return name not in records[i]

    whole = ColumnRows(table, range(len(records)), lacks=lacks)
    assert _text(_by_columns(whole)) == _text(_by_records(records))
    picked = array("q", range(0, len(records), 3))
    some = ColumnRows(table, picked, lacks=lacks)
    assert _text(_by_columns(some)) == _text(_by_records(records[::3]))


def test_signed_zeros_keep_their_order_where_a_quantile_reads_them() -> None:
    # Every value is a zero, so each level and interval bound reads one:
    # an unstable sort without the fix-up would flip signs.
    rng = random.Random(7)
    zeros = [rng.choice([0.0, -0.0]) for _ in range(400)]
    records = [
        {"status": "ok", "ttft_ms": z, "e2e_latency_ms": z, "stream": True}
        for z in zeros
    ]
    whole = ColumnRows(RecordColumns(records), range(len(records)))
    assert _text(latency_summary((), columns=whole)) == _text(latency_summary(records))
    ordered = sorted_floats(np.asarray(zeros))
    assert [repr(v) for v in ordered.tolist()] == [repr(v) for v in sorted(zeros)]


@pytest.mark.parametrize("seed", range(20))
def test_sorted_floats_is_sorted_bit_for_bit(seed: int) -> None:
    rng = random.Random(seed)
    pool = [0.0, -0.0, 1.0, -1.0, 5e-324, -5e-324, 1e308, INF, -INF, 2.5]
    values = [rng.choice(pool) for _ in range(rng.randint(0, 2000))]
    got = sorted_floats(np.asarray(values, dtype=np.float64)).tolist()
    assert [repr(v) for v in got] == [repr(v) for v in sorted(values)]


def test_the_cell_rules_are_is_number_and_int_value() -> None:
    huge = 2**1024 - 2**970  # the first int float() overflows on
    assert is_number(huge - 1) and not is_number(huge) and not is_number(-huge)
    cells = list(ODD) + [1, -3, 2**70, 4.0, huge, -huge, huge - 1, 10**400]
    assert numbers_at(cells) == [k for k, v in enumerate(cells) if is_number(v)]
    assert int_sum(cells) == sum(int_value(v) for v in cells)
    ints = [1, None, 2**70, -4, None, 10**400, -huge, huge - 1]
    assert numbers_at(ints) == [0, 2, 3, 7]
    reals = [1.5, None, huge, -(huge - 1)]
    assert numbers_at(reals) == [0, 3]
    assert int_sum(ints) == sum(int_value(v) for v in ints) == 1 + 2**70 - 4 + huge - 1
    # A column sum over every cell is the sum over the numbers only if
    # int_value reads what is_number refuses as 0 (Fable's P2-1).
    for cell in cells + ints + reals:
        assert is_number(cell) or int_value(cell) == 0, cell
    assert int_sum([10**400]) == int_sum([-huge, None]) == 0


@pytest.mark.parametrize("field", HUGE_FIELDS)
@pytest.mark.parametrize("huge", [10**400, -(10**400)], ids=["e400", "-e400"])
def test_an_int_past_a_floats_range_is_no_number_by_record_or_by_column(
    field: str, huge: int
) -> None:
    # Fable's P2-1 and P2-2: such an int raised OverflowError in a token
    # rate or slo._number, or counted in a column sum and a window while
    # the record path left it out.
    records = [_streamed(0), _streamed(1, **{field: huge}), _streamed(2)]
    whole = ColumnRows(RecordColumns(records), range(len(records)))
    assert _text(_by_columns(whole)) == _text(_by_records(records))


def test_a_window_over_int_columns_skips_an_int_past_a_floats_range() -> None:
    # Fable's P2-1: _all_ints took the plain maximum, 10**400, while the
    # record path left the cell out of the window.
    records = [_streamed(0), _streamed(1, ended_at_ns=10**400), _streamed(2)]
    columns = ColumnRows(RecordColumns(records), range(len(records)))
    assert request_window(records) == (0, 2 * SECOND + SECOND // 2)
    assert request_window((), columns=columns) == request_window(records)


def test_records_and_columns_are_never_mixed() -> None:
    record = {"status": "ok", "ttft_ms": 1.0, "e2e_latency_ms": 2.0}
    columns = ColumnRows(RecordColumns([record]), range(1))
    with pytest.raises(ValueError, match="not both"):
        latency_summary([record], columns=columns)
    with pytest.raises(ValueError, match="not both"):
        token_coverage([record], columns=columns)


def test_every_client_metric_is_summarized_by_column() -> None:
    record = {"status": "ok", "ttft_ms": 1.0, "e2e_latency_ms": 2.0}
    columns = ColumnRows(RecordColumns([record]), range(1))
    assert sorted(latency_summary((), columns=columns)["metrics"]) == sorted(
        CLIENT_METRICS
    )
