"""How many token counts a set of requests has, where they came from, and rates.

A missing count is not zero tokens. The analyzer's ``tokens`` block sums
the counts it has, so a sum over requests that lack counts reads low with
nothing to say so. ``token_coverage`` says how many successful requests
had a count, how many were exact, and where each came from, beside the
same sum.

``throughput`` is the analyzer's rate block: successful requests and their
tokens per second of a case's rate interval.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .populations import MeasuredInterval, rate
from .report_stats import int_value, is_number
from .request_columns import int_sum, no_records, numbers_at, pick

if TYPE_CHECKING:
    from .request_columns import ColumnRows

TOKEN_KINDS = ("output", "prompt")


def token_coverage(
    requests: Iterable[Mapping[str, Any]], *, columns: ColumnRows | None = None
) -> dict[str, dict[str, Any]]:
    """Coverage of the output and prompt token counts of successful requests.

    For each kind: ``eligible`` successful requests; ``known`` of them with
    a numeric count; ``exact`` of those flagged exact; ``by_source`` the
    eligible requests by recorded source (``unknown`` when none is
    recorded); ``known_sum`` the sum of the known counts, as the analyzer
    sums them; ``partial`` when some eligible request has no count.

    With ``columns`` the requests are those rows, read by column, and
    ``requests`` must be empty; the coverage is the same.
    """
    if columns is not None:
        no_records(requests)
        ok = columns.successful()
        return {kind: _columnar_coverage(columns, ok, kind) for kind in TOKEN_KINDS}
    successful = [r for r in requests if r.get("status") == "ok"]
    return {kind: _coverage(successful, kind) for kind in TOKEN_KINDS}


def _coverage(successful: list[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    known = [r for r in successful if is_number(r.get(f"{kind}_tokens"))]
    sources = Counter(str(r.get(f"{kind}_token_source", "unknown")) for r in successful)
    return {
        "eligible": len(successful),
        "known": len(known),
        "exact": sum(1 for r in known if r.get(f"{kind}_token_exact") is True),
        "by_source": dict(sorted(sources.items())),
        "known_sum": sum(int_value(r.get(f"{kind}_tokens")) for r in known),
        "partial": len(known) < len(successful),
    }


def _columnar_coverage(
    columns: ColumnRows, ok: Sequence[int], kind: str
) -> dict[str, Any]:
    """``_coverage`` of the rows at positions ``ok``, by column."""
    counts = pick(columns.column(f"{kind}_tokens"), ok)
    exact = pick(columns.column(f"{kind}_token_exact"), ok)
    known = numbers_at(counts)
    return {
        "eligible": len(ok),
        "known": len(known),
        "exact": sum(1 for j in known if exact[j] is True),
        "by_source": dict(sorted(_columnar_sources(columns, ok, kind).items())),
        # int_value is 0 for every cell that is not a number, so the sum
        # over all cells is the sum over the known ones.
        "known_sum": int_sum(counts),
        "partial": len(known) < len(ok),
    }


def _columnar_sources(
    columns: ColumnRows, ok: Sequence[int], kind: str
) -> Counter[str]:
    """``str(record.get(source, "unknown"))`` counted, by column.

    A None cell is a missing key (``"unknown"``) or a null one
    (``"None"``); only those rows are asked which.
    """
    name = f"{kind}_token_source"
    cells = columns.column(name)
    sources = Counter(map(str, pick(cells, ok)))
    for k in ok if sources["None"] else ():
        if cells[k] is None and columns.lacks(k, name):
            sources["None"] -= 1
            sources["unknown"] += 1
    return +sources


def throughput(
    successful: int,
    output_tokens: int,
    total_tokens: int,
    interval: MeasuredInterval | None,
) -> dict[str, Any]:
    """Successful requests and their tokens per second of the rate interval.

    The rates are null when the interval has no length, instead of zero.
    """
    return {
        "interval_seconds": interval.seconds if interval is not None else None,
        "interval_kind": interval.kind if interval is not None else None,
        "numerator_cohort": (
            interval.numerator_cohort if interval is not None else None
        ),
        "requests_per_second": rate(successful, interval),
        "output_tokens_per_second": rate(output_tokens, interval),
        "total_tokens_per_second": rate(total_tokens, interval),
    }


def successful_throughput(
    requests: Iterable[Mapping[str, Any]],
    interval: MeasuredInterval | None,
    *,
    columns: ColumnRows | None = None,
) -> dict[str, Any]:
    """``throughput`` of the successful requests among ``requests``.

    The analyzer's rate block for a case or a segment: successful requests,
    and the sums of their output and total token counts (a count that is
    not a number adds 0, as the analyzer sums), per second of ``interval``.
    With ``columns`` the requests are those rows, read by column, and
    ``requests`` must be empty.
    """
    if columns is not None:
        no_records(requests)
        ok = columns.successful()
        output, total = columns.column("output_tokens"), columns.column("total_tokens")
        return throughput(
            len(ok), int_sum(pick(output, ok)), int_sum(pick(total, ok)), interval
        )
    successful = [r for r in requests if r.get("status") == "ok"]
    return throughput(
        len(successful),
        sum(int_value(r.get("output_tokens")) for r in successful),
        sum(int_value(r.get("total_tokens")) for r in successful),
        interval,
    )


__all__ = ["TOKEN_KINDS", "successful_throughput", "throughput", "token_coverage"]
