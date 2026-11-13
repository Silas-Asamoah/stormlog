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
from collections.abc import Iterable, Mapping
from typing import Any

from .populations import MeasuredInterval, rate
from .report_stats import int_value, is_number

TOKEN_KINDS = ("output", "prompt")


def token_coverage(
    requests: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Coverage of the output and prompt token counts of successful requests.

    For each kind: ``eligible`` successful requests; ``known`` of them with
    a numeric count; ``exact`` of those flagged exact; ``by_source`` the
    eligible requests by recorded source (``unknown`` when none is
    recorded); ``known_sum`` the sum of the known counts, as the analyzer
    sums them; ``partial`` when some eligible request has no count.
    """
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


__all__ = ["TOKEN_KINDS", "throughput", "token_coverage"]
