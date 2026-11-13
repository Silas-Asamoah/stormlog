"""Client system samples during a set of requests, and their peaks by device.

``infer.system_sample`` records come from the machine running Stormlog's
client (``observation_scope`` ``client_local``), and one sampler watches one
device. Pooling every sample into one peak mixes devices, so ``sample_peaks``
gives one entry per ``(sampler, device_id)``.

``request_window`` and ``samples_in_window`` are the analyzer's rules for the
samples it counts against a case: from the first request's start to the last
request's end, both ends included. ``sample_gaps`` says where one series of
samples breaks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .observers import SCRAPE_GAP_FACTOR
from .quantiles import quantile
from .report_stats import int_value, is_number
from .request_columns import no_records

if TYPE_CHECKING:
    from .request_columns import ColumnRows

PEAK_FIELDS = ("device_used_bytes", "process_rss_bytes")


def request_window(
    requests: Iterable[Mapping[str, Any]], *, columns: ColumnRows | None = None
) -> tuple[int, int] | None:
    """The earliest start and latest end of the requests that record both.

    None when no request records a numeric start and end. With ``columns``
    the requests are those rows, read by column, and ``requests`` must be
    empty.
    """
    if columns is not None:
        no_records(requests)
        started, ended = columns.column("started_at_ns"), columns.column("ended_at_ns")
        if started and _all_ints(started) and _all_ints(ended):
            return min(started), max(ended)
        pairs: Iterable[tuple[Any, Any]] = zip(started, ended)
    else:
        pairs = ((r.get("started_at_ns"), r.get("ended_at_ns")) for r in requests)
    starts: list[int] = []
    ends: list[int] = []
    for started, ended in pairs:
        if is_number(started) and is_number(ended):
            starts.append(int_value(started))
            ends.append(int_value(ended))
    if not starts:
        return None
    return min(starts), max(ends)


def _all_ints(cells: Sequence[Any]) -> bool:
    """Every cell a JSON int: then ``is_number`` holds and ``int_value``
    is the cell itself, so the window is the plain minimum and maximum."""
    return set(map(type, cells)) == {int}


def samples_in_window(
    samples: Iterable[Mapping[str, Any]], start_ns: int, end_ns: int
) -> list[Mapping[str, Any]]:
    """The samples with a numeric timestamp in ``[start_ns, end_ns]``."""
    return [
        sample
        for sample in samples
        if is_number(sample.get("timestamp_ns"))
        and start_ns <= int_value(sample.get("timestamp_ns")) <= end_ns
    ]


def sample_peaks(samples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per ``(sampler, device_id)``: its sample count and peaks.

    A peak is the largest numeric value of its field, as the analyzer takes
    it, or None when no sample of the group has one. Entries are ordered by
    sampler, then device, as text.
    """
    return [
        _entry(sampler, device_id, members)
        for sampler, device_id, members in group_samples(samples)
    ]


def group_samples(
    samples: Iterable[Mapping[str, Any]],
) -> list[tuple[Any, Any, list[Mapping[str, Any]]]]:
    """The samples by ``(sampler, device_id)``, as ``sample_peaks`` groups them.

    Each group is ``(sampler, device_id, members)``, members in the order
    given, with the group's first ``sampler`` and ``device_id`` values as
    written (never a stand-in for them, so a NaN id and the text "nan" stay
    two groups with two ids); groups are ordered by sampler, then device,
    as text.
    """
    groups: dict[
        tuple[tuple[str, Any], tuple[str, Any]],
        tuple[Any, Any, list[Mapping[str, Any]]],
    ] = {}
    for sample in samples:
        sampler, device_id = sample.get("sampler"), sample.get("device_id")
        key = (_part(sampler), _part(device_id))
        group = groups.get(key)
        if group is None:
            group = groups[key] = (sampler, device_id, [])
        group[2].append(sample)
    return sorted(groups.values(), key=lambda group: (str(group[0]), str(group[1])))


def sample_gaps(
    times: Sequence[int], *, factor: float = SCRAPE_GAP_FACTOR
) -> list[tuple[int, int]]:
    """Where a series of samples breaks: each pair of consecutive times
    (ascending) more than ``factor`` times the series' median interval
    apart.

    The series has no configured interval to compare with, so its own
    median interval stands in for the one ``observers`` applies
    ``SCRAPE_GAP_FACTOR`` to (a scrape gap). None when the median is 0 or
    there are fewer than two times.
    """
    intervals = [b - a for a, b in zip(times, times[1:])]
    median = quantile(intervals, 0.5)
    if not median:
        return []
    return [(a, b) for a, b in zip(times, times[1:]) if b - a > factor * median]


def _part(value: Any) -> tuple[str, Any]:
    """A hashable group key that keeps 1, 1.0, True and "1" apart: the
    type's name, and the value or, for one that is not hashable or not
    equal to itself (a list, an object, NaN), its repr."""
    if value is None or isinstance(value, (str, int)):
        return type(value).__name__, value
    if isinstance(value, float) and math.isfinite(value):
        return "float", value
    return type(value).__name__, repr(value)


def _entry(
    sampler: Any, device_id: Any, members: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "sampler": sampler,
        "device_id": device_id,
        "samples": len(members),
    }
    for field in PEAK_FIELDS:
        entry[f"peak_{field}"] = max(
            (int_value(s.get(field)) for s in members if is_number(s.get(field))),
            default=None,
        )
    return entry


__all__ = [
    "PEAK_FIELDS",
    "group_samples",
    "request_window",
    "sample_gaps",
    "sample_peaks",
    "samples_in_window",
]
