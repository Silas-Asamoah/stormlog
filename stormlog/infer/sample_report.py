"""Client system samples during a set of requests, and their peaks by device.

``infer.system_sample`` records come from the machine running Stormlog's
client (``observation_scope`` ``client_local``), and one sampler watches one
device. Pooling every sample into one peak mixes devices, so ``sample_peaks``
gives one entry per ``(sampler, device_id)``.

``request_window`` and ``samples_in_window`` are the analyzer's rules for the
samples it counts against a case: from the first request's start to the last
request's end, both ends included.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .report_stats import int_value, is_number

PEAK_FIELDS = ("device_used_bytes", "process_rss_bytes")


def request_window(requests: Iterable[Mapping[str, Any]]) -> tuple[int, int] | None:
    """The earliest start and latest end of the requests that record both.

    None when no request records a numeric start and end.
    """
    starts: list[int] = []
    ends: list[int] = []
    for record in requests:
        started, ended = record.get("started_at_ns"), record.get("ended_at_ns")
        if is_number(started) and is_number(ended):
            starts.append(int_value(started))
            ends.append(int_value(ended))
    if not starts:
        return None
    return min(starts), max(ends)


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
    given; groups are ordered by sampler, then device, as text.
    """
    groups: dict[tuple[tuple[str, Any], tuple[str, Any]], list[Mapping[str, Any]]]
    groups = {}
    for sample in samples:
        key = (_part(sample.get("sampler")), _part(sample.get("device_id")))
        groups.setdefault(key, []).append(sample)
    ordered = [(key[0][1], key[1][1], members) for key, members in groups.items()]
    return sorted(ordered, key=lambda group: (str(group[0]), str(group[1])))


def _part(value: Any) -> tuple[str, Any]:
    """A hashable group key that keeps 1, 1.0, True and "1" apart."""
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
    "sample_peaks",
    "samples_in_window",
]
