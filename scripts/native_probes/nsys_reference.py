"""Inspect a pinned Nsight Systems SQLite export without inferring missing events."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

MEASURED_RANGE = "stormlog-native-probe-measured"


def measured_kernels(path: Path, *, range_name: str = MEASURED_RANGE) -> dict[str, Any]:
    """Attribute kernels by correlated host launches within one NVTX range.

    CUDA launch and device execution are asynchronous. A kernel launched inside
    the range may execute after its end, so device timestamps do not select the
    measured population. An incomplete correlation cannot be a trusted reference.
    """
    if not path.is_file():
        return _unavailable("Nsight SQLite export is missing")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            return _inspect(connection, range_name)
    except sqlite3.DatabaseError as error:
        return _unavailable(f"Nsight SQLite export is malformed: {error}")


def _inspect(connection: sqlite3.Connection, range_name: str) -> dict[str, Any]:
    required = {
        "NVTX_EVENTS": {"start", "end", "text", "textId"},
        "CUPTI_ACTIVITY_KIND_RUNTIME": {
            "start",
            "end",
            "correlationId",
            "nameId",
            "returnValue",
        },
        "CUPTI_ACTIVITY_KIND_KERNEL": {"start", "end", "correlationId", "shortName"},
        "StringIds": {"id", "value"},
    }
    for table, columns in required.items():
        actual = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if not columns <= actual:
            return _unavailable(
                f"Nsight export missing {table} columns: {sorted(columns - actual)}"
            )
    ranges = connection.execute(
        """SELECT n.start, n.end FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON n.textId = s.id
        WHERE n.text = ? OR s.value = ?""",
        (range_name, range_name),
    ).fetchall()
    if len(ranges) != 1 or ranges[0][1] is None or ranges[0][0] >= ranges[0][1]:
        return _unavailable(
            f"expected one complete measured NVTX range; found {len(ranges)}"
        )
    start, end = ranges[0]
    launches = connection.execute(
        """SELECT r.correlationId, r.start, r.end, r.returnValue,
                  rn.value, k.start, k.end, kn.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        LEFT JOIN StringIds AS rn ON r.nameId = rn.id
        LEFT JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k
          ON r.correlationId = k.correlationId
        LEFT JOIN StringIds AS kn ON k.shortName = kn.id
        WHERE r.start >= ? AND r.end <= ?
          AND (rn.value LIKE '%LaunchKernel%' OR rn.value LIKE '%launchKernel%')
        ORDER BY r.start, r.correlationId""",
        (start, end),
    ).fetchall()
    if not launches:
        return _unavailable("no measured CUDA kernel launches were exported")
    identifiers = [row[0] for row in launches]
    missing = sum(row[0] is None or row[5] is None for row in launches)
    duplicates = len(identifiers) - len(set(identifiers))
    failures = sum(row[3] != 0 for row in launches)
    invalid = sum(
        row[1] > row[2] or (row[5] is not None and row[5] >= row[6]) for row in launches
    )
    unnamed = sum(row[7] is None for row in launches)
    names = Counter(row[7] for row in launches if row[7] is not None)
    after_range = sum(row[5] is not None and row[5] >= end for row in launches)
    problems = []
    for label, count in (
        ("missing kernel correlation", missing),
        ("duplicate launch correlation", duplicates),
        ("failed CUDA launch", failures),
        ("invalid interval", invalid),
        ("unnamed kernel", unnamed),
    ):
        if count:
            problems.append(f"{label}: {count}")
    return {
        "status": "valid_export" if not problems else "partial",
        "range_name": range_name,
        "range_start_ns": start,
        "range_end_ns": end,
        "measured_launches": len(launches),
        "kernel_counts_by_name": dict(sorted(names.items())),
        "kernels_executing_after_range_end": after_range,
        "missing_kernel_correlations": missing,
        "duplicate_launch_correlations": duplicates,
        "failed_cuda_launches": failures,
        "problems": problems,
        "external_capture_complete": None,
        "comparison_with_direct_cupti": "pending",
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "measured_launches": None,
        "kernel_counts_by_name": None,
        "external_capture_complete": None,
        "comparison_with_direct_cupti": "pending",
    }
