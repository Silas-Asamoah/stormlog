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
        "CUPTI_ACTIVITY_KIND_KERNEL": {
            "start",
            "end",
            "correlationId",
            "shortName",
            "graphId",
            "graphNodeId",
        },
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
    rows = connection.execute(
        """SELECT r.rowid, r.correlationId, r.start, r.end, r.returnValue,
                  rn.value, k.rowid, k.start, k.end, kn.value,
                  k.graphId, k.graphNodeId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        LEFT JOIN StringIds AS rn ON r.nameId = rn.id
        LEFT JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k
          ON r.correlationId = k.correlationId
        LEFT JOIN StringIds AS kn ON k.shortName = kn.id
        WHERE r.start >= ? AND r.end <= ?
          AND (rn.value LIKE '%LaunchKernel%' OR rn.value LIKE '%launchKernel%'
               OR rn.value LIKE '%GraphLaunch%')
        ORDER BY r.start, r.correlationId""",
        (start, end),
    ).fetchall()
    if not rows:
        return _unavailable("no measured CUDA kernel launches were exported")
    grouped: dict[int, list[tuple[Any, ...]]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)
    missing = sum(
        any(row[1] is None or row[6] is None for row in group)
        for group in grouped.values()
    )
    duplicates = 0
    graph_replays = 0
    for group in grouped.values():
        if "GraphLaunch" in group[0][5]:
            graph_replays += 1
            nodes = [(row[10], row[11]) for row in group]
            if any(graph is None or node is None for graph, node in nodes):
                duplicates += 1
            duplicates += len(nodes) - len(set(nodes))
        else:
            duplicates += max(0, len(group) - 1)
    correlations = [group[0][1] for group in grouped.values()]
    duplicates += len(correlations) - len(set(correlations))
    failures = sum(group[0][4] != 0 for group in grouped.values())
    invalid = sum(
        row[2] > row[3] or (row[7] is not None and row[7] >= row[8]) for row in rows
    )
    unnamed = sum(row[6] is not None and row[9] is None for row in rows)
    names = Counter(row[9] for row in rows if row[6] is not None and row[9] is not None)
    after_range = sum(row[7] is not None and row[7] >= end for row in rows)
    measured_count = sum(row[6] is not None for row in rows)
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
        "measured_launches": measured_count,
        "measured_host_launch_calls": len(grouped),
        "measured_graph_replays": graph_replays,
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
