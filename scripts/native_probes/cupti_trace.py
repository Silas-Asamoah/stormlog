"""Stream a Stormlog CUPTI trace and an Nsight export into comparable summaries.

Both summaries attribute device kernels to the host runtime call that launched
them through the CUPTI correlation ID. A window, when given, selects kernels by
their launch's CPU start time, because a kernel launched inside the window may
execute after it ends.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from scripts.native_probes.workloads.vllm_open_loop import (
    CUPTI_FRAME_HEADER,
    CUPTI_TRACE_MAGIC,
)

_TEMPLATE_OR_ARGS = re.compile(r"[<(]")


def short_kernel_name(name: str) -> str:
    """Reduce a demangled or plain kernel name to its unqualified base name."""
    base = _TEMPLATE_OR_ARGS.split(name, maxsplit=1)[0].strip()
    base = base.split(" ")[-1]
    return base.rsplit("::", 1)[-1]


def demangle(names: Iterable[str]) -> dict[str, str]:
    """Demangle names with c++filt when present; unknown names map to themselves."""
    unique = sorted(set(names))
    tool = shutil.which("c++filt")
    if not unique or tool is None:
        return {name: name for name in unique}
    output = subprocess.run(
        [tool],
        input="\n".join(unique) + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if len(output) != len(unique):
        return {name: name for name in unique}
    return dict(zip(unique, output))


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a finalized trace; raise on any framing defect."""
    with path.open("rb") as source:
        if source.read(len(CUPTI_TRACE_MAGIC)) != CUPTI_TRACE_MAGIC:
            raise ValueError(f"{path}: not a Stormlog CUPTI trace")
        while True:
            header = source.read(CUPTI_FRAME_HEADER.size)
            if len(header) != CUPTI_FRAME_HEADER.size:
                raise ValueError(f"{path}: end frame missing")
            encoded, raw_size, count = CUPTI_FRAME_HEADER.unpack(header)
            if (encoded, raw_size, count) == (0, 0, 0):
                return
            raw = zlib.decompress(source.read(encoded))
            lines = raw.splitlines()
            if len(raw) != raw_size or len(lines) != count:
                raise ValueError(f"{path}: frame size or record count mismatch")
            for line in lines:
                yield json.loads(line)


def _overlap_ns(intervals: Iterable[tuple[int, int, str]]) -> int:
    """Total time during which kernels on two or more streams run together."""
    events: list[tuple[int, int, str]] = []
    for start, end, stream in intervals:
        events.append((start, 1, stream))
        events.append((end, -1, stream))
    events.sort(key=lambda event: (event[0], event[1]))
    active: Counter[str] = Counter()
    overlap = 0
    previous: int | None = None
    for at, delta, stream in events:
        if previous is not None and sum(1 for n in active.values() if n > 0) >= 2:
            overlap += at - previous
        active[stream] += delta
        previous = at
    return overlap


def summarize_kernels(
    launches: dict[str, int],
    kernels: list[dict[str, Any]],
    window: tuple[int, int] | None,
) -> dict[str, Any]:
    """Summarize kernels given launch CPU starts by correlation ID."""
    if window is not None:
        selected = [
            k
            for k in kernels
            if k["correlation_id"] in launches
            and window[0] <= launches[k["correlation_id"]] <= window[1]
        ]
    else:
        selected = kernels
    correlated = sum(k["correlation_id"] in launches for k in selected)
    graph_kernels = sum(k["graph_id"] is not None for k in selected)
    return {
        "kernels": len(selected),
        "correlated_kernels": correlated,
        "correlation_coverage": correlated / len(selected) if selected else None,
        "graph_node_kernels": graph_kernels,
        "streams": len({k["stream_id"] for k in selected}),
        "kernels_missing_device_timestamps": sum(
            k["start"] is None or k["end"] is None for k in selected
        ),
        "stream_overlap_ns": _overlap_ns(
            (k["start"], k["end"], k["stream_id"])
            for k in selected
            if k["start"] is not None and k["end"] is not None
        ),
        "kernel_counts_by_name": dict(
            sorted(Counter(short_kernel_name(k["name"]) for k in selected).items())
        ),
    }


def cupti_summary(
    paths: Iterable[Path], window: tuple[int, int] | None = None
) -> dict[str, Any]:
    """Summarize one or more per-process Stormlog CUPTI traces."""
    launches: dict[str, int] = {}
    kernels: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    for path in paths:
        for record in iter_records(path):
            kind = record["activity_kind"]
            kinds[kind] += 1
            key = f"{path.parent.name}:{record['correlation_id']}"
            if kind in ("runtime", "driver"):
                launches[key] = record["cpu_start_ns"]
            elif kind == "kernel":
                kernels.append(
                    {
                        "correlation_id": key,
                        "start": record["device_start_ns"],
                        "end": record["device_end_ns"],
                        "stream_id": record["stream_id"],
                        "graph_id": record["graph_id"],
                        "name": record["metadata"]["name"],
                    }
                )
    names = demangle(kernel["name"] for kernel in kernels)
    for kernel in kernels:
        kernel["name"] = names[kernel["name"]]
    return {
        "records_by_kind": dict(kinds),
        **summarize_kernels(launches, kernels, window),
    }


def nsys_summary(path: Path, window: tuple[int, int] | None = None) -> dict[str, Any]:
    """Summarize an Nsight Systems SQLite export the same way."""
    uri = f"{path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        launches = {
            f"{gid}:{cid}": start
            for gid, cid, start in connection.execute(
                "SELECT globalTid >> 24, correlationId, start "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            )
        }
        kernels = [
            {
                "correlation_id": f"{gid}:{cid}",
                "start": start,
                "end": end,
                "stream_id": str(stream),
                "graph_id": graph,
                "name": name or "",
            }
            for gid, cid, start, end, stream, graph, name in connection.execute(
                "SELECT k.globalPid >> 24, k.correlationId, k.start, k.end, "
                "k.streamId, k.graphId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL AS k "
                "LEFT JOIN StringIds AS s ON k.demangledName = s.id"
            )
        ]
    return summarize_kernels(launches, kernels, window)


def compare(
    cupti: dict[str, Any], nsys: dict[str, Any], *, exact: bool
) -> dict[str, Any]:
    """Compare two summaries; exact counts only for deterministic workloads."""
    cupti_names = set(cupti["kernel_counts_by_name"])
    nsys_names = set(nsys["kernel_counts_by_name"])
    result: dict[str, Any] = {
        "names_only_in_cupti": sorted(cupti_names - nsys_names),
        "names_only_in_nsys": sorted(nsys_names - cupti_names),
        "cupti_correlation_coverage": cupti["correlation_coverage"],
        "nsys_correlation_coverage": nsys["correlation_coverage"],
        "graph_node_kernels": [cupti["graph_node_kernels"], nsys["graph_node_kernels"]],
    }
    if exact:
        result["kernel_count_match"] = cupti["kernels"] == nsys["kernels"]
        result["per_name_count_match"] = (
            cupti["kernel_counts_by_name"] == nsys["kernel_counts_by_name"]
        )
    result["pass"] = (
        not result["names_only_in_cupti"]
        and not result["names_only_in_nsys"]
        and (cupti["correlation_coverage"] or 0) >= 0.99
        and (not exact or result["per_name_count_match"])
    )
    return result
