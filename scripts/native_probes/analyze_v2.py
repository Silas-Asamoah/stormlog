"""Reduce protocol v2 window-scaling trials to per-mode, per-window tables.

Every trial is counted, including failed and timed-out ones. Latency
perturbation compares a trial's window p95 with the median window p95 of the
`off` trials over the same request indices.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

METRICS = {
    "added_rss_mib": lambda r: _mib(r["memory"].get("target_rss_added_peak_bytes")),
    "stopping_rss_peak_mib": lambda r: _mib(
        r["memory"].get("target_rss_stopping_peak_bytes")
    ),
    "stop_seconds": lambda r: r.get("stop_seconds"),
    "window_p95_ms": lambda r: r["latency_window"]["e2e_p95_ms"],
    "window_ttft_p95_ms": lambda r: r["latency_window"]["ttft_p95_ms"],
    "after_window_p95_ms": lambda r: r["latency_after_window"]["e2e_p95_ms"],
    "artifact_mib": lambda r: _mib(r.get("artifacts", {}).get("bytes")),
    "records": lambda r: r.get("artifacts", {}).get("delivered_records"),
    "dropped": lambda r: r.get("artifacts", {}).get("dropped_records"),
    "kernels_without_timestamps": lambda r: r.get("artifacts", {}).get(
        "kernels_without_timestamps"
    ),
    "buffer_peak_mib": lambda r: _mib(
        r.get("artifacts", {}).get("buffer_peak_outstanding_bytes")
    ),
    "callback_cpu_s": lambda r: r.get("artifacts", {}).get(
        "buffer_completed_cpu_seconds"
    ),
    "target_cpu_s": lambda r: r.get("target_cpu_seconds_capture_to_end"),
    "over_slo_or_failed": lambda r: r["stall"]["over_slo_or_failed"],
    "max_e2e_ms": lambda r: r["stall"]["max_e2e_ms"],
}


def _mib(value: float | None) -> float | None:
    return None if value is None else value / 1024**2


SLO_MS = 2000.0


def _stall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Requests that missed the 2 s SLO or failed, and the worst latency."""
    latencies = [r["e2e_latency_ms"] for r in rows if r.get("status") == "ok"]
    return {
        "over_slo_or_failed": sum(
            r.get("status") != "ok" or r["e2e_latency_ms"] > SLO_MS for r in rows
        ),
        "max_e2e_ms": max(latencies) if latencies else None,
    }


def load(root: Path) -> list[dict[str, Any]]:
    trials = []
    for path in sorted(root.glob("*/result.json")):
        if path.parent.name.startswith("smoke-"):
            continue
        trial = json.loads(path.read_text())
        requests = path.parent / "requests.jsonl"
        rows = [json.loads(line) for line in requests.read_text().splitlines()]
        trial["stall"] = _stall(rows)
        trials.append(trial)
    return trials


def _spread(values: Iterable[float | None]) -> dict[str, Any]:
    present = [v for v in values if v is not None]
    if not present:
        return {"n": 0}
    return {
        "n": len(present),
        "min": min(present),
        "median": statistics.median(present),
        "max": max(present),
    }


def off_window_p95(rows_by_trial: dict[str, list[dict]], window: int) -> float | None:
    """Median over off trials of p95 for requests indexed below the window."""
    values = []
    for rows in rows_by_trial.values():
        latencies = sorted(
            row["e2e_latency_ms"]
            for row in rows
            if row.get("status") == "ok"
            and int(row["request_id"].rsplit("-", 1)[1]) < window
        )
        if latencies:
            index = (len(latencies) - 1) * 0.95
            lower = int(index)
            upper = min(lower + 1, len(latencies) - 1)
            values.append(
                latencies[lower]
                + (latencies[upper] - latencies[lower]) * (index - lower)
            )
    return statistics.median(values) if values else None


def _off_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        trial["trial_id"]: [
            json.loads(line)
            for line in (root / trial["trial_id"] / "requests.jsonl")
            .read_text()
            .splitlines()
        ]
        for trial in load(root)
        if trial["mode"] == "off"
    }


def analyze(root: Path, off_root: Path | None = None) -> dict[str, Any]:
    """Summarize trials in root; `off` baselines come from off_root if given."""
    trials = load(root)
    off_rows = _off_rows(off_root or root)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[(trial["mode"], trial["window_requests"])].append(trial)
    table = []
    for (mode, window), members in sorted(grouped.items()):
        baseline = off_window_p95(off_rows, window)
        row: dict[str, Any] = {
            "mode": mode,
            "window": window,
            "trials": len(members),
            "ok": sum(m["ok"] for m in members),
            "errors": sorted({e for m in members for e in m["errors"]}),
            "off_window_p95_ms": baseline,
        }
        for name, getter in METRICS.items():
            row[name] = _spread(_safe(getter, m) for m in members)
        if baseline:
            row["window_p95_perturbation"] = _spread(
                (m["latency_window"]["e2e_p95_ms"] - baseline) / baseline
                for m in members
                if m["latency_window"]["e2e_p95_ms"] is not None
            )
        table.append(row)
    return {"trials": len(trials), "table": table}


def _safe(getter: Any, trial: dict[str, Any]) -> Any:
    try:
        return getter(trial)
    except (KeyError, TypeError):
        return None


def markdown(summary: dict[str, Any]) -> str:
    columns = [
        ("added_rss_mib", "added RSS MiB"),
        ("stop_seconds", "stop s"),
        ("window_p95_perturbation", "window p95 Δ"),
        ("artifact_mib", "artifact MiB"),
        ("records", "records"),
        ("dropped", "dropped"),
        ("buffer_peak_mib", "CUPTI buf MiB"),
        ("over_slo_or_failed", "reqs >2s/failed"),
    ]
    header = "| mode | W | ok/n | " + " | ".join(label for _, label in columns) + " |"
    lines = [header, "|" + "---|" * (3 + len(columns))]
    for row in summary["table"]:
        cells = []
        for key, _ in columns:
            spread = row.get(key) or {"n": 0}
            if not spread["n"]:
                cells.append("–")
            elif key == "window_p95_perturbation":
                cells.append(
                    f"{spread['median']:+.1%} [{spread['min']:+.1%}, {spread['max']:+.1%}]"
                )
            else:
                cells.append(
                    f"{spread['median']:.4g} [{spread['min']:.4g}, {spread['max']:.4g}]"
                )
        lines.append(
            f"| {row['mode']} | {row['window']} | {row['ok']}/{row['trials']} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--off-runs", type=Path)
    parser.add_argument("--name", default="window_scaling")
    args = parser.parse_args()
    summary = analyze(args.runs, args.off_runs)
    summary["off_baseline_runs"] = (
        args.off_runs.name if args.off_runs else args.runs.name
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(summary, indent=1) + "\n")
    (args.out / f"{args.name}.md").write_text(markdown(summary))
    print(markdown(summary))


if __name__ == "__main__":
    main()
