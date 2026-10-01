"""Tracking-session W&B export helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..session import SessionSummary
from .attribution import log_attribution_outputs
from .core import (
    WandbExportConfig,
    coerce_existing_dir,
    coerce_existing_file,
    log_directory_artifact,
    log_file_artifact,
    materialize_html_file,
    resolve_run,
    session_slug,
    session_summary_fields,
    update_summary,
)
from .dashboard import tracking_dashboard_html

_ALERT_EVENT_TYPES = frozenset({"warning", "critical", "error", "peak"})
_TIMELINE_MAX_POINTS = 250


def export_tracking_run_to_wandb(
    config: WandbExportConfig,
    *,
    command_name: str,
    session_summary: SessionSummary | None,
    stats: Mapping[str, Any],
    events: Sequence[Any],
    output_path: str | Path | None = None,
    telemetry_sink_dir: str | Path | None = None,
    oom_dump_path: str | Path | None = None,
    attribution_bundle_dir: str | Path | None = None,
) -> None:
    """Export one completed tracking session to W&B."""
    if not config.enabled:
        return

    timeline_rows = tracking_timeline_rows(events)
    wandb, run, managed = resolve_run(
        config,
        command_name=command_name,
        session_summary=session_summary,
    )
    safe_session = session_slug(session_summary)
    output_file = coerce_existing_file(output_path)
    sink_dir = coerce_existing_dir(telemetry_sink_dir)
    oom_dir = coerce_existing_dir(oom_dump_path)
    attribution_dir = coerce_existing_dir(attribution_bundle_dir) or oom_dir

    try:
        update_summary(
            run,
            tracking_metrics(stats)
            | {"stormlog_chart_point_count": len(timeline_rows)}
            | session_summary_fields(session_summary)
            | tracking_summary_fields(stats, output_path=output_path),
        )

        log_tracking_time_series(run, timeline_rows)

        if config.log_tables:
            log_alerts_table(wandb, run, events)
            update_summary(
                run,
                log_tracking_visualizations(
                    wandb,
                    run,
                    timeline_rows,
                    session_slug=safe_session,
                    dashboard_root=_tracking_dashboard_root(output_file, sink_dir),
                    allow_artifact_logging=config.log_artifacts,
                ),
            )

        if config.log_artifacts:
            _log_tracking_artifacts(
                wandb,
                run,
                safe_session=safe_session,
                output_file=output_file,
                sink_dir=sink_dir,
                oom_dir=oom_dir,
                attribution_dir=attribution_dir,
                attribution_bundle_dir=attribution_bundle_dir,
            )

        if config.log_attribution and attribution_dir is not None:
            update_summary(
                run,
                log_attribution_outputs(
                    wandb,
                    run,
                    root=attribution_dir,
                    session_slug=safe_session,
                    allow_artifact_logging=config.log_artifacts,
                ),
            )
    finally:
        if managed:
            run.finish()


def tracking_metrics(stats: Mapping[str, Any]) -> dict[str, Any]:
    metric_names = {
        "stormlog_peak_memory_bytes": "peak_memory",
        "stormlog_total_events": "total_events",
        "stormlog_alert_count": "alert_count",
        "stormlog_current_memory_allocated_bytes": "current_memory_allocated",
        "stormlog_current_memory_reserved_bytes": "current_memory_reserved",
        "stormlog_memory_utilization_percent": "memory_utilization_percent",
        "stormlog_total_allocations": "total_allocations",
        "stormlog_total_deallocations": "total_deallocations",
        "stormlog_total_allocation_bytes": "total_allocation_bytes",
        "stormlog_total_deallocation_bytes": "total_deallocation_bytes",
        "stormlog_tracking_duration_seconds": "tracking_duration_seconds",
        "stormlog_allocations_per_second": "allocations_per_second",
        "stormlog_bytes_allocated_per_second": "bytes_allocated_per_second",
        "stormlog_history_retained_events": "history_retained_events",
        "stormlog_history_dropped_events": "history_dropped_events",
        "stormlog_sink_rollover_count": "rollover_count",
        "stormlog_sink_pruned_segment_count": "pruned_segment_count",
        "stormlog_sink_pruned_bytes": "pruned_bytes",
        "stormlog_sink_retained_files": "final_retained_files",
        "stormlog_sink_retained_bytes": "final_retained_bytes",
    }
    metrics: dict[str, Any] = {}
    for wandb_key, stats_key in metric_names.items():
        value = stats.get(stats_key)
        if isinstance(value, (int, float, bool)) and not isinstance(value, complex):
            metrics[wandb_key] = value
    return metrics


def tracking_summary_fields(
    stats: Mapping[str, Any],
    *,
    output_path: str | Path | None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for source_key, target_key in (
        ("backend", "stormlog_backend"),
        ("collector_health_status", "stormlog_collector_health_status"),
        ("collector_last_error", "stormlog_collector_last_error"),
        ("session_status", "stormlog_session_status"),
    ):
        value = stats.get(source_key)
        if value is not None:
            fields[target_key] = value

    output_file = coerce_existing_file(output_path)
    if output_file is not None:
        fields["stormlog_output_file"] = output_file.name
    return fields


def log_alerts_table(wandb: Any, run: Any, events: Sequence[Any]) -> None:
    rows: list[list[Any]] = []
    for event in events:
        event_type = event_value(event, "event_type") or event_value(event, "type")
        if event_type not in _ALERT_EVENT_TYPES:
            continue
        rows.append(
            [
                event_timestamp_seconds(event),
                event_type,
                event_value(event, "context"),
                event_int_value(event, "memory_allocated", "allocator_allocated_bytes"),
                event_int_value(event, "memory_reserved", "allocator_reserved_bytes"),
                event_int_value(event, "memory_change", "allocator_change_bytes"),
                event_value(event, "job_id"),
                event_value(event, "rank"),
            ]
        )
    if not rows:
        return
    run.log(
        {
            "stormlog_alerts": wandb.Table(
                columns=[
                    "timestamp_s",
                    "event_type",
                    "context",
                    "memory_allocated_bytes",
                    "memory_reserved_bytes",
                    "memory_change_bytes",
                    "job_id",
                    "rank",
                ],
                data=rows[-250:],
            )
        }
    )


def log_tracking_time_series(run: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        payload = {
            "stormlog_timeline_elapsed_seconds": row["elapsed_seconds"],
            "stormlog_timeline_allocated_bytes": row["allocated_bytes"],
            "stormlog_timeline_reserved_bytes": row["reserved_bytes"],
            "stormlog_timeline_change_bytes": row["change_bytes"],
            "stormlog_timeline_device_used_bytes": row["device_used_bytes"],
            "stormlog_timeline_utilization_percent": row["utilization_percent"],
        }
        filtered_payload = {
            key: value for key, value in payload.items() if value is not None
        }
        if filtered_payload:
            run.log(filtered_payload)


def log_tracking_visualizations(
    wandb: Any,
    run: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    session_slug: str,
    dashboard_root: Path | None,
    allow_artifact_logging: bool,
) -> dict[str, Any]:
    if not rows:
        return {}

    run.log(
        {
            "stormlog_memory_timeline_table": wandb.Table(
                columns=[
                    "sample_index",
                    "elapsed_seconds",
                    "event_type",
                    "memory_allocated_bytes",
                    "memory_reserved_bytes",
                    "memory_change_bytes",
                    "device_used_bytes",
                    "utilization_percent",
                    "context",
                    "rank",
                ],
                data=[
                    [
                        row["sample_index"],
                        row["elapsed_seconds"],
                        row["event_type"],
                        row["allocated_bytes"],
                        row["reserved_bytes"],
                        row["change_bytes"],
                        row["device_used_bytes"],
                        row["utilization_percent"],
                        row["context"],
                        row["rank"],
                    ]
                    for row in rows
                ],
            )
        }
    )

    plot_api = getattr(wandb, "plot", None)
    line_series = getattr(plot_api, "line_series", None)
    if callable(line_series):
        elapsed = [float(row["elapsed_seconds"]) for row in rows]
        keys, ys = _memory_plot_series(rows)
        if keys:
            run.log(
                {
                    "stormlog_memory_timeline_plot": line_series(
                        xs=elapsed,
                        ys=ys,
                        keys=keys,
                        title="Stormlog Memory Timeline",
                        xname="Elapsed Seconds",
                    )
                }
            )

        utilization_series = _series_for_plot(rows, "utilization_percent")
        if any(not math.isnan(value) for value in utilization_series):
            run.log(
                {
                    "stormlog_memory_utilization_plot": line_series(
                        xs=elapsed,
                        ys=[utilization_series],
                        keys=["utilization_percent"],
                        title="Stormlog Memory Utilization",
                        xname="Elapsed Seconds",
                    )
                }
            )

    dashboard_html = tracking_dashboard_html(rows, alert_event_types=_ALERT_EVENT_TYPES)
    run.log({"stormlog_tracking_dashboard": wandb.Html(dashboard_html)})

    if not allow_artifact_logging:
        return {}

    dashboard_path = materialize_html_file(
        html_text=dashboard_html,
        file_name="stormlog_tracking_dashboard.html",
        output_root=dashboard_root,
    )
    log_file_artifact(
        wandb,
        run,
        artifact_name=f"stormlog-tracking-dashboard-{session_slug}",
        artifact_type="stormlog-tracking-dashboard",
        path=dashboard_path,
    )
    return {"stormlog_tracking_dashboard_file": dashboard_path.name}


def tracking_timeline_rows(events: Sequence[Any]) -> list[dict[str, Any]]:
    timeline_rows: list[dict[str, Any]] = []
    first_timestamp: float | None = None

    for event in events:
        timestamp_s = event_timestamp_seconds(event)
        if timestamp_s is None:
            continue
        if first_timestamp is None:
            first_timestamp = timestamp_s

        allocated = event_int_value(
            event, "memory_allocated", "allocator_allocated_bytes"
        )
        reserved = event_int_value(event, "memory_reserved", "allocator_reserved_bytes")
        change = event_int_value(event, "memory_change", "allocator_change_bytes")
        device_used = event_int_value(event, "device_used", "device_used_bytes")
        device_total = event_int_value(event, "device_total", "device_total_bytes")

        device_used, utilization_percent = _device_usage(
            device_used, device_total, allocated, reserved
        )

        timeline_rows.append(
            {
                "sample_index": len(timeline_rows),
                "elapsed_seconds": timestamp_s - first_timestamp,
                "event_type": str(
                    event_value(event, "event_type")
                    or event_value(event, "type")
                    or "sample"
                ),
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "change_bytes": change,
                "device_used_bytes": device_used,
                "utilization_percent": utilization_percent,
                "context": event_value(event, "context"),
                "rank": event_value(event, "rank"),
            }
        )

    return sample_timeline_rows(timeline_rows)


def sample_timeline_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= _TIMELINE_MAX_POINTS:
        return list(rows)

    pinned_by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_index = row.get("sample_index")
        if (
            isinstance(sample_index, int)
            and row.get("event_type") in _ALERT_EVENT_TYPES
        ):
            pinned_by_index[sample_index] = row

    last_row = rows[-1]
    last_index = last_row.get("sample_index")
    if isinstance(last_index, int):
        pinned_by_index[last_index] = last_row

    pinned_rows = sorted(
        pinned_by_index.values(),
        key=lambda row: int(row["sample_index"]),
    )
    remaining_budget = _TIMELINE_MAX_POINTS - len(pinned_rows)
    if remaining_budget <= 0:
        return pinned_rows[-_TIMELINE_MAX_POINTS:]

    pinned_indices = set(pinned_by_index)
    unpinned_rows = [
        row for row in rows if row.get("sample_index") not in pinned_indices
    ]
    stride = max(1, math.ceil(len(unpinned_rows) / remaining_budget))
    sampled_rows = unpinned_rows[::stride][:remaining_budget]
    return sorted(
        [*pinned_rows, *sampled_rows],
        key=lambda row: int(row["sample_index"]),
    )[:_TIMELINE_MAX_POINTS]


def _memory_plot_series(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[list[float]]]:
    keys: list[str] = []
    series: list[list[float]] = []
    for row_key, label in (
        ("allocated_bytes", "allocated_bytes"),
        ("reserved_bytes", "reserved_bytes"),
        ("device_used_bytes", "device_used_bytes"),
    ):
        values = _series_for_plot(rows, row_key)
        if any(not math.isnan(value) for value in values):
            keys.append(label)
            series.append(values)
    return keys, series


def _series_for_plot(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        else:
            values.append(math.nan)
    return values


def event_value(event: Any, name: str) -> Any:
    if isinstance(event, Mapping):
        return event.get(name)
    return getattr(event, name, None)


def event_int_value(event: Any, *names: str) -> int | None:
    for name in names:
        value = event_value(event, name)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def event_timestamp_seconds(event: Any) -> float | None:
    value = event_value(event, "timestamp")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    value_ns = event_value(event, "timestamp_ns")
    if isinstance(value_ns, int) and not isinstance(value_ns, bool):
        return float(value_ns) / 1_000_000_000.0
    return None


def _tracking_dashboard_root(
    output_file: Path | None,
    sink_dir: Path | None,
) -> Path | None:
    if output_file is not None:
        return output_file.parent
    if sink_dir is not None:
        return sink_dir
    return None


def _log_tracking_artifacts(
    wandb: Any,
    run: Any,
    *,
    safe_session: str,
    output_file: Path | None,
    sink_dir: Path | None,
    oom_dir: Path | None,
    attribution_dir: Path | None,
    attribution_bundle_dir: str | Path | None,
) -> None:
    if output_file is not None:
        log_file_artifact(
            wandb,
            run,
            artifact_name=f"stormlog-track-output-{safe_session}",
            artifact_type="stormlog-track-output",
            path=output_file,
        )

    if sink_dir is not None:
        log_directory_artifact(
            wandb,
            run,
            artifact_name=f"stormlog-telemetry-sink-{safe_session}",
            artifact_type="stormlog-telemetry-sink",
            path=sink_dir,
        )

    if oom_dir is not None:
        log_directory_artifact(
            wandb,
            run,
            artifact_name=f"stormlog-oom-dump-{safe_session}",
            artifact_type="stormlog-oom-dump",
            path=oom_dir,
        )

    if (
        attribution_bundle_dir is not None
        and attribution_dir is not None
        and attribution_dir != oom_dir
    ):
        log_directory_artifact(
            wandb,
            run,
            artifact_name=f"stormlog-attribution-bundle-{safe_session}",
            artifact_type="stormlog-attribution-bundle",
            path=attribution_dir,
        )


def _device_usage(
    device_used: int | None,
    device_total: int | None,
    allocated: int | None,
    reserved: int | None,
) -> tuple[int | None, float | None]:
    if device_used is None:
        candidates = [value for value in (allocated, reserved) if value is not None]
        device_used = max(candidates) if candidates else None

    utilization_percent: float | None = None
    if (
        isinstance(device_used, int)
        and isinstance(device_total, int)
        and device_total > 0
    ):
        utilization_percent = (float(device_used) / float(device_total)) * 100.0

    return device_used, utilization_percent
