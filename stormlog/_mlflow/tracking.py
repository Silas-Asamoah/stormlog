"""Tracking-session MLflow export helpers.

Mirrors ``stormlog._wandb.tracking`` for MLflow.  The pure timeline/metric
helpers live in the W&B exporter modules and are backend-agnostic, so they are
reused here to keep both integrations in sync.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._wandb.core import (
    coerce_existing_dir,
    coerce_existing_file,
    session_slug,
    session_summary_fields,
)
from .._wandb.dashboard import tracking_dashboard_html
from .._wandb.tracking import (
    _ALERT_EVENT_TYPES,
    _memory_plot_series,
    _series_for_plot,
    _tracking_dashboard_root,
    event_int_value,
    event_timestamp_seconds,
    event_value,
    tracking_metrics,
    tracking_summary_fields,
    tracking_timeline_rows,
)
from ..session import SessionSummary
from .attribution import log_attribution_outputs
from .core import (
    MlflowExportConfig,
    log_directory_artifact,
    log_file_artifact,
    materialize_html_file,
    new_export_artifact_prefix,
    resolve_run,
    update_summary,
)


def export_tracking_run_to_mlflow(
    config: MlflowExportConfig,
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
    """Export one completed tracking session to MLflow."""
    if not config.enabled:
        return

    timeline_rows = tracking_timeline_rows(events)
    mlflow, run, managed = resolve_run(
        config,
        command_name=command_name,
        session_summary=session_summary,
    )
    safe_session = session_slug(session_summary)
    artifact_prefix = new_export_artifact_prefix(safe_session)
    output_file = coerce_existing_file(output_path)
    sink_dir = coerce_existing_dir(telemetry_sink_dir)
    oom_dir = coerce_existing_dir(oom_dump_path)
    attribution_dir = coerce_existing_dir(attribution_bundle_dir) or oom_dir

    try:
        update_summary(
            mlflow,
            tracking_metrics(stats)
            | {"stormlog_chart_point_count": len(timeline_rows)}
            | session_summary_fields(session_summary)
            | tracking_summary_fields(stats, output_path=output_path),
        )

        log_tracking_time_series(mlflow, timeline_rows)

        if config.log_tables:
            log_alerts_table(mlflow, events, artifact_prefix=artifact_prefix)
            update_summary(
                mlflow,
                log_tracking_visualizations(
                    mlflow,
                    timeline_rows,
                    artifact_prefix=artifact_prefix,
                    session_slug=safe_session,
                    dashboard_root=_tracking_dashboard_root(output_file, sink_dir),
                    allow_artifact_logging=config.log_artifacts,
                ),
            )

        if config.log_artifacts:
            _log_tracking_artifacts(
                mlflow,
                safe_session=safe_session,
                output_file=output_file,
                sink_dir=sink_dir,
                oom_dir=oom_dir,
                attribution_dir=attribution_dir,
                attribution_bundle_dir=attribution_bundle_dir,
            )

        if config.log_attribution and attribution_dir is not None:
            update_summary(
                mlflow,
                log_attribution_outputs(
                    mlflow,
                    root=attribution_dir,
                    artifact_prefix=artifact_prefix,
                    session_slug=safe_session,
                    allow_artifact_logging=config.log_artifacts,
                ),
            )
    finally:
        if managed:
            mlflow.end_run()


def log_alerts_table(
    mlflow: Any,
    events: Sequence[Any],
    *,
    artifact_prefix: str,
) -> None:
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
    columns = [
        "timestamp_s",
        "event_type",
        "context",
        "memory_allocated_bytes",
        "memory_reserved_bytes",
        "memory_change_bytes",
        "job_id",
        "rank",
    ]
    mlflow.log_table(
        data=_rows_as_columns(columns, rows[-250:]),
        artifact_file=f"{artifact_prefix}/stormlog_alerts.json",
    )


def log_tracking_time_series(mlflow: Any, rows: Sequence[Mapping[str, Any]]) -> None:
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
            mlflow.log_metrics(filtered_payload, step=int(row["sample_index"]))


def log_tracking_visualizations(
    mlflow: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_prefix: str,
    session_slug: str,
    dashboard_root: Path | None,
    allow_artifact_logging: bool,
) -> dict[str, Any]:
    if not rows:
        return {}

    columns = [
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
    ]
    data_rows = [
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
    ]
    mlflow.log_table(
        data=_rows_as_columns(columns, data_rows),
        artifact_file=f"{artifact_prefix}/stormlog_memory_timeline.json",
    )

    log_memory_plots(mlflow, rows)

    dashboard_html = tracking_dashboard_html(rows, alert_event_types=_ALERT_EVENT_TYPES)
    mlflow.log_text(dashboard_html, artifact_file="stormlog_tracking_dashboard.html")

    if not allow_artifact_logging:
        return {}

    dashboard_path = materialize_html_file(
        html_text=dashboard_html,
        file_name="stormlog_tracking_dashboard.html",
        output_root=dashboard_root,
    )
    mlflow.log_artifact(
        local_path=str(dashboard_path),
        artifact_path=f"stormlog-tracking-dashboard-{session_slug}",
    )
    return {"stormlog_tracking_dashboard_file": dashboard_path.name}


def log_memory_plots(mlflow: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """Log matplotlib figures when matplotlib is available; otherwise skip."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    elapsed = [float(row["elapsed_seconds"]) for row in rows]

    keys, series = _memory_plot_series(rows)
    if keys:
        figure = plt.figure(figsize=(10, 4))
        for label, values in zip(keys, series):
            plt.plot(elapsed, values, label=label)
        plt.xlabel("Elapsed Seconds")
        plt.ylabel("Bytes")
        plt.title("Stormlog Memory Timeline")
        plt.legend()
        plt.tight_layout()
        mlflow.log_figure(figure, artifact_file="stormlog_memory_timeline.png")
        plt.close(figure)

    utilization_series = _series_for_plot(rows, "utilization_percent")
    if any(not _is_missing(value) for value in utilization_series):
        figure = plt.figure(figsize=(10, 4))
        plt.plot(elapsed, utilization_series)
        plt.xlabel("Elapsed Seconds")
        plt.ylabel("Utilization (%)")
        plt.title("Stormlog Memory Utilization")
        plt.tight_layout()
        mlflow.log_figure(figure, artifact_file="stormlog_memory_utilization.png")
        plt.close(figure)


def _rows_as_columns(columns: list[str], rows: list[list[Any]]) -> dict[str, list[Any]]:
    return {
        column: [row[index] if index < len(row) else None for row in rows]
        for index, column in enumerate(columns)
    }


def _is_missing(value: float) -> bool:
    return math.isnan(value)


def _log_tracking_artifacts(
    mlflow: Any,
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
            mlflow,
            artifact_path=f"stormlog-track-output-{safe_session}",
            path=output_file,
        )

    if sink_dir is not None:
        log_directory_artifact(
            mlflow,
            artifact_path=f"stormlog-telemetry-sink-{safe_session}",
            path=sink_dir,
        )

    if oom_dir is not None:
        log_directory_artifact(
            mlflow,
            artifact_path=f"stormlog-oom-dump-{safe_session}",
            path=oom_dir,
        )

    if (
        attribution_bundle_dir is not None
        and attribution_dir is not None
        and attribution_dir != oom_dir
    ):
        log_directory_artifact(
            mlflow,
            artifact_path=f"stormlog-attribution-bundle-{safe_session}",
            path=attribution_dir,
        )
