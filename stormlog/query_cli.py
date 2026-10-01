"""Command-line interface for local Stormlog artifact queries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .correlation import DEFAULT_CORRELATION_WINDOW_NS

if TYPE_CHECKING:
    from .query import QueryStore


@dataclass(frozen=True)
class _OutputMode:
    json: bool = False
    csv: bool = False
    table: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `stormlog query` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    from . import query as query_api

    store = query_api.open(args.paths)
    output_mode = _OutputMode(
        json=bool(getattr(args, "json", False)),
        csv=bool(getattr(args, "csv", False)),
        table=bool(getattr(args, "table", False)),
    )

    try:
        return int(args.query_handler(store, args, output_mode))
    except BrokenPipeError:
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _query_runs(
    store: QueryStore, args: argparse.Namespace, output_mode: _OutputMode
) -> int:
    from . import query as query_api

    rows = [
        row.as_dict()
        for row in store.list_runs(
            query_api.RunFilter(
                run_id=args.run_id,
                session_id=args.session_id,
                job_id=args.job_id,
                rank=args.rank,
                source_namespace=args.source_namespace,
                source_ref=args.source_ref,
            )
        )
    ]
    _emit_rows(rows, output_mode, _RUN_COLUMNS)
    return 0


def _query_attachments(
    store: QueryStore, args: argparse.Namespace, output_mode: _OutputMode
) -> int:
    from . import query as query_api

    rows = [
        row.as_dict()
        for row in store.list_run_attachments(
            query_api.RunAttachmentFilter(
                run_id=args.run_id,
                session_id=args.session_id,
                job_id=args.job_id,
                rank=args.rank,
                kind=args.kind,
                source_namespace=args.source_namespace,
                source_ref=args.source_ref,
            )
        )
    ]
    _emit_rows(rows, output_mode, _ATTACHMENT_COLUMNS)
    return 0


def _query_sessions(
    store: QueryStore, args: argparse.Namespace, output_mode: _OutputMode
) -> int:
    from . import query as query_api

    rows = [
        row.as_dict()
        for row in store.list_sessions(
            query_api.SessionFilter(
                session_id=args.session_id,
                status=args.status,
                job_id=args.job_id,
                rank=args.rank,
                world_size=args.world_size,
                has_oom_bundle=args.has_oom_bundle,
                source_kind=args.source_kind,
            )
        )
    ]
    _emit_rows(rows, output_mode, _SESSION_COLUMNS)
    return 0


def _query_events(
    store: QueryStore, args: argparse.Namespace, output_mode: _OutputMode
) -> int:
    from . import query as query_api

    rows = [
        row.as_dict()
        for row in store.query_events(
            query_api.EventFilter(
                session_id=args.session_id,
                event_type=args.event_type,
                rank=args.rank,
                collector=args.collector,
                status=args.status,
                time_start_ns=args.time_start_ns,
                time_end_ns=args.time_end_ns,
                has_alert=args.has_alert,
                collector_health_status=args.collector_health_status,
                backend=args.backend,
                limit=args.limit,
            )
        )
    ]
    _emit_rows(rows, output_mode, _EVENT_COLUMNS)
    return 0


def _query_ooms(
    store: QueryStore, args: argparse.Namespace, output_mode: _OutputMode
) -> int:
    from . import query as query_api

    rows = [
        row.as_dict()
        for row in store.list_oom_bundles(
            query_api.OOMBundleFilter(
                session_id=args.session_id,
                backend=args.backend,
                reason=args.reason,
                created_after=args.created_after,
                created_before=args.created_before,
            )
        )
    ]
    _emit_rows(rows, output_mode, _OOM_COLUMNS)
    return 0


def _query_issues(
    store: QueryStore, args: argparse.Namespace, output_mode: _OutputMode
) -> int:
    from . import query as query_api

    if output_mode.csv:
        print("Error: --csv is not supported for issue queries", file=sys.stderr)
        return 2
    rows = [
        row.as_dict()
        for row in store.list_issues(
            query_api.IssueFilter(
                fingerprint_id=args.fingerprint_id,
                kind=args.kind,
                state=args.state,
                severity=args.severity,
                session_id=args.session_id,
            )
        )
    ]
    _emit_rows(rows, output_mode, _ISSUE_COLUMNS)
    return 0


def _query_summary(
    store: QueryStore, args: argparse.Namespace, output_mode: _OutputMode
) -> int:
    if output_mode.csv:
        print("Error: --csv is not supported for summary queries", file=sys.stderr)
        return 2
    rows = [
        row.as_dict() for row in store.summarize(args.metric, group_by=args.group_by)
    ]
    _emit_rows(rows, output_mode, _SUMMARY_COLUMNS)
    return 0


def _query_correlate(
    store: QueryStore, args: argparse.Namespace, output_mode: _OutputMode
) -> int:
    from . import query as query_api

    result = store.correlate(
        query_api.CorrelationFilter(
            session_id=args.session_id,
            job_id=args.job_id,
            rank=args.rank,
            scope=args.scope,
            at_ns=args.at_ns,
            record_id=args.record_id,
            window_ns=args.window_ns,
            kinds=tuple(args.kind or ()),
            limit=args.limit,
        )
    )
    if output_mode.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        _emit_rows(
            [row.as_dict() for row in result.evidence],
            output_mode,
            _CORRELATION_COLUMNS,
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stormlog query",
        description="Query local Stormlog telemetry sessions and artifact bundles.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Query targets")

    runs = subparsers.add_parser("runs", help="List run envelopes")
    runs.set_defaults(query_handler=_query_runs)
    _add_paths(runs)
    _add_output_flags(runs, include_csv=True)
    runs.add_argument("--run-id")
    runs.add_argument("--session-id", "--session", dest="session_id")
    runs.add_argument("--job-id")
    runs.add_argument("--rank", type=int)
    runs.add_argument("--source-namespace")
    runs.add_argument("--source-ref")

    attachments = subparsers.add_parser(
        "attachments",
        help="List run-indexed local and external attachments",
    )
    attachments.set_defaults(query_handler=_query_attachments)
    _add_paths(attachments)
    _add_output_flags(attachments, include_csv=True)
    attachments.add_argument("--run-id")
    attachments.add_argument("--session-id", "--session", dest="session_id")
    attachments.add_argument("--job-id")
    attachments.add_argument("--rank", type=int)
    attachments.add_argument("--kind")
    attachments.add_argument("--source-namespace")
    attachments.add_argument("--source-ref")

    sessions = subparsers.add_parser("sessions", help="List artifact sessions")
    sessions.set_defaults(query_handler=_query_sessions)
    _add_paths(sessions)
    _add_output_flags(sessions, include_csv=True)
    sessions.add_argument("--session-id")
    sessions.add_argument("--status")
    sessions.add_argument("--job-id")
    sessions.add_argument("--rank", type=int)
    sessions.add_argument("--world-size", type=int)
    sessions.add_argument("--source-kind")
    sessions.add_argument(
        "--has-oom-bundle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Filter sessions by linked OOM bundle presence.",
    )

    events = subparsers.add_parser("events", help="Query telemetry events")
    events.set_defaults(query_handler=_query_events)
    _add_paths(events)
    _add_output_flags(events, include_csv=True)
    events.add_argument("--session-id", "--session", dest="session_id")
    events.add_argument("--event-type")
    events.add_argument("--rank", type=int)
    events.add_argument("--collector")
    events.add_argument("--status")
    events.add_argument("--time-start-ns", type=int)
    events.add_argument("--time-end-ns", type=int)
    events.add_argument(
        "--has-alert",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Filter warning, critical, error, or severity-marked events.",
    )
    events.add_argument("--collector-health-status")
    events.add_argument("--backend")
    events.add_argument("--limit", type=int)

    ooms = subparsers.add_parser("ooms", help="List OOM dump bundles")
    ooms.set_defaults(query_handler=_query_ooms)
    _add_paths(ooms)
    _add_output_flags(ooms, include_csv=True)
    ooms.add_argument("--session-id", "--session", dest="session_id")
    ooms.add_argument("--backend")
    ooms.add_argument("--reason")
    ooms.add_argument("--created-after")
    ooms.add_argument("--created-before")

    issues = subparsers.add_parser("issues", help="List grouped recurring issues")
    issues.set_defaults(query_handler=_query_issues)
    _add_paths(issues)
    _add_output_flags(issues, include_csv=False)
    issues.add_argument("--fingerprint-id")
    issues.add_argument(
        "--kind",
        choices=[
            "oom",
            "collector_degradation",
            "alert",
            "hidden_memory_anomaly",
        ],
    )
    issues.add_argument(
        "--state",
        choices=["open", "resolved", "ignored", "regressed"],
    )
    issues.add_argument("--severity")
    issues.add_argument("--session-id", "--session", dest="session_id")

    summary = subparsers.add_parser("summary", help="Run built-in summaries")
    summary.set_defaults(query_handler=_query_summary)
    _add_paths(summary)
    _add_output_flags(summary, include_csv=False)
    summary.add_argument(
        "--metric",
        required=True,
        choices=[
            "session_count_by_status",
            "peak_allocator_allocated_bytes",
            "peak_allocator_reserved_bytes",
            "peak_device_used_bytes",
            "alert_count",
            "collector_degradation_transitions",
            "interrupted_sessions_with_oom_bundles",
            "hidden_memory_gap_growth",
        ],
    )
    summary.add_argument(
        "--group-by",
        choices=["session", "session-rank", "rank", "status"],
        default=None,
    )

    correlate = subparsers.add_parser(
        "correlate",
        help="Find evidence correlated with a timestamp or telemetry record",
    )
    correlate.set_defaults(query_handler=_query_correlate)
    _add_paths(correlate)
    _add_output_flags(correlate, include_csv=False)
    anchor = correlate.add_mutually_exclusive_group(required=True)
    anchor.add_argument("--at-ns", type=int)
    anchor.add_argument("--record-id")
    correlate.add_argument("--session-id", "--session", dest="session_id")
    correlate.add_argument("--job-id")
    correlate.add_argument("--rank", type=int)
    correlate.add_argument(
        "--scope",
        choices=["session", "distributed"],
        default="session",
    )
    correlate.add_argument(
        "--window-ns",
        type=int,
        default=DEFAULT_CORRELATION_WINDOW_NS,
    )
    correlate.add_argument(
        "--kind",
        action="append",
        choices=[
            "telemetry_event",
            "timeline_marker",
            "alert",
            "oom_bundle",
            "diagnose_bundle",
            "rollup_window",
            "attachment",
        ],
    )
    correlate.add_argument("--limit", type=int)
    return parser


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paths", nargs="+", help="Artifact paths to query")


def _add_output_flags(
    parser: argparse.ArgumentParser,
    *,
    include_csv: bool,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="Emit JSON rows")
    group.add_argument("--table", action="store_true", help="Emit a readable table")
    if include_csv:
        group.add_argument("--csv", action="store_true", help="Emit CSV rows")


def _emit_rows(
    rows: Sequence[Mapping[str, Any]],
    output_mode: _OutputMode,
    preferred_columns: Sequence[str],
) -> None:
    columns = _columns_for_rows(rows, preferred_columns)
    if output_mode.json:
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
    elif output_mode.csv:
        _emit_csv(rows, columns)
    else:
        print(_format_table(rows, columns))


def _columns_for_rows(
    rows: Sequence[Mapping[str, Any]],
    preferred_columns: Sequence[str],
) -> list[str]:
    discovered = {key for row in rows for key in row}
    columns = [column for column in preferred_columns if column in discovered]
    columns.extend(sorted(discovered - set(columns)))
    return columns or list(preferred_columns)


def _emit_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _format_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> str:
    if not rows:
        return "No rows."
    labels = [_label(column) for column in columns]
    widths = [
        max(len(label), *[len(_table_value(row.get(column))) for row in rows])
        for label, column in zip(labels, columns)
    ]
    header = "  ".join(
        label.ljust(width) for label, width in zip(labels, widths)
    ).rstrip()
    separator = "  ".join("-" * width for width in widths).rstrip()
    body = [
        "  ".join(
            _table_value(row.get(column)).ljust(width)
            for column, width in zip(columns, widths)
        ).rstrip()
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _table_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _label(column: str) -> str:
    return column.replace("_", " ").title()


_RUN_COLUMNS = (
    "run_id",
    "explicit",
    "title",
    "job_id",
    "started_at_ns",
    "ended_at_ns",
    "session_count",
    "attachment_count",
    "source_kind",
    "source_path",
    "source_namespace",
    "source_ref",
    "sessions",
    "ranks",
)
_ATTACHMENT_COLUMNS = (
    "run_id",
    "kind",
    "title",
    "storage",
    "session_id",
    "job_id",
    "rank",
    "source_kind",
    "source_path",
    "source_namespace",
    "source_ref",
    "url",
    "path",
)
_SESSION_COLUMNS = (
    "session_id",
    "status",
    "started_at_ns",
    "ended_at_ns",
    "host",
    "pid",
    "job_id",
    "rank",
    "local_rank",
    "world_size",
    "source_kind",
    "source_path",
    "source_count",
    "warning_count",
    "event_count",
    "oom_bundle_count",
)
_EVENT_COLUMNS = (
    "session_id",
    "timestamp_ns",
    "event_type",
    "rank",
    "local_rank",
    "world_size",
    "collector",
    "allocator_allocated_bytes",
    "allocator_reserved_bytes",
    "device_used_bytes",
    "source_kind",
    "source_path",
    "session_status",
)
_OOM_COLUMNS = (
    "bundle_path",
    "created_at_utc",
    "backend",
    "reason",
    "event_count",
    "session_id",
    "session_status",
    "exception_type",
    "exception_module",
)
_ISSUE_COLUMNS = (
    "fingerprint_id",
    "kind",
    "state",
    "severity",
    "title",
    "hit_count",
    "first_seen_ns",
    "last_seen_ns",
    "affected_sessions",
    "representative_evidence",
)
_SUMMARY_COLUMNS = (
    "metric",
    "group_by",
    "session_id",
    "rank",
    "status",
    "value",
    "timestamp_ns",
    "first_gap_bytes",
    "latest_gap_bytes",
    "peak_gap_bytes",
    "sample_count",
)
_CORRELATION_COLUMNS = (
    "confidence",
    "kind",
    "title",
    "session_id",
    "job_id",
    "rank",
    "start_ns",
    "end_ns",
    "source_kind",
    "source_path",
    "reasons",
)


__all__ = ["main"]
