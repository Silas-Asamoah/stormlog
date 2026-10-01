"""HTML dashboard helpers for Stormlog W&B exports."""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

_DASHBOARD_WIDTH = 720
_DASHBOARD_HEIGHT = 260
_DASHBOARD_PADDING = 18


def tracking_dashboard_html(
    rows: Sequence[Mapping[str, Any]],
    *,
    alert_event_types: frozenset[str],
) -> str:
    allocated_series = _numeric_series(rows, "allocated_bytes")
    reserved_series = _numeric_series(rows, "reserved_bytes")
    utilization_values = [
        float(value)
        for value in (row.get("utilization_percent") for row in rows)
        if isinstance(value, (int, float))
    ]
    alert_rows = [
        row for row in rows if str(row.get("event_type", "")) in alert_event_types
    ][-8:]

    chart_values = allocated_series + reserved_series
    chart_max = float(max(chart_values) if chart_values else 1)
    allocated_points = _svg_polyline_points(
        [float(value) for value in allocated_series],
        width=_DASHBOARD_WIDTH,
        height=_DASHBOARD_HEIGHT,
        min_value=0.0,
        max_value=chart_max,
    )
    reserved_points = _svg_polyline_points(
        [float(value) for value in reserved_series],
        width=_DASHBOARD_WIDTH,
        height=_DASHBOARD_HEIGHT,
        min_value=0.0,
        max_value=chart_max,
    )

    card_html = _dashboard_cards(
        rows, allocated_series, reserved_series, utilization_values
    )
    alerts_body = _dashboard_alerts(alert_rows)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:18px;color:#1f2937;background:#fff;}"
        ".cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px;}"
        ".card{border:1px solid #dbe3ea;border-radius:12px;padding:12px;background:#f8fafc;}"
        ".label{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;}"
        ".value{font-size:22px;font-weight:700;margin-top:6px;}"
        ".legend{display:flex;gap:18px;margin:10px 0 14px;font-size:13px;color:#475569;}"
        ".swatch{display:inline-block;width:10px;height:10px;border-radius:999px;margin-right:6px;}"
        "table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px;}"
        "th,td{padding:8px 10px;border-bottom:1px solid #e2e8f0;text-align:left;}"
        "th{color:#475569;font-weight:600;background:#f8fafc;}"
        "h2{margin:0 0 12px;font-size:20px;}"
        "p{margin:0 0 10px;color:#475569;}"
        "</style></head><body>"
        "<h2>Stormlog Tracking Dashboard</h2>"
        "<p>Sampled timeline exported to Weights & Biases from Stormlog tracking events.</p>"
        f"<div class='cards'>{card_html}</div>"
        "<svg viewBox='0 0 720 260' width='100%' role='img' aria-label='Stormlog memory timeline'>"
        "<rect x='0' y='0' width='720' height='260' fill='#ffffff' stroke='#e2e8f0' rx='12'/>"
        f"<polyline fill='none' stroke='#2563eb' stroke-width='3' points='{allocated_points}'/>"
        f"<polyline fill='none' stroke='#f97316' stroke-width='3' points='{reserved_points}'/>"
        "</svg>"
        "<div class='legend'>"
        "<span><span class='swatch' style='background:#2563eb;'></span>Allocated</span>"
        "<span><span class='swatch' style='background:#f97316;'></span>Reserved</span>"
        "</div>"
        "<table><thead><tr><th>sample</th><th>event</th><th>elapsed (s)</th><th>context</th></tr></thead>"
        f"<tbody>{alerts_body}</tbody></table>"
        "</body></html>"
    )


def _numeric_series(rows: Sequence[Mapping[str, Any]], key: str) -> list[int]:
    return [
        int(value)
        for value in (row.get(key) for row in rows)
        if isinstance(value, int) and not isinstance(value, bool)
    ]


def _svg_polyline_points(
    values: Sequence[float],
    *,
    width: int,
    height: int,
    min_value: float,
    max_value: float,
) -> str:
    if not values:
        return ""
    inner_width = float(width - (_DASHBOARD_PADDING * 2))
    inner_height = float(height - (_DASHBOARD_PADDING * 2))
    span = max(max_value - min_value, 1.0)
    point_count = max(len(values) - 1, 1)
    points: list[str] = []
    for index, value in enumerate(values):
        x = _DASHBOARD_PADDING + (float(index) / float(point_count)) * inner_width
        normalized = (float(value) - min_value) / span
        y = height - _DASHBOARD_PADDING - (normalized * inner_height)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    scaled = float(value)
    unit = "B"
    for unit in units:
        scaled /= 1024.0
        if scaled < 1024.0:
            return f"{scaled:.2f} {unit}"
    return f"{scaled:.2f} {unit}"


def _format_alert_sample_index(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_alert_elapsed_seconds(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.2f}"
    return "n/a"


def _dashboard_cards(
    rows: Sequence[Mapping[str, Any]],
    allocated_series: Sequence[int],
    reserved_series: Sequence[int],
    utilization_values: Sequence[float],
) -> str:
    cards = [
        ("samples", str(len(rows))),
        (
            "peak allocated",
            _format_bytes(max(allocated_series)) if allocated_series else "n/a",
        ),
        (
            "peak reserved",
            _format_bytes(max(reserved_series)) if reserved_series else "n/a",
        ),
        (
            "max utilization",
            f"{max(utilization_values):.1f}%" if utilization_values else "n/a",
        ),
    ]
    card_html = "".join(
        f"<div class='card'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div></div>"
        for label, value in cards
    )
    return card_html


def _dashboard_alerts(alert_rows: Sequence[Mapping[str, Any]]) -> str:
    alerts_html = "".join(
        "<tr>"
        f"<td>{html.escape(_format_alert_sample_index(row.get('sample_index')))}</td>"
        f"<td>{html.escape(str(row.get('event_type', '')))}</td>"
        f"<td>{html.escape(_format_alert_elapsed_seconds(row.get('elapsed_seconds')))}</td>"
        f"<td>{html.escape(str(row.get('context') or ''))}</td>"
        "</tr>"
        for row in alert_rows
    )
    alerts_body = (
        alerts_html or "<tr><td colspan='4'>No alert events captured.</td></tr>"
    )
    return alerts_body
