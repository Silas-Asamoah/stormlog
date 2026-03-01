"""Table widgets used by the Textual TUI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, List

from textual.widgets import DataTable

from ..distributed_diagnostics import AnomalyIndicator, RankDiagnosticsRow
from ..profiles import ProfileRow


class GPUStatsTable(DataTable):
    """Live-updating table of GPU stats."""

    def __init__(
        self,
        title: str,
        provider: Callable[[], list[dict[str, Any]]],
        refresh_interval: float = 2.0,
    ) -> None:
        super().__init__(show_header=True, zebra_stripes=True, id=f"table-{title}")
        self.title_text = title
        self.provider = provider
        self.refresh_interval = refresh_interval

    def on_mount(self) -> None:
        self.add_columns("Device", "Current (MB)", "Peak (MB)", "Reserved (MB)")
        self.refresh_rows()
        self.set_interval(self.refresh_interval, self.refresh_rows)

    def refresh_rows(self) -> None:
        stats = self.provider() or []
        self.clear()
        if not stats:
            self.add_row("N/A", "-", "-", "-")
            return

        for row in stats:
            self.add_row(
                row.get("device", "N/A"),
                f"{row.get('current', 0):.2f}",
                f"{row.get('peak', 0):.2f}",
                f"{row.get('reserved', 0):.2f}",
            )


class KeyValueTable(DataTable):
    """Simple key/value table for monitoring stats."""

    def on_mount(self) -> None:
        if not self.columns:
            self.add_columns("Metric", "Value")


class AlertHistoryTable(DataTable):
    """Table displaying recent alerts."""

    def on_mount(self) -> None:
        if not self.columns:
            self.add_columns("Time", "Type", "Message")

    def update_rows(self, events: List[dict]) -> None:
        self.clear()
        if not events:
            self.add_row("-", "-", "No alerts yet.")
            return
        for event in events:
            timestamp = event.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            else:
                timestamp_str = str(timestamp or "-")
            event_type = str(event.get("type", "-")).upper()
            message = event.get("message", "")
            self.add_row(timestamp_str, event_type, message)


class ProfileResultsTable(DataTable):
    """Reusable table for displaying profile summaries."""

    def on_mount(self) -> None:
        if not self.columns:
            self.add_columns(
                "Name",
                "Peak (MB)",
                "Δ Avg (MB)",
                "Duration (ms)",
                "Calls",
                "Recorded",
            )

    def update_rows(self, rows: List[ProfileRow]) -> None:
        self.clear()
        if not rows:
            self.add_row("No profiles", "-", "-", "-", "-", "-")
            return

        for row in rows:
            timestamp = (
                datetime.fromtimestamp(row.recorded_at).strftime("%H:%M:%S")
                if row.recorded_at
                else "-"
            )
            self.add_row(
                row.name,
                f"{row.peak_mb:.2f}",
                f"{row.delta_mb:.2f}",
                f"{row.duration_ms:.2f}",
                str(row.call_count),
                timestamp,
            )


class DistributedRankTable(DataTable):
    """Table displaying per-rank distributed diagnostics metrics."""

    def on_mount(self) -> None:
        if not self.columns:
            self.add_columns(
                "Rank",
                "Status",
                "Samples",
                "Δ Allocated",
                "Δ Reserved",
                "Gap Latest",
                "Gap Peak |abs|",
                "Anomaly",
            )

    def update_rows(self, rows: list[RankDiagnosticsRow]) -> None:
        self.clear()
        if not rows:
            self.add_row("-", "-", "-", "-", "-", "-", "-", "No rank data.")
            return

        for row in rows:
            anomaly_label = "Yes" if row.has_anomaly else "No"
            self.add_row(
                str(row.rank),
                row.availability,
                str(row.samples),
                self._format_bytes(row.allocated_delta_bytes),
                self._format_bytes(row.reserved_delta_bytes),
                self._format_bytes(row.hidden_gap_latest_bytes),
                self._format_bytes(row.hidden_gap_peak_abs_bytes),
                anomaly_label,
                key=f"rank-{row.rank}",
            )

    @staticmethod
    def rank_from_row_key(row_key: Any) -> int | None:
        raw_key = getattr(row_key, "value", row_key)
        text = str(raw_key)
        if text.startswith("rank-"):
            rank_text = text.removeprefix("rank-")
            if rank_text.isdigit():
                return int(rank_text)
        return None

    @staticmethod
    def _format_bytes(value: int) -> str:
        sign = "-" if value < 0 else ""
        absolute = abs(value)
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(absolute)
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
        return f"{sign}{size:.2f} {units[unit_index]}"


class AnomalySummaryTable(DataTable):
    """Table displaying first-cause anomaly indicators."""

    def on_mount(self) -> None:
        if not self.columns:
            self.add_columns("Indicator", "Rank", "Severity", "Time (UTC)", "Signal")

    def update_rows(self, indicators: list[AnomalyIndicator]) -> None:
        self.clear()
        if not indicators:
            self.add_row("-", "-", "-", "-", "No anomaly indicators detected.")
            return

        for indicator in indicators:
            time_label = datetime.utcfromtimestamp(
                indicator.timestamp_ns / 1e9
            ).strftime("%H:%M:%S")
            self.add_row(
                indicator.kind,
                str(indicator.rank),
                indicator.severity,
                time_label,
                indicator.signal,
            )
