"""Timeline widgets used by the Textual TUI."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from textual.widgets import Static

from stormlog.timeline_markers import TimelineMarker


class TimelineCanvas(Static):
    """ASCII timeline renderer for quick visual feedback."""

    def __init__(self, width: int = 72, height: int = 10, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.canvas_width = width
        self.canvas_height = height

    def render_timeline(self, timeline: dict[str, Any]) -> None:
        allocated, reserved, values = self._timeline_usage_series(timeline)
        if not values:
            self.render_placeholder(
                "No timeline data yet. Start live tracking and press Refresh."
            )
            return

        label = "Allocated" if values is allocated else "Device Used"
        numeric_values = [float(value) for value in values if value is not None]
        allocated_lines = self._build_chart_lines(label, numeric_values)
        reserved_lines = self._reserved_chart_lines(reserved)
        text = (
            "\n".join(allocated_lines + [""] + reserved_lines)
            if reserved_lines
            else "\n".join(allocated_lines)
        )
        self.update(text)

    @staticmethod
    def _timeline_usage_series(timeline: dict[str, Any]) -> tuple[Any, Any, Any]:
        allocated = timeline.get("allocated") if timeline else None
        reserved = timeline.get("reserved") if timeline else None
        device_used = timeline.get("device_used") if timeline else None
        values = (
            allocated
            if allocated and any(v is not None for v in allocated)
            else device_used
        )
        return allocated, reserved, values

    def _reserved_chart_lines(self, reserved: Any) -> list[str]:
        return (
            self._build_chart_lines(
                "Reserved", [float(value) for value in reserved if value is not None]
            )
            if reserved and any(value is not None for value in reserved)
            else []
        )

    def render_placeholder(self, message: str) -> None:
        self.update(message)

    def _build_chart_lines(self, label: str, values: Sequence[float]) -> list[str]:
        samples = self._resample(values)
        samples_mb = [v / (1024**2) for v in samples]
        if not samples_mb:
            return [f"{label}: no samples"]

        sparkline = self._generate_sparkline(samples_mb)
        max_val = max(samples_mb) if samples_mb else 0.0
        latest = samples_mb[-1] if samples_mb else 0.0

        return [
            f"{label} (max {max_val:.2f} MB, latest {latest:.2f} MB)",
            f"[{sparkline}]",
        ]

    def _resample(self, values: Sequence[float]) -> list[float]:
        if not values:
            return []
        if len(values) <= self.canvas_width:
            return list(values)

        step = len(values) / self.canvas_width
        sampled = []
        for i in range(self.canvas_width):
            idx = min(int(round(i * step)), len(values) - 1)
            sampled.append(values[idx])
        return sampled

    def _generate_sparkline(self, values: Sequence[float]) -> str:
        if not values:
            return ""
        max_val = max(values) or 1.0
        palette = " .:-=+*#%@"
        last_index = len(palette) - 1
        chars = []
        for value in values:
            ratio = min(value / max_val, 1.0)
            idx = int(ratio * last_index)
            chars.append(palette[idx])
        return "".join(chars)


class DistributedTimelineCanvas(Static):
    """ASCII renderer for comparing per-rank timeline trends."""

    def __init__(self, width: int = 72, max_ranks: int = 8, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.canvas_width = width
        self.max_ranks = max_ranks

    def render_rank_timelines(
        self,
        timelines: Mapping[int, Mapping[str, Sequence[int | None]]],
        active_rank: int | None = None,
        markers_by_rank: Mapping[int, Sequence[TimelineMarker]] | None = None,
    ) -> None:
        if not timelines:
            self.render_placeholder(
                "No distributed timelines yet. Load live or artifact data."
            )
            return

        ranks = sorted(timelines.keys())
        if active_rank in ranks:
            ordered = [active_rank] + [rank for rank in ranks if rank != active_rank]
        else:
            ordered = ranks

        chosen_ranks = ordered[: self.max_ranks]
        lines: list[str] = []
        for rank in chosen_ranks:
            lines.extend(
                self._build_rank_lines(
                    rank,
                    timelines.get(rank, {}),
                    is_active=rank == active_rank,
                    markers=markers_by_rank.get(rank, []) if markers_by_rank else [],
                )
            )

        if len(ordered) > self.max_ranks:
            lines.append(
                f"... showing {self.max_ranks}/{len(ordered)} ranks (apply filter for more)."
            )

        self.update("\n".join(lines) if lines else "No timeline samples to render.")

    def _build_rank_lines(
        self,
        rank: int,
        payload: Mapping[str, Sequence[int | None]],
        *,
        is_active: bool,
        markers: Sequence[TimelineMarker],
    ) -> list[str]:
        allocated = payload.get("allocated", [])
        values = allocated or payload.get("device_used", [])
        alloc_mb = self._sample_megabytes(values)
        if not alloc_mb:
            return []

        alloc_latest = self._latest_memory_text(values)
        alloc_max = max(alloc_mb)
        gap_latest = self._latest_memory_text(payload.get("gap", []))
        marker = "*" if is_active else " "
        if allocated:
            heading = (
                f"{marker}r{rank:02d} alloc(max={alloc_max:.1f}MB "
                f"latest={alloc_latest}) gap_latest={gap_latest}"
            )
        else:
            heading = (
                f"{marker}r{rank:02d} device-used(max={alloc_max:.1f}MB "
                f"latest={alloc_latest}) allocator=N/A"
            )
        lines = [heading, f"    [{self._generate_sparkline(alloc_mb)}]"]
        if markers:
            lines.append(f"    markers: {self._format_marker_summary(markers)}")
        return lines

    @staticmethod
    def _latest_memory_text(values: Sequence[int | None]) -> str:
        latest = values[-1] if values else None
        return f"{latest / (1024**2):.1f}MB" if latest is not None else "N/A"

    def _sample_megabytes(self, values: Sequence[int | None]) -> list[float]:
        return [
            value / (1024**2)
            for value in self._resample([float(v) for v in values if v is not None])
        ]

    def render_placeholder(self, message: str) -> None:
        self.update(message)

    def _resample(self, values: Sequence[float]) -> list[float]:
        if not values:
            return []
        if len(values) <= self.canvas_width:
            return list(values)

        step = len(values) / self.canvas_width
        sampled = []
        for index in range(self.canvas_width):
            source_index = min(int(round(index * step)), len(values) - 1)
            sampled.append(values[source_index])
        return sampled

    def _generate_sparkline(self, values: Sequence[float]) -> str:
        if not values:
            return ""
        max_value = max(values) or 1.0
        palette = " .:-=+*#%@"
        max_index = len(palette) - 1

        rendered: list[str] = []
        for value in values:
            ratio = min(max(value / max_value, 0.0), 1.0)
            rendered.append(palette[int(ratio * max_index)])
        return "".join(rendered)

    def _format_marker_summary(self, markers: Sequence[TimelineMarker]) -> str:
        rendered = []
        display_markers = sorted(
            enumerate(markers),
            key=lambda item: self._marker_display_sort_key(item[0], item[1]),
        )
        for _, marker in display_markers[:3]:
            rendered.append(f"{self._marker_token(marker)} {self._short_label(marker)}")
        if len(markers) > 3:
            rendered.append(f"+{len(markers) - 3} more")
        return " | ".join(rendered)

    def _marker_display_sort_key(
        self, original_index: int, marker: TimelineMarker
    ) -> tuple[int, int, int]:
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        return (
            severity_order.get(marker.severity, 3),
            -marker.start_ns,
            original_index,
        )

    def _marker_token(self, marker: TimelineMarker) -> str:
        if marker.severity == "critical":
            return "!"
        if marker.severity == "warning":
            return "~"
        if marker.is_interval:
            return "="
        return "i"

    def _short_label(self, marker: TimelineMarker) -> str:
        label = marker.label.strip() or marker.kind
        if len(label) <= 36:
            return label
        return f"{label[:33]}..."
