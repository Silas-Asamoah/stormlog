"""Timeline widgets used by the Textual TUI."""

from __future__ import annotations

from typing import Any, Sequence

from textual.widgets import Static


class TimelineCanvas(Static):
    """ASCII timeline renderer for quick visual feedback."""

    def __init__(self, width: int = 72, height: int = 10, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.canvas_width = width
        self.canvas_height = height

    def render_timeline(self, timeline: dict[str, Any]) -> None:
        allocated = timeline.get("allocated") if timeline else None
        reserved = timeline.get("reserved") if timeline else None
        if not allocated:
            self.render_placeholder(
                "No timeline data yet. Start live tracking and press Refresh."
            )
            return

        allocated_lines = self._build_chart_lines("Allocated", allocated)
        reserved_lines = (
            self._build_chart_lines("Reserved", reserved) if reserved else []
        )
        text = (
            "\n".join(allocated_lines + [""] + reserved_lines)
            if reserved_lines
            else "\n".join(allocated_lines)
        )
        self.update(text)

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
        timelines: dict[int, dict[str, list[int]]],
        active_rank: int | None = None,
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
            rank_payload = timelines.get(rank, {})
            allocated = rank_payload.get("allocated", [])
            gaps = rank_payload.get("gap", [])
            if not allocated:
                continue

            sampled_allocated = self._resample([float(value) for value in allocated])
            alloc_mb = [value / (1024**2) for value in sampled_allocated]
            alloc_latest = alloc_mb[-1] if alloc_mb else 0.0
            alloc_max = max(alloc_mb) if alloc_mb else 0.0

            sampled_gap = (
                self._resample([float(value) for value in gaps]) if gaps else []
            )
            gap_mb = [value / (1024**2) for value in sampled_gap] if sampled_gap else []
            gap_latest = gap_mb[-1] if gap_mb else 0.0
            marker = "*" if rank == active_rank else " "
            lines.append(
                f"{marker}r{rank:02d} alloc(max={alloc_max:.1f}MB latest={alloc_latest:.1f}MB) "
                f"gap_latest={gap_latest:.1f}MB"
            )
            lines.append(f"    [{self._generate_sparkline(alloc_mb)}]")

        if len(ordered) > self.max_ranks:
            lines.append(
                f"... showing {self.max_ranks}/{len(ordered)} ranks (apply filter for more)."
            )

        self.update("\n".join(lines) if lines else "No timeline samples to render.")

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
