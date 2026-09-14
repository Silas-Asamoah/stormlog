from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("textual")

from stormlog.timeline_markers import TimelineMarker
from stormlog.tui.widgets.timeline import DistributedTimelineCanvas


def _marker(
    *,
    start_ns: int,
    severity: str,
    label: str,
    kind: str = "alert",
    end_ns: int | None = None,
) -> TimelineMarker:
    return TimelineMarker(
        session_id="session-1",
        start_ns=start_ns,
        end_ns=end_ns,
        kind=kind,
        source="telemetry_event",
        severity=severity,
        label=label,
        metadata={},
    )


def test_marker_summary_prioritizes_severity_then_recency() -> None:
    canvas = DistributedTimelineCanvas()
    markers = [
        _marker(start_ns=100, severity="info", label="Tracking started"),
        _marker(
            start_ns=200,
            end_ns=300,
            severity="info",
            label="Phase: warmup",
            kind="phase",
        ),
        _marker(start_ns=400, severity="warning", label="Fragmentation warning"),
        _marker(start_ns=500, severity="info", label="Checkpoint saved"),
        _marker(start_ns=600, severity="critical", label="OOM detected"),
    ]

    summary = canvas._format_marker_summary(markers)

    assert summary == (
        "! OOM detected | ~ Fragmentation warning | i Checkpoint saved | +2 more"
    )


def test_marker_summary_keeps_recent_marker_within_severity() -> None:
    canvas = DistributedTimelineCanvas()
    markers = [
        _marker(start_ns=100, severity="warning", label="Older warning"),
        _marker(start_ns=300, severity="warning", label="Newer warning"),
        _marker(start_ns=200, severity="critical", label="Critical event"),
    ]

    summary = canvas._format_marker_summary(markers)

    assert summary == "! Critical event | ~ Newer warning | ~ Older warning"


def test_rank_timeline_rendering_preserves_focus_gaps_and_marker_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canvas = DistributedTimelineCanvas(width=3, max_ranks=2)
    update = Mock()
    monkeypatch.setattr(canvas, "update", update)
    mb = 1024**2

    canvas.render_rank_timelines(
        {
            0: {"allocated": [mb], "gap": []},
            1: {"allocated": [mb, 2 * mb, 3 * mb], "gap": [-mb, 0, mb]},
            2: {"allocated": [4 * mb]},
        },
        active_rank=1,
        markers_by_rank={1: [_marker(start_ns=1, severity="warning", label="Spike")]},
    )

    update.assert_called_once_with(
        "*r01 alloc(max=3.0MB latest=3.0MB) gap_latest=1.0MB\n"
        "    [-*@]\n"
        "    markers: ~ Spike\n"
        " r00 alloc(max=1.0MB latest=1.0MB) gap_latest=0.0MB\n"
        "    [@]\n"
        "... showing 2/3 ranks (apply filter for more)."
    )


@pytest.mark.parametrize(
    ("timelines", "expected"),
    [
        ({}, "No distributed timelines yet. Load live or artifact data."),
        ({0: {"allocated": []}}, "No timeline samples to render."),
    ],
)
def test_rank_timeline_empty_rendering(
    monkeypatch: pytest.MonkeyPatch,
    timelines: dict[int, dict[str, list[int]]],
    expected: str,
) -> None:
    canvas = DistributedTimelineCanvas()
    update = Mock()
    monkeypatch.setattr(canvas, "update", update)

    canvas.render_rank_timelines(timelines, active_rank=7)

    update.assert_called_once_with(expected)
