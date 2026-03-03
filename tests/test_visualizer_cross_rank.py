"""Tests for cross-rank timeline visualization."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from gpumemprof.visualizer import MemoryVisualizer
from tests.gap_test_helpers import BASE_NS, INTERVAL_NS, build_gap_event

_GB = 1024**3


def _build_cross_rank_events() -> list:
    events = []
    for rank, offset_ns, values in (
        (0, 0, [1 * _GB, 1 * _GB, 1 * _GB, int(1.34 * _GB)]),
        (1, 20_000_000, [1 * _GB, 1 * _GB, 1 * _GB, int(1.30 * _GB)]),
        (2, 40_000_000, [1 * _GB, 1 * _GB, int(1.38 * _GB), int(1.40 * _GB)]),
    ):
        for index, device_used in enumerate(values):
            allocator_reserved = max(device_used - 128 * 1024**2, 0)
            allocator_allocated = max(allocator_reserved - 64 * 1024**2, 0)
            events.append(
                build_gap_event(
                    index=index,
                    allocator_allocated=allocator_allocated,
                    allocator_reserved=allocator_reserved,
                    device_used=device_used,
                    collector="gpumemprof.cuda_tracker",
                    rank=rank,
                    local_rank=rank,
                    world_size=3,
                    job_id="viz-job",
                    host=f"host-{rank}",
                    timestamp_ns=BASE_NS + offset_ns + index * INTERVAL_NS,
                )
            )
    return events


def test_plot_cross_rank_timeline_saves_png(tmp_path) -> None:
    visualizer = MemoryVisualizer()
    output_path = tmp_path / "cross_rank_timeline.png"

    fig = visualizer.plot_cross_rank_timeline(
        events=_build_cross_rank_events(),
        save_path=str(output_path),
    )

    assert output_path.exists()
    assert fig.axes
    line_labels = [line.get_label() for line in fig.axes[0].lines]
    assert "Cluster onset" in line_labels
