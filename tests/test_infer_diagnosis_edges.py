"""The fixed edge table, as the qualification scorer (#221) imports it."""

from __future__ import annotations

from stormlog.infer.diagnosis_edges import (
    EDGES,
    EDGES_VERSION,
    edge_between,
    settle_order,
    table,
)


def test_the_table_is_six_edges_at_fixed_components() -> None:
    """Pinned: #221 ships this content under the same version until it
    imports the table. Any change is a new version."""
    assert EDGES_VERSION == "diagnosis_edges_v1"
    assert [
        (
            e.upstream_kind,
            *e.upstream_components,
            e.downstream_kind,
            *e.downstream_components,
        )
        for e in EDGES.values()
    ] == [
        ("capture_pause", "profiler", "host_stall", "engine_core"),
        ("host_stall", "engine_core", "queue_saturation", "scheduler"),
        ("kv_preemption_pressure", "kv_cache", "queue_saturation", "scheduler"),
        (
            "kv_preemption_pressure",
            "kv_cache",
            "mixed_prefill_interference",
            "scheduler",
        ),
        ("queue_saturation", "scheduler", "mixed_prefill_interference", "scheduler"),
        (
            "prefix_cache_loss",
            "prefix_cache",
            "mixed_prefill_interference",
            "scheduler",
        ),
    ]
    assert all(
        name == f"{e.upstream_kind}->{e.downstream_kind}" for name, e in EDGES.items()
    )
    assert [row["name"] for row in table()] == list(EDGES)


def test_edges_join_locations_not_just_kinds() -> None:
    engine_stall = ("host_stall", "engine_core")
    api_stall = ("host_stall", "api_server")
    queue = ("queue_saturation", "scheduler")
    assert edge_between(engine_stall, queue) is not None
    # A frontend stall is a different mechanism: no edge to the queue.
    assert edge_between(api_stall, queue) is None
    assert edge_between(("capture_pause", "profiler"), api_stall) is None


def test_every_kind_is_settled_after_the_kinds_upstream_of_it() -> None:
    order = settle_order()
    assert order == ["host_stall", "queue_saturation", "mixed_prefill_interference"]
    for edge in EDGES.values():
        if edge.upstream_kind in order:
            assert order.index(edge.upstream_kind) < order.index(edge.downstream_kind)
