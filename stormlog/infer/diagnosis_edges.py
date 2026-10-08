"""The fixed table of edges: which finding may stand behind which.

An edge names an upstream kind at its components and a downstream kind at
its components. A downstream finding becomes the upstream's ``secondary``
only through an edge in this table, only for the same subject, and only
when the upstream finding is eligible (``diagnosis_roles``). Each edge's
evidence is computed by the downstream class, which marks the competitor
``upstream`` when that evidence reaches the edge's share.

The table is versioned: the qualification scorer (#221) imports it, and a
change to its content is a new version, never the same name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .diagnosis_vocabulary import (
    CAPTURE_PAUSE,
    COMPONENT_ENGINE_CORE,
    COMPONENT_KV_CACHE,
    COMPONENT_PREFIX_CACHE,
    COMPONENT_PROFILER,
    COMPONENT_SCHEDULER,
    HOST_STALL,
    KV_PREEMPTION_PRESSURE,
    MIXED_PREFILL_INTERFERENCE,
    PREFIX_CACHE_LOSS,
    QUEUE_SATURATION,
)

EDGES_VERSION = "diagnosis_edges_v1"


@dataclass(frozen=True)
class Edge:
    """``upstream_kind`` at one of ``upstream_components`` may stand behind
    ``downstream_kind`` at one of ``downstream_components``."""

    name: str
    upstream_kind: str
    upstream_components: frozenset[str]
    downstream_kind: str
    downstream_components: frozenset[str]

    def links(self, upstream: tuple[str, str], downstream: tuple[str, str]) -> bool:
        """Whether this edge joins the two (kind, component) locations."""
        return (
            upstream[0] == self.upstream_kind
            and upstream[1] in self.upstream_components
            and downstream[0] == self.downstream_kind
            and downstream[1] in self.downstream_components
        )


def _edge(upstream: str, at: str, downstream: str, to: str) -> Edge:
    return Edge(
        f"{upstream}->{downstream}",
        upstream,
        frozenset({at}),
        downstream,
        frozenset({to}),
    )


# In topological order: every edge's upstream is settled before any edge
# that reads its eligibility.
EDGES: Mapping[str, Edge] = MappingProxyType(
    {
        edge.name: edge
        for edge in (
            # E1: engine-loop stalls next to a profiler stop.
            _edge(CAPTURE_PAUSE, COMPONENT_PROFILER, HOST_STALL, COMPONENT_ENGINE_CORE),
            # E2: engine-loop stalls holding the waits back.
            _edge(
                HOST_STALL, COMPONENT_ENGINE_CORE, QUEUE_SATURATION, COMPONENT_SCHEDULER
            ),
            # E3: the subject's own preempted requests at the head of the queue.
            _edge(
                KV_PREEMPTION_PRESSURE,
                COMPONENT_KV_CACHE,
                QUEUE_SATURATION,
                COMPONENT_SCHEDULER,
            ),
            # E4: the treated prefill is mostly recompute after preemption.
            _edge(
                KV_PREEMPTION_PRESSURE,
                COMPONENT_KV_CACHE,
                MIXED_PREFILL_INTERFERENCE,
                COMPONENT_SCHEDULER,
            ),
            # E5: the treated prefill is mostly requests admitted from the queue.
            _edge(
                QUEUE_SATURATION,
                COMPONENT_SCHEDULER,
                MIXED_PREFILL_INTERFERENCE,
                COMPONENT_SCHEDULER,
            ),
            # E6: the treated prefill is mostly prefixes the cache lost.
            _edge(
                PREFIX_CACHE_LOSS,
                COMPONENT_PREFIX_CACHE,
                MIXED_PREFILL_INTERFERENCE,
                COMPONENT_SCHEDULER,
            ),
        )
    }
)


def settle_order() -> list[str]:
    """Downstream kinds in an order where each kind is settled after every
    kind upstream of it."""
    order: list[str] = []
    for edge in EDGES.values():
        if edge.downstream_kind not in order:
            order.append(edge.downstream_kind)
    return order


def edge_between(upstream: tuple[str, str], downstream: tuple[str, str]) -> Edge | None:
    """The table's edge joining two (kind, component) locations, if any."""
    return next((e for e in EDGES.values() if e.links(upstream, downstream)), None)


def table() -> list[dict[str, object]]:
    """The table as written into a diagnosis payload."""
    return [
        {
            "name": edge.name,
            "upstream": {
                "kind": edge.upstream_kind,
                "components": sorted(edge.upstream_components),
            },
            "downstream": {
                "kind": edge.downstream_kind,
                "components": sorted(edge.downstream_components),
            },
        }
        for edge in EDGES.values()
    ]


__all__ = ["EDGES", "EDGES_VERSION", "Edge", "edge_between", "settle_order", "table"]
