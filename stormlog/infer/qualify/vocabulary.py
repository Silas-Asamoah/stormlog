"""The diagnosis vocabulary and edge table the qualification labels use.

These are #218's names (its design v2.2, §4.4 kinds and §4.6 edges), copied
here until #218's ``diagnosis_vocabulary`` and edge table reach
``release/dev``; whichever of the two lands second makes this module import
them instead. A test compares the two when both are present.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

QUEUE_SATURATION = "queue_saturation"
KV_PREEMPTION_PRESSURE = "kv_preemption_pressure"
PREFIX_CACHE_LOSS = "prefix_cache_loss"
MIXED_PREFILL_INTERFERENCE = "mixed_prefill_interference"
HOST_STALL = "host_stall"
RANK_DELAY = "rank_delay"
TRANSFER_DEGRADATION = "transfer_degradation"
CAPTURE_PAUSE = "capture_pause"
CLIENT_ADMISSION = "client_admission"
LOAD_INCREASE = "load_increase"
LONGER_INPUTS = "longer_inputs"
LONGER_OUTPUTS = "longer_outputs"
PREFIX_SHARING_DROP = "prefix_sharing_drop"

# Where each kind may be located (#218 §4.4).
KIND_COMPONENTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        QUEUE_SATURATION: frozenset({"scheduler"}),
        KV_PREEMPTION_PRESSURE: frozenset({"kv_cache"}),
        PREFIX_CACHE_LOSS: frozenset({"prefix_cache"}),
        MIXED_PREFILL_INTERFERENCE: frozenset({"scheduler"}),
        HOST_STALL: frozenset({"engine_core", "worker", "api_server"}),
        RANK_DELAY: frozenset({"worker"}),
        TRANSFER_DEGRADATION: frozenset({"interconnect"}),
        CAPTURE_PAUSE: frozenset({"profiler"}),
        CLIENT_ADMISSION: frozenset({"client"}),
        LOAD_INCREASE: frozenset({"workload"}),
        LONGER_INPUTS: frozenset({"workload"}),
        LONGER_OUTPUTS: frozenset({"workload"}),
        PREFIX_SHARING_DROP: frozenset({"workload"}),
    }
)
KINDS = frozenset(KIND_COMPONENTS)
WORKLOAD_KINDS = frozenset(
    {LOAD_INCREASE, LONGER_INPUTS, LONGER_OUTPUTS, PREFIX_SHARING_DROP}
)

CAUSE_FAULT = "fault"
CAUSE_WORKLOAD_CHANGE = "workload_change"
CAUSE_INSTRUMENTATION = "instrumentation"
CAUSE_UNDETERMINED = "undetermined"
CAUSES = frozenset(
    {CAUSE_FAULT, CAUSE_WORKLOAD_CHANGE, CAUSE_INSTRUMENTATION, CAUSE_UNDETERMINED}
)

# #218's severities, lowest first; a fault claim is at ``warning``.
SEVERITIES = ("info", "warning")

PRIMARY = "primary"
SECONDARY = "secondary"
CLAIM_FAULT = "fault"
CLAIM_CONDITION = "condition"
CLAIM_OBSERVATION = "observation"


@dataclass(frozen=True)
class Edge:
    """A primary-to-secondary relation #218 may assert, with where each end
    may be located."""

    upstream: str
    upstream_components: frozenset[str]
    downstream: str
    downstream_components: frozenset[str]

    @property
    def name(self) -> str:
        return f"{self.upstream}->{self.downstream}"


EDGE_TABLE_VERSION = "diagnosis_edges_v1"
EDGES: Mapping[str, Edge] = MappingProxyType(
    {
        edge.name: edge
        for edge in (
            Edge(
                KV_PREEMPTION_PRESSURE,
                frozenset({"kv_cache"}),
                QUEUE_SATURATION,
                frozenset({"scheduler"}),
            ),
            Edge(
                KV_PREEMPTION_PRESSURE,
                frozenset({"kv_cache"}),
                MIXED_PREFILL_INTERFERENCE,
                frozenset({"scheduler"}),
            ),
            Edge(
                QUEUE_SATURATION,
                frozenset({"scheduler"}),
                MIXED_PREFILL_INTERFERENCE,
                frozenset({"scheduler"}),
            ),
            Edge(
                HOST_STALL,
                frozenset({"engine_core"}),
                QUEUE_SATURATION,
                frozenset({"scheduler"}),
            ),
            Edge(
                CAPTURE_PAUSE,
                frozenset({"profiler"}),
                HOST_STALL,
                frozenset({"engine_core", "worker"}),
            ),
            Edge(
                PREFIX_CACHE_LOSS,
                frozenset({"prefix_cache"}),
                MIXED_PREFILL_INTERFERENCE,
                frozenset({"scheduler"}),
            ),
        )
    }
)


def severity_at_least(severity: str, floor: str) -> bool:
    """Whether ``severity`` reaches ``floor`` in #218's order."""
    return SEVERITIES.index(severity) >= SEVERITIES.index(floor)


__all__ = [
    "CAUSES",
    "CAUSE_FAULT",
    "CAUSE_INSTRUMENTATION",
    "CAUSE_UNDETERMINED",
    "CAUSE_WORKLOAD_CHANGE",
    "CLAIM_CONDITION",
    "CLAIM_FAULT",
    "CLAIM_OBSERVATION",
    "EDGES",
    "EDGE_TABLE_VERSION",
    "Edge",
    "KINDS",
    "KIND_COMPONENTS",
    "PRIMARY",
    "SECONDARY",
    "SEVERITIES",
    "WORKLOAD_KINDS",
    "severity_at_least",
]
