"""The diagnosis vocabulary and edge table the qualification labels use.

The kinds, where each may be located, the workload kinds and the causes are
#218's, imported from ``stormlog.infer.diagnosis_vocabulary``; the severity
names are ``stormlog.report``'s. What #218's later PRs add is not on
``release/dev`` yet, so it is copied from its design (v2.2 §4.4 and §4.6)
until they land, then imported: a finding's role and claim, the order of
the two severities a diagnosis finding takes, and the edge table. A test
pins the edge table to #218's PR 2 table.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from stormlog.infer.diagnosis_vocabulary import (
    CAPTURE_PAUSE,
    CAUSE_FAULT,
    CAUSE_INSTRUMENTATION,
    CAUSE_UNDETERMINED,
    CAUSE_WORKLOAD_CHANGE,
    CAUSES,
    CLIENT_ADMISSION,
    HOST_STALL,
    KIND_COMPONENTS,
    KINDS,
    KV_PREEMPTION_PRESSURE,
    LOAD_INCREASE,
    LONGER_INPUTS,
    LONGER_OUTPUTS,
    MIXED_PREFILL_INTERFERENCE,
    PREFIX_CACHE_LOSS,
    PREFIX_SHARING_DROP,
    QUEUE_SATURATION,
    RANK_DELAY,
    TRANSFER_DEGRADATION,
    WORKLOAD_KINDS,
)
from stormlog.report import SEVERITY_INFO, SEVERITY_WARNING

# The severities a diagnosis finding takes, lowest first; a fault claim is
# at ``warning``. #218 never emits the report's ``critical``, so the scorer
# refuses it as outside the vocabulary.
SEVERITIES = (SEVERITY_INFO, SEVERITY_WARNING)

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


# #218's edge table is canonical. Until its PR 2 is below this one, the
# content here is #218's PR 2 table (plan v2, E1-E6), edge for edge, under
# its name: never two contents under one name. Once it lands, this imports
# it and the copy goes. (#218's PR 1b has only E3, as kind pairs.)
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
                frozenset({"engine_core"}),
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
    "CAPTURE_PAUSE",
    "CAUSES",
    "CAUSE_FAULT",
    "CAUSE_INSTRUMENTATION",
    "CAUSE_UNDETERMINED",
    "CAUSE_WORKLOAD_CHANGE",
    "CLAIM_CONDITION",
    "CLAIM_FAULT",
    "CLAIM_OBSERVATION",
    "CLIENT_ADMISSION",
    "EDGES",
    "EDGE_TABLE_VERSION",
    "Edge",
    "HOST_STALL",
    "KINDS",
    "KIND_COMPONENTS",
    "KV_PREEMPTION_PRESSURE",
    "LOAD_INCREASE",
    "LONGER_INPUTS",
    "LONGER_OUTPUTS",
    "MIXED_PREFILL_INTERFERENCE",
    "PREFIX_CACHE_LOSS",
    "PREFIX_SHARING_DROP",
    "PRIMARY",
    "QUEUE_SATURATION",
    "RANK_DELAY",
    "SECONDARY",
    "SEVERITIES",
    "TRANSFER_DEGRADATION",
    "WORKLOAD_KINDS",
    "severity_at_least",
]
