"""The closed vocabulary of inference diagnosis findings.

Every finding names one kind, located at one component, and states one cause.
The vocabulary is closed: adding a kind is a change to the diagnosis payload's
schema, not a new string a producer may invent. Online triggers, the offline
diagnoser and the qualification scorer all read these names from here.
"""

from __future__ import annotations

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

COMPONENT_CLIENT = "client"
COMPONENT_API_SERVER = "api_server"
COMPONENT_SCHEDULER = "scheduler"
COMPONENT_KV_CACHE = "kv_cache"
COMPONENT_PREFIX_CACHE = "prefix_cache"
COMPONENT_ENGINE_CORE = "engine_core"
COMPONENT_WORKER = "worker"
COMPONENT_INTERCONNECT = "interconnect"
COMPONENT_PROFILER = "profiler"
COMPONENT_WORKLOAD = "workload"

CAUSE_FAULT = "fault"
CAUSE_WORKLOAD_CHANGE = "workload_change"
CAUSE_INSTRUMENTATION = "instrumentation"
CAUSE_UNDETERMINED = "undetermined"

# Where each kind may be located. A host stall is located where the stalled
# process is; every other kind has one component.
KIND_COMPONENTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        QUEUE_SATURATION: frozenset({COMPONENT_SCHEDULER}),
        KV_PREEMPTION_PRESSURE: frozenset({COMPONENT_KV_CACHE}),
        PREFIX_CACHE_LOSS: frozenset({COMPONENT_PREFIX_CACHE}),
        MIXED_PREFILL_INTERFERENCE: frozenset({COMPONENT_SCHEDULER}),
        HOST_STALL: frozenset(
            {COMPONENT_ENGINE_CORE, COMPONENT_WORKER, COMPONENT_API_SERVER}
        ),
        RANK_DELAY: frozenset({COMPONENT_WORKER}),
        TRANSFER_DEGRADATION: frozenset({COMPONENT_INTERCONNECT}),
        CAPTURE_PAUSE: frozenset({COMPONENT_PROFILER}),
        CLIENT_ADMISSION: frozenset({COMPONENT_CLIENT}),
        LOAD_INCREASE: frozenset({COMPONENT_WORKLOAD}),
        LONGER_INPUTS: frozenset({COMPONENT_WORKLOAD}),
        LONGER_OUTPUTS: frozenset({COMPONENT_WORKLOAD}),
        PREFIX_SHARING_DROP: frozenset({COMPONENT_WORKLOAD}),
    }
)
KINDS = frozenset(KIND_COMPONENTS)
# Changes in what the workload asked for; always reported at ``info``.
WORKLOAD_KINDS = frozenset(
    {LOAD_INCREASE, LONGER_INPUTS, LONGER_OUTPUTS, PREFIX_SHARING_DROP}
)
INSTRUMENTATION_KINDS = frozenset({CAPTURE_PAUSE, CLIENT_ADMISSION})
MECHANISM_KINDS = KINDS - WORKLOAD_KINDS - INSTRUMENTATION_KINDS
COMPONENTS = frozenset().union(*KIND_COMPONENTS.values())
CAUSES = frozenset(
    {CAUSE_FAULT, CAUSE_WORKLOAD_CHANGE, CAUSE_INSTRUMENTATION, CAUSE_UNDETERMINED}
)


def check_kind(kind: str) -> str:
    """Return ``kind`` if the vocabulary has it.

    Raises:
        ValueError: for a kind outside the closed vocabulary.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown diagnosis kind: {kind!r}")
    return kind


__all__ = [
    "CAPTURE_PAUSE",
    "CAUSES",
    "CAUSE_FAULT",
    "CAUSE_INSTRUMENTATION",
    "CAUSE_UNDETERMINED",
    "CAUSE_WORKLOAD_CHANGE",
    "CLIENT_ADMISSION",
    "COMPONENTS",
    "COMPONENT_API_SERVER",
    "COMPONENT_CLIENT",
    "COMPONENT_ENGINE_CORE",
    "COMPONENT_INTERCONNECT",
    "COMPONENT_KV_CACHE",
    "COMPONENT_PREFIX_CACHE",
    "COMPONENT_PROFILER",
    "COMPONENT_SCHEDULER",
    "COMPONENT_WORKER",
    "COMPONENT_WORKLOAD",
    "HOST_STALL",
    "INSTRUMENTATION_KINDS",
    "KINDS",
    "KIND_COMPONENTS",
    "KV_PREEMPTION_PRESSURE",
    "LOAD_INCREASE",
    "LONGER_INPUTS",
    "LONGER_OUTPUTS",
    "MECHANISM_KINDS",
    "MIXED_PREFILL_INTERFERENCE",
    "PREFIX_CACHE_LOSS",
    "PREFIX_SHARING_DROP",
    "QUEUE_SATURATION",
    "RANK_DELAY",
    "TRANSFER_DEGRADATION",
    "WORKLOAD_KINDS",
    "check_kind",
]
