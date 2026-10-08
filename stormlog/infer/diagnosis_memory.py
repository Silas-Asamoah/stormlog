"""Where the server's memory went, as a ledger of what each source says.

GPU memory is reported by different owners at different scopes: the device
(NVML), the process on it, PyTorch's caching allocator, vLLM's KV cache.
They nest under the default allocator (KV blocks within the KV pool, within
what the allocator reserved, within the process's GPU memory, within the
device), but they are sampled at different instants by different tools, so
the ledger never adds or subtracts across them. Each category says what
observed it, at what cadence and on which clock, and what its peak means; a
category nothing measures says so, and a pool no public interface exposes
is ``unsupported``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .diagnosis_context import Context
from .scrape_window import gauge_window
from .telemetry import TelemetrySample
from .vllm_telemetry import VllmScrapeRecord

OBSERVED = "observed"
NOT_COLLECTED = "not_collected"
UNSUPPORTED = "unsupported"
EXPORTER_SCOPED = "exporter_scoped"
KV_USAGE = "vllm:kv_cache_usage_perc"
NESTING = (
    "Under the default caching allocator: KV blocks allocated to running "
    "requests lie within the KV pool, within what the allocator reserved, "
    "within the process's GPU memory, within the device; runtime memory "
    "(CUDA context, NCCL, workspaces) lies outside the allocator. Allocator "
    "context verified on vLLM 0.30.0 (gpu_worker.py:312-333); where tensors "
    "are created and where the graph pool lives are not."
)

# (category, telemetry metric, scope)
_TELEMETRY = (
    ("physical_device", "device_memory_used_bytes", "gpu_device"),
    ("physical_mig_instance", "instance_memory_used_bytes", "gpu_instance"),
    ("physical_process_gpu", "process_gpu_used_bytes", "server_process"),
    ("host_process_rss", "process_rss_bytes", "server_process"),
    ("allocator_allocated", "allocator_allocated_bytes", "server_process"),
    ("allocator_reserved", "allocator_reserved_bytes", "server_process"),
)


def memory_ledger(
    context: Context, telemetry: Sequence[TelemetrySample] = ()
) -> dict[str, Any]:
    """The run's memory, category by category; never a sum."""
    entries = [
        _telemetry(category, metric, scope, telemetry)
        for category, metric, scope in _TELEMETRY
    ]
    entries.append(
        _unsupported(
            "runtime", "no public interface reports context, NCCL or workspace memory"
        )
    )
    entries.append(
        _unsupported(
            "cuda_graph_pools", "no public interface reports graph pool memory"
        )
    )
    entries.append(_kv(context))
    return {
        "entries": entries,
        "nesting": _nesting(context),
        "rule": "categories are never added or subtracted: their sources sample different instants",
    }


def _telemetry(
    category: str, metric: str, scope: str, samples: Sequence[TelemetrySample]
) -> dict[str, Any]:
    valid = [
        s
        for s in samples
        if s.metric == metric and s.state == "valid" and s.value_bytes is not None
    ]
    entry: dict[str, Any] = {
        "category": category,
        "scope": scope,
        "source": f"telemetry:{metric}",
        "unit": "bytes",
    }
    if not valid:
        return {**entry, "status": NOT_COLLECTED, "peak": None}
    peak = max(valid, key=lambda s: s.value_bytes or 0)
    return {
        **entry,
        "status": OBSERVED,
        "provenance": peak.provenance,
        "cadence_ms": peak.interval_ms,
        "clock": peak.identity.clock_domain,
        "samples": len(valid),
        "peak": {
            "scope": "sampled_max",
            "value": peak.value_bytes,
            "at_ns": peak.observed_at_ns,
        },
    }


def _unsupported(category: str, reason: str) -> dict[str, Any]:
    return {"category": category, "status": UNSUPPORTED, "reason": reason, "peak": None}


def _kv(context: Context) -> dict[str, Any]:
    """KV blocks held by running requests: the usage gauge times the
    engine's block count. In tokens these are capacity slots, not live
    tokens, and cached blocks that are free count as free."""
    entry: dict[str, Any] = {
        "category": "kv_blocks_allocated",
        "scope": "engine",
        "source": f"metrics:{KV_USAGE} x hello:num_gpu_blocks",
        "unit": "blocks",
    }
    scrapes = _scrapes(context)
    gauge = gauge_window(scrapes, KV_USAGE) if scrapes else None
    blocks = _engine_blocks(context)
    if gauge is None or gauge.max is None:
        return {**entry, "status": NOT_COLLECTED, "peak": None}
    if blocks is None:
        return {
            **entry,
            "status": NOT_COLLECTED,
            "peak": None,
            "reason": "no engine block count",
        }
    status = OBSERVED if context.metrics_from_engine else EXPORTER_SCOPED
    return {
        **entry,
        "status": status,
        "provenance": "reported",
        "clock": context.view.clock_domain,
        "samples": gauge.n,
        "peak": {"scope": "sampled_max", "value": round(gauge.max * blocks)},
        "binding": "asserted" if context.metrics_from_engine else "exporter_scoped",
    }


def _scrapes(context: Context) -> list[VllmScrapeRecord]:
    records = []
    for line in context.view.scrapes:
        try:
            records.append(VllmScrapeRecord.from_record(dict(line.raw or {})))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(records, key=lambda r: r.observed_at_ns)


def _engine_blocks(context: Context) -> int | None:
    counts = {
        epoch.config.get("num_gpu_blocks")
        for epoch in context.view.engines.values()
        if isinstance(epoch.config.get("num_gpu_blocks"), int)
    }
    return counts.pop() if len(counts) == 1 else None


def _nesting(context: Context) -> dict[str, Any]:
    flags = {
        epoch.config.get("enable_cumem_allocator")
        for epoch in context.view.engines.values()
    }
    if flags == {False}:
        return {"holds": True, "basis": NESTING}
    return {
        "holds": None,
        "basis": "unknown: the engine's hello does not say the default allocator was used",
    }


__all__ = ["memory_ledger"]
