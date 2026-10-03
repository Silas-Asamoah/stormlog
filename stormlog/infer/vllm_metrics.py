"""Prometheus text parsing and the vLLM metric catalog.

The parser reads the text exposition that ``prometheus_client`` writes for
vLLM: ``# HELP`` and ``# TYPE`` lines name a family, a counter family is
named ``*_total`` with a separate ``*_created`` gauge, and a histogram family
exposes ``_bucket{le=...}``, ``_sum`` and ``_count`` samples. Nothing here
imports vLLM, and nothing here turns a scrape into a per-request number: a
scrape describes every request the engine served, from every client.

The catalog names what vLLM 0.30.0 exposes and what each series means. A
series the catalog does not name is kept under its native name, never
dropped; a name vLLM has retired is recognised and mapped to its successor.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

VERIFIED_VLLM_VERSION = "0.30.0"
VLLM_PREFIX = "vllm:"
ENGINE_LABELS = ("engine", "model_name")

_SAMPLE_LINE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>\S+)(?:\s+(?P<timestamp>-?\d+))?\s*$"
)
_LABEL = re.compile(
    r'\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"\s*,?'
)
_UNESCAPE = {"\\\\": "\\", '\\"': '"', "\\n": "\n"}

Labels = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Sample:
    """One exposed series: its full sample name, sorted labels and value."""

    name: str
    labels: Labels
    value: float


@dataclass(frozen=True)
class MetricFamily:
    """A ``# TYPE`` family and the samples that belong to it."""

    name: str
    kind: str
    help: str
    samples: tuple[Sample, ...] = ()


def parse_prometheus_text(text: str) -> dict[str, MetricFamily]:
    """Parse a text scrape into families keyed by their ``# TYPE`` name.

    A sample without a preceding ``# TYPE`` line belongs to an ``untyped``
    family of its own name. A line that is neither a comment nor a sample
    raises ``ValueError`` naming the line, so a truncated or HTML response
    never passes as an empty scrape.
    """
    kinds: dict[str, str] = {}
    helps: dict[str, str] = {}
    samples: dict[str, list[Sample]] = {}
    order: list[str] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            _parse_comment(line, kinds, helps, order)
            continue
        sample = _parse_sample(line, line_number)
        family = _family_of(sample.name, kinds)
        if family not in kinds:
            kinds[family] = "untyped"
            order.append(family)
        samples.setdefault(family, []).append(sample)
    return {
        name: MetricFamily(
            name, kinds[name], helps.get(name, ""), tuple(samples.get(name, ()))
        )
        for name in order
    }


def _parse_comment(
    line: str, kinds: dict[str, str], helps: dict[str, str], order: list[str]
) -> None:
    parts = line.split(None, 3)
    if len(parts) < 3:
        return
    marker, name = parts[1], parts[2]
    if marker == "TYPE":
        kinds[name] = parts[3].strip() if len(parts) > 3 else "untyped"
        if name not in order:
            order.append(name)
    elif marker == "HELP":
        helps[name] = parts[3] if len(parts) > 3 else ""


def _parse_sample(line: str, line_number: int) -> Sample:
    match = _SAMPLE_LINE.match(line)
    if match is None:
        raise ValueError(f"line {line_number} is not a Prometheus sample: {line!r}")
    try:
        value = float(match.group("value"))
    except ValueError as exc:
        raise ValueError(f"line {line_number} has a non-numeric value") from exc
    return Sample(
        match.group("name"), _parse_labels(match.group("labels"), line_number), value
    )


def _parse_labels(raw: str | None, line_number: int) -> Labels:
    if not raw or not raw.strip():
        return ()
    labels: dict[str, str] = {}
    position = 0
    while position < len(raw):
        match = _LABEL.match(raw, position)
        if match is None:
            raise ValueError(f"line {line_number} has malformed labels: {raw!r}")
        labels[match.group("key")] = _unescape(match.group("value"))
        position = match.end()
    return tuple(sorted(labels.items()))


def _unescape(value: str) -> str:
    return re.sub(r'\\\\|\\"|\\n', lambda m: _UNESCAPE[m.group(0)], value)


def _family_of(sample_name: str, kinds: dict[str, str]) -> str:
    """Map a sample name to its family: histogram and summary parts, else itself."""
    if sample_name in kinds:
        return sample_name
    for suffix in ("_bucket", "_sum", "_count", "_total"):
        if sample_name.endswith(suffix):
            base = sample_name[: -len(suffix)]
            if base in kinds:
                return base
    return sample_name


# --------------------------------------------------------------------------- compact
@dataclass(frozen=True)
class HistogramValue:
    """Cumulative bucket counts under their native ``le`` boundaries.

    ``sum`` or ``count`` is None when the exposition lacked that sample: it
    is kept missing, never invented as zero, so nothing differences it.
    """

    buckets: tuple[tuple[str, float], ...]
    sum: float | None
    count: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "buckets": [[le, encode_number(count)] for le, count in self.buckets],
            "sum": None if self.sum is None else encode_number(self.sum),
            "count": None if self.count is None else encode_number(self.count),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> HistogramValue:
        buckets = tuple(
            (str(le), decode_number(count)) for le, count in record.get("buckets", [])
        )
        total, count = record["sum"], record["count"]
        return cls(
            buckets,
            None if total is None else decode_number(total),
            None if count is None else decode_number(count),
        )

    @property
    def boundaries(self) -> tuple[float, ...]:
        return tuple(bucket_boundary(le) for le, _count in self.buckets)


_NON_FINITE = {"NaN": math.nan, "+Inf": math.inf, "-Inf": -math.inf}


def encode_number(value: float) -> float | str:
    """A sample value as strict JSON: NaN and infinities become strings."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return value


def decode_number(value: Any) -> float:
    """The inverse of :func:`encode_number`; other strings are an error."""
    if isinstance(value, str):
        if value in _NON_FINITE:
            return _NON_FINITE[value]
        raise ValueError(f"not a sample value: {value!r}")
    return float(value)


def bucket_boundary(le: str) -> float:
    """The numeric upper bound of a bucket; ``+Inf`` is infinity."""
    return math.inf if le in {"+Inf", "inf", "Inf"} else float(le)


@dataclass(frozen=True)
class CompactScrape:
    """One scrape with every series kept, label sets shared across series.

    ``values`` maps a family name to ``{label set id: value}``; a value is a
    float for gauges, counters and untyped series, and a ``HistogramValue``
    for histograms. Summaries keep their quantiles as a histogram-like value
    whose bucket boundaries are quantiles.
    """

    families: dict[str, str]
    label_sets: dict[str, dict[str, str]]
    values: dict[str, dict[str, float | HistogramValue]]
    helps: dict[str, str] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        values: dict[str, dict[str, Any]] = {}
        for name, by_set in self.values.items():
            values[name] = {
                set_id: (
                    value.to_record()
                    if isinstance(value, HistogramValue)
                    else encode_number(value)
                )
                for set_id, value in by_set.items()
            }
        return {
            "families": dict(self.families),
            "label_sets": {k: dict(v) for k, v in self.label_sets.items()},
            "values": values,
            "helps": dict(self.helps),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CompactScrape:
        families = {str(k): str(v) for k, v in record["families"].items()}
        values: dict[str, dict[str, float | HistogramValue]] = {}
        for name, by_set in record["values"].items():
            values[name] = {
                str(set_id): (
                    HistogramValue.from_record(value)
                    if isinstance(value, dict)
                    else decode_number(value)
                )
                for set_id, value in by_set.items()
            }
        return cls(
            families,
            {
                str(k): {str(a): str(b) for a, b in v.items()}
                for k, v in record["label_sets"].items()
            },
            values,
            {str(k): str(v) for k, v in record.get("helps", {}).items()},
        )

    def series(self, name: str) -> dict[str, float | HistogramValue]:
        """Values of one family by label set id; empty when absent."""
        return self.values.get(name, {})

    def labels(self, set_id: str) -> dict[str, str]:
        return self.label_sets.get(set_id, {})

    def label_values(self, label: str) -> tuple[str, ...]:
        """Every distinct value of one label, sorted."""
        return tuple(
            sorted(
                {
                    labels[label]
                    for labels in self.label_sets.values()
                    if label in labels
                }
            )
        )

    def scalar(self, name: str, set_id: str) -> float | None:
        value = self.values.get(name, {}).get(set_id)
        return value if isinstance(value, float) else None


def compact_scrape(families: dict[str, MetricFamily]) -> CompactScrape:
    """Fold parsed families into the compact per-scrape form."""
    ids: dict[Labels, str] = {}
    label_sets: dict[str, dict[str, str]] = {}
    values: dict[str, dict[str, float | HistogramValue]] = {}
    kinds: dict[str, str] = {}
    helps: dict[str, str] = {}
    for family in families.values():
        kinds[family.name] = family.kind
        if family.help:
            helps[family.name] = family.help
        folded: dict[Labels, float | HistogramValue]
        if family.kind in {"histogram", "summary"}:
            folded = dict(_fold_distribution(family))
        else:
            folded = {sample.labels: sample.value for sample in family.samples}
        for labels, value in folded.items():
            set_id = ids.setdefault(labels, str(len(ids)))
            label_sets.setdefault(set_id, dict(labels))
            values.setdefault(family.name, {})[set_id] = value
    return CompactScrape(kinds, label_sets, values, helps)


def _fold_distribution(family: MetricFamily) -> dict[Labels, HistogramValue]:
    """Group ``_bucket``/``quantile``, ``_sum`` and ``_count`` samples per label set."""
    key = "le" if family.kind == "histogram" else "quantile"
    buckets: dict[Labels, list[tuple[str, float]]] = {}
    sums: dict[Labels, float] = {}
    counts: dict[Labels, float] = {}
    for sample in family.samples:
        if sample.name.endswith("_sum"):
            sums[sample.labels] = sample.value
        elif sample.name.endswith("_count"):
            counts[sample.labels] = sample.value
        else:
            bound = dict(sample.labels).get(key)
            if bound is None:
                continue
            rest = tuple(item for item in sample.labels if item[0] != key)
            buckets.setdefault(rest, []).append((bound, sample.value))
    # Every label set any component named, in first-seen order: a set with
    # only a sum and a count (a quantile-free summary) keeps no buckets, and
    # a missing component stays None rather than becoming a zero.
    seen = {**dict.fromkeys(buckets), **dict.fromkeys(sums), **dict.fromkeys(counts)}
    return {
        labels: HistogramValue(
            tuple(
                sorted(
                    buckets.get(labels, []), key=lambda item: bucket_boundary(item[0])
                )
            ),
            sums.get(labels),
            counts.get(labels),
        )
        for labels in seen
    }


# --------------------------------------------------------------------------- catalog
@dataclass(frozen=True)
class CatalogEntry:
    """What one vLLM series means and how Stormlog refers to it."""

    name: str
    kind: str
    field: str
    group: str
    unit: str | None
    meaning: str
    labels: tuple[str, ...] = ()
    requires: str | None = None
    provenance: str = "observed"


def _entry(
    name: str,
    kind: str,
    field_name: str,
    group: str,
    unit: str | None,
    meaning: str,
    *,
    labels: tuple[str, ...] = (),
    requires: str | None = None,
    provenance: str = "observed",
) -> CatalogEntry:
    return CatalogEntry(
        name, kind, field_name, group, unit, meaning, labels, requires, provenance
    )


RESIDENCY_NOTE = (
    "wall-clock residency in a scheduler phase, measured on the engine's clock; "
    "not execution time and not GPU time"
)

# Verified against vLLM 0.30.0 (vllm/v1/metrics/loggers.py) and a live scrape.
CATALOG_0_30: tuple[CatalogEntry, ...] = (
    # Scheduler state
    _entry(
        "vllm:num_requests_running",
        "gauge",
        "running_requests",
        "scheduler",
        "requests",
        "requests in the RUNNING state at scrape time",
    ),
    _entry(
        "vllm:num_requests_waiting",
        "gauge",
        "queue_depth",
        "scheduler",
        "requests",
        "requests in the WAITING state at scrape time",
    ),
    _entry(
        "vllm:num_requests_waiting_by_reason",
        "gauge",
        "queue_depth_by_reason",
        "scheduler",
        "requests",
        "waiting requests split by why they wait",
        labels=("reason",),
    ),
    _entry(
        "vllm:engine_sleep_state",
        "gauge",
        "engine_sleep_state",
        "scheduler",
        None,
        "1 for the engine's current sleep state",
        labels=("sleep_state",),
    ),
    _entry(
        "vllm:num_preemptions_total",
        "counter",
        "preemptions",
        "scheduler",
        "preemptions",
        "requests preempted by the scheduler",
    ),
    _entry(
        "vllm:request_num_preemptions",
        "histogram",
        "preemptions_per_request",
        "scheduler",
        "preemptions",
        "preemptions a finished request went through",
    ),
    _entry(
        "vllm:request_queue_time_seconds",
        "histogram",
        "queue_time",
        "scheduler",
        "seconds",
        f"time a finished request spent WAITING; {RESIDENCY_NOTE}",
    ),
    # KV cache
    _entry(
        "vllm:kv_cache_usage_perc",
        "gauge",
        "kv_cache_usage",
        "kv_cache",
        "fraction",
        "fraction of KV cache blocks in use at scrape time (0 to 1 despite the "
        "name); logical occupancy, not device memory",
    ),
    _entry(
        "vllm:cache_config_info",
        "gauge",
        "cache_config",
        "kv_cache",
        None,
        "cache configuration carried as labels; the value is always 1",
    ),
    _entry(
        "vllm:kv_block_lifetime_seconds",
        "histogram",
        "kv_block_lifetime",
        "kv_cache",
        "seconds",
        "sampled KV block lifetime from allocation to eviction",
        requires="--kv-cache-metrics",
    ),
    _entry(
        "vllm:kv_block_idle_before_evict_seconds",
        "histogram",
        "kv_block_idle_before_evict",
        "kv_cache",
        "seconds",
        "sampled idle time of a KV block before eviction",
        requires="--kv-cache-metrics",
    ),
    _entry(
        "vllm:kv_block_reuse_gap_seconds",
        "histogram",
        "kv_block_reuse_gap",
        "kv_cache",
        "seconds",
        "sampled gap between reuses of a KV block",
        requires="--kv-cache-metrics",
    ),
    # Prefix and multimodal caches
    _entry(
        "vllm:prefix_cache_queries_total",
        "counter",
        "prefix_cache_queries",
        "prefix_cache",
        "tokens",
        "prompt tokens looked up in the local prefix cache",
    ),
    _entry(
        "vllm:prefix_cache_hits_total",
        "counter",
        "prefix_cache_hits",
        "prefix_cache",
        "tokens",
        "prompt tokens served from the local prefix cache",
    ),
    _entry(
        "vllm:external_prefix_cache_queries_total",
        "counter",
        "external_prefix_cache_queries",
        "prefix_cache",
        "tokens",
        "prompt tokens looked up in an external KV cache",
        requires="a KV connector",
    ),
    _entry(
        "vllm:external_prefix_cache_hits_total",
        "counter",
        "external_prefix_cache_hits",
        "prefix_cache",
        "tokens",
        "prompt tokens served from an external KV cache",
        requires="a KV connector",
    ),
    _entry(
        "vllm:mm_cache_queries_total",
        "counter",
        "mm_cache_queries",
        "prefix_cache",
        "items",
        "multimodal cache lookups",
    ),
    _entry(
        "vllm:mm_cache_hits_total",
        "counter",
        "mm_cache_hits",
        "prefix_cache",
        "items",
        "multimodal cache hits",
    ),
    # Tokens
    _entry(
        "vllm:prompt_tokens_total",
        "counter",
        "prompt_tokens",
        "tokens",
        "tokens",
        "prefill tokens processed",
    ),
    _entry(
        "vllm:prompt_tokens_cached_total",
        "counter",
        "prompt_tokens_cached",
        "tokens",
        "tokens",
        "prompt tokens that did not need prefill compute",
    ),
    _entry(
        "vllm:prompt_tokens_by_source_total",
        "counter",
        "prompt_tokens_by_source",
        "tokens",
        "tokens",
        "prompt tokens by where they came from",
        labels=("source",),
    ),
    _entry(
        "vllm:generation_tokens_total",
        "counter",
        "generation_tokens",
        "tokens",
        "tokens",
        "tokens generated",
    ),
    _entry(
        "vllm:iteration_tokens_total",
        "histogram",
        "iteration_tokens",
        "tokens",
        "tokens",
        "tokens scheduled per engine step (a histogram despite the suffix)",
    ),
    _entry(
        "vllm:request_success_total",
        "counter",
        "request_success",
        "requests",
        "requests",
        "finished requests by finish reason",
        labels=("finished_reason",),
    ),
    # Request latency, all residency
    _entry(
        "vllm:time_to_first_token_seconds",
        "histogram",
        "time_to_first_token",
        "request_latency",
        "seconds",
        f"arrival to first token; {RESIDENCY_NOTE}",
    ),
    _entry(
        "vllm:inter_token_latency_seconds",
        "histogram",
        "inter_token_latency",
        "request_latency",
        "seconds",
        f"gap between consecutive output tokens; {RESIDENCY_NOTE}",
    ),
    _entry(
        "vllm:request_time_per_output_token_seconds",
        "histogram",
        "time_per_output_token",
        "request_latency",
        "seconds",
        f"decode time per output token of a finished request; {RESIDENCY_NOTE}",
    ),
    _entry(
        "vllm:e2e_request_latency_seconds",
        "histogram",
        "e2e_latency",
        "request_latency",
        "seconds",
        f"arrival to finish; {RESIDENCY_NOTE}",
    ),
    _entry(
        "vllm:request_inference_time_seconds",
        "histogram",
        "inference_time",
        "request_latency",
        "seconds",
        "from the scheduler admitting the request to the engine processing "
        "its last token's output; the same expression as the OpenTelemetry "
        "attribute gen_ai.latency.time_in_model_inference; with async "
        "scheduling it can start while the previous step is still running; "
        f"{RESIDENCY_NOTE}",
    ),
    _entry(
        "vllm:request_prefill_time_seconds",
        "histogram",
        "prefill_time",
        "request_latency",
        "seconds",
        f"first schedule to first token; {RESIDENCY_NOTE}",
    ),
    _entry(
        "vllm:request_decode_time_seconds",
        "histogram",
        "decode_time",
        "request_latency",
        "seconds",
        f"first token to last token; {RESIDENCY_NOTE}",
    ),
    # Request shape
    _entry(
        "vllm:request_prompt_tokens",
        "histogram",
        "request_prompt_tokens",
        "request_shape",
        "tokens",
        "prompt length of finished requests",
    ),
    _entry(
        "vllm:request_generation_tokens",
        "histogram",
        "request_generation_tokens",
        "request_shape",
        "tokens",
        "output length of finished requests",
    ),
    _entry(
        "vllm:request_max_num_generation_tokens",
        "histogram",
        "request_max_generation_tokens",
        "request_shape",
        "tokens",
        "largest output length among a request's sequences",
    ),
    _entry(
        "vllm:request_params_n",
        "histogram",
        "request_params_n",
        "request_shape",
        None,
        "the n parameter of finished requests",
    ),
    _entry(
        "vllm:request_params_max_tokens",
        "histogram",
        "request_params_max_tokens",
        "request_shape",
        "tokens",
        "the max_tokens parameter of finished requests",
    ),
    _entry(
        "vllm:request_prefill_kv_computed_tokens",
        "histogram",
        "request_prefill_kv_computed_tokens",
        "request_shape",
        "tokens",
        "prompt tokens whose KV was computed rather than reused",
    ),
    # Speculative decoding
    _entry(
        "vllm:spec_decode_num_drafts_total",
        "counter",
        "spec_decode_drafts",
        "spec_decode",
        "drafts",
        "speculative drafts proposed",
        requires="speculative decoding",
    ),
    _entry(
        "vllm:spec_decode_num_draft_tokens_total",
        "counter",
        "spec_decode_draft_tokens",
        "spec_decode",
        "tokens",
        "draft tokens proposed",
        requires="speculative decoding",
    ),
    _entry(
        "vllm:spec_decode_num_accepted_tokens_total",
        "counter",
        "spec_decode_accepted_tokens",
        "spec_decode",
        "tokens",
        "draft tokens accepted",
        requires="speculative decoding",
    ),
    _entry(
        "vllm:spec_decode_num_accepted_tokens_per_pos_total",
        "counter",
        "spec_decode_accepted_tokens_per_position",
        "spec_decode",
        "tokens",
        "draft tokens accepted by draft position",
        labels=("position",),
        requires="speculative decoding",
    ),
    # Model FLOPs utilisation: estimates from the model's shapes, per GPU
    _entry(
        "vllm:estimated_flops_per_gpu_total",
        "counter",
        "estimated_flops_per_gpu",
        "mfu",
        "flops",
        "estimated floating point operations per GPU; advances only with "
        "--enable-mfu-metrics",
        requires="--enable-mfu-metrics",
        provenance="estimated",
    ),
    _entry(
        "vllm:estimated_read_bytes_per_gpu_total",
        "counter",
        "estimated_read_bytes_per_gpu",
        "mfu",
        "bytes",
        "estimated bytes read from memory per GPU; advances only with "
        "--enable-mfu-metrics",
        requires="--enable-mfu-metrics",
        provenance="estimated",
    ),
    _entry(
        "vllm:estimated_write_bytes_per_gpu_total",
        "counter",
        "estimated_write_bytes_per_gpu",
        "mfu",
        "bytes",
        "estimated bytes written to memory per GPU; advances only with "
        "--enable-mfu-metrics",
        requires="--enable-mfu-metrics",
        provenance="estimated",
    ),
    # Other optional families
    _entry(
        "vllm:lora_requests_info",
        "gauge",
        "lora_requests",
        "lora",
        None,
        "running LoRA adapter state carried as labels",
        labels=("max_lora", "running_lora_adapters", "waiting_lora_adapters"),
        requires="LoRA",
    ),
    _entry(
        "vllm:corrupted_requests_total",
        "counter",
        "corrupted_requests",
        "requests",
        "requests",
        "requests whose logits held NaNs",
        requires="VLLM_COMPUTE_NANS_IN_LOGITS=1",
    ),
)

CATALOG: dict[str, CatalogEntry] = {entry.name: entry for entry in CATALOG_0_30}

# Family name prefixes vLLM uses for optional subsystems; a series under one of
# them that the catalog does not name is still known to be vLLM's.
OPTIONAL_PREFIXES: dict[str, str] = {
    "vllm:kv_offload_": "kv_transfer",
    "vllm:spec_decode_": "spec_decode",
    "vllm:diffusion_": "diffusion",
}

# Names vLLM has retired, mapped to the series that carries the same meaning.
# None means the meaning is no longer exposed at all.
DEPRECATED_ALIASES: dict[str, tuple[str | None, str]] = {
    "vllm:gpu_cache_usage_perc": (
        "vllm:kv_cache_usage_perc",
        "renamed; the value is a fraction of KV blocks",
    ),
    "vllm:gpu_prefix_cache_queries_total": (
        "vllm:prefix_cache_queries_total",
        "renamed",
    ),
    "vllm:gpu_prefix_cache_hits_total": ("vllm:prefix_cache_hits_total", "renamed"),
    "vllm:time_in_queue_requests": (
        "vllm:request_queue_time_seconds",
        "renamed; both record WAITING residency per finished request",
    ),
    "vllm:model_forward_time_milliseconds": (
        None,
        "removed; no 0.30.0 series records model forward time",
    ),
    "vllm:model_execute_time_milliseconds": (
        None,
        "removed; no 0.30.0 series records model execute time",
    ),
    "vllm:num_requests_swapped": (None, "removed with the V0 engine"),
    "vllm:cpu_cache_usage_perc": (None, "removed with the V0 engine"),
    "vllm:request_params_best_of": (None, "removed with the V0 engine"),
    "vllm:kv_offload_total_bytes": (None, "replaced by load and store counters"),
    "vllm:kv_offload_total_time": (None, "replaced by load and store timers"),
    "vllm:kv_offload_size": (None, "replaced by load and store sizes"),
}


def created_family_for(name: str, kind: str = "counter") -> str | None:
    """The ``*_created`` gauge that marks when a counter or histogram was made.

    prometheus_client names a counter's stamp without the ``_total`` suffix
    (``vllm:prompt_tokens_created``) but a histogram's with its full name,
    so ``vllm:iteration_tokens_total``, a histogram despite the suffix, is
    stamped as ``vllm:iteration_tokens_total_created``.
    """
    if name.endswith("_created"):
        return None
    base = name
    if kind == "counter" and name.endswith("_total"):
        base = name[: -len("_total")]
    return f"{base}_created"


def resolve_name(name: str) -> tuple[str, str | None]:
    """Return the current catalog name and the retired alias it replaced, if any."""
    if name in CATALOG:
        return name, None
    successor, _note = DEPRECATED_ALIASES.get(name, (None, ""))
    if successor is not None:
        return successor, name
    return name, None


@dataclass(frozen=True)
class Discovery:
    """What one scrape exposes, measured against the catalog."""

    present: tuple[str, ...]
    absent: tuple[str, ...]
    optional_absent: tuple[str, ...]
    deprecated_present: tuple[str, ...]
    removed_present: tuple[str, ...]
    unknown: tuple[str, ...]
    engines: tuple[str, ...]
    model_names: tuple[str, ...]
    process_start_ns: int | None
    # Series of an optional subsystem (see OPTIONAL_PREFIXES) the catalog does
    # not name one by one: known to be vLLM's, kept raw, never "unknown".
    optional_present: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "verified_vllm_version": VERIFIED_VLLM_VERSION,
            "present": list(self.present),
            "absent": list(self.absent),
            "optional_absent": list(self.optional_absent),
            "optional_present": list(self.optional_present),
            "deprecated_present": list(self.deprecated_present),
            "removed_present": list(self.removed_present),
            "unknown": list(self.unknown),
            "engines": list(self.engines),
            "model_names": list(self.model_names),
            "process_start_ns": self.process_start_ns,
        }


def discover(scrape: CompactScrape) -> Discovery:
    """Compare a scrape with the catalog without discarding anything."""
    names = set(scrape.families)
    present, absent, optional_absent = _catalog_presence(names)
    deprecated, removed, unknown, optional_present = _classify_vllm_names(names)
    return Discovery(
        present,
        absent,
        optional_absent,
        deprecated,
        removed,
        unknown,
        scrape.label_values("engine"),
        scrape.label_values("model_name"),
        process_start_ns(scrape),
        optional_present=optional_present,
    )


def _catalog_presence(
    names: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Catalog names found, missing although expected, and missing but optional."""
    present = tuple(sorted(name for name in CATALOG if name in names))
    missing = [name for name in CATALOG if name not in names]
    absent = tuple(n for n in missing if CATALOG[n].requires is None)
    optional_absent = tuple(n for n in missing if CATALOG[n].requires is not None)
    return present, absent, optional_absent


def _classify_vllm_names(
    names: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Retired names with a successor, retired names without one, unknown
    names, and names of an optional subsystem the catalog knows by prefix."""
    deprecated: list[str] = []
    removed: list[str] = []
    unknown: list[str] = []
    optional: list[str] = []
    for name in sorted(names):
        if name.endswith("_created") or not name.startswith(VLLM_PREFIX):
            continue
        if name in DEPRECATED_ALIASES:
            successor = DEPRECATED_ALIASES[name][0]
            (deprecated if successor else removed).append(name)
        elif name in CATALOG:
            continue
        elif any(name.startswith(prefix) for prefix in OPTIONAL_PREFIXES):
            optional.append(name)
        else:
            unknown.append(name)
    return tuple(deprecated), tuple(removed), tuple(unknown), tuple(optional)


def process_start_ns(scrape: CompactScrape) -> int | None:
    """The exporting process's start time, or None when a scrape has no single one."""
    start = scrape.series("process_start_time_seconds")
    value = next(iter(start.values()), None) if len(start) == 1 else None
    return round(value * 1e9) if isinstance(value, float) else None


def family_group(name: str) -> str:
    """The catalog group of a series; optional prefixes name their subsystem."""
    entry = CATALOG.get(resolve_name(name)[0])
    if entry is not None:
        return entry.group
    for prefix, group in OPTIONAL_PREFIXES.items():
        if name.startswith(prefix):
            return group
    return "other"
