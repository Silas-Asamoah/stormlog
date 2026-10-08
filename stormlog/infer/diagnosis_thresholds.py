"""The versioned threshold table shared by online triggers and the diagnoser.

A trigger and a diagnosis read the same value under the same key, so they can
never disagree about what "saturated" means. Every value here is provisional
until it has been read from real runs; a change to any value is a new table
version, and every result records the version it used and whether a caller
overrode a value.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

THRESHOLDS_VERSION = "diagnosis_thresholds_v1"

QUEUE_MEDIAN_WAITING = "queue_saturation.median_waiting_requests"
KV_PREEMPTIONS = "kv_preemption_pressure.preemptions"
PREFIX_HIT_RATIO_DROP = "prefix_cache_loss.hit_ratio_drop"
PREFIX_MIN_QUERIED = "prefix_cache_loss.min_queried_tokens"
LOOP_STALL_FACTOR = "host_stall.stall_factor"
LOOP_STALL_FLOOR_NS = "host_stall.stall_floor_ns"
LOOP_NO_BASELINE_FLOOR_NS = "host_stall.no_baseline_floor_ns"
LOOP_BASELINE_WINDOW_NS = "host_stall.baseline_window_ns"
LOOP_MIN_BUSY_STEPS = "host_stall.min_busy_steps"
LOOP_MATCHED_BIN_MIN_STEPS = "host_stall.matched_bin_min_steps"
LOOP_HEARTBEAT_GRACE_NS = "host_stall.heartbeat_grace_ns"
QUEUE_WITNESS_SHARE = "queue_saturation.witness_step_share"
QUEUE_CONTRIBUTION = "queue_saturation.ttft_excess_share"
QUEUE_STALL_SHARE = "queue_saturation.stall_excess_share"
QUEUE_FRONT_SHARE = "queue_saturation.front_excess_share"
QUEUE_COMPETITOR_FLOOR = "queue_saturation.competitor_floor_share"
WORKLOAD_RATE_RATIO = "load_increase.arrival_rate_ratio"
WORKLOAD_LENGTH_RATIO = "workload.length_ratio"
WORKLOAD_SHARE_DROP = "prefix_sharing_drop.share_drop"
SELECTION_WINDOW_S = "selection.window_seconds"
SELECTION_SPAN_CAP_S = "selection.span_cap_seconds"
SELECTION_MIN_REQUESTS = "selection.min_requests"
SELECTION_REFERENCE_MIN = "selection.reference_min_requests"
SELECTION_ALPHA = "selection.alpha"
SELECTION_MIN_ABOVE = "selection.min_above"

DEFAULT_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        # Requests waiting in the median scrape of the window.
        QUEUE_MEDIAN_WAITING: 1.0,
        # Preemptions counted in the window: any preemption is evidence.
        KV_PREEMPTIONS: 1.0,
        # Fall of the prefix-cache hit ratio below the caller's reference.
        PREFIX_HIT_RATIO_DROP: 0.2,
        # Tokens the window must have queried the cache for before its ratio
        # decides: vLLM counts every prompt token of a new request as a
        # query, so one short, unseen prompt alone has a ratio of 0.
        PREFIX_MIN_QUERIED: 2048.0,
        # An engine-loop stall is at least this many times the median
        # completion cadence of the busy steps before it...
        LOOP_STALL_FACTOR: 10.0,
        # ...and at least this long,
        LOOP_STALL_FLOOR_NS: 50_000_000.0,
        # or, with no such steps to compare with, at least this long.
        LOOP_NO_BASELINE_FLOOR_NS: 500_000_000.0,
        # The busy steps compared with lie in this window before the stall,
        LOOP_BASELINE_WINDOW_NS: 30_000_000_000.0,
        # and there must be this many of them,
        LOOP_MIN_BUSY_STEPS: 20.0,
        # or this many of the stall's own work bucket (scheduled tokens within
        # a factor of two) to compare it with steps of its size.
        LOOP_MATCHED_BIN_MIN_STEPS: 20.0,
        # A stall still going on is judged only while the hook's writer was
        # heard from this recently: three of its one-second heartbeats, since
        # under load they slip (2.3 s apart on a real vLLM 0.30.0 run).
        LOOP_HEARTBEAT_GRACE_NS: 3_000_000_000.0,
        # A capacity witness: the share of steps spanning the waits that ran
        # at max_num_seqs or at the max_num_batched_tokens budget.
        QUEUE_WITNESS_SHARE: 0.5,
        # The queue explains the incident when its wait excess is at least
        # this share of the TTFT excess.
        QUEUE_CONTRIBUTION: 0.5,
        # Engine stalls during or just before the waits explain them instead
        # when they last this share of the wait excess: a stall holds a
        # request back by no more than its own length.
        QUEUE_STALL_SHARE: 0.5,
        # Time before the queue (engine ingress, the API server) explains the
        # excess instead when its own excess is this share of the wait's.
        QUEUE_FRONT_SHARE: 0.25,
        # Below this share of the wait excess a competitor is ruled out;
        # between it and the competitor's own cut it is contributing, which
        # leaves the queue no fault claim.
        QUEUE_COMPETITOR_FLOOR: 0.1,
        # Workload changes: arrivals faster by this ratio (interval's lower
        # bound), prompts or outputs longer by this ratio, or this much less
        # of the requests declaring a shared prefix.
        WORKLOAD_RATE_RATIO: 1.25,
        WORKLOAD_LENGTH_RATIO: 1.1,
        WORKLOAD_SHARE_DROP: 0.1,
        # Incident selection: base windows of 1 s, joined until each holds
        # 20 requests or spans 30 s.
        SELECTION_WINDOW_S: 1.0,
        SELECTION_SPAN_CAP_S: 30.0,
        SELECTION_MIN_REQUESTS: 20.0,
        # Reference requests needed before a window is tested: the fewest
        # that bound a p90 under the shared sufficiency rule (#213's
        # quantile_minimum_n(0.90) with margin 5 at 95%).
        SELECTION_REFERENCE_MIN: 114.0,
        # One-sided Fisher's exact test level, and the fewest requests above
        # the reference p90 a flagged window must hold.
        SELECTION_ALPHA: 0.01,
        SELECTION_MIN_ABOVE: 3.0,
    }
)


def resolve_threshold(
    key: str, overrides: Mapping[str, float] | None = None
) -> tuple[float, bool]:
    """The threshold for ``key`` and whether a caller overrode the table.

    Raises:
        KeyError: for a key the table does not have.
    """
    if overrides and key in overrides:
        return float(overrides[key]), True
    return DEFAULT_THRESHOLDS[key], False


__all__ = [
    "DEFAULT_THRESHOLDS",
    "KV_PREEMPTIONS",
    "LOOP_BASELINE_WINDOW_NS",
    "LOOP_HEARTBEAT_GRACE_NS",
    "LOOP_MATCHED_BIN_MIN_STEPS",
    "LOOP_MIN_BUSY_STEPS",
    "LOOP_NO_BASELINE_FLOOR_NS",
    "LOOP_STALL_FACTOR",
    "LOOP_STALL_FLOOR_NS",
    "PREFIX_HIT_RATIO_DROP",
    "PREFIX_MIN_QUERIED",
    "QUEUE_COMPETITOR_FLOOR",
    "QUEUE_CONTRIBUTION",
    "QUEUE_FRONT_SHARE",
    "QUEUE_MEDIAN_WAITING",
    "QUEUE_STALL_SHARE",
    "QUEUE_WITNESS_SHARE",
    "SELECTION_ALPHA",
    "SELECTION_MIN_ABOVE",
    "SELECTION_MIN_REQUESTS",
    "SELECTION_REFERENCE_MIN",
    "SELECTION_SPAN_CAP_S",
    "SELECTION_WINDOW_S",
    "THRESHOLDS_VERSION",
    "WORKLOAD_LENGTH_RATIO",
    "WORKLOAD_RATE_RATIO",
    "WORKLOAD_SHARE_DROP",
    "resolve_threshold",
]
