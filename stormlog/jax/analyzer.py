"""JAX Memory Analysis.

Advanced analysis tools for JAX memory profiling data.  Provides feature
parity with the PyTorch and TensorFlow analyzer modules while adapting
recommendations and heuristics to JAX-specific idioms (XLA compilation,
``jax.jit``, device memory pools, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, cast

try:
    import numpy as np
except ImportError as _np_exc:
    raise ImportError(
        "numpy is required for the JAX memory analyzer. "
        "Install it with: pip install numpy"
    ) from _np_exc

# ---------------------------------------------------------------------------
# Graceful imports for optional stormlog sub-packages
# ---------------------------------------------------------------------------

try:
    from stormlog.collective_attribution import (
        CollectiveAttributionConfig,
        CollectiveAttributionResult,
        attribute_collective_memory,
        resolve_collective_attribution_config,
    )
except ImportError:
    CollectiveAttributionConfig = Any  # type: ignore[assignment,misc]
    CollectiveAttributionResult = Any  # type: ignore[assignment,misc]

    def attribute_collective_memory(  # type: ignore[misc]
        events: Any,
        config: Any,
        phase_resolver: Any = None,
    ) -> list:
        return []

    def resolve_collective_attribution_config(  # type: ignore[misc]
        sensitivity: str,
        overrides: Any,
    ) -> Any:
        return {}


try:
    from stormlog.gap_analysis import GapFinding, analyze_hidden_memory_gaps
except ImportError:
    GapFinding = Any  # type: ignore[assignment,misc]

    def analyze_hidden_memory_gaps(  # type: ignore[misc]
        events: Any,
        thresholds: Any,
        format_memory: Any = None,
        remediation_by_classification: Any = None,
        phase_resolver: Any = None,
    ) -> list:
        return []


try:
    from stormlog.phases import (
        PhaseAttribution,
        PhaseReplayIndex,
        phase_attribution_to_payload,
    )
except ImportError:  # phase package may land in another slice
    PhaseAttribution = Any  # type: ignore[assignment,misc]
    PhaseReplayIndex = Any  # type: ignore[assignment,misc]

    def phase_attribution_to_payload(
        attribution: PhaseAttribution | None,
    ) -> dict[str, Any] | None:
        return None


try:
    from stormlog.telemetry import TelemetryEventV2
except ImportError:
    TelemetryEventV2 = Any  # type: ignore[assignment,misc]

from .utils import format_memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JAX-specific remediation guidance
# ---------------------------------------------------------------------------

_GAP_REMEDIATION_BY_CLASSIFICATION: Dict[str, List[str]] = {
    "transient_spike": [
        "Investigate non-allocator memory consumers active during spikes "
        "(XLA temporaries, collective-ops buffers, other frameworks).",
        "Use jax.live_arrays() around spike windows for detailed attribution.",
        "Consider pinning XLA workspace size via XLA_PYTHON_CLIENT_MEM_FRACTION.",
    ],
    "persistent_drift": [
        "Look for non-JAX device allocations accumulating over time "
        "(e.g. custom kernels, third-party libraries).",
        "Monitor nvidia-smi used memory alongside JAX allocator counters.",
        "If gap stabilises after warmup, it may be one-time XLA context overhead.",
    ],
    "fragmentation_like": [
        "Call jax.clear_caches() periodically to release unused XLA buffers.",
        "Reduce allocation churn by pre-allocating arrays or reusing buffers.",
        "Set XLA_PYTHON_CLIENT_PREALLOCATE=false to use a grow-only allocator.",
    ],
}


class MemoryAnalyzer:
    """Advanced analyzer for JAX memory profiling data.

    Mirrors the public interface of the PyTorch and TensorFlow
    ``MemoryAnalyzer`` classes, adapting heuristics and recommendations
    for JAX's XLA runtime and device memory model.
    """

    def __init__(
        self,
        sensitivity: float = 0.05,
        collective_sensitivity: str = "medium",
        collective_threshold_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialise the analyzer.

        Args:
            sensitivity: General sensitivity multiplier for pattern
                detection thresholds (e.g. leak slope threshold).
            collective_sensitivity: Preset sensitivity for collective-memory
                attribution (``"low"``, ``"medium"``, ``"high"``).
            collective_threshold_overrides: Optional per-threshold overrides
                for collective-memory attribution heuristics.
        """
        self.sensitivity = sensitivity
        self.collective_attribution_config: Any = resolve_collective_attribution_config(
            collective_sensitivity,
            collective_threshold_overrides,
        )

        # Hidden-memory gap analysis thresholds
        self.thresholds: Dict[str, float] = {
            "gap_ratio_threshold": 0.05,
            "gap_spike_zscore": 2.0,
            "gap_drift_r_squared": 0.6,
            "gap_fragmentation_ratio": 0.3,
        }

    # ------------------------------------------------------------------
    # Leak & pattern detection (carried over from original minimal impl)
    # ------------------------------------------------------------------

    def detect_memory_leaks(self, results: Any) -> List[Dict[str, Any]]:
        """Detect potential memory leaks in JAX telemetry.

        Uses linear regression over the memory-usage series to detect
        sustained upward drift.

        Args:
            results: Object with a ``memory_usage`` attribute (numeric
                sequence of at least 10 samples).

        Returns:
            List of leak-detection dicts with ``type``, ``severity``,
            ``description``, and ``slope`` keys.
        """
        leaks: List[Dict[str, Any]] = []
        if not hasattr(results, "memory_usage") or len(results.memory_usage) < 10:
            return leaks

        usage = np.array(results.memory_usage, dtype=float)

        # Simple linear regression to detect upward trend
        x = np.arange(len(usage))
        slope, _intercept = np.polyfit(x, usage, 1)

        # Scale threshold by sensitivity
        threshold = usage.max() * max(self.sensitivity * 0.02, 0.001)
        if slope > threshold:
            if usage[0] != 0:
                ratio_str = f"{usage[-1] / usage[0]:.1f}x"
            else:
                ratio_str = "∞x"
            leaks.append(
                {
                    "type": "leak",
                    "severity": "medium" if slope < (usage.max() * 0.01) else "high",
                    "description": (
                        f"Significant drift detected: "
                        f"{ratio_str} increase over session."
                    ),
                    "slope": float(slope),
                }
            )

        return leaks

    def detect_patterns(self, results: Any) -> List[Dict[str, Any]]:
        """Detect allocation patterns in JAX telemetry.

        Performs simplified autocorrelation analysis to identify periodic
        memory usage behaviour (e.g. per-step allocations in a training
        loop).

        Args:
            results: Object with a ``memory_usage`` attribute (numeric
                sequence of at least 10 samples).

        Returns:
            List of detected pattern dicts.
        """
        patterns: List[Dict[str, Any]] = []
        if not hasattr(results, "memory_usage") or len(results.memory_usage) < 10:
            return patterns

        usage = np.array(results.memory_usage, dtype=float)

        # Detect periodic spikes via autocorrelation secondary peaks
        centered = usage - usage.mean()
        n = len(centered)
        fft_len = 1
        while fft_len < 2 * n:
            fft_len <<= 1
        f = np.fft.rfft(centered, n=fft_len)
        autocorr = np.fft.irfft(f * np.conj(f), n=fft_len)[:n]
        center_peak = float(autocorr[0])

        if center_peak > 0:
            # Exclude immediate neighbors (±5 lags) around lag-0
            margin = min(5, n - 1)
            if margin + 1 < n:
                secondary_peak = float(autocorr[margin + 1 :].max())
            else:
                secondary_peak = 0.0

            if secondary_peak > 0.5 * center_peak:
                patterns.append(
                    {
                        "type": "periodic",
                        "description": "Strong step-to-step memory correlation detected.",
                    }
                )

        return patterns

    # ------------------------------------------------------------------
    # Fragmentation analysis
    # ------------------------------------------------------------------

    def analyze_fragmentation(self, profile_result: Any) -> Dict[str, float]:
        """Analyse memory fragmentation patterns.

        Computes fragmentation as ``1 − (used / reserved)`` across
        profiling snapshots.

        Args:
            profile_result: Profiling result with a ``snapshots`` attribute
                where each snapshot exposes ``device_memory_mb`` and
                ``device_memory_reserved_mb``.

        Returns:
            Dictionary with ``fragmentation_score``, ``trend``,
            ``max_fragmentation``, and ``min_fragmentation``.
        """
        if (
            not hasattr(profile_result, "snapshots")
            or len(profile_result.snapshots) < 2
        ):
            return {"fragmentation_score": 0.0, "trend": 0.0}

        fragmentation_scores: List[float] = []

        for snapshot in profile_result.snapshots:
            if snapshot.device_memory_reserved_mb > 0:
                utilization = (
                    snapshot.device_memory_mb / snapshot.device_memory_reserved_mb
                )
                fragmentation = 1.0 - utilization
                fragmentation_scores.append(fragmentation)

        if not fragmentation_scores:
            return {"fragmentation_score": 0.0, "trend": 0.0}

        avg_fragmentation = sum(fragmentation_scores) / len(fragmentation_scores)

        # Calculate trend
        if len(fragmentation_scores) >= 10:
            early = sum(fragmentation_scores[:5]) / 5.0
            late = sum(fragmentation_scores[-5:]) / 5.0
            trend = late - early
        else:
            trend = 0.0

        return {
            "fragmentation_score": avg_fragmentation,
            "trend": trend,
            "max_fragmentation": max(fragmentation_scores),
            "min_fragmentation": min(fragmentation_scores),
        }

    # ------------------------------------------------------------------
    # Efficiency analysis
    # ------------------------------------------------------------------

    def analyze_efficiency(self, profile_result: Any) -> float:
        """Analyse memory usage efficiency.

        Returns a score on a 0.0–1.0 scale (1.0 = excellent).  The score
        starts at 1.0 and is reduced by penalties for high peak memory,
        high growth rate, fragmentation, and detected leaks.

        Args:
            profile_result: Profiling result with ``peak_memory_mb`` and
                optionally ``memory_growth_rate``, ``snapshots``, and
                ``memory_usage``.

        Returns:
            Efficiency score in [0.0, 1.0].
        """
        if not hasattr(profile_result, "peak_memory_mb"):
            return 0.0

        score = 1.0

        # Penalise high peak memory
        if profile_result.peak_memory_mb > 8000:  # > 8 GB
            score -= 0.30
        elif profile_result.peak_memory_mb > 4000:  # > 4 GB
            score -= 0.15

        # Penalise high memory growth rate
        if hasattr(profile_result, "memory_growth_rate"):
            if profile_result.memory_growth_rate > 200:  # > 200 MB/s
                score -= 0.20
            elif profile_result.memory_growth_rate > 100:  # > 100 MB/s
                score -= 0.10

        # Penalise fragmentation
        if hasattr(profile_result, "snapshots"):
            frag_info = self.analyze_fragmentation(profile_result)
            if frag_info["fragmentation_score"] > 0.5:
                score -= 0.20
            elif frag_info["fragmentation_score"] > 0.3:
                score -= 0.10

        score = self._apply_leak_penalty(profile_result, score)

        return max(0.0, min(1.0, score))

    def _apply_leak_penalty(self, profile_result: Any, score: float) -> float:
        # Penalise memory leaks
        if hasattr(profile_result, "memory_usage") or hasattr(
            profile_result, "snapshots"
        ):

            class _SimpleTrackingResult:
                def __init__(self, memory_usage: List[float]) -> None:
                    self.memory_usage = memory_usage
                    self.timestamps = list(range(len(memory_usage)))
                    self.memory_growth_rate = 0

            if hasattr(profile_result, "snapshots") and profile_result.snapshots:
                mem_usage = [s.device_memory_bytes for s in profile_result.snapshots]
            else:
                mem_usage = getattr(profile_result, "memory_usage", [])

            simple_result = _SimpleTrackingResult(mem_usage)
            leaks = self.detect_memory_leaks(simple_result)

            high_severity_leaks = [
                leak for leak in leaks if leak.get("severity") == "high"
            ]
            if high_severity_leaks:
                score -= 0.30
            elif leaks:
                score -= 0.15

        return score

    # ------------------------------------------------------------------
    # Performance correlation
    # ------------------------------------------------------------------

    def correlate_with_performance(self, profile_result: Any) -> Dict[str, Any]:
        """Correlate memory usage with performance metrics.

        Analyses per-function efficiency based on memory consumption and
        execution duration.

        Args:
            profile_result: Profiling result with ``function_profiles``
                mapping function names to dicts containing ``calls``,
                ``total_memory_delta``, and ``total_duration``.

        Returns:
            Dictionary with ``memory_duration_correlation``,
            ``function_efficiency``, and ``recommendations``.
        """
        correlation_data: Dict[str, Any] = {
            "memory_duration_correlation": 0.0,
            "function_efficiency": {},
            "recommendations": [],
        }
        function_efficiency = cast(
            Dict[str, Dict[str, Any]], correlation_data["function_efficiency"]
        )
        recommendations = cast(List[str], correlation_data["recommendations"])

        if not hasattr(profile_result, "function_profiles"):
            return correlation_data

        for func_name, profile in profile_result.function_profiles.items():
            if profile.get("calls", 0) > 0:
                avg_memory_per_call = (
                    profile.get("total_memory_delta", 0) / profile["calls"]
                )
                avg_duration_per_call = (
                    profile.get("total_duration", 0) / profile["calls"]
                )

                # Calculate efficiency score
                efficiency = 1.0
                if avg_memory_per_call > 1024**3:  # > 1 GiB per call
                    efficiency *= 0.5
                if avg_duration_per_call > 1.0:  # > 1 second per call
                    efficiency *= 0.7

                function_efficiency[func_name] = {
                    "avg_memory_per_call": avg_memory_per_call,
                    "avg_duration_per_call": avg_duration_per_call,
                    "efficiency_score": efficiency,
                    "total_calls": profile["calls"],
                }

                # Generate recommendations
                if avg_memory_per_call > 2 * 1024**3:  # > 2 GiB per call
                    recommendations.append(
                        f"Function '{func_name}' uses high memory per call "
                        f"— consider using jax.checkpoint or reducing "
                        f"intermediate array sizes"
                    )

                if profile["calls"] > 100 and avg_duration_per_call > 0.1:
                    recommendations.append(
                        f"Function '{func_name}' called frequently "
                        f"— consider wrapping with @jax.jit"
                    )

        return correlation_data

    # ------------------------------------------------------------------
    # Optimization scoring
    # ------------------------------------------------------------------

    def score_optimization(
        self,
        profile_result: Any,
        events: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Generate an overall optimisation score with recommendations.

        Combines memory efficiency, fragmentation, and per-function
        performance scores into a single summary.

        Args:
            profile_result: JAX profiling result object.
            events: Optional telemetry event series for gap analysis.
                When provided, the result includes ``gap_analysis`` and
                ``collective_attribution`` sections.

        Returns:
            Dictionary with ``overall_score``, ``categories``,
            ``top_recommendations``, and ``priority_actions``.
        """
        optimization_score: Dict[str, Any] = {
            "overall_score": 0.0,
            "categories": {},
            "top_recommendations": [],
            "priority_actions": [],
        }
        categories = cast(Dict[str, float], optimization_score["categories"])
        priority_actions = cast(List[str], optimization_score["priority_actions"])

        # Memory efficiency (convert 0-1 → 0-10 scale for internal averaging)
        efficiency_score_01 = self.analyze_efficiency(profile_result)
        efficiency_score = efficiency_score_01 * 10.0
        categories["memory_efficiency"] = efficiency_score

        # Fragmentation analysis
        if hasattr(profile_result, "snapshots"):
            frag_info = self.analyze_fragmentation(profile_result)
            frag_score = max(0.0, 10.0 - frag_info["fragmentation_score"] * 10.0)
            categories["fragmentation"] = frag_score
        else:
            frag_score = 5.0

        # Performance correlation
        perf_corr = self.correlate_with_performance(profile_result)
        if perf_corr["function_efficiency"]:
            eff_scores = [
                func["efficiency_score"]
                for func in perf_corr["function_efficiency"].values()
            ]
            avg_efficiency = sum(eff_scores) / len(eff_scores)
            perf_score = avg_efficiency * 10.0
        else:
            perf_score = 5.0

        categories["performance"] = perf_score

        # Overall score
        optimization_score["overall_score"] = (
            efficiency_score + frag_score + perf_score
        ) / 3.0

        # Generate priority actions
        if efficiency_score < 6.0:
            priority_actions.append("Address memory efficiency issues")
        if frag_score < 6.0:
            priority_actions.append("Reduce memory fragmentation")
        if perf_score < 6.0:
            priority_actions.append("Optimise function performance")

        # Top recommendations (JAX-specific)
        top_recommendations = _suggest_jax_optimizations(profile_result)
        optimization_score["top_recommendations"] = top_recommendations[:5]

        # Hidden-memory gap analysis (only when telemetry events supplied).
        if events is not None:
            optimization_score.update(self._telemetry_analysis(events))

        return optimization_score

    # ------------------------------------------------------------------
    # Hidden-memory gap analysis (operates on TelemetryEventV2 series)
    # ------------------------------------------------------------------

    def _telemetry_analysis(self, events: List) -> Dict[str, Any]:
        optimization_score: Dict[str, Any] = {}
        phase_resolver = (
            PhaseReplayIndex.from_events(events)
            if hasattr(PhaseReplayIndex, "from_events")
            else None
        )
        gap_findings = self.analyze_memory_gaps(
            events,
            phase_resolver=phase_resolver,
        )
        collective_attribution = self.analyze_collective_attribution(
            events,
            phase_resolver=phase_resolver,
        )
        optimization_score["gap_analysis"] = [
            _serialize_gap_finding(f) for f in gap_findings
        ]
        optimization_score["collective_attribution"] = [
            _serialize_collective_attribution(result)
            for result in collective_attribution
        ]

        return optimization_score

    def analyze_memory_gaps(
        self,
        events: List,
        *,
        phase_resolver: Any | None = None,
    ) -> List:
        """Classify allocator-vs-device hidden memory gaps over time.

        Args:
            events: Chronologically ordered telemetry samples.
            phase_resolver: Optional ``PhaseReplayIndex`` for phase
                attribution.

        Returns:
            Prioritised list of gap findings (severity desc, confidence
            desc).  Returns an empty list when the ``gap_analysis``
            sub-package is not available.
        """
        return analyze_hidden_memory_gaps(
            events=events,
            thresholds=self.thresholds,
            format_memory=format_memory,
            remediation_by_classification=_GAP_REMEDIATION_BY_CLASSIFICATION,
            phase_resolver=phase_resolver,
        )

    # ------------------------------------------------------------------
    # Collective attribution
    # ------------------------------------------------------------------

    def analyze_collective_attribution(
        self,
        events: List,
        *,
        phase_resolver: Any | None = None,
    ) -> List:
        """Attribute hidden-memory spikes to collective communication phases.

        Args:
            events: Chronologically ordered telemetry samples.
            phase_resolver: Optional ``PhaseReplayIndex`` for phase
                attribution.

        Returns:
            List of ``CollectiveAttributionResult`` objects.  Returns an
            empty list when the ``collective_attribution`` sub-package is
            not available.
        """
        return attribute_collective_memory(
            events=events,
            config=self.collective_attribution_config,
            phase_resolver=phase_resolver,
        )


# ======================================================================
# Module-level helpers
# ======================================================================


def _suggest_jax_optimizations(profile_result: Any) -> List[str]:
    """Generate JAX-specific optimisation suggestions.

    Args:
        profile_result: Profiling result (duck-typed).

    Returns:
        Deduplicated list of suggestion strings (up to 10).
    """
    suggestions: List[str] = []

    if hasattr(profile_result, "peak_memory_mb"):
        peak = profile_result.peak_memory_mb
        if peak > 8000:
            suggestions.extend(
                [
                    "Consider using jax.checkpoint (rematerialisation) for large models",
                    "Enable bfloat16 mixed precision via jax.default_matmul_precision",
                    "Reduce batch size or use gradient accumulation",
                ]
            )
        elif peak > 4000:
            suggestions.extend(
                [
                    "Consider reducing batch size or using gradient accumulation",
                    "Use jax.lax.scan instead of Python loops to reduce tracing overhead",
                    "Set XLA_PYTHON_CLIENT_PREALLOCATE=false for grow-only allocation",
                ]
            )

    if (
        hasattr(profile_result, "memory_growth_rate")
        and profile_result.memory_growth_rate > 100
    ):
        suggestions.extend(
            [
                "High memory growth detected — check for leaked array references",
                "Wrap hot-path functions with @jax.jit to avoid retracing",
                "Call jax.clear_caches() periodically to free compilation artefacts",
            ]
        )

    # Always-applicable general JAX advice
    suggestions.extend(
        [
            "Use jax.lax.scan over Python for-loops for sequential computation",
            "Consider sharding large arrays with jax.sharding for multi-device setups",
            "Enable persistent compilation cache via jax.config.update("
            "'jax_compilation_cache_dir', '/tmp/jax_cache')",
        ]
    )

    return list(dict.fromkeys(suggestions))[:10]


def _serialize_gap_finding(finding: Any) -> dict[str, Any]:
    """Serialise a ``GapFinding`` dataclass to a plain dict.

    Adds ``phase_attribution`` payload when the phases package is
    available.
    """
    payload = asdict(finding)
    payload["phase_attribution"] = phase_attribution_to_payload(
        getattr(finding, "phase_attribution", None)
    )
    return payload


def _serialize_collective_attribution(result: Any) -> dict[str, Any]:
    """Serialise a ``CollectiveAttributionResult`` to a plain dict.

    Adds ``phase_attribution`` payload when the phases package is
    available.
    """
    payload = asdict(result)
    payload["phase_attribution"] = phase_attribution_to_payload(
        getattr(result, "phase_attribution", None)
    )
    return payload
