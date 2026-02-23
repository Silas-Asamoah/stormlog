"""Advanced analysis tools for memory profiling data."""

import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from .gap_analysis import GapFinding, analyze_hidden_memory_gaps
from .profiler import GPUMemoryProfiler, ProfileResult
from .telemetry import TelemetryEventV2
from .utils import format_bytes


@dataclass
class MemoryPattern:
    """Represents a detected memory usage pattern."""

    pattern_type: str
    description: str
    severity: str  # 'info', 'warning', 'critical'
    affected_functions: List[str]
    metrics: Dict[str, Any]
    suggestions: List[str]


@dataclass
class PerformanceInsight:
    """Performance insight derived from profiling data."""

    category: str
    title: str
    description: str
    impact: str  # 'low', 'medium', 'high'
    confidence: float  # 0.0 to 1.0
    data: Dict[str, Any]
    recommendations: List[str]


_GAP_REMEDIATION_BY_CLASSIFICATION: Dict[str, List[str]] = {
    "transient_spike": [
        "Investigate non-allocator memory consumers active during spikes "
        "(cuDNN workspace, NCCL buffers, other frameworks).",
        "Use torch.cuda.memory_snapshot() around spike windows for detailed attribution.",
        "Consider pinning cuDNN workspace size with torch.backends.cudnn.benchmark = False.",
    ],
    "persistent_drift": [
        "Look for non-PyTorch CUDA allocations accumulating over time "
        "(e.g. custom CUDA kernels, third-party libraries).",
        "Monitor nvidia-smi used memory alongside torch allocator counters.",
        "If gap stabilises after warmup, it may be one-time CUDA context overhead.",
    ],
    "fragmentation_like": [
        "Call torch.cuda.empty_cache() periodically to release unused reserved blocks.",
        "Reduce allocation churn by pre-allocating tensors or using memory pools.",
        "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.",
    ],
}


class MemoryAnalyzer:
    """Advanced analyzer for memory profiling data."""

    def __init__(self, profiler: Optional[GPUMemoryProfiler] = None):
        """
        Initialize the analyzer.

        Args:
            profiler: GPUMemoryProfiler instance to analyze
        """
        self.profiler = profiler

        # Analysis thresholds
        self.thresholds = {
            "memory_leak_ratio": 0.1,  # 10% memory growth indicates potential leak
            "fragmentation_ratio": 0.3,  # 30% fragmentation is concerning
            "inefficient_allocation_ratio": 0.5,  # 50% waste in allocations
            "slow_function_percentile": 0.9,  # Top 10% slowest functions
            "high_memory_percentile": 0.9,  # Top 10% memory-heavy functions
            "min_calls_for_analysis": 3,  # Minimum calls to consider for analysis
            # Hidden-memory gap analysis thresholds
            "gap_ratio_threshold": 0.05,  # 5% of device total = significant gap
            "gap_spike_zscore": 2.0,  # z-score for transient spike detection
            "gap_drift_r_squared": 0.6,  # R-squared for persistent drift
            "gap_fragmentation_ratio": 0.3,  # reserved-allocated / reserved
        }

    def analyze_memory_patterns(
        self, results: Optional[List[ProfileResult]] = None
    ) -> List[MemoryPattern]:
        """
        Detect memory usage patterns in profiling data.

        Args:
            results: List of ProfileResults to analyze

        Returns:
            List of detected patterns
        """
        if results is None and self.profiler:
            results = self.profiler.results

        if not results:
            return []

        patterns = []

        # Detect different types of patterns
        patterns.extend(self._detect_memory_leaks(results))
        patterns.extend(self._detect_fragmentation_issues(results))
        patterns.extend(self._detect_inefficient_allocations(results))
        patterns.extend(self._detect_memory_spikes(results))
        patterns.extend(self._detect_repeated_allocations(results))

        return patterns

    def _detect_memory_leaks(self, results: List[ProfileResult]) -> List[MemoryPattern]:
        """Detect potential memory leaks."""
        patterns = []

        # Group results by function
        function_memory = defaultdict(list)
        for result in results:
            memory_change = result.memory_diff()
            function_memory[result.function_name].append(memory_change)

        # Look for functions with consistently positive memory growth
        for func_name, memory_changes in function_memory.items():
            if len(memory_changes) < self.thresholds["min_calls_for_analysis"]:
                continue

            total_growth = sum(memory_changes)
            avg_growth = total_growth / len(memory_changes)

            # Check if function consistently allocates more than it frees
            positive_ratio = sum(1 for change in memory_changes if change > 0) / len(
                memory_changes
            )

            if (
                avg_growth > 0
                and positive_ratio > self.thresholds["memory_leak_ratio"]
                and total_growth > 100 * 1024 * 1024
            ):  # At least 100MB total growth

                severity = "critical" if total_growth > 1024**3 else "warning"

                patterns.append(
                    MemoryPattern(
                        pattern_type="memory_leak",
                        description=f"Function '{func_name}' shows potential memory leak pattern",
                        severity=severity,
                        affected_functions=[func_name],
                        metrics={
                            "total_memory_growth": total_growth,
                            "average_growth_per_call": avg_growth,
                            "positive_growth_ratio": positive_ratio,
                            "call_count": len(memory_changes),
                        },
                        suggestions=[
                            f"Review memory management in '{func_name}'",
                            "Check for uncleaned tensors or variables",
                            "Use torch.cuda.empty_cache() if appropriate",
                            "Consider using context managers for tensor lifecycle",
                        ],
                    )
                )

        return patterns

    def _detect_fragmentation_issues(
        self, results: List[ProfileResult]
    ) -> List[MemoryPattern]:
        """Detect memory fragmentation patterns."""
        patterns = []

        # Calculate fragmentation metrics across all results
        fragmentation_ratios = []
        high_frag_functions = []

        for result in results:
            # Use reserved vs allocated memory as fragmentation indicator
            allocated = result.memory_after.allocated_memory
            reserved = result.memory_after.reserved_memory

            if reserved > 0:
                fragmentation_ratio = (reserved - allocated) / reserved
                fragmentation_ratios.append(fragmentation_ratio)

                if fragmentation_ratio > self.thresholds["fragmentation_ratio"]:
                    high_frag_functions.append(result.function_name)

        if fragmentation_ratios:
            avg_fragmentation = statistics.mean(fragmentation_ratios)
            max_fragmentation = max(fragmentation_ratios)

            if avg_fragmentation > self.thresholds["fragmentation_ratio"]:
                severity = "critical" if avg_fragmentation > 0.5 else "warning"

                patterns.append(
                    MemoryPattern(
                        pattern_type="fragmentation",
                        description="High memory fragmentation detected across operations",
                        severity=severity,
                        affected_functions=list(set(high_frag_functions)),
                        metrics={
                            "average_fragmentation": avg_fragmentation,
                            "max_fragmentation": max_fragmentation,
                            "high_fragmentation_functions": len(
                                set(high_frag_functions)
                            ),
                        },
                        suggestions=[
                            "Call torch.cuda.empty_cache() periodically",
                            "Restructure code to reduce allocation/deallocation cycles",
                            "Consider pre-allocating tensors when possible",
                            "Use tensor.detach() to break computation graphs early",
                        ],
                    )
                )

        return patterns

    def _detect_inefficient_allocations(
        self, results: List[ProfileResult]
    ) -> List[MemoryPattern]:
        """Detect inefficient memory allocation patterns."""
        patterns: List[MemoryPattern] = []

        # Look for functions that allocate much more than they actually use
        inefficient_functions: List[Dict[str, Any]] = []

        for result in results:
            allocated = result.memory_allocated
            peak_usage = (
                result.peak_memory_usage() - result.memory_before.allocated_memory
            )

            if allocated > 0 and peak_usage > 0:
                efficiency_ratio = peak_usage / allocated

                if efficiency_ratio < self.thresholds["inefficient_allocation_ratio"]:
                    inefficient_functions.append(
                        {
                            "function": result.function_name,
                            "efficiency_ratio": efficiency_ratio,
                            "allocated": allocated,
                            "peak_usage": peak_usage,
                        }
                    )

        if inefficient_functions:
            # Group by function name
            func_efficiency: Dict[str, List[float]] = defaultdict(list)
            for item in inefficient_functions:
                func_efficiency[str(item["function"])].append(
                    float(item["efficiency_ratio"])
                )

            # Find consistently inefficient functions
            consistently_inefficient: List[str] = []
            for func_name, ratios in func_efficiency.items():
                if len(ratios) >= self.thresholds["min_calls_for_analysis"]:
                    avg_efficiency = statistics.mean(ratios)
                    if avg_efficiency < self.thresholds["inefficient_allocation_ratio"]:
                        consistently_inefficient.append(func_name)

            if consistently_inefficient:
                patterns.append(
                    MemoryPattern(
                        pattern_type="inefficient_allocation",
                        description="Functions with inefficient memory allocation patterns detected",
                        severity="warning",
                        affected_functions=consistently_inefficient,
                        metrics={
                            "inefficient_function_count": len(consistently_inefficient),
                            "average_efficiency": statistics.mean(
                                [
                                    statistics.mean(func_efficiency[func])
                                    for func in consistently_inefficient
                                ]
                            ),
                        },
                        suggestions=[
                            "Review allocation patterns in inefficient functions",
                            "Consider using in-place operations where possible",
                            "Pre-allocate tensors to avoid repeated allocations",
                            "Use tensor views instead of copies when appropriate",
                        ],
                    )
                )

        return patterns

    def _detect_memory_spikes(
        self, results: List[ProfileResult]
    ) -> List[MemoryPattern]:
        """Detect sudden memory spikes."""
        patterns: List[MemoryPattern] = []

        # Calculate memory allocation statistics
        allocations = [r.memory_allocated for r in results if r.memory_allocated > 0]

        if len(allocations) < 3:
            return patterns

        # Use statistical methods to detect outliers
        allocation_array = np.asarray(allocations, dtype=float)
        q75 = float(np.quantile(allocation_array, 0.75))
        q25 = float(np.quantile(allocation_array, 0.25))
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr

        spike_functions = []
        for result in results:
            if result.memory_allocated > outlier_threshold:
                spike_functions.append(result.function_name)

        if spike_functions:
            spike_count = len(spike_functions)
            unique_spike_functions = list(set(spike_functions))

            patterns.append(
                MemoryPattern(
                    pattern_type="memory_spikes",
                    description=f"Detected {spike_count} memory allocation spikes",
                    severity="warning" if spike_count < 5 else "critical",
                    affected_functions=unique_spike_functions,
                    metrics={
                        "spike_count": spike_count,
                        "spike_threshold": outlier_threshold,
                        "max_allocation": max(allocations),
                        "median_allocation": statistics.median(allocations),
                    },
                    suggestions=[
                        "Investigate functions causing memory spikes",
                        "Consider batch processing to reduce peak memory",
                        "Use gradient checkpointing for large models",
                        "Implement memory monitoring in spike-prone functions",
                    ],
                )
            )

        return patterns

    def _detect_repeated_allocations(
        self, results: List[ProfileResult]
    ) -> List[MemoryPattern]:
        """Detect patterns of repeated allocations that could be optimized."""
        patterns = []

        # Count function calls and total memory allocated
        function_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "total_memory": 0}
        )

        for result in results:
            func_name = result.function_name
            function_stats[func_name]["calls"] += 1
            function_stats[func_name]["total_memory"] += result.memory_allocated

        # Find functions with many small allocations
        repeated_allocation_functions: List[Dict[str, Any]] = []

        for func_name, func_stats in function_stats.items():
            if func_stats["calls"] >= 10:  # Many calls
                avg_allocation = func_stats["total_memory"] / func_stats["calls"]

                # Check if allocations are small but frequent
                if avg_allocation < 50 * 1024 * 1024:  # Less than 50MB per call
                    repeated_allocation_functions.append(
                        {
                            "function": func_name,
                            "calls": func_stats["calls"],
                            "avg_allocation": avg_allocation,
                            "total_memory": func_stats["total_memory"],
                        }
                    )

        if repeated_allocation_functions:
            # Sort by total memory impact
            repeated_allocation_functions.sort(
                key=lambda x: float(x["total_memory"]), reverse=True
            )
            top_functions = [
                str(f["function"]) for f in repeated_allocation_functions[:5]
            ]

            patterns.append(
                MemoryPattern(
                    pattern_type="repeated_allocations",
                    description="Functions with frequent small allocations detected",
                    severity="info",
                    affected_functions=top_functions,
                    metrics={
                        "function_count": len(repeated_allocation_functions),
                        "total_impact_functions": top_functions,
                        "total_memory_from_repeated": sum(
                            f["total_memory"] for f in repeated_allocation_functions
                        ),
                    },
                    suggestions=[
                        "Consider pre-allocating memory for frequently called functions",
                        "Use tensor pools or memory buffers for repeated allocations",
                        "Batch operations to reduce allocation overhead",
                        "Cache tensors between function calls when possible",
                    ],
                )
            )

        return patterns

    def generate_performance_insights(
        self, results: Optional[List[ProfileResult]] = None
    ) -> List[PerformanceInsight]:
        """
        Generate performance insights from profiling data.

        Args:
            results: List of ProfileResults to analyze

        Returns:
            List of performance insights
        """
        if results is None and self.profiler:
            results = self.profiler.results

        if not results:
            return []

        insights = []

        # Performance analysis
        insights.extend(self._analyze_execution_times(results))
        insights.extend(self._analyze_memory_efficiency(results))
        insights.extend(self._analyze_function_correlations(results))
        insights.extend(self._analyze_temporal_patterns(results))

        return insights

    def _analyze_execution_times(
        self, results: List[ProfileResult]
    ) -> List[PerformanceInsight]:
        """Analyze execution time patterns."""
        insights = []

        # Group by function
        function_times = defaultdict(list)
        for result in results:
            function_times[result.function_name].append(result.execution_time)

        # Find slowest functions
        function_avg_times = {}
        for func_name, times in function_times.items():
            if len(times) >= self.thresholds["min_calls_for_analysis"]:
                function_avg_times[func_name] = statistics.mean(times)

        if function_avg_times:
            # Find top slowest functions
            sorted_functions = sorted(
                function_avg_times.items(), key=lambda x: x[1], reverse=True
            )

            slow_threshold = np.percentile(
                list(function_avg_times.values()),
                self.thresholds["slow_function_percentile"] * 100,
            )

            slow_functions = [
                func for func, time in sorted_functions if time > slow_threshold
            ]

            if slow_functions:
                insights.append(
                    PerformanceInsight(
                        category="execution_time",
                        title="Slow Function Detection",
                        description=f"Identified {len(slow_functions)} functions with high execution times",
                        impact="high" if len(slow_functions) > 3 else "medium",
                        confidence=0.8,
                        data={
                            "slow_functions": slow_functions[:5],
                            "slowest_function": sorted_functions[0][0],
                            "slowest_time": sorted_functions[0][1],
                            "threshold": slow_threshold,
                        },
                        recommendations=[
                            "Profile slow functions in detail",
                            "Consider algorithmic optimizations",
                            "Look for GPU/CPU synchronization issues",
                            "Check for unnecessary memory transfers",
                        ],
                    )
                )

        # Analyze execution time variance
        high_variance_functions = []
        for func_name, times in function_times.items():
            if len(times) >= self.thresholds["min_calls_for_analysis"]:
                # Coefficient of variation
                cv = statistics.stdev(times) / statistics.mean(times)
                if cv > 0.5:  # High variance
                    high_variance_functions.append((func_name, cv))

        if high_variance_functions:
            insights.append(
                PerformanceInsight(
                    category="execution_time",
                    title="Inconsistent Execution Times",
                    description="Functions with highly variable execution times detected",
                    impact="medium",
                    confidence=0.7,
                    data={
                        "variable_functions": [
                            func for func, _ in high_variance_functions
                        ],
                        "highest_variance": max(
                            high_variance_functions, key=lambda x: x[1]
                        ),
                    },
                    recommendations=[
                        "Investigate causes of execution time variance",
                        "Check for resource contention",
                        "Look for input size dependencies",
                        "Consider warming up functions before timing",
                    ],
                )
            )

        return insights

    def _analyze_memory_efficiency(
        self, results: List[ProfileResult]
    ) -> List[PerformanceInsight]:
        """Analyze memory usage efficiency."""
        insights = []

        # Memory-to-time ratio analysis
        memory_time_ratios = []
        for result in results:
            if result.execution_time > 0:
                ratio = result.memory_allocated / result.execution_time
                memory_time_ratios.append((result.function_name, ratio))

        if memory_time_ratios:
            # Find functions with high memory/time ratios (memory-intensive)
            ratios = [ratio for _, ratio in memory_time_ratios]
            high_ratio_threshold = np.percentile(
                ratios, self.thresholds["high_memory_percentile"] * 100
            )

            memory_intensive_functions = [
                func
                for func, ratio in memory_time_ratios
                if ratio > high_ratio_threshold
            ]

            if memory_intensive_functions:
                unique_functions = list(set(memory_intensive_functions))

                insights.append(
                    PerformanceInsight(
                        category="memory_efficiency",
                        title="Memory-Intensive Functions",
                        description=f"Identified {len(unique_functions)} memory-intensive functions",
                        impact="medium",
                        confidence=0.75,
                        data={
                            "memory_intensive_functions": unique_functions[:5],
                            "threshold_ratio": high_ratio_threshold,
                        },
                        recommendations=[
                            "Optimize memory usage in intensive functions",
                            "Consider using smaller data types",
                            "Implement memory streaming for large operations",
                            "Use gradient accumulation to reduce memory peaks",
                        ],
                    )
                )

        return insights

    def _analyze_function_correlations(
        self, results: List[ProfileResult]
    ) -> List[PerformanceInsight]:
        """Analyze correlations between different metrics."""
        insights: List[PerformanceInsight] = []

        if len(results) < 10:  # Need enough data for correlation analysis
            return insights

        # Extract metrics for correlation analysis
        execution_times = [r.execution_time for r in results]
        memory_allocated = [r.memory_allocated for r in results]
        memory_peak = [r.peak_memory_usage() for r in results]

        # Calculate correlations
        time_memory_corr = stats.pearsonr(execution_times, memory_allocated)[0]
        _time_peak_corr = stats.pearsonr(execution_times, memory_peak)[0]

        # Strong correlation between time and memory
        if abs(time_memory_corr) > 0.7:
            insights.append(
                PerformanceInsight(
                    category="correlation",
                    title="Time-Memory Correlation",
                    description=f"Strong correlation ({time_memory_corr:.2f}) between execution time and memory",
                    impact="medium",
                    confidence=0.8,
                    data={
                        "correlation_coefficient": time_memory_corr,
                        "interpretation": (
                            "positive" if time_memory_corr > 0 else "negative"
                        ),
                    },
                    recommendations=[
                        "Memory allocation is a significant factor in execution time",
                        "Consider memory pre-allocation strategies",
                        "Optimize memory access patterns",
                        "Monitor memory bandwidth utilization",
                    ],
                )
            )

        return insights

    def _analyze_temporal_patterns(
        self, results: List[ProfileResult]
    ) -> List[PerformanceInsight]:
        """Analyze temporal patterns in the profiling data."""
        insights: List[PerformanceInsight] = []

        # Sort results by timestamp
        sorted_results = sorted(results, key=lambda r: r.memory_before.timestamp)

        if len(sorted_results) < 5:
            return insights

        # Analyze memory growth over time
        timestamps = [r.memory_before.timestamp for r in sorted_results]
        memory_usage = [r.memory_after.allocated_memory for r in sorted_results]

        # Calculate trend
        if len(timestamps) > 1:
            time_diffs = [
                (timestamps[i] - timestamps[0]) for i in range(len(timestamps))
            ]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                time_diffs, memory_usage
            )

            # Significant upward trend indicates potential memory leak
            if slope > 0 and abs(r_value) > 0.5 and p_value < 0.05:
                insights.append(
                    PerformanceInsight(
                        category="temporal",
                        title="Memory Growth Trend",
                        description="Detected upward trend in memory usage over time",
                        impact="high" if slope > 1e6 else "medium",  # 1MB/s growth
                        confidence=abs(r_value),
                        data={
                            "growth_rate": slope,
                            "correlation": r_value,
                            "p_value": p_value,
                            "time_span": max(time_diffs),
                        },
                        recommendations=[
                            "Investigate potential memory leaks",
                            "Implement periodic memory cleanup",
                            "Monitor long-running processes",
                            "Consider memory usage limits",
                        ],
                    )
                )

        return insights

    # ------------------------------------------------------------------
    # Hidden-memory gap analysis (operates on TelemetryEventV2 series)
    # ------------------------------------------------------------------

    def analyze_memory_gaps(self, events: List[TelemetryEventV2]) -> List[GapFinding]:
        """Classify allocator-vs-device hidden memory gaps over time.

        Args:
            events: Chronologically ordered telemetry samples.

        Returns:
            Prioritized list of gap findings (severity desc, confidence desc).
        """
        return analyze_hidden_memory_gaps(
            events=events,
            thresholds=self.thresholds,
            format_memory=format_bytes,
            remediation_by_classification=_GAP_REMEDIATION_BY_CLASSIFICATION,
        )

    def generate_optimization_report(
        self,
        results: Optional[List[ProfileResult]] = None,
        events: Optional[List[TelemetryEventV2]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive optimization report.

        Args:
            results: List of ProfileResults to analyze
            events: Optional telemetry event series for gap analysis.
                    When provided, the report includes a ``gap_analysis`` section.

        Returns:
            Comprehensive optimization report
        """
        if results is None and self.profiler:
            results = self.profiler.results

        effective_results = results or []

        patterns = self.analyze_memory_patterns(effective_results)
        insights = self.generate_performance_insights(effective_results)

        # Categorize findings by severity/impact
        critical_issues = [p for p in patterns if p.severity == "critical"]
        high_impact_insights = [i for i in insights if i.impact == "high"]

        # Generate summary statistics
        total_memory_allocated = sum(r.memory_allocated for r in effective_results)
        total_execution_time = sum(r.execution_time for r in effective_results)
        unique_functions = len(set(r.function_name for r in effective_results))

        report: Dict[str, Any] = {
            "summary": {
                "total_functions_analyzed": unique_functions,
                "total_function_calls": len(effective_results),
                "total_memory_allocated": total_memory_allocated,
                "total_execution_time": total_execution_time,
                "analysis_timestamp": (
                    effective_results[-1].memory_after.timestamp
                    if effective_results
                    else None
                ),
            },
            "critical_issues": [p.__dict__ for p in critical_issues],
            "high_impact_insights": [i.__dict__ for i in high_impact_insights],
            "all_patterns": [p.__dict__ for p in patterns],
            "all_insights": [i.__dict__ for i in insights],
            "recommendations": self._generate_priority_recommendations(
                patterns, insights
            ),
            "optimization_score": self._calculate_optimization_score(
                patterns, insights
            ),
        }

        # Hidden-memory gap analysis (only when telemetry events are supplied).
        if events is not None:
            gap_findings = self.analyze_memory_gaps(events)
            report["gap_analysis"] = [asdict(f) for f in gap_findings]

        return report

    def _generate_priority_recommendations(
        self, patterns: List[MemoryPattern], insights: List[PerformanceInsight]
    ) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Critical issues first
        for pattern in patterns:
            if pattern.severity == "critical":
                recommendations.append(
                    {
                        "priority": "high",
                        "category": pattern.pattern_type,
                        "description": pattern.description,
                        "suggestions": pattern.suggestions,
                    }
                )

        # High impact insights
        for insight in insights:
            if insight.impact == "high":
                recommendations.append(
                    {
                        "priority": "high",
                        "category": insight.category,
                        "description": insight.description,
                        "suggestions": insight.recommendations,
                    }
                )

        # Other important issues
        for pattern in patterns:
            if pattern.severity == "warning":
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": pattern.pattern_type,
                        "description": pattern.description,
                        "suggestions": pattern.suggestions,
                    }
                )

        return recommendations[:10]  # Top 10 recommendations

    def _calculate_optimization_score(
        self, patterns: List[MemoryPattern], insights: List[PerformanceInsight]
    ) -> Dict[str, Any]:
        """Calculate an optimization score based on detected issues."""
        base_score = 100

        # Deduct points for issues
        for pattern in patterns:
            if pattern.severity == "critical":
                base_score -= 20
            elif pattern.severity == "warning":
                base_score -= 10
            else:
                base_score -= 5

        for insight in insights:
            if insight.impact == "high":
                base_score -= 15
            elif insight.impact == "medium":
                base_score -= 8
            else:
                base_score -= 3

        score = max(0, base_score)

        if score >= 90:
            grade = "A"
            description = "Excellent memory usage patterns"
        elif score >= 80:
            grade = "B"
            description = "Good memory usage with minor issues"
        elif score >= 70:
            grade = "C"
            description = "Acceptable memory usage with some optimization potential"
        elif score >= 60:
            grade = "D"
            description = "Poor memory usage patterns requiring attention"
        else:
            grade = "F"
            description = "Critical memory usage issues requiring immediate attention"

        return {
            "score": score,
            "grade": grade,
            "description": description,
            "issues_found": len(patterns) + len(insights),
        }
