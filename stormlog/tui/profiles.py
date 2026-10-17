"""Helpers for exposing profile summaries inside the Textual TUI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

get_pt_results: Optional[Callable[..., List[Any]]] = None
clear_pt_results: Optional[Callable[[], None]] = None
_pt_profiles_import_attempted = False


def _ensure_pytorch_profile_helpers() -> None:
    global get_pt_results, clear_pt_results, _pt_profiles_import_attempted
    if _pt_profiles_import_attempted:
        return
    _pt_profiles_import_attempted = True
    try:
        from stormlog.context_profiler import clear_results, get_profile_results

        get_pt_results, clear_pt_results = get_profile_results, clear_results
    except ImportError:
        return


get_tf_summaries: Optional[Callable[..., List[Dict[str, Any]]]] = None
clear_tf_profiles: Optional[Callable[[], None]] = None
_tf_profiles_import_attempted = False


def _ensure_tensorflow_profile_helpers() -> None:
    """Import TensorFlow profile helpers lazily for import hardening."""
    global clear_tf_profiles
    global get_tf_summaries
    global _tf_profiles_import_attempted

    if _tf_profiles_import_attempted:
        return
    _tf_profiles_import_attempted = True

    try:
        from stormlog.tensorflow.context_profiler import (
            clear_profiles as _clear_tf_profiles,
        )
        from stormlog.tensorflow.context_profiler import (
            get_profile_summaries as _get_tf_summaries,
        )
    except Exception as exc:
        logger.debug("TensorFlow context profiler unavailable: %s", exc)
        get_tf_summaries = None
        clear_tf_profiles = None
        return

    get_tf_summaries = _get_tf_summaries
    clear_tf_profiles = _clear_tf_profiles


@dataclass
class ProfileRow:
    """Lightweight view model used by the TUI tables."""

    name: str
    peak_mb: float | None
    delta_mb: float | None
    duration_ms: float
    call_count: int
    recorded_at: float
    completion_verified: bool | None = None


def fetch_pytorch_profiles(limit: int = 15) -> List[ProfileRow]:
    """Return recent PyTorch profile rows."""
    _ensure_pytorch_profile_helpers()
    if get_pt_results is None:
        return []

    try:
        results = get_pt_results(limit=limit)
    except Exception as exc:
        logger.debug("fetch_pytorch_profiles failed: %s", exc)
        return []

    rows: List[ProfileRow] = []
    for result in results:
        timestamp = getattr(result.memory_after, "timestamp", None) or getattr(
            result.memory_peak, "timestamp", 0.0
        )
        recorded_at = float(timestamp or 0.0)
        rows.append(
            ProfileRow(
                name=result.function_name,
                peak_mb=result.peak_memory_usage() / (1024**2),
                delta_mb=result.memory_diff() / (1024**2),
                duration_ms=result.execution_time * 1000.0,
                call_count=result.call_count,
                recorded_at=recorded_at,
            )
        )

    rows.sort(key=lambda row: row.recorded_at or 0.0, reverse=True)
    if limit:
        return rows[:limit]
    return rows


def clear_pytorch_profiles() -> bool:
    """Clear global PyTorch profile results."""
    _ensure_pytorch_profile_helpers()
    if clear_pt_results is None:
        return False

    try:
        clear_pt_results()
        return True
    except Exception as exc:
        logger.debug("clear_pytorch_profiles failed: %s", exc)
        return False


def fetch_tensorflow_profiles(limit: int = 15) -> List[ProfileRow]:
    """Return aggregated TensorFlow profile summaries."""
    _ensure_tensorflow_profile_helpers()
    if get_tf_summaries is None:
        return []

    try:
        summaries = get_tf_summaries(limit=limit)
    except Exception as exc:
        logger.debug("fetch_tensorflow_profiles failed: %s", exc)
        return []

    rows: List[ProfileRow] = []
    for summary in summaries:
        calls = max(int(summary.get("calls", 0)), 1)
        total_duration = float(summary.get("total_duration", 0.0))
        total_memory = float(summary.get("total_memory_used", 0.0))
        peak_memory = float(summary.get("peak_memory", 0.0))
        timestamp = float(summary.get("last_timestamp") or 0.0)

        rows.append(
            ProfileRow(
                name=str(summary.get("name", "context")),
                peak_mb=peak_memory,
                delta_mb=total_memory / calls if calls else 0.0,
                duration_ms=(total_duration / calls) * 1000.0 if calls else 0.0,
                call_count=int(summary.get("calls", 0)),
                recorded_at=timestamp,
            )
        )

    rows.sort(key=lambda row: row.recorded_at or 0.0, reverse=True)
    if limit:
        return rows[:limit]
    return rows


def clear_tensorflow_profiles() -> bool:
    """Clear TensorFlow profile summaries if available."""
    _ensure_tensorflow_profile_helpers()
    if clear_tf_profiles is None:
        return False

    try:
        clear_tf_profiles()
        return True
    except Exception as exc:
        logger.debug("clear_tensorflow_profiles failed: %s", exc)
        return False


def fetch_mlx_profiles(limit: int = 15, *, path: str | None = None) -> List[ProfileRow]:
    from stormlog.mlx.context_profiler import get_profile_summaries
    from stormlog.mlx.profile_artifact import load_profiles

    results = load_profiles(path)["profiles"] if path else get_profile_summaries(limit)
    rows = []
    for result in results:
        before, after = (
            result["baseline"]["active_bytes"],
            result["final"]["active_bytes"],
        )
        peak = result["sampled_peak_bytes"]
        rows.append(
            ProfileRow(
                name=result["name"],
                peak_mb=None if peak is None else peak / 1024**2,
                delta_mb=(
                    None
                    if before is None or after is None
                    else (after - before) / 1024**2
                ),
                duration_ms=result["elapsed_ns"] / 1e6,
                call_count=1,
                recorded_at=result["ended_at_ns"] / 1e9,
                completion_verified=result["completion_verified"],
            )
        )
    rows.sort(key=lambda row: row.recorded_at, reverse=True)
    return rows[:limit] if limit else rows


def clear_mlx_profiles() -> bool:
    from stormlog.mlx.context_profiler import clear_profiles

    clear_profiles()
    return True
