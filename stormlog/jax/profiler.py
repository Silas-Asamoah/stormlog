"""JAX Memory Profiler.

Provides snapshot-based memory profiling, function/context profiling,
and a global profiler singleton for JAX workloads.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

from .jax_env import configure_jax_logging
from .utils import _device_zero, get_device_memory_capability, resolve_jax_device

configure_jax_logging()

jax: Any

try:
    import jax as _jax  # noqa: E402

    jax = _jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    """Point-in-time JAX memory snapshot."""

    timestamp: float
    name: str
    device_memory_bytes: int
    cpu_memory_bytes: int
    device_id: int
    device_memory_reserved_bytes: int = 0
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    operation_name: Optional[str] = None
    device_memory_available: bool = True
    device_memory_unavailable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.device_memory_bytes < 0:
            self.device_memory_bytes = 0
        if self.device_memory_reserved_bytes < 0:
            self.device_memory_reserved_bytes = 0
        if self.cpu_memory_bytes < 0:
            self.cpu_memory_bytes = 0

    @property
    def device_memory_mb(self) -> float:
        return self.device_memory_bytes / (1024 * 1024)

    @property
    def device_memory_reserved_mb(self) -> float:
        return self.device_memory_reserved_bytes / (1024 * 1024)

    @property
    def cpu_memory_mb(self) -> float:
        return self.cpu_memory_bytes / (1024 * 1024)


@dataclass
class ProfileResult:
    """Aggregated profiling results for a JAX session."""

    start_time: float
    end_time: float
    peak_memory_bytes: int
    average_memory_bytes: int
    min_memory_bytes: int
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    function_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    device_memory_available: bool = False
    device_memory_unavailable_reason: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def peak_memory_mb(self) -> float:
        return self.peak_memory_bytes / (1024 * 1024)

    @property
    def average_memory_mb(self) -> float:
        return self.average_memory_bytes / (1024 * 1024)

    @property
    def min_memory_mb(self) -> float:
        return self.min_memory_bytes / (1024 * 1024)

    @property
    def memory_growth_rate(self) -> float:
        """Memory growth rate in MB/second.

        Stormlog versions before 0.3.6 reported this property in bytes/second;
        0.3.6 and later report MB/second.
        """
        if self.duration <= 0:
            return 0.0
        return (
            (self.peak_memory_bytes - self.min_memory_bytes)
            / (1024 * 1024)
            / self.duration
        )


# ---------------------------------------------------------------------------
# Core profiler
# ---------------------------------------------------------------------------


class JAXMemoryProfiler:
    """JAX memory profiler with snapshot capture and function profiling.

    Provides:

    * ``capture_snapshot()`` – point-in-time device memory reading
    * ``profile_function()`` – decorator-based before/after profiling
    * ``profile_context()`` – with-block profiling
    * ``start_continuous_profiling()`` / ``stop_continuous_profiling()``
    * ``get_results()`` – aggregate into :class:`ProfileResult`

    Example::

        profiler = JAXMemoryProfiler()
        with profiler:
            s = profiler.capture_snapshot("after_init")
        result = profiler.get_results()
    """

    def __init__(
        self, device_index: Union[int, str] = 0, *, device: Any = None
    ) -> None:
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX not available. Install with `pip install 'stormlog[jax]'`."
            )

        self._device_index = int(device_index) if isinstance(device_index, int) else 0
        self._device: Any = None
        self._lock = threading.Lock()

        self._snapshots: List[MemorySnapshot] = []
        self._function_profiles: Dict[str, Dict[str, Any]] = {}

        self._continuous_thread: Optional[threading.Thread] = None
        self._continuous_stop = threading.Event()

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        try:
            if device is not None:
                self._device = device
                self._device_index = int(getattr(device, "id", device_index))
            elif isinstance(device_index, int):
                devices = jax.local_devices()
                if 0 <= device_index < len(devices):
                    self._device = devices[device_index]
            else:
                self._device, self._device_index = resolve_jax_device(device_index)
        except Exception as exc:
            logger.debug("Could not resolve JAX device %s: %s", device_index, exc)

        # Cache a scalar sentinel for sync barriers
        self._sync_sentinel: Any = None
        if self._device is not None:
            try:
                self._sync_sentinel = _device_zero(self._device)
            except Exception:
                pass

    # -- Snapshot capture --------------------------------------------------

    def capture_snapshot(
        self,
        name: str = "snapshot",
        *,
        operation_name: Optional[str] = None,
    ) -> MemorySnapshot:
        """Capture a point-in-time memory snapshot.

        Args:
            name: Human-readable label for this snapshot.
            operation_name: Optional operation being profiled.

        Returns:
            A :class:`MemorySnapshot`.
        """
        device_bytes = 0
        reserved_bytes = 0
        memory_stats: Dict[str, Any] = {}
        device_memory_available = False
        unavailable_reason: Optional[str] = "No JAX device was selected"

        if self._device is not None:
            try:
                # Flush XLA async dispatch before reading memory stats.
                # JAX dispatches operations asynchronously to XLA — without
                # this synchronisation barrier, memory_stats() may return
                # stale values that exclude memory from operations still
                # "in flight".  The trade-off is a small allocation
                # (1-element array) and a forced sync; at high snapshot
                # frequencies this can slightly perturb workload timing.
                if self._sync_sentinel is not None:
                    self._sync_sentinel.block_until_ready()
                else:
                    _device_zero(self._device).block_until_ready()
                capability = get_device_memory_capability(self._device)
                memory_stats = capability["memory_stats"]
                device_memory_available = bool(capability["memory_stats_available"])
                unavailable_reason = capability["memory_stats_error"]
                if device_memory_available:
                    device_bytes = int(memory_stats.get("bytes_in_use", 0))
                    reserved_bytes = int(
                        memory_stats.get("bytes_reserved", device_bytes)
                    )
            except Exception as exc:
                logger.debug("Snapshot memory_stats failed: %s", exc)
                unavailable_reason = str(exc)

        cpu_bytes = 0
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                cpu_bytes = psutil.Process().memory_info().rss
            except Exception as exc:
                logger.debug("CPU memory read failed: %s", exc)

        snap = MemorySnapshot(
            timestamp=time.time(),
            name=name,
            device_memory_bytes=device_bytes,
            device_memory_reserved_bytes=reserved_bytes,
            cpu_memory_bytes=cpu_bytes,
            device_id=self._device_index,
            memory_stats=memory_stats,
            operation_name=operation_name,
            device_memory_available=device_memory_available,
            device_memory_unavailable_reason=unavailable_reason,
        )

        with self._lock:
            self._snapshots.append(snap)
        return snap

    # -- Function profiling ------------------------------------------------

    def profile_function(
        self,
        func: Optional[F] = None,
        *,
        name: Optional[str] = None,
    ) -> Any:
        """Decorator that profiles a function's memory impact.

        Usage::

            @profiler.profile_function
            def train_step():
                ...

            # or with options:
            @profiler.profile_function(name="custom_name")
            def train_step():
                ...
        """

        def decorator(f: F) -> F:
            profiled_name = name or f.__name__

            @functools.wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                before = self.capture_snapshot(
                    f"{profiled_name}_before",
                    operation_name=profiled_name,
                )
                start = time.monotonic()
                try:
                    result = f(*args, **kwargs)
                finally:
                    elapsed = time.monotonic() - start
                    after = self.capture_snapshot(
                        f"{profiled_name}_after",
                        operation_name=profiled_name,
                    )

                    memory_available = (
                        before.device_memory_available and after.device_memory_available
                    )
                    delta = (
                        after.device_memory_bytes - before.device_memory_bytes
                        if memory_available
                        else 0
                    )
                    peak = max(
                        before.device_memory_bytes,
                        after.device_memory_bytes,
                    )

                    with self._lock:
                        entry = self._function_profiles.setdefault(
                            profiled_name,
                            {
                                "calls": 0,
                                "total_duration": 0.0,
                                "peak_memory_bytes": 0,
                                "total_memory_delta": 0,
                                "last_memory_delta": 0,
                                "device_memory_available": memory_available,
                            },
                        )
                        entry["calls"] += 1
                        entry["total_duration"] += elapsed
                        entry["peak_memory_bytes"] = max(
                            entry["peak_memory_bytes"], peak
                        )
                        entry["total_memory_delta"] += delta
                        entry["last_memory_delta"] = delta
                        entry["device_memory_available"] = memory_available

                return result

            return cast(F, wrapper)

        if func is None:
            return decorator
        return decorator(func)

    # -- Context profiling -------------------------------------------------

    @contextmanager
    def profile_context(
        self,
        name: str = "context",
    ) -> Iterator[MemorySnapshot]:
        """Context manager that captures before/after snapshots.

        Usage::

            with profiler.profile_context("matmul") as snap_before:
                result = jax.numpy.dot(a, b)
        """
        before = self.capture_snapshot(f"{name}_before", operation_name=name)
        start = time.monotonic()
        try:
            yield before
        finally:
            elapsed = time.monotonic() - start
            after = self.capture_snapshot(f"{name}_after", operation_name=name)
            memory_available = (
                before.device_memory_available and after.device_memory_available
            )
            delta = (
                after.device_memory_bytes - before.device_memory_bytes
                if memory_available
                else 0
            )
            peak = max(
                before.device_memory_bytes,
                after.device_memory_bytes,
            )

            with self._lock:
                entry = self._function_profiles.setdefault(
                    name,
                    {
                        "calls": 0,
                        "total_duration": 0.0,
                        "peak_memory_bytes": 0,
                        "total_memory_delta": 0,
                        "last_memory_delta": 0,
                        "device_memory_available": memory_available,
                    },
                )
                entry["calls"] += 1
                entry["total_duration"] += elapsed
                entry["peak_memory_bytes"] = max(entry["peak_memory_bytes"], peak)
                entry["total_memory_delta"] += delta
                entry["last_memory_delta"] = delta
                entry["device_memory_available"] = memory_available

    # -- Continuous profiling ----------------------------------------------

    def start_continuous_profiling(self, interval: float = 1.0) -> None:
        """Start background snapshot capture at *interval* seconds."""
        if self._continuous_thread is not None:
            logger.warning("Continuous profiling already running")
            return
        self._continuous_stop.clear()
        self._continuous_thread = threading.Thread(
            target=self._continuous_loop,
            args=(interval,),
            daemon=True,
        )
        self._continuous_thread.start()

    def stop_continuous_profiling(self) -> None:
        """Stop the background snapshot loop."""
        self._continuous_stop.set()
        if self._continuous_thread is not None:
            self._continuous_thread.join(timeout=5.0)
            self._continuous_thread = None

    def _continuous_loop(self, interval: float) -> None:
        counter = 0
        while not self._continuous_stop.is_set():
            self.capture_snapshot(f"continuous_{counter}")
            counter += 1
            self._continuous_stop.wait(interval)

    # -- Results -----------------------------------------------------------

    def get_results(self) -> ProfileResult:
        """Aggregate captured snapshots into a :class:`ProfileResult`."""
        with self._lock:
            snapshots = list(self._snapshots)
            profiles = {k: dict(v) for k, v in self._function_profiles.items()}
        snapshots.sort(key=lambda snapshot: snapshot.timestamp)

        if not snapshots:
            now = time.time()
            return ProfileResult(
                start_time=self._start_time or now,
                end_time=self._end_time or now,
                peak_memory_bytes=0,
                average_memory_bytes=0,
                min_memory_bytes=0,
                snapshots=[],
                function_profiles=profiles,
            )

        memories, memory_available, unavailable_reason = _snapshot_memory_values(
            snapshots
        )
        return ProfileResult(
            start_time=snapshots[0].timestamp,
            end_time=snapshots[-1].timestamp,
            peak_memory_bytes=max(memories) if memories else 0,
            average_memory_bytes=int(sum(memories) / len(memories)) if memories else 0,
            min_memory_bytes=min(memories) if memories else 0,
            snapshots=snapshots,
            function_profiles=profiles,
            device_memory_available=memory_available,
            device_memory_unavailable_reason=unavailable_reason,
        )

    def reset(self) -> None:
        """Clear all captured data."""
        with self._lock:
            self._snapshots.clear()
            self._function_profiles.clear()
        self._start_time = None
        self._end_time = None

    # -- Context manager ---------------------------------------------------

    def __enter__(self) -> "JAXMemoryProfiler":
        self._start_time = time.time()
        self.capture_snapshot("session_start")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.capture_snapshot("session_end")
        self._end_time = time.time()
        self.stop_continuous_profiling()


# ---------------------------------------------------------------------------
# Global profiler singleton
# ---------------------------------------------------------------------------

_global_profiler: Optional[JAXMemoryProfiler] = None
_profiler_lock = threading.Lock()


def get_global_profiler() -> JAXMemoryProfiler:
    """Get or create the global :class:`JAXMemoryProfiler` instance."""
    global _global_profiler
    with _profiler_lock:
        if _global_profiler is None:
            _global_profiler = JAXMemoryProfiler()
        return _global_profiler


def set_global_profiler(profiler: JAXMemoryProfiler) -> None:
    """Replace the global profiler instance."""
    global _global_profiler
    with _profiler_lock:
        if _global_profiler is not None:
            _global_profiler.stop_continuous_profiling()
        _global_profiler = profiler


def clear_global_profiler() -> None:
    """Reset and discard the global profiler."""
    global _global_profiler
    with _profiler_lock:
        if _global_profiler is not None:
            _global_profiler.stop_continuous_profiling()
            _global_profiler.reset()
        _global_profiler = None


def clear_profiles() -> None:
    """Reset the global profiler without discarding it."""
    with _profiler_lock:
        if _global_profiler is not None:
            _global_profiler.reset()


def get_profile_summaries(
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return aggregated profile summaries from the global profiler.

    Args:
        limit: Maximum number of summaries to return.

    Returns:
        A list of dicts, one per profiled function/context block.
    """
    with _profiler_lock:
        prof = _global_profiler
    if prof is None:
        return []

    result = prof.get_results()
    summaries: List[Dict[str, Any]] = []
    for name, entry in result.function_profiles.items():
        summaries.append({"name": name, **entry})
    summaries.sort(key=lambda s: s.get("peak_memory_bytes", 0), reverse=True)

    if limit is not None:
        summaries = summaries[:limit]
    return summaries


def _snapshot_memory_values(
    snapshots: List[MemorySnapshot],
) -> tuple[List[int], bool, Optional[str]]:
    memory_available = all(snapshot.device_memory_available for snapshot in snapshots)
    unavailable_reason = next(
        (
            snapshot.device_memory_unavailable_reason
            for snapshot in snapshots
            if not snapshot.device_memory_available
        ),
        None,
    )
    memories = [s.device_memory_bytes for s in snapshots] if memory_available else []
    return memories, memory_available, unavailable_reason
