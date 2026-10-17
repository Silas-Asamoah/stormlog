"""Narrow versioned MLX allocation-failure recognition; no torch discovery."""

from __future__ import annotations

import re

from stormlog.oom_flight_recorder import OOMExceptionClassification

# MLX v0.32.3 mlx/backend/metal/allocator.cpp. Resource-count and single-buffer
# limit failures are distinct conditions, deliberately not matched here.
_ALLOCATION_FAILURE = re.compile(r"\[malloc\] Unable to allocate \d+ bytes\.")


def classify_oom_exception(exc: BaseException) -> OOMExceptionClassification:
    if isinstance(exc, MemoryError):
        return OOMExceptionClassification(True, "host.MemoryError")
    if isinstance(exc, RuntimeError) and _ALLOCATION_FAILURE.fullmatch(
        str(exc).strip()
    ):
        return OOMExceptionClassification(
            True, "mlx.metal.allocator_allocation_failure"
        )
    return OOMExceptionClassification(False, None)
