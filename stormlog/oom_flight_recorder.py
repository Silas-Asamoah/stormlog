"""OOM flight recorder helpers for bounded event capture and dump artifacts."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .utils import get_system_info

logger = logging.getLogger(__name__)


_OOM_MESSAGE_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "hip out of memory",
    "resource exhausted",
    "failed to allocate",
    "allocation failed",
)


@dataclass(frozen=True)
class OOMFlightRecorderConfig:
    """Runtime configuration for OOM flight recorder dumps."""

    enabled: bool = False
    dump_dir: str = "oom_dumps"
    buffer_size: int = 10_000
    max_dumps: int = 5
    max_total_mb: int = 256


@dataclass(frozen=True)
class OOMExceptionClassification:
    """Normalized classification result for an exception."""

    is_oom: bool
    reason: Optional[str]


def classify_oom_exception(exc: BaseException) -> OOMExceptionClassification:
    """Classify whether an exception corresponds to an OOM condition."""

    try:
        import torch

        torch_oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if torch_oom_type is not None and isinstance(exc, torch_oom_type):
            return OOMExceptionClassification(True, "torch.cuda.OutOfMemoryError")
    except Exception:
        # Keep classification resilient even if torch isn't importable in this runtime.
        pass

    exc_type = type(exc)
    exc_name = exc_type.__name__.lower()
    exc_module = exc_type.__module__.lower()

    if exc_name.endswith("resourceexhaustederror"):
        return OOMExceptionClassification(True, "tensorflow.ResourceExhaustedError")

    if exc_name == "outofmemoryerror" and "torch" in exc_module:
        return OOMExceptionClassification(True, "torch.cuda.OutOfMemoryError")

    message = str(exc).strip().lower()
    for pattern in _OOM_MESSAGE_PATTERNS:
        if pattern in message:
            return OOMExceptionClassification(True, f"message_pattern:{pattern}")

    return OOMExceptionClassification(False, None)


class OOMFlightRecorder:
    """Bounded recorder that writes dump bundles on OOM."""

    def __init__(self, config: OOMFlightRecorderConfig) -> None:
        self.config = config
        bounded_size = max(1, int(config.buffer_size))
        self._events: deque[dict[str, Any]] = deque(maxlen=bounded_size)
        self._events_lock = threading.Lock()
        self._dump_sequence = 0
        self._sequence_lock = threading.Lock()

    def record_event(self, event: dict[str, Any]) -> None:
        """Append one event payload to the in-memory ring buffer."""
        with self._events_lock:
            self._events.append(dict(event))

    def snapshot_events(self) -> list[dict[str, Any]]:
        """Return buffered events in chronological order."""
        with self._events_lock:
            return [dict(event) for event in self._events]

    def dump(
        self,
        *,
        reason: str,
        exception: BaseException,
        context: Optional[str],
        backend: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Write an OOM diagnostic bundle and enforce retention constraints."""
        if not self.config.enabled:
            return None

        root = Path(self.config.dump_dir)
        root.mkdir(parents=True, exist_ok=True)

        bundle_dir = self._next_bundle_dir(root=root, backend=backend)
        bundle_dir.mkdir(parents=True, exist_ok=False)

        events_payload = self.snapshot_events()
        metadata_payload = {
            "reason": reason,
            "exception_type": type(exception).__name__,
            "exception_module": type(exception).__module__,
            "exception_message": str(exception),
            "context": context,
            "backend": backend,
            "captured_event_count": len(events_payload),
            "custom_metadata": dict(metadata or {}),
        }
        environment_payload = {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "system": get_system_info(),
        }

        self._write_json(bundle_dir / "events.json", events_payload)
        self._write_json(bundle_dir / "metadata.json", metadata_payload)
        self._write_json(bundle_dir / "environment.json", environment_payload)

        manifest_payload = {
            "schema_version": 1,
            "bundle_name": bundle_dir.name,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reason": reason,
            "backend": backend,
            "event_count": len(events_payload),
            "files": [
                "manifest.json",
                "events.json",
                "metadata.json",
                "environment.json",
            ],
        }
        self._write_json(bundle_dir / "manifest.json", manifest_payload)

        self._prune_retention(root)
        return str(bundle_dir)

    def _next_bundle_dir(self, *, root: Path, backend: str) -> Path:
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_backend = re.sub(r"[^a-zA-Z0-9_-]+", "_", backend) or "unknown"

        while True:
            with self._sequence_lock:
                self._dump_sequence += 1
                candidate = root / (
                    f"oom_dump_{timestamp_utc}_{os.getpid()}_{safe_backend}_{self._dump_sequence}"
                )
            if not candidate.exists():
                return candidate

    def _prune_retention(self, root: Path) -> None:
        # Retention consistently uses oldest->newest ordering.
        bundles = self._list_bundles_oldest_first(root)

        if self.config.max_dumps > 0 and len(bundles) > self.config.max_dumps:
            for stale in bundles[: -self.config.max_dumps]:
                shutil.rmtree(stale, ignore_errors=True)

        max_total_bytes = self.config.max_total_mb * 1024 * 1024
        if max_total_bytes <= 0:
            return

        bundles = self._list_bundles_oldest_first(root)
        total_bytes = sum(self._bundle_size_bytes(path) for path in bundles)

        while bundles and total_bytes > max_total_bytes:
            oldest = bundles.pop(0)
            bundle_size = self._bundle_size_bytes(oldest)
            shutil.rmtree(oldest, ignore_errors=True)
            total_bytes = max(total_bytes - bundle_size, 0)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    @staticmethod
    def _list_bundles_oldest_first(root: Path) -> list[Path]:
        return sorted(
            [
                path
                for path in root.iterdir()
                if path.is_dir() and path.name.startswith("oom_dump_")
            ],
            key=lambda path: path.stat().st_mtime,
        )

    @staticmethod
    def _bundle_size_bytes(bundle_dir: Path) -> int:
        total = 0
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total


__all__ = [
    "OOMFlightRecorder",
    "OOMFlightRecorderConfig",
    "OOMExceptionClassification",
    "classify_oom_exception",
]
