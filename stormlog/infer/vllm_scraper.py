"""Scrape vLLM's ``/metrics`` during a profile and keep each response whole.

The profiler scrapes at the start and end of every phase and on a cadence
inside it. Each scrape becomes one ``infer.vllm_scrape`` record on the
client's clock, so no clock alignment is needed to place it against the
phase windows. A scrape that fails is recorded as a failure with its reason,
never as an empty set of series. Deltas and rates are computed at analysis
time, where resets and restarts can be recognised.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cache_state import redact_url
from .correlation_events import CapabilityEvent, CorrelationContext
from .vllm_metrics import (
    CATALOG,
    VERIFIED_VLLM_VERSION,
    Discovery,
    compact_scrape,
    discover,
    parse_prometheus_text,
)
from .vllm_telemetry import (
    MARKER_INTERVAL,
    SCRAPE_ERROR,
    SCRAPE_OK,
    VllmScrapeRecord,
)

DEFAULT_METRICS_PATH = "/metrics"
AUTO_METRICS_URL = "auto"
CAPABILITY_COMPONENT = "vllm.metrics"


def resolve_metrics_url(endpoint: str, requested: str | None) -> str | None:
    """None when scraping is off; ``auto`` is the endpoint's origin plus ``/metrics``."""
    if requested is None:
        return None
    if requested != AUTO_METRICS_URL:
        return requested
    parts = urllib.parse.urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("--vllm-metrics needs a URL when the endpoint is not http(s)")
    return f"{parts.scheme}://{parts.netloc}{DEFAULT_METRICS_PATH}"


@dataclass(frozen=True)
class FetchResult:
    """What one GET of the metrics URL returned, or why it did not."""

    text: str | None
    http_status: int | None
    error: str | None
    duration_ms: float


def fetch_metrics(
    url: str, *, timeout_seconds: float, api_key: str | None = None
) -> FetchResult:
    """GET the metrics text; every failure becomes a result, never an exception."""
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            elapsed = (time.perf_counter() - started) * 1000.0
            return FetchResult(body, int(response.status), None, elapsed)
    except urllib.error.HTTPError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return FetchResult(None, exc.code, f"HTTP {exc.code}", elapsed)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return FetchResult(None, None, f"{type(exc).__name__}: {exc}", elapsed)


class VllmMetricsScraper:
    """Turn metrics responses into records and remember what they exposed."""

    def __init__(
        self,
        *,
        url: str,
        interval_seconds: float,
        timeout_seconds: float,
        session_id: str,
        run_id: str,
        clock_domain: str,
        api_key: str | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.url = url
        self.source_url = redact_url(url) or url
        self.interval_seconds = interval_seconds
        self.interval_ms = max(1, round(interval_seconds * 1000))
        self.timeout_seconds = timeout_seconds
        self.session_id = session_id
        self.run_id = run_id
        self.clock_domain = clock_domain
        self.api_key = api_key
        self.on_warning = on_warning
        self.ok_scrapes = 0
        self.failed_scrapes = 0
        self.first_discovery: Discovery | None = None
        self.last_error: str | None = None

    def scrape(
        self, *, marker: str, case_id: str | None = None, phase: str | None = None
    ) -> VllmScrapeRecord:
        """Fetch and parse once; the record says what happened either way."""
        observed_at_ns = time.time_ns()
        result = fetch_metrics(
            self.url, timeout_seconds=self.timeout_seconds, api_key=self.api_key
        )
        if result.text is None:
            return self._failed(observed_at_ns, marker, case_id, phase, result)
        try:
            compact = compact_scrape(parse_prometheus_text(result.text))
        except ValueError as exc:
            failed = FetchResult(
                None,
                result.http_status,
                f"unparseable response: {exc}",
                result.duration_ms,
            )
            return self._failed(observed_at_ns, marker, case_id, phase, failed)
        found = discover(compact)
        if self.first_discovery is None:
            self.first_discovery = found
        self.ok_scrapes += 1
        encoded = result.text.encode("utf-8")
        return VllmScrapeRecord(
            session_id=self.session_id,
            run_id=self.run_id,
            observed_at_ns=observed_at_ns,
            source_url=self.source_url,
            marker=marker,
            interval_ms=self.interval_ms,
            status=SCRAPE_OK,
            clock_domain=self.clock_domain,
            case_id=case_id,
            phase=phase,
            duration_ms=result.duration_ms,
            http_status=result.http_status,
            content_digest=hashlib.sha256(encoded).hexdigest(),
            content_bytes=len(encoded),
            scrape=compact,
            discovery=found,
        )

    def _failed(
        self,
        observed_at_ns: int,
        marker: str,
        case_id: str | None,
        phase: str | None,
        result: FetchResult,
    ) -> VllmScrapeRecord:
        self.failed_scrapes += 1
        error = result.error or "metrics request failed"
        if self.last_error is None and self.on_warning is not None:
            self.on_warning(f"vLLM metrics scrape of {self.source_url} failed: {error}")
        self.last_error = error
        return VllmScrapeRecord(
            session_id=self.session_id,
            run_id=self.run_id,
            observed_at_ns=observed_at_ns,
            source_url=self.source_url,
            marker=marker,
            interval_ms=self.interval_ms,
            status=SCRAPE_ERROR,
            clock_domain=self.clock_domain,
            case_id=case_id,
            phase=phase,
            duration_ms=result.duration_ms,
            http_status=result.http_status,
            error=error,
        )

    async def interval_loop(
        self,
        *,
        append: Callable[[dict[str, Any]], None],
        case_id: str,
        phase: str,
        stop_event: asyncio.Event,
    ) -> None:
        """Scrape every interval until the phase ends; the fetch runs off the loop."""
        while True:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass
            else:
                return
            record = await asyncio.to_thread(
                self.scrape, marker=MARKER_INTERVAL, case_id=case_id, phase=phase
            )
            append(record.to_record())

    def config_record(self) -> dict[str, Any]:
        return {
            "url": self.source_url,
            "interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds,
        }

    def capability_event(self, context: CorrelationContext) -> CapabilityEvent:
        """What the engine exposed, as a v2 capability record for the artifact.

        ``supported`` lists the catalog's normalized fields, ``enabled`` the
        ones the first successful scrape exposed, and ``collected`` repeats
        ``enabled`` once at least one scrape succeeded. An endpoint that never
        answered is unavailable with empty lists.
        """
        found = self.first_discovery
        available = found is not None and self.ok_scrapes > 0
        enabled = (
            sorted(CATALOG[name].field for name in found.present)
            if found is not None and available
            else []
        )
        return CapabilityEvent(
            context=context,
            event_id=f"capability:{CAPABILITY_COMPONENT}",
            component=CAPABILITY_COMPONENT,
            available=available,
            supported=(
                sorted(entry.field for entry in CATALOG.values()) if available else []
            ),
            enabled=enabled,
            collected=list(enabled),
            metadata=self._capability_metadata(found),
        )

    def _capability_metadata(self, found: Discovery | None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "url": self.source_url,
            "verified_vllm_version": VERIFIED_VLLM_VERSION,
            "ok_scrapes": self.ok_scrapes,
            "failed_scrapes": self.failed_scrapes,
            "last_error": self.last_error,
            "observation_scope": "engine_aggregate",
            "per_request_attribution": "none",
        }
        if found is not None:
            metadata.update(
                {
                    "unknown_series": list(found.unknown),
                    "deprecated_series": list(found.deprecated_present),
                    "removed_series": list(found.removed_present),
                    "optional_absent": list(found.optional_absent),
                    "engines": list(found.engines),
                }
            )
        return metadata
