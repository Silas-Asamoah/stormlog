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
import http.client
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

from ..scrub import redact_url
from .correlation_events import CapabilityEvent, CorrelationContext
from .vllm_metrics import (
    CATALOG,
    VERIFIED_VLLM_VERSION,
    CompactScrape,
    Discovery,
    ScrapeTooLarge,
    compact_scrape,
    discover,
    parse_prometheus_text,
)
from .vllm_telemetry import MARKER_INTERVAL, SCRAPE_ERROR, SCRAPE_OK, VllmScrapeRecord

DEFAULT_METRICS_PATH = "/metrics"
AUTO_METRICS_URL = "auto"
CAPABILITY_COMPONENT = "vllm.metrics"
# The phase-end scrape of a run being stopped waits at most this long, so a
# server that stopped answering cannot hold Ctrl+C back.
INTERRUPT_SCRAPE_TIMEOUT_SECONDS = 2.0
# A response is read at most this far, and parsed into records only below this
# many series, so a misbehaving endpoint cannot grow the client's memory. A
# vLLM 0.30.0 response for one model is about 90 KB and 360 series.
MAX_SCRAPE_BYTES = 8 * 1024 * 1024
MAX_SCRAPE_SERIES = 20_000


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


def metrics_api_key(
    endpoint: str,
    url: str,
    api_key: str | None,
    on_warning: Callable[[str], None] | None = None,
) -> str | None:
    """The bearer token for scrapes: the endpoint's, and only on its origin.

    A metrics URL on another scheme, host or port is scraped without
    credentials, so the token given for the inference endpoint never goes
    anywhere else; the run says so once.
    """
    if api_key is None or _origin(endpoint) == _origin(url) is not None:
        return api_key
    if on_warning is not None:
        on_warning(
            f"the bearer token is not sent to {redact_url(url) or url}: it is not "
            "the endpoint's origin, so scrapes go out without credentials"
        )
    return None


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str, int | None] | None:
    """Lowercase scheme and host plus the effective port, so ``https://h`` and
    ``https://h:443`` are one origin; None for a URL without a usable port."""
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:
        return None
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    return (scheme, (parts.hostname or "").lower(), port)


@dataclass(frozen=True)
class FetchResult:
    """What one GET of the metrics URL returned, or why it did not."""

    text: str | None
    http_status: int | None
    error: str | None
    duration_ms: float


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect, so the 3xx is answered as an error.

    urllib would follow it and forward the Authorization header to wherever
    it points, so a /metrics on the endpoint's origin that redirects would
    hand the endpoint's token to another origin. A scrape is of one URL:
    the redirect is recorded with its target and never followed.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())


def fetch_metrics(
    url: str,
    *,
    timeout_seconds: float,
    api_key: str | None = None,
    max_bytes: int = MAX_SCRAPE_BYTES,
) -> FetchResult:
    """GET the metrics text; every failure becomes a result, never an exception.

    The body is read at most ``max_bytes + 1`` bytes far: a longer response
    is a failed scrape, and the rest of it is never read.
    """
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    started = time.perf_counter()
    try:
        with _OPENER.open(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            try:
                raw = _read_capped(response, max_bytes)
            except http.client.HTTPException as exc:
                # A body cut short of its Content-Length (IncompleteRead) is
                # an http.client error, not an OSError: still a failed
                # scrape, never the run's end.
                elapsed = (time.perf_counter() - started) * 1000.0
                return FetchResult(
                    None, status, f"{type(exc).__name__}: {exc}", elapsed
                )
            elapsed = (time.perf_counter() - started) * 1000.0
            if len(raw) > max_bytes:
                error = f"oversized: the response is over {max_bytes} bytes"
                return FetchResult(None, status, error, elapsed)
            body = raw.decode("utf-8", errors="replace")
            return FetchResult(body, status, None, elapsed)
    except urllib.error.HTTPError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return FetchResult(None, exc.code, _http_error_text(exc), elapsed)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return FetchResult(None, None, f"{type(exc).__name__}: {exc}", elapsed)


def _read_capped(response: Any, max_bytes: int) -> bytes:
    """At most ``max_bytes + 1`` bytes of the body.

    A bounded read returns short at EOF instead of raising, so a body cut
    short of its Content-Length is raised here as ``IncompleteRead``, as an
    unbounded read would.
    """
    raw = bytes(response.read(max_bytes + 1))
    remaining = getattr(response, "length", None)
    if len(raw) <= max_bytes and remaining:
        raise http.client.IncompleteRead(raw, remaining)
    return raw


def _http_error_text(exc: urllib.error.HTTPError) -> str:
    """``HTTP 503``, or for a refused redirect also where it pointed."""
    location = exc.headers.get("Location") if exc.headers is not None else None
    if 300 <= exc.code < 400 and location:
        return f"HTTP {exc.code}: redirect to {redact_url(location) or location} not followed"
    return f"HTTP {exc.code}"


def _fetch_and_parse(
    url: str,
    timeout_seconds: float,
    api_key: str | None,
    max_bytes: int = MAX_SCRAPE_BYTES,
    max_series: int = MAX_SCRAPE_SERIES,
) -> tuple[FetchResult, CompactScrape | None]:
    """Fetch and parse, touching no scraper state, so the work can run on a
    thread whose result may be dropped."""
    result = fetch_metrics(
        url, timeout_seconds=timeout_seconds, api_key=api_key, max_bytes=max_bytes
    )
    if result.text is None:
        return result, None
    try:
        families = parse_prometheus_text(result.text, max_series=max_series)
        return result, compact_scrape(families)
    except ScrapeTooLarge as exc:
        return _failed_fetch(result, f"oversized: {exc}"), None
    except ValueError as exc:
        return _failed_fetch(result, f"unparseable response: {exc}"), None


def _failed_fetch(result: FetchResult, error: str) -> FetchResult:
    return FetchResult(None, result.http_status, error, result.duration_ms)


T = TypeVar("T")


async def _off_loop(func: Callable[[], T]) -> T:
    """Run ``func`` on a daemon thread; cancelling the await abandons it.

    ``asyncio.to_thread`` would make a cancelled caller, and then the
    interpreter's exit, wait for a fetch blocked on a silent endpoint until
    its socket timeout. A daemon thread is dropped instead, and its result
    is discarded because nothing awaits it any more.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def deliver(outcome: Callable[[], None]) -> None:
        if not future.done():
            outcome()

    def run() -> None:
        try:
            value = func()
        except BaseException as exc:
            outcome = partial(future.set_exception, exc)
        else:
            outcome = partial(future.set_result, value)
        try:
            loop.call_soon_threadsafe(deliver, outcome)
        except RuntimeError:
            pass  # the loop is closed: nobody is waiting

    threading.Thread(target=run, name="stormlog-vllm-scrape", daemon=True).start()
    return await future


def _started() -> tuple[int, int]:
    """A scrape's wall stamp, and a monotonic read to time the scrape by."""
    return time.time_ns(), time.monotonic_ns()


def _sampled(started: tuple[int, int]) -> tuple[int, int]:
    """``(observed_at_ns, completed_at_ns)``: the end is the wall stamp plus
    the monotonic time since, so a wall clock stepped during the scrape
    cannot put the end before the start."""
    wall, mono = started
    return wall, wall + time.monotonic_ns() - mono


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
        max_scrape_bytes: int = MAX_SCRAPE_BYTES,
        max_scrape_series: int = MAX_SCRAPE_SERIES,
    ) -> None:
        self.url = url
        self.source_url = redact_url(url) or url
        self.max_scrape_bytes = max_scrape_bytes
        self.max_scrape_series = max_scrape_series
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
        # Fetches still running on their threads, abandoned ones included.
        self._fetching = 0
        self._fetching_lock = threading.Lock()

    def scrape(
        self,
        *,
        marker: str,
        case_id: str | None = None,
        phase: str | None = None,
        timeout_seconds: float | None = None,
    ) -> VllmScrapeRecord:
        """Fetch and parse once, here; the record says what happened either way.

        ``timeout_seconds`` overrides the scraper's own for one scrape, for
        the one taken on the way out of an interrupted run.
        """
        started = _started()
        result, compact = self._fetch(timeout_seconds)()
        sampled = _sampled(started)
        return self._record(sampled, marker, case_id, phase, result, compact)

    async def scrape_async(
        self,
        *,
        marker: str,
        case_id: str | None = None,
        phase: str | None = None,
        timeout_seconds: float | None = None,
    ) -> VllmScrapeRecord:
        """``scrape`` with the fetch on a thread a cancelled caller abandons.

        Counting and the record happen back on the loop, so a fetch dropped
        by cancellation leaves no trace: no counter moves and no record is
        written for it.
        """
        started = _started()
        fetch = self._fetch(timeout_seconds)

        def tracked() -> tuple[FetchResult, CompactScrape | None]:
            try:
                return fetch()
            finally:
                with self._fetching_lock:
                    self._fetching -= 1

        with self._fetching_lock:
            self._fetching += 1
        result, compact = await _off_loop(tracked)
        # Stamped back on the loop, so the interval also covers the time the
        # fetch thread took to start, which duration_ms does not.
        sampled = _sampled(started)
        return self._record(sampled, marker, case_id, phase, result, compact)

    def fetching(self) -> bool:
        """A fetch is still running on its thread, one whose caller gave up
        waiting for it included: a socket timeout bounds each read, not the
        whole response, so a trickling one can run on for long."""
        with self._fetching_lock:
            return self._fetching > 0

    def abandoned(
        self,
        *,
        marker: str,
        case_id: str | None,
        phase: str | None,
        observed_at_ns: int,
        deadline_seconds: float,
        reason: str | None = None,
    ) -> VllmScrapeRecord:
        """The record of a scrape given up at an overall deadline.

        The fetch itself may still be reading on its thread; its result is
        dropped, so this failed record is the only trace of the scrape.
        """
        error = reason or (
            f"the {deadline_seconds:g} s deadline of the interrupted run passed "
            "while the response was still arriving"
        )
        result = FetchResult(
            None, None, f"abandoned: {error}", deadline_seconds * 1000.0
        )
        sampled = (observed_at_ns, max(observed_at_ns, time.time_ns()))
        return self._failed(sampled, marker, case_id, phase, result)

    def _timeout(self, override: float | None) -> float:
        return self.timeout_seconds if override is None else override

    def _fetch(
        self, timeout_seconds: float | None
    ) -> Callable[[], tuple[FetchResult, CompactScrape | None]]:
        """One bounded fetch-and-parse, as a call that touches no scraper state."""
        return partial(
            _fetch_and_parse,
            self.url,
            self._timeout(timeout_seconds),
            self.api_key,
            self.max_scrape_bytes,
            self.max_scrape_series,
        )

    def _record(
        self,
        sampled: tuple[int, int],
        marker: str,
        case_id: str | None,
        phase: str | None,
        result: FetchResult,
        compact: CompactScrape | None,
    ) -> VllmScrapeRecord:
        if compact is None:
            return self._failed(sampled, marker, case_id, phase, result)
        found = discover(compact)
        if self.first_discovery is None:
            self.first_discovery = found
        self.ok_scrapes += 1
        encoded = (result.text or "").encode("utf-8")
        return VllmScrapeRecord(
            session_id=self.session_id,
            run_id=self.run_id,
            observed_at_ns=sampled[0],
            completed_at_ns=sampled[1],
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
        sampled: tuple[int, int],
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
            observed_at_ns=sampled[0],
            completed_at_ns=sampled[1],
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
        """Scrape every interval until the phase ends; the fetch runs off the loop.

        Cancelling the loop while a fetch is in flight abandons that fetch
        (see ``scrape_async``), so a stop never waits on a silent endpoint.
        """
        while True:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass
            else:
                return
            record = await self.scrape_async(
                marker=MARKER_INTERVAL, case_id=case_id, phase=phase
            )
            append(record.to_record())

    def config_record(self) -> dict[str, Any]:
        return {
            "url": self.source_url,
            "interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "authorization": "bearer" if self.api_key else None,
            "max_scrape_bytes": self.max_scrape_bytes,
            "max_scrape_series": self.max_scrape_series,
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
                    "optional_present": list(found.optional_present),
                    "engines": list(found.engines),
                }
            )
        return metadata
