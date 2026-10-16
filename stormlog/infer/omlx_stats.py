"""Read-only, allowlisted oMLX admin statistics with explicit provenance.

The pinned admin response includes credentials and configuration. They are
never retained: normalization emits only these source-qualified numeric fields.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from .http_transport import BoundedResponse, inference_opener, validate_http_endpoint

SOURCE_REVISION = "013bc3f3a48fa9a4b7dffbfeb6cdc7c1face1c5d"
# raw path -> (unit, metric kind). These values are engine counters/gauges,
# not native allocator samples, request spans or model-compute measurements.
FIELDS = {
    "total_tokens_served": ("tokens", "counter"),
    "total_cached_tokens": ("tokens", "counter"),
    "total_prompt_tokens": ("tokens", "counter"),
    "total_completion_tokens": ("tokens", "counter"),
    "total_requests": ("requests", "counter"),
    "cache_efficiency": ("percent", "gauge"),
    "avg_prefill_tps": ("tokens_per_second", "gauge"),
    "avg_generation_tps": ("tokens_per_second", "gauge"),
    "uptime_seconds": ("seconds", "gauge"),
    "active_models.total_active_requests": ("requests", "gauge"),
    "active_models.total_waiting_requests": ("requests", "gauge"),
}
RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "event_type",
        "session_id",
        "run_id",
        "engine",
        "case_id",
        "phase",
        "marker",
        "source_revision",
        "server_version",
        "version_provenance",
        "counter_scope",
        "model_scope",
        "observation_scope",
        "other_client_contamination",
        "scrape_started_at_ns",
        "scrape_ended_at_ns",
        "observed_at_ns",
        "scrape_elapsed_ns",
        "clock_uncertainty_ns",
        "clock_provenance",
        "metrics",
        "status",
        "error_type",
        "http_status",
        "epoch_change",
        "decreased_fields",
        "preceded_by_outage",
    }
)


def stats_identity(url: str) -> dict[str, Any]:
    validate_http_endpoint(url)
    parsed = urllib.parse.urlparse(url)
    if parsed.path.rstrip("/") != "/admin/api/stats":
        raise ValueError(
            "oMLX statistics require the read-only /admin/api/stats endpoint"
        )
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if set(query) - {"scope", "model"} or any(
        len(values) != 1 for values in query.values()
    ):
        raise ValueError("oMLX stats query supports only one scope and one model")
    scope = query.get("scope", ["session"])[0]
    if scope not in {"session", "alltime"}:
        raise ValueError("oMLX statistics scope must be session or alltime")
    return {
        "counter_scope": scope,
        "model_scope": query.get("model", [""])[0] or None,
        "observation_scope": "engine_admin_aggregate",
        "other_client_contamination": "possible",
    }


def _field(payload: dict[str, Any], path: str) -> float | int | None:
    value: Any = payload
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return (
        value
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
        else None
    )


def normalize_stats(payload: Any, identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("oMLX stats must be an object")
    metrics = []
    for path, (unit, kind) in FIELDS.items():
        value = _field(payload, path)
        metrics.append(
            {
                "raw_path": path,
                "name": path.replace(".", "_"),
                "unit": unit,
                "kind": kind,
                "value": value,
                "validity": "valid" if value is not None else "missing_or_invalid",
                "counter_scope": (
                    identity["counter_scope"] if kind == "counter" else None
                ),
                "scope": (
                    "server"
                    if path.startswith("active_models.")
                    else "model" if identity["model_scope"] else "server"
                ),
            }
        )
    return metrics


class OmlxStatsSampler:
    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        interval: float,
        timeout: float,
        session_id: str,
        run_id: str,
        server_version: str | None,
        cookie: str | None = None,
    ) -> None:
        self.identity = stats_identity(url)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("oMLX statistics interval must be positive and finite")
        self.url, self.api_key, self.interval = url, api_key, interval
        _validate_admin_credentials(api_key, cookie)
        self.cookie = cookie
        self.timeout = min(timeout, 5.0)
        self.session_id, self.run_id, self.server_version = (
            session_id,
            run_id,
            server_version,
        )
        self.opener = inference_opener()
        self.previous: dict[str, float | int] = {}
        self.outage = False

    def fetch(self, case_id: str, phase: str, marker: str) -> dict[str, Any]:
        start_ns, started = time.time_ns(), time.perf_counter()
        record: dict[str, Any] = {
            "schema_version": 1,
            "event_type": "infer.mlx.stats",
            "session_id": self.session_id,
            "run_id": self.run_id,
            "engine": "omlx",
            "case_id": case_id,
            "phase": phase,
            "marker": marker,
            "source_revision": SOURCE_REVISION,
            "server_version": self.server_version,
            "version_provenance": "user_declared" if self.server_version else "unknown",
            **self.identity,
            "scrape_started_at_ns": start_ns,
            "metrics": [],
            "status": "ok",
        }
        try:
            record["metrics"] = self._fetch_metrics(started)
        except Exception as exc:
            record.update(
                status="error",
                error_type=type(exc).__name__,
                http_status=(
                    exc.code if isinstance(exc, urllib.error.HTTPError) else None
                ),
            )
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
        elapsed_ns = round((time.perf_counter() - started) * 1e9)
        end_ns = time.time_ns()
        record.update(
            scrape_ended_at_ns=end_ns,
            observed_at_ns=(start_ns + end_ns) // 2,
            scrape_elapsed_ns=elapsed_ns,
            clock_uncertainty_ns=elapsed_ns // 2 + abs(end_ns - start_ns - elapsed_ns),
            clock_provenance="client_scrape_midpoint_not_server_clock",
        )
        return record

    def _fetch_metrics(self, started: float) -> list[dict[str, Any]]:
        headers = {"Accept": "application/json"}
        # This is exclusively the separately supplied admin key.
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        if self.cookie:
            headers["Cookie"] = "omlx_admin_session=" + self.cookie
        request = urllib.request.Request(self.url, headers=headers, method="GET")
        with self.opener.open(request, timeout=self.timeout) as raw:
            with BoundedResponse(
                raw, deadline=started + self.timeout, max_bytes=1024**2
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        return normalize_stats(payload, self.identity)

    async def sample(
        self,
        append: Callable[[dict[str, Any]], None],
        case_id: str,
        phase: str,
        marker: str,
    ) -> None:
        started_ns, started = time.time_ns(), time.perf_counter()
        try:
            record = await asyncio.to_thread(self.fetch, case_id, phase, marker)
        except asyncio.CancelledError:
            ended_ns = time.time_ns()
            elapsed_ns = round((time.perf_counter() - started) * 1e9)
            record = {
                "schema_version": 1,
                "event_type": "infer.mlx.stats",
                "engine": "omlx",
                "session_id": self.session_id,
                "run_id": self.run_id,
                "case_id": case_id,
                "phase": phase,
                "marker": marker,
                **self.identity,
                "status": "abandoned",
                "metrics": [],
                "source_revision": SOURCE_REVISION,
                "server_version": self.server_version,
                "version_provenance": (
                    "user_declared" if self.server_version else "unknown"
                ),
                "scrape_started_at_ns": started_ns,
                "scrape_ended_at_ns": ended_ns,
                "scrape_elapsed_ns": elapsed_ns,
                "observed_at_ns": (started_ns + ended_ns) // 2,
                "clock_uncertainty_ns": elapsed_ns // 2
                + abs(ended_ns - started_ns - elapsed_ns),
                "clock_provenance": "abandoned_client_scrape_not_server_clock",
            }
            self._epoch(record)
            append(record)
            raise
        self._epoch(record)
        append(record)

    def _epoch(self, record: dict[str, Any]) -> None:
        current = {
            metric["raw_path"]: metric["value"]
            for metric in record["metrics"]
            if metric["value"] is not None
        }
        resets = [
            name
            for name, value in current.items()
            if name in self.previous
            and value < self.previous[name]
            and _epoch_field(name)
        ]
        record["epoch_change"] = (
            "restart_or_uptime_reset"
            if "uptime_seconds" in resets
            else "counter_reset_or_restart" if resets else None
        )
        record["decreased_fields"] = resets
        record["preceded_by_outage"] = self.outage
        self.outage = record["status"] != "ok"
        if not self.outage:
            self.previous = current

    def observe(self, case_id: str, phase: str, marker: str) -> dict[str, Any]:
        """One blocking scrape with epoch/outage tracking, for thread samplers."""
        record = self.fetch(case_id, phase, marker)
        self._epoch(record)
        return record

    @asynccontextmanager
    async def phase(
        self, append: Callable[[dict[str, Any]], None], case_id: str, phase: str
    ) -> AsyncIterator[None]:
        await self.sample(append, case_id, phase, "phase_start")
        stop = asyncio.Event()
        task = asyncio.create_task(self._interval_loop(append, case_id, phase, stop))
        try:
            yield
        finally:
            stop.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self.sample(append, case_id, phase, "phase_end")

    async def _interval_loop(
        self,
        append: Callable[[dict[str, Any]], None],
        case_id: str,
        phase: str,
        stop: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), self.interval)
            except asyncio.TimeoutError:
                await self.sample(append, case_id, phase, "interval")


def _epoch_field(name: str) -> bool:
    return FIELDS[name][1] == "counter" or name == "uptime_seconds"


def _validate_admin_credentials(api_key: str | None, cookie: str | None) -> None:
    if api_key and cookie:
        raise ValueError("Choose one separate admin bearer key or session cookie")
    for value in (api_key, cookie):
        _validate_admin_value(value)
    if cookie is not None and any(ch in cookie for ch in ";,="):
        raise ValueError(
            "Supply only the omlx_admin_session token, not a Cookie header"
        )


def _validate_admin_value(value: str | None) -> None:
    if value is not None and (
        not value
        or len(value) > 8192
        or any(ord(ch) < 33 or ord(ch) > 126 for ch in value)
    ):
        raise ValueError(
            "Admin credentials must be bounded printable ASCII without whitespace"
        )


def validate_stats_record(record: dict[str, Any]) -> None:
    if set(record) - RECORD_FIELDS:
        raise ValueError("Unknown oMLX statistics fields")
    if (
        record.get("event_type") != "infer.mlx.stats"
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
    ):
        raise ValueError("Unsupported MLX statistics event schema")
    if record.get("engine") != "omlx" or record.get("counter_scope") not in {
        "session",
        "alltime",
    }:
        raise ValueError("Invalid oMLX statistics identity/scope")
    if record.get("status") not in {"ok", "error", "abandoned"}:
        raise ValueError("Invalid oMLX statistics status")
    _validate_stats_times(record)
    seen: set[str] = set()
    metrics = record.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("oMLX statistics requires metrics list")
    for metric in metrics:
        _validate_metric(metric, seen)


def _validate_stats_times(record: dict[str, Any]) -> None:
    for key in (
        "scrape_started_at_ns",
        "scrape_ended_at_ns",
        "observed_at_ns",
        "scrape_elapsed_ns",
        "clock_uncertainty_ns",
    ):
        value = record.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid oMLX statistics {key}")
    if (
        not record["scrape_started_at_ns"]
        <= record["observed_at_ns"]
        <= record["scrape_ended_at_ns"]
    ):
        raise ValueError("Invalid oMLX scrape time window")


def _validate_metric(metric: Any, seen: set[str]) -> None:
    if not isinstance(metric, dict) or set(metric) != {
        "raw_path",
        "name",
        "unit",
        "kind",
        "value",
        "validity",
        "counter_scope",
        "scope",
    }:
        raise ValueError("Invalid oMLX metric object")
    path = metric.get("raw_path")
    if path not in FIELDS or path in seen:
        raise ValueError("Unknown or duplicate oMLX metric path")
    seen.add(path)
    unit, kind = FIELDS[path]
    if metric.get("unit") != unit or metric.get("kind") != kind:
        raise ValueError("Invalid oMLX metric units/kind")
    _validate_metric_value(metric.get("value"))


def _validate_metric_value(value: Any) -> None:
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
    ):
        raise ValueError("Invalid oMLX metric value")


def stats_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        record for record in records if record.get("event_type") == "infer.mlx.stats"
    ]
    for record in samples:
        validate_stats_record(record)
    return {
        "observation_scope": "engine_admin_aggregate",
        "available": bool(samples),
        "samples": len(samples),
        "failed_samples": sum(record["status"] != "ok" for record in samples),
        "epoch_changes": sum(
            record.get("epoch_change") is not None for record in samples
        ),
        "other_client_contamination": "possible",
        "request_ownership": "unavailable",
        "metrics": _metric_report(samples),
        "version_provenance": sorted(
            {record.get("version_provenance", "unknown") for record in samples}
        ),
        "counter_scopes": sorted({record["counter_scope"] for record in samples}),
    }


def _metric_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        # Separate model/global and session/lifetime scopes rather than merging
        # counters that happen to use the same raw path.
        scope = f"{sample['counter_scope']}:{sample.get('model_scope') or 'server'}"
        for metric in sample["metrics"]:
            by_path.setdefault(scope + ":" + metric["raw_path"], []).append(metric)
    result = {}
    for path, metrics in by_path.items():
        values = [metric["value"] for metric in metrics if metric["value"] is not None]
        result[path] = {
            "unit": metrics[0]["unit"],
            "kind": metrics[0]["kind"],
            "scope": metrics[0]["scope"],
            "valid_samples": len(values),
            "missing_samples": len(metrics) - len(values),
            "latest": values[-1] if values else None,
            "sampled_max": max(values) if values else None,
            "provenance": "reported_admin_field",
            "source_revision": SOURCE_REVISION,
        }
    return result
