"""Audio profiling composed with the shared request scheduler."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .. import __version__
from ..scrub import redact_url
from ..session import new_session_id
from .analysis import analyze_inference_events
from .arrivals import arrival_offsets
from .audio_client import AudioCallError, AudioClient
from .audio_config import AudioCase, AudioProfileConfig
from .audio_events import AudioRequestEvent
from .audio_workload import AudioWorkload
from .correlation_events import ArtifactIdentityEvent, CorrelationContext
from .engine_adapters import get_adapter
from .events import JsonlEventWriter
from .host_clock import host_boot_id, wall_clock_domain
from .http_transport import ignored_proxies
from .open_loop import Arrival
from .profile import classify_failure
from .request_scheduler import LateCalls, RequestScheduler


class AudioProfiler:
    def __init__(self, config: AudioProfileConfig) -> None:
        self.config = config
        self.workload = AudioWorkload(config)
        self.session_id = new_session_id()
        self.run_id = config.run_id or new_session_id()
        self.client = AudioClient(config)
        self.late_calls = LateCalls()
        self.retained_bytes = 0
        self._validate_schedules()
        self.media_root: Path | None = None
        if config.retain_media_dir is not None:
            root = Path(config.retain_media_dir).resolve()
            try:
                root.relative_to(Path(config.output_path).parent.resolve())
            except ValueError as exc:
                raise ValueError(
                    "Retained media directory must be within the artifact directory"
                ) from exc
            self.media_root = root / ("run-" + self.session_id)

    def _validate_schedules(self) -> None:
        config = self.config
        for case in config.cases():
            if case.arrival.open_loop:
                for count, duration in [
                    (config.request_count, config.duration_seconds)
                ] + (
                    [(config.warmup_requests, None)] if config.warmup_requests else []
                ):
                    arrival_offsets(
                        case.arrival,
                        count=count,
                        duration_seconds=duration,
                        seed=config.seed,
                    )

    def run(self) -> dict[str, Any]:
        with ThreadPoolExecutor(
            max_workers=max(case.concurrency for case in self.config.cases()),
            thread_name_prefix="stormlog-audio",
        ) as pool:
            asyncio.run(self._capture(pool))
        report = analyze_inference_events(self.config.output_path)
        summary = {
            "schema_version": 1,
            "event_type": "infer.audio.summary",
            "engine": self.config.engine,
            "task": self.config.task,
            "session_id": self.session_id,
            "summary": report["summary"],
        }
        with Path(self.config.output_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
        return report

    async def _capture(self, pool: ThreadPoolExecutor) -> None:
        config = self.config
        with JsonlEventWriter(config.output_path) as writer:
            writer.append(self._session("running"))
            writer.append(self._artifact_identity().to_record())
            writer.append(self.workload.record(self.session_id))
            status = "completed"
            try:
                for case in config.cases():
                    if config.warmup_requests:
                        await self._phase(
                            case, "warmup", config.warmup_requests, None, writer, pool
                        )
                    await self._phase(
                        case,
                        "measured",
                        config.request_count,
                        config.duration_seconds,
                        writer,
                        pool,
                    )
            except (KeyboardInterrupt, asyncio.CancelledError):
                status = "interrupted"
                raise
            except BaseException:
                status = "incomplete"
                raise
            finally:
                writer.append(self._session(status))

    def _artifact_identity(self) -> ArtifactIdentityEvent:
        host, boot_id = platform.node(), host_boot_id()
        return ArtifactIdentityEvent(
            context=CorrelationContext(
                run_id=self.run_id,
                session_id=self.session_id,
                producer_id="stormlog.infer.profile_audio",
                source="stormlog.infer.profile_audio",
                source_version=__version__,
                host=host,
                pid=os.getpid(),
                clock_domain=wall_clock_domain(host, boot_id),
                clock_kind="wall",
                collection_mode="active",
                provenance="observed",
            ),
            event_id="artifact",
            metadata={"boot_id": boot_id},
            artifact_kind="inference_jsonl",
            created_at_ns=time.time_ns(),
        )

    def _session(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_type": "infer.session",
            "session_id": self.session_id,
            "run_id": self.run_id,
            "task": self.config.task,
            "engine": self.config.engine,
            "timestamp_ns": time.time_ns(),
            "status": status,
            "config": {
                "endpoint": redact_url(self.config.endpoint),
                "model": self.config.model,
                "engine_adapter": get_adapter(
                    self.config.engine, self.config.task
                ).to_record(self.config.task, self.config.server_version),
                "environment_proxies": ignored_proxies(),
            },
        }

    async def _phase(
        self,
        case: AudioCase,
        phase: str,
        count: int | None,
        duration: float | None,
        writer: JsonlEventWriter,
        pool: ThreadPoolExecutor,
    ) -> None:
        abandoned = await self.late_calls.wait(self.config.timeout_seconds)

        async def send(request_id: str, arrival: Arrival) -> None:
            await self._send(request_id, case, phase, arrival, writer, pool)

        def drop(request_id: str, arrival: Arrival, reason: str) -> None:
            fields = self._fields(request_id, case, phase, arrival, sent=False)
            writer.append(
                AudioRequestEvent(
                    **fields,
                    status="dropped",
                    error_type="DroppedArrival",
                    error_message=reason,
                ).to_record()
            )

        scheduler = RequestScheduler(
            case_id=case.case_id,
            phase=phase,
            arrival=case.arrival,
            concurrency=case.concurrency,
            seed=self.config.seed,
            overflow=self.config.overflow,
            drain_timeout=self.config.drain_timeout,
            send=send,
            drop=drop,
        )
        window = await (
            scheduler.open(count, duration)
            if case.arrival.open_loop
            else scheduler.closed(count, duration)
        )
        writer.append(
            {
                "schema_version": 1,
                "event_type": "infer.phase_window",
                "session_id": self.session_id,
                "task": self.config.task,
                "case_id": case.case_id,
                "phase": phase,
                "arrival_mode": case.arrival.mode,
                **asdict(window),
                "drain_timeout_seconds": self.config.drain_timeout,
                "abandoned_requests": abandoned.to_record(),
            }
        )

    def _fields(
        self,
        request_id: str,
        case: AudioCase,
        phase: str,
        arrival: Arrival,
        *,
        sent: bool,
    ) -> dict[str, Any]:
        fixture = self.workload.select(arrival.index)
        now = time.time_ns()
        return {
            "session_id": self.session_id,
            "request_id": request_id,
            "case_id": case.case_id,
            "phase": phase,
            "task": self.config.task,
            "model": self.config.model,
            "fixture_id": fixture.fixture_id,
            "fixture_digest": fixture.content_digest,
            "request_index": arrival.index,
            "started_at_ns": now,
            "ended_at_ns": now,
            "endpoint": redact_url(self.config.endpoint),
            "stream": self.config.stream,
            "arrival_mode": arrival.mode,
            "intended_at_ns": arrival.intended_at_ns,
            "x_request_id": f"stormlog-{self.run_id}-{request_id}" if sent else None,
            "concurrency": None if case.arrival.open_loop else case.concurrency,
            "max_in_flight": case.concurrency if case.arrival.open_loop else None,
            "dispatch_lag_ms": (now - arrival.intended_at_ns) / 1e6 if sent else None,
            "held_for_slot": arrival.held_for_slot,
            "in_flight_at_dispatch": arrival.in_flight_at_dispatch,
            "input_audio_seconds": fixture.audio_seconds,
        }

    async def _send(
        self,
        request_id: str,
        case: AudioCase,
        phase: str,
        arrival: Arrival,
        writer: JsonlEventWriter,
        pool: ThreadPoolExecutor,
    ) -> None:
        fields = self._fields(request_id, case, phase, arrival, sent=True)
        fixture = self.workload.select(arrival.index)
        started_perf = time.perf_counter()
        call = pool.submit(
            self.client.complete, fixture, request_id=fields["x_request_id"]
        )
        self.late_calls.track(call)
        try:
            result = await asyncio.wrap_future(call)
        except asyncio.CancelledError:
            fields.update(
                ended_at_ns=time.time_ns(),
                e2e_latency_ms=(time.perf_counter() - started_perf) * 1000,
            )
            writer.append(
                AudioRequestEvent(
                    **fields, status="cancelled", error_type="CancelledError"
                ).to_record()
            )
            raise
        except AudioCallError as exc:
            status, http_status = classify_failure(exc.cause)
            fields.update(exc.observations)
            writer.append(
                AudioRequestEvent(
                    **fields,
                    status=status,
                    http_status=http_status,
                    error_type=type(exc.cause).__name__,
                    error_message="Audio request failed; response content omitted",
                ).to_record()
            )
            return
        values = asdict(result)
        media = values.pop("media")
        fields.update(values)
        retained = self._retain(request_id, media, fixture.audio)
        writer.append(
            AudioRequestEvent(
                **fields, status="ok", retained_media=retained
            ).to_record()
        )

    def _retain(
        self, request_id: str, response: bytes | None, input_audio: bytes | None
    ) -> dict[str, Any] | None:
        if self.media_root is None:
            return None
        records: dict[str, Any] = {}
        for kind, data in [("request", input_audio), ("response", response)]:
            if data is None:
                continue
            if self.retained_bytes + len(data) > self.config.max_retained_bytes:
                records[kind] = {"status": "limit_exceeded", "retained_bytes": 0}
                continue
            destination = self.media_root / f"{request_id}-{kind}.bin"
            try:
                self.media_root.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(data)
                self.retained_bytes += len(data)
                records[kind] = {
                    "status": "retained",
                    "path": str(
                        destination.relative_to(
                            Path(self.config.output_path).parent.resolve()
                        )
                    ),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            except OSError:
                records[kind] = {"status": "retention_failed", "retained_bytes": 0}
        return records
