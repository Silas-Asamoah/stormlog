"""Active inference profiling runner."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .. import __version__
from ..session import (
    SESSION_STATUS_INCOMPLETE,
    create_session_summary,
    finalize_session_summary,
    new_session_id,
    session_summary_to_dict,
    update_session_summary,
)
from .analysis import analyze_inference_events
from .arrivals import CLOSED, arrival_offsets
from .config import ProfileConfig, WorkloadCase
from .correlation_events import ArtifactIdentityEvent, CorrelationContext
from .events import InferenceRequestEvent, InferenceSummaryEvent, JsonlEventWriter
from .host_clock import host_boot_id, wall_clock_domain
from .open_loop import Arrival, InFlightLimiter, dispatch_schedule
from .openai_client import EndpointHTTPError, OpenAIChatCompletionsClient
from .samplers import SystemSampler, build_system_sampler
from .tokens import TokenCount, TokenCounter, build_token_counter, generate_prompt


class InferenceProfiler:
    """Profile an OpenAI-compatible chat completions endpoint."""

    def __init__(self, config: ProfileConfig, *, run_id: str | None = None) -> None:
        self.config = config
        self.session = create_session_summary(source="stormlog.infer.profile")
        self.run_id = run_id or config.run_id or new_session_id()
        self.token_counter = build_token_counter(
            tokenizer=config.tokenizer,
            model=config.model,
            tokenizer_model=config.tokenizer_model,
            tiktoken_encoding=config.tiktoken_encoding,
            strict=config.strict_token_counts,
        )
        self.sampler = build_system_sampler(config.system_sampler)
        self.client = OpenAIChatCompletionsClient(
            endpoint=config.endpoint,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            api_key=config.api_key,
            max_tokens_field=config.max_tokens_field,
        )
        self.request_executor = ThreadPoolExecutor(
            max_workers=max(case.concurrency for case in config.cases()),
            thread_name_prefix="stormlog-infer",
        )

    def run(self) -> dict[str, Any]:
        """Run profiling and return an aggregate report."""
        try:
            return asyncio.run(self._run_async())
        finally:
            self.request_executor.shutdown(wait=True, cancel_futures=True)

    async def _run_async(self) -> dict[str, Any]:
        output_path = Path(self.config.output_path)
        with JsonlEventWriter(output_path) as writer:
            writer.append(
                {
                    "schema_version": 1,
                    "event_type": "infer.session",
                    "session_id": self.session.session_id,
                    "timestamp_ns": self.session.started_at_ns,
                    "status": "running",
                    "config": {
                        "endpoint": self.config.endpoint,
                        "model": self.config.model,
                        "concurrency": list(self.config.concurrency),
                        "input_tokens": list(self.config.input_tokens),
                        "output_tokens": list(self.config.output_tokens),
                        "duration_seconds": self.config.duration_seconds,
                        "request_count": self.config.request_count,
                        "stream": self.config.stream,
                        "stream_include_usage": self.config.stream_include_usage,
                        "tokenizer": self.config.tokenizer,
                        "system_sampler": self.sampler.name,
                        "arrivals": [
                            spec.to_record() for spec in self.config.arrival_specs()
                        ],
                        "max_in_flight": self.config.max_in_flight,
                        "overflow": self.config.overflow,
                    },
                }
            )
            writer.append(self._artifact_identity().to_record())
            stop_sampling = asyncio.Event()
            sample_task = asyncio.create_task(
                self._sample_system_loop(
                    writer=writer,
                    sampler=self.sampler,
                    stop_event=stop_sampling,
                )
            )
            try:
                for case in self.config.cases():
                    await self._run_case(case=case, writer=writer)
            finally:
                stop_sampling.set()
                await sample_task

        try:
            report = analyze_inference_events(output_path)
        except Exception:
            self._write_terminal_session(output_path=output_path, report=None)
            raise
        self._write_terminal_session(output_path=output_path, report=report)
        return report

    def _artifact_identity(self) -> ArtifactIdentityEvent:
        session = self.session
        boot_id = host_boot_id()
        return ArtifactIdentityEvent(
            context=CorrelationContext(
                run_id=self.run_id,
                session_id=session.session_id,
                producer_id="stormlog.infer.profile",
                source="stormlog.infer.profile",
                source_version=__version__,
                host=session.host,
                pid=session.pid,
                rank=session.rank,
                local_rank=session.local_rank,
                world_size=session.world_size,
                clock_domain=wall_clock_domain(session.host, boot_id),
                clock_kind="wall",
                collection_mode="active",
                provenance="observed",
            ),
            event_id="artifact",
            metadata={"boot_id": boot_id},
            artifact_kind="inference_jsonl",
            created_at_ns=session.started_at_ns,
        )

    def _write_terminal_session(
        self,
        *,
        output_path: Path,
        report: dict[str, Any] | None,
    ) -> None:
        session = self.session
        if report is None:
            session = update_session_summary(
                session,
                status=SESSION_STATUS_INCOMPLETE,
            )
        completed_session = finalize_session_summary(session)
        with output_path.open("a", encoding="utf-8") as handle:
            if report is not None:
                event = InferenceSummaryEvent(
                    session_id=self.session.session_id,
                    timestamp_ns=time.time_ns(),
                    summary=report,
                )
                handle.write(json.dumps(event.to_record(), sort_keys=True) + "\n")
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "event_type": "infer.session",
                        "session_id": self.session.session_id,
                        "timestamp_ns": completed_session.ended_at_ns,
                        "status": completed_session.status,
                        "summary": session_summary_to_dict(completed_session),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    async def _sample_system_loop(
        self,
        *,
        writer: JsonlEventWriter,
        sampler: SystemSampler,
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                sample = await asyncio.to_thread(
                    sampler.sample,
                    session_id=self.session.session_id,
                )
            except Exception:
                sample = None
            if sample is not None:
                writer.append(sample.to_record())
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.config.sample_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def _run_case(
        self,
        *,
        case: WorkloadCase,
        writer: JsonlEventWriter,
    ) -> None:
        prompt = generate_prompt(
            case.input_tokens,
            self.token_counter,
            seed=self.config.seed + case.input_tokens,
        )
        prompt_count = self.token_counter.count_text(prompt)

        if self.config.warmup_requests > 0:
            await self._run_phase(
                case=case,
                writer=writer,
                prompt=prompt,
                prompt_count=prompt_count,
                phase="warmup",
                total_requests=self.config.warmup_requests,
                duration_seconds=None,
            )

        await self._run_phase(
            case=case,
            writer=writer,
            prompt=prompt,
            prompt_count=prompt_count,
            phase="measured",
            total_requests=self.config.request_count,
            duration_seconds=self.config.duration_seconds,
        )

    async def _run_phase(
        self,
        *,
        case: WorkloadCase,
        writer: JsonlEventWriter,
        prompt: str,
        prompt_count: TokenCount,
        phase: str,
        total_requests: int | None,
        duration_seconds: float | None,
    ) -> None:
        request = _PhaseRequest(case, writer, prompt, prompt_count, phase)
        if case.arrival.open_loop:
            await self._run_open_phase(request, total_requests, duration_seconds)
            return
        if total_requests is None and duration_seconds is None:
            total_requests = 1
        counter = _RequestCounter(limit=total_requests)
        limiter = InFlightLimiter(case.concurrency)
        end_time = (
            time.monotonic() + duration_seconds
            if duration_seconds is not None
            else None
        )
        tasks = [
            asyncio.create_task(
                self._worker(
                    worker_id=index,
                    request=request,
                    counter=counter,
                    limiter=limiter,
                    end_time=end_time,
                )
            )
            for index in range(case.concurrency)
        ]
        await asyncio.gather(*tasks)

    async def _worker(
        self,
        *,
        worker_id: int,
        request: "_PhaseRequest",
        counter: "_RequestCounter",
        limiter: InFlightLimiter,
        end_time: float | None,
    ) -> None:
        case, phase = request.case, request.phase
        while True:
            if end_time is not None and time.monotonic() >= end_time:
                return
            request_index = await counter.next()
            if request_index is None:
                return
            # In a closed loop a request is due as soon as its worker is free.
            in_flight = await limiter.acquire()
            arrival = Arrival(
                index=request_index,
                mode=CLOSED,
                intended_at_ns=time.time_ns(),
                in_flight_at_dispatch=in_flight,
            )
            try:
                event = await self._run_one_request(
                    request_id=f"{case.case_id}_{phase}_{worker_id}_{request_index}",
                    request=request,
                    arrival=arrival,
                )
            finally:
                limiter.release()
            request.writer.append(event.to_record())

    async def _run_open_phase(
        self,
        request: "_PhaseRequest",
        total_requests: int | None,
        duration_seconds: float | None,
    ) -> None:
        case = request.case
        offsets = arrival_offsets(
            case.arrival,
            count=total_requests,
            duration_seconds=duration_seconds,
            seed=self.config.seed,
        )

        async def send(arrival: Arrival) -> None:
            event = await self._run_one_request(
                request_id=f"{case.case_id}_{request.phase}_{arrival.index}",
                request=request,
                arrival=arrival,
            )
            request.writer.append(event.to_record())

        def drop(arrival: Arrival) -> None:
            event = self._dropped_event(
                request_id=f"{case.case_id}_{request.phase}_{arrival.index}",
                request=request,
                arrival=arrival,
            )
            request.writer.append(event.to_record())

        tasks = await dispatch_schedule(
            offsets,
            mode=case.arrival.mode,
            limiter=InFlightLimiter(case.concurrency),
            overflow=self.config.overflow,
            send=send,
            drop=drop,
        )
        await asyncio.gather(*tasks)

    def _request_fields(
        self,
        *,
        request_id: str,
        request: "_PhaseRequest",
        arrival: Arrival,
        prompt_count: TokenCount,
    ) -> dict[str, Any]:
        """Fields every request event carries, whatever its outcome."""
        case = request.case
        return {
            "session_id": self.session.session_id,
            "request_id": request_id,
            "case_id": case.case_id,
            "phase": request.phase,
            "endpoint": self.config.endpoint,
            "model": self.config.model,
            "concurrency": case.concurrency,
            "target_input_tokens": case.input_tokens,
            "target_output_tokens": case.output_tokens,
            "stream": self.config.stream,
            "prompt_tokens": prompt_count.value,
            "prompt_token_source": prompt_count.source,
            "prompt_token_exact": prompt_count.exact,
            "arrival_mode": arrival.mode,
            "request_index": arrival.index,
            "intended_at_ns": arrival.intended_at_ns,
            "held_for_slot": arrival.held_for_slot,
            "in_flight_at_dispatch": arrival.in_flight_at_dispatch,
        }

    async def _run_one_request(
        self,
        *,
        request_id: str,
        request: "_PhaseRequest",
        arrival: Arrival,
    ) -> InferenceRequestEvent:
        case = request.case
        started_at_ns = time.time_ns()
        started_perf = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.request_executor,
                partial(
                    self.client.complete,
                    prompt=request.prompt,
                    output_tokens=case.output_tokens,
                    stream=self.config.stream,
                    stream_include_usage=self.config.stream_include_usage,
                ),
            )
            output_count = _resolve_output_count(
                result.usage,
                result.text,
                self.token_counter,
            )
            prompt_count = _resolve_prompt_count(result.usage, request.prompt_count)
            total_tokens = _resolve_total_tokens(
                result.usage,
                prompt_count,
                output_count,
            )
            return InferenceRequestEvent(
                **self._request_fields(
                    request_id=request_id,
                    request=request,
                    arrival=arrival,
                    prompt_count=prompt_count,
                ),
                started_at_ns=result.started_at_ns,
                ended_at_ns=result.ended_at_ns,
                dispatch_lag_ms=_lag_ms(arrival, result.started_at_ns),
                status="ok",
                e2e_latency_ms=result.e2e_latency_ms,
                ttft_ms=result.ttft_ms,
                first_chunk_latency_ms=result.first_chunk_latency_ms,
                chunk_interarrival_ms=result.chunk_interarrival_ms,
                output_tokens=output_count.value,
                output_token_source=output_count.source,
                output_token_exact=output_count.exact,
                total_tokens=total_tokens,
                finish_reason=result.finish_reason,
            )
        except Exception as exc:
            ended_at_ns = time.time_ns()
            status, http_status = classify_failure(exc)
            return InferenceRequestEvent(
                **self._request_fields(
                    request_id=request_id,
                    request=request,
                    arrival=arrival,
                    prompt_count=request.prompt_count,
                ),
                started_at_ns=started_at_ns,
                ended_at_ns=ended_at_ns,
                dispatch_lag_ms=_lag_ms(arrival, started_at_ns),
                status=status,
                e2e_latency_ms=(time.perf_counter() - started_perf) * 1000.0,
                ttft_ms=None,
                first_chunk_latency_ms=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
                http_status=http_status,
            )

    def _dropped_event(
        self,
        *,
        request_id: str,
        request: "_PhaseRequest",
        arrival: Arrival,
    ) -> InferenceRequestEvent:
        """A request the in-flight limit turned away; it was never sent."""
        now_ns = time.time_ns()
        return InferenceRequestEvent(
            **self._request_fields(
                request_id=request_id,
                request=request,
                arrival=arrival,
                prompt_count=request.prompt_count,
            ),
            started_at_ns=now_ns,
            ended_at_ns=now_ns,
            status="dropped",
            e2e_latency_ms=None,
            ttft_ms=None,
            first_chunk_latency_ms=None,
            error_message=f"in-flight limit of {request.case.concurrency} reached",
        )


@dataclass(frozen=True)
class _PhaseRequest:
    """What every request of one case phase shares."""

    case: WorkloadCase
    writer: JsonlEventWriter
    prompt: str
    prompt_count: TokenCount
    phase: str


def _lag_ms(arrival: Arrival, sent_at_ns: int) -> float:
    return (sent_at_ns - arrival.intended_at_ns) / 1_000_000.0


# The server declined the request: rate limited or overloaded.
REJECTED_HTTP_STATUSES = frozenset({429, 503})


def classify_failure(exc: BaseException) -> tuple[str, int | None]:
    """Return the request status for a failure and its HTTP status, if any."""
    http_status = exc.status if isinstance(exc, EndpointHTTPError) else None
    if http_status in REJECTED_HTTP_STATUSES:
        return "rejected", http_status
    if _is_timeout(exc):
        return "timeout", http_status
    return "error", http_status


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    # urllib wraps a connect timeout in URLError.
    return isinstance(exc, urllib.error.URLError) and isinstance(
        exc.reason, TimeoutError
    )


class _RequestCounter:
    def __init__(self, *, limit: int | None) -> None:
        self.limit = limit
        self.value = 0
        self.lock = asyncio.Lock()

    async def next(self) -> int | None:
        async with self.lock:
            if self.limit is not None and self.value >= self.limit:
                return None
            current = self.value
            self.value += 1
            return current


def run_profile(config: ProfileConfig) -> dict[str, Any]:
    """Run an inference profile from a resolved config."""
    return InferenceProfiler(config).run()


def _resolve_prompt_count(
    usage: dict[str, Any] | None,
    fallback: TokenCount,
) -> TokenCount:
    if usage and isinstance(usage.get("prompt_tokens"), int):
        return TokenCount(
            value=int(usage["prompt_tokens"]),
            source="server_usage",
            exact=True,
        )
    return fallback


def _resolve_output_count(
    usage: dict[str, Any] | None,
    text: str,
    counter: TokenCounter,
) -> TokenCount:
    if usage and isinstance(usage.get("completion_tokens"), int):
        return TokenCount(
            value=int(usage["completion_tokens"]),
            source="server_usage",
            exact=True,
        )
    return counter.count_text(text)


def _resolve_total_tokens(
    usage: dict[str, Any] | None,
    prompt: TokenCount,
    output: TokenCount,
) -> int | None:
    if usage and isinstance(usage.get("total_tokens"), int):
        return int(usage["total_tokens"])
    return prompt.value + output.value
