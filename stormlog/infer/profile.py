"""Active inference profiling runner."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

from .. import __version__
from ..session import (
    SESSION_STATUS_INCOMPLETE,
    SESSION_STATUS_INTERRUPTED,
    create_session_summary,
    finalize_session_summary,
    new_session_id,
    session_summary_to_dict,
    update_session_summary,
)
from .analysis import analyze_inference_events
from .arrivals import CLOSED, arrival_offsets
from .cache_state import cache_state_record, redact_url, reset_cache
from .config import ProfileConfig, WorkloadCase
from .correlation_events import ArtifactIdentityEvent, CorrelationContext
from .events import InferenceRequestEvent, InferenceSummaryEvent, JsonlEventWriter
from .host_clock import host_boot_id, wall_clock_domain
from .open_loop import Arrival, InFlightLimiter, cancel_all, dispatch_schedule
from .openai_client import (
    ChatCompletionResult,
    EndpointHTTPError,
    OpenAIChatCompletionsClient,
)
from .prompts import Prompt, PromptSource
from .samplers import SystemSampler, build_system_sampler
from .tokens import TokenCount, TokenCounter, build_token_counter
from .vllm_scraper import (
    INTERRUPT_SCRAPE_TIMEOUT_SECONDS,
    VllmMetricsScraper,
    metrics_api_key,
)
from .vllm_spans import OTLP_EXTRA_HINT, OtlpSpanReceiver, span_capability_event
from .vllm_telemetry import MARKER_PHASE_END, MARKER_PHASE_START
from .workload import workload_record


class InferenceProfiler:
    """Profile an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        config: ProfileConfig,
        *,
        run_id: str | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.on_warning = on_warning
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
        # Built now so a bad setting fails before the artifact is opened.
        self.prompt_spec = config.prompt_spec()
        self._check_schedules()
        # The repeated prompt by (seed, length): it is the same in every phase.
        self._repeated_prompts: dict[tuple[int, int], Prompt] = {}
        self.client = OpenAIChatCompletionsClient(
            endpoint=config.endpoint,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            api_key=config.api_key,
            max_tokens_field=config.max_tokens_field,
            extra_body=config.extra_body,
        )
        self.request_executor = ThreadPoolExecutor(
            max_workers=max(case.concurrency for case in config.cases()),
            thread_name_prefix="stormlog-infer",
        )
        # Requests on the pool, including ones a drain deadline gave up on:
        # their HTTP calls keep running until they finish or time out.
        self._unfinished: set[Future[Any]] = set()
        # Pool threads remove finished calls while the event loop reads the set.
        self._unfinished_lock = threading.Lock()
        self._opened_artifact = False
        self.vllm_scraper = self._build_vllm_scraper()
        self.span_receiver: OtlpSpanReceiver | None = None
        self._span_receiver_error: str | None = None

    def _build_vllm_scraper(self) -> VllmMetricsScraper | None:
        """The ``/metrics`` scraper when native vLLM telemetry is on."""
        config = self.config
        if config.vllm_metrics_url is None:
            return None
        return VllmMetricsScraper(
            url=config.vllm_metrics_url,
            interval_seconds=config.vllm_metrics_interval_seconds,
            timeout_seconds=config.timeout_seconds,
            session_id=self.session.session_id,
            run_id=self.run_id,
            clock_domain=wall_clock_domain(self.session.host, host_boot_id()),
            api_key=metrics_api_key(
                config.endpoint,
                config.vllm_metrics_url,
                config.api_key,
                on_warning=self.on_warning,
            ),
            on_warning=self.on_warning,
        )

    def _x_request_id(self, request_id: str) -> str:
        """The run-scoped ``X-Request-Id`` sent with a request and recorded on it."""
        return f"stormlog-{self.run_id}-{request_id}"

    def _check_schedules(self) -> None:
        """Refuse an open-loop schedule above the arrival cap up front."""
        config = self.config
        phases = [(config.request_count, config.duration_seconds)]
        if config.warmup_requests > 0:
            phases.append((config.warmup_requests, None))
        for case in config.cases():
            for count, duration_seconds in phases if case.arrival.open_loop else ():
                arrival_offsets(
                    case.arrival,
                    count=count,
                    duration_seconds=duration_seconds,
                    seed=config.seed,
                )

    def run(self) -> dict[str, Any]:
        """Run profiling and return an aggregate report."""
        try:
            return asyncio.run(self._run_async())
        finally:
            self.request_executor.shutdown(wait=True, cancel_futures=True)

    async def _run_async(self) -> dict[str, Any]:
        output_path = Path(self.config.output_path)
        try:
            await self._capture(output_path)
        except BaseException as exc:
            # A crash or Ctrl+C still ends the artifact with a session record,
            # once this run has opened it; an older file at the path is left be.
            if self._opened_artifact:
                self._write_terminal_session(
                    output_path=output_path, report=None, status=_stop_status(exc)
                )
            raise

        try:
            report = analyze_inference_events(output_path)
        except Exception:
            self._write_terminal_session(output_path=output_path, report=None)
            raise
        self._write_terminal_session(output_path=output_path, report=report)
        return report

    async def _capture(self, output_path: Path) -> None:
        self._start_span_receiver()
        with JsonlEventWriter(output_path) as writer:
            self._opened_artifact = True
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
                        "prompts": self.prompt_spec.to_record(),
                        "cache_state": self.config.cache_state,
                        "cache_reset_url": redact_url(self.config.cache_reset_url),
                        "vllm_metrics": (
                            self.vllm_scraper.config_record()
                            if self.vllm_scraper is not None
                            else None
                        ),
                        "vllm_spans": (
                            {
                                **self.span_receiver.config_record(),
                                "drain_seconds": self.config.vllm_spans_drain_seconds,
                            }
                            if self.span_receiver is not None
                            else None
                        ),
                    },
                }
            )
            writer.append(self._artifact_identity().to_record())
            writer.append(
                workload_record(
                    self.config,
                    session_id=self.session.session_id,
                    counter=self.token_counter,
                    prompt_spec=self.prompt_spec,
                )
            )
            stop_sampling = asyncio.Event()
            sample_task = asyncio.create_task(
                self._sample_system_loop(
                    writer=writer,
                    sampler=self.sampler,
                    stop_event=stop_sampling,
                )
            )
            stop_spans = asyncio.Event()
            span_task = asyncio.create_task(
                self._drain_spans_loop(writer=writer, stop_event=stop_spans)
            )
            completed = False
            try:
                for case in self.config.cases():
                    await self._run_case(case=case, writer=writer)
                completed = True
            finally:
                # The helpers are waited for, never awaited: under a real
                # Ctrl+C asyncio.run has cancelled them too, and a bare
                # await would raise here and skip the rest of this block.
                stop_sampling.set()
                await _wait_for(sample_task)
                if completed:
                    await self._wait_for_late_spans()
                stop_spans.set()
                await _wait_for(span_task)
                self._stop_span_receiver(writer)
                # Written on the way out of an interrupted run too, so the
                # artifact says what the engine exposed before it says why
                # the run stopped.
                self._write_capabilities(writer)

    async def _wait_for_late_spans(self) -> None:
        """Keep the receiver up after the last phase for the exporter's last batch.

        vLLM's span exporter flushes on a schedule (5 s by default), so the
        spans of the final requests usually leave the server after the last
        request has been answered. The wait is skipped when no receiver runs.
        """
        if self.span_receiver is None or self.config.vllm_spans_drain_seconds <= 0:
            return
        await asyncio.sleep(self.config.vllm_spans_drain_seconds)

    def _start_span_receiver(self) -> None:
        """Bind the OTLP receiver before any request can produce a span."""
        listen = self.config.vllm_spans_listen
        if listen is None:
            return
        try:
            receiver = OtlpSpanReceiver(
                listen=listen, session_id=self.session.session_id, run_id=self.run_id
            )
            receiver.start()
        except OSError as exc:
            self._span_receiver_error = f"{type(exc).__name__}: {exc}"
            if self.on_warning is not None:
                self.on_warning(
                    f"vLLM span receiver could not listen on {listen}: "
                    f"{self._span_receiver_error}; spans are not collected"
                )
            return
        self.span_receiver = receiver
        if not receiver.protobuf_available and self.on_warning is not None:
            self.on_warning(
                f"vLLM span receiver on {receiver.listen} cannot decode OTLP "
                f"protobuf, which is what vLLM exports: {OTLP_EXTRA_HINT}; "
                "protobuf exports are refused and recorded"
            )

    def _stop_span_receiver(self, writer: JsonlEventWriter) -> None:
        receiver = self.span_receiver
        if receiver is None:
            return
        receiver.stop()
        for record in receiver.drain():
            writer.append(record.to_record())

    async def _drain_spans_loop(
        self, *, writer: JsonlEventWriter, stop_event: asyncio.Event
    ) -> None:
        """Move received spans onto the artifact from the event loop thread."""
        receiver = self.span_receiver
        if receiver is None:
            return
        while not stop_event.is_set():
            for record in receiver.drain():
                writer.append(record.to_record())
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass

    def _write_capabilities(self, writer: JsonlEventWriter) -> None:
        """Say what the engine exposed, once the run knows."""
        if self.vllm_scraper is None and self.config.vllm_spans_listen is None:
            return
        context = self._artifact_identity().context
        if self.vllm_scraper is not None:
            writer.append(self.vllm_scraper.capability_event(context).to_record())
        if self.config.vllm_spans_listen is not None:
            event = span_capability_event(
                context,
                receiver=self.span_receiver,
                listen=self.config.vllm_spans_listen,
                error=self._span_receiver_error,
            )
            writer.append(event.to_record())

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
        status: str = SESSION_STATUS_INCOMPLETE,
    ) -> None:
        session = self.session
        if report is None:
            session = update_session_summary(session, status=status)
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
        # Let calls the previous case gave up on finish first, so the cache
        # reset does not land while they are still running on the server.
        abandoned: _AbandonedWait | None = await self._wait_for_abandoned()
        reset = None
        if self.config.cache_reset_url is not None:
            reset = await asyncio.to_thread(
                reset_cache,
                self.config.cache_reset_url,
                timeout_seconds=self.config.timeout_seconds,
                api_key=self.config.api_key,
            )
            if not reset.succeeded and self.on_warning is not None:
                self.on_warning(
                    f"cache reset failed before {case.case_id} "
                    f"({reset.error or reset.status}); the case is not a cold start"
                )
        writer.append(
            cache_state_record(
                session_id=self.session.session_id,
                case_id=case.case_id,
                requested=self.config.cache_state,
                reset=reset,
                warmup_requests=self.config.warmup_requests,
            )
        )
        if self.config.warmup_requests > 0:
            await self._run_phase(
                case=case,
                writer=writer,
                phase="warmup",
                total_requests=self.config.warmup_requests,
                duration_seconds=None,
                abandoned=abandoned,
            )
            abandoned = None

        await self._run_phase(
            case=case,
            writer=writer,
            phase="measured",
            total_requests=self.config.request_count,
            duration_seconds=self.config.duration_seconds,
            abandoned=abandoned,
        )

    async def _run_phase(
        self,
        *,
        case: WorkloadCase,
        writer: JsonlEventWriter,
        phase: str,
        total_requests: int | None,
        duration_seconds: float | None,
        abandoned: "_AbandonedWait | None" = None,
    ) -> None:
        """Run one phase; ``abandoned`` is a wait already done for it."""
        prompts = PromptSource(
            self.prompt_spec,
            counter=self.token_counter,
            seed=self.config.seed,
            case_id=case.case_id,
            phase=phase,
            input_tokens=case.input_tokens,
            repeated=self._repeated_prompts,
        )
        request = _PhaseRequest(case, writer, prompts, phase)
        prompts.warm()
        if abandoned is None:
            abandoned = await self._wait_for_abandoned()
        window = await self._run_scraped_phase(
            request, total_requests, duration_seconds
        )
        writer.append(
            window.to_record(
                session_id=self.session.session_id,
                request=request,
                drain_timeout_seconds=self._drain_timeout(),
                abandoned=abandoned,
            )
        )

    async def _run_scraped_phase(
        self,
        request: "_PhaseRequest",
        total_requests: int | None,
        duration_seconds: float | None,
    ) -> "_PhaseWindow":
        """Run the phase's arrivals, scraping vLLM around and during them.

        The first scrape lands just before the first send and the last one
        after the drain, so counters and histograms between them cover the
        requests this phase completed; the interval scrapes catch the gauges
        while the phase runs.
        """
        scraper = self.vllm_scraper
        if scraper is None:
            return await self._run_arrivals(request, total_requests, duration_seconds)
        case_id, phase, writer = request.case.case_id, request.phase, request.writer
        await self._scrape(scraper, MARKER_PHASE_START, case_id, phase, writer)
        stop_scraping = asyncio.Event()
        scrape_task = asyncio.create_task(
            scraper.interval_loop(
                append=writer.append,
                case_id=case_id,
                phase=phase,
                stop_event=stop_scraping,
            )
        )
        completed = False
        try:
            window = await self._run_arrivals(request, total_requests, duration_seconds)
            completed = True
            return window
        finally:
            stop_scraping.set()
            # A finished phase keeps a straddling interval scrape, however
            # long its fetch takes. A cancelled one waits briefly for a
            # scrape in flight and then gives it up: its thread is dropped,
            # and its late result with it, so nothing from it is recorded.
            bound = None if completed else INTERRUPT_SCRAPE_TIMEOUT_SECONDS
            if not await _wait_for(scrape_task, bound):
                scrape_task.cancel()
                await _wait_for(scrape_task)
            # A cancelled phase still gets its end scrape, on a short clock.
            if completed:
                await self._scrape(scraper, MARKER_PHASE_END, case_id, phase, writer)
            else:
                await self._bounded_end_scrape(scraper, case_id, phase, writer)

    async def _bounded_end_scrape(
        self,
        scraper: VllmMetricsScraper,
        case_id: str,
        phase: str,
        writer: JsonlEventWriter,
    ) -> None:
        """The end scrape of a cancelled phase, under one overall deadline.

        The short timeout given to the fetch is a socket inactivity timeout,
        so a response that keeps dribbling bytes inside it would hold the
        stop open for as long as it liked. The same bound is applied to the
        scrape as a whole; past it the fetch is left to its daemon thread
        and the record says the deadline passed.
        """
        observed_at_ns = time.time_ns()
        deadline = INTERRUPT_SCRAPE_TIMEOUT_SECONDS
        task = asyncio.ensure_future(
            scraper.scrape_async(
                marker=MARKER_PHASE_END,
                case_id=case_id,
                phase=phase,
                timeout_seconds=deadline,
            )
        )
        if await _wait_for(task, deadline):
            record = task.result()
        else:
            task.cancel()
            await _wait_for(task)
            record = scraper.abandoned(
                marker=MARKER_PHASE_END,
                case_id=case_id,
                phase=phase,
                observed_at_ns=observed_at_ns,
                deadline_seconds=deadline,
            )
        writer.append(record.to_record())

    async def _run_arrivals(
        self,
        request: "_PhaseRequest",
        total_requests: int | None,
        duration_seconds: float | None,
    ) -> "_PhaseWindow":
        if request.case.arrival.open_loop:
            return await self._run_open_phase(request, total_requests, duration_seconds)
        return await self._run_closed_phase(request, total_requests, duration_seconds)

    async def _scrape(
        self,
        scraper: VllmMetricsScraper,
        marker: str,
        case_id: str,
        phase: str,
        writer: JsonlEventWriter,
        timeout_seconds: float | None = None,
    ) -> None:
        record = await scraper.scrape_async(
            marker=marker,
            case_id=case_id,
            phase=phase,
            timeout_seconds=timeout_seconds,
        )
        writer.append(record.to_record())

    def _track(self, call: Future[Any]) -> None:
        with self._unfinished_lock:
            self._unfinished.add(call)
        call.add_done_callback(self._untrack)

    def _untrack(self, call: Future[Any]) -> None:
        with self._unfinished_lock:
            self._unfinished.discard(call)

    def _running_calls(self) -> list[Future[Any]]:
        with self._unfinished_lock:
            return [future for future in self._unfinished if not future.done()]

    async def _wait_for_abandoned(self) -> "_AbandonedWait":
        """Let requests an earlier drain gave up on finish before a phase starts.

        Their HTTP calls still hold pool threads and server capacity, so a
        phase that started alongside them would queue behind them and measure
        their load as its own. Each call is bounded by the request timeout.
        """
        running = self._running_calls()
        if not running:
            return _AbandonedWait()
        started = time.perf_counter()
        await asyncio.wait(
            [asyncio.wrap_future(future) for future in running],
            timeout=self.config.timeout_seconds + 1.0,
        )
        return _AbandonedWait(
            running_at_start=len(running),
            waited_seconds=time.perf_counter() - started,
            still_running=sum(1 for future in running if not future.done()),
        )

    async def _run_closed_phase(
        self,
        request: "_PhaseRequest",
        total_requests: int | None,
        duration_seconds: float | None,
    ) -> "_PhaseWindow":
        case = request.case
        if total_requests is None and duration_seconds is None:
            total_requests = 1
        started_at_ns = time.time_ns()
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
        if duration_seconds is None:
            # Every request counts, so each one finishes or times out.
            await _drain(tasks, timeout=None)
            window_ended_at_ns = counter.last_issued_at_ns or time.time_ns()
        else:
            await _drain(tasks, timeout=duration_seconds + self._drain_timeout())
            window_ended_at_ns = started_at_ns + round(duration_seconds * 1e9)
        return _PhaseWindow(started_at_ns, window_ended_at_ns, time.time_ns())

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
                await self._send(
                    f"{case.case_id}_{phase}_{worker_id}_{request_index}",
                    request,
                    arrival,
                )
            finally:
                limiter.release()

    async def _run_open_phase(
        self,
        request: "_PhaseRequest",
        total_requests: int | None,
        duration_seconds: float | None,
    ) -> "_PhaseWindow":
        case = request.case
        offsets = arrival_offsets(
            case.arrival,
            count=total_requests,
            duration_seconds=duration_seconds,
            seed=self.config.seed,
        )

        async def send(arrival: Arrival) -> None:
            request_id = f"{case.case_id}_{request.phase}_{arrival.index}"
            await self._send(request_id, request, arrival)

        def drop(arrival: Arrival, reason: str) -> None:
            event = self._dropped_event(
                request_id=f"{case.case_id}_{request.phase}_{arrival.index}",
                request=request,
                arrival=arrival,
                reason=reason,
            )
            request.prompts.forget(arrival.index)
            request.writer.append(event.to_record())

        drain_timeout = self._drain_timeout()
        dispatch = await dispatch_schedule(
            offsets,
            mode=case.arrival.mode,
            limiter=InFlightLimiter(case.concurrency),
            overflow=self.config.overflow,
            send=send,
            drop=drop,
            deadline=(
                duration_seconds + drain_timeout
                if duration_seconds is not None
                else None
            ),
        )
        if duration_seconds is None:
            # Counted arrivals are all sent, held ones included, so the
            # window closes with the last send and the whole drain follows.
            window_ended_at_ns = time.time_ns()
        else:
            # The drain is measured from the window end: arrivals held past
            # it have used up part of it, or all of it.
            window_ended_at_ns = dispatch.started_at_ns + round(duration_seconds * 1e9)
            drain_timeout += (window_ended_at_ns - time.time_ns()) / 1e9
        await _drain(dispatch.tasks, timeout=max(drain_timeout, 0.0))
        return _PhaseWindow(
            dispatch.started_at_ns,
            window_ended_at_ns,
            time.time_ns(),
            scheduled_arrivals=len(offsets),
        )

    async def _send(
        self, request_id: str, request: "_PhaseRequest", arrival: Arrival
    ) -> None:
        """Send one request and record it, or record that it was cancelled."""
        sent_at_ns = time.time_ns()
        try:
            event = await self._run_one_request(
                request_id=request_id, request=request, arrival=arrival
            )
        except asyncio.CancelledError:
            cancelled = self._cancelled_event(
                request_id=request_id,
                request=request,
                arrival=arrival,
                sent_at_ns=sent_at_ns,
            )
            request.writer.append(cancelled.to_record())
            request.prompts.forget(arrival.index)
            raise
        # Keep only the prompt's digest once its request is done.
        request.prompts.forget(arrival.index)
        request.writer.append(event.to_record())

    def _drain_timeout(self) -> float:
        if self.config.drain_timeout_seconds is not None:
            return self.config.drain_timeout_seconds
        return self.config.timeout_seconds

    def _request_fields(
        self,
        *,
        request_id: str,
        request: "_PhaseRequest",
        arrival: Arrival,
        prompt: Prompt,
        prompt_count: TokenCount,
        sent: bool = True,
    ) -> dict[str, Any]:
        """Fields every request event carries, whatever its outcome.

        ``prompt_count`` is the server's count when it reported one, an exact
        count made on the request's thread, or the planned size for a
        request that never completed. ``sent`` is False for a request that
        never went out, which therefore carried no ``X-Request-Id``.
        """
        case = request.case
        return {
            "session_id": self.session.session_id,
            "request_id": request_id,
            "x_request_id": self._x_request_id(request_id) if sent else None,
            "case_id": case.case_id,
            "phase": request.phase,
            "endpoint": self.config.endpoint,
            "model": self.config.model,
            "concurrency": None if case.arrival.open_loop else case.concurrency,
            "max_in_flight": case.concurrency if case.arrival.open_loop else None,
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
            "prompt_mode": request.prompts.spec.mode,
            "prompt_id": prompt.prompt_id,
            "prefix_group": prompt.prefix_group,
            "shared_prefix_tokens": prompt.shared_prefix_tokens,
            "prompt_digest": prompt.digest,
        }

    async def _run_one_request(
        self,
        *,
        request_id: str,
        request: "_PhaseRequest",
        arrival: Arrival,
    ) -> InferenceRequestEvent:
        case = request.case
        prompt = request.prompts.take(arrival.index)
        call = self.request_executor.submit(
            self._call_and_count,
            prompt,
            partial(
                self.client.complete,
                prompt=prompt.text,
                output_tokens=case.output_tokens,
                stream=self.config.stream,
                stream_include_usage=self.config.stream_include_usage,
                request_id=self._x_request_id(request_id),
            ),
        )
        self._track(call)
        outcome = await asyncio.wrap_future(call)
        if outcome.error is None:
            try:
                return self._ok_event(request_id, request, arrival, prompt, outcome)
            except Exception as exc:
                outcome = replace(outcome, error=exc)
        return self._failure_event(request_id, request, arrival, prompt, outcome)

    def _call_and_count(
        self, prompt: Prompt, call: Callable[[], ChatCompletionResult]
    ) -> "_TimedCall":
        """Make the call, then any token counts it needs, on this pool thread.

        Counting a long prompt or answer takes milliseconds; on the event
        loop that time would delay every other request's dispatch.
        """
        outcome = _timed_call(call)
        result = outcome.result
        try:
            if result is None:
                return replace(outcome, prompt_count=prompt.count)
            output_count = _resolve_output_count(
                result.usage, result.text, self.token_counter
            )
            prompt_count = _server_prompt_count(result.usage) or prompt.count
        except Exception as exc:
            return replace(outcome, error=exc, prompt_count=prompt.planned_count)
        return replace(outcome, prompt_count=prompt_count, output_count=output_count)

    def _ok_event(
        self,
        request_id: str,
        request: "_PhaseRequest",
        arrival: Arrival,
        prompt: Prompt,
        outcome: "_TimedCall",
    ) -> InferenceRequestEvent:
        result = outcome.result
        output_count = outcome.output_count
        prompt_count = outcome.prompt_count
        assert result is not None and output_count and prompt_count
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
                prompt=prompt,
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

    def _failure_event(
        self,
        request_id: str,
        request: "_PhaseRequest",
        arrival: Arrival,
        prompt: Prompt,
        outcome: "_TimedCall",
    ) -> InferenceRequestEvent:
        error = outcome.error
        assert error is not None
        status, http_status = classify_failure(error)
        return InferenceRequestEvent(
            **self._request_fields(
                request_id=request_id,
                request=request,
                arrival=arrival,
                prompt=prompt,
                prompt_count=outcome.prompt_count or prompt.planned_count,
            ),
            started_at_ns=outcome.started_at_ns,
            ended_at_ns=outcome.ended_at_ns,
            dispatch_lag_ms=_lag_ms(arrival, outcome.started_at_ns),
            status=status,
            e2e_latency_ms=outcome.elapsed_ms,
            ttft_ms=None,
            first_chunk_latency_ms=None,
            error_type=type(error).__name__,
            error_message=str(error),
            http_status=http_status,
        )

    def _cancelled_event(
        self,
        *,
        request_id: str,
        request: "_PhaseRequest",
        arrival: Arrival,
        sent_at_ns: int,
    ) -> InferenceRequestEvent:
        """A request still running when the drain deadline passed."""
        prompt = request.prompts.take(arrival.index)
        return InferenceRequestEvent(
            **self._request_fields(
                request_id=request_id,
                request=request,
                arrival=arrival,
                prompt=prompt,
                prompt_count=prompt.planned_count,
            ),
            started_at_ns=sent_at_ns,
            ended_at_ns=time.time_ns(),
            dispatch_lag_ms=_lag_ms(arrival, sent_at_ns),
            status="cancelled",
            e2e_latency_ms=None,
            ttft_ms=None,
            first_chunk_latency_ms=None,
            error_message="still in flight when the drain deadline passed",
        )

    def _dropped_event(
        self,
        *,
        request_id: str,
        request: "_PhaseRequest",
        arrival: Arrival,
        reason: str,
    ) -> InferenceRequestEvent:
        """A request that was never sent; ``reason`` says why.

        The in-flight limit turned it away, or it was still waiting for a
        slot when the drain deadline passed.
        """
        prompt = request.prompts.take(arrival.index)
        now_ns = time.time_ns()
        return InferenceRequestEvent(
            **self._request_fields(
                request_id=request_id,
                request=request,
                arrival=arrival,
                prompt=prompt,
                prompt_count=prompt.planned_count,
                sent=False,
            ),
            started_at_ns=now_ns,
            ended_at_ns=now_ns,
            status="dropped",
            e2e_latency_ms=None,
            ttft_ms=None,
            first_chunk_latency_ms=None,
            error_message=reason,
        )


@dataclass(frozen=True)
class _PhaseRequest:
    """What every request of one case phase shares."""

    case: WorkloadCase
    writer: JsonlEventWriter
    prompts: PromptSource
    phase: str


@dataclass(frozen=True)
class _TimedCall:
    """A client call timed on the thread that made it, whatever its outcome."""

    started_at_ns: int
    ended_at_ns: int
    elapsed_ms: float
    result: ChatCompletionResult | None = None
    error: Exception | None = None
    prompt_count: TokenCount | None = None
    output_count: TokenCount | None = None


def _timed_call(call: Callable[[], ChatCompletionResult]) -> _TimedCall:
    """Run ``call`` on a pool thread; time starts when the thread picks it up.

    Timing a failure from before the pool would add any wait for a free
    thread to its dispatch lag in one place and not the other; a successful
    call is already timed by the client on its own thread.
    """
    started_at_ns = time.time_ns()
    started = time.perf_counter()
    try:
        result = call()
    except Exception as exc:
        return _TimedCall(
            started_at_ns,
            time.time_ns(),
            (time.perf_counter() - started) * 1000.0,
            error=exc,
        )
    return _TimedCall(
        started_at_ns,
        time.time_ns(),
        (time.perf_counter() - started) * 1000.0,
        result=result,
    )


@dataclass(frozen=True)
class _AbandonedWait:
    """Requests from an earlier drain still running when a phase was ready."""

    running_at_start: int = 0
    waited_seconds: float = 0.0
    still_running: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "running_at_start": self.running_at_start,
            "waited_seconds": self.waited_seconds,
            "still_running": self.still_running,
        }


@dataclass(frozen=True)
class _PhaseWindow:
    """When a phase's arrivals ran and when its last request finished.

    The window is when requests arrive: until the duration ends, or until
    the last scheduled or counted request is sent. The drain follows, until
    every request has finished or the drain deadline has passed, when the
    requests still running are cancelled and the open-loop arrivals still
    waiting for a slot are dropped.
    """

    started_at_ns: int
    window_ended_at_ns: int
    drained_at_ns: int
    scheduled_arrivals: int | None = None

    def to_record(
        self,
        *,
        session_id: str,
        request: _PhaseRequest,
        drain_timeout_seconds: float,
        abandoned: _AbandonedWait,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_type": "infer.phase_window",
            "session_id": session_id,
            "case_id": request.case.case_id,
            "phase": request.phase,
            "arrival_mode": request.case.arrival.mode,
            "started_at_ns": self.started_at_ns,
            "window_ended_at_ns": self.window_ended_at_ns,
            "drained_at_ns": self.drained_at_ns,
            "drain_timeout_seconds": drain_timeout_seconds,
            "scheduled_arrivals": self.scheduled_arrivals,
            "prompts_digest": request.prompts.digest(),
            "abandoned_requests": abandoned.to_record(),
        }


async def _wait_for(task: asyncio.Task[Any], timeout: float | None = None) -> bool:
    """Wait for a helper task; True once it has finished.

    ``await task`` would re-raise the CancelledError of a task that
    ``asyncio.run`` has already cancelled (a Ctrl+C on Python 3.10 cancels
    every task of the run at once) and abort the ``finally`` doing the
    waiting. A finished task's own failure is still raised, so a bug in a
    helper is not hidden.
    """
    done, _pending = await asyncio.wait({task}, timeout=timeout)
    if task in done and not task.cancelled():
        task.result()
    return task in done


async def _drain(tasks: list[asyncio.Task[None]], *, timeout: float | None) -> None:
    """Wait for requests to finish, cancelling any still running at the timeout.

    A cancelled request records itself as cancelled. Other failures inside a
    task are bugs and propagate.
    """
    if not tasks:
        return
    try:
        _done, pending = await asyncio.wait(tasks, timeout=timeout)
    except asyncio.CancelledError:
        # Ctrl+C: requests record themselves while the artifact is still open.
        await cancel_all(tasks)
        raise
    for task in pending:
        task.cancel()
    for result in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(result, Exception):
            raise result


def _stop_status(exc: BaseException) -> str:
    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
        return SESSION_STATUS_INTERRUPTED
    return SESSION_STATUS_INCOMPLETE


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
        self.last_issued_at_ns: int | None = None
        self.lock = asyncio.Lock()

    async def next(self) -> int | None:
        async with self.lock:
            if self.limit is not None and self.value >= self.limit:
                return None
            current = self.value
            self.value += 1
            if self.value == self.limit:
                self.last_issued_at_ns = time.time_ns()
            return current


def run_profile(
    config: ProfileConfig, *, on_warning: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Run an inference profile from a resolved config."""
    return InferenceProfiler(config, on_warning=on_warning).run()


def _server_prompt_count(usage: dict[str, Any] | None) -> TokenCount | None:
    if usage and isinstance(usage.get("prompt_tokens"), int):
        return TokenCount(
            value=int(usage["prompt_tokens"]),
            source="server_usage",
            exact=True,
        )
    return None


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
