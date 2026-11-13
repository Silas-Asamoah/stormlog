"""Inference artifacts for Studio's tests, written by Stormlog's own code.

Nothing here writes JSON by hand. ``profile_artifact`` runs the real
``InferenceProfiler`` against ``fake_openai``; ``with_execution_import``
runs the real ``infer import-execution`` over hook logs built with
``tests/vllm_execution_helpers.py``; ``synthetic_artifact`` writes request
events through ``JsonlEventWriter`` and the infer dataclasses, around the
header, phase and session records of a real ``InferenceProfiler`` that
never sends. The other helpers cut, grow, concatenate or rewrite those
files the way a killed run, a running one, ``cat`` or a rerun to the same
path would (design §4.2, §8.2, §10.3 C5).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import math
import random
import signal
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from unittest import mock

from stormlog.cpu_profiler import _CPU_MEMORY_CAPABILITIES
from stormlog.exit_codes import ExitCode
from stormlog.infer.arrivals import CLOSED, FIXED_RATE, RATE_MODES, ArrivalSpec
from stormlog.infer.audio_events import AudioRequestEvent
from stormlog.infer.cache_state import cache_state_record
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.correlation_codec import expand_inference_records
from stormlog.infer.events import (
    InferencePhaseStartEvent,
    InferenceRequestEvent,
    InferenceSystemSample,
    JsonlEventWriter,
)
from stormlog.infer.profile import InferenceProfiler, _PhaseRequest, _PhaseWindow
from stormlog.infer.prompts import PromptSource
from stormlog.infer.request_scheduler import AbandonedWait
from stormlog.telemetry import (
    SCHEMA_VERSION_V4,
    TelemetryEventV4,
    telemetry_event_to_dict,
)
from stormlog.telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig
from tests.studio.fake_openai import CasePolicy, fake_openai
from tests.vllm_execution_helpers import (
    SECOND,
    WALL_OFFSET,
    alias,
    completed,
    done,
    engine_log,
    goodbye,
    heartbeat,
    member,
    scheduled,
    terminal,
)

MODEL = "fake-model"


@dataclass(frozen=True)
class CaseSpec:
    """One workload case of a profile run (design §10.3 C5).

    ``fail_every`` fails every Nth measured request at the server and
    ``all_fail`` every request; ``usage`` False makes the server leave out
    token counts, so the profiler estimates them. One profile run sends the
    same request count and the same stream setting to every case, and its
    cases are the product of its concurrencies and token lengths, so the
    cases given together must agree on those (``ProfileConfig.cases``).
    In an open-loop run ``concurrency`` is the in-flight limit, one for the
    whole run. ``stop_after`` ends the measured window through the run's
    stop file when the Nth measured request reaches the server;
    ``interrupt_after`` presses Ctrl+C then. ``slow_every`` holds every Nth
    measured request for ``delay_seconds`` at the server: past the run's
    ``timeout_seconds``, the profiler records a timeout.
    """

    concurrency: int
    input_tokens: int
    output_tokens: int
    requests: int
    fail_every: int | None = None
    all_fail: bool = False
    stream: bool = True
    usage: bool = True
    stop_after: int | None = None
    interrupt_after: int | None = None
    slow_every: int | None = None
    delay_seconds: float = 0.0

    @property
    def case_id(self) -> str:
        """The ID the profiler gives this case (closed-loop arrivals)."""
        return f"c{self.concurrency}_in{self.input_tokens}_out{self.output_tokens}"

    def run_case_id(self, arrival: ArrivalSpec) -> str:
        """The ID the profiler gives this case under ``arrival``."""
        if not arrival.open_loop:
            return self.case_id
        return f"{arrival.case_label()}_in{self.input_tokens}_out{self.output_tokens}"

    def policy(self) -> CasePolicy:
        return CasePolicy(
            fail_every=self.fail_every,
            all_fail=self.all_fail,
            usage=self.usage,
            stop_after=self.stop_after,
            interrupt_after=self.interrupt_after,
            slow_every=self.slow_every,
            delay_seconds=self.delay_seconds,
        )


def arrival_spec(arrival: str, rate_per_second: float | None) -> ArrivalSpec:
    """A run's arrival shape: ``closed``, or a rate mode with its rate."""
    if arrival == CLOSED:
        if rate_per_second is not None:
            raise ValueError("a closed loop takes no rate")
        return ArrivalSpec()
    if arrival not in RATE_MODES:
        raise ValueError(f"arrival must be closed or one of {sorted(RATE_MODES)}")
    return ArrivalSpec(mode=arrival, rate_per_second=rate_per_second)


def _unique(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(values))


def _run_shape(
    cases: Sequence[CaseSpec],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """The concurrencies and token lengths of one run holding ``cases``."""
    if not cases:
        raise ValueError("a profile needs at least one case")
    if len({(case.requests, case.stream) for case in cases}) != 1:
        raise ValueError("every case of one run has the same requests and stream")
    shape = (
        _unique([case.concurrency for case in cases]),
        _unique([case.input_tokens for case in cases]),
        _unique([case.output_tokens for case in cases]),
    )
    given = [
        (case.concurrency, case.input_tokens, case.output_tokens) for case in cases
    ]
    if len(set(given)) != len(given) or set(given) != set(itertools.product(*shape)):
        raise ValueError("the cases must be the product of their shapes, once each")
    return shape


def profile_config(
    cases: Sequence[CaseSpec],
    *,
    endpoint: str,
    output: Path,
    warmup: int,
    seed: int,
    system_sampler: str = "none",
    sample_interval_seconds: float = 1.0,
    arrival: str = CLOSED,
    rate_per_second: float | None = None,
    stop_file: Path | None = None,
    timeout_seconds: float = 60.0,
) -> ProfileConfig:
    """The config one profile run of ``cases`` takes.

    Raises ``ValueError`` for cases one run cannot hold: mixed request
    counts or stream settings, or a set that is not the full product of its
    concurrencies and token lengths. ``system_sampler`` and
    ``sample_interval_seconds`` are the run's, as ``--system-sampler`` and
    its interval set them. ``arrival`` and ``rate_per_second`` are the
    run's too (``--arrival``, ``--rate``): an open loop sends every case at
    that rate with the cases' one ``concurrency`` as its in-flight limit.
    ``stop_file`` is ``--stop-file`` and ``timeout_seconds`` ``--timeout``,
    which also bounds the drain.
    """
    concurrency, inputs, outputs = _run_shape(cases)
    spec = arrival_spec(arrival, rate_per_second)
    if spec.open_loop and len(concurrency) != 1:
        raise ValueError("an open-loop run has one in-flight limit for every case")
    return ProfileConfig(
        arrival_mode=spec.mode,
        rates=() if spec.rate_per_second is None else (spec.rate_per_second,),
        max_in_flight=concurrency[0],
        stop_file=None if stop_file is None else str(stop_file),
        timeout_seconds=timeout_seconds,
        endpoint=endpoint,
        model=MODEL,
        concurrency=concurrency,
        input_tokens=inputs,
        output_tokens=outputs,
        output_path=str(output),
        request_count=cases[0].requests,
        stream=cases[0].stream,
        warmup_requests=warmup,
        seed=seed,
        system_sampler=system_sampler,
        sample_interval_seconds=sample_interval_seconds,
        tokenizer="none",
        server_probe="none",
    )


def profile_artifact(
    directory: Path,
    cases: Sequence[CaseSpec],
    *,
    name: str = "run.jsonl",
    warmup: int = 2,
    seed: int = 0,
    system_sampler: str = "none",
    sample_interval_seconds: float = 1.0,
    arrival: str = CLOSED,
    rate_per_second: float | None = None,
    timeout_seconds: float = 60.0,
) -> Path:
    """Profile ``fake_openai`` with the real ``InferenceProfiler``.

    The artifact is complete: header records, ``warmup`` requests and the
    measured ones per case, phase windows, the analyzer's summary and the
    terminal session record. An existing file at the path is truncated and
    rewritten in place, as a rerun of ``infer profile`` does.
    ``system_sampler="psutil"`` writes the run's real ``infer.system_sample``
    rows (this process's RSS), one every ``sample_interval_seconds``.
    ``arrival`` (``fixed-rate`` or ``poisson``) and ``rate_per_second``
    make the run open-loop, for every case: one profile run has one arrival
    shape per rate, as ``requests`` and ``stream`` are the run's.
    ``timeout_seconds`` is the run's request timeout (``--timeout``).

    A case with ``stop_after`` ends its measured window early through
    ``<name>.stop``, which is removed first. A case with
    ``interrupt_after`` stops the run with a Ctrl+C: the artifact ends with
    an ``interrupted`` session record, the case has no phase window and the
    cases after it never start. That needs the main thread; the profiler's
    own handler takes the signal for the run.
    """
    output = Path(directory) / name
    spec = arrival_spec(arrival, rate_per_second)
    policies = {case.run_case_id(spec): case.policy() for case in cases}
    stop_file = output.with_suffix(".stop")
    stop_file.unlink(missing_ok=True)
    with fake_openai(policies, stop_file=stop_file) as server:
        config = profile_config(
            cases,
            endpoint=server.endpoint,
            output=output,
            warmup=warmup,
            seed=seed,
            system_sampler=system_sampler,
            sample_interval_seconds=sample_interval_seconds,
            arrival=arrival,
            rate_per_second=rate_per_second,
            stop_file=stop_file if _stops(cases) else None,
            timeout_seconds=timeout_seconds,
        )
        _run(InferenceProfiler(config), interrupted=_interrupts(cases))
        if server.state.interrupted != _interrupts(cases):
            raise RuntimeError("the run ended before its interrupt was sent")
    return output


def _stops(cases: Sequence[CaseSpec]) -> bool:
    return any(case.stop_after is not None for case in cases)


def _interrupts(cases: Sequence[CaseSpec]) -> bool:
    return any(case.interrupt_after is not None for case in cases)


def _run(profiler: InferenceProfiler, *, interrupted: bool) -> None:
    """Run ``profiler``; a run meant to be interrupted must end so."""
    with contextlib.redirect_stdout(io.StringIO()):
        if not interrupted:
            profiler.run()
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("an interrupted profile needs the main thread")
        # A backgrounded process starts with SIGINT ignored; the profiler
        # only takes the signal over from Python's default handler.
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            profiler.run()
        except KeyboardInterrupt:
            return
        finally:
            signal.signal(signal.SIGINT, previous)
    raise RuntimeError("the profile was meant to be interrupted, and completed")


# --- Synthetic artifacts -------------------------------------------------

# Where a synthetic run starts on the wall clock, so a seed gives the same
# request times on every machine (2026-09-21T13:46:40Z).
SYNTHETIC_ORIGIN_NS = 1_790_000_000_000_000_000
# Gap and TTFT values are drawn once per seed into pools; each request takes
# a slice, so a large artifact costs no random draws per gap.
_POOL = 8192
MAX_SYNTHETIC_GAPS = 1024
TokenMode = Literal["exact", "estimated", "missing", "mixed"]
_TOKEN_CYCLE: tuple[TokenMode, ...] = ("exact", "estimated", "missing")


@dataclass(frozen=True)
class _Pools:
    gaps: list[float]
    prefix: list[float]  # prefix[k] is sum(gaps[:k])
    ttft: list[float]

    @classmethod
    def draw(cls, seed: int) -> _Pools:
        rng = random.Random(seed)
        gaps = [round(2.0 + 18.0 * rng.random(), 3) for _ in range(2 * _POOL)]
        ttft = [round(15.0 + 85.0 * rng.random(), 3) for _ in range(_POOL)]
        return cls(gaps, list(itertools.accumulate(gaps, initial=0.0)), ttft)


@dataclass(frozen=True)
class SyntheticPlan:
    """What ``synthetic_artifact`` writes for its one case."""

    requests: int
    gaps_per_request: int
    concurrency: int = 64
    input_tokens: int = 128
    output_tokens: int = 64
    fail_every: int | None = None
    tokens: TokenMode = "exact"
    nan_gap: bool = False

    @property
    def case_id(self) -> str:
        return f"c{self.concurrency}_in{self.input_tokens}_out{self.output_tokens}"


GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True)
class SamplePlan:
    """The ``infer.system_sample`` rows ``synthetic_artifact`` writes.

    One tick every ``interval_ms`` from the measured phase's start to its
    drain. Each tick gives a ``psutil`` row (the client's RSS) when
    ``client`` is set, and an ``nvidia-smi`` row (device memory and
    utilisation, ``client_local`` like every client sample) for each of
    ``devices``. ``stale_device`` writes nothing in ticks
    ``gap_ticks[0]`` to ``gap_ticks[1] - 1`` and from ``stop_tick`` on, so
    its series has a gap and then goes stale; the real samplers write no row
    when they cannot read a value, so that is how a missing sample looks.
    Values are a function of the tick and device only. A real run has one
    sampler watching one device (``samplers.py``); mixing two here exercises
    grouping by ``(sampler, device_id)``.
    """

    interval_ms: int = 50
    client: bool = True
    devices: tuple[int, ...] = (0, 1)
    stale_device: int | None = 1
    gap_ticks: tuple[int, int] = (6, 12)
    stop_tick: int = 24

    @property
    def sampler(self) -> str:
        """The sampler the header records name."""
        return "psutil" if self.client else "nvidia-smi"

    def writes(self, device: int, tick: int) -> bool:
        if device != self.stale_device:
            return True
        start, end = self.gap_ticks
        return not (start <= tick < end or tick >= self.stop_tick)

    def rows(self, session_id: str, tick: int, at_ns: int) -> Iterator[dict[str, Any]]:
        """The rows of one tick, client first, then by device ID."""
        if self.client:
            yield InferenceSystemSample(
                session_id=session_id,
                timestamp_ns=at_ns,
                sampler="psutil",
                process_rss_bytes=180 * MIB + (tick * 7 % 17) * MIB,
            ).to_record()
        for device in self.devices:
            if self.writes(device, tick):
                yield _device_sample(session_id, device, tick, at_ns + 1_000 * device)


def _device_sample(
    session_id: str, device: int, tick: int, at_ns: int
) -> dict[str, Any]:
    used = (8 + 2 * device) * GIB + (tick * 37 % 101) * MIB
    return InferenceSystemSample(
        session_id=session_id,
        timestamp_ns=at_ns,
        sampler="nvidia-smi",
        device_id=device,
        device_name="Fake GPU 80GB",
        device_used_bytes=used,
        device_free_bytes=80 * GIB - used,
        device_total_bytes=80 * GIB,
        gpu_utilization_percent=float(tick * 13 % 100),
    ).to_record()


class _SampleClock:
    """Writes a ``SamplePlan``'s ticks as the phase's requests pass them."""

    def __init__(
        self, plan: SamplePlan | None, session_id: str, started_ns: int
    ) -> None:
        self.plan = plan
        self.session_id = session_id
        self.started_ns = started_ns
        self.tick = 0

    def until(self, writer: JsonlEventWriter, at_ns: int) -> None:
        """Write every tick at or before ``at_ns`` not yet written."""
        plan = self.plan
        if plan is None:
            return
        step = plan.interval_ms * 1_000_000
        while self.started_ns + self.tick * step <= at_ns:
            tick_ns = self.started_ns + self.tick * step
            for row in plan.rows(self.session_id, self.tick, tick_ns):
                writer.append(row)
            self.tick += 1


def _tokens(plan: SyntheticPlan, index: int, chunks: int) -> dict[str, Any]:
    """The token fields of a successful request in ``plan``'s mode."""
    mode = plan.tokens
    if mode == "mixed":
        mode = _TOKEN_CYCLE[index % len(_TOKEN_CYCLE)]
    exact = mode == "exact"
    source = "server_usage" if exact else "estimated"
    output = None if mode == "missing" else chunks
    return {
        "prompt_tokens": plan.input_tokens,
        "prompt_token_source": source,
        "prompt_token_exact": exact,
        "output_tokens": output,
        "output_token_source": "unknown" if output is None else source,
        "output_token_exact": exact,
        "total_tokens": None if output is None else plan.input_tokens + output,
    }


def _ok_fields(
    plan: SyntheticPlan, pools: _Pools, index: int
) -> tuple[dict[str, Any], float]:
    """A successful request's outcome fields, and its end-to-end ms."""
    count = plan.gaps_per_request
    start = (index * 7919) % (len(pools.gaps) - count)
    gaps = pools.gaps[start : start + count]
    if plan.nan_gap and index == 0 and count:
        # json.dumps writes it as a bare NaN, as events.py would.
        gaps[count // 2] = math.nan
    ttft = pools.ttft[(index * 104729) % _POOL]
    e2e = round(ttft + pools.prefix[start + count] - pools.prefix[start] + 0.5, 3)
    fields: dict[str, Any] = {
        "status": "ok",
        "ttft_ms": ttft,
        "first_chunk_latency_ms": round(ttft * 0.9, 3),
        "chunk_interarrival_ms": gaps,
        "e2e_latency_ms": e2e,
        "finish_reason": "length",
        "http_status": 200,
        "error_type": None,
        "error_message": None,
        **_tokens(plan, index, count + 1),
    }
    return fields, e2e


def _failed_fields(plan: SyntheticPlan, index: int) -> tuple[dict[str, Any], float]:
    """A request the server refused with a 500, as the profiler records it."""
    e2e = 3.0 + index % 7
    fields: dict[str, Any] = {
        "status": "error",
        "ttft_ms": None,
        "first_chunk_latency_ms": None,
        "chunk_interarrival_ms": [],
        "e2e_latency_ms": e2e,
        "finish_reason": None,
        "http_status": 500,
        "error_type": "EndpointHTTPError",
        "error_message": f'HTTP 500: {{"error": {{"message": "forced failure {index}"}}}}',
        "prompt_tokens": plan.input_tokens,
        "prompt_token_source": "estimated",
        "prompt_token_exact": False,
        "output_tokens": None,
        "output_token_source": "unknown",
        "output_token_exact": False,
        "total_tokens": None,
    }
    return fields, e2e


class _Timeline:
    """Closed-loop workers sending back to back from ``started_at_ns``."""

    def __init__(self, plan: SyntheticPlan, started_at_ns: int) -> None:
        self.plan = plan
        self.clocks = [
            started_at_ns + worker * 50_000 for worker in range(plan.concurrency)
        ]
        self.last_sent_ns = started_at_ns
        self.last_ended_ns = started_at_ns

    def place(self, index: int, e2e_ms: float) -> dict[str, Any]:
        worker = index % self.plan.concurrency
        started = self.clocks[worker]
        ended = started + int(e2e_ms * 1_000_000)
        self.clocks[worker] = ended + 80_000
        self.last_sent_ns = max(self.last_sent_ns, started)
        self.last_ended_ns = max(self.last_ended_ns, ended)
        return {
            "request_id": f"{self.plan.case_id}_measured_{worker}_{index}",
            "request_index": index,
            "intended_at_ns": started - 30_000,
            "started_at_ns": started,
            "timestamp_ns": started,
            "ended_at_ns": ended,
            "in_flight_at_dispatch": min(index + 1, self.plan.concurrency),
        }


def synthetic_requests(
    plan: SyntheticPlan, template: InferenceRequestEvent, timeline: _Timeline, seed: int
) -> Iterator[dict[str, Any]]:
    """The measured requests' records, as ``template.to_record()`` with each
    request's own fields; ``InferenceRequestEvent(**fields).to_record()``
    gives the same dict (a test checks it), but building one per request
    would cost more than the rest of the generation together."""
    pools = _Pools.draw(seed)
    base = template.to_record()
    prefix = template.x_request_id or ""
    every = plan.fail_every
    for index in range(plan.requests):
        failed = every is not None and (index + 1) % every == 0
        fields, e2e = (
            _failed_fields(plan, index) if failed else _ok_fields(plan, pools, index)
        )
        record = dict(base)
        record.update(fields)
        placed = timeline.place(index, e2e)
        record.update(placed)
        record["x_request_id"] = prefix + placed["request_id"]
        yield record


def _synthetic_profiler(
    plan: SyntheticPlan, path: Path, seed: int, samples: SamplePlan | None = None
) -> InferenceProfiler:
    """A profiler whose header records describe ``plan``; it never sends."""
    rng = random.Random(seed)
    sampling: dict[str, Any] = {"system_sampler": "none"}
    if samples is not None:
        sampling = {
            "system_sampler": samples.sampler,
            "sample_interval_seconds": samples.interval_ms / 1000,
        }
    config = ProfileConfig(
        endpoint="http://127.0.0.1:9/v1/chat/completions",
        model=MODEL,
        concurrency=(plan.concurrency,),
        input_tokens=(plan.input_tokens,),
        output_tokens=(plan.output_tokens,),
        output_path=str(path),
        request_count=plan.requests,
        seed=seed,
        tokenizer="none",
        server_probe="none",
        run_id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
        **sampling,
    )
    profiler = InferenceProfiler(config)
    profiler.request_executor.shutdown()
    profiler.session = replace(
        profiler.session,
        session_id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
        started_at_ns=SYNTHETIC_ORIGIN_NS,
    )
    return profiler


def _template_request(
    profiler: InferenceProfiler, plan: SyntheticPlan, prompt_digest: str
) -> InferenceRequestEvent:
    config = profiler.config
    return InferenceRequestEvent(
        session_id=profiler.session.session_id,
        request_id="",
        case_id=plan.case_id,
        phase="measured",
        started_at_ns=0,
        ended_at_ns=0,
        endpoint=config.endpoint,
        model=config.model,
        concurrency=plan.concurrency,
        target_input_tokens=plan.input_tokens,
        target_output_tokens=plan.output_tokens,
        stream=True,
        status="ok",
        e2e_latency_ms=None,
        ttft_ms=None,
        first_chunk_latency_ms=None,
        dispatch_lag_ms=0.03,
        held_for_slot=False,
        prompt_mode="repeat",
        prompt_id="repeat",
        prompt_digest=prompt_digest,
        x_request_id=f"stormlog-{profiler.run_id}-",
    )


def _write_phase(
    writer: JsonlEventWriter,
    profiler: InferenceProfiler,
    plan: SyntheticPlan,
    samples: SamplePlan | None = None,
) -> int:
    """The measured phase of ``plan``'s case; returns when it drained.

    ``samples`` rows are written in time order among the requests, as the
    profiler's sampling task writes them while requests finish."""
    case = profiler.config.cases()[0]
    session_id = profiler.session.session_id
    started = SYNTHETIC_ORIGIN_NS + 5_000_000
    writer.append(
        cache_state_record(
            session_id=session_id,
            case_id=case.case_id,
            requested=profiler.config.cache_state,
            reset=None,
            warmup_requests=0,
        )
    )
    writer.append(
        InferencePhaseStartEvent(
            session_id=session_id,
            case_id=case.case_id,
            phase="measured",
            arrival_mode=case.arrival.mode,
            started_at_ns=started,
        ).to_record()
    )
    prompts = PromptSource(
        profiler.prompt_spec,
        counter=profiler.token_counter,
        seed=profiler.config.seed,
        case_id=case.case_id,
        phase="measured",
        input_tokens=case.input_tokens,
    )
    prompt = prompts.take(0)
    template = _template_request(profiler, plan, prompt.digest)
    timeline = _Timeline(plan, started)
    clock = _SampleClock(samples, session_id, started)
    for record in synthetic_requests(plan, template, timeline, profiler.config.seed):
        clock.until(writer, record["started_at_ns"])
        writer.append(record)
    clock.until(writer, timeline.last_ended_ns)
    for index in range(1, plan.requests):
        prompts.take(index)
    window = _PhaseWindow(started, timeline.last_sent_ns, timeline.last_ended_ns)
    writer.append(
        window.to_record(
            session_id=session_id,
            request=_PhaseRequest(case, writer, prompts, "measured"),
            drain_timeout_seconds=profiler._drain_timeout(),
            abandoned=AbandonedWait(),
        )
    )
    return timeline.last_ended_ns


def synthetic_artifact(
    directory: Path,
    *,
    requests: int,
    gaps_per_request: int,
    seed: int = 0,
    name: str = "synthetic.jsonl",
    concurrency: int = 64,
    fail_every: int | None = None,
    tokens: TokenMode = "exact",
    samples: SamplePlan | bool = False,
    nan_gap: bool = False,
) -> Path:
    """One closed-loop case of ``requests`` streamed requests.

    The header, cache state, phase and terminal session records come from
    a real ``InferenceProfiler`` that never sends; each request is an
    ``InferenceRequestEvent`` written through ``JsonlEventWriter``. Times
    start at ``SYNTHETIC_ORIGIN_NS``, so a seed always writes the same
    requests. ``fail_every`` makes every Nth request a server 500;
    ``tokens`` picks exact, estimated or missing output counts, or all three
    in turn. ``samples`` writes ``InferenceSystemSample`` rows through the
    same writer: True for the default ``SamplePlan``. ``nan_gap`` makes the
    middle gap of request 0 a NaN, which the writer's ``json.dumps`` writes
    as a bare ``NaN`` (needs ``gaps_per_request`` of at least 1). No
    ``infer.dispatch``, ``infer.first_content`` or ``infer.summary``
    records are written.
    """
    if nan_gap and (gaps_per_request < 1 or requests < 1):
        raise ValueError("nan_gap needs a request with at least one gap")
    if not 0 <= gaps_per_request <= MAX_SYNTHETIC_GAPS:
        raise ValueError(f"gaps_per_request must be 0 to {MAX_SYNTHETIC_GAPS}")
    plan = SyntheticPlan(
        requests=requests,
        gaps_per_request=gaps_per_request,
        concurrency=concurrency,
        fail_every=fail_every,
        tokens=tokens,
        nan_gap=nan_gap,
    )
    sample_plan = samples if isinstance(samples, SamplePlan) else None
    if samples is True:
        sample_plan = SamplePlan()
    path = Path(directory) / name
    profiler = _synthetic_profiler(plan, path, seed, sample_plan)
    with JsonlEventWriter(path) as writer:
        writer.append(profiler._session_record())
        profiler._write_run_records(writer)
        drained = _write_phase(writer, profiler, plan, sample_plan)
    with mock.patch("stormlog.session.now_ns", return_value=drained + 1_000_000):
        profiler._write_terminal_session(
            output_path=path, report=None, status="completed"
        )
    return path


# --- Execution import ----------------------------------------------------


def decoded_records(path: Path) -> list[dict[str, Any]]:
    """Every record of a complete artifact, with v4 contexts expanded."""
    lines = path.read_bytes().split(b"\n")
    return expand_inference_records(json.loads(line) for line in lines if line)


def _step_times(records: Sequence[Mapping[str, Any]]) -> list[tuple[int, str, str]]:
    """(mono ns, kind, x_request_id) of each hook event the requests imply.

    Each successful measured request gets a prefill step just after it was
    sent and a decode step that ends as it ends, on the engine's monotonic
    clock (``wall - WALL_OFFSET`` in the helpers).
    """
    events = []
    for record in records:
        if not _is_measured_ok(record):
            continue
        start = int(record["started_at_ns"]) - WALL_OFFSET
        end = int(record["ended_at_ns"]) - WALL_OFFSET
        x_id = str(record["x_request_id"])
        events.append((start + 100_000, "prefill", x_id))
        events.append((max(start + 400_000, end - 300_000), "decode", x_id))
    return sorted(events)


def _is_measured_ok(record: Mapping[str, Any]) -> bool:
    return (
        record.get("event_type") == "infer.request"
        and record.get("phase") == "measured"
        and record.get("status") == "ok"
        and isinstance(record.get("x_request_id"), str)
    )


def _hook_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The engine's hook log for ``records``' requests, step by step."""
    log: list[dict[str, Any]] = []
    for iteration, (mono, kind, x_id) in enumerate(_step_times(records)):
        # vLLM's own ID: the external one plus a random suffix.
        internal = f"chatcmpl-{x_id}-0f3a9c1d"
        if kind == "prefill":
            log.append(alias(internal, f"chatcmpl-{x_id}", mono - 50_000))
            step = member(internal, scheduled=8)
        else:
            step = member(internal, scheduled=1, sighting="repeat")
        log.append(scheduled(iteration, mono, [step], duration_ns=20_000))
        if kind == "decode":
            log.append(terminal(internal, mono + 25_000, output_tokens=2))
        log.append(completed(iteration, mono + 30_000, [done(internal)]))
    return log


def with_execution_import(artifact: Path) -> Path:
    """Import a vLLM execution log of ``artifact``'s requests into it.

    The log is built with ``tests/vllm_execution_helpers.py`` in a
    sibling directory, then the real ``infer import-execution`` appends the
    iteration, membership, request, stage and clock records, replacing the
    file through ``os.replace`` (a new inode with the old bytes as prefix).
    Returns ``artifact``.
    """
    records = decoded_records(artifact)
    log = _hook_records(records)
    if not log:
        raise ValueError(f"{artifact} has no successful measured request")
    first = int(log[0]["mono_ns"])
    last = int(log[-1]["mono_ns"])
    hook = artifact.with_name(f"{artifact.stem}.hook")
    log.append(heartbeat(last + 1_000_000, len(log)))
    log.append(goodbye(last + 2_000_000, len(log) + 1))
    engine_log(hook, log, hello_mono_ns=first - SECOND)
    with contextlib.redirect_stdout(io.StringIO()):
        code = infer_main(["import-execution", str(artifact), str(hook)])
    if code != int(ExitCode.OK):
        raise RuntimeError(f"import-execution exited {code}")
    return artifact


# --- Files Studio does not read as a text inference artifact -------------


def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _audio_request(session_id: str, index: int, ok: bool) -> dict[str, Any]:
    started = SYNTHETIC_ORIGIN_NS + 5_000_000 + index * 400_000_000
    return AudioRequestEvent(
        session_id=session_id,
        request_id=f"tts_c1_measured_0_{index}",
        case_id="tts_c1",
        phase="measured",
        task="tts",
        model="fake-tts",
        fixture_id="hello",
        fixture_digest=hashlib.sha256(b"hello").hexdigest(),
        request_index=index,
        started_at_ns=started,
        ended_at_ns=started + 300_000_000,
        status="ok" if ok else "error",
        endpoint="http://127.0.0.1:9/v1/audio/speech",
        stream=False,
        arrival_mode=CLOSED,
        intended_at_ns=started,
        response_bytes=48_000 if ok else 0,
        concurrency=1,
        e2e_latency_ms=300.0,
        first_response_byte_ms=120.0 if ok else None,
        output_audio_seconds=1.5 if ok else None,
        rtf=0.2 if ok else None,
        http_status=200 if ok else 500,
        error_type=None if ok else "EndpointHTTPError",
    ).to_record()


def audio_artifact(directory: Path, *, name: str = "audio.jsonl") -> Path:
    """Two ``infer.audio.request`` lines, an ok and a failed text-to-speech
    request, written by ``AudioRequestEvent`` through ``JsonlEventWriter``:
    the shape Studio refuses as ``unsupported_artifact``, reason ``audio``."""
    session_id = _uuid(random.Random(11))
    path = Path(directory) / name
    with JsonlEventWriter(path) as writer:
        writer.append(_audio_request(session_id, 0, ok=True))
        writer.append(_audio_request(session_id, 1, ok=False))
    return path


def telemetry_artifact(directory: Path, *, name: str = "telemetry.jsonl") -> Path:
    """A memory-profiler telemetry JSONL, not an inference artifact.

    ``AppendOnlyTelemetrySink`` writes ``TelemetryEventV4`` records into
    ``<stem>.sink/``, built as the CPU tracker builds them
    (``CPUMemoryTracker._telemetry_record_from_event``, with its private
    capability table); the sink's one segment is copied to ``name``. Four
    samples of the process's RSS, one per 100 ms."""
    path = Path(directory) / name
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(root_dir=path.with_suffix(".sink"), write_rollups=False)
    )
    session = sink.start_session()
    for index in range(4):
        rss = 2 * GIB + index * 64 * MIB
        event = TelemetryEventV4(
            schema_version=SCHEMA_VERSION_V4,
            session_id=session.session_id,
            timestamp_ns=SYNTHETIC_ORIGIN_NS + index * 100_000_000,
            event_type="sample",
            collector="stormlog.cpu_tracker",
            sampling_interval_ms=100,
            pid=1,
            host="fixture",
            device_id=-1,
            allocator_allocated_bytes=rss,
            allocator_reserved_bytes=rss,
            allocator_active_bytes=None,
            allocator_inactive_bytes=None,
            allocator_change_bytes=0 if index == 0 else 64 * MIB,
            device_used_bytes=rss,
            device_free_bytes=None,
            device_total_bytes=None,
            context=None,
            metadata={"memory_capabilities": dict(_CPU_MEMORY_CAPABILITIES)},
        )
        sink.append(telemetry_event_to_dict(event))
    sink.close()
    (segment,) = sorted(path.with_suffix(".sink").glob("*.jsonl"))
    path.write_bytes(segment.read_bytes())
    return path


def unsupported_version_artifact(
    directory: Path, *, name: str = "unsupported_version.jsonl"
) -> Path:
    """A real profile with one more line: its last measured client request
    again, declared ``schema_version`` 2 and still without a ``context``.

    ``JsonlEventWriter`` refuses that record (a v2 request must carry the
    context it compacts), and no Stormlog writer makes one, so the line is
    the record ``json.dumps``-ed with ``sort_keys`` as the writer would
    have, and appended after the terminal session record. It stands for
    the design's "client request at version 2 with no context" (§4.4):
    Studio counts it as an ``unsupported_version`` problem and excludes it.
    """
    path = profile_artifact(directory, [CaseSpec(2, 8, 4, 8)], name=name)
    requests = [r for r in decoded_records(path) if _is_measured_ok(r)]
    record = dict(requests[-1], schema_version=2)
    with JsonlEventWriter(path.with_suffix(".refused")) as writer:
        try:
            writer.append(record)
        except ValueError:
            pass
        else:
            raise RuntimeError("JsonlEventWriter now writes v2 client requests")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


# --- Cuts, growth, concatenation and reruns ------------------------------

Cut = Literal["mid_line", "after_newline"]


def last_line_start(data: bytes) -> int:
    """Where the last complete line of ``data`` begins."""
    if not data.endswith(b"\n"):
        raise ValueError("the source must end with a newline")
    return data.rfind(b"\n", 0, len(data) - 1) + 1


def truncated_copy(source: Path, target: Path, *, cut: Cut) -> Path:
    """Copy ``source`` without the end of its last line.

    ``after_newline`` drops the last line whole, so the copy ends with a
    newline: a run killed before its terminal session record. ``mid_line``
    keeps the first half of the last line's bytes, without its newline: a
    write cut off part way (design §4.2 hazard 3).
    """
    data = Path(source).read_bytes()
    start = last_line_start(data)
    end = start
    if cut == "mid_line":
        end = start + max(1, (len(data) - 1 - start) // 2)
    elif cut != "after_newline":
        raise ValueError(f"unknown cut {cut!r}")
    Path(target).write_bytes(data[:end])
    return Path(target)


class GrowingCopy:
    """A prefix of a finished artifact that grows as a running profile's
    file does: appended to in place, line by line, on the same inode."""

    def __init__(self, source: Path, target: Path, *, lines: int) -> None:
        self.data = Path(source).read_bytes()
        self.ends = [i + 1 for i, byte in enumerate(self.data) if byte == 0x0A]
        if not 0 <= lines <= len(self.ends):
            raise ValueError(f"the source has {len(self.ends)} lines")
        self.target = Path(target)
        self.offset = self.ends[lines - 1] if lines else 0
        self.target.write_bytes(self.data[: self.offset])

    @property
    def remaining(self) -> int:
        """Complete lines not yet written."""
        return sum(1 for end in self.ends if end > self.offset)

    def _write_to(self, end: int) -> None:
        with self.target.open("ab") as handle:
            handle.write(self.data[self.offset : end])
        self.offset = end

    def append(self, count: int = 1) -> None:
        """Write the next ``count`` lines, finishing a partial one first."""
        pending = [end for end in self.ends if end > self.offset]
        if count < 1 or count > len(pending):
            raise ValueError(f"{len(pending)} lines remain")
        self._write_to(pending[count - 1])

    def append_partial(self) -> None:
        """Write the first half of the next line, without its newline."""
        end = next(end for end in self.ends if end > self.offset)
        self._write_to(self.offset + max(1, (end - 1 - self.offset) // 2))

    def finish(self) -> None:
        """Write the rest: the file now equals its source."""
        self._write_to(len(self.data))


def concatenated(sources: Sequence[Path], target: Path) -> Path:
    """``cat sources > target``: the only way one file gets two sessions."""
    Path(target).write_bytes(b"".join(Path(path).read_bytes() for path in sources))
    return Path(target)


def rewrite_in_place(
    path: Path, *, cases: Sequence[CaseSpec], warmup: int = 2, seed: int = 0
) -> None:
    """Rerun a profile to ``path``: the writer opens it with ``"w"``, so the
    same inode is truncated and rewritten with a new session (§4.2 hazard 4)."""
    profile_artifact(
        Path(path).parent, cases, name=Path(path).name, warmup=warmup, seed=seed
    )


# --- The standard set ----------------------------------------------------

Builder = Callable[[Path], Path]

# Two cases per concurrency, one of them losing every 4th measured request.
MIXED_CASES = (
    CaseSpec(2, 8, 4, 12, fail_every=4),
    CaseSpec(2, 8, 16, 12),
    CaseSpec(4, 8, 4, 12),
    CaseSpec(4, 8, 16, 12, fail_every=3),
)


OPEN_LOOP_CASES = (CaseSpec(4, 8, 4, 12, fail_every=4), CaseSpec(4, 8, 16, 12))
INTERRUPTED_CASES = (CaseSpec(2, 8, 4, 12), CaseSpec(2, 8, 16, 12, interrupt_after=5))


def _profiled(name: str, cases: Sequence[CaseSpec], **options: Any) -> Builder:
    def build(directory: Path) -> Path:
        return profile_artifact(directory, cases, name=f"{name}.jsonl", **options)

    return build


def _synthetic(name: str, requests: int, **options: Any) -> Builder:
    def build(directory: Path) -> Path:
        return synthetic_artifact(
            directory,
            requests=requests,
            gaps_per_request=15,
            seed=7,
            name=f"{name}.jsonl",
            concurrency=8,
            **options,
        )

    return build


def _truncated(name: str, cut: Cut) -> Builder:
    def build(directory: Path) -> Path:
        source = profile_artifact(directory, MIXED_CASES[:1], name=f"{name}.src.jsonl")
        return truncated_copy(source, directory / f"{name}.jsonl", cut=cut)

    return build


def _growing(directory: Path) -> Path:
    """A profile caught mid-run: its measured phase has begun and only part
    of its requests are written; no phase window, no terminal record."""
    source = profile_artifact(directory, MIXED_CASES[:1], name="growing.src.jsonl")
    lines = source.read_bytes().split(b"\n")
    starts = [i for i, line in enumerate(lines) if b'"infer.phase_start"' in line]
    cut = starts[-1] + 8  # the measured phase's start, then a few records
    return GrowingCopy(source, directory / "growing.jsonl", lines=cut).target


def _execution_import(directory: Path) -> Path:
    cases = [CaseSpec(2, 8, 4, 8, fail_every=4)]
    return with_execution_import(
        profile_artifact(directory, cases, name="execution_import.jsonl")
    )


def _multi_session(directory: Path) -> Path:
    first = profile_artifact(directory, MIXED_CASES[:1], name="multi_session.a.jsonl")
    second = profile_artifact(directory, MIXED_CASES[:1], name="multi_session.b.jsonl")
    return concatenated([first, second], directory / "multi_session.jsonl")


# Each builder writes ``<name>.jsonl`` (and maybe ``<name>.*`` helpers) into
# the directory it is given, so the whole set can share one directory.
STANDARD_ARTIFACTS: Mapping[str, Builder] = {
    "mixed": _profiled("mixed", MIXED_CASES),
    "all_failed": _profiled("all_failed", [CaseSpec(2, 8, 4, 8, all_fail=True)]),
    "non_streaming": _profiled("non_streaming", [CaseSpec(2, 8, 4, 8, stream=False)]),
    "tokens_exact": _profiled("tokens_exact", [CaseSpec(2, 8, 4, 8)]),
    "tokens_estimated": _profiled(
        "tokens_estimated", [CaseSpec(2, 8, 4, 8, usage=False)]
    ),
    "tokens_missing": _synthetic("tokens_missing", 60, tokens="missing"),
    "tokens_mixed": _synthetic("tokens_mixed", 60, tokens="mixed", fail_every=10),
    "warmup_measured": _profiled("warmup_measured", [CaseSpec(2, 8, 4, 8)], warmup=4),
    "empty": _synthetic("empty", 0),
    # Open loop at 40/s: each case's phase window has its scheduled endpoint,
    # so the case rate is over the schedule and a window covering the whole
    # case gives the same rate (§3.3 rules 1 to 3).
    "open_loop": _profiled(
        "open_loop", OPEN_LOOP_CASES, arrival=FIXED_RATE, rate_per_second=40.0
    ),
    # 40 arrivals scheduled over 2 s, the window ended by the stop file at the
    # 8th: stopped_early, with the endpoint where the window stopped.
    "stopped_early": _profiled(
        "stopped_early",
        [CaseSpec(4, 8, 4, 40, stop_after=8)],
        arrival=FIXED_RATE,
        rate_per_second=20.0,
    ),
    # Ctrl+C at the 5th measured request of the second case: the first case
    # is whole, the second has requests and no phase window, so no rate.
    "interrupted": _profiled("interrupted", INTERRUPTED_CASES),
    "samples": _synthetic("samples", 60, samples=True),
    "samples_psutil": _profiled(
        "samples_psutil",
        [CaseSpec(2, 8, 4, 8)],
        system_sampler="psutil",
        sample_interval_seconds=0.02,
    ),
    "synthetic": _synthetic("synthetic", 2_000, fail_every=50),
    "execution_import": _execution_import,
    "truncated_mid_line": _truncated("truncated_mid_line", "mid_line"),
    "truncated_after_newline": _truncated("truncated_after_newline", "after_newline"),
    "growing": _growing,
    "multi_session": _multi_session,
    # Every 4th measured request held 1 s at the server, past the 0.25 s
    # request timeout: two timeouts among eight measured requests.
    "timeouts": _profiled(
        "timeouts",
        [CaseSpec(2, 8, 4, 8, slow_every=4, delay_seconds=1.0)],
        timeout_seconds=0.25,
    ),
    "nan_gap": _synthetic("nan_gap", 60, nan_gap=True),
    "audio": lambda directory: audio_artifact(directory, name="audio.jsonl"),
    "not_an_inference_artifact": lambda directory: telemetry_artifact(
        directory, name="not_an_inference_artifact.jsonl"
    ),
    "unsupported_version": lambda directory: unsupported_version_artifact(
        directory, name="unsupported_version.jsonl"
    ),
}

# What the rest of the suite may rely on about each standard artifact.
# Not text inference artifacts: no infer.session, no infer.request. Studio
# answers each with 422 and this code (design §5.2, §8.2 unsupported).
UNSUPPORTED: Mapping[str, str] = {
    "audio": "unsupported_artifact",  # details.reason "audio"
    "not_an_inference_artifact": "not_an_inference_artifact",
}
# One bad line fails the file; the analyzer exits INVALID_INPUT on these.
ANALYZE_FAILS = frozenset({"truncated_mid_line", *UNSUPPORTED})
MULTI_SESSION = frozenset({"multi_session"})  # Studio refuses it (decision 7)
NO_TERMINAL_RECORD = frozenset(
    {"truncated_mid_line", "truncated_after_newline", "growing"}
)
# Read by Studio, with lines it counts as these problems (contracts.ProblemCode).
PROBLEM_LINES: Mapping[str, str] = {
    "truncated_mid_line": "truncated_final_line",
    "unsupported_version": "unsupported_version",
}

LARGE_REQUESTS = 500_000
LARGE_GAPS = 15


def large_artifact(directory: Path) -> Path:
    """The 500,000-request, 15-gap artifact for the 2 s route target and
    Fable's D-1/F-7 timings: one case at concurrency 64, every 100th
    request failed, token counts mixed."""
    return synthetic_artifact(
        directory,
        requests=LARGE_REQUESTS,
        gaps_per_request=LARGE_GAPS,
        seed=1,
        name="large.jsonl",
        fail_every=100,
        tokens="mixed",
    )


# --- What differs between two generations of one fixture -----------------

# Keys whose values differ between two generations of the same synthetic
# fixture, or between two Studio instances serving it (design §9.2). From
# the process: ``pid`` (infer.context, the terminal infer.session summary),
# ``context_id`` (a digest of the context, so it follows ``pid``), and
# ``environment_proxies`` (the proxy variables set, recorded in the run
# config). From the machine or the checkout: ``host``, ``clock_domain`` and
# ``boot_id``, and ``source_version``. From Studio: ``instance_id`` and
# ``index_id``. Synthetic timestamps are fixed; profiled fixtures also vary
# in every timestamp, ID and measured value, so they need ``timestamps``
# and more than this list.
VARYING_FIELDS = frozenset(
    {
        "context_id",
        "pid",
        "host",
        "clock_domain",
        "boot_id",
        "source_version",
        "environment_proxies",
        "instance_id",
        "index_id",
    }
)
VARYING = "<varying>"


def is_timestamp(key: str) -> bool:
    """A wall-clock key: ``*_ns``, ``*_at`` or ``timestamp``."""
    return key.endswith(("_ns", "_at")) or key == "timestamp"


def normalised(value: Any, *, timestamps: bool = False) -> Any:
    """``value`` with every ``VARYING_FIELDS`` value, at any depth, replaced
    by ``VARYING``; with ``timestamps``, every timestamp key's too."""
    if isinstance(value, Mapping):
        return {
            key: (
                VARYING
                if key in VARYING_FIELDS or (timestamps and is_timestamp(key))
                else normalised(item, timestamps=timestamps)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalised(item, timestamps=timestamps) for item in value]
    return value


def varying_paths(first: Any, second: Any, path: str = "") -> set[str]:
    """The dotted paths where ``first`` and ``second`` differ (``[]`` for a
    list element), down to the first differing key."""
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        found: set[str] = set()
        for key in set(first) | set(second):
            inner = f"{path}.{key}" if path else str(key)
            found |= varying_paths(first.get(key), second.get(key), inner)
        return found
    if isinstance(first, list) and isinstance(second, list):
        if len(first) == len(second):
            found = set()
            for one, other in zip(first, second):
                found |= varying_paths(one, other, f"{path}[]")
            return found
    return set() if first == second else {path}


def main(argv: Sequence[str]) -> int:
    """``python -m tests.studio.artifacts NAME DIRECTORY``: build one standard
    artifact into DIRECTORY and print its path, in a fresh process."""
    name, directory = argv
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    print(STANDARD_ARTIFACTS[name](target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
