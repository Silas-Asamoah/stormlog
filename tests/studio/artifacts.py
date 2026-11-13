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
import io
import itertools
import json
import random
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from unittest import mock

from stormlog.exit_codes import ExitCode
from stormlog.infer.cache_state import cache_state_record
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.correlation_codec import expand_inference_records
from stormlog.infer.events import (
    InferencePhaseStartEvent,
    InferenceRequestEvent,
    JsonlEventWriter,
)
from stormlog.infer.profile import InferenceProfiler, _PhaseRequest, _PhaseWindow
from stormlog.infer.prompts import PromptSource
from stormlog.infer.request_scheduler import AbandonedWait
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
    """

    concurrency: int
    input_tokens: int
    output_tokens: int
    requests: int
    fail_every: int | None = None
    all_fail: bool = False
    stream: bool = True
    usage: bool = True

    @property
    def case_id(self) -> str:
        """The ID the profiler gives this case (closed-loop arrivals)."""
        return f"c{self.concurrency}_in{self.input_tokens}_out{self.output_tokens}"

    def policy(self) -> CasePolicy:
        return CasePolicy(
            fail_every=self.fail_every, all_fail=self.all_fail, usage=self.usage
        )


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
) -> ProfileConfig:
    """The config one profile run of ``cases`` takes.

    Raises ``ValueError`` for cases one run cannot hold: mixed request
    counts or stream settings, or a set that is not the full product of its
    concurrencies and token lengths.
    """
    concurrency, inputs, outputs = _run_shape(cases)
    return ProfileConfig(
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
        system_sampler="none",
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
) -> Path:
    """Profile ``fake_openai`` with the real ``InferenceProfiler``.

    The artifact is complete: header records, ``warmup`` requests and the
    measured ones per case, phase windows, the analyzer's summary and the
    terminal session record. An existing file at the path is truncated and
    rewritten in place, as a rerun of ``infer profile`` does.
    """
    output = Path(directory) / name
    policies = {case.case_id: case.policy() for case in cases}
    with fake_openai(policies) as server:
        config = profile_config(
            cases, endpoint=server.endpoint, output=output, warmup=warmup, seed=seed
        )
        with contextlib.redirect_stdout(io.StringIO()):
            InferenceProfiler(config).run()
    return output


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

    @property
    def case_id(self) -> str:
        return f"c{self.concurrency}_in{self.input_tokens}_out{self.output_tokens}"


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
    plan: SyntheticPlan, path: Path, seed: int
) -> InferenceProfiler:
    """A profiler whose header records describe ``plan``; it never sends."""
    rng = random.Random(seed)
    config = ProfileConfig(
        endpoint="http://127.0.0.1:9/v1/chat/completions",
        model=MODEL,
        concurrency=(plan.concurrency,),
        input_tokens=(plan.input_tokens,),
        output_tokens=(plan.output_tokens,),
        output_path=str(path),
        request_count=plan.requests,
        seed=seed,
        system_sampler="none",
        tokenizer="none",
        server_probe="none",
        run_id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
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
    writer: JsonlEventWriter, profiler: InferenceProfiler, plan: SyntheticPlan
) -> int:
    """The measured phase of ``plan``'s case; returns when it drained."""
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
    for record in synthetic_requests(plan, template, timeline, profiler.config.seed):
        writer.append(record)
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
) -> Path:
    """One closed-loop case of ``requests`` streamed requests.

    The header, cache state, phase and terminal session records come from
    a real ``InferenceProfiler`` that never sends; each request is an
    ``InferenceRequestEvent`` written through ``JsonlEventWriter``. Times
    start at ``SYNTHETIC_ORIGIN_NS``, so a seed always writes the same
    requests. ``fail_every`` makes every Nth request a server 500;
    ``tokens`` picks exact, estimated or missing output counts, or all three
    in turn. No ``infer.dispatch``, ``infer.first_content`` or
    ``infer.summary`` records are written.
    """
    if not 0 <= gaps_per_request <= MAX_SYNTHETIC_GAPS:
        raise ValueError(f"gaps_per_request must be 0 to {MAX_SYNTHETIC_GAPS}")
    plan = SyntheticPlan(
        requests=requests,
        gaps_per_request=gaps_per_request,
        concurrency=concurrency,
        fail_every=fail_every,
        tokens=tokens,
    )
    path = Path(directory) / name
    profiler = _synthetic_profiler(plan, path, seed)
    with JsonlEventWriter(path) as writer:
        writer.append(profiler._session_record())
        profiler._write_run_records(writer)
        drained = _write_phase(writer, profiler, plan)
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
    "synthetic": _synthetic("synthetic", 2_000, fail_every=50),
    "execution_import": _execution_import,
    "truncated_mid_line": _truncated("truncated_mid_line", "mid_line"),
    "truncated_after_newline": _truncated("truncated_after_newline", "after_newline"),
    "growing": _growing,
    "multi_session": _multi_session,
}

# What the rest of the suite may rely on about each standard artifact.
ANALYZE_FAILS = frozenset({"truncated_mid_line"})  # one bad line fails the file
MULTI_SESSION = frozenset({"multi_session"})  # Studio refuses it (decision 7)
NO_TERMINAL_RECORD = frozenset(
    {"truncated_mid_line", "truncated_after_newline", "growing"}
)
