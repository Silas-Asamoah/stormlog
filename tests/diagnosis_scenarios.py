"""Synthetic runs for diagnosis tests: a client and a toy vLLM engine.

The engine steps at a fixed cadence, admits waiting requests first come first
served up to ``max_num_seqs``, prefills a new request in one step and decodes
one token per step. It writes the hook log (``docs/vllm_execution.md``) the
way the hook would, and the client writes its records the way ``infer
profile`` does; the artifact is then imported through the real
``import_execution_into_artifact``. Engine and client share one host, so the
client's wall clock is the engine's: wall = mono + WALL_OFFSET.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stormlog.infer.correlation_events import (
    ArtifactIdentityEvent,
    CorrelationContext,
)
from stormlog.infer.vllm_execution_import import import_execution_into_artifact
from tests.vllm_execution_helpers import (
    BOOT,
    HOST,
    SECOND,
    WALL_OFFSET,
    importer,
    stamp,
    write_epoch,
)

MS = 1_000_000
RUN = "run-1"
SESSION = "session-1"
PID, START = 2600, 1_790_000_000_000_000_000
EPOCH = f"engine-{PID}-{START}"
OBSERVES = ["cache_reset", "enqueued", "pause"]


@dataclass
class SimRequest:
    """One client request; times are on the engine's monotonic clock."""

    request_id: str
    sent_ns: int
    case_id: str = "c1"
    phase: str = "measured"
    prompt: int = 8
    output: int = 4
    intended_ns: int | None = None
    ingress_ns: int = 1 * MS  # send to the engine's alias
    enqueue_ns: int = 100_000  # alias to entering the scheduler
    delivery_ns: int = 500_000  # a step's completion to the client
    structured_output: bool = False
    resumable: bool = False  # a streaming-input request
    held_for_slot: bool = False
    closed_loop: bool = False  # no intended arrival time
    shared_prefix_tokens: int | None = None
    prefix_group: int | None = None
    cached: int = 0  # prefix-cache hit at its first step
    # Filled by the engine.
    first_step: int | None = None
    first_done_ns: int | None = None
    last_done_ns: int | None = None

    @property
    def x_request_id(self) -> str:
        return f"stormlog-{RUN}-{self.request_id}"

    @property
    def internal(self) -> str:
        return f"chatcmpl-{self.x_request_id}-0f3a9c1d"

    @property
    def admitted_ns(self) -> int:
        return self.sent_ns + self.ingress_ns

    @property
    def enqueued_ns(self) -> int:
        return self.admitted_ns + self.enqueue_ns


@dataclass
class Engine:
    max_num_seqs: int = 8
    step_ns: int = 10 * MS
    gap_ns: int = 100_000
    observes: list[str] | None = field(default_factory=lambda: list(OBSERVES))
    max_num_batched_tokens: int = 2048
    # Extra raw records to interleave by time, e.g. pauses and resets.
    extra: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    # False: a hook from before wall_after_ns, whose stamps are unbracketed.
    bracketed: bool = True
    # False: a hook from before enqueued records.
    enqueued_records: bool = True
    # KV slots, in tokens: a step whose growth would not fit preempts.
    kv_tokens: int | None = None
    # From this monotonic time, heartbeats report one oversized cache reset
    # the writer dropped.
    dropped_from: int | None = None
    # Extra settings the hello's config reports.
    config: dict[str, Any] = field(default_factory=dict)
    # (mono_ns, delta_ns): the host's wall clock steps by delta at mono_ns.
    wall_jump: tuple[int, int] | None = None
    # (mono_ns, duration_ns): the engine loop stops for duration after the
    # first step that completes at or after mono_ns.
    stall: tuple[int, int] | None = None
    # More such stops, each as ``stall`` is.
    stalls: list[tuple[int, int]] = field(default_factory=list)
    # How long an idle engine takes from a request's entry to its schedule()
    # call; vLLM 0.30.0's loop took 20 us and more on an A30.
    wake_ns: int = 0

    def wall(self, mono_ns: int) -> int:
        """The shared wall clock at an engine monotonic time."""
        jump = self.wall_jump
        step = jump[1] if jump is not None and mono_ns >= jump[0] else 0
        return mono_ns + WALL_OFFSET + step

    def run(self, requests: list[SimRequest]) -> list[dict[str, Any]]:
        """Serve ``requests``; return the epoch's raw records after the hello."""
        timed: list[tuple[int, int, dict[str, Any]]] = []
        for request in requests:
            if self.enqueued_records:
                timed.append(_enqueued(request))
            timed.append(
                (
                    request.admitted_ns,
                    0,
                    {
                        "kind": "alias",
                        "internal": request.internal,
                        "external": f"chatcmpl-{request.x_request_id}",
                        **stamp(request.admitted_ns),
                    },
                )
            )
        timed.extend((at, 0, record) for at, record in self.extra)
        timed.extend(self._steps(requests))
        timed.sort(key=lambda item: (item[0], item[1]))
        end = max(at for at, _, _ in timed) + SECOND
        records = [record for _, _, record in timed]
        records.append({"kind": "goodbye", **stamp(end), "last_seq": len(records) + 1})
        beats = _with_heartbeats(records, self.dropped_from)
        return [self._restamp(record) for record in beats]

    def _restamp(self, record: dict[str, Any]) -> dict[str, Any]:
        """Apply the wall clock's jump, or drop the second reads."""
        out = dict(record)
        for prefix in ("", "start_", "end_"):
            mono = out.get(f"{prefix}mono_ns")
            if not isinstance(mono, int):
                continue
            if f"{prefix}wall_ns" in out:
                out[f"{prefix}wall_ns"] = self.wall(mono)
            if not self.bracketed:
                out.pop(f"{prefix}wall_after_ns", None)
            elif f"{prefix}wall_after_ns" in out:
                out[f"{prefix}wall_after_ns"] = self.wall(mono) + 800
        return out

    def _steps(
        self, requests: list[SimRequest]
    ) -> list[tuple[int, int, dict[str, Any]]]:
        waiting = deque(sorted(requests, key=lambda r: r.enqueued_ns))
        running: dict[str, tuple[SimRequest, int]] = {}  # internal -> tokens
        resumed: dict[str, int] = {}  # preempted, waiting: tokens kept
        out: list[tuple[int, int, dict[str, Any]]] = []
        now = waiting[0].enqueued_ns if waiting else 0
        iteration = 0
        while waiting or running:
            if not running and waiting[0].enqueued_ns > now:
                now = waiting[0].enqueued_ns + self.wake_ns
                continue
            preempted = self._make_room(running, waiting, resumed)
            # vLLM schedules waiting requests only in a step that preempted none.
            admitted = [] if preempted else self._admit(now, running, waiting, resumed)
            members = [_admission(r, resumed.pop(r.internal, None)) for r in admitted]
            members += [_member(r, first=False, done=n) for r, n in running.values()]
            for request in admitted:
                if request.first_step is None:
                    request.first_step = iteration
                tokens = next(
                    (
                        m["output_before"]
                        for m in members
                        if m["internal"] == request.internal
                    ),
                    0,
                )
                running[request.internal] = (request, tokens)
            total = sum(m["scheduled"] for m in members)
            out.append((now, 1, _scheduled(iteration, now, members, total, preempted)))
            finished_at = now + self.step_ns
            out.extend(self._complete(iteration, finished_at, running))
            iteration += 1
            now = finished_at + self.gap_ns
            now += self._stalled(finished_at)
        return out

    def _stalled(self, finished_at: int) -> int:
        """How long the loop stops after a step completing at finished_at:
        each stall stops it once, after the first step at or past its time."""
        pending = sorted([*self.stalls, *([self.stall] if self.stall else [])])
        due = [stall for stall in pending if finished_at >= stall[0]]
        self.stall = None
        self.stalls = [stall for stall in pending if stall not in due]
        return sum(length for _, length in due)

    def _make_room(
        self,
        running: dict[str, tuple[SimRequest, int]],
        waiting: deque[SimRequest],
        resumed: dict[str, int],
    ) -> list[str]:
        """Preempt the latest admitted requests until the step's growth fits
        the KV budget; they go back to the front of the queue."""
        preempted: list[str] = []
        while (
            self.kv_tokens is not None
            and running
            and _usage(running) + len(running) > self.kv_tokens
        ):
            internal = next(reversed(running))
            request, tokens = running.pop(internal)
            resumed[internal] = tokens
            waiting.appendleft(request)
            preempted.append(internal)
        return preempted

    def _admit(
        self,
        now: int,
        running: dict[str, tuple[SimRequest, int]],
        waiting: deque[SimRequest],
        resumed: dict[str, int],
    ) -> list[SimRequest]:
        admitted: list[SimRequest] = []
        usage = _usage(running) + len(running)
        while (
            waiting
            and waiting[0].enqueued_ns <= now
            and len(running) + len(admitted) < self.max_num_seqs
        ):
            need = waiting[0].prompt + resumed.get(waiting[0].internal, 0) + 1
            if self.kv_tokens is not None and usage + need > self.kv_tokens:
                break
            usage += need
            admitted.append(waiting.popleft())
        return admitted

    def _complete(
        self,
        iteration: int,
        finished_at: int,
        running: dict[str, tuple[SimRequest, int]],
    ) -> list[tuple[int, int, dict[str, Any]]]:
        """Every running request gains a token; finished ones are freed."""
        out: list[tuple[int, int, dict[str, Any]]] = []
        done = []
        for internal, (request, tokens) in list(running.items()):
            tokens += 1
            if tokens == 1:
                request.first_done_ns = finished_at
            finish = "length" if tokens >= request.output else None
            done.append(_done(internal, tokens, request, finish))
            running[internal] = (request, tokens)
            if finish is not None:
                request.last_done_ns = finished_at
                del running[internal]
                out.append((finished_at - 1, 2, _terminal(request, finished_at - 1)))
        out.append(
            (
                finished_at,
                3,
                {
                    "kind": "completed",
                    "iteration": str(iteration),
                    **stamp(finished_at),
                    "members": done,
                },
            )
        )
        return out


def _usage(running: dict[str, tuple[SimRequest, int]]) -> int:
    """KV slots the running requests hold: prompt and output so far."""
    return sum(request.prompt + tokens for request, tokens in running.values())


def _terminal(request: SimRequest, at: int) -> dict[str, Any]:
    return {
        "kind": "terminal",
        "internal": request.internal,
        "status": "FINISHED_LENGTH_CAPPED",
        "finish_reason": "length",
        "output_tokens": request.output,
        **stamp(at),
    }


def _admission(request: SimRequest, kept: int | None) -> dict[str, Any]:
    """A first schedule, or the resume of a preempted request, which
    recomputes its prompt and the output it kept."""
    if kept is None:
        return _member(request, first=True)
    return {
        "internal": request.internal,
        "sighting": "repeat",
        "phase": "context",
        "scheduled": request.prompt + kept,
        "computed_before": 0,
        "prompt_tokens": request.prompt,
        "prefill_scheduled": request.prompt,
        "past_prompt_scheduled": kept,
        "drafts_scheduled": 0,
        "cached_at_admission": None,
        "recompute": True,
        "output_before": kept,
        "resumable": False,
    }


def _enqueued(request: SimRequest) -> tuple[int, int, dict[str, Any]]:
    return (
        request.enqueued_ns,
        0,
        {
            "kind": "enqueued",
            "internal": request.internal,
            "structured_output": request.structured_output,
            "resumable": request.resumable,
            **stamp(request.enqueued_ns),
        },
    )


def _member(request: SimRequest, *, first: bool, done: int = 0) -> dict[str, Any]:
    computed = 0 if first else request.prompt + done - 1
    scheduled = request.prompt - request.cached if first else 1
    return {
        "internal": request.internal,
        "sighting": "first" if first else "repeat",
        "phase": "context" if first else "generation",
        "scheduled": scheduled,
        "computed_before": request.cached if first else computed,
        "prompt_tokens": request.prompt,
        "prefill_scheduled": scheduled if first else 0,
        "past_prompt_scheduled": 0 if first else 1,
        "drafts_scheduled": 0,
        "cached_at_admission": request.cached if first else None,
        "recompute": False,
        "output_before": 0 if first else done,
        "resumable": request.resumable,
    }


def _done(
    internal: str, tokens: int, request: SimRequest, finish: str | None
) -> dict[str, Any]:
    return {
        "internal": internal,
        "outcome": "kept",
        "stale": False,
        "sampled": 1,
        "accepted_drafts": 0,
        "retained": 1,
        "finish_reason": finish,
        "computed_after": request.prompt + tokens - 1,
    }


def _scheduled(
    iteration: int,
    now: int,
    members: list[dict[str, Any]],
    total: int,
    preempted: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "scheduled",
        "iteration": str(iteration),
        **stamp(now, "start_"),
        **stamp(now + 50_000, "end_"),
        "total_tokens": total,
        "zero_token": total == 0,
        "preempted": list(preempted or []),
        "pause_state": "UNPAUSED",
        "members": members,
    }


def _with_heartbeats(
    records: list[dict[str, Any]], dropped_from: int | None = None
) -> list[dict[str, Any]]:
    """Insert a heartbeat every second of engine time; its counters are
    clean, or from ``dropped_from`` count one dropped oversized reset."""
    out: list[dict[str, Any]] = []
    next_beat: int | None = None
    for record in records:
        at = record.get("mono_ns", record.get("start_mono_ns"))
        if isinstance(at, int):
            if next_beat is None:
                next_beat = at
            while at >= next_beat:
                out.append(
                    {
                        "kind": "heartbeat",
                        **stamp(next_beat),
                        "last_seq": len(out),
                        "dropped": (
                            {"cache_reset_oversized": 1}
                            if dropped_from is not None and next_beat >= dropped_from
                            else {}
                        ),
                        "errors": 0,
                        "bytes": 1024,
                        "capped": False,
                        "queued": 0,
                        "pending": 0,
                        "reserved": 0,
                    }
                )
                next_beat += SECOND
        out.append(record)
    return out


def hello_record(engine: Engine) -> dict[str, Any]:
    first = {
        "kind": "hello",
        "role": "engine",
        "host": HOST,
        "boot_id": BOOT,
        "pid": PID,
        "start_ns": START,
        "vllm_version": "0.30.0",
        "enabled": True,
        "refused": None,
        "producer": f"vllm:{HOST}:{BOOT}:{PID}:{START}",
        "config": {
            "executor": "uni",
            "tp": 1,
            "max_num_seqs": engine.max_num_seqs,
            "max_num_batched_tokens": engine.max_num_batched_tokens,
            "request_id_randomization": True,
            **engine.config,
        },
        "clock": {**stamp(0), "gap_ns": 800},
    }
    if engine.observes is not None:
        first["observes"] = engine.observes
    return first


def client_records(
    request: SimRequest, engine: Engine | None = None
) -> list[dict[str, Any]]:
    """What ``infer profile`` writes for one request, in append order."""
    wall = (engine or Engine()).wall
    sent = wall(request.sent_ns)
    common = {
        "schema_version": 1,
        "session_id": SESSION,
        "request_id": request.request_id,
        "x_request_id": request.x_request_id,
        "case_id": request.case_id,
        "phase": request.phase,
    }
    intended: int | None = wall(
        request.intended_ns if request.intended_ns is not None else request.sent_ns
    )
    if request.closed_loop:
        intended = None
    records = [
        {
            **common,
            "event_type": "infer.dispatch",
            "intended_at_ns": intended,
            "started_at_ns": sent,
            "timestamp_ns": sent,
        }
    ]
    if request.first_done_ns is None or request.last_done_ns is None:
        return records
    first = wall(request.first_done_ns + request.delivery_ns)
    ended = wall(request.last_done_ns + request.delivery_ns)
    records.append(
        {
            **common,
            "event_type": "infer.first_content",
            "first_content_at_ns": first,
            "timestamp_ns": first,
        }
    )
    records.append(
        {
            **common,
            "event_type": "infer.request",
            "started_at_ns": sent,
            "ended_at_ns": ended,
            "timestamp_ns": sent,
            "status": "ok",
            "ttft_ms": (first - sent) / MS,
            "e2e_latency_ms": (ended - sent) / MS,
            "prompt_tokens": request.prompt,
            "output_tokens": request.output,
            "target_input_tokens": request.prompt,
            "target_output_tokens": request.output,
            "arrival_mode": "poisson",
            "intended_at_ns": intended,
            "dispatch_lag_ms": None if intended is None else (sent - intended) / MS,
            "held_for_slot": request.held_for_slot,
            "shared_prefix_tokens": request.shared_prefix_tokens,
            "prefix_group": request.prefix_group,
        }
    )
    return records


def identity_record(host: str = HOST) -> dict[str, Any]:
    return ArtifactIdentityEvent(
        context=CorrelationContext(
            run_id=RUN,
            session_id=SESSION,
            producer_id="stormlog.infer.profile",
            source="stormlog.infer.profile",
            clock_domain=f"{host}/{BOOT}/unix_epoch_ns",
            clock_kind="wall",
            collection_mode="active",
            provenance="observed",
        ),
        event_id="artifact",
        artifact_kind="inference_jsonl",
        created_at_ns=WALL_OFFSET,
    ).to_record()


def build_run(
    tmp_path: Path,
    requests: list[SimRequest],
    engine: Engine | None = None,
    *,
    windows: list[dict[str, Any]] | None = None,
    client_host: str = HOST,
    client: bool = True,
) -> Path:
    """Serve ``requests``, write the client artifact, import the hook log."""
    engine = engine or Engine()
    raw = engine.run(requests)
    write_epoch(tmp_path / "hook", "engine", PID, START, [hello_record(engine), *raw])
    lines = [identity_record(client_host)]
    events: list[tuple[int, dict[str, Any]]] = []
    for request in requests if client else ():
        events.extend((r["timestamp_ns"], r) for r in client_records(request, engine))
    for window in windows or []:
        events.append((window["timestamp_ns"], window))
    events.sort(key=lambda item: item[0])
    lines.extend(record for _, record in events)
    artifact = tmp_path / "infer.jsonl"
    artifact.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    end = max(r.get("mono_ns", 0) for r in raw)
    import_execution_into_artifact(
        artifact, tmp_path / "hook", importer=importer(end + SECOND)
    )
    return artifact


def poisson_free(
    count: int, start_ns: int, gap_ns: int, **fields: Any
) -> list[SimRequest]:
    """``count`` requests sent every ``gap_ns`` from ``start_ns``."""
    prefix = fields.pop("prefix", "r")
    return [
        SimRequest(f"{prefix}{index}", start_ns + index * gap_ns, **fields)
        for index in range(count)
    ]


__all__ = [
    "EPOCH",
    "MS",
    "OBSERVES",
    "RUN",
    "SESSION",
    "Engine",
    "SimRequest",
    "build_run",
    "client_records",
    "poisson_free",
]
