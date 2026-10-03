"""A simulated continuous-batching engine, scheduled the way vLLM 0.30's V1
scheduler is: running requests first, then waiting ones in arrival order.

Nothing is computed. Each step sleeps for its simulated cost; a request whose
context is fully computed samples one token. KV blocks, prefix-cache reuse and
recompute preemption follow vLLM closely enough that each mechanism moves the
counters vLLM exports, and the execution hook's records describe the steps
exactly. Observers (the hook log, the span exporter, the profiler) are called
outside the engine lock.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .blocks import BlockPool
from .config import FakeEngineConfig
from .hold import Hold
from .stats import EngineStats, StatsSnapshot, snapshot

# A fixed chat-template prefix, as a server's template adds to every prompt.
TEMPLATE_TOKENS = ("<|im_start|>", "user", "\n", "<|im_end|>")
# Every sampled token; its blocks still hash apart, as hashes chain.
GENERATED_TOKEN = "tok"
STATUS_FOR_REASON = {
    "length": "FINISHED_LENGTH_CAPPED",
    "stop": "FINISHED_STOPPED",
    "abort": "FINISHED_ABORTED",
}


def prompt_tokens(text: str) -> tuple[str, ...]:
    """The fake tokenizer: the template, then the prompt's whitespace words."""
    return TEMPLATE_TOKENS + tuple(text.split())


@dataclass(eq=False)
class FakeRequest:
    internal_id: str
    external_id: str
    prompt: tuple[str, ...]
    max_tokens: int
    arrival_ns: int
    traceparent: str | None = None
    status: str = "WAITING"
    computed: int = 0
    committed: int = 0
    output_tokens: int = 0
    block_ids: list[int] = field(default_factory=list)
    block_hashes: list[int] = field(default_factory=list)
    preemptions: int = 0
    # Preempted and not yet scheduled again: vLLM's output lists it as new.
    resumed: bool = False
    cached_at_admission: int | None = None
    first_scheduled_ns: int | None = None
    first_token_ns: int | None = None
    finished_ns: int | None = None
    finish_reason: str | None = None
    token_ns: list[int] = field(default_factory=list)
    events: "queue.Queue[tuple[str, Any]]" = field(default_factory=queue.Queue)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)

    @property
    def num_tokens(self) -> int:
        return self.prompt_len + self.output_tokens

    @property
    def tokens(self) -> tuple[str, ...]:
        """The prompt and every sampled token, as vLLM's block hashes see them."""
        return self.prompt + (GENERATED_TOKEN,) * self.output_tokens

    @property
    def finished(self) -> bool:
        return self.finished_ns is not None


@dataclass
class ScheduledMember:
    request: FakeRequest
    tokens: int
    computed_before: int
    first_sighting: bool
    recompute: bool
    output_before: int
    context: bool
    sampled: int = 0
    outcome: str = "kept"
    finish_reason: str | None = None


@dataclass
class Step:
    iteration: int
    start_wall_ns: int
    start_mono_ns: int
    members: list[ScheduledMember]
    preempted: list[str]
    end_wall_ns: int = 0
    end_mono_ns: int = 0
    exec_start_ns: int = 0
    exec_end_ns: int = 0

    @property
    def total_tokens(self) -> int:
        return sum(member.tokens for member in self.members)

    @property
    def prefill_tokens(self) -> int:
        return sum(
            max(0, min(m.tokens, m.request.prompt_len - m.computed_before))
            for m in self.members
        )


class EngineObserver:
    """Callbacks from the engine; each default does nothing."""

    def on_admit(self, request: FakeRequest) -> None:
        return None

    def on_scheduled(self, step: Step) -> None:
        return None

    def on_executed(self, step: Step) -> None:
        return None

    def on_completed(self, step: Step) -> None:
        return None

    def on_free(self, request: FakeRequest) -> None:
        return None


@dataclass
class _LoopCall:
    action: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class Engine:
    """The scheduler and its step loop; every public method is thread-safe."""

    def __init__(self, config: FakeEngineConfig) -> None:
        self.config = config
        self.pool = BlockPool(
            config.num_gpu_blocks,
            config.block_size,
            caching=config.enable_prefix_caching,
        )
        self.observers: list[EngineObserver] = []
        self.waiting: deque[FakeRequest] = deque()
        self.running: list[FakeRequest] = []
        self.live: dict[str, FakeRequest] = {}
        self.finished: list[FakeRequest] = []
        # (victim, running IDs in order just before it was popped).
        self.preemption_log: list[tuple[str, tuple[str, ...]]] = []
        self.steps: list[Step] = []
        self.start_ns = time.time_ns()
        self.stats = EngineStats(created_s=self.start_ns / 1e9)
        self.waiting_capacity = 0
        self._sighted: set[str] = set()
        # Victims of a reset, listed in the next step's preempted as vLLM's
        # reset_preempted_req_ids are.
        self._pending_preempted: list[str] = []
        # A request finished since the last schedule: vLLM's finished_req_ids,
        # which keep has_requests() true for one more, maybe empty, step.
        self._finished_since_schedule = False
        self._iteration = 0
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._gate = Hold()
        self._calls: deque[_LoopCall] = deque()
        # Whether the loop takes calls; before start and after its last
        # drain, a call runs at once on the caller's thread.
        self._loop_running = False
        self._stopping = False
        self._snapshot = self._take_snapshot()
        self._thread = threading.Thread(
            target=self._run, name="fake-engine-core", daemon=True
        )

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._loop_running = True
        self._thread.start()

    def stop(self) -> None:
        with self._wake:
            self._stopping = True
            self._wake.notify_all()
        self._gate.resume()
        self._thread.join(timeout=10)
        with self._lock:
            pending = [*self.waiting, *self.running]
        for request in pending:
            self.abort(request)

    @property
    def loop_thread_id(self) -> int | None:
        return self._thread.ident

    # ------------------------------------------------------------ requests

    def submit(self, request: FakeRequest) -> FakeRequest:
        request.block_hashes = self.pool.block_hashes(request.prompt)
        # Observed before the loop can see it, as the hook writes alias before
        # vLLM hands the request to its scheduler.
        for observer in self.observers:
            observer.on_admit(request)
        with self._wake:
            self.live[request.internal_id] = request
            self.waiting.append(request)
            self._wake.notify_all()
        return request

    def abort(self, request: FakeRequest) -> None:
        with self._wake:
            if request.finished:
                return
            self._finish(request, "abort")
        self._notify_free(request)

    # ------------------------------------------------------------ controls

    def pause(self, seconds: float | None = None) -> None:
        """Hold the step loop before its next step; release after ``seconds``,
        or on ``resume()``. Pauses stack: the last to end releases it."""
        self._gate.pause(seconds)

    def resume(self) -> None:
        self._gate.resume()

    @property
    def paused(self) -> bool:
        return self._gate.held

    def call_in_loop(self, action: Callable[[], Any], timeout: float = 60.0) -> Any:
        """Run ``action`` on the step loop between steps, as vLLM runs its
        utility calls; a paused loop runs it once released."""
        call = _LoopCall(action)
        with self._wake:
            queued = self._loop_running
            if queued:
                self._calls.append(call)
                self._wake.notify_all()
        if not queued:
            # No loop runs steps now (a test drives them by hand, or it has
            # stopped), so now is between steps.
            return action()
        if not call.done.wait(timeout):
            raise TimeoutError("the engine loop did not run the call in time")
        if call.error is not None:
            raise call.error
        return call.result

    def reset_prefix_cache(self, reset_running_requests: bool) -> bool:
        """vLLM's semantics: refused while blocks are held, unless every
        running request is preempted first. Like vLLM's utility call it runs on
        the step loop, between steps, so no step is in flight when it acts."""
        result: bool = self.call_in_loop(
            lambda: self._reset_in_loop(reset_running_requests)
        )
        return result

    def _reset_in_loop(self, reset_running_requests: bool) -> bool:
        with self._lock:
            if reset_running_requests:
                while self.running:
                    self._preempt(self.running.pop(), self._pending_preempted)
            return self.pool.reset()

    def metrics_snapshot(self) -> StatsSnapshot:
        with self._lock:
            return self._snapshot

    # ------------------------------------------------------------ the loop

    def _run(self) -> None:
        while not self._stopping:
            if not self._gate.wait(timeout=0.05):
                continue
            self._run_calls()
            step = self._schedule()
            if step is None:
                with self._wake:
                    if not self._stopping and not self._calls:
                        self._wake.wait(timeout=0.01)
                continue
            for observer in self.observers:
                observer.on_scheduled(step)
            self._execute(step)
            for observer in self.observers:
                observer.on_executed(step)
            self._complete(step)
            if not step.total_tokens and self._has_requests():
                time.sleep(0.001)  # vLLM's yield after a step that ran nothing
        with self._lock:
            self._loop_running = False
        self._run_calls()  # those queued before the loop stopped taking calls

    def _run_calls(self) -> None:
        while True:
            with self._lock:
                if not self._calls:
                    return
                call = self._calls.popleft()
            try:
                call.result = call.action()
            except BaseException as exc:  # handed back to the caller
                call.error = exc
            call.done.set()

    def _schedule(self) -> Step | None:
        start = (time.time_ns(), time.monotonic_ns())
        with self._lock:
            # vLLM's step() schedules whenever has_requests(), so a step can
            # have no members: after the last finish, or while none fit.
            if not self._has_requests_locked():
                return None
            self._finished_since_schedule = False
            budget = self.config.max_num_batched_tokens
            members: list[ScheduledMember] = []
            preempted: list[str] = []
            budget = self._schedule_running(members, preempted, budget)
            # As in vLLM, only this step's own preemptions stop admission.
            if not preempted:
                self._schedule_waiting(members, budget)
            carried, self._pending_preempted = self._pending_preempted, []
            preempted = carried + preempted
            step = Step(self._iteration, start[0], start[1], members, preempted)
            self._iteration += 1
            step.end_wall_ns, step.end_mono_ns = time.time_ns(), time.monotonic_ns()
            return step

    def _has_requests(self) -> bool:
        with self._lock:
            return self._has_requests_locked()

    def _has_requests_locked(self) -> bool:
        return bool(self.waiting or self.running or self._finished_since_schedule)

    def _schedule_running(
        self, members: list[ScheduledMember], preempted: list[str], budget: int
    ) -> int:
        index = 0
        while index < len(self.running) and budget > 0:
            request = self.running[index]
            want = min(request.num_tokens - request.computed, budget)
            while not self._grow(request, want):
                self._preempt(self.running.pop(), preempted)
                if request.status != "RUNNING":
                    return budget
            members.append(self._member(request, want))
            budget -= want
            index += 1
        return budget

    def _schedule_waiting(self, members: list[ScheduledMember], budget: int) -> None:
        self.waiting_capacity = 0
        while (
            self.waiting and budget > 0 and len(self.running) < self.config.max_num_seqs
        ):
            request = self.waiting[0]
            # A resumed request can reuse its generated blocks too.
            request.block_hashes = self.pool.block_hashes(request.tokens)
            cached = self.pool.lookup(request.block_hashes, request.num_tokens)
            self.pool.touch(cached)
            request.block_ids = list(cached)
            request.computed = len(cached) * self.config.block_size
            want = min(request.num_tokens - request.computed, budget)
            if not self._grow(request, want):
                self.pool.free(request.block_ids)
                request.block_ids, request.computed = [], 0
                self.waiting_capacity = len(self.waiting)
                return
            self.waiting.popleft()
            self.running.append(request)
            self._admit(request)
            members.append(self._member(request, want))
            budget -= want
        if self.waiting and len(self.running) >= self.config.max_num_seqs:
            self.waiting_capacity = len(self.waiting)

    def _admit(self, request: FakeRequest) -> None:
        request.status = "RUNNING"
        self.stats.prefix_queries += request.num_tokens
        self.stats.prefix_hits += request.computed
        if request.first_scheduled_ns is None:
            request.first_scheduled_ns = time.time_ns()
            request.cached_at_admission = request.computed
            self.stats.observe(
                "vllm:request_queue_time_seconds",
                (request.first_scheduled_ns - request.arrival_ns) / 1e9,
            )

    def _member(self, request: FakeRequest, tokens: int) -> ScheduledMember:
        first = request.internal_id not in self._sighted
        self._sighted.add(request.internal_id)
        before = request.computed
        resumed, request.resumed = request.resumed, False
        return ScheduledMember(
            request=request,
            tokens=tokens,
            computed_before=before,
            first_sighting=first,
            recompute=(not first) and before < request.committed,
            output_before=request.output_tokens,
            # vLLM's phase: context for a request new in this output, a resumed
            # one included, or one still in its context phase.
            context=resumed
            or request.output_tokens == 0
            or request.num_tokens - before > 1,
        )

    def _grow(self, request: FakeRequest, tokens: int) -> bool:
        needed = math.ceil((request.computed + tokens) / self.config.block_size)
        extra = needed - len(request.block_ids)
        if extra <= 0:
            return True
        taken = self.pool.allocate(extra)
        if taken is None:
            return False
        request.block_ids.extend(taken)
        return True

    def _preempt(self, victim: FakeRequest, preempted: list[str]) -> None:
        order = tuple(request.internal_id for request in [*self.running, victim])
        self.preemption_log.append((victim.internal_id, order))
        self.pool.free(victim.block_ids)
        victim.block_ids = []
        victim.computed = 0
        victim.status = "PREEMPTED"
        victim.resumed = True
        victim.preemptions += 1
        self.waiting.appendleft(victim)
        self.stats.preemptions += 1
        preempted.append(victim.internal_id)

    def _execute(self, step: Step) -> None:
        config = self.config
        decode = sum(1 for member in step.members if not member.context)
        # A step with no tokens runs no forward pass.
        seconds = (
            config.step_seconds
            + step.prefill_tokens * config.prefill_token_seconds
            + decode * config.decode_token_seconds
            if step.total_tokens
            else 0.0
        )
        step.exec_start_ns = time.time_ns()
        time.sleep(seconds)
        step.exec_end_ns = time.time_ns()

    def _complete(self, step: Step) -> None:
        freed: list[FakeRequest] = []
        with self._lock:
            for member in step.members:
                self._complete_member(member, freed)
            self._observe_iteration(step)
            self.steps.append(step)
            self._snapshot = self._take_snapshot()
        for request in freed:
            self._notify_free(request)
        for observer in self.observers:
            observer.on_completed(step)

    def _observe_iteration(self, step: Step) -> None:
        """vLLM's front end records an iteration only for an output that
        carries tokens: the prompt tokens computed for the requests whose
        first token it carries, plus the tokens generated."""
        kept = [m for m in step.members if m.outcome == "kept" and m.sampled]
        if not kept:
            return
        computed = sum(
            m.request.prompt_len - (m.request.cached_at_admission or 0)
            for m in kept
            if m.output_before == 0
        )
        generated = sum(m.sampled for m in kept)
        self.stats.observe("vllm:iteration_tokens_total", computed + generated)

    def _complete_member(
        self, member: ScheduledMember, freed: list[FakeRequest]
    ) -> None:
        request = member.request
        # The sampler's count, which the hook reads before the update whatever
        # the outcome: a token once the step completes the request's context.
        context_end = request.prompt_len + member.output_before
        member.sampled = int(member.computed_before + member.tokens >= context_end)
        if request.finished:
            member.outcome = "discarded_finished"
            return
        if request.status != "RUNNING":
            # Only async scheduling can preempt a request while its step is in
            # flight (vLLM then records dropped_stale); this engine schedules
            # synchronously and runs resets between steps.
            raise AssertionError(f"{request.internal_id} was preempted mid-step")
        request.computed += member.tokens
        request.committed = request.computed
        self._cache_full_blocks(request)
        if not member.sampled:
            return
        self._sample(request)
        if request.output_tokens >= request.max_tokens:
            self._finish(request, "length")
            member.finish_reason = "length"
            freed.append(request)

    def _cache_full_blocks(self, request: FakeRequest) -> None:
        """Cache every full computed block, generated tokens included, as
        vLLM does."""
        full = request.computed // self.config.block_size
        if len(request.block_hashes) < full:
            request.block_hashes = self.pool.block_hashes(request.tokens)
        for index in range(min(full, len(request.block_hashes))):
            self.pool.cache(request.block_ids[index], request.block_hashes[index])

    def _sample(self, request: FakeRequest) -> None:
        now = time.time_ns()
        if request.first_token_ns is None:
            request.first_token_ns = now
            self.stats.prompt_tokens += request.prompt_len
            self.stats.prompt_tokens_cached += request.cached_at_admission or 0
            self.stats.observe(
                "vllm:time_to_first_token_seconds",
                (now - request.arrival_ns) / 1e9,
            )
        else:
            self.stats.observe(
                "vllm:inter_token_latency_seconds", (now - request.token_ns[-1]) / 1e9
            )
        request.output_tokens += 1
        request.token_ns.append(now)
        self.stats.generation_tokens += 1
        request.events.put(("token", now))

    def _finish(self, request: FakeRequest, reason: str) -> None:
        """Mark a request done and free it; the caller holds the lock."""
        now = time.time_ns()
        request.status = STATUS_FOR_REASON.get(reason, "FINISHED_STOPPED")
        request.finish_reason = reason
        request.finished_ns = now
        self._finished_since_schedule = True
        if request in self.running:
            self.running.remove(request)
        if request in self.waiting:
            self.waiting.remove(request)
        self.pool.free(request.block_ids)
        request.block_ids = []
        self.live.pop(request.internal_id, None)
        self.finished.append(request)
        self._observe_finish(request, now)
        request.events.put(("finish", reason))

    def _observe_finish(self, request: FakeRequest, now: int) -> None:
        stats = self.stats
        stats.success[request.finish_reason or "stop"] += 1
        stats.observe(
            "vllm:e2e_request_latency_seconds", (now - request.arrival_ns) / 1e9
        )
        stats.observe("vllm:request_prompt_tokens", request.prompt_len)
        stats.observe("vllm:request_generation_tokens", request.output_tokens)
        stats.observe("vllm:request_num_preemptions", request.preemptions)
        stats.observe(
            "vllm:request_prefill_kv_computed_tokens",
            request.prompt_len - (request.cached_at_admission or 0),
        )
        # One completion per request, so n is 1 and its longest is its own.
        stats.observe("vllm:request_max_num_generation_tokens", request.output_tokens)
        stats.observe("vllm:request_params_n", 1)
        stats.observe("vllm:request_params_max_tokens", request.max_tokens)
        if request.first_scheduled_ns is None or request.first_token_ns is None:
            return
        scheduled = request.first_scheduled_ns
        stats.observe("vllm:request_inference_time_seconds", (now - scheduled) / 1e9)
        stats.observe(
            "vllm:request_prefill_time_seconds",
            (request.first_token_ns - scheduled) / 1e9,
        )
        stats.observe(
            "vllm:request_decode_time_seconds", (now - request.first_token_ns) / 1e9
        )
        if request.output_tokens > 1:
            stats.observe(
                "vllm:request_time_per_output_token_seconds",
                (now - request.first_token_ns) / 1e9 / (request.output_tokens - 1),
            )

    def _notify_free(self, request: FakeRequest) -> None:
        for observer in self.observers:
            observer.on_free(request)

    def _take_snapshot(self) -> StatsSnapshot:
        return snapshot(
            taken_ns=time.time_ns(),
            running=len(self.running),
            waiting=len(self.waiting),
            waiting_capacity=self.waiting_capacity,
            kv_usage=self.pool.usage(),
            stats=self.stats,
        )


__all__ = [
    "GENERATED_TOKEN",
    "TEMPLATE_TOKENS",
    "Engine",
    "EngineObserver",
    "FakeRequest",
    "ScheduledMember",
    "Step",
    "prompt_tokens",
]
