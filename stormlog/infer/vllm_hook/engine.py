"""Engine-core snapshots: admissions, scheduled steps, their outputs, and exits.

Every value is a scalar copied at the moment it is observed; nothing keeps a
live vLLM object. Per-step context comes from the scheduler output's own fields
because ``Scheduler.schedule`` has already advanced each request's
``num_computed_tokens`` by the time it returns, and under async scheduling the
next step has been scheduled before this one's output is processed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .writer import EpochWriter, stamp

ITERATION_ATTRIBUTE = "_stormlog_iteration"
# vLLM keeps at most a few steps in flight; more pending means outputs were
# never processed (an error or shutdown), and the oldest are forgotten.
MAX_PENDING = 64


@dataclass
class _Member:
    internal: str
    scheduled: int
    computed_before: int
    prompt_tokens: int | None
    drafts: int


@dataclass
class _Pending:
    iteration: str
    members: dict[str, _Member]


@dataclass
class EngineRecorder:
    """Records for one scheduler instance."""

    writer: EpochWriter
    producer: str
    next_iteration: int = 0
    # Per live request: its prompt length, and its committed context. Both are
    # dropped when vLLM frees the request, so they never outgrow its live set.
    prompt_tokens: dict[str, int] = field(default_factory=dict)
    committed: dict[str, int] = field(default_factory=dict)
    pending: dict[str, _Pending] = field(default_factory=dict)

    # ------------------------------------------------------------ admission

    def on_admit(self, request: Any) -> None:
        self.writer.emit(
            "alias",
            {
                "internal": str(request.request_id),
                "external": _optional_str(getattr(request, "external_req_id", None)),
                **stamp(),
            },
        )

    def on_enqueue(self, request: Any, at: dict[str, int]) -> None:
        """A request entering the scheduler's waiting queue, stamped just
        before ``add_request`` ran."""
        self.writer.emit(
            "enqueued",
            {
                "internal": str(request.request_id),
                "structured_output": _optional_bool(
                    getattr(request, "use_structured_output", None)
                ),
                "resumable": _optional_bool(getattr(request, "resumable", None)),
                **at,
            },
        )

    # ------------------------------------------------------------ schedule

    def on_schedule(self, scheduler: Any, output: Any, start: dict[str, int]) -> None:
        iteration = str(self.next_iteration)
        self.next_iteration += 1
        setattr(output, ITERATION_ATTRIBUTE, (self.producer, iteration))
        members = [
            *self._new_members(scheduler, output),
            *self._cached_members(scheduler, output),
        ]
        self.pending[iteration] = _Pending(
            iteration, {member.internal: member for member, _ in members}
        )
        while len(self.pending) > MAX_PENDING:
            self.pending.pop(next(iter(self.pending)))
        end = stamp()
        self.writer.emit(
            "scheduled",
            {
                "iteration": iteration,
                **{f"start_{name}": value for name, value in start.items()},
                **{f"end_{name}": value for name, value in end.items()},
                "total_tokens": int(output.total_num_scheduled_tokens),
                "zero_token": int(output.total_num_scheduled_tokens) == 0,
                "preempted": sorted(getattr(output, "preempted_req_ids", None) or ()),
                "pause_state": _state_name(getattr(scheduler, "pause_state", None)),
                "members": [fields for _, fields in members],
            },
        )

    def _new_members(
        self, scheduler: Any, output: Any
    ) -> list[tuple[_Member, dict[str, Any]]]:
        members = []
        for data in output.scheduled_new_reqs:
            internal = str(data.req_id)
            first = internal not in self.prompt_tokens
            if first:
                self.prompt_tokens[internal] = int(data.prompt_len)
            members.append(
                self._member(
                    scheduler,
                    output,
                    internal,
                    computed_before=int(data.num_computed_tokens),
                    first=first,
                    context=True,
                )
            )
        return members

    def _cached_members(
        self, scheduler: Any, output: Any
    ) -> list[tuple[_Member, dict[str, Any]]]:
        cached = output.scheduled_cached_reqs
        is_context = getattr(cached, "is_context_phase", None)
        return [
            self._member(
                scheduler,
                output,
                str(internal),
                computed_before=int(cached.num_computed_tokens[index]),
                first=str(internal) not in self.prompt_tokens,
                context=bool(is_context(internal)) if callable(is_context) else None,
            )
            for index, internal in enumerate(cached.req_ids)
        ]

    def _member(
        self,
        scheduler: Any,
        output: Any,
        internal: str,
        *,
        computed_before: int,
        first: bool,
        context: bool | None,
    ) -> tuple[_Member, dict[str, Any]]:
        scheduled = int(output.num_scheduled_tokens.get(internal, 0))
        drafts = len(output.scheduled_spec_decode_tokens.get(internal, ()) or ())
        request = scheduler.requests.get(internal)
        if request is not None:
            # A resumable (streaming-input) request grows its prompt between
            # sessions, so the live length wins over the first sighting's.
            self.prompt_tokens[internal] = int(request.num_prompt_tokens)
        prompt = self.prompt_tokens.get(internal)
        prefill = max(0, min(scheduled, prompt - computed_before)) if prompt else 0
        member = _Member(internal, scheduled, computed_before, prompt, drafts)
        fields = {
            "internal": internal,
            "sighting": "first" if first else "repeat",
            # vLLM's own classification: new in this output, or a cached request
            # still in its context phase. A recomputed request is context.
            "phase": (
                None if context is None else ("context" if context else "generation")
            ),
            "scheduled": scheduled,
            "computed_before": computed_before,
            "prompt_tokens": prompt,
            "prefill_scheduled": prefill,
            "past_prompt_scheduled": scheduled - prefill,
            "drafts_scheduled": drafts,
            "cached_at_admission": computed_before if first else None,
            "recompute": (not first)
            and computed_before < self.committed.get(internal, computed_before),
            "output_before": (
                int(request.num_output_tokens) if request is not None else None
            ),
            "resumable": bool(getattr(request, "resumable", False)),
        }
        return member, fields

    # ------------------------------------------------------------ output

    def before_update(
        self, scheduler: Any, output: Any, model_output: Any
    ) -> dict[str, dict[str, Any]]:
        """Per member, what vLLM's own output loop is about to read."""
        sampled = getattr(model_output, "sampled_token_ids", None) or []
        index_of = getattr(model_output, "req_id_to_index", None) or {}
        snapshot = {}
        for internal in output.num_scheduled_tokens:
            request = scheduler.requests.get(internal)
            index = index_of.get(internal)
            snapshot[internal] = {
                "exists": request is not None,
                "finished": request is not None and bool(request.is_finished()),
                "stale": request is not None
                and int(getattr(request, "num_stale_output_tokens", 0)) > 0,
                "drop_stale": bool(getattr(request, "drop_stale_output", False)),
                # Copied now: stop handling trims the sampled list in place.
                "sampled": (
                    len(sampled[index])
                    if index is not None and index < len(sampled)
                    else 0
                ),
            }
        return snapshot

    def after_update(
        self,
        scheduler: Any,
        output: Any,
        before: dict[str, dict[str, Any]],
        *,
        result: Any = None,
        failed: bool = False,
    ) -> None:
        """Record the step's outcome; a failed update is recorded as unknown."""
        identity = getattr(output, ITERATION_ATTRIBUTE, None)
        if identity is None:
            return
        pending = self.pending.pop(identity[1], None)
        emitted = None if failed else _emitted(result)
        per_step = int(getattr(scheduler, "num_sampled_tokens_per_step", 1))
        members = [
            (
                _failed_member(name)
                if failed
                else self._completed_member(
                    scheduler, member, before.get(name, {}), emitted, per_step
                )
            )
            for name, member in (pending.members.items() if pending else ())
        ]
        fields: dict[str, Any] = {"iteration": identity[1], **stamp()}
        if failed:
            fields["update_failed"] = True
        fields["members"] = members
        self.writer.emit("completed", fields)

    def _completed_member(
        self,
        scheduler: Any,
        member: _Member,
        before: dict[str, Any],
        emitted: dict[str, tuple[int, str | None]] | None,
        per_step: int,
    ) -> dict[str, Any]:
        outcome = _outcome(before)
        sampled = int(before.get("sampled", 0))
        accepted = (
            max(sampled - per_step, 0)
            if member.drafts and (sampled or per_step == 0)
            else 0
        )
        stale = bool(before.get("stale"))
        computed_after = None
        retained = None
        finish = None
        if outcome == "kept":
            rejected = 0 if stale else member.drafts - accepted
            computed_after = member.computed_before + member.scheduled - rejected
            if emitted is not None:
                retained, finish = emitted.get(member.internal, (0, None))
            if member.internal in scheduler.requests:
                self.committed[member.internal] = computed_after
        return {
            "internal": member.internal,
            "outcome": outcome,
            "stale": stale,
            "sampled": sampled,
            "accepted_drafts": accepted,
            "retained": retained,
            "finish_reason": finish,
            "computed_after": computed_after,
        }

    # ------------------------------------------------------------ pauses

    def on_pause(self, before: Any, after: Any) -> None:
        """A call to ``set_pause_state``, stamped once it has returned."""
        self.writer.emit(
            "pause",
            {"from": _state_name(before), "to": _state_name(after), **stamp()},
        )

    # ------------------------------------------------------------ cache resets

    def reset_call(
        self, scheduler: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """A ``reset_prefix_cache`` call's settings and running requests, and
        its start, read before vLLM runs it."""
        named = dict(zip(("reset_running_requests", "reset_connector"), args))
        named.update(kwargs)
        return {
            "reset_running_requests": bool(named.get("reset_running_requests")),
            "reset_connector": bool(named.get("reset_connector")),
            "running": [str(request.request_id) for request in scheduler.running],
            **{f"start_{name}": value for name, value in stamp().items()},
        }

    def on_cache_reset(
        self, call: dict[str, Any] | None, succeeded: bool | None
    ) -> None:
        """The call's outcome: its return value, or null when it raised."""
        if call is None:
            return
        self.writer.emit(
            "cache_reset",
            {
                **call,
                "succeeded": succeeded,
                "raised": succeeded is None,
                **{f"end_{name}": value for name, value in stamp().items()},
            },
        )

    # ------------------------------------------------------------ exit

    def on_free(self, request: Any) -> None:
        internal = str(request.request_id)
        output_tokens = int(request.num_output_tokens)
        self.prompt_tokens.pop(internal, None)
        self.committed.pop(internal, None)
        status = getattr(request, "status", None)
        reason = (
            request.get_finished_reason()
            if hasattr(request, "get_finished_reason")
            else None
        )
        self.writer.emit(
            "terminal",
            {
                "internal": internal,
                "status": getattr(status, "name", None) or _optional_str(status),
                "finish_reason": _optional_str(reason),
                "output_tokens": output_tokens,
                **stamp(),
            },
        )


def _emitted(result: Any) -> dict[str, tuple[int, str | None]] | None:
    """Tokens and finish reason vLLM sent each request in this step's output.

    ``update_from_output`` returns ``{client: EngineCoreOutputs}``; counting the
    emitted tokens is right whatever vLLM's own counters do: stop trimming,
    a finished request already freed, or a streaming-input session reset.
    Any other shape is unknown, not zero.
    """
    if not isinstance(result, dict):
        return None
    emitted: dict[str, tuple[int, str | None]] = {}
    for outputs in result.values():
        for item in getattr(outputs, "outputs", None) or ():
            emitted[str(item.request_id)] = (
                len(item.new_token_ids or ()),
                # vLLM's FinishReason prints as "stop", "length" and so on.
                _optional_str(getattr(item, "finish_reason", None)),
            )
    return emitted


def _failed_member(internal: str) -> dict[str, Any]:
    return {
        "internal": internal,
        "outcome": "unknown",
        "stale": None,
        "sampled": None,
        "accepted_drafts": None,
        "retained": None,
        "finish_reason": None,
        "computed_after": None,
    }


def _outcome(before: dict[str, Any]) -> str:
    if not before:
        return "unknown"
    if not before["exists"] or before["finished"]:
        return "discarded_finished"
    if before["stale"] and before["drop_stale"]:
        return "dropped_stale"
    return "kept"


def _state_name(state: Any) -> str | None:
    """vLLM's PauseState by name: UNPAUSED, PAUSED_NEW or PAUSED_ALL."""
    if state is None:
        return None
    name = getattr(state, "name", None)
    return name if isinstance(name, str) else str(state)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = ["ITERATION_ATTRIBUTE", "EngineRecorder"]
