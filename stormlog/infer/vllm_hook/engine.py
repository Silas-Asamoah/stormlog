"""Engine-core snapshots: admissions, scheduled steps, their outputs, and exits.

Every value is a scalar copied at the moment it is observed; nothing keeps a
live vLLM object. Per-step context comes from the scheduler output's own fields
because ``Scheduler.schedule`` has already advanced each request's
``num_computed_tokens`` by the time it returns, and under async scheduling the
next step has been scheduled before this one's output is processed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .writer import EpochWriter

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
    prompt_tokens: dict[str, int] = field(default_factory=dict)
    committed: dict[str, int] = field(default_factory=dict)
    pending: dict[str, _Pending] = field(default_factory=dict)
    freed_output_tokens: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------ admission

    def on_admit(self, request: Any) -> None:
        self.writer.emit(
            "alias",
            {
                "internal": str(request.request_id),
                "external": _optional_str(getattr(request, "external_req_id", None)),
                **_stamp(),
            },
        )

    # ------------------------------------------------------------ schedule

    def on_schedule(self, scheduler: Any, output: Any, start: tuple[int, int]) -> None:
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
        end = _stamp()
        self.writer.emit(
            "scheduled",
            {
                "iteration": iteration,
                "start_wall_ns": start[0],
                "start_mono_ns": start[1],
                "end_wall_ns": end["wall_ns"],
                "end_mono_ns": end["mono_ns"],
                "total_tokens": int(output.total_num_scheduled_tokens),
                "zero_token": int(output.total_num_scheduled_tokens) == 0,
                "preempted": sorted(getattr(output, "preempted_req_ids", None) or ()),
                "members": [fields for _, fields in members],
            },
            size_hint=256 + 220 * len(members),
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
        prompt = self.prompt_tokens.get(internal)
        prefill = max(0, min(scheduled, prompt - computed_before)) if prompt else 0
        request = scheduler.requests.get(internal)
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
        }
        return member, fields

    # ------------------------------------------------------------ output

    def before_update(
        self, scheduler: Any, output: Any, model_output: Any
    ) -> dict[str, dict[str, Any]]:
        """Per member, what vLLM's own output loop is about to read."""
        self.freed_output_tokens.clear()
        sampled = getattr(model_output, "sampled_token_ids", None) or []
        index_of = getattr(model_output, "req_id_to_index", None) or {}
        snapshot = {}
        for internal in output.num_scheduled_tokens:
            request = scheduler.requests.get(internal)
            index = index_of.get(internal)
            snapshot[internal] = {
                "exists": request is not None,
                "finished": request is not None and bool(request.is_finished()),
                "output_before": (
                    int(request.num_output_tokens) if request is not None else None
                ),
                "stale": request is not None
                and int(getattr(request, "num_stale_output_tokens", 0)) > 0,
                "drop_stale": bool(getattr(request, "drop_stale_output", False)),
                "sampled": (
                    len(sampled[index])
                    if index is not None and index < len(sampled)
                    else 0
                ),
            }
        return snapshot

    def after_update(
        self, scheduler: Any, output: Any, before: dict[str, dict[str, Any]]
    ) -> None:
        identity = getattr(output, ITERATION_ATTRIBUTE, None)
        if identity is None:
            return
        pending = self.pending.pop(identity[1], None)
        per_step = int(getattr(scheduler, "num_sampled_tokens_per_step", 1))
        members = [
            self._completed_member(scheduler, member, before.get(name, {}), per_step)
            for name, member in (pending.members.items() if pending else ())
        ]
        self.writer.emit(
            "completed",
            {"iteration": identity[1], **_stamp(), "members": members},
            size_hint=128 + 160 * len(members),
        )

    def _completed_member(
        self,
        scheduler: Any,
        member: _Member,
        before: dict[str, Any],
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
        if outcome == "kept":
            rejected = 0 if stale else member.drafts - accepted
            computed_after = member.computed_before + member.scheduled - rejected
            self.committed[member.internal] = computed_after
        return {
            "internal": member.internal,
            "outcome": outcome,
            "stale": stale,
            "sampled": sampled,
            "accepted_drafts": accepted,
            "retained": self._retained(scheduler, member.internal, before),
            "computed_after": computed_after,
        }

    def _retained(
        self, scheduler: Any, internal: str, before: dict[str, Any]
    ) -> int | None:
        output_before = before.get("output_before")
        if output_before is None:
            return None
        request = scheduler.requests.get(internal)
        if request is not None:
            return int(request.num_output_tokens) - int(output_before)
        freed = self.freed_output_tokens.get(internal)
        return None if freed is None else freed - int(output_before)

    # ------------------------------------------------------------ exit

    def on_free(self, request: Any) -> None:
        internal = str(request.request_id)
        output_tokens = int(request.num_output_tokens)
        self.freed_output_tokens[internal] = output_tokens
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
                **_stamp(),
            },
        )


def _outcome(before: dict[str, Any]) -> str:
    if not before:
        return "unknown"
    if not before["exists"] or before["finished"]:
        return "discarded_finished"
    if before["stale"] and before["drop_stale"]:
        return "dropped_stale"
    return "kept"


def _stamp() -> dict[str, int]:
    return {"wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns()}


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = ["ITERATION_ATTRIBUTE", "EngineRecorder"]
