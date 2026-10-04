"""The experiment runner: every run of a plan, on a fresh server, recorded.

``run_plan`` runs a plan's blocks in the plan's order. Before a block's
first run it runs the block's preludes. Each run then:

1. starts the arm's server, waits for ``/health``, and checks the server's
   process tree holds only vLLM's own processes;
2. describes the server (before);
3. starts the arm's treatments and waits for each to be ready;
4. runs the workload steps in order;
5. checks each treatment stayed up for the whole workload, then stops it;
6. describes the server again (after) and attaches that, and the probe it
   took before measuring, to each artifact;
7. stops the server's whole process group and checks nothing is left;
8. checks the promised artifacts and their labels, appends the run's state
   to each (``infer.run_state``), writes ``SHA256SUMS`` and renames
   ``runs/<label>.partial`` to ``runs/<label>``.

Every run ends in one state, recorded in ``index.jsonl`` with its reasons:

- ``completed``;
- ``outcome_failure``: the server died, a step failed or timed out, a
  treatment was not ready or stopped early. Outcomes are data: a
  comparison counts them, and a retry never replaces them;
- ``protocol_failure``: the measurement failed before the treatment was
  applied (a server that never became healthy, unexpected processes,
  affinity not applied), or the run's records cannot be trusted (labels,
  a cleanup that left processes). Protocol failures are excluded and may
  be retried.

When both apply, the outcome wins unless the protocol fault came before the
first workload step started.

A cleanup that left processes, a run's or a prelude's, stops the
experiment: nothing more starts beside them. Every planned run after it is
indexed ``not_run``, and a resume runs them once the host is clean: it
refuses while a survivor, known by its PID and start time, still runs.

A run whose runner was killed (a preempted box, an operator's abort) leaves
its attempt in ``runs/<label>.partial`` with no state. A resume refuses to
go on until each such attempt is explained: by an external cause with its
evidence (``external_causes``), which makes it a ``protocol_failure`` that
may be retried, or by marking it an outcome (``interrupted_as_outcome``),
an ``outcome_failure`` (``runner_interrupted``) that is kept. It then
finishes the attempt in place and indexes it. An attempt killed after it
recorded its state keeps that state, and no cause can be named for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .describe_server import (
    DescribeOptions,
    describe_server,
    load_description,
    write_description,
)
from .errors import InferInputError, InferUsageError
from .experiment_plan import (
    Arm,
    ExperimentPlan,
    Prelude,
    RunOrder,
    Step,
    Treatment,
    block_seed,
    expand,
    plan_order,
)
from .experiment_process import (
    Launched,
    journaled,
    launch,
    listens,
    parse_cpu_list,
    process_key,
    remembered_tree,
    run_step,
    still_there,
    stop,
    stop_journaled,
    unexpected_roles,
    verify_cleanup,
    wait_for_file,
)
from .host_clock import host_boot_id
from .manifest import BEFORE as MANIFEST_BEFORE
from .manifest import (
    attach_manifest,
    before_refusals,
    description_record,
    manifests,
    model_identity_record,
)
from .model_identity import VerifiedModel, changed_files, prepare_model
from .observers import TREATMENTS_EVENT
from .run_summary import RUN_STATE_EVENT
from .sanitize import sanitize_bundle
from .server_collector import NvmlUnavailableError
from .server_probe import AUTO, BEFORE, SERVER_INFO, probe_server

COMPLETED = "completed"
OUTCOME_FAILURE = "outcome_failure"
PROTOCOL_FAILURE = "protocol_failure"
# A planned run the runner never started: an earlier cleanup left processes.
NOT_RUN = "not_run"
NEVER_HEALTHY = "server_never_healthy"
INDEX = "index.jsonl"
# Written when an attempt starts: the slot it runs in.
ATTEMPT = "attempt.json"
# One line per process an attempt or a prelude launches, as it starts.
LAUNCHES = "launches.ndjson"
# What stops the experiment: a process that may still run, or a server port
# something else holds.
STOP_REASONS = ("cleanup_unverified", "server_port_in_use")
# What may set aside an interrupted attempt; anything else is an outcome.
EXTERNAL_REASONS = ("spot_preemption", "operator_abort", "infra_fault")
SUMS = "SHA256SUMS"
HEALTH_POLL_SECONDS = 1.0
ROLE_WAIT_SECONDS = 10.0

Events = Callable[[dict[str, Any]], None]


@dataclass
class RunRecord:
    """One attempt at one run, as ``index.jsonl`` records it."""

    label: str
    arm: str
    block: int
    position_planned: int
    position_actual: int
    attempt: int
    run_dir: str
    state: str = COMPLETED
    reasons: list[str] = field(default_factory=list)
    before_treatment: list[str] = field(default_factory=list)
    processes: list[dict[str, Any]] = field(default_factory=list)
    started_at_ns: int = 0
    ended_at_ns: int | None = None
    cleanup: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    # A retried attempt runs later than its planned slot.
    order_broken: bool = False
    # The runner was killed during it; finished on resume.
    interrupted: bool = False
    external_cause: dict[str, str] | None = None
    # Whether its server came up; None when none was launched.
    server_healthy: bool | None = None
    # For a server that never came up: which rule made it an outcome or not.
    decided_by: str | None = None

    def outcome(self, reason: str) -> None:
        self.reasons.append(reason)
        self.state = OUTCOME_FAILURE

    def protocol(self, reason: str, *, before_treatment: bool) -> None:
        self.reasons.append(reason)
        if before_treatment:
            self.before_treatment.append(reason)
        if self.state == COMPLETED:
            self.state = PROTOCOL_FAILURE

    def settle(self) -> None:
        """Precedence: a protocol fault before the treatment outranks an outcome."""
        if self.before_treatment:
            self.state = PROTOCOL_FAILURE

    def to_record(self) -> dict[str, Any]:
        return {"type": "run", **self.__dict__}


@dataclass(frozen=True)
class ExternalCause:
    """Why an interrupted attempt does not count: one of three causes, with
    the evidence an operator or a driver can show for it."""

    reason: str
    evidence: str

    def __post_init__(self) -> None:
        if self.reason not in EXTERNAL_REASONS:
            raise InferUsageError(
                f"external cause {self.reason!r} is not spot_preemption, "
                "operator_abort or infra_fault"
            )
        if not self.evidence.strip():
            raise InferUsageError(f"external cause {self.reason} needs evidence")

    def to_record(self) -> dict[str, str]:
        return {"reason": self.reason, "evidence": self.evidence}


@dataclass(frozen=True)
class Environment:
    """What a run's commands get beyond the plan: interpreter and secrets."""

    python: str = sys.executable
    secrets: Mapping[str, str] = field(default_factory=dict)
    model: VerifiedModel | None = None


def run_plan(
    plan: ExperimentPlan,
    output_dir: Path,
    *,
    resume: bool = False,
    retry_incomplete: bool = False,
    environment: Environment | None = None,
    on_event: Events | None = None,
    external_causes: Mapping[str, ExternalCause] | None = None,
    interrupted_as_outcome: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Run every block of the plan; return the index entries written.

    Every attempt the runner was killed in must be named, by label, in
    ``external_causes`` (why it does not count) or ``interrupted_as_outcome``
    (it counts against its arm); a resume refuses until each is.
    """
    causes = dict(external_causes or {})
    _check_interrupted(output_dir, causes, set(interrupted_as_outcome), resume)
    if resume:
        _stop_left_launches(output_dir)
        _check_nothing_left(output_dir)
    env = environment or Environment(secrets=_secrets(plan))
    order = plan_order(plan)
    _prepare(plan, order, output_dir, resume)
    model = _verified_model(plan)
    if model is not None:
        env = replace(env, model=model)
    written: list[dict[str, Any]] = []
    for block, arms in enumerate(order.blocks):
        records = _run_block(
            plan,
            block,
            arms,
            output_dir,
            env,
            _Resume(resume, retry_incomplete, causes),
            on_event,
        )
        written += records
        left = [r for r in records if _left_running(r)]
        if left:
            # Nothing more starts beside a process that would not stop.
            written += _not_run(plan, order, left[0], output_dir, on_event)
            break
    report = sanitize_bundle(output_dir, env.secrets.values())
    (output_dir / "sanitizer.json").write_text(json.dumps(report, indent=2) + "\n")
    return written


@dataclass(frozen=True)
class _Resume:
    resume: bool
    retry: bool
    causes: Mapping[str, ExternalCause]


def _check_interrupted(
    output: Path,
    causes: Mapping[str, ExternalCause],
    outcomes: set[str],
    resume: bool,
) -> None:
    """Every attempt the runner was killed in is explained one way, and only
    those are.

    A finished attempt keeps the state it recorded, so an outcome failure
    can never be turned into a set-aside; and an interruption no one
    explained is neither set aside nor counted by default, since a
    forgotten flag after a preemption would decide it.
    """
    named = set(causes) | outcomes
    if named and not resume:
        raise InferUsageError(
            "external causes and interrupted outcomes are given only on resume"
        )
    if not resume:
        return
    both = sorted(set(causes) & outcomes)
    if both:
        raise InferUsageError(
            f"{', '.join(both)}: both an external cause and an outcome"
        )
    interrupted = _interrupted_labels(output / "runs")
    unknown = sorted(named - interrupted)
    if unknown:
        raise InferUsageError(
            f"{', '.join(unknown)} is not an interrupted attempt: a finished "
            "attempt keeps the state it recorded"
        )
    unexplained = sorted(interrupted - named)
    if unexplained:
        raise InferUsageError(
            f"{', '.join(unexplained)} was interrupted: give its external cause "
            "(external_causes, --external-cause) or mark it an outcome "
            "(interrupted_as_outcome, --interrupted-as-outcome)"
        )


def _check_nothing_left(output: Path) -> None:
    """Refuse to resume while a process an earlier cleanup left still runs."""
    for where, cleanup in _cleanups(output):
        if cleanup.get("verified") is not False:
            continue
        # Survivors, and processes it could not judge that may be the launch's.
        left = [*cleanup.get("survivors", []), *_blind_of(cleanup)]
        running = [s["pid"] for s in left if still_there(s)]
        if running:
            pids = ", ".join(str(pid) for pid in running)
            raise InferUsageError(
                f"{where}: its cleanup left {pids} running; stop it, then resume"
            )


def _stop_left_launches(output: Path) -> None:
    """Stop what a runner that was killed left running, or refuse to resume.

    Every launch of an unfinished attempt or a prelude is in its journal by
    PID, start time and mark. One whose leader is still that process has its
    group stopped; then its group, session and mark are verified gone.
    """
    journals = [
        *sorted((output / "runs").glob(f"*.partial/{LAUNCHES}")),
        *sorted((output / "preludes").glob(f"*/{LAUNCHES}")),
    ]
    for journal in journals:
        for entry in journaled(journal):
            cleanup = stop_journaled(entry)
            if not cleanup.verified:
                left = [s["pid"] for s in [*cleanup.survivors, *cleanup.blind]]
                raise InferUsageError(
                    f"{journal.parent.name} {entry.get('name')}: the runner was "
                    f"stopped and left {left} running; stop them, then resume"
                )


def _blind_of(cleanup: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    search = cleanup.get("mark_search") or {}
    return [item for item in search.get("blind", []) if isinstance(item, Mapping)]


def _cleanups(output: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Every cleanup the experiment recorded: each run's server and
    treatments, and each prelude's server."""
    for path in sorted((output / "runs").glob("*/run.json")):
        record = json.loads(path.read_text())
        if record.get("cleanup"):
            yield path.parent.name, record["cleanup"]
        for process in record.get("processes", []):
            if process.get("cleanup"):
                yield f"{path.parent.name} {process['name']}", process["cleanup"]
    for path in sorted((output / "preludes").glob("*/cleanup.json")):
        yield f"prelude {path.parent.name}", json.loads(path.read_text())


def _interrupted_labels(runs: Path) -> set[str]:
    if not runs.is_dir():
        return set()
    return {
        path.name[: -len(".partial")]
        for path in runs.iterdir()
        if path.name.endswith(".partial")
        and path.is_dir()
        and _recorded_state(path) is None
    }


def _verified_model(plan: ExperimentPlan) -> VerifiedModel | None:
    """Fix the weights once, before any server starts, if the plan says how."""
    spec = plan.server.model or {}
    return prepare_model(spec) if spec.get("route") else None


def _secrets(plan: ExperimentPlan) -> dict[str, str]:
    missing = [name for name in plan.secret_env if name not in os.environ]
    if missing:
        raise InferUsageError(f"secret_env not set: {', '.join(missing)}")
    return {name: os.environ[name] for name in plan.secret_env}


# ------------------------------------------------------------- the bundle


def _prepare(plan: ExperimentPlan, order: RunOrder, output: Path, resume: bool) -> None:
    """Write plan.json, prereg.json and order.json, or check them on resume."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    plan_path = output / "plan.json"
    record = {
        "plan_digest": plan.digest,
        "prereg_digest": plan.prereg_digest,
        "experiment_id": plan.experiment_id,
    }
    if plan_path.exists():
        recorded = json.loads(plan_path.read_text())
        if not resume:
            raise InferInputError(f"{output} already holds an experiment; resume it")
        if recorded.get("plan_digest") != plan.digest:
            raise InferInputError("the plan changed since this experiment started")
        if recorded.get("prereg_digest") != plan.prereg_digest:
            raise InferInputError(
                "the pre-registration changed since this experiment started"
            )
        return
    plan_path.write_text(json.dumps(record, indent=2) + "\n")
    if plan.prereg is not None:
        (output / "prereg.json").write_text(
            json.dumps(plan.prereg, indent=2, sort_keys=True) + "\n"
        )
    (output / "order.json").write_text(json.dumps(order.to_record(), indent=2) + "\n")


def _run_block(
    plan: ExperimentPlan,
    block: int,
    arms: list[str],
    output: Path,
    env: Environment,
    resuming: _Resume,
    on_event: Events | None,
) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    prelude_failures = [
        reason
        for prelude in plan.preludes
        for reason in _prelude(plan, prelude, block, output, env)
    ]
    held: list[dict[str, Any]] = []
    for position, arm in enumerate(arms):
        records = _attempts(
            plan,
            plan.arms[arm],
            block,
            position,
            output,
            env,
            resuming,
            prelude_failures,
        )
        for record in records:
            if NEVER_HEALTHY in record["reasons"]:
                held.append(record)  # until the block's control has run
            else:
                _publish(output, record, on_event)
        written += records
        if any(_left_running(r) for r in records):
            break
    for record in held:
        _decide_unhealthy(plan, record, output)
        _publish(output, record, on_event)
    return written


def _publish(output: Path, record: dict[str, Any], on_event: Events | None) -> None:
    _append_index(output, record)
    if on_event is not None:
        on_event(record)


def _decide_unhealthy(
    plan: ExperimentPlan, record: dict[str, Any], output: Path
) -> None:
    """A server that never became healthy is its arm's outcome when the arm's
    own launch differs from the control's and the control's server came up
    in the block; otherwise it stays a protocol failure. Its run.json is
    rewritten with the decision; such a run has no artifacts."""
    state, decided_by = _unhealthy_ruling(plan, record, output)
    record["decided_by"] = decided_by
    if state == OUTCOME_FAILURE:
        record["state"] = OUTCOME_FAILURE
        record["before_treatment"] = [
            reason for reason in record["before_treatment"] if reason != NEVER_HEALTHY
        ]
    run_dir = Path(record["run_dir"])
    (run_dir / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    _write_sums(run_dir)


def _unhealthy_ruling(
    plan: ExperimentPlan, record: Mapping[str, Any], output: Path
) -> tuple[str, str]:
    control = plan.control_arm
    if control is None:
        return PROTOCOL_FAILURE, "no_control_arm"
    if _same_launch(plan.arms[record["arm"]], plan.arms[control]):
        return PROTOCOL_FAILURE, "identical_launch"
    healthy = _control_health(plan, control, record["block"], output)
    if True in healthy:
        return OUTCOME_FAILURE, "arm_launch_differs"
    if False in healthy:
        return PROTOCOL_FAILURE, "control_also_unhealthy"
    return PROTOCOL_FAILURE, "control_not_launched"


def _same_launch(arm: Arm, control: Arm) -> bool:
    return arm.server_args == control.server_args and dict(arm.server_env) == dict(
        control.server_env
    )


def _control_health(
    plan: ExperimentPlan, control: str, block: int, output: Path
) -> list[bool | None]:
    """Whether each attempt of the control in the block, in this invocation
    or an earlier one, got its server up."""
    pattern = re.compile(
        rf"{re.escape(plan.experiment_id)}-b{block:02d}-p\d+-"
        rf"{re.escape(control)}-a\d+$"
    )
    found = []
    for path in sorted((output / "runs").iterdir()):
        if pattern.match(path.name) and (path / "run.json").is_file():
            found.append(
                json.loads((path / "run.json").read_text()).get("server_healthy")
            )
    return found


def _not_run(
    plan: ExperimentPlan,
    order: RunOrder,
    stopped: Mapping[str, Any],
    output: Path,
    on_event: Events | None,
) -> list[dict[str, Any]]:
    """Index every planned run after the one that stopped the experiment.

    A resume runs them, once the host is clean.
    """
    after = (stopped["block"], stopped["position_planned"])
    records = [
        {
            "type": NOT_RUN,
            "label": f"{plan.experiment_id}-b{block:02d}-p{position}-{arm}",
            "arm": arm,
            "block": block,
            "position_planned": position,
            "state": NOT_RUN,
            "reasons": [_stop_reason(stopped)],
            "stopped_after": stopped["label"],
        }
        for block, arms in enumerate(order.blocks)
        for position, arm in enumerate(arms)
        if (block, position) > after
    ]
    for record in records:
        _append_index(output, record)
        if on_event is not None:
            on_event(record)
    return records


def _attempts(
    plan: ExperimentPlan,
    arm: Arm,
    block: int,
    position: int,
    output: Path,
    env: Environment,
    resuming: _Resume,
    prelude_failures: list[str],
) -> list[dict[str, Any]]:
    """Any attempt the runner was killed in, finished; the run's attempt; and
    one more on a fresh server if its probe did not finish and its cleanup
    verified, so the fresh server never starts beside a survivor."""
    base = f"{plan.experiment_id}-b{block:02d}-p{position}-{arm.name}"
    interrupted = [
        _finish_leftover(leftover, arm, block, position, resuming.causes)
        for leftover in _leftovers(output / "runs", base)
    ]
    first = _attempt(
        plan,
        arm,
        block,
        position,
        output,
        env,
        resuming.resume,
        resuming.retry,
        prelude_failures,
    )
    if first is None:
        return interrupted
    if "probe_incomplete" not in first["reasons"] or _left_running(first):
        return [*interrupted, first]
    again = _attempt(
        plan, arm, block, position, output, env, False, True, prelude_failures
    )
    return [*interrupted, first] + ([again] if again is not None else [])


def _left_running(record: Mapping[str, Any]) -> bool:
    """Whether the experiment must stop after this attempt: a server,
    treatment, step or prelude whose cleanup did not verify, or a server
    port something else already held."""
    return _stop_reason(record) is not None


def _stop_reason(record: Mapping[str, Any]) -> str | None:
    for kind in STOP_REASONS:
        if any(kind in reason for reason in record["reasons"]):
            return kind
    return None


def _leftovers(runs: Path, base: str) -> list[Path]:
    """A run's attempts its runner was killed in, oldest first."""
    pattern = re.compile(re.escape(base) + r"-a(\d+)\.partial$")
    found = [
        (int(match.group(1)), path)
        for path in runs.iterdir()
        if (match := pattern.match(path.name)) is not None
    ]
    return [path for _, path in sorted(found)]


def _finish_leftover(
    leftover: Path,
    arm: Arm,
    block: int,
    position: int,
    causes: Mapping[str, ExternalCause],
) -> dict[str, Any]:
    recorded = _recorded_state(leftover)
    if recorded is None:
        return _finish_interrupted(leftover, arm, block, position, causes)
    return _finish_recorded(leftover, recorded, arm, block, position)


def _recorded_state(leftover: Path) -> dict[str, Any] | None:
    """The state an attempt recorded before its runner was killed, if any:
    its run.json, or else the run state it appended to its artifacts."""
    try:
        record = json.loads((leftover / "run.json").read_text())
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and record.get("state"):
        return record
    for path in _session_artifacts(leftover):
        states = [
            item
            for item in _artifact_records(path)
            if item.get("event_type") == RUN_STATE_EVENT
        ]
        if states:
            return states[-1]
    return None


def _finish_recorded(
    leftover: Path,
    recorded: Mapping[str, Any],
    arm: Arm,
    block: int,
    position: int,
) -> dict[str, Any]:
    """An attempt killed after it recorded its state: kept in that state,
    which no cause named on resume can change."""
    label = leftover.name[: -len(".partial")]
    final = leftover.with_name(label)
    if recorded.get("type") == "run":
        record = dict(recorded)
    else:
        attempt = int(label.rsplit("-a", 1)[1])
        rebuilt = RunRecord(
            label=label,
            arm=arm.name,
            block=block,
            position_planned=position,
            position_actual=_slot(leftover, position),
            attempt=attempt,
            run_dir=str(final),
            state=recorded["state"],
            reasons=list(recorded.get("reasons", [])),
            before_treatment=list(recorded.get("before_treatment", [])),
            order_broken=attempt > 1,
            external_cause=recorded.get("external_cause"),
        )
        rebuilt.notes.append("the runner was stopped after this attempt's state")
        record = rebuilt.to_record()
        (leftover / "run.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    _write_sums(leftover)
    leftover.rename(final)
    return record


def _finish_interrupted(
    leftover: Path,
    arm: Arm,
    block: int,
    position: int,
    causes: Mapping[str, ExternalCause],
) -> dict[str, Any]:
    """An attempt the runner was killed in, finished in place and kept.

    It is an outcome failure, since what stopped the runner may be the
    treatment, unless an external cause is given for it.
    """
    label = leftover.name[: -len(".partial")]
    attempt = int(label.rsplit("-a", 1)[1])
    final = leftover.with_name(label)
    record = RunRecord(
        label=label,
        arm=arm.name,
        block=block,
        position_planned=position,
        position_actual=_slot(leftover, position),
        attempt=attempt,
        run_dir=str(final),
        order_broken=attempt > 1,
        interrupted=True,
    )
    record.notes.append("the runner was stopped during this attempt")
    cause = causes.get(label)
    if cause is None:
        record.outcome("runner_interrupted")
    else:
        record.protocol(cause.reason, before_treatment=False)
        record.external_cause = cause.to_record()
    _append_run_state(_session_artifacts(leftover), record, label)
    (leftover / "run.json").write_text(
        json.dumps(record.to_record(), indent=2, sort_keys=True) + "\n"
    )
    _write_sums(leftover)
    leftover.rename(final)
    return record.to_record()


def _started_in_block(runs: Path, experiment_id: str, block: int) -> int:
    """How many attempts the block started, in this invocation or earlier."""
    prefix = f"{experiment_id}-b{block:02d}-p"
    return sum(1 for path in runs.iterdir() if path.name.startswith(prefix))


def _slot(leftover: Path, planned: int) -> int:
    """The slot an interrupted attempt ran in, as it recorded at its start."""
    try:
        return int(json.loads((leftover / ATTEMPT).read_text())["position_actual"])
    except (OSError, ValueError, KeyError, TypeError):
        return planned


def _session_artifacts(run_dir: Path) -> list[Path]:
    """The infer artifacts in a run's directory: files that open with a session."""
    found = []
    for path in sorted(run_dir.rglob("*.jsonl")):
        try:
            with path.open() as handle:
                first = json.loads(handle.readline())
        except (OSError, ValueError):
            continue
        if isinstance(first, dict) and first.get("event_type") == "infer.session":
            found.append(path)
    return found


def _append_run_state(paths: list[Path], record: RunRecord, run_id: str) -> None:
    """Tell each artifact how the run ended, so a comparison need not trust the
    session alone: an outcome is compared and counted against its arm, and a
    protocol failure is the external cause that sets aside its block."""
    state: dict[str, Any] = {
        "event_type": RUN_STATE_EVENT,
        "run_id": run_id,
        "state": record.state,
        "reasons": list(record.reasons),
        "before_treatment": list(record.before_treatment),
    }
    if record.external_cause is not None:
        state["external_cause"] = dict(record.external_cause)
    for path in paths:
        with path.open("a") as handle:
            handle.write(json.dumps(state, sort_keys=True) + "\n")


def _attempt(
    plan: ExperimentPlan,
    arm: Arm,
    block: int,
    position: int,
    output: Path,
    env: Environment,
    resume: bool,
    retry: bool,
    prelude_failures: list[str],
) -> dict[str, Any] | None:
    """Run one planned run, unless a kept attempt already stands for it.

    On resume a run is skipped when its last attempt verifies and ended
    ``completed`` or ``outcome_failure``: an outcome is never retried away.
    A ``protocol_failure`` is retried only with ``retry``.
    """
    runs = output / "runs"
    base = f"{plan.experiment_id}-b{block:02d}-p{position}-{arm.name}"
    finished = _finished_attempts(runs, base)
    if resume and finished and not _retry_wanted(finished[-1], retry):
        return None
    run = _Run(plan, arm, block, position, len(finished) + 1, runs, env)
    if prelude_failures:
        run.record.protocol(
            "prelude_failed:" + ",".join(prelude_failures), before_treatment=True
        )
        return run.finish()
    overlap = _affinity_overlap(plan, arm)
    if overlap:
        run.record.protocol(f"affinity_overlap:{overlap}", before_treatment=True)
        return run.finish()
    return run.execute()


def _finished_attempts(runs: Path, base: str) -> list[Path]:
    """The run's finished attempts, oldest first."""
    pattern = re.compile(re.escape(base) + r"-a(\d+)$")
    found = [
        (int(match.group(1)), path)
        for path in runs.iterdir()
        if (match := pattern.match(path.name)) is not None
    ]
    return [path for _, path in sorted(found)]


def _retry_wanted(last: Path, retry: bool) -> bool:
    """Whether a run's last attempt should be followed by another."""
    if not _sums_verify(last):
        return retry
    state = json.loads((last / "run.json").read_text()).get("state")
    return state == PROTOCOL_FAILURE and retry


def _affinity_overlap(plan: ExperimentPlan, arm: Arm) -> str | None:
    """The CPUs a declared-disjoint server shares with a command, if any."""
    if not plan.affinity_disjoint or not plan.server.cpu_affinity:
        return None
    server = parse_cpu_list(plan.server.cpu_affinity)
    others = [s.cpu_affinity for s in arm.workload] + [
        t.cpu_affinity for t in arm.treatments
    ]
    shared = set().union(*(server & parse_cpu_list(c) for c in others if c))
    return ",".join(str(cpu) for cpu in sorted(shared)) or None


def _append_index(output: Path, record: Mapping[str, Any]) -> None:
    with (output / INDEX).open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _sums_verify(run_dir: Path) -> bool:
    sums = run_dir / SUMS
    if not sums.exists():
        return False
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition("  ")
        path = run_dir / name
        if not path.exists() or _sha256(path) != digest:
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ a run


class _Run:
    """One attempt: its directory, its processes and its record."""

    def __init__(
        self,
        plan: ExperimentPlan,
        arm: Arm,
        block: int,
        position: int,
        attempt: int,
        runs: Path,
        env: Environment,
    ) -> None:
        self.plan, self.arm, self.env = plan, arm, env
        self.label = (
            f"{plan.experiment_id}-b{block:02d}-p{position}-{arm.name}-a{attempt}"
        )
        self.final_dir = runs / self.label
        self.dir = runs / f"{self.label}.partial"
        # The slot it runs in: every attempt the block started before it.
        started = _started_in_block(runs, plan.experiment_id, block)
        self.dir.mkdir(parents=True)
        self.record = RunRecord(
            label=self.label,
            arm=arm.name,
            block=block,
            position_planned=position,
            position_actual=started,
            attempt=attempt,
            run_dir=str(self.final_dir),
            started_at_ns=time.time_ns(),
            order_broken=attempt > 1,
        )
        # Kept should the runner be killed before the attempt ends.
        (self.dir / ATTEMPT).write_text(
            json.dumps({"position_actual": started}, indent=2) + "\n"
        )
        self.values: dict[str, Any] = {
            "run_dir": str(self.dir),
            "run_id": self.label,
            "label": self.label,
            "experiment_id": plan.experiment_id,
            "arm": arm.name,
            "block": block,
            "position": position,
            "attempt": attempt,
            "block_seed": block_seed(plan, block),
            "base_url": plan.server.base_url,
            "python": env.python,
            "model": _model_name(plan, env),
        }
        self.commands: list[str] = []
        self.server: Launched | None = None
        self.server_key: tuple[int, int] | None = None
        self.treatments: list[tuple[Treatment, Launched]] = []
        self.steps_started = False

    # The run, in order -------------------------------------------------

    def execute(self) -> dict[str, Any]:
        try:
            if self._start_server():
                self._describe("before")
                if self._start_treatments():
                    self._run_steps()
                self._stop_treatments()
                self._describe("after")
                self._attach_after()
        finally:
            self._stop_server()
        self._check_artifacts()
        self._check_model()
        return self.finish()

    def _check_model(self) -> None:
        """The verified weights must not have moved during the run."""
        model = self.env.model
        if model is None:
            return
        (self.dir / "model_identity.json").write_text(
            json.dumps(model.record(), indent=2, sort_keys=True) + "\n"
        )
        changed = changed_files(model)
        if changed:
            self.record.protocol("model_changed", before_treatment=False)
            self.record.notes.append(f"model files changed: {changed[:5]}")

    def finish(self) -> dict[str, Any]:
        self.record.ended_at_ns = time.time_ns()
        self.record.settle()
        self._record_state()
        self._write_commands()
        (self.dir / "run.json").write_text(
            json.dumps(self.record.to_record(), indent=2, sort_keys=True) + "\n"
        )
        _write_sums(self.dir)
        self.dir.rename(self.final_dir)
        return self.record.to_record()

    def _record_state(self) -> None:
        _append_run_state(self._infer_artifacts(), self.record, self.label)

    # Server -------------------------------------------------------------

    def _start_server(self) -> bool:
        server = self.plan.server
        if _port_in_use(server.base_url):
            # Another server, perhaps one a killed runner left: launching
            # would measure it.
            self.record.protocol("server_port_in_use", before_treatment=True)
            return False
        command, env = _server_launch(self.plan, self.arm, self.values, self.env)
        self.server = self._launch("server", command, env, server.cpu_affinity)
        self.values["server_pid"] = self.server.pid
        # Read now: the start ticks name this process, and no restart.
        self.server_key = process_key(self.server.pid)
        if self.server.affinity_applied is False:
            self.record.protocol("affinity_not_applied:server", before_treatment=True)
            return False
        healthy = _wait_healthy(server.base_url, self.server, server.start_timeout_s)
        self.record.server_healthy = healthy
        if not healthy:
            # Decided once the block is done (_decide_unhealthy).
            self.record.protocol(NEVER_HEALTHY, before_treatment=True)
            return False
        if not self._probe():
            return False
        return self._only_vllm_processes()

    def _probe(self) -> bool:
        """Ask the server about itself once, before anything is measured.

        A /server_info that does not answer in time may leave vLLM's
        environment collector running; the run stops here, and the plan
        runs it again on a fresh server.
        """
        probe = probe_server(self.plan.server.base_url, mode=AUTO, phase=BEFORE)
        (self.dir / "server-probe.json").write_text(
            json.dumps(probe.to_record(session_id=self.label), indent=2) + "\n"
        )
        if probe.incomplete:
            self.record.protocol("probe_incomplete", before_treatment=True)
            return False
        return True

    def _only_vllm_processes(self) -> bool:
        assert self.server is not None
        deadline = time.monotonic() + ROLE_WAIT_SECONDS
        while True:
            extra = unexpected_roles(self.server.pid)
            if not extra:
                return True
            if time.monotonic() >= deadline:
                self.record.protocol(
                    "unexpected_server_children", before_treatment=True
                )
                self.record.notes.append(f"unexpected server processes: {extra}")
                return False
            time.sleep(0.5)

    def _stop_server(self) -> None:
        if self.server is None:
            return
        remembered = remembered_tree(self.server.pid)
        if self.server.poll() is not None:
            self.record.outcome(f"server_exited:{self.server.exit_code}")
        stop(self.server, signals=(2, 15), timeout_s=self.plan.server.stop_timeout_s)
        cleanup = verify_cleanup(
            self.server.pid,
            remembered,
            mark=self.server.mark,
            since=self.server.identity,
        )
        self.record.cleanup = cleanup.to_record()
        if not cleanup.verified:
            self.record.protocol("collector_cleanup_unverified", before_treatment=False)
        self.record.processes.append(self.server.to_record())

    # Description ---------------------------------------------------------

    def _describe(self, phase: str) -> None:
        if not self.plan.describe.get(phase, True) or self.server is None:
            return
        log = (
            self.server.log_path if self.plan.describe.get("server_log", True) else None
        )
        model = self.env.model
        options = DescribeOptions(
            pid=self.server.pid,
            run_id=self.label,
            server_log=log,
            model_identity=model.record() if model is not None else None,
        )
        try:
            try:
                document = describe_server(options)
            except NvmlUnavailableError:
                document = describe_server(_without_gpu(options))
                self.record.notes.append(f"describe {phase}: NVML unavailable")
        except (InferUsageError, InferInputError, OSError) as exc:
            self.record.notes.append(f"describe {phase}: {exc}")
            return
        write_description(document, self.dir / f"describe-{phase}.json")

    def _attach_after(self) -> None:
        self._record_treatments()
        self._attach_before()
        self._attach_probe()
        self._record_model_identity()
        after = self.dir / "describe-after.json"
        if not after.exists():
            return
        for path in self._infer_artifacts():
            try:
                attach_manifest(path, after)
            except InferInputError as exc:
                self.record.notes.append(
                    f"after description not attached to {path.name}: {exc}"
                )

    def _attach_before(self) -> None:
        """Give each artifact the runner's before description, unless it has it.

        A workload step need not pass ``--describe-server``: the runner took
        the description, and the artifact needs it for its after
        description, and for the weights the runner verified to bind.
        """
        path = self.dir / "describe-before.json"
        if not path.exists():
            return
        description = load_description(path)
        refusals = before_refusals(description, run_id=self.label)
        if refusals:
            self.record.notes.append(f"before description: {'; '.join(refusals)}")
            return
        for artifact in self._infer_artifacts():
            records = _artifact_records(artifact)
            if manifests(records)[MANIFEST_BEFORE]:
                continue
            record = description_record(
                description,
                role=MANIFEST_BEFORE,
                session_id=_artifact_session(records),
                run_id=self.label,
            )
            with artifact.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _attach_probe(self) -> None:
        """Give each artifact the probe taken before measuring, unless it has one.

        The workload probes only the basic routes, so that no collector runs
        beside it; ``/server_info``, and the configuration a comparison
        needs, come from this probe.
        """
        path = self.dir / "server-probe.json"
        if not path.exists():
            return
        probe = json.loads(path.read_text())
        for artifact in self._infer_artifacts():
            records = _artifact_records(artifact)
            if any(_answered_server_info(r) for r in records):
                continue
            record = {
                **probe,
                "session_id": _artifact_session(records),
                "run_id": self.label,
                "taken_by": "experiment_runner",
            }
            with artifact.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _record_model_identity(self) -> None:
        """Bind the weights verified before launch to the server launched.

        The record names the server by boot, PID and start ticks; a
        comparison counts the weights as observed only for the server the
        run's before description shows.
        """
        model = self.env.model
        if model is None:
            return
        if self.server_key is None:
            self.record.notes.append(
                "model identity not bound: the server's start time was not read"
            )
            return
        pid, start_ticks = self.server_key
        record = model_identity_record(
            model.record(),
            session_id=self.label,
            run_id=self.label,
            server={"pid": pid, "start_ticks": start_ticks},
            boot_id=host_boot_id(),
        )
        for path in self._infer_artifacts():
            with path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _record_treatments(self) -> None:
        """Tell each artifact which treatments ran beside it, and how they held up.

        A treatment is an observer of the run, so a comparison's mode
        contract sees it; the record says whether it was ready and stayed up.
        """
        if not self.treatments:
            return
        reasons = set(self.record.reasons)
        entries = [
            {
                "name": treatment.name,
                # The plan's template: the same in every run of the arm.
                "command_sha256": hashlib.sha256(
                    "\0".join(treatment.command).encode()
                ).hexdigest(),
                "cpu_affinity": treatment.cpu_affinity,
                "ready": f"treatment_not_ready:{treatment.name}" not in reasons,
                "healthy": not any(
                    reason.split(":")[1:2] == [treatment.name]
                    for reason in reasons
                    if reason.startswith("treatment_")
                ),
                "exit_code": launched.exit_code,
            }
            for treatment, launched in self.treatments
        ]
        for path in self._infer_artifacts():
            record = {
                "event_type": TREATMENTS_EVENT,
                "run_id": self.label,
                "treatments": entries,
            }
            with path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    # Treatments -------------------------------------------------------------

    def _start_treatments(self) -> bool:
        for treatment in self.arm.treatments:
            command = [expand(part, self.values) for part in treatment.command]
            launched = self._launch(
                f"treatment:{treatment.name}",
                command,
                treatment.env,
                treatment.cpu_affinity,
            )
            self.treatments.append((treatment, launched))
            ready = (
                expand(treatment.ready_file, self.values)
                if treatment.ready_file
                else None
            )
            if ready and not wait_for_file(
                Path(ready), treatment.ready_timeout_s, launched
            ):
                self.record.outcome(f"treatment_not_ready:{treatment.name}")
                return False
        return True

    def _stop_treatments(self) -> None:
        for treatment, launched in self.treatments:
            name = treatment.name
            if launched.poll() is not None:
                # Gone before the workload ended, whatever its exit code.
                self.record.outcome(f"treatment_unhealthy:{name}")
            code = stop(
                launched,
                signals=(treatment.stop_signal,),
                timeout_s=treatment.stop_timeout_s,
            )
            unhealthy = f"treatment_unhealthy:{name}" in self.record.reasons
            if code not in treatment.expect_exit and not unhealthy:
                self.record.outcome(f"treatment_failed:{name}:{code}")
            # Whatever it started must not run on into the next run.
            cleanup = verify_cleanup(
                launched.pid, mark=launched.mark, since=launched.identity
            )
            if not cleanup.verified:
                self.record.protocol(
                    f"treatment_cleanup_unverified:{name}", before_treatment=False
                )
            self.record.processes.append(
                {**launched.to_record(), "cleanup": cleanup.to_record()}
            )

    # Steps ----------------------------------------------------------------

    def _run_steps(self) -> None:
        self.steps_started = True
        for step in self.arm.workload:
            if not self._run_step(step):
                return
            if not self._treatments_alive():
                return

    def _run_step(self, step: Step) -> bool:
        command = [expand(part, self.values) for part in step.command]
        env = {k: expand(v, self.values) for k, v in step.env.items()}
        self.commands.append(_shell_line(env, command, self.env.secrets))
        launched, timed_out = run_step(
            f"step:{step.name}",
            command,
            env={**env, **self.env.secrets},
            cpu_affinity=step.cpu_affinity,
            timeout_s=step.timeout_s,
            log_path=self.dir / f"{step.name}.log",
            journal=self.dir / LAUNCHES,
        )
        self.record.processes.append(launched.to_record())
        if timed_out:
            self.record.outcome(f"step_timeout:{step.name}")
            return False
        if launched.exit_code not in step.expect_exit:
            self.record.outcome(f"step_failed:{step.name}:{launched.exit_code}")
            return False
        return True

    def _treatments_alive(self) -> bool:
        return all(launched.poll() is None for _, launched in self.treatments)

    # Artifacts --------------------------------------------------------------

    def _check_artifacts(self) -> None:
        if not self.steps_started:
            return  # nothing was measured, so nothing is missing
        for step in self.arm.workload:
            for template in step.artifacts:
                path = Path(expand(template, self.values))
                if not path.exists():
                    self.record.outcome(f"artifact_missing:{path.name}")
        for path in self._infer_artifacts():
            if not _labels_match(
                path, self.plan.experiment_id, self.arm.name, self.record.block
            ):
                self.record.protocol(
                    f"label_mismatch:{path.name}", before_treatment=False
                )

    def _infer_artifacts(self) -> list[Path]:
        found = []
        for step in self.arm.workload:
            for template in step.artifacts:
                path = Path(expand(template, self.values))
                if path.exists() and path.suffix == ".jsonl":
                    found.append(path)
        return found

    # Helpers ----------------------------------------------------------------

    def _launch(
        self, name: str, command: list[str], env: Mapping[str, str], cpus: str | None
    ) -> Launched:
        expanded_env = {k: expand(v, self.values) for k, v in env.items()}
        self.commands.append(_shell_line(expanded_env, command, self.env.secrets))
        return launch(
            name,
            command,
            env={**expanded_env, **self.env.secrets},
            cpu_affinity=cpus,
            log_path=self.dir / f"{name.replace(':', '-')}.log",
            journal=self.dir / LAUNCHES,
        )

    def _write_commands(self) -> None:
        lines = ["#!/bin/sh", f"# {self.label}", *self.commands]
        (self.dir / "commands.sh").write_text("\n".join(lines) + "\n")


def _prelude(
    plan: ExperimentPlan, prelude: Prelude, block: int, output: Path, env: Environment
) -> list[str]:
    """Run a block prelude, with an arm's server if it asks for one."""
    directory = output / "preludes" / f"b{block:02d}-{prelude.step.name}"
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        "run_dir": str(directory),
        "experiment_id": plan.experiment_id,
        "block": block,
        "block_seed": block_seed(plan, block),
        "base_url": plan.server.base_url,
        "python": env.python,
        "model": _model_name(plan, env),
    }
    server = None
    if prelude.server_arm is not None:
        if _port_in_use(plan.server.base_url):
            return [f"{prelude.step.name}:server_port_in_use"]
        arm = plan.arms[prelude.server_arm]
        command, server_env = _server_launch(plan, arm, values, env)
        server = launch(
            "server",
            command,
            env=server_env,
            cpu_affinity=plan.server.cpu_affinity,
            log_path=directory / "server.log",
            journal=directory / LAUNCHES,
        )
    try:
        ran = _prelude_ran(plan, prelude, server, values, directory, env)
    finally:
        left = server is not None and not _stop_prelude_server(plan, server, directory)
    name = prelude.step.name
    return ([] if ran else [name]) + ([f"{name}:cleanup_unverified"] if left else [])


def _server_launch(
    plan: ExperimentPlan, arm: Arm, values: Mapping[str, Any], env: Environment
) -> tuple[list[str], dict[str, str]]:
    """An arm's server command and environment, a run's or a prelude's."""
    server = plan.server
    command = [expand(part, values) for part in (*server.command, *arm.server_args)]
    server_env = {**server.env, **arm.server_env}
    if env.model is not None:
        # The weights verified before launch, and nothing else.
        command += list(env.model.server_args)
        server_env.update(env.model.env)
    return command, server_env


def _prelude_ran(
    plan: ExperimentPlan,
    prelude: Prelude,
    server: Launched | None,
    values: Mapping[str, Any],
    directory: Path,
    env: Environment,
) -> bool:
    """Whether the prelude's server came up and its step exited as expected."""
    if server is not None and not _wait_healthy(
        plan.server.base_url, server, plan.server.start_timeout_s
    ):
        return False
    launched, timed_out = run_step(
        prelude.step.name,
        [expand(p, values) for p in prelude.step.command],
        env={
            **{k: expand(v, values) for k, v in prelude.step.env.items()},
            **env.secrets,
        },
        cpu_affinity=prelude.step.cpu_affinity,
        timeout_s=prelude.step.timeout_s,
        log_path=directory / "step.log",
        journal=directory / LAUNCHES,
    )
    return not timed_out and launched.exit_code in prelude.step.expect_exit


def _stop_prelude_server(
    plan: ExperimentPlan, server: Launched, directory: Path
) -> bool:
    """Stop a prelude's server as a run's is stopped; whether nothing is left."""
    remembered = remembered_tree(server.pid)
    stop(server, signals=(2, 15), timeout_s=plan.server.stop_timeout_s)
    cleanup = verify_cleanup(
        server.pid, remembered, mark=server.mark, since=server.identity
    )
    (directory / "cleanup.json").write_text(
        json.dumps(cleanup.to_record(), indent=2) + "\n"
    )
    return cleanup.verified


def _model_name(plan: ExperimentPlan, env: Environment) -> str:
    if env.model is not None:
        return env.model.model
    return str((plan.server.model or {}).get("name", ""))


def _artifact_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _answered_server_info(record: Mapping[str, Any]) -> bool:
    """A before probe whose /server_info answered with a body."""
    if record.get("event_type") != "infer.server_probe":
        return False
    if record.get("phase") != BEFORE:
        return False
    answer = (record.get("answers") or {}).get(SERVER_INFO) or {}
    return bool(answer.get("body"))


def _artifact_session(records: list[dict[str, Any]]) -> str:
    return next(
        (str(r["session_id"]) for r in records if r.get("session_id")), "unknown"
    )


def _without_gpu(options: DescribeOptions) -> DescribeOptions:
    return replace(options, no_gpu=True)


def _wait_healthy(base_url: str, server: Launched, timeout_s: float) -> bool:
    """Whether ``/health`` answers 200 before the timeout from this launch:
    the server still runs, and listens on the port where that can be read,
    so another server's answer never counts."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    port = _port_of(base_url)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                answered = response.status == 200
        except (urllib.error.URLError, OSError):
            answered = False
        if (
            answered
            and server.poll() is None
            and listens(server.pid, port) is not False
        ):
            return True
        time.sleep(HEALTH_POLL_SECONDS)
    return False


def _port_of(base_url: str) -> int:
    parsed = urllib.parse.urlsplit(base_url)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def _port_in_use(base_url: str) -> bool:
    """Whether something already accepts connections on the server's port."""
    host = urllib.parse.urlsplit(base_url).hostname or "127.0.0.1"
    try:
        with socket.create_connection((host, _port_of(base_url)), timeout=1):
            return True
    except OSError:
        return False


def _labels_match(path: Path, experiment: str, arm: str, block: int) -> bool:
    """Whether an infer artifact says it is this run: experiment, arm, block."""
    try:
        with path.open() as handle:
            first = json.loads(handle.readline())
    except (OSError, ValueError):
        return False
    labels = (first.get("config") or {}).get("labels") or {}
    return (
        labels.get("experiment") == experiment
        and labels.get("arm") == arm
        and str(labels.get("block")) == str(block)
    )


def _shell_line(
    env: Mapping[str, str], command: list[str], secrets: Mapping[str, str]
) -> str:
    """A command as a shell line; secrets appear only as ``${NAME}``."""
    assignments = [f"{k}={shlex.quote(v)}" for k, v in env.items()]
    assignments += [f"{name}=${{{name}}}" for name in secrets]
    return " ".join([*assignments, shlex.join(command)])


def _write_sums(run_dir: Path) -> None:
    """SHA256SUMS of every file in the run, written last."""
    lines = [
        f"{_sha256(path)}  {path.relative_to(run_dir).as_posix()}"
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != SUMS
    ]
    (run_dir / SUMS).write_text("\n".join(lines) + "\n")


__all__ = [
    "COMPLETED",
    "EXTERNAL_REASONS",
    "INDEX",
    "OUTCOME_FAILURE",
    "PROTOCOL_FAILURE",
    "Environment",
    "ExternalCause",
    "RunRecord",
    "run_plan",
]
