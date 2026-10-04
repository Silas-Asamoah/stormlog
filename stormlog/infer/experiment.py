"""The experiment runner: every run of a plan, on a fresh server, recorded.

``run_plan`` runs a plan's blocks in the plan's order. Before a block's
first run it runs the block's preludes. Each run then:

1. starts the arm's server, waits for ``/health``, and checks the server's
   process tree holds only vLLM's own processes;
2. describes the server (before);
3. starts the arm's treatments and waits for each to be ready;
4. runs the workload steps in order;
5. checks each treatment stayed up for the whole workload, then stops it;
6. describes the server again (after) and attaches that to each artifact;
7. stops the server's whole process group and checks nothing is left;
8. checks the promised artifacts and their labels, writes ``SHA256SUMS``
   and renames ``runs/<label>.partial`` to ``runs/<label>``.

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
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
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
    launch,
    parse_cpu_list,
    process_key,
    remembered_tree,
    run_step,
    stop,
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
from .sanitize import sanitize_bundle
from .server_collector import NvmlUnavailableError
from .server_probe import AUTO, BEFORE, probe_server

COMPLETED = "completed"
OUTCOME_FAILURE = "outcome_failure"
PROTOCOL_FAILURE = "protocol_failure"
INDEX = "index.jsonl"
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
) -> list[dict[str, Any]]:
    """Run every block of the plan; return the index entries written."""
    env = environment or Environment(secrets=_secrets(plan))
    order = plan_order(plan)
    _prepare(plan, order, output_dir, resume)
    model = _verified_model(plan)
    if model is not None:
        env = replace(env, model=model)
    written: list[dict[str, Any]] = []
    for block, arms in enumerate(order.blocks):
        written += _run_block(
            plan, block, arms, output_dir, env, resume, retry_incomplete, on_event
        )
    report = sanitize_bundle(output_dir, env.secrets.values())
    (output_dir / "sanitizer.json").write_text(json.dumps(report, indent=2) + "\n")
    return written


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
    resume: bool,
    retry: bool,
    on_event: Events | None,
) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    prelude_failures = [
        reason
        for prelude in plan.preludes
        for reason in _prelude(plan, prelude, block, output, env)
    ]
    for position, arm in enumerate(arms):
        records = _attempts(
            plan,
            plan.arms[arm],
            block,
            position,
            output,
            env,
            resume,
            retry,
            prelude_failures,
        )
        for record in records:
            _append_index(output, record)
            if on_event is not None:
                on_event(record)
        written += records
        if any("collector_cleanup_unverified" in r["reasons"] for r in records):
            break  # never measure beside a process that would not stop
    return written


def _attempts(
    plan: ExperimentPlan,
    arm: Arm,
    block: int,
    position: int,
    output: Path,
    env: Environment,
    resume: bool,
    retry: bool,
    prelude_failures: list[str],
) -> list[dict[str, Any]]:
    """A run's attempt, and one more on a fresh server if its probe did not finish."""
    first = _attempt(
        plan, arm, block, position, output, env, resume, retry, prelude_failures
    )
    if first is None:
        return []
    if "probe_incomplete" not in first["reasons"]:
        return [first]
    again = _attempt(
        plan, arm, block, position, output, env, False, True, prelude_failures
    )
    return [first] + ([again] if again is not None else [])


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
    A ``protocol_failure`` is retried only with ``retry``. A leftover
    ``.partial`` directory is kept, renamed ``.abandoned``.
    """
    runs = output / "runs"
    base = f"{plan.experiment_id}-b{block:02d}-p{position}-{arm.name}"
    finished = _finished_attempts(runs, base)
    for leftover in runs.glob(f"{base}-a*.partial"):
        leftover.rename(
            leftover.with_name(leftover.name[: -len(".partial")] + ".abandoned")
        )
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
        self.dir.mkdir(parents=True)
        self.record = RunRecord(
            label=self.label,
            arm=arm.name,
            block=block,
            position_planned=position,
            position_actual=position,
            attempt=attempt,
            run_dir=str(self.final_dir),
            started_at_ns=time.time_ns(),
            order_broken=attempt > 1,
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
        self._write_commands()
        (self.dir / "run.json").write_text(
            json.dumps(self.record.to_record(), indent=2, sort_keys=True) + "\n"
        )
        _write_sums(self.dir)
        self.dir.rename(self.final_dir)
        return self.record.to_record()

    # Server -------------------------------------------------------------

    def _start_server(self) -> bool:
        server, model = self.plan.server, self.env.model
        command = [
            expand(part, self.values)
            for part in (*server.command, *self.arm.server_args)
        ]
        env = {**server.env, **self.arm.server_env}
        if model is not None:
            # The weights verified before launch, and nothing else.
            command += list(model.server_args)
            env.update(model.env)
        self.server = self._launch("server", command, env, server.cpu_affinity)
        self.values["server_pid"] = self.server.pid
        # Read now: the start ticks name this process, and no restart.
        self.server_key = process_key(self.server.pid)
        if self.server.affinity_applied is False:
            self.record.protocol("affinity_not_applied:server", before_treatment=True)
            return False
        if not _wait_healthy(server.base_url, self.server, server.start_timeout_s):
            self.record.protocol("server_never_healthy", before_treatment=True)
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
        cleanup = verify_cleanup(self.server.pid, remembered)
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
            self.record.processes.append(launched.to_record())

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
        arm = plan.arms[prelude.server_arm]
        command = [expand(p, values) for p in (*plan.server.command, *arm.server_args)]
        server = launch(
            "server",
            command,
            env={**plan.server.env, **arm.server_env},
            cpu_affinity=plan.server.cpu_affinity,
            log_path=directory / "server.log",
        )
        if not _wait_healthy(plan.server.base_url, server, plan.server.start_timeout_s):
            stop(server, signals=(2, 15), timeout_s=plan.server.stop_timeout_s)
            return [prelude.step.name]
    try:
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
        )
    finally:
        if server is not None:
            stop(server, signals=(2, 15), timeout_s=plan.server.stop_timeout_s)
            verify_cleanup(server.pid)
    if timed_out or launched.exit_code not in prelude.step.expect_exit:
        return [prelude.step.name]
    return []


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


def _artifact_session(records: list[dict[str, Any]]) -> str:
    return next(
        (str(r["session_id"]) for r in records if r.get("session_id")), "unknown"
    )


def _without_gpu(options: DescribeOptions) -> DescribeOptions:
    return replace(options, no_gpu=True)


def _wait_healthy(base_url: str, server: Launched, timeout_s: float) -> bool:
    """Whether ``/health`` answers 200 before the timeout, while the server runs."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(HEALTH_POLL_SECONDS)
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
    "INDEX",
    "OUTCOME_FAILURE",
    "PROTOCOL_FAILURE",
    "Environment",
    "RunRecord",
    "run_plan",
]
