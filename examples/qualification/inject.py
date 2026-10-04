"""``inject``: run a victim, inject a plan's episodes, and write the truth.

One run (#221 design A.4) against a server someone else launched:

1. The victim starts (``examples.qualification.victim``) and the reference
   channel is polled every second.
2. **Priming**, then the priming check: the victim's median cached fraction
   over the last 10 s must be at least 0.9, or every episode of the run is
   a protocol failure.
3. **Baseline**, measured for recovery's thresholds.
4. **Episodes** in the plan's order. Each starts once the previous one's
   recovery has held, and no sooner than its minimum; a recovery timeout
   skips the rest, which are published as not actuated.
5. **Final recovery**, then the victim is stopped, and every attempted
   episode's ``stormlog.qualify.injection/1`` record is written with its
   four validity layers and status. The run is published atomically.

The harness never launches the server: it is given the server's URL, the
pid of each role it may pulse, and the hook directory the server writes.
"""

from __future__ import annotations

import hashlib
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from stormlog.infer.host_clock import host_boot_id, wall_clock_domain
from stormlog.infer.qualify.ground_truth import (
    Impact,
    Injection,
    Interval,
    Neutral,
    PhaseWindow,
    RunRecord,
    Times,
    Validity,
    assess_impact,
    decide_status,
    is_aligned,
    write_injections,
    write_run,
)
from stormlog.infer.qualify.recovery import (
    SECOND,
    START,
    TIMEOUT,
    WAIT,
    Actions,
    Baseline,
    Context,
    Signals,
    Thresholds,
    Timing,
    added_mechanisms,
    effect_timing,
    next_episode,
    priming_check,
    realization,
)
from stormlog.infer.server_clock import client_clock_domain

from .capture import capture_window
from .catalog import CAPTURE, NEIGHBOR, PULSE
from .fake_engine.process import _environment
from .neighbor import Neighbor
from .outcomes import Slo, count_outcomes
from .plan import EpisodePlan, Plan
from .pulser import Pulser, Target
from .reference import ReferenceChannel
from .run_dir import RunDirectory
from .victim import read_marker

MARKER_TIMEOUT_SECONDS = 120.0
# After its stop file appears the victim drains (its --timeout at most) and
# runs its post-run imports; past this it is interrupted instead.
VICTIM_STOP_SECONDS = 180.0


def neighbor_name(run_id: str, index: int) -> str:
    """An opaque name for a run's ``index``-th neighbor: its request IDs reach
    the hook log a diagnosed configuration may import, so they must not give
    away the episode order (C.4)."""
    return hashlib.sha256(f"{run_id}:{index}".encode()).hexdigest()[:12]


def victim_prefix(run_id: str) -> str:
    """The victim's request IDs, as vLLM names them: the victim runs under the
    run's own label, so its artifact names the run its truth belongs to."""
    return f"chatcmpl-stormlog-{run_id}-"


@dataclass(frozen=True)
class Server:
    """The server under test, as the harness is given it."""

    base_url: str
    model: str
    hook_root: Path
    # Role (engine_core, api_server, sidecar, ...) -> the process it may
    # pulse, named by pid and start time when the run began.
    targets: dict[str, Target] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"


@dataclass
class _Attempt:
    """An episode as it ran, before its status can be decided."""

    index: int
    plan: EpisodePlan
    actions: Actions
    action_onset_ns: int
    action_end_ns: int
    actuated: bool
    injected: dict[str, Any]
    timing: Timing
    decision: str
    realized: bool
    checks: list[dict[str, Any]]
    clean_since_ns: int
    actions_record: list[dict[str, Any]] = field(default_factory=list)
    # Mechanisms realized beyond the label's (A.4), as "kind@component".
    added: tuple[str, ...] = ()


@dataclass
class _Progress:
    """What a run has done so far, kept as it goes, so that a run that fails
    part way still publishes every episode it attempted."""

    started_ns: int
    attempts: list[_Attempt] = field(default_factory=list)
    measured_start: int | None = None
    priming_end: int | None = None
    baseline_end: int | None = None
    final_start: int | None = None
    priming: tuple[bool, float | None] = (False, None)
    failure: str | None = None

    def windows(self, now: int) -> _Windows:
        start = self.measured_start or self.started_ns
        priming_end = self.priming_end or start
        baseline_end = self.baseline_end or priming_end
        return _Windows(start, priming_end, baseline_end, self.final_start or now, now)

    def protocol_failure(self) -> str | None:
        if self.failure is not None:
            return self.failure
        return None if self.priming[0] else "priming_check_failed"


@dataclass(frozen=True)
class _Windows:
    """The run's own windows, on the victim's clock."""

    measured_start: int
    priming_end: int
    baseline_end: int
    final_start: int
    measured_end: int

    def run_record(
        self, run_id: str, clock_domain: str | None, failure: str | None
    ) -> RunRecord:
        return RunRecord(
            run_id=run_id,
            clock_domain=clock_domain,
            measured=Interval(self.measured_start, self.measured_end),
            priming=Interval(self.measured_start, self.priming_end),
            baseline=Interval(self.priming_end, self.baseline_end),
            final_recovery=Interval(self.final_start, self.measured_end),
            protocol_failure=failure,
        )


class InjectionRun:
    """One injection run."""

    def __init__(
        self,
        plan: Plan,
        directory: RunDirectory,
        server: Server,
        victim_arguments: Sequence[str] = (),
        *,
        poll_seconds: float = 1.0,
        clock: Callable[[], int] = time.time_ns,
    ) -> None:
        self.plan = plan
        self.directory = directory
        self.server = server
        self.victim_arguments = list(victim_arguments)
        self.poll_seconds = poll_seconds
        self.clock = clock
        self.thresholds: Thresholds = plan.recovery_thresholds()
        self.channel: ReferenceChannel | None = None
        # The run's own failure, if it had one (episodes' are in the truth).
        self.failure: str | None = None
        self._lock = threading.Lock()
        self._stop_polling = threading.Event()
        self._closed = False

    # ------------------------------------------------------------ the run

    def execute(self) -> Path:
        """Run the plan; return the published run directory. Whatever ends
        the run, every episode it attempted is written and published: an
        episode whose actuation raised is not actuated, and a run that fails
        or is interrupted records why in its run record (``self.failure``).
        An interruption is re-raised once the run is published."""
        directory = self.directory.create()
        (directory.truth / "plan.json").write_text(
            json.dumps(self.plan.to_record(), indent=2, sort_keys=True)
        )
        self.channel = self._channel()
        poller = threading.Thread(target=self._poll_loop, name="reference", daemon=True)
        poller.start()
        progress = _Progress(self.clock())
        victim: subprocess.Popen[bytes] | None = None
        try:
            victim = self._start_victim()
            self._episodes(victim, progress)
        except Exception as error:  # the run's own failure; publish it
            progress.failure = f"run_failed: {error!r}"
        except BaseException:
            progress.failure = "interrupted"
            self._finish(victim, poller, progress)
            raise
        return self._finish(victim, poller, progress)

    def _finish(
        self,
        victim: subprocess.Popen[bytes] | None,
        poller: threading.Thread,
        progress: _Progress,
    ) -> Path:
        if victim is not None:
            self._stop_victim(victim)
        self._close_channel()
        poller.join(timeout=10)
        if self.channel is not None:
            # The hook log the replay cuts by first-seen time, in the truth.
            self.channel.tailer.copy_to(self.directory.reference / "hook")
        self.failure = progress.failure
        self._write_truth(progress)
        return self.directory.publish()

    def _episodes(self, victim: subprocess.Popen[bytes], progress: _Progress) -> None:
        t0 = progress.measured_start = self._wait_for_measured(victim)
        t = self.plan.timeline
        priming_end = progress.priming_end = t0 + int(t.priming * SECOND)
        self._sleep_until(priming_end)
        progress.priming = priming_check(self._signals(), self.clock(), self.thresholds)
        baseline_end = progress.baseline_end = priming_end + int(t.baseline * SECOND)
        self._sleep_until(baseline_end)
        baseline = Baseline.measure(self._signals(), priming_end, baseline_end)
        # The baseline itself is the first episode's clean time.
        self._run_episodes(baseline, priming_end, progress.attempts)
        progress.final_start = self.clock()
        self._sleep_for(t.final_recovery)

    def _run_episodes(
        self, baseline: Baseline, clean_since: int, attempts: list[_Attempt]
    ) -> None:
        for index, episode in enumerate(self.plan.episodes):
            attempt = self._attempt(index, episode, baseline, clean_since)
            attempts.append(attempt)
            if attempt.decision == TIMEOUT:
                break
            # Clean from the start of the recovery hold: the effect's end.
            clean_since = attempt.timing.end_ns or self.clock()

    def _attempt(
        self, index: int, episode: EpisodePlan, baseline: Baseline, clean_since: int
    ) -> _Attempt:
        # An episode starts only once its clean time has passed (alignment).
        self._sleep_until(clean_since + int(self.plan.timeline.min_clean * SECOND))
        started, started_mono = self.clock(), time.monotonic_ns()
        try:
            actions, actuated, injected = self._actuate(index, episode)
        except Exception as error:  # a failed actuation: not actuated, on record
            actions, actuated = Actions(), False
            injected = {"method": episode.row.method, "error": repr(error)}
        ended, ended_mono = self.clock(), time.monotonic_ns()
        result = "ok" if actuated else str(injected.get("error", "failed"))
        actions_record = [
            {"kind": f"{episode.row.method}_start", "at_wall_ns": started,
             "at_mono_ns": started_mono, "result": "ok"},
            {"kind": f"{episode.row.method}_end", "at_wall_ns": ended,
             "at_mono_ns": ended_mono, "result": result},
        ]  # fmt: skip
        # The action's own span: a twin's realization is judged over all of
        # it, and a null run's slot is it.
        actions = replace(actions, action_end_ns=ended, slot_ns=(started, ended))
        onset = _action_onset(actions, started)
        decision, timing = self._recover(episode, baseline, actions, onset, ended)
        if episode.row.method == PULSE:
            self._mark_landings(injected)
        context = Context(
            self._signals(), baseline, actions, onset, self.clock(), self.thresholds
        )
        realized, checks = realization(episode.type, context, timing)
        return _Attempt(
            index, episode, actions, onset, ended, actuated, injected, timing,
            decision, realized, [check.to_record() for check in checks], clean_since,
            actions_record, added_mechanisms(episode.type, checks),
        )  # fmt: skip

    # ------------------------------------------------------------ actuation

    def _actuate(
        self, index: int, episode: EpisodePlan
    ) -> tuple[Actions, bool, dict[str, Any]]:
        method = episode.row.method
        if method == NEIGHBOR:
            return self._neighbor(index, episode)
        if method == PULSE:
            return self._pulse(episode)
        if method == CAPTURE:
            return self._capture(episode)
        self._sleep_for(self.plan.timeline.episode)
        return Actions(), True, {"method": "none"}

    def _neighbor(
        self, index: int, episode: EpisodePlan
    ) -> tuple[Actions, bool, dict[str, Any]]:
        neighbor = Neighbor(
            name=neighbor_name(self.directory.label, index),
            shape=episode.neighbor_shape(),
            endpoint=self.server.endpoint,
            model=self.server.model,
            duration_seconds=self.plan.timeline.episode,
            output=self.directory.truth / f"neighbor-{index}.jsonl",
            seed=self.plan.seed + index,
        )
        neighbor.start()
        neighbor.join()
        actuation = neighbor.actuation()
        with self._lock:
            assert self.channel is not None
            admitted = self.channel.view.first_admission(neighbor.external_prefix)
        actions = Actions(
            first_send_ns=actuation.first_send_ns, first_admission_ns=admitted
        )
        injected = {
            "method": NEIGHBOR,
            "dose": dict(episode.dose),
            "actuation": actuation.to_record(),
        }
        return actions, actuation.ok, injected

    def _pulse(self, episode: EpisodePlan) -> tuple[Actions, bool, dict[str, Any]]:
        role = episode.row.pulse_role or ""
        target = self.server.targets.get(role)
        if target is None:
            return (
                Actions(),
                False,
                {"method": PULSE, "error": f"no process for {role}"},
            )
        if not target.is_alive():
            # Named when the run began; a pid since reused is never signalled.
            error = f"the {role} process (pid {target.pid}) exited or was replaced"
            return Actions(), False, {"method": PULSE, "error": error}
        pulse_s = episode.dose["pulse_ms"] / 1000
        period_s = episode.dose["period_ms"] / 1000
        count = max(1, int(self.plan.timeline.episode / period_s))
        with Pulser(target, max_pulse_seconds=pulse_s) as pulser:
            pulses = pulser.run(pulse_s, period_s, count)
        actions = Actions(
            first_stop_confirmed_ns=pulses[0].stopped_ns,
            last_continue_ns=pulses[-1].continue_sent_ns,
            pulses=[(p.stopped_ns, p.continue_sent_ns) for p in pulses],
        )
        injected = {
            "method": PULSE,
            "dose": dict(episode.dose),
            "target": target.to_record(),
            "pulses": [pulse.to_record() for pulse in pulses],
        }
        return actions, len(pulses) == count, injected

    def _mark_landings(self, injected: dict[str, Any]) -> None:
        """Where each pulse landed in the step loop (A.4, #218 R12), from the
        hook records read by the end of its recovery."""
        with self._lock:
            assert self.channel is not None
            view = self.channel.view
            for pulse in injected.get("pulses") or ():
                pulse["landed"] = view.landing(int(pulse["stop_sent_ns"]))

    def _capture(self, episode: EpisodePlan) -> tuple[Actions, bool, dict[str, Any]]:
        window = capture_window(self.server.base_url, float(episode.dose["seconds"]))
        stop = window.stop
        actions = Actions(
            # Started only once the server said so: an ambiguous start isn't.
            capture_started_ns=(
                window.start.returned_ns if window.start.status == 200 else None
            ),
            stop_requested_ns=None if stop is None else stop.requested_ns,
            stop_returned_ns=None if stop is None else stop.returned_ns,
        )
        return actions, window.ok, {"method": CAPTURE, "window": window.to_record()}

    # ------------------------------------------------------------ recovery

    def _recover(
        self,
        episode: EpisodePlan,
        baseline: Baseline,
        actions: Actions,
        onset: int,
        action_end: int,
    ) -> tuple[str, Timing]:
        """Wait until the next episode may start; the effect's timing."""
        while True:
            now = self.clock()
            timing = self._timing(episode, baseline, actions, onset, action_end, now)
            decision = next_episode(
                action_end, timing.recovery_held_at_ns, now, self.thresholds
            )
            if decision != WAIT:
                return decision, timing
            time.sleep(self.poll_seconds / 4)

    def _timing(
        self,
        episode: EpisodePlan,
        baseline: Baseline,
        actions: Actions,
        onset: int,
        action_end: int,
        now: int,
    ) -> Timing:
        context = Context(
            self._signals(), baseline, actions, onset, now, self.thresholds
        )
        return effect_timing(episode.type, context)

    # ------------------------------------------------------------ the truth

    def _write_truth(self, progress: _Progress) -> None:
        attempts, priming = progress.attempts, progress.priming
        windows = progress.windows(self.clock())
        records = self._victim_records()
        clock = _victim_clock(records)
        truth = _Truth(
            run_id=self.directory.label,
            window=PhaseWindow(windows.measured_start, windows.measured_end),
            priming=priming,
            slo=Slo(self.plan.victim.slo_ttft_ms, self.plan.victim.slo_e2e_ms),
            records=records,
            baseline=(windows.priming_end, windows.baseline_end),
            clock_domain=clock,
            same_clock=clock is not None and clock == _harness_clock(),
        )
        write_run(
            self.directory.truth / "run.json",
            windows.run_record(
                self.directory.label, clock, progress.protocol_failure()
            ),
        )
        injections = [self._injection(attempt, truth) for attempt in attempts]
        injections += [
            _skipped(truth, index, episode)
            for index, episode in enumerate(self.plan.episodes)
            if index >= len(attempts)
        ]
        write_injections(self.directory.truth / "injections.jsonl", injections)
        episodes = [
            {
                "index": a.index,
                "type": a.plan.type,
                "injected": a.injected,
                "decision": a.decision,
            }
            for a in attempts
        ]
        (self.directory.truth / "episodes.json").write_text(
            json.dumps(episodes, indent=2, sort_keys=True)
        )

    def _injection(self, attempt: _Attempt, truth: _Truth) -> Injection:
        row = attempt.plan.row
        times = Times(
            action_onset_ns=attempt.action_onset_ns,
            action_end_ns=attempt.action_end_ns,
            effect_onset_ns=attempt.timing.onset_ns,
            effect_end_ns=attempt.timing.end_ns,
            effect_basis=attempt.timing.basis,
            recovery_held_at_ns=attempt.timing.recovery_held_at_ns,
            priming_check=truth.priming_record(),
        )
        impact = _impact(attempt.timing, truth.slo, truth.records, truth.baseline)
        status = decide_status(
            protocol_failure=not truth.priming[0],
            same_clock=truth.same_clock,
            actuated=attempt.actuated,
            aligned=is_aligned(
                times,
                truth.window,
                clean_since_ns=attempt.clean_since_ns,
                min_baseline_ns=int(self.plan.timeline.min_clean * SECOND),
            ),
            realized=attempt.realized,
            recovered=attempt.decision == START,
        )
        labelled = [f"{e.kind}@{e.component}" for e in row.expects]
        validity = Validity(
            actuation="ok" if attempt.actuated else "failed",
            realization="realized" if attempt.realized else "not_realized",
            observation="not_assessed",
            impact=impact,
            realized_mechanisms=tuple(
                (labelled if attempt.realized else []) + list(attempt.added)
            ),
            checks=tuple(attempt.checks),
        )
        # A mechanism the episode realized beyond its label is allowed: a
        # diagnosis naming it is not wrong.
        allows = row.allows + tuple(_neutral(mechanism) for mechanism in attempt.added)
        return Injection(
            episode_id=f"{truth.run_id}-e{attempt.index}",
            run_id=truth.run_id,
            episode_type=row.id,
            cause_class=row.cause_class,
            injected=attempt.injected,
            expects=row.expects,
            secondary=row.secondary,
            allows=allows,
            times=times,
            clock_domain=truth.clock_domain,
            status=status,
            actions=tuple(attempt.actions_record),
            validity=validity,
        )

    # ------------------------------------------------------------ helpers

    def _channel(self) -> ReferenceChannel:
        return ReferenceChannel(
            hook_root=self.server.hook_root,
            metrics_url=f"{self.server.base_url}/metrics",
            victim_prefix=victim_prefix(self.directory.label),
            shared_prefix_tokens=max(1, self.plan.victim.shared_prefix_tokens),
            reference_dir=self.directory.reference,
            probes_dir=self.directory.probes,
            victim_artifact=self.directory.run / "victim.jsonl",
        )

    def _poll_loop(self) -> None:
        """Read the hook and scrape once a second. A poll that fails is
        recorded in ``probes/poll-errors.jsonl``, and polling goes on."""
        while not self._stop_polling.wait(self.poll_seconds):
            with self._lock:
                if self._closed:
                    return
                assert self.channel is not None
                try:
                    self.channel.poll()
                except Exception as error:  # the judge must outlive one bad poll
                    _append_error(self.directory.probes / "poll-errors.jsonl", error)

    def _close_channel(self) -> None:
        """Stop polling: the lock waits out a poll in progress, so nothing is
        appended to the run once it is hashed."""
        self._stop_polling.set()
        with self._lock:
            self._closed = True

    def _signals(self) -> Signals:
        """The signals so far, with the hook read up to now (scrapes stay at
        the poller's cadence)."""
        with self._lock:
            assert self.channel is not None
            self.channel.poll(scrape=False)
            return self.channel.signals()

    def _start_victim(self) -> subprocess.Popen[bytes]:
        victim = self.plan.victim
        arguments = [
            "--endpoint", self.server.endpoint,
            "--model", self.server.model,
            "--arrival", "poisson",
            "--rate", str(victim.rate_per_second),
            "--duration", str(self.plan.victim_duration_seconds()),
            "--input-tokens", str(victim.input_tokens),
            "--output-tokens", str(victim.output_tokens),
            "--prompt-mode", "shared-prefix",
            "--shared-prefix-ratio", str(victim.shared_prefix_ratio),
            "--prefix-groups", str(victim.prefix_groups),
            "--seed", str(self.plan.seed),
            "--run-id", self.directory.label,
            "--stop-file", str(self._victim_stop_file),
            "--output", str(self.directory.run / "victim.jsonl"),
            *self.victim_arguments,
        ]  # fmt: skip
        command = [
            sys.executable, "-m", "examples.qualification.victim",
            "--probes", str(self.directory.probes), "--", *arguments,
        ]  # fmt: skip
        log = (self.directory.probes / "victim.log").open("wb")
        return subprocess.Popen(
            command, env=_environment(), stdout=log, stderr=subprocess.STDOUT
        )

    @property
    def _victim_stop_file(self) -> Path:
        return self.directory.probes / "stop-victim"

    def _stop_victim(self, victim: subprocess.Popen[bytes]) -> None:
        """End the victim's measured window with its stop file, so it drains,
        imports and completes; interrupt it only if that doesn't end it."""
        if victim.poll() is not None:
            return
        self._victim_stop_file.touch()
        try:
            victim.wait(timeout=VICTIM_STOP_SECONDS)
            return
        except subprocess.TimeoutExpired:
            victim.send_signal(signal.SIGINT)
        try:
            victim.wait(timeout=60)
        except subprocess.TimeoutExpired:
            victim.kill()
            victim.wait()

    def _wait_for_measured(self, victim: subprocess.Popen[bytes]) -> int:
        deadline = time.monotonic() + MARKER_TIMEOUT_SECONDS
        markers = self.directory.probes / "markers"
        while time.monotonic() < deadline:
            marker = read_marker(markers, "measured", "started")
            if marker is not None:
                return int(marker["at_ns"])
            if victim.poll() is not None:
                raise RuntimeError(
                    f"the victim exited with {victim.returncode} before measuring"
                )
            time.sleep(0.05)
        raise TimeoutError("the victim never started measuring")

    def _victim_records(self) -> list[dict[str, Any]]:
        path = self.directory.run / "victim.jsonl"
        if not path.exists():
            return []
        data = path.read_bytes()
        complete = data[: data.rfind(b"\n") + 1]
        return [json.loads(line) for line in complete.splitlines() if line.strip()]

    def _sleep_until(self, at_ns: int) -> None:
        delay = (at_ns - self.clock()) / SECOND
        if delay > 0:
            time.sleep(delay)

    def _sleep_for(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))


def _action_onset(actions: Actions, started: int) -> int:
    for value in (
        actions.first_stop_confirmed_ns,
        actions.first_send_ns,
        actions.stop_requested_ns,
    ):
        if value is not None:
            return value
    return started


def _impact(
    timing: Timing,
    slo: Slo,
    records: list[dict[str, Any]],
    baseline: tuple[int, int],
) -> Impact | None:
    if not slo.defined or timing.onset_ns is None or timing.end_ns is None:
        return None
    effect = count_outcomes(records, timing.onset_ns, timing.end_ns, slo)
    reference = count_outcomes(records, baseline[0], baseline[1], slo)
    return assess_impact(effect, reference)


def _neutral(mechanism: str) -> Neutral:
    kind, _, component = mechanism.partition("@")
    return Neutral(kind, component)


def _append_error(path: Path, error: Exception) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at_ns": time.time_ns(), "error": repr(error)}) + "\n")


@dataclass(frozen=True)
class _Truth:
    """What every injection record of a run shares."""

    run_id: str
    window: PhaseWindow
    priming: tuple[bool, float | None]
    slo: Slo
    records: list[dict[str, Any]]
    baseline: tuple[int, int]
    clock_domain: str | None
    same_clock: bool

    def priming_record(self) -> dict[str, Any]:
        return {"passed": self.priming[0], "cached_fraction_median": self.priming[1]}


def _victim_clock(records: list[dict[str, Any]]) -> str | None:
    """The victim artifact's wall clock domain, from its first context."""
    for record in records:
        if "context" in record:
            return client_clock_domain(record)
    return None


def _harness_clock() -> str:
    """The clock the harness stamps its own times on: this host's."""
    return wall_clock_domain(socket.gethostname(), host_boot_id())


def _skipped(truth: _Truth, index: int, episode: EpisodePlan) -> Injection:
    """An episode skipped after a recovery timeout: published, never run."""
    row = episode.row
    return Injection(
        episode_id=f"{truth.run_id}-e{index}",
        run_id=truth.run_id,
        episode_type=row.id,
        cause_class=row.cause_class,
        injected={"method": row.method, "skipped": "recovery_timeout"},
        expects=row.expects,
        secondary=row.secondary,
        allows=row.allows,
        times=Times(priming_check=truth.priming_record()),
        clock_domain=truth.clock_domain,
        status=decide_status(protocol_failure=not truth.priming[0], actuated=False),
        validity=Validity(
            actuation="skipped", realization="not_assessed", observation="not_assessed"
        ),
    )


__all__ = ["InjectionRun", "Server", "neighbor_name", "victim_prefix"]
