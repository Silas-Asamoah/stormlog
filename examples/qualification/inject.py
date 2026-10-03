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

import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from stormlog.infer.qualify.ground_truth import (
    Impact,
    Injection,
    PhaseWindow,
    Times,
    Validity,
    assess_impact,
    decide_status,
    is_aligned,
    write_injections,
)
from stormlog.infer.qualify.recovery import (
    MECHANISMS,
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
    effect_timing,
    next_episode,
    priming_check,
    realization,
)

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

VICTIM_RUN_ID = "victim"
VICTIM_PREFIX = f"chatcmpl-stormlog-{VICTIM_RUN_ID}-"
MARKER_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class Server:
    """The server under test, as the harness is given it."""

    base_url: str
    model: str
    hook_root: Path
    # Role (engine_core, api_server, sidecar, ...) -> pid it may pulse.
    targets: dict[str, int] = field(default_factory=dict)

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
        self._lock = threading.Lock()
        self._stop_polling = threading.Event()

    # ------------------------------------------------------------ the run

    def execute(self) -> Path:
        """Run the plan; return the published run directory."""
        directory = self.directory.create()
        (directory.truth / "plan.json").write_text(
            json.dumps(self.plan.to_record(), indent=2, sort_keys=True)
        )
        self.channel = self._channel()
        poller = threading.Thread(target=self._poll_loop, name="reference", daemon=True)
        poller.start()
        victim = self._start_victim()
        try:
            attempts, window, priming = self._episodes(victim)
        finally:
            self._stop_victim(victim)
            self._stop_polling.set()
            poller.join(timeout=10)
        self._write_truth(attempts, window, priming)
        return directory.publish()

    def _episodes(
        self, victim: subprocess.Popen[bytes]
    ) -> tuple[list[_Attempt], PhaseWindow, tuple[bool, float | None]]:
        t0 = self._wait_for_measured(victim)
        t = self.plan.timeline
        self._sleep_until(t0 + int(t.priming * SECOND))
        priming = priming_check(self._signals(), self.clock(), self.thresholds)
        baseline_end = t0 + int((t.priming + t.baseline) * SECOND)
        self._sleep_until(baseline_end)
        baseline = Baseline.measure(
            self._signals(), t0 + int(t.priming * SECOND), baseline_end
        )
        # The baseline itself is the first episode's clean time.
        attempts = self._run_episodes(
            baseline, clean_since=t0 + int(t.priming * SECOND)
        )
        self._sleep_for(t.final_recovery)
        return attempts, PhaseWindow(t0, self.clock()), priming

    def _run_episodes(self, baseline: Baseline, clean_since: int) -> list[_Attempt]:
        attempts: list[_Attempt] = []
        for index, episode in enumerate(self.plan.episodes):
            attempt = self._attempt(index, episode, baseline, clean_since)
            attempts.append(attempt)
            if attempt.decision == TIMEOUT:
                break
            # Clean from the start of the recovery hold: the effect's end.
            clean_since = attempt.timing.end_ns or self.clock()
        return attempts

    def _attempt(
        self, index: int, episode: EpisodePlan, baseline: Baseline, clean_since: int
    ) -> _Attempt:
        # An episode starts only once its clean time has passed (alignment).
        self._sleep_until(clean_since + int(self.plan.timeline.min_clean * SECOND))
        started = self.clock()
        actions, actuated, injected = self._actuate(index, episode)
        ended = self.clock()
        onset = _action_onset(actions, started)
        decision, timing = self._recover(episode, baseline, actions, onset, ended)
        context = Context(
            self._signals(), baseline, actions, onset, self.clock(), self.thresholds
        )
        realized, checks = (
            realization(episode.type, context, timing)
            if episode.type in _REALIZED
            else (True, [])
        )
        return _Attempt(
            index, episode, actions, onset, ended, actuated, injected, timing,
            decision, realized, [check.to_record() for check in checks], clean_since,
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
            name=str(index),
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
        pid = self.server.targets.get(role)
        if pid is None:
            return Actions(), False, {"method": PULSE, "error": f"no pid for {role}"}
        pulse_s = episode.dose["pulse_ms"] / 1000
        period_s = episode.dose["period_ms"] / 1000
        count = max(1, int(self.plan.timeline.episode / period_s))
        target = Target.of(pid, role)
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

    def _capture(self, episode: EpisodePlan) -> tuple[Actions, bool, dict[str, Any]]:
        window = capture_window(self.server.base_url, float(episode.dose["seconds"]))
        stop = window.stop
        actions = Actions(
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
        if episode.type not in MECHANISMS and episode.type != "I1":
            # N and the like: the effect window is the action's.
            return Timing(onset, "scheduled_window", action_end, action_end)
        context = Context(
            self._signals(), baseline, actions, onset, now, self.thresholds
        )
        return effect_timing(episode.type, context)

    # ------------------------------------------------------------ the truth

    def _write_truth(
        self,
        attempts: list[_Attempt],
        window: PhaseWindow,
        priming: tuple[bool, float | None],
    ) -> None:
        records = self._victim_records()
        t = self.plan.timeline
        baseline_start = window.start_ns + int(t.priming * SECOND)
        baseline_end = baseline_start + int(t.baseline * SECOND)
        slo = Slo(self.plan.victim.slo_ttft_ms, self.plan.victim.slo_e2e_ms)
        injections = [
            self._injection(
                attempt, window, priming, slo, records, (baseline_start, baseline_end)
            )
            for attempt in attempts
        ]
        injections += [
            _skipped(f"{self.directory.label}-e{index}", episode, priming)
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

    def _injection(
        self,
        attempt: _Attempt,
        window: PhaseWindow,
        priming: tuple[bool, float | None],
        slo: Slo,
        records: list[dict[str, Any]],
        baseline: tuple[int, int],
    ) -> Injection:
        row = attempt.plan.row
        times = Times(
            action_onset_ns=attempt.action_onset_ns,
            action_end_ns=attempt.action_end_ns,
            effect_onset_ns=attempt.timing.onset_ns,
            effect_end_ns=attempt.timing.end_ns,
            effect_basis=attempt.timing.basis,
            recovery_held_at_ns=attempt.timing.recovery_held_at_ns,
            priming_check={"passed": priming[0], "cached_fraction_median": priming[1]},
        )
        impact = _impact(attempt.timing, slo, records, baseline)
        status = decide_status(
            protocol_failure=not priming[0],
            actuated=attempt.actuated,
            aligned=is_aligned(
                times,
                window,
                clean_since_ns=attempt.clean_since_ns,
                min_baseline_ns=int(self.plan.timeline.min_clean * SECOND),
            ),
            realized=attempt.realized,
            recovered=attempt.decision == START,
        )
        validity = Validity(
            actuation="ok" if attempt.actuated else "failed",
            realization="realized" if attempt.realized else "not_realized",
            observation="not_assessed",
            impact=impact,
            checks=tuple(attempt.checks),
        )
        return Injection(
            episode_id=f"{self.directory.label}-e{attempt.index}",
            episode_type=row.id,
            cause_class=row.cause_class,
            injected=attempt.injected,
            expects=row.expects,
            secondary=row.secondary,
            allows=row.allows,
            times=times,
            clock_domain=None,
            status=status,
            actions=(),
            validity=validity,
        )

    # ------------------------------------------------------------ helpers

    def _channel(self) -> ReferenceChannel:
        return ReferenceChannel(
            hook_root=self.server.hook_root,
            metrics_url=f"{self.server.base_url}/metrics",
            victim_prefix=VICTIM_PREFIX,
            shared_prefix_tokens=max(1, self.plan.victim.shared_prefix_tokens),
            reference_dir=self.directory.reference,
            probes_dir=self.directory.probes,
            victim_artifact=self.directory.run / "victim.jsonl",
        )

    def _poll_loop(self) -> None:
        """Read the hook and scrape once a second."""
        while not self._stop_polling.wait(self.poll_seconds):
            with self._lock:
                assert self.channel is not None
                self.channel.poll()

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
            "--run-id", VICTIM_RUN_ID,
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

    def _stop_victim(self, victim: subprocess.Popen[bytes]) -> None:
        if victim.poll() is None:
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


# The types whose realization the catalog checks (A.4's realization column).
_REALIZED = frozenset(
    {"F1", "T1", "F2", "T2", "F3", "T3", "T3b", "F4a", "F4b", "H0", "I1"}
)


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


def _skipped(
    episode_id: str, episode: EpisodePlan, priming: tuple[bool, float | None]
) -> Injection:
    """An episode skipped after a recovery timeout: published, never run."""
    row = episode.row
    return Injection(
        episode_id=episode_id,
        episode_type=row.id,
        cause_class=row.cause_class,
        injected={"method": row.method, "skipped": "recovery_timeout"},
        expects=row.expects,
        secondary=row.secondary,
        allows=row.allows,
        times=Times(
            priming_check={"passed": priming[0], "cached_fraction_median": priming[1]}
        ),
        clock_domain=None,
        status=decide_status(protocol_failure=not priming[0], actuated=False),
        validity=Validity(
            actuation="skipped", realization="not_assessed", observation="not_assessed"
        ),
    )


__all__ = ["VICTIM_PREFIX", "VICTIM_RUN_ID", "InjectionRun", "Server"]
