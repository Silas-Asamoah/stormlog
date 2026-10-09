"""An injection run end to end, against the fake vLLM engine as its own
process: priming, baseline, three episodes with recovery, and the truth."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, cast

import psutil
import pytest

from examples.qualification.__main__ import main
from examples.qualification.fake_engine.process import FakeEngineProcess, _environment
from examples.qualification.inject import _harness_clock
from examples.qualification.run_dir import verify
from stormlog.infer.qualify.ground_truth import load_injections, load_run
from tests.qualification_fake_engine_helpers import post, wait_until

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(signal, "SIGSTOP"),
    reason="needs SIGSTOP and SIGCONT",
)


def _plan(path: Path) -> Path:
    plan: dict[str, Any] = {
        "format": "stormlog.qualify.plan/1",
        "profile": "smoke",
        "seed": 7,
        "victim": {
            "rate_per_second": 10.0,
            "input_tokens": 64,
            "output_tokens": 8,
            "prefix_groups": 2,
            "shared_prefix_ratio": 0.75,
            "slo_ttft_ms": 2000,
            "slo_e2e_ms": 5000,
        },
        "timeline": {
            "priming": 2,
            "baseline": 3,
            "episode": 2,
            "min_recovery": 1,
            "recovery_timeout": 15,
            "final_recovery": 1,
            "min_clean": 0.5,
        },
        # Holds of at least two scrape periods: a hold shorter than the
        # scrape spacing can contain no gauge sample.
        "thresholds": {
            "window": 1,
            "hold": 2.5,
            "cadence_hold": 1,
            "priming_window": 1,
        },
        "episodes": [
            {"type": "N"},
            {
                "type": "T3b",
                "dose": {"rate_per_second": 2, "input_tokens": 64, "output_tokens": 4},
            },
            {"type": "F4a", "dose": {"pulse_ms": 100, "period_ms": 400}},
            {
                "type": "F2",
                "dose": {"concurrency": 4, "input_tokens": 256, "output_tokens": 64},
            },
        ],
    }
    path.write_text(json.dumps(plan))
    return path


def test_a_run_injects_its_plan_and_publishes_the_truth(tmp_path: Path) -> None:
    hook = tmp_path / "hook"
    arguments = [
        "--step-seconds", "0.002",
        "--decode-token-seconds", "0.0005",
        "--num-gpu-blocks", "40",
        "--max-num-seqs", "8",
        "--hook-dir", str(hook),
    ]  # fmt: skip
    with FakeEngineProcess(arguments) as server:
        code = main(
            [
                "inject",
                "--plan", str(_plan(tmp_path / "plan.json")),
                "--out", str(tmp_path / "runs"),
                "--label", "q221-0123456789abcdef",
                "--base-url", server.base_url,
                "--model", "fake/qwen-0.5b",
                "--reference-channel", str(hook),
                "--target", f"engine_core={server.pid}",
                "--", "--tokenizer", "none", "--system-sampler", "none",
            ]  # fmt: skip
        )
    assert code == 0
    run = tmp_path / "runs" / "q221-0123456789abcdef"
    assert run.is_dir() and verify(run) == []
    injections = {
        i.episode_type: i for i in load_injections(run / "truth" / "injections.jsonl")
    }
    assert sorted(injections) == ["F2", "F4a", "N", "T3b"]
    for injection in injections.values():
        assert injection.times.priming_check == {
            "passed": True,
            "cached_fraction_median": pytest.approx(1.0),
        }
    stall = injections["F4a"]
    assert stall.status == "valid", stall.validity
    assert stall.injected["pulses"]
    landings = {"in_schedule", "in_step", "between_steps"}
    assert all(pulse["landed"] in landings for pulse in stall.injected["pulses"])
    assert stall.times.effect_onset_ns == stall.times.action_onset_ns
    assert injections["N"].status == "valid"
    # T3b: the neighbor's unique prompts pull the engine-wide hit ratio down,
    # read from the scraped prefix-cache counters, while the victim's own
    # cached fraction holds.
    twin = injections["T3b"]
    assert twin.status == "valid", twin.validity
    checks = {c["name"]: c for c in twin.validity.checks}
    assert checks["engine_hit_ratio_fell"]["passed"] is True
    assert not checks["engine_hit_ratio_fell"]["incomplete"]
    # Its effect, the benign change, spans its whole action, as N's does.
    times = twin.times
    assert times.effect_onset_ns is not None and times.action_end_ns is not None
    assert times.effect_end_ns is not None
    assert times.effect_end_ns >= times.action_end_ns > times.effect_onset_ns
    # Every label at an engine component names the engine #218 will name:
    # the producer in the fake engine's hello, for L2.
    (hello,) = [
        json.loads(line)
        for path in (run / "truth" / "reference" / "hook").rglob("*.jsonl*")
        if "engine-" in str(path)
        for line in path.read_text().splitlines()
        if '"hello"' in line
    ]
    assert hello["producer"]
    assert stall.expects[0].component == "engine_core"
    assert stall.expects[0].engine == hello["producer"]
    assert injections["F2"].expects[0].engine == hello["producer"]
    # F2's neighbor preempts victim requests: the KV fault is realized.
    kv = injections["F2"]
    assert kv.status == "valid", kv.validity
    (preempted,) = [c for c in kv.validity.checks if c["name"] == "victim_preempted"]
    assert preempted["value"] > 0
    assert (run / "run" / "victim.jsonl").exists()
    assert (run / "truth" / "reference" / "scrapes.jsonl").exists()
    assert (run / "probes" / "hook-firstseen.jsonl").exists()
    # The record lag the hang check allowed was measured once the baseline
    # ended (close-221-delta, N4), and is on record.
    used = json.loads((run / "probes" / "record-lag.json").read_text())
    assert used["measured_p99_ns"] is not None and used["lag_ns"] >= 1_000_000_000
    # The run record: its label, its windows and the victim's clock, which
    # every episode shares; the victim ran under the same label.
    record = load_run(run / "truth" / "run.json")
    assert record.run_id == "q221-0123456789abcdef"
    assert record.clock_domain is not None
    assert record.priming is not None and record.final_recovery is not None
    assert record.measured.start_ns == record.priming.start_ns
    assert record.final_recovery.end_ns == record.measured.end_ns
    # close-221-delta, H4: run.json says how near the dose check each
    # cadence series came, and each episode how long after its action its
    # effect ended, so a late recovery just under the limit can be read.
    dose = record.baseline_checks["dose_check"]
    assert dose["limit_per_hold"] == 1.0
    assert 0 <= dose["per_hold"]["busy step gaps"] < 1
    times = stall.times
    assert times.effect_end_ns is not None and times.action_end_ns is not None
    lateness = times.effect_end_ns - times.action_end_ns
    assert stall.injected["recovery_lateness_ns"] == lateness
    for injection in injections.values():
        assert injection.run_id == record.run_id
        assert injection.clock_domain == record.clock_domain
        assert [a["kind"].rsplit("_", 1)[1] for a in injection.actions] == [
            "start",
            "end",
        ]
    (artifact,) = [
        json.loads(line)
        for line in (run / "run" / "victim.jsonl").read_text().splitlines()
        if '"infer.artifact"' in line
    ]
    assert artifact["context"]["run_id"] == record.run_id
    assert artifact["context"]["clock_domain"] == record.clock_domain
    # The victim ended its measured window when the run was done, drained
    # and completed: the diagnoser gets a whole artifact, not an
    # interrupted one.
    victim_records = [
        json.loads(line)
        for line in (run / "run" / "victim.jsonl").read_text().splitlines()
    ]
    (measured,) = [
        r
        for r in victim_records
        if r.get("event_type") == "infer.phase_window" and r["phase"] == "measured"
    ]
    assert measured["stopped_early"] is True
    # The run's measured window ends where the victim's did, before its
    # drain (Fable's A2 delta N5).
    assert record.measured.end_ns == measured["window_ended_at_ns"]
    sessions = [r for r in victim_records if r.get("event_type") == "infer.session"]
    assert sessions[-1]["status"] == "completed"
    assert "KeyboardInterrupt" not in (run / "probes" / "victim.log").read_text()
    # The reference hook log is in the truth, beside the first-seen notes
    # that cut it for the replay.
    copied = [
        line
        for path in (run / "truth" / "reference" / "hook").rglob("*.jsonl*")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    noted = (run / "probes" / "hook-firstseen.jsonl").read_text().splitlines()
    assert len(copied) >= len(noted) > 0


def test_a_bad_plan_is_refused_before_anything_runs(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"format": "other"}))
    command = [
        sys.executable, "-m", "examples.qualification", "inject",
        "--plan", str(plan), "--out", str(tmp_path), "--base-url", "http://127.0.0.1:9",
        "--model", "m", "--reference-channel", str(tmp_path),
    ]  # fmt: skip
    finished = subprocess.run(
        command, env=_environment(), capture_output=True, text=True
    )
    assert finished.returncode == 2
    assert "format is not" in finished.stderr


def test_a_target_replaced_since_the_run_began_is_never_pulsed(
    tmp_path: Path,
) -> None:
    # The CLI binds each target to its start time at startup; an episode
    # whose target exited, or whose pid now names another process, is not
    # actuated, and nothing is signalled.
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.pulser import Target
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan({**record, "episodes": [{"type": "F4a"}]})
    stand_in = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        replaced = Target(
            stand_in.pid, Target.of(stand_in.pid).start_time - 10, "engine_core"
        )
        server = Server("http://127.0.0.1:9", "m", tmp_path, {"engine_core": replaced})
        run = InjectionRun(plan, RunDirectory(tmp_path / "runs", "q221-x"), server)
        _actions, actuated, injected = run._pulse(plan.episodes[0])
        assert not actuated
        assert "exited or was replaced" in injected["error"]
    finally:
        stand_in.kill()
        stand_in.wait()


def _after_the_stop_is_confirmed(
    monkeypatch: pytest.MonkeyPatch, act: Callable[[], None]
) -> None:
    """Run ``act`` in a thread once the pulser has confirmed its first stop.
    Acting on the target's own STOPPED state raced the pulser's first read
    (gate-221, F1 and F2: 4 runs in 30): the stop could end before the
    pulser saw it."""
    import threading

    from examples.qualification import pulser as pulser_module

    real = pulser_module.Pulser._confirm_stopped
    confirmed = threading.Event()

    def confirm(self: Any, stop_sent: int) -> int:
        seen = real(self, stop_sent)
        confirmed.set()
        return seen

    def wait_then_act() -> None:
        if confirmed.wait(30):
            act()

    monkeypatch.setattr(pulser_module.Pulser, "_confirm_stopped", confirm)
    threading.Thread(target=wait_then_act, daemon=True).start()


def test_the_pulse_a_failure_cut_short_is_in_what_was_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fable's and rev-220-a's second A2 deltas: the engine went through a
    # stop that a failure cut short (here its process killed mid-pulse),
    # but the truth listed completed pulses only.
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.pulser import PulseRefused, Target
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    dose = {"pulse_ms": 300, "period_ms": 800}
    plan = parse_plan({**record, "episodes": [{"type": "F4a", "dose": dose}]})
    loop = "import time\nwhile True:\n    time.sleep(0.01)\n"
    stand_in = subprocess.Popen([sys.executable, "-c", loop], start_new_session=True)

    def kill() -> None:
        stand_in.kill()
        stand_in.wait()

    try:
        target = Target.of(stand_in.pid, "engine_core")
        server = Server("http://127.0.0.1:9", "m", tmp_path, {"engine_core": target})
        run = InjectionRun(plan, RunDirectory(tmp_path / "runs", "q221-x"), server)
        # Inside the first 300 ms pulse, once its stop is confirmed.
        _after_the_stop_is_confirmed(monkeypatch, kill)
        with pytest.raises(PulseRefused, match="^target_gone: .* exited during"):
            run._pulse(plan.episodes[0])
        done = run._done_so_far(plan.episodes[0])
    finally:
        kill()
    (cut,) = done["pulses"]
    assert cut["completed"] is False and cut["continue_sent_ns"] is None


def test_a_pulse_continued_by_someone_else_leaves_the_episode_not_actuated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rev-220-a's final A2 delta-2 note: nothing read continued_by_other.
    # A pulse someone else continued wasn't the stop the dose asked for, so
    # the episode is not actuated, with actuation interrupted_by_other (the
    # lead's ruling), and its record names the pulse.
    from types import SimpleNamespace
    from typing import cast

    from examples.qualification.inject import InjectionRun, Server, _actuation
    from examples.qualification.plan import parse_plan
    from examples.qualification.pulser import Target
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    dose = {"pulse_ms": 300, "period_ms": 800}
    plan = parse_plan({**record, "episodes": [{"type": "F4a", "dose": dose}]})
    loop = "import time\nwhile True:\n    time.sleep(0.01)\n"
    stand_in = subprocess.Popen([sys.executable, "-c", loop], start_new_session=True)

    def continue_it() -> None:
        os.kill(stand_in.pid, signal.SIGCONT)  # the operator's

    try:
        target = Target.of(stand_in.pid, "engine_core")
        server = Server("http://127.0.0.1:9", "m", tmp_path, {"engine_core": target})
        run = InjectionRun(plan, RunDirectory(tmp_path / "runs", "q221-y"), server)
        _after_the_stop_is_confirmed(monkeypatch, continue_it)
        _actions, actuated, injected = run._pulse(plan.episodes[0])
    finally:
        stand_in.kill()
        stand_in.wait()
    assert actuated is False
    assert injected["interrupted_by_other"] == [0]
    attempt = cast(Any, SimpleNamespace(actuated=actuated, injected=injected))
    assert _actuation(attempt) == "interrupted_by_other"


def test_a_pulse_continued_before_its_stop_was_seen_reads_interrupted_by_other(
    tmp_path: Path,
) -> None:
    # gate-221, F2's product half: a pulse someone continued before the
    # pulser saw its stop raised "did not stop", and the episode read as a
    # failed actuation. It is named as continued by another now, and the
    # episode's actuation is interrupted_by_other.
    from types import SimpleNamespace
    from typing import cast

    from examples.qualification.inject import InjectionRun, Server, _actuation
    from examples.qualification.plan import parse_plan
    from examples.qualification.pulser import ContinuedByOther
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan({**record, "episodes": [{"type": "F4a"}]})
    run = InjectionRun(
        plan, RunDirectory(tmp_path / "runs", "q221-z"), Server("", "m", tmp_path, {})
    )
    cut = {"stop_sent_ns": 1, "stopped_ns": None, "completed": False}
    run._pulser = cast(Any, SimpleNamespace(pulses=[], cut_short=cut))
    error = ContinuedByOther("continued_by_other: pid 1 ran on after its SIGSTOP")
    injected = run._failed(plan.episodes[0], error)
    assert injected["interrupted_by_other"] == [0]
    attempt = cast(Any, SimpleNamespace(actuated=False, injected=injected))
    assert _actuation(attempt) == "interrupted_by_other"


@pytest.mark.parametrize("hung", [True, False])
def test_the_next_episode_waits_while_the_engine_looks_hung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hung: bool
) -> None:
    # rev-220-b's delta-3 closure, G4: the recovery was found in an idle
    # stretch, and START came at once, although a victim request sent at
    # 99 s has had no step since (an engine hung with an open-loop victim).
    # Now the decision reads the present, and a hung engine runs into the
    # recovery timeout instead.
    from examples.qualification import inject as module
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory
    from stormlog.infer.qualify.recovery import (
        START,
        TIMEOUT,
        Actions,
        Baseline,
        Signals,
        Timing,
    )

    second = 1_000_000_000
    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan({**record, "episodes": [{"type": "F4a"}]})
    now = [100 * second]

    def clock() -> int:
        now[0] += second // 4
        return now[0]

    run = InjectionRun(
        plan, RunDirectory(tmp_path / "runs", "q221-z"), Server("", "m", tmp_path, {}),
        clock=clock,
    )  # fmt: skip
    last_step = 99 * second if hung else 1000 * second
    steps = [tick * 20_000_000 for tick in range(last_step // 20_000_000)]
    signals = Signals(in_flight=[(0, 10**18)], step_starts=steps)
    baseline = Baseline.measure(signals, 0, 45 * second)
    monkeypatch.setattr(run, "_signals", lambda: signals)
    monkeypatch.setattr(
        run,
        "_timing",
        lambda *_args: Timing(90 * second, "test", 90 * second, 95 * second),
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    decision, _timing = run._recover(
        plan.episodes[0], baseline, Actions(), 90 * second, 95 * second
    )
    assert decision == (TIMEOUT if hung else START)


@pytest.mark.parametrize(
    ("late_ms", "measured_ms", "hung", "decision"),
    [
        (15, 15, False, "start"),
        (1500, 1500, False, "start"),
        (1500, None, False, "timeout"),
        (0, 60_000, True, "timeout"),
    ],
)
def test_a_healthy_engine_whose_records_arrive_late_is_not_hung(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    late_ms: int,
    measured_ms: int | None,
    hung: bool,
    decision: str,
) -> None:
    # Astra's closure of delta 3, H5: engine_stalled compared the gap open
    # at the poll with twice the baseline's p99 (40 ms here), but the hook's
    # records arrive after their steps start, so a healthy engine read as
    # hung at most polls and every episode ran into the recovery timeout.
    # The run allows at least a poll period, or the baseline's measured lag
    # if longer; a lag past both still reads as a hang. Which it used, and
    # why, is logged in probes/record-lag.json. close-221-delta, N3: a
    # measured p99 of a minute let an engine hung since 80 s start the next
    # episode; the allowance is capped at 10 s (a tenth of the 150 s
    # timeout at most), and the hang times out.
    from types import SimpleNamespace
    from typing import cast

    from examples.qualification import inject as module
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory
    from stormlog.infer.qualify.recovery import Actions, Baseline, Signals, Timing

    second, step = 1_000_000_000, 20_000_000
    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    record["timeline"]["recovery_timeout"] = 150
    plan = parse_plan({**record, "episodes": [{"type": "F4a"}]})
    now = [100 * second]

    def clock() -> int:
        now[0] += second // 4
        return now[0]

    run = InjectionRun(
        plan, RunDirectory(tmp_path / "runs", "q221-y"), Server("", "m", tmp_path, {}),
        clock=clock,
    )  # fmt: skip
    run.directory.create()
    lag = None if measured_ms is None else measured_ms * 1_000_000
    run.channel = cast(
        Any,
        SimpleNamespace(record_lag_ns=lambda *_: lag, stop_noting_lags=lambda: None),
    )
    run._measure_record_lag(0, 45 * second)
    # The lag used, and where it came from, is on record.
    used = json.loads((run.directory.probes / "record-lag.json").read_text())
    assert used["measured_p99_ns"] == lag
    assert used["cap_ns"] == 10 * second
    assert used["lag_ns"] == min(max(second, lag or 0), 10 * second)
    assert (
        used["source"]
        == {
            15: "poll_period: longer than the measured lag",
            1500: "measured",
            60_000: "cap: the measured lag was longer",
            None: "poll_period: no step record in the baseline to measure",
        }[measured_ms]
    )

    def seen() -> Signals:
        visible = now[0] - late_ms * 1_000_000
        if hung:
            visible = min(visible, 80 * second)
        steps = list(range(0, visible + 1, step))
        return Signals(in_flight=[(0, 10**18)], step_starts=steps)

    baseline = Baseline.measure(seen(), 0, 45 * second)
    monkeypatch.setattr(run, "_signals", seen)
    monkeypatch.setattr(
        run,
        "_timing",
        lambda *_args: Timing(90 * second, "test", 90 * second, 95 * second),
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    found, _timing = run._recover(
        plan.episodes[0], baseline, Actions(), 90 * second, 95 * second
    )
    assert found == decision


def test_a_recovery_that_can_never_hold_ends_the_episode_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # close-221-delta, H4: an engine refused by G0's dose check waited out
    # the full recovery timeout (150 s) before its episode ended. Its
    # recovery can never hold, so the decision is a timeout at once, with
    # no poll waited.
    from examples.qualification import inject as module
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory
    from stormlog.infer.qualify.recovery import (
        TIMEOUT,
        Actions,
        Baseline,
        GapStats,
        Signals,
        Timing,
    )

    second = 1_000_000_000
    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan({**record, "episodes": [{"type": "F4a"}]})
    now = [100 * second]

    def clock() -> int:
        now[0] += second // 4
        return now[0]

    run = InjectionRun(
        plan, RunDirectory(tmp_path / "runs", "q221-x"), Server("", "m", tmp_path, {}),
        clock=clock,
    )  # fmt: skip
    # 30 too-long gaps among 1,000 busy ones of 20 ms: 1.5 per 1 s hold.
    steps = GapStats(count=1000, mean=0.02, p95=0.03, p99=0.03, too_long_count=30)
    baseline = Baseline(
        wait_p95=1.0, waiting_low=0.0, waiting_high=1.0, kv_max=0.5,
        steps=steps, chunks=GapStats(), cached_median=1.0,
    )  # fmt: skip
    slept: list[float] = []
    monkeypatch.setattr(run, "_signals", lambda: Signals(in_flight=[(0, 10**18)]))
    monkeypatch.setattr(
        run, "_timing", lambda *_args: Timing(90 * second, "test", None, None)
    )
    monkeypatch.setattr(module.time, "sleep", slept.append)
    decision, _timing = run._recover(
        plan.episodes[0], baseline, Actions(), 90 * second, 95 * second
    )
    assert (decision, slept) == (TIMEOUT, [])


def test_the_baseline_is_measured_with_the_plans_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The run measured its baseline with the design's default thresholds,
    # so a plan's long_gap_factor never reached which baseline gaps were
    # long (the allowance) or too long (the dose check). Gaps of 20 ms with
    # one in 100 of 50 ms: long at a factor of 2 (over 40 ms) are the 50 ms
    # ones; at a factor of 3 (over 60 ms), none.
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory
    from stormlog.infer.qualify.recovery import Signals

    second, ms = 1_000_000_000, 1_000_000
    gaps = [50 if index % 100 == 50 else 20 for index in range(2000)]
    steps = [sum(gaps[:index]) * ms for index in range(len(gaps) + 1)]
    signals = Signals(in_flight=[(0, 10**18)], step_starts=steps)
    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    for factor, long in ((2.0, 20), (3.0, 0)):
        thresholds = {**record["thresholds"], "long_gap_factor": factor}
        plan = parse_plan({**record, "thresholds": thresholds})
        run = InjectionRun(
            plan,
            RunDirectory(tmp_path / "runs", "q221-w"),
            Server("", "m", tmp_path, {}),
        )
        monkeypatch.setattr(run, "_signals", lambda: signals)
        assert run._measure_baseline(0, 45 * second).steps.long_count == long


def test_targets_are_bound_at_startup(tmp_path: Path) -> None:
    from examples.qualification.__main__ import _targets

    stand_in = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        bound = _targets(None, [f"sidecar={stand_in.pid}"])
        assert bound["sidecar"].is_alive()
    finally:
        stand_in.kill()
        stand_in.wait()
    with pytest.raises(ValueError, match="sidecar="):
        _targets(None, [f"sidecar={stand_in.pid}"])


def test_queue_episodes_recover_end_to_end(tmp_path: Path) -> None:
    # Fable's A2 deltas, N0: the e2e above has no queue episode, so a plan
    # whose queue recovery could never hold passed CI. T1 and F1 with a
    # baseline and hold long enough for the waits and the waiting gauge:
    # each recovers, and F1's neighbor saturates the queue. 400 blocks keep
    # F1's load from preempting the victim, which would make it invalid.
    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    # A timeout well past the 6 s hold: on a loaded host the waits take
    # longer to settle back into the baseline's band, and the test asks
    # that they do, not how soon.
    record["timeline"].update(baseline=7, recovery_timeout=60)
    record["thresholds"]["hold"] = 6
    shape = {"input_tokens": 128, "output_tokens": 16}
    record["episodes"] = [
        {"type": "T1", "dose": {"rate_per_second": 2, **shape}},
        {"type": "F1", "dose": {"rate_per_second": 120, **shape}},
    ]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(record))
    # 20 ms steps, so a victim's wait (admission to its first step, about
    # half a step) is the engine's own time, not the host's. With 2 ms
    # steps the baseline's waits were 0.5-2 ms, its 2x p99 ceiling 7-30 ms
    # and its 1.25x mean bound a fraction of a millisecond: the host's
    # scheduling delays under the full suite's load broke every hold for
    # the 60 s timeout (2 of 3 full runs). Under 16 busy loops, two runs in
    # four lost a third to nearly half of their steady 6 s windows to the
    # mean bound, the ceiling or the exceedance count; at 20 ms none fails
    # the mean or the ceiling, and 82-96% hold.
    arguments = [
        "--step-seconds", "0.02",
        "--decode-token-seconds", "0.0005",
        "--num-gpu-blocks", "400",
        "--max-num-seqs", "8",
        "--hook-dir", str(tmp_path / "hook"),
    ]  # fmt: skip
    with FakeEngineProcess(arguments) as server:
        assert _inject(server, tmp_path, plan) == 0
    run = tmp_path / "runs" / "q221-00000000000000bb"
    assert verify(run) == []
    twin, fault = load_injections(run / "truth" / "injections.jsonl")
    for injection in (twin, fault):
        assert injection.validity.actuation == "ok", injection.validity
        assert injection.times.effect_end_ns is not None  # it recovered
    assert fault.status == "valid", fault.validity
    checks = {c["name"]: c["passed"] for c in fault.validity.checks}
    assert checks == {"onset_reached": True, "no_victim_preemption": True}
    # The twin is valid on a quiet host. On a loaded one (load average 20
    # in the full suite) the victim's own waits can pass the baseline's p95
    # for a window, and the harness rightly calls the twin not realized:
    # its one check then fails, nothing else, and only just. The check's
    # value, the highest window median over the p95, tells host noise from
    # a twin that saturated the queue (F1's own waits read many times over).
    if twin.status != "valid":
        (check,) = twin.validity.checks
        assert (check["name"], check["passed"]) == ("waits_within_baseline", False)
        assert check["value"] is not None and check["value"] < 3, twin.validity


def test_a_baseline_too_thin_to_recover_is_named_not_a_bare_timeout(
    tmp_path: Path,
) -> None:
    # fable-design's A2 delta 2, N0: a run whose baseline was too thin for a
    # queue rule timed out with only "recovery_timeout" in its truth. Here
    # /metrics fails, so the baseline has no waiting count: T1 can never
    # recover, its record says which series was too thin, and the N it
    # skips says baseline_too_thin.
    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    record["timeline"].update(baseline=7, recovery_timeout=4)
    record["thresholds"]["hold"] = 6
    record["episodes"] = [
        {"type": "T1", "dose": {"rate_per_second": 2, "input_tokens": 128,
                                "output_tokens": 16}},
        {"type": "N"},
    ]  # fmt: skip
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(record))
    arguments = ["--step-seconds", "0.002", "--hook-dir", str(tmp_path / "hook")]
    with FakeEngineProcess(arguments) as server:
        controls = json.dumps({"metrics_mode": "fail"}).encode()
        assert post(f"{server.base_url}/_fault/controls", controls)[0] == 200
        _inject(server, tmp_path, plan)
    run = tmp_path / "runs" / "q221-00000000000000bb"
    assert verify(run) == []
    twin, skipped = load_injections(run / "truth" / "injections.jsonl")
    assert twin.injected["recovery_blocked"] == [
        "baseline_too_thin: 0 waiting counts of the 5 a hold needs"
    ]
    assert skipped.injected["skipped"] == "baseline_too_thin"


def test_an_engine_whose_prefill_steps_outlast_a_dose_is_not_evaluable(
    tmp_path: Path,
) -> None:
    # The lead's note on Astra's H4, end to end. A fake engine whose victim
    # prefill steps take about 90 ms, past the 60 ms smallest dose, and come
    # about once in 200 busy steps (0.5 a second, long outputs) fails G0's
    # dose check: F4a can't recover, its record says dose_check_failed, the
    # N it skips says so, and the scorer counts it as not evaluable. At its
    # default prefill cost (about 2 ms a step) the engine passes the check:
    # the first e2e's F4a is valid. (With prefill steps over 1% of the busy
    # steps, twice the p99 is a prefill step and the check passes: the 2%
    # residual the guide leaves to G0's step times.)
    from stormlog.infer.qualify.ground_truth import load_run
    from stormlog.infer.qualify.scoring import ScoreConfig, score_run

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    record["victim"].update(rate_per_second=0.5, output_tokens=512)
    record["timeline"].update(baseline=15, recovery_timeout=10)
    record["thresholds"]["cadence_hold"] = 4
    record["episodes"] = [{"type": "F4a"}, {"type": "N"}]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(record))
    arguments = [
        "--step-seconds", "0.002",
        "--prefill-token-seconds", "0.004",
        "--hook-dir", str(tmp_path / "hook"),
    ]  # fmt: skip
    with FakeEngineProcess(arguments) as server:
        _inject(server, tmp_path, plan, "--target", f"engine_core={server.pid}")
    run = tmp_path / "runs" / "q221-00000000000000bb"
    assert verify(run) == []
    stall, null = load_injections(run / "truth" / "injections.jsonl")
    assert stall.validity.actuation == "ok"
    (reason,) = stall.injected["recovery_blocked"]
    assert reason.startswith("dose_check_failed: ") and "busy step gaps" in reason
    assert stall.injected["recovery_lateness_ns"] is None
    per_hold = load_run(run / "truth" / "run.json").baseline_checks["dose_check"]
    assert per_hold["per_hold"]["busy step gaps"] >= 1
    assert stall.status == "recovery_incomplete"
    assert null.injected["skipped"] == "dose_check_failed"
    nothing: dict[str, Any] = {"payload": {"findings_detail": {}, "coverage": {}}}
    config = ScoreConfig(supported_types=frozenset({"F4a"}))
    truth = load_run(run / "truth" / "run.json")
    scored = score_run(truth, [stall, null], nothing, config)
    assert scored.episodes[0].not_evaluable == "dose_check_failed"


def _inject(server: FakeEngineProcess, tmp_path: Path, plan: Path, *extra: str) -> int:
    return main(
        [
            "inject",
            "--plan", str(plan),
            "--out", str(tmp_path / "runs"),
            "--label", "q221-00000000000000bb",
            "--base-url", server.base_url,
            "--model", "fake/qwen-0.5b",
            "--reference-channel", str(tmp_path / "hook"),
            *extra,
            "--", "--tokenizer", "none", "--system-sampler", "none",
        ]  # fmt: skip
    )


def _short_plan(path: Path, *episodes: dict[str, Any]) -> Path:
    record = json.loads(_plan(path).read_text())
    record["episodes"] = list(episodes)
    path.write_text(json.dumps(record))
    return path


def test_the_skipped_episodes_name_why_recovery_could_never_hold() -> None:
    # Astra's closure of delta 3, H4: a cadence episode refused by G0's dose
    # check also times out. The episodes it skips say so, not that the
    # baseline was too thin.
    from types import SimpleNamespace

    from examples.qualification.inject import _timeout_reason

    def skipped_after(injected: dict[str, Any]) -> str:
        attempt = SimpleNamespace(injected=injected)
        return _timeout_reason(cast(Any, SimpleNamespace(attempts=[attempt])))

    dose = "dose_check_failed: 16 of 2021 busy step gaps in the baseline are ..."
    thin = "baseline_too_thin: 3 waiting counts of the 5 a hold needs"
    assert skipped_after({"recovery_blocked": [dose]}) == "dose_check_failed"
    assert skipped_after({"recovery_blocked": [thin, dose]}) == "baseline_too_thin"
    assert skipped_after({}) == "recovery_timeout"


def test_an_actuation_that_raises_is_published_not_actuated(tmp_path: Path) -> None:
    # F4a's target can't be stopped (a zombie): the pulser raises, the
    # episode is not actuated with the error kept, and the N before it and
    # the run are still published.
    zombie = subprocess.Popen([sys.executable, "-c", "pass"])
    zombie_pid = zombie.pid
    arguments = ["--step-seconds", "0.002", "--hook-dir", str(tmp_path / "hook")]
    try:
        assert wait_until(
            lambda: psutil.Process(zombie_pid).status() == psutil.STATUS_ZOMBIE
        )
        plan = _short_plan(tmp_path / "plan.json", {"type": "N"}, {"type": "F4a"})
        with FakeEngineProcess(arguments) as server:
            code = _inject(
                server, tmp_path, plan, "--target", f"engine_core={zombie_pid}"
            )
    finally:
        zombie.wait()
    assert code == 0
    run = tmp_path / "runs" / "q221-00000000000000bb"
    assert verify(run) == []
    by_type = {
        i.episode_type: i for i in load_injections(run / "truth" / "injections.jsonl")
    }
    assert by_type["N"].status == "valid"
    assert by_type["F4a"].status == "not_actuated"
    assert "did not stop" in by_type["F4a"].injected["error"]


def test_a_run_that_fails_is_published_with_its_reason(tmp_path: Path) -> None:
    # The victim exits before measuring (an argument infer profile refuses):
    # nothing is attempted, and the run says why, exit code 1.
    arguments = ["--step-seconds", "0.002", "--hook-dir", str(tmp_path / "hook")]
    plan = _short_plan(tmp_path / "plan.json", {"type": "N"})
    with FakeEngineProcess(arguments) as server:
        code = main(
            [
                "inject", "--plan", str(plan), "--out", str(tmp_path / "runs"),
                "--label", "q221-00000000000000cc", "--base-url", server.base_url,
                "--model", "fake/qwen-0.5b", "--reference-channel", str(tmp_path / "hook"),
                "--", "--no-such-flag",
            ]  # fmt: skip
        )
    assert code == 1
    run = tmp_path / "runs" / "q221-00000000000000cc"
    assert verify(run) == []
    record = load_run(run / "truth" / "run.json")
    assert record.protocol_failure is not None
    assert "before measuring" in record.protocol_failure
    # No victim clock: the run's times are the harness's, on this host.
    assert record.clock_domain == _harness_clock()
    (skipped,) = load_injections(run / "truth" / "injections.jsonl")
    assert skipped.status in ("protocol_failure", "not_actuated")


class _FlakyChannel:
    """A reference channel whose first poll raises."""

    def __init__(self) -> None:
        self.polls = 0

    def poll(self, *, scrape: bool = True) -> None:
        self.polls += 1
        if self.polls == 1:
            raise OSError("disk full")


def test_the_reference_poller_survives_a_failed_poll_and_stops_cleanly(
    tmp_path: Path,
) -> None:
    import threading
    import time

    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    directory = RunDirectory(tmp_path / "runs", "q221-x").create()
    run = InjectionRun(
        parse_plan(record), directory, Server("http://127.0.0.1:9", "m", tmp_path),
        poll_seconds=0.01,
    )  # fmt: skip
    channel = _FlakyChannel()
    run.channel = channel  # type: ignore[assignment]
    poller = threading.Thread(target=run._poll_loop, daemon=True)
    poller.start()
    assert wait_until(lambda: channel.polls >= 3)
    errors = (directory.probes / "poll-errors.jsonl").read_text()
    assert "disk full" in errors
    # Closing waits out any poll in progress; none follows it.
    run._close_channel()
    polls = channel.polls
    time.sleep(0.1)
    assert channel.polls == polls
    poller.join(timeout=5)
    assert not poller.is_alive()


def test_neighbor_names_say_nothing_about_the_episode_order() -> None:
    # The hook log a diagnosed configuration may import carries each
    # neighbor's request IDs; they mustn't give away which episode ran when
    # (C.4's blinding).
    from examples.qualification.inject import neighbor_name

    names = [neighbor_name("q221-0123456789abcdef", index) for index in range(3)]
    assert len(set(names)) == 3
    assert all(re.fullmatch(r"[0-9a-f]{12}", name) for name in names)
    assert names == [neighbor_name("q221-0123456789abcdef", i) for i in range(3)]
    assert neighbor_name("q221-fedcba9876543210", 0) != names[0]


def test_an_api_server_pulse_that_stalled_the_engine_adds_that_mechanism(
    tmp_path: Path,
) -> None:
    # The fake engine is one process, so pulsing its API server stops its
    # engine too. A.4: F4b stays realized, its realized set adds
    # host_stall@engine_core, and the label allows it.
    arguments = ["--step-seconds", "0.002", "--hook-dir", str(tmp_path / "hook")]
    plan = _short_plan(
        tmp_path / "plan.json",
        {"type": "F4b", "dose": {"pulse_ms": 100, "period_ms": 400}},
    )
    with FakeEngineProcess(arguments) as server:
        code = _inject(server, tmp_path, plan, "--target", f"api_server={server.pid}")
    assert code == 0
    run = tmp_path / "runs" / "q221-00000000000000bb"
    (f4b,) = load_injections(run / "truth" / "injections.jsonl")
    assert f4b.validity.realization == "realized", f4b.validity
    assert "host_stall@engine_core" in f4b.validity.realized_mechanisms
    assert ("host_stall", "engine_core") in {(a.kind, a.component) for a in f4b.allows}


def test_a_capture_records_its_start_for_i1s_realization(tmp_path: Path) -> None:
    # A1 realizes I1 only when the capture both started and stopped.
    from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan(
        {**record, "episodes": [{"type": "I1", "dose": {"seconds": 0.1}}]}
    )
    config = FakeEngineConfig(step_seconds=0.001, trace_dir=tmp_path / "traces")
    with FakeEngine(config) as engine:
        server = Server(engine.base_url, engine.config.model, tmp_path)
        run = InjectionRun(plan, RunDirectory(tmp_path / "runs", "q221-y"), server)
        actions, actuated, _injected = run._capture(plan.episodes[0])
    assert actuated
    assert actions.capture_started_ns is not None
    assert actions.stop_requested_ns is not None
    assert actions.capture_started_ns <= actions.stop_requested_ns


def test_a_twin_without_its_scraped_ratio_is_published_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A server whose /metrics has no prefix-cache counters: T3b's hit-ratio
    # check has nothing to judge. The twin stays valid on its victim check,
    # and its observation says the reference channel was incomplete.
    from examples.qualification import reference

    scrape = reference.scrape_metrics

    def without_counters(url: str, **kwargs: Any) -> reference.Scrape:
        taken = scrape(url, **kwargs)
        return replace(taken, prefix_queries=None, prefix_hits=None)

    monkeypatch.setattr(reference, "scrape_metrics", without_counters)
    dose = {"rate_per_second": 2, "input_tokens": 64, "output_tokens": 4}
    plan = _short_plan(tmp_path / "plan.json", {"type": "T3b", "dose": dose})
    with FakeEngineProcess(
        ["--step-seconds", "0.002", "--hook-dir", str(tmp_path / "hook")]
    ) as server:
        assert _inject(server, tmp_path, plan) == 0
    run = tmp_path / "runs" / "q221-00000000000000bb"
    (twin,) = load_injections(run / "truth" / "injections.jsonl")
    assert twin.status == "valid", twin.validity
    assert twin.validity.observation == "incomplete"
    (ratio,) = [c for c in twin.validity.checks if c["name"] == "engine_hit_ratio_fell"]
    assert ratio["incomplete"] is True


def _signal_mid_first_episode(
    tmp_path: Path, episodes: list[dict[str, Any]], signum: int, *targets: str
) -> Path:
    """Run ``episodes`` (6 s each) in their own session, signal the
    harness's process group inside the first one, and return the run."""
    hook = tmp_path / "hook"
    plan = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan["timeline"]["episode"] = 6
    plan["episodes"] = episodes
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    label = "q221-00000000000000cc"
    engine = ["--step-seconds", "0.002", "--hook-dir", str(hook)]
    with FakeEngineProcess(engine) as server:
        # fmt: off
        argv = [
            "inject", "--plan", str(tmp_path / "plan.json"),
            "--out", str(tmp_path / "runs"), "--label", label,
            "--base-url", server.base_url, "--model", "fake/qwen-0.5b",
            "--reference-channel", str(hook),
            *(arg.format(pid=server.pid) for arg in targets),
            "--", "--tokenizer", "none", "--system-sampler", "none",
        ]
        # fmt: on
        harness = subprocess.Popen(
            [sys.executable, "-m", "examples.qualification", *argv],
            env=_environment(),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        markers = tmp_path / "runs" / f".{label}.partial" / "probes" / "markers"
        assert wait_until(lambda: any(markers.glob("*measured_started*")), timeout=60)
        # Past priming and the baseline (5 s): inside the first episode.
        time.sleep(7)
        os.killpg(harness.pid, signum)
        assert harness.wait(timeout=120) == 128 + signum
        assert psutil.Process(server.pid).status() != psutil.STATUS_STOPPED
    return tmp_path / "runs" / label


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_a_run_without_pulses_signalled_mid_run_is_published(
    tmp_path: Path, signum: int
) -> None:
    # A job's SIGTERM or an ssh disconnect's SIGHUP during [N, N], before
    # any pulser exists: the run is still published, interrupted, with the
    # episode it attempted. The handlers used to come only with a pulser.
    run = _signal_mid_first_episode(tmp_path, [{"type": "N"}, {"type": "N"}], signum)
    assert verify(run) == []
    assert load_run(run / "truth" / "run.json").protocol_failure == "interrupted"
    attempted = load_injections(run / "truth" / "injections.jsonl")
    assert [injection.episode_type for injection in attempted] == ["N", "N"]


def _victim_of(harness: subprocess.Popen[bytes]) -> psutil.Process:
    """The harness's own victim process, from its process tree: found by
    label across the machine, a victim of the same case run elsewhere (the
    other interpreter's suite, at once) read as one left behind."""
    (victim,) = [
        child
        for child in psutil.Process(harness.pid).children(recursive=True)
        if "examples.qualification.victim" in " ".join(child.cmdline())
    ]
    return victim


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (signal.SIGTERM, signal.SIGTERM),
        (signal.SIGTERM, signal.SIGINT),
        # rev-220-a's final note: a job runner's SIGHUP, and an operator's
        # second Ctrl+C, pinned as well.
        (signal.SIGHUP, signal.SIGHUP),
        (signal.SIGINT, signal.SIGINT),
    ],
    ids=["term-term", "term-int", "hup-hup", "int-int"],
)
def test_a_second_signal_while_the_run_finishes_still_publishes_it(
    tmp_path: Path, first: int, second: int
) -> None:
    # rev-220-a's second A2 delta, D1: SIGTERM to the harness alone, then
    # another signal while it waited for its victim to drain. The second
    # one ended the harness there: no truth, the run left in .partial and
    # the victim still loading the engine. Now the second signal cuts the
    # drain short, and the run is published before the harness exits.
    hook = tmp_path / "hook"
    plan = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan["timeline"]["episode"] = 6
    plan["episodes"] = [{"type": "N"}, {"type": "N"}]
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    label = f"q221-{0xDD00 + 64 * first + second:016x}"  # cases may run at once
    engine = ["--step-seconds", "0.002", "--hook-dir", str(hook)]
    victim: psutil.Process | None = None
    try:
        with FakeEngineProcess(engine) as server:
            # fmt: off
            argv = [
                "inject", "--plan", str(tmp_path / "plan.json"),
                "--out", str(tmp_path / "runs"), "--label", label,
                "--base-url", server.base_url, "--model", "fake/qwen-0.5b",
                "--reference-channel", str(hook),
                "--", "--tokenizer", "none", "--system-sampler", "none",
            ]
            # fmt: on
            harness = subprocess.Popen(
                [sys.executable, "-m", "examples.qualification", *argv],
                env=_environment(),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            partial = tmp_path / "runs" / f".{label}.partial"
            markers = partial / "probes" / "markers"
            assert wait_until(
                lambda: any(markers.glob("*measured_started*")), timeout=60
            )
            victim = _victim_of(harness)
            time.sleep(7)  # inside the first episode
            # A held engine: the victim's requests sent from now on can't
            # finish (about ten in the next second), so its drain lasts as
            # long as a real victim's long outputs would.
            assert post(f"{server.base_url}/_fault/pause?target=engine")[0] == 200
            time.sleep(1)
            os.kill(harness.pid, first)  # the harness alone
            time.sleep(1)
            assert harness.poll() is None  # still waiting for the victim
            os.kill(harness.pid, second)
            code = harness.wait(timeout=120)
            # close-221-delta, N8: the harness's own victim, by its pid and
            # start time; matched by label across the machine, the same
            # case's victim in another suite running at once was "left".
            left = victim.is_running()
            post(f"{server.base_url}/_fault/resume?target=engine")
    finally:
        if victim is not None and victim.is_running():
            victim.kill()
    assert code == 128 + first  # it exits as the first signal would have
    assert not left
    run = tmp_path / "runs" / label
    assert not partial.exists() and verify(run) == []
    assert load_run(run / "truth" / "run.json").protocol_failure == "interrupted"


def test_a_third_signal_is_not_held() -> None:
    # rev-220-b's delta-3 closure, G5: once the second signal had killed the
    # victim, every further one was only noted, so a publish that hung (a
    # full disk) could be ended only by SIGKILL. The third goes to the
    # handler held before, as if nothing had been held.
    from examples.qualification.inject import _HeldSignals

    seen: list[int] = []
    previous = signal.signal(signal.SIGHUP, lambda signum, _frame: seen.append(signum))
    try:
        with _HeldSignals() as held:
            held.holding = True
            signal.raise_signal(signal.SIGHUP)
            signal.raise_signal(signal.SIGHUP)
            assert (held.received, seen) == ([signal.SIGHUP] * 2, [])
            signal.raise_signal(signal.SIGHUP)
            assert seen == [signal.SIGHUP]
            assert not held.installed
    finally:
        signal.signal(signal.SIGHUP, previous)


def test_a_signal_ignored_before_the_run_stays_ignored() -> None:
    # close-221-delta, N6: _pass_on's SIG_IGN branch was unpinned. A signal
    # the process ignored before the run is ignored while it is passed on:
    # it neither ends the run nor turns holding on, and the holder stays.
    from examples.qualification.inject import _HeldSignals

    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with _HeldSignals() as held:
            signal.raise_signal(signal.SIGHUP)
            assert (held.holding, held.received, held.installed) == (False, [], True)
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)


def test_a_holder_cut_short_while_installing_installs_the_rest_next_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # close-221-delta, N1: a signal between two installs left __enter__
    # with SIGTERM held and the rest not; the finish's entry then returned
    # at once (something was installed), so a Ctrl+C during the publish
    # was not held. Entering again installs what isn't yet.
    from examples.qualification import inject as module

    real = signal.signal
    saved = {signum: signal.getsignal(signum) for signum in module._HeldSignals.SIGNALS}
    cut = [True]

    def cut_after_sigterm(signum: int, handler: Any) -> Any:
        if signum == signal.SIGHUP and cut[0]:
            cut[0] = False
            raise KeyboardInterrupt  # a Ctrl+C between two installs
        return real(signum, handler)

    monkeypatch.setattr(module.signal, "signal", cut_after_sigterm)
    held = module._HeldSignals()
    try:
        with pytest.raises(KeyboardInterrupt):
            held.__enter__()
        assert signal.getsignal(signal.SIGINT) is saved[signal.SIGINT]
        held.__enter__()
        assert all(signal.getsignal(s) == held._note for s in held.SIGNALS)
        held.__exit__()
        assert {s: signal.getsignal(s) for s in saved} == saved
    finally:
        for signum, handler in saved.items():
            real(signum, handler)


def test_a_signal_while_the_run_sets_up_still_publishes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # close-221-delta, N1's pre-existing half: a signal after the run's
    # directory was made but before its try (the plan written, the channel
    # and poller starting) left the run in .partial. The set-up is inside
    # the try now: the run is published, interrupted, with every episode
    # skipped, and the interruption goes on.
    from examples.qualification import pulser
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan({**record, "episodes": [{"type": "N"}]})
    directory = RunDirectory(tmp_path / "runs", "q221-s")
    run = InjectionRun(plan, directory, Server("", "m", tmp_path, {}))

    def interrupted() -> None:
        raise KeyboardInterrupt  # a Ctrl+C as the channel starts

    saved = {
        s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    }
    monkeypatch.setattr(run, "_channel", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            run.execute()
    finally:
        for saved_signum, handler in saved.items():
            signal.signal(saved_signum, handler)
        pulser._HANDLED.clear()
    assert not directory.partial.exists() and verify(directory.final) == []
    assert load_run(directory.final / "truth" / "run.json").protocol_failure == (
        "interrupted"
    )
    (skipped,) = load_injections(directory.final / "truth" / "injections.jsonl")
    assert skipped.injected["skipped"] == "run_ended"


def test_a_signal_before_holding_is_passed_on() -> None:
    # Installed for the whole run, the handlers act as the ones they
    # replaced until the run's end sets holding: mid-run, a signal still
    # ends the run (and the run is published on the way out).
    from examples.qualification.inject import _HeldSignals

    seen: list[int] = []
    previous = signal.signal(signal.SIGHUP, lambda signum, _frame: seen.append(signum))
    try:
        with _HeldSignals() as held:
            signal.raise_signal(signal.SIGHUP)
            assert (held.received, seen) == ([], [signal.SIGHUP])
            with pytest.raises(KeyboardInterrupt):
                signal.raise_signal(signal.SIGINT)
            assert held.installed
    finally:
        signal.signal(signal.SIGHUP, previous)


def test_a_handler_installed_from_c_is_restored_as_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # G5: signal.signal returns None for a handler installed from C, and
    # __exit__ skipped it, leaving the note-taker installed for good.
    from examples.qualification import inject as module

    real = signal.signal
    saved = {signum: signal.getsignal(signum) for signum in module._HeldSignals.SIGNALS}

    def from_c(signum: int, handler: Any) -> Any:
        before = real(signum, handler)
        return None if signum == signal.SIGHUP and handler != signal.SIG_DFL else before

    monkeypatch.setattr(module.signal, "signal", from_c)
    try:
        with module._HeldSignals():
            pass
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
    finally:
        for signum, handler in saved.items():
            real(signum, handler)


def test_the_run_holds_signals_before_it_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # G5: on the normal path a signal that landed between the episodes'
    # end and _finish holding them raised out of execute() with nothing
    # published. Signals are now held inside the try, before _finish runs.
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan({**record, "episodes": [{"type": "N"}]})
    run = InjectionRun(
        plan, RunDirectory(tmp_path / "runs", "q221-w"), Server("", "m", tmp_path, {})
    )
    holding: list[bool] = []

    def finish(victim: Any, poller: Any, progress: Any, held: Any) -> tuple[Path, None]:
        holding.append(held.holding)
        held.__exit__()
        return tmp_path, None

    monkeypatch.setattr(run, "_channel", lambda: None)
    monkeypatch.setattr(run, "_poll_loop", lambda: None)
    monkeypatch.setattr(run, "_start_victim", lambda: None)
    monkeypatch.setattr(run, "_episodes", lambda victim, progress: None)
    monkeypatch.setattr(run, "_finish", finish)
    run.execute()
    assert holding == [True]


@pytest.mark.parametrize("first", ["run_failure", "keyboard_interrupt"])
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_on_the_way_to_the_finish_still_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str, signum: int
) -> None:
    # Astra's closure of delta 3, H6: after the run's own failure, or a
    # first Ctrl+C, signals were held only once _finish entered them, so a
    # signal landing in between (a double Ctrl+C) raised out of execute()
    # with nothing published. It lands here at _finish's entry, the last
    # moment of that window: the run is still published, then ends as the
    # first cause, or the held signal, says.
    from examples.qualification import pulser
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan({**record, "episodes": [{"type": "N"}]})
    run = InjectionRun(
        plan, RunDirectory(tmp_path / "runs", "q221-v"), Server("", "m", tmp_path, {})
    )
    published: list[bool] = []

    def finish(victim: Any, poller: Any, progress: Any, held: Any) -> Any:
        signal.raise_signal(signum)
        with held:
            published.append(True)
        return tmp_path, held.received[0] if held.received else None

    def episodes(victim: Any, progress: Any) -> None:
        if first == "run_failure":
            raise RuntimeError("engine died")
        raise KeyboardInterrupt

    saved = {
        s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    }
    monkeypatch.setattr(run, "_channel", lambda: None)
    monkeypatch.setattr(run, "_poll_loop", lambda: None)
    monkeypatch.setattr(run, "_start_victim", lambda: None)
    monkeypatch.setattr(run, "_episodes", episodes)
    monkeypatch.setattr(run, "_finish", finish)
    expected = {
        "keyboard_interrupt": KeyboardInterrupt,
        "run_failure": KeyboardInterrupt if signum == signal.SIGINT else SystemExit,
    }[first]
    try:
        with pytest.raises(expected):
            run.execute()
    finally:
        for saved_signum, handler in saved.items():
            signal.signal(saved_signum, handler)
        pulser._HANDLED.clear()
    assert published == [True]


@pytest.mark.parametrize("first", ["run_failure", "keyboard_interrupt"])
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_at_the_last_check_before_the_runs_except_still_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str, signum: int
) -> None:
    # The lead's proof for Astra's H6: the run's end is raised from deep in
    # the episodes, and a signal is made pending in the last __exit__ the
    # unwind runs before execute's except block, the last place CPython can
    # run a handler before it (from 3.11; see execute). After a failure the
    # signal raises into the except and the run is published interrupted;
    # after a first Ctrl+C it is held, the first goes on, and the run is
    # published. Run on 3.11 and 3.12 as well as 3.10.
    from examples.qualification import pulser
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan({**record, "episodes": [{"type": "N"}]})
    run = InjectionRun(
        plan, RunDirectory(tmp_path / "runs", "q221-u"), Server("", "m", tmp_path, {})
    )
    seen: dict[str, Any] = {}

    class LastExit:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> None:
            signal.raise_signal(signum)

    def deep(depth: int) -> None:
        if depth:
            deep(depth - 1)
        elif first == "run_failure":
            raise RuntimeError("engine died")
        else:
            signal.raise_signal(signal.SIGINT)  # a first Ctrl+C, deep in the run

    def episodes(victim: Any, progress: Any) -> None:
        with LastExit():
            deep(8)

    def finish(victim: Any, poller: Any, progress: Any, held: Any) -> Any:
        with held:
            seen.update(failure=progress.failure, received=list(held.received))
        return tmp_path, None

    saved = {
        s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    }
    monkeypatch.setattr(run, "_channel", lambda: None)
    monkeypatch.setattr(run, "_poll_loop", lambda: None)
    monkeypatch.setattr(run, "_start_victim", lambda: None)
    monkeypatch.setattr(run, "_episodes", episodes)
    monkeypatch.setattr(run, "_finish", finish)
    held_first = first == "keyboard_interrupt"
    raised = KeyboardInterrupt if held_first or signum == signal.SIGINT else SystemExit
    try:
        with pytest.raises(raised):
            run.execute()
    finally:
        for saved_signum, handler in saved.items():
            signal.signal(saved_signum, handler)
        pulser._HANDLED.clear()
    assert seen["failure"] == "interrupted"
    assert seen["received"] == ([signum] if held_first else [])


# Instructions that run no Python code and don't check for pending signals
# (POP_TOP drops the caught exception, which the handler still holds).
_NO_HANDLER_RUNS = {
    "PUSH_EXC_INFO", "LOAD_GLOBAL", "CHECK_EXC_MATCH", "POP_JUMP_IF_FALSE",
    "POP_JUMP_FORWARD_IF_FALSE", "POP_TOP", "STORE_FAST", "LOAD_CONST",
    "LOAD_FAST", "STORE_ATTR", "NOP", "CACHE",
}  # fmt: skip


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="3.10 checks for signals entering an except"
)
def test_no_signal_handler_can_run_between_the_runs_except_and_holding() -> None:
    # The other half of H6's proof: from each of execute's except clauses'
    # first instruction to its store of ``holding``, the bytecode holds
    # nothing CPython checks for pending signals at (no call, no backward
    # jump, no RESUME), and the store itself runs no Python code (no
    # __setattr__, no descriptor). A signal is handled before an except,
    # and raises into it, or after the store, and is held.
    import dis

    from examples.qualification.inject import InjectionRun, _HeldSignals

    code = list(dis.get_instructions(InjectionRun.execute))
    handlers = []
    for entry, ins in enumerate(code):
        if ins.opname != "PUSH_EXC_INFO":
            continue
        store = next(
            index
            for index in range(entry, len(code))
            if code[index].opname == "STORE_ATTR" and code[index].argval == "holding"
        )
        names = [c.opname for c in code[entry : store + 1]]
        if "CHECK_EXC_MATCH" in names:
            handlers.append(names)
    assert len(handlers) == 2  # the run's except, and the one around it
    for names in handlers:
        assert set(names) <= _NO_HANDLER_RUNS, names
    assert _HeldSignals.__setattr__ is object.__setattr__
    assert "holding" not in vars(_HeldSignals)


def _fail_with_signals_pending(*signums: int) -> None:
    """Raise TypeError from one C call that has just set ``signums``
    pending, with no Python code run and no check made in between: ctypes
    calls PyErr_SetInterruptEx for the first, each errcheck (a C partial of
    the next such call) sets the next, and the last errcheck, the C builtin
    pow, raises. CPython runs their handlers at its next checks, which on
    3.10 include the entry to an except block."""
    import ctypes
    import functools

    pointer: Any = type(ctypes.pythonapi.PyErr_SetInterruptEx)

    def setter(extra: int) -> Any:
        function = pointer(("PyErr_SetInterruptEx", ctypes.pythonapi))
        function.argtypes = [ctypes.c_int] + [ctypes.py_object] * extra
        function.restype = ctypes.c_int
        return function

    errcheck: Any = pow
    for signum in reversed(signums[1:]):
        chained = setter(3)  # called as errcheck: (signum, result, func, args)
        chained.errcheck = errcheck
        errcheck = functools.partial(chained, signum)
    first = setter(0)
    first.errcheck = errcheck
    first(signums[0])


@pytest.mark.parametrize(
    "pending", [(signal.SIGINT,), (signal.SIGINT, signal.SIGTERM)], ids=["one", "two"]
)
def test_a_run_that_fails_with_signals_pending_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pending: tuple[int, ...]
) -> None:
    # The lead's 3.10 question on H6: the run fails in a C call with SIGINT
    # (and SIGTERM) left pending. On 3.10, CPython runs the first handler
    # as the run's except is entered, so the KeyboardInterrupt skipped it
    # and left the run unpublished; it now lands in the except around it,
    # and the second signal, handled as that one is entered, is held. From
    # 3.11 both are handled after the except has set holding, and held.
    from examples.qualification import pulser
    from examples.qualification.inject import InjectionRun, Server
    from examples.qualification.plan import parse_plan
    from examples.qualification.run_dir import RunDirectory

    record = json.loads(_plan(tmp_path / "plan.json").read_text())
    plan = parse_plan({**record, "episodes": [{"type": "N"}]})
    run = InjectionRun(
        plan, RunDirectory(tmp_path / "runs", "q221-t"), Server("", "m", tmp_path, {})
    )
    seen: dict[str, Any] = {}

    def finish(victim: Any, poller: Any, progress: Any, held: Any) -> Any:
        with held:
            seen.update(failure=progress.failure, received=list(held.received))
        return tmp_path, held.received[0] if held.received else None

    saved = {
        s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    }
    monkeypatch.setattr(run, "_channel", lambda: None)
    monkeypatch.setattr(run, "_poll_loop", lambda: None)
    monkeypatch.setattr(run, "_start_victim", lambda: None)
    monkeypatch.setattr(
        run, "_episodes", lambda victim, progress: _fail_with_signals_pending(*pending)
    )
    monkeypatch.setattr(run, "_finish", finish)
    try:
        with pytest.raises(KeyboardInterrupt):
            run.execute()
    finally:
        for saved_signum, handler in saved.items():
            signal.signal(saved_signum, handler)
        pulser._HANDLED.clear()
    if sys.version_info < (3, 11):
        assert seen == {"failure": "interrupted", "received": list(pending[1:])}
    else:
        assert seen["failure"].startswith("run_failed: TypeError")
        assert seen["received"] == list(pending)


def test_an_episode_interrupted_mid_pulse_records_its_pulses(tmp_path: Path) -> None:
    # SIGTERM during F4a's pulses: the engine was stopped several times, so
    # the truth says so, with each completed pulse, and that it was cut
    # short. The next episode is skipped because the run ended.
    episodes: list[dict[str, Any]] = [
        {"type": "F4a", "dose": {"pulse_ms": 100, "period_ms": 400}},
        {"type": "N"},
    ]
    run = _signal_mid_first_episode(
        tmp_path, episodes, signal.SIGTERM, "--target", "engine_core={pid}"
    )
    assert verify(run) == []
    stall, null = load_injections(run / "truth" / "injections.jsonl")
    assert stall.status == "not_actuated"
    assert stall.validity.actuation == "interrupted"
    assert stall.injected["interrupted"] is True
    assert len(stall.injected["pulses"]) >= 1
    # Fable's lens-a closure, K5: an interrupted episode's pulses say where
    # they landed in the step loop, as a whole episode's do.
    assert all("landed" in pulse for pulse in stall.injected["pulses"])
    assert null.injected == {"method": "none", "skipped": "run_ended"}


def test_pulses_delivered_before_a_failing_one_are_on_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fable's A2 delta N6: pulse 3 of 5 fails to confirm its stop; the two
    # real stops before it are in the hook log, and now in the truth too,
    # beside the error.
    from examples.qualification import pulser as pulser_module

    real_pulse = pulser_module.Pulser.pulse
    calls = [0]

    def third_fails(self: Any, seconds: float, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == 3:
            raise pulser_module.PulseRefused("did not stop within 1 s")
        return real_pulse(self, seconds, **kwargs)

    monkeypatch.setattr(pulser_module.Pulser, "pulse", third_fails)
    dose = {"pulse_ms": 100, "period_ms": 400}
    plan = _short_plan(tmp_path / "plan.json", {"type": "F4a", "dose": dose})
    engine = ["--step-seconds", "0.002", "--hook-dir", str(tmp_path / "hook")]
    with FakeEngineProcess(engine) as server:
        assert (
            _inject(server, tmp_path, plan, "--target", f"engine_core={server.pid}")
            == 0
        )
    run = tmp_path / "runs" / "q221-00000000000000bb"
    (stall,) = load_injections(run / "truth" / "injections.jsonl")
    assert stall.status == "not_actuated"
    assert "did not stop" in stall.injected["error"]
    assert len(stall.injected["pulses"]) == 2
    assert all("landed" in pulse for pulse in stall.injected["pulses"])
