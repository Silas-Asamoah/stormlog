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
from typing import Any, Callable

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
    # The run record: its label, its windows and the victim's clock, which
    # every episode shares; the victim ran under the same label.
    record = load_run(run / "truth" / "run.json")
    assert record.run_id == "q221-0123456789abcdef"
    assert record.clock_domain is not None
    assert record.priming is not None and record.final_recovery is not None
    assert record.measured.start_ns == record.priming.start_ns
    assert record.final_recovery.end_ns == record.measured.end_ns
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
    # its one check then fails, nothing else.
    if twin.status != "valid":
        twin_checks = {c["name"]: c["passed"] for c in twin.validity.checks}
        assert twin_checks == {"waits_within_baseline": False}, twin.validity


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


def _victims_of(label: str) -> list[psutil.Process]:
    found = []
    for process in psutil.process_iter(["cmdline"]):
        command = " ".join(process.info["cmdline"] or [])
        if "examples.qualification.victim" in command and label in command:
            found.append(process)
    return found


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
            left = [process.pid for process in _victims_of(label)]
            post(f"{server.base_url}/_fault/resume?target=engine")
    finally:
        for process in _victims_of(label):
            process.kill()
    assert code == 128 + first  # it exits as the first signal would have
    assert left == []
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
            signal.raise_signal(signal.SIGHUP)
            signal.raise_signal(signal.SIGHUP)
            assert (held.received, seen) == ([signal.SIGHUP] * 2, [])
            signal.raise_signal(signal.SIGHUP)
            assert seen == [signal.SIGHUP]
            assert not held.holding
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
