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
from typing import Any

import psutil
import pytest

from examples.qualification.__main__ import main
from examples.qualification.fake_engine.process import FakeEngineProcess, _environment
from examples.qualification.run_dir import verify
from stormlog.infer.qualify.ground_truth import load_injections, load_run
from tests.qualification_fake_engine_helpers import wait_until

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
