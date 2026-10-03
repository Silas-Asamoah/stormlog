"""An injection run end to end, against the fake vLLM engine as its own
process: priming, baseline, three episodes with recovery, and the truth."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from examples.qualification.__main__ import main
from examples.qualification.fake_engine.process import FakeEngineProcess, _environment
from examples.qualification.run_dir import verify
from stormlog.infer.qualify.ground_truth import load_injections

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
            "recovery_timeout": 8,
            "final_recovery": 1,
            "min_clean": 0.5,
        },
        "thresholds": {"window": 1, "hold": 1, "cadence_hold": 1, "priming_window": 1},
        "episodes": [
            {"type": "N"},
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
    assert sorted(injections) == ["F2", "F4a", "N"]
    for injection in injections.values():
        assert injection.times.priming_check == {
            "passed": True,
            "cached_fraction_median": pytest.approx(1.0),
        }
    stall = injections["F4a"]
    assert stall.status == "valid", stall.validity
    assert stall.injected["pulses"]
    assert stall.times.effect_onset_ns == stall.times.action_onset_ns
    assert injections["N"].status == "valid"
    kv = injections["F2"]
    assert kv.validity.actuation == "ok"
    assert kv.status in ("valid", "not_realized")
    assert (run / "run" / "victim.jsonl").exists()
    assert (run / "truth" / "reference" / "scrapes.jsonl").exists()
    assert (run / "probes" / "hook-firstseen.jsonl").exists()


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
