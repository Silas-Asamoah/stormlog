"""Inference command dispatch and help do not require Unix watch locking."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _infer_without_fcntl(*argv: str) -> subprocess.CompletedProcess[str]:
    code = """
        import sys

        # On Windows this module is absent. Block it before any CLI import
        # in a fresh process so a previously loaded watcher cannot mask it.
        sys.modules["fcntl"] = None

        from stormlog.entrypoint import main

        raise SystemExit(main(["infer", *sys.argv[1:]]))
        """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *argv],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("--help",),
        ("profile", "--help"),
        ("analyze", "--help"),
        ("collect-server", "--help"),
        ("import-trace", "--help"),
        ("import-execution", "--help"),
        ("watch", "--help"),
    ],
)
def test_inference_help_without_fcntl(argv: tuple[str, ...]) -> None:
    result = _infer_without_fcntl(*argv)

    assert result.returncode == 0, result.stderr
    assert "usage: stormlog infer" in result.stdout


def test_inference_analysis_without_fcntl(tmp_path: Path) -> None:
    artifact = tmp_path / "infer.jsonl"
    artifact.write_text(
        json.dumps(
            {
                "event_type": "infer.request",
                "phase": "measured",
                "case_id": "c1",
                "status": "ok",
                "started_at_ns": 0,
                "ended_at_ns": 100_000_000,
                "e2e_latency_ms": 100.0,
                "output_tokens": 1,
                "total_tokens": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _infer_without_fcntl("analyze", str(artifact), "--format", "json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["summary"]["total_requests"] == 1
    assert "c1" in report["cases"]
