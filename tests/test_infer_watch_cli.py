"""``stormlog infer watch`` on the command line: exit codes and signals."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from stormlog.infer.cli import main as infer_main
from tests.watch_test_helpers import FakeMetrics, serve_metrics, watch_config

REPO = Path(__file__).resolve().parents[1]


def _watch(tmp_path: Path, *extra: str) -> int:
    return infer_main(["watch", "--root", str(tmp_path / "watch"), *extra])


def test_an_unreadable_config_exits_five(tmp_path: Path) -> None:
    assert _watch(tmp_path, "--config", str(tmp_path / "absent.json")) == 5
    garbled = tmp_path / "garbled.json"
    garbled.write_text("{", encoding="utf-8")
    assert _watch(tmp_path, "--config", str(garbled)) == 5


def test_an_unusable_config_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "watch.json"
    path.write_text(json.dumps(watch_config("http://x", surprise=1)), encoding="utf-8")
    assert _watch(tmp_path, "--config", str(path)) == 2


def test_a_second_watcher_on_a_root_exits_two(tmp_path: Path) -> None:
    from stormlog.infer.watch.config import resolve_watch_config
    from stormlog.infer.watch.watcher import Watcher

    config = resolve_watch_config(watch_config("http://127.0.0.1:9"))
    first = Watcher(config, tmp_path / "watch")
    try:
        assert _watch(tmp_path, "--base-url", "http://127.0.0.1:9") == 2
    finally:
        first.close()


@pytest.mark.parametrize(
    "extra",
    [
        (),  # no server
        ("--base-url", "http://127.0.0.1:9", "--test-trigger", "sometimes"),
        ("--base-url", "http://127.0.0.1:9", "--test-trigger", "every=0"),
        ("--base-url", "http://127.0.0.1:9", "--duration", "0"),
        ("--base-url", "http://127.0.0.1:9", "--interval", "-1"),
        ("--base-url", "http://127.0.0.1:9", "--api-key-env", "STORMLOG_UNSET_KEY"),
        # JSON and float() accept these; each crashed with exit 1, no report.
        ("--base-url", "http://127.0.0.1:9", "--duration", "inf"),
        ("--base-url", "http://127.0.0.1:9", "--duration", "nan"),
        ("--base-url", "http://127.0.0.1:9", "--interval", "nan"),
        ("--base-url", "http://127.0.0.1:9", "--interval", "1e-12"),
        ("--base-url", "http://127.0.0.1:9", "--test-trigger", "every=inf"),
        ("--base-url", "ftp://127.0.0.1:9"),
        ("--base-url", "not a url"),
    ],
)
def test_bad_arguments_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]
) -> None:
    monkeypatch.delenv("STORMLOG_UNSET_KEY", raising=False)
    assert _watch(tmp_path, *extra) == 2


def test_a_watch_of_a_quiet_server_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "watch.json"
    with serve_metrics(FakeMetrics()) as base_url:
        path.write_text(json.dumps(watch_config(base_url)), encoding="utf-8")
        code = _watch(tmp_path, "--config", str(path), "--duration", "0.5")
    assert code == 0
    report = tmp_path / "watch" / "report.json"
    assert f"Watch report: {report}" in capsys.readouterr().out
    assert json.loads(report.read_text())["verdict"]["exit_code"] == 0


def test_sigterm_ends_the_watch_with_a_report(tmp_path: Path) -> None:
    root = tmp_path / "watch"
    ready = tmp_path / "ready"
    with serve_metrics(FakeMetrics()) as base_url:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "stormlog.entrypoint",
                "infer",
                "watch",
                "--root",
                str(root),
                "--base-url",
                base_url,
                "--interval",
                "0.1",
                "--ready-file",
                str(ready),
            ],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 60
            while not ready.exists():
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "the watcher never became ready"
                time.sleep(0.05)
            process.send_signal(signal.SIGTERM)
            out, err = process.communicate(timeout=60)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert process.returncode == 0, err
    assert "Watch report:" in out
    report = json.loads((root / "report.json").read_text())
    assert report["verdict"]["exit_code"] == 0
    assert report["metrics"]["scrapes_ok"] >= 1


def test_sigterm_ends_a_watch_whose_scrape_trickles(tmp_path: Path) -> None:
    """A /metrics sending a byte every 0.2 s kept the watch past SIGTERM
    until it was killed, losing its report."""
    root = tmp_path / "watch"
    metrics = FakeMetrics()
    metrics.dribble = 0.2
    with serve_metrics(metrics) as base_url:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "stormlog.entrypoint",
                "infer",
                "watch",
                "--root",
                str(root),
                "--base-url",
                base_url,
                "--interval",
                "0.5",
            ],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 60
            while metrics.scrapes == 0:
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "the watcher never scraped"
                time.sleep(0.05)
            time.sleep(1.0)  # inside a trickling scrape, or past one
            process.send_signal(signal.SIGTERM)
            out, err = process.communicate(timeout=15)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert process.returncode == 1, err  # no scrape ever finished
    report = json.loads((root / "report.json").read_text())
    assert report["payload"]["unsound"] == ["no_successful_scrape"]
