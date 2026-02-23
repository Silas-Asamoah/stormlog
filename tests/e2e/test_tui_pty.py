import os
import shutil

import pytest

pytest.importorskip("pexpect")
import pexpect  # type: ignore[import-untyped, unused-ignore]

pytestmark = [
    pytest.mark.tui_pty,
    pytest.mark.skipif(os.name == "nt", reason="PTY tests require a POSIX terminal."),
]


def _spawn_tui() -> pexpect.spawn:
    executable = shutil.which("gpu-profiler")
    if executable is None:
        pytest.skip("gpu-profiler entrypoint is unavailable in this environment.")

    env = os.environ.copy()
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("COLORTERM", "truecolor")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("PYTHONUNBUFFERED", "1")

    return pexpect.spawn(
        executable,
        env=env,
        encoding="utf-8",
        timeout=20,
        dimensions=(40, 120),
    )


def _expect_with_retry(
    child: pexpect.spawn,
    pattern: object,
    *,
    attempts: int = 3,
    timeout: float = 4.0,
) -> None:
    last_error = None
    for _ in range(attempts):
        try:
            child.expect(pattern, timeout=timeout)
            return
        except pexpect.TIMEOUT as error:
            last_error = error

    tail = (child.before or "")[-1000:]
    raise AssertionError(
        f"Timed out waiting for pattern {pattern!r}. Last output tail:\n{tail}"
    ) from last_error


def _assert_still_running(child: pexpect.spawn, *, timeout: float = 1.0) -> None:
    try:
        child.expect(pexpect.EOF, timeout=timeout)
    except pexpect.TIMEOUT:
        return
    raise AssertionError("gpu-profiler exited unexpectedly before quit key.")


def test_gpu_profiler_pty_smoke_start_interact_quit() -> None:
    child = _spawn_tui()
    try:
        _expect_with_retry(child, "Refresh Overview", attempts=4, timeout=4)
        _expect_with_retry(child, "Focus Log", attempts=2, timeout=2)
        _expect_with_retry(child, "Quit", attempts=2, timeout=2)

        for key in ("r", "f"):
            child.send(key)
            _assert_still_running(child)

        child.send("q")
        _expect_with_retry(child, pexpect.EOF, attempts=4, timeout=5)
        child.close()

        assert child.exitstatus == 0
        assert child.signalstatus is None
    finally:
        if child.isalive():
            child.send("q")
            try:
                child.expect(pexpect.EOF, timeout=2)
            except pexpect.ExceptionPexpect:
                child.terminate(force=True)
            child.close()
