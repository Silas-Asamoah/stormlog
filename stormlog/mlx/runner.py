"""Execute a Python script under passive tracking in the same interpreter."""

from __future__ import annotations

import runpy
import sys
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from stormlog.exit_codes import ExitCode

from .artifacts import CaptureBundle, CaptureOptions, Outcome, capture_failure
from .oom import classify_oom_exception
from .runtime import Runtime


@contextmanager
def script_environment(script: Path, args: Sequence[str]) -> Iterator[None]:
    """Mirror script argv/import paths, retaining the invocation cwd."""
    original_argv, original_path = sys.argv, sys.path
    sys.argv = [str(script), *args]
    sys.path = [str(script.parent), *original_path[1:]]
    try:
        yield
    finally:
        sys.argv, sys.path = original_argv, original_path


def workload_failure(exc: BaseException) -> Outcome:
    details: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }
    code = _failure_code(exc)
    if isinstance(exc, SystemExit):
        value = exc.code
        details["system_exit"] = (
            value if value is None or isinstance(value, (str, int)) else repr(value)
        )
    details["oom_occurred"] = classify_oom_exception(exc).is_oom
    details["status"] = "completed" if code == ExitCode.OK else "failed"
    return Outcome(
        int(code),
        None if code == ExitCode.OK else str(exc) or type(exc).__name__,
        details,
    )


def _failure_code(exc: BaseException) -> ExitCode:
    if isinstance(exc, SystemExit):
        return ExitCode.OK if exc.code is None or exc.code == 0 else ExitCode.ERROR
    if isinstance(exc, KeyboardInterrupt):
        return ExitCode.INTERRUPTED
    return ExitCode.FINDINGS if classify_oom_exception(exc).is_oom else ExitCode.ERROR


def run_in_bundle(
    bundle: CaptureBundle,
    call: Callable[[], Any],
    *,
    workload: Mapping[str, Any],
    runtime: Runtime | None = None,
    context: str = "script",
) -> Outcome:
    """Run ``call`` in this interpreter under ``bundle``'s passive capture.

    Capture starts before ``call`` and is finalized in ``finally``. The
    workload's own failure is recorded and re-reported, never turned into a
    passing result; the exception itself is not re-raised, so callers read
    the returned outcome.
    """
    outcome = Outcome(workload=dict(workload))
    try:
        bundle.start(runtime)
    except BaseException as exc:
        failure = capture_failure(exc)
        failure.workload = outcome.workload
        return bundle.finish(failure)
    try:
        call()
        outcome.workload["status"] = "completed"
    except BaseException as exc:
        failure = workload_failure(exc)
        failure.workload = {**outcome.workload, **failure.workload}
        outcome = failure
        if bundle.tracker is not None:
            with suppress(BaseException):
                outcome.oom_dump = bundle.tracker.record_exception(exc, context=context)
    finally:
        bundle.finish(outcome)
    return outcome


def run_script(
    script: str | Path,
    args: Sequence[str] = (),
    *,
    output: str | Path | None,
    options: CaptureOptions | None = None,
    runtime: Runtime | None = None,
) -> tuple[Path, Outcome]:
    bundle = CaptureBundle(output, "run", options or CaptureOptions())
    path = Path(script).resolve()
    outcome = Outcome(
        workload={"script": str(path), "argv": list(args), "status": "not_started"}
    )
    try:
        if not path.is_file():
            raise OSError(f"Script is not a readable file: {path}")
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        outcome.exit_code, outcome.error = int(ExitCode.INVALID_INPUT), str(exc)
        return bundle.directory, bundle.finish(outcome)

    def execute() -> None:
        with script_environment(path, args):
            runpy.run_path(str(path), run_name="__main__")

    outcome = run_in_bundle(bundle, execute, workload=outcome.workload, runtime=runtime)
    return bundle.directory, outcome
