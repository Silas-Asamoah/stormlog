"""The out-of-process analyzer and its cache (design §2.1, §8.1; decision 12).

Studio shows the analyzer report's observers, SLO and server blocks beside
its own numbers. ``analyze_inference_events`` loads the whole file as dicts
(``analysis.py:88,743-758``), so :class:`AnalyzerRunner` runs
``stormlog infer analyze --format json`` as a subprocess: its memory goes
back when it exits, and it can be killed.

Rules:

- **One at a time.** One worker thread runs the queued analyses in turn,
  one subprocess each. A newer request for the same path replaces a queued
  older one.
- **Size cap.** Artifacts over ``Limits.analysis_bytes`` (128 MiB) are
  ``unavailable`` with ``over_size_cap``; nothing runs.
- **When.** Only for an index snapshot that is ``ready`` or ``partial`` and
  not ``growing``: a building index is ``pending`` (``index_building``), a
  growing file ``unavailable`` (``growing``), a failed index ``unavailable``
  with the index's failure. A snapshot of the same index once the file stops
  growing starts the analysis.
- **Truncated last line (decision 12).** ``infer analyze`` fails the whole
  file on one bad line (``correlation_codec.py:270-273``), which hits exactly
  the killed runs people investigate. When the file does not end with a
  newline, the analysis covers its complete lines only; the result says so
  (``input.kind: complete_line_prefix``, the bytes left out).
- **Verified bytes only.** The subprocess never opens the artifact's path,
  which a writer can grow, rewrite or replace (or a directory on the way
  swap, §7.4) after the identity check. The runner copies the complete
  lines (the whole file when it ends with a newline) from the descriptor
  it checked to a private temporary file under ``<state>/analysis/``,
  checks the identity again, and analyses the copy; a file that changed
  by then is ``unavailable`` with ``changed``, so no report is cached
  under a key it does not describe.
- **Cache.** Results are keyed by path, index ID, generation and file
  identity, and kept for the last 8 keys. A failure is cached like a result,
  so a key is analysed once; a new generation or a rewrite is a new key.
- **Timeout.** ``Limits.analysis_seconds`` (300 s) of wall time per run.
  At the timeout the runner kills the subprocess's process group with
  SIGKILL and records ``failed`` with reason ``timeout`` and no exit code;
  that key is not retried.
- **Unexpected errors.** Anything else a run raises, such as a ``Popen``
  that cannot start the interpreter, is ``failed`` with reason
  ``internal`` and the scrubbed error as ``detail``; the worker goes on
  to the next queued analysis, and a dead worker is restarted on the
  next enqueue.
- **Shutdown.** :meth:`AnalyzerRunner.close` kills a running subprocess the
  same way, drops queued work, removes temporary files, and from then on
  every state is ``unavailable`` with ``closed``.

The subprocess runs ``sys.executable -m stormlog``, in the private analysis
directory (so a ``stormlog`` package in the workspace cannot shadow the
installed one), with an environment that keeps no API key, token, secret
or proxy (:func:`analyzer_environment`), and with stdin and stdout closed
and stderr captured; the last 512 bytes of stderr, scrubbed, become the
document's ``detail`` on failure.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO

from stormlog.scrub import scrub_text
from stormlog.studio.contracts import (
    AnalysisState,
    FileIdentity,
    IndexSnapshot,
    Limits,
    NotReady,
    StudioError,
    WorkspaceRoot,
    envelope,
    finite_copy,
)

__all__ = [
    "ANALYZE_COMMAND",
    "KEPT_RESULTS",
    "RETRY_AFTER_S",
    "AnalyzerRunner",
    "analyzer_environment",
]

# The argv before the input path and the output options.
ANALYZE_COMMAND: tuple[str, ...] = (
    sys.executable,
    "-m",
    "stormlog",
    "infer",
    "analyze",
)
# Finished results kept, least recently used dropped first.
KEPT_RESULTS = 8
# Retry-After for the 202 while an analysis is queued or runs.
RETRY_AFTER_S = 2

_BLOCK = 64 * 2**10
_COPY_BLOCK = 2**20
_DETAIL_BYTES = 512
_Key = tuple[str, str, int, int, int, int, int]
# The analyzer's environment. ``infer analyze`` reads no variable it
# needs: its report is byte-identical under an empty environment. These
# keep the interpreter finding stormlog when it is a user-site install
# (HOME), a PYTHONPATH checkout or an activated venv, and the locale.
_ENV_KEPT = frozenset({"PATH", "HOME", "LANG", "LANGUAGE", "TMPDIR", "VIRTUAL_ENV"})
_ENV_KEPT_PREFIXES = ("LC_", "PYTHON")
# Never passed on, even when kept above (PYTHON_API_KEY, LC_TOKEN).
_ENV_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")


@dataclass(frozen=True)
class _Input:
    """What one run analysed: the file, or its complete-line prefix."""

    kind: str  # "file" or "complete_line_prefix"
    file_size: int
    analysed_bytes: int

    def record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "file_size": self.file_size,
            "analysed_bytes": self.analysed_bytes,
            "omitted_tail_bytes": self.file_size - self.analysed_bytes,
        }


@dataclass
class _Entry:
    """One key's analysis: queued, running or finished."""

    rel: str
    key: _Key
    state: AnalysisState
    input: _Input | None = None
    detail: str | None = None
    elapsed_s: float | None = None
    non_finite: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Outcome:
    state: AnalysisState
    input: _Input | None = None
    detail: str | None = None
    non_finite: tuple[str, ...] = ()


def _state(
    status: Any,
    reason: str | None = None,
    *,
    exit_code: int | None = None,
    report: Mapping[str, Any] | None = None,
    analysed_bytes: int | None = None,
) -> AnalysisState:
    return AnalysisState(status, reason, exit_code, report, analysed_bytes)


class AnalyzerRunner:
    """C4's :class:`~stormlog.studio.contracts.AnalysisCache` (decision 12).

    ``state_dir`` is Studio's state directory; outputs and prefix copies go
    in ``<state_dir>/analysis/`` (0700) and are removed after each run.
    ``command`` replaces :data:`ANALYZE_COMMAND`, for tests.
    """

    def __init__(
        self,
        root: WorkspaceRoot,
        state_dir: Path,
        limits: Limits | None = None,
        *,
        command: Sequence[str] = ANALYZE_COMMAND,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = root
        self._dir = Path(state_dir) / "analysis"
        self._limits = limits or Limits()
        self._command = tuple(command)
        self._clock = clock
        self._lock = threading.Condition()
        self._entries: OrderedDict[_Key, _Entry] = OrderedDict()
        self._queue: deque[_Entry] = deque()
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    # --- the AnalysisCache protocol ---------------------------------------------

    def get(self, rel: str, snapshot: IndexSnapshot) -> AnalysisState:
        """The analysis of this snapshot's file, queueing one if none exists."""
        return self._lookup(rel, snapshot).state

    def close(self) -> None:
        """Kill the running subprocess, drop queued work; idempotent."""
        with self._lock:
            self._closed = True
            self._queue.clear()
            process = self._process
            thread = self._thread
            self._lock.notify_all()
        if process is not None:
            _kill(process)
        if thread is not None:
            thread.join(timeout=10)

    # --- documents ------------------------------------------------------------

    def document(self, rel: str, snapshot: IndexSnapshot) -> dict[str, Any]:
        """The enveloped ``stormlog.studio.analysis`` document."""
        entry = self._lookup(rel, snapshot)
        state = entry.state
        pending = state.status in ("pending", "running")
        body = {
            "status": state.status,
            "reason": state.reason,
            "exit_code": state.exit_code,
            "analysed_bytes": state.analysed_bytes,
            "input": None if entry.input is None else entry.input.record(),
            "detail": entry.detail,
            "elapsed_s": entry.elapsed_s,
            "retry_after_s": RETRY_AFTER_S if pending else None,
            "limits": {
                "max_bytes": self._limits.analysis_bytes,
                "timeout_s": self._limits.analysis_seconds,
            },
            "report": None if state.report is None else dict(state.report),
            "non_finite_fields": list(entry.non_finite),
        }
        return envelope("analysis", body, snapshot)

    def respond(self, rel: str, snapshot: IndexSnapshot) -> dict[str, Any]:
        """The document, or NotReady carrying it while the analysis is queued
        or runs, which httpd answers with 202 and Retry-After."""
        document = self.document(rel, snapshot)
        if document["status"] in ("pending", "running"):
            raise NotReady(document, retry_after_s=RETRY_AFTER_S)
        return document

    # --- lookup and queue ----------------------------------------------------

    def _lookup(self, rel: str, snapshot: IndexSnapshot) -> _Entry:
        early = self._precheck(snapshot)
        if early is not None:
            return _Entry(rel, _key(rel, snapshot), early)
        key = _key(rel, snapshot)
        with self._lock:
            if self._closed:
                return _Entry(rel, key, _state("unavailable", "closed"))
            entry = self._entries.get(key)
            if entry is None:
                entry = self._enqueue(rel, key)
            self._entries.move_to_end(key)
            # A copy, so the caller sees one consistent state while the
            # worker updates the entry.
            return replace(entry, non_finite=list(entry.non_finite))

    def _precheck(self, snapshot: IndexSnapshot) -> AnalysisState | None:
        """The state a snapshot gets without running anything, if any."""
        identity = snapshot.identity
        if snapshot.status == "failed":
            return _state("unavailable", snapshot.failure or "index_failed")
        if snapshot.status == "building" or identity is None:
            return _state("pending", "index_building")
        if snapshot.growing:
            return _state("unavailable", "growing")
        if identity.size > self._limits.analysis_bytes:
            return _state("unavailable", "over_size_cap")
        if identity.size == 0:
            return _state("unavailable", "empty")
        return None

    def _enqueue(self, rel: str, key: _Key) -> _Entry:
        """Queue ``key``, replacing queued work for the same path (locked)."""
        for old in [queued for queued in self._queue if queued.rel == rel]:
            self._queue.remove(old)
            self._entries.pop(old.key, None)
        entry = _Entry(rel, key, _state("pending", "queued"))
        self._entries[key] = entry
        self._queue.append(entry)
        self._trim()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._work, name="stormlog-studio-analyzer", daemon=True
            )
            self._thread.start()
        self._lock.notify_all()
        return entry

    def _trim(self) -> None:
        """Keep at most KEPT_RESULTS finished entries (locked)."""
        finished = [
            key
            for key, entry in self._entries.items()
            if entry.state.status not in ("pending", "running")
        ]
        for key in finished[: max(len(finished) - KEPT_RESULTS, 0)]:
            del self._entries[key]

    def _work(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._closed:
                    self._lock.wait()
                if self._closed:
                    return
                entry = self._queue.popleft()
                entry.state = _state("running")
            started = self._clock()
            outcome = self._run_guarded(entry)
            with self._lock:
                entry.state = outcome.state
                entry.input = outcome.input
                entry.detail = outcome.detail
                entry.non_finite = list(outcome.non_finite)
                entry.elapsed_s = round(self._clock() - started, 3)
                self._trim()

    # --- one run ---------------------------------------------------------------

    def _run_guarded(self, entry: _Entry) -> _Outcome:
        """:meth:`_run`, with anything it raises recorded as ``internal``.

        A ``Popen`` that raises (a missing interpreter, a failed fork,
        ENOMEM) must fail this entry, not end the worker thread and leave
        every later analysis ``pending``.
        """
        try:
            return self._run(entry)
        except Exception as error:
            return _Outcome(_state("failed", "internal"), detail=_scrubbed(str(error)))

    def _run(self, entry: _Entry) -> _Outcome:
        token = secrets.token_hex(16)
        prefix = self._dir / f"{token}.jsonl"
        try:  # the copy goes however the run ends, a skip included
            return self._run_copy(entry, prefix, token)
        finally:
            _remove(prefix)

    def _run_copy(self, entry: _Entry, prefix: Path, token: str) -> _Outcome:
        try:
            self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            source, analysed = self._prepare(entry, prefix)
        except _Skip as skip:
            return _Outcome(skip.state)
        except (OSError, StudioError) as error:
            return _Outcome(
                _state("failed", "unreadable"), detail=_scrubbed(str(error))
            )
        return self._analyse(source, analysed, token)

    def _prepare(self, entry: _Entry, prefix: Path) -> tuple[Path, _Input]:
        """A private copy of the verified bytes to analyse, and what it holds.

        The copy is read from the descriptor whose identity matched the
        key, and that identity is checked again once it is written: the
        subprocess then analyses exactly the bytes the key names, never a
        path that a writer or a directory swap has changed since.
        """
        with self._root.open(entry.rel) as handle:
            info = os.fstat(handle.fileno())
            if _identity(info) != entry.key[3:]:
                raise _Skip(_state("unavailable", "changed"))
            size = info.st_size
            complete = _complete_prefix(handle, size)
            if complete == 0:
                raise _Skip(_state("unavailable", "no_complete_lines"))
            _copy_prefix(handle, prefix, complete)
            if _identity(os.fstat(handle.fileno())) != entry.key[3:]:
                raise _Skip(_state("unavailable", "changed"))
        kind = "file" if complete == size else "complete_line_prefix"
        return prefix, _Input(kind, size, complete)

    def _analyse(self, source: Path, analysed: _Input, token: str) -> _Outcome:
        output = self._dir / f"{token}.json"
        errors = self._dir / f"{token}.stderr"
        argv = [*self._command, str(source), "--format", "json"]
        argv += ["--output", str(output)]
        try:
            code = self._spawn(argv, errors)
            if code is None:
                return _Outcome(_state("failed", "timeout"), analysed)
            if self._closed:
                return _Outcome(_state("unavailable", "closed"), analysed)
            if code != 0:
                return _Outcome(
                    _state("failed", "nonzero_exit", exit_code=code),
                    analysed,
                    detail=_tail(errors),
                )
            return self._read_output(output, analysed)
        finally:
            _remove(output)
            _remove(errors)

    def _spawn(self, argv: list[str], errors: Path) -> int | None:
        """Run ``argv`` to the end or the timeout; None on a timeout."""
        with errors.open("wb") as stderr:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                cwd=self._dir,
                env=analyzer_environment(os.environ),
                start_new_session=True,
            )
        with self._lock:
            self._process = process
            closed = self._closed
        if closed:
            _kill(process)
        try:
            return process.wait(timeout=self._limits.analysis_seconds)
        except subprocess.TimeoutExpired:
            _kill(process)
            return None
        finally:
            with self._lock:
                self._process = None

    def _read_output(self, output: Path, analysed: _Input) -> _Outcome:
        try:
            size = output.stat().st_size
            if size > self._limits.report_bytes:
                return _Outcome(_state("failed", "output_too_large"), analysed)
            report = json.loads(output.read_bytes())
        except (OSError, ValueError, RecursionError) as error:
            return _Outcome(
                _state("failed", "bad_output"), analysed, _scrubbed(str(error))
            )
        if not isinstance(report, dict):
            return _Outcome(_state("failed", "bad_output"), analysed)
        clean, non_finite = finite_copy(report)
        state = _state(
            "ready",
            None if analysed.kind == "file" else "truncated_final_line",
            exit_code=0,
            report=clean,
            analysed_bytes=analysed.analysed_bytes,
        )
        return _Outcome(state, analysed, non_finite=tuple(non_finite))


class _Skip(Exception):
    """A run that ends before the subprocess, with this state."""

    def __init__(self, state: AnalysisState) -> None:
        super().__init__(state.reason)
        self.state = state


def analyzer_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The variables of ``environ`` the analyzer subprocess gets.

    Only ``PATH``, ``HOME``, ``LANG``, ``LANGUAGE``, ``TMPDIR``,
    ``VIRTUAL_ENV``, ``LC_*`` and ``PYTHON*``, and never a name ending in
    ``_API_KEY``, ``_TOKEN`` or ``_SECRET`` or holding ``PROXY``, in any
    case: Studio holds no credentials (§7.6), so its analyzer is given
    none of the user's, and it reads only local files, so no proxy.
    """
    return {name: value for name, value in environ.items() if _env_kept(name.upper())}


def _env_kept(name: str) -> bool:
    if name.endswith(_ENV_SECRET_SUFFIXES) or "PROXY" in name:
        return False
    return name in _ENV_KEPT or name.startswith(_ENV_KEPT_PREFIXES)


def _key(rel: str, snapshot: IndexSnapshot) -> _Key:
    # Optional in the revised contract: no identity before the first read.
    identity: FileIdentity | None = snapshot.identity
    if identity is None:
        return (rel, snapshot.index_id, snapshot.generation, -1, -1, -1, -1)
    return (
        rel,
        snapshot.index_id,
        snapshot.generation,
        identity.device,
        identity.inode,
        identity.size,
        identity.mtime_ns,
    )


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _complete_prefix(handle: BinaryIO, size: int) -> int:
    """The length of the file's complete lines: up to its last newline."""
    end = size
    while end > 0:
        start = max(end - _BLOCK, 0)
        handle.seek(start)
        block = handle.read(end - start)
        newline = block.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        end = start
    return 0


def _copy_prefix(handle: BinaryIO, target: Path, length: int) -> None:
    """Copy the first ``length`` bytes of ``handle`` to a new 0600 file."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    handle.seek(0)
    with os.fdopen(os.open(target, flags, 0o600), "wb") as out:
        remaining = length
        while remaining > 0:
            block = handle.read(min(_COPY_BLOCK, remaining))
            if not block:
                raise OSError("the artifact shrank while it was copied")
            out.write(block)
            remaining -= len(block)


def _kill(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL the subprocess's process group, then reap it."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _tail(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            handle.seek(max(size - _DETAIL_BYTES, 0))
            text = handle.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return None
    return _scrubbed(text) if text else None


def _scrubbed(text: str) -> str:
    return scrub_text(text, max_bytes=_DETAIL_BYTES)


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
