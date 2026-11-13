"""A local OpenAI-compatible server for Studio's fixture artifacts.

It extends the fake that ``tests/test_infer_profile.py`` already serves:
the same ``/v1/models`` answer, the same streamed chunks and the same
non-streamed body. What it adds is a behaviour per workload case, keyed by
the ``X-Request-Id`` the profiler sends (``stormlog-<run>-<case>_<phase>_
<worker>_<index>`` in a closed loop, ``..._<phase>_<index>`` in an open
one): fail every Nth measured request, fail every request, leave usage out
so the profiler has to estimate the token counts, end the measured window
through the run's stop file, or press Ctrl+C in the middle of a case.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

from tests.test_infer_profile import _FakeOpenAIHandler

# What the profiler puts after the run ID: the case ID, then the phase,
# then the worker (closed loop only) and the request index.
_REQUEST_ID = re.compile(r"-(?P<case>[^-]+)_(?P<phase>warmup|measured)(?:_\d+)?_\d+$")

# The body a non-streamed request gets when its case reports no usage; the
# other bodies are the base fake's own.
_NO_USAGE_BODY = {
    "choices": [
        {
            "message": {"role": "assistant", "content": "hello world"},
            "finish_reason": "stop",
        }
    ]
}


@dataclass(frozen=True)
class CasePolicy:
    """How the server answers one case's requests.

    ``fail_every`` fails the Nth, 2Nth, ... measured request to arrive;
    ``all_fail`` fails every request of the case, warmup included;
    ``usage`` False leaves the token counts out of every answer.
    ``stop_after`` creates the server's ``stop_file`` when the Nth measured
    request of the case arrives, so the profiler ends the measured window
    early (``--stop-file``); ``interrupt_after`` sends this process a
    SIGINT then, as a Ctrl+C would. Both answer that request as usual.
    ``slow_every`` holds the Nth, 2Nth, ... measured request for
    ``delay_seconds`` before answering, so a run whose request timeout is
    shorter records those requests as timed out.
    """

    fail_every: int | None = None
    all_fail: bool = False
    usage: bool = True
    stop_after: int | None = None
    interrupt_after: int | None = None
    slow_every: int | None = None
    delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fail_every", "stop_after", "interrupt_after", "slow_every"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must not be negative")


@dataclass(frozen=True)
class Answer:
    """What the server does with one request."""

    fail: bool = False
    usage: bool = True
    stop: bool = False
    interrupt: bool = False
    delay_seconds: float = 0.0


@dataclass
class ServerState:
    """What one server instance was told and what it has answered.

    ``stop_file`` is the run's ``--stop-file`` for policies with
    ``stop_after``; ``stopped`` and ``interrupted`` say whether the server
    ended the window or pressed Ctrl+C.
    """

    policies: Mapping[str, CasePolicy] = field(default_factory=dict)
    measured: dict[str, int] = field(default_factory=dict)
    answered: int = 0
    failed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_file: Path | None = None
    stopped: bool = False
    interrupted: bool = False

    def decide(self, request_id: str | None) -> tuple[bool, bool]:
        """Whether to fail this request, and whether to report usage."""
        answer = self.answer(request_id)
        return answer.fail, answer.usage

    def answer(self, request_id: str | None) -> Answer:
        """Count the request and say what to do with it."""
        match = _REQUEST_ID.search(request_id or "")
        policy = (
            self.policies.get(match.group("case"), CasePolicy())
            if match is not None
            else CasePolicy()
        )
        count = 0
        with self.lock:
            self.answered += 1
            if match is not None and match.group("phase") == "measured":
                count = self.measured.get(match.group("case"), 0) + 1
                self.measured[match.group("case")] = count
            answer = _answer(policy, count)
            self.failed += answer.fail
            self.stopped = self.stopped or answer.stop
            self.interrupted = self.interrupted or answer.interrupt
        return answer


def _answer(policy: CasePolicy, measured: int) -> Answer:
    """``policy``'s answer to its case's ``measured``th measured request
    (0 for a warmup request)."""
    return Answer(
        fail=policy.all_fail or _nth(measured, policy.fail_every),
        usage=policy.usage,
        stop=measured > 0 and measured == policy.stop_after,
        interrupt=measured > 0 and measured == policy.interrupt_after,
        delay_seconds=(
            policy.delay_seconds if _nth(measured, policy.slow_every) else 0.0
        ),
    )


def _nth(measured: int, every: int | None) -> bool:
    return every is not None and measured > 0 and measured % every == 0


class _PolicyHandler(_FakeOpenAIHandler):
    """The base fake, answering by its server's per-case policy."""

    state: ClassVar[ServerState]

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        answer = self.state.answer(self.headers.get("X-Request-Id"))
        self._act(answer)
        if answer.delay_seconds:
            time.sleep(answer.delay_seconds)
            # The client has given up and closed the connection.
            with contextlib.suppress(OSError):
                self._answer(answer)
            self.close_connection = True
            return
        self._answer(answer)

    def _answer(self, answer: Answer) -> None:
        fail, usage = answer.fail, answer.usage
        if not fail and usage:
            super().do_POST()  # the base fake reads and answers the request
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload: dict[str, Any] = json.loads(self.rfile.read(length).decode("utf-8"))
        if fail:
            body = json.dumps({"error": {"message": "forced failure"}})
            self._send_json(body.encode("utf-8"), status=500)
        elif payload.get("stream"):
            self._send_stream(include_usage=False)
        else:
            self._send_json(json.dumps(_NO_USAGE_BODY).encode("utf-8"))

    def _act(self, answer: Answer) -> None:
        """End the window or press Ctrl+C, before answering."""
        if answer.stop:
            if self.state.stop_file is None:
                raise RuntimeError("a stop_after policy needs the server's stop_file")
            self.state.stop_file.touch()
        if answer.interrupt:
            # Delivered to the main thread, where the profiler's handler
            # turns it into the run's KeyboardInterrupt.
            os.kill(os.getpid(), signal.SIGINT)


@dataclass(frozen=True)
class FakeOpenAI:
    """A running server: its chat completions URL and its state."""

    endpoint: str
    state: ServerState


@contextlib.contextmanager
def fake_openai(
    policies: Mapping[str, CasePolicy] | None = None,
    *,
    stop_file: Path | None = None,
) -> Iterator[FakeOpenAI]:
    """Serve on 127.0.0.1 at a free port until the block ends.

    ``stop_file`` is the path a ``stop_after`` policy creates."""
    state = ServerState(policies=dict(policies or {}), stop_file=stop_file)
    handler = type("PolicyHandler", (_PolicyHandler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield FakeOpenAI(
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            state=state,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
