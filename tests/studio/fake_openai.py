"""A local OpenAI-compatible server for Studio's fixture artifacts.

It extends the fake that ``tests/test_infer_profile.py`` already serves:
the same ``/v1/models`` answer, the same streamed chunks and the same
non-streamed body. What it adds is a behaviour per workload case, keyed by
the ``X-Request-Id`` the profiler sends (``stormlog-<run>-<case>_<phase>_
<worker>_<index>``): fail every Nth measured request, fail every request,
or leave usage out so the profiler has to estimate the token counts.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from typing import Any, ClassVar

from tests.test_infer_profile import _FakeOpenAIHandler

# What the profiler puts after the run ID: the case ID, then the phase.
_REQUEST_ID = re.compile(r"-(?P<case>[^-]+)_(?P<phase>warmup|measured)_\d+_\d+$")

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
    """

    fail_every: int | None = None
    all_fail: bool = False
    usage: bool = True

    def __post_init__(self) -> None:
        if self.fail_every is not None and self.fail_every < 1:
            raise ValueError("fail_every must be at least 1")


@dataclass
class ServerState:
    """What one server instance was told and what it has answered."""

    policies: Mapping[str, CasePolicy] = field(default_factory=dict)
    measured: dict[str, int] = field(default_factory=dict)
    answered: int = 0
    failed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def decide(self, request_id: str | None) -> tuple[bool, bool]:
        """Whether to fail this request, and whether to report usage."""
        match = _REQUEST_ID.search(request_id or "")
        policy = (
            self.policies.get(match.group("case"), CasePolicy())
            if match is not None
            else CasePolicy()
        )
        with self.lock:
            self.answered += 1
            fail = policy.all_fail
            if match is not None and match.group("phase") == "measured":
                count = self.measured.get(match.group("case"), 0) + 1
                self.measured[match.group("case")] = count
                every = policy.fail_every
                fail = fail or (every is not None and count % every == 0)
            if fail:
                self.failed += 1
        return fail, policy.usage


class _PolicyHandler(_FakeOpenAIHandler):
    """The base fake, answering by its server's per-case policy."""

    state: ClassVar[ServerState]

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        fail, usage = self.state.decide(self.headers.get("X-Request-Id"))
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


@dataclass(frozen=True)
class FakeOpenAI:
    """A running server: its chat completions URL and its state."""

    endpoint: str
    state: ServerState


@contextlib.contextmanager
def fake_openai(
    policies: Mapping[str, CasePolicy] | None = None,
) -> Iterator[FakeOpenAI]:
    """Serve on 127.0.0.1 at a free port until the block ends."""
    state = ServerState(policies=dict(policies or {}))
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
