"""T6: no credential and no server text leaves through an export without consent.

Canaries are planted wherever a run meets a secret or free text: the API
key, the endpoint's query, OTLP headers from flags and variables, resource
attributes under keys that are not listed, a server error that echoes the
prompt with a bearer token and an OpenAI ``param``, and a collector's own
error message. Every export (the spans the collector stored, the
Prometheus textfile, the capability records and every warning) is searched
for each canary in its raw, percent-encoded, JSON-escaped and base64 forms.
"""

import base64
import contextlib
import json
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from stormlog.infer.config import ProfileConfig
from stormlog.infer.export_config import ExportConfig
from stormlog.infer.profile import InferenceProfiler
from tests.fake_otlp_collector import FakeCollector, Reply, running
from tests.test_infer_profile import _FakeOpenAIHandler

pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")

API_KEY = "sk-apikey-canary-0123456789"
QUERY = "query-canary-0123456789"
HEADER = "header-canary-0123456789"
ENV_HEADER = "envTOKENcanary0123456789"
BODY_TOKEN = "sk-bodytoken-canary-0123456789"
PARAM = "param-canary-0123456789"
COLLECTOR = "collector-canary-0123456789"
RESOURCE = {
    "db.password": "hunter2-canary-0123",
    "api.key": "opaque-canary-0123456",
    "service.password": "svc-canary-0123456",
    "k8s.secret.x": "k8s-canary-0123456",
}
SECRETS = [API_KEY, QUERY, HEADER, ENV_HEADER, BODY_TOKEN, *RESOURCE.values()]
SERVER_TEXT = [PARAM, COLLECTOR, "echoed:"]


class _EchoingHandler(_FakeOpenAIHandler):
    """Every other request fails with an error that echoes the prompt."""

    count = 0
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        self.path = urllib.parse.urlsplit(self.path).path
        with self.lock:
            type(self).count += 1
            fail = self.count % 2 == 0
        if not fail:
            super().do_POST()
            return
        length = int(self.headers.get("Content-Length", "0"))
        prompt = json.loads(self.rfile.read(length))["messages"][-1]["content"]
        body = json.dumps(
            {
                "error": {
                    "message": f"echoed: {prompt[:40]} Bearer {BODY_TOKEN}",
                    "type": "invalid_request_error",
                    "param": PARAM,
                    "code": "context_length_exceeded",
                }
            }
        ).encode()
        self._send_json(body, status=400)


@contextlib.contextmanager
def _echoing_server() -> Iterator[str]:
    _EchoingHandler.count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_port
        yield f"http://127.0.0.1:{port}/v1/chat/completions?token={QUERY}"
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()


def _forms(value: str) -> set[str]:
    raw = value.encode()
    forms = {
        value,
        urllib.parse.quote(value, safe=""),
        json.dumps(value)[1:-1],
    }
    for encoded in (base64.b64encode(raw), base64.urlsafe_b64encode(raw)):
        forms.add(encoded.decode().rstrip("="))
    return forms


def _found(canary: str, outputs: dict[str, str]) -> list[str]:
    return [
        f"{name} holds {canary!r}"
        for name, text in outputs.items()
        for form in _forms(canary)
        if form in text
    ]


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: frozenset[str] = frozenset(),
) -> tuple[dict[str, str], FakeCollector]:
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "authorization=Bearer%20" + ENV_HEADER,
    )
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        ",".join(f"{key}={value}" for key, value in RESOURCE.items()),
    )
    # The collector refuses the first export with a message of its own,
    # which echoes the bare token of the Bearer header, as auth errors do.
    message = f"{COLLECTOR} unknown credential {ENV_HEADER}".encode()
    status = b"\x08\x03\x12" + bytes([len(message)]) + message
    replies = [Reply(400, status, store=False)]
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    output = tmp_path / "infer.jsonl"
    warnings: list[str] = []
    with running(replies) as collector, _echoing_server() as endpoint:
        config = ProfileConfig(
            endpoint=endpoint,
            model="fake-model",
            api_key=API_KEY,
            concurrency=(1,),
            input_tokens=(8,),
            output_tokens=(4,),
            output_path=str(output),
            request_count=6,
            stream=False,
            tokenizer="none",
            system_sampler="none",
            export=ExportConfig(
                otlp_endpoint=collector.url,
                otlp_headers=(f"x-api-key={HEADER}",),
                prometheus_textfile_dir=metrics,
                export_content=content,
            ),
        )
        InferenceProfiler(config, on_warning=warnings.append).run()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    capabilities = [
        r for r in records if str(r.get("component", "")).startswith("export.")
    ]
    outputs = {
        "collector": "\n".join(r.body.decode("latin-1") for r in collector.received),
        "decoded spans": repr([s for r in collector.received for s in r.spans]),
        "textfile": (metrics / "stormlog-default.prom").read_text(),
        "capability records": json.dumps(capabilities),
        "warnings": "\n".join(warnings),
    }
    return outputs, collector


def test_nothing_secret_or_from_a_server_leaves_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs, collector = _run(tmp_path, monkeypatch)
    # The run did export, and did fail requests, so the test means something.
    spans = [span for received in collector.received for span in received.spans]
    assert "stormlog.infer.request" in {span.name for span in spans}
    statuses = {span.attributes.get("stormlog.request.status") for span in spans}
    assert {"ok", "error"} <= statuses
    leaks = [
        leak for canary in SECRETS + SERVER_TEXT for leak in _found(canary, outputs)
    ]
    assert leaks == []
    # The header canaries were really sent, to the collector only.
    sent = collector.received[0].headers
    assert sent["x-api-key"] == HEADER
    assert sent["authorization"] == f"Bearer {ENV_HEADER}"
    # The left-out resource keys are named, without their values.
    capability = json.loads(outputs["capability records"])
    otlp = next(c for c in capability if c["component"] == "export.otlp")
    assert set(RESOURCE) <= set(otlp["metadata"]["dropped_resource_keys"])


def test_with_error_consent_server_text_leaves_but_no_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs, _ = _run(tmp_path, monkeypatch, frozenset({"errors"}))
    leaks = [leak for canary in SECRETS for leak in _found(canary, outputs)]
    assert leaks == []
    # The documented boundary: an error that echoes the request is exported,
    # and so is the collector's own message, scrubbed.
    assert _found("echoed:", outputs)
    assert _found(PARAM, outputs)
    assert _found(COLLECTOR, {"capability records": outputs["capability records"]})
