"""Records a profile writes while a request is under way: its send and its
first content, each before the request's terminal record."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any, cast
from unittest import mock

from stormlog.infer.config import ProfileConfig
from stormlog.infer.events import JsonlEventWriter
from stormlog.infer.open_loop import Arrival
from stormlog.infer.openai_client import OpenAIChatCompletionsClient
from stormlog.infer.profile import InferenceProfiler, _PhaseRequest

STREAM = b"".join(
    f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
    for chunk in (
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "hello"}}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
    )
)
ANSWER = json.dumps(
    {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
).encode("utf-8")


def _complete(body: bytes, *, stream: bool) -> tuple[Any, list[Any]]:
    """Run one completion against ``body``; return it and the calls seen,
    in order: the callbacks' and the request's opening."""
    calls: list[Any] = []

    def urlopen(_request: Any, timeout: float) -> io.BytesIO:
        calls.append("open")
        return io.BytesIO(body)

    client = OpenAIChatCompletionsClient(
        endpoint="http://localhost/v1/chat/completions", model="m", timeout_seconds=1
    )
    with mock.patch.object(client, "_opener", mock.Mock(open=urlopen)):
        result = client.complete(
            prompt="hi",
            output_tokens=4,
            stream=stream,
            stream_include_usage=False,
            on_sent=lambda at_ns: calls.append(("sent", at_ns)),
            on_first_content=lambda at_ns: calls.append(("first", at_ns)),
        )
    return result, calls


def test_the_send_is_reported_before_it_goes_out_and_first_content_at_its_ttft() -> (
    None
):
    result, calls = _complete(STREAM + b"data: [DONE]\n\n", stream=True)

    assert calls[:2] == [("sent", result.started_at_ns), "open"]
    # Once, for the first of two content pieces, on the send's clock.
    [(kind, first_at_ns)] = calls[2:]
    assert kind == "first"
    assert result.ttft_ms is not None
    assert abs(first_at_ns - result.started_at_ns - result.ttft_ms * 1e6) <= 1


def test_no_first_content_is_reported_without_streamed_content() -> None:
    role_only = STREAM.split(b"\n\n")[0] + b"\n\ndata: [DONE]\n\n"
    for body, stream in ((ANSWER, False), (role_only, True)):
        result, calls = _complete(body, stream=stream)

        assert calls == [("sent", result.started_at_ns), "open"]


def test_a_progress_record_after_the_capture_ends_is_dropped(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    profiler = InferenceProfiler(
        ProfileConfig(
            endpoint="http://127.0.0.1:1/v1/chat/completions",
            model="m",
            concurrency=(1,),
            input_tokens=(8,),
            output_tokens=(4,),
            output_path=str(output),
            system_sampler="none",
            tokenizer="none",
        )
    )
    case = profiler.config.cases()[0]
    errors: list[dict[str, Any]] = []

    async def scenario() -> Any:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        with JsonlEventWriter(output) as writer:
            request = _PhaseRequest(case, writer, cast(Any, None), "measured")
            on_sent, on_first_content = profiler._progress_records(
                "r0", request, Arrival(index=0, mode="closed", intended_at_ns=5)
            )
            await asyncio.to_thread(on_sent, 10)
            await asyncio.sleep(0)
        # A call a drain gave up on reports after the writer has closed.
        await asyncio.to_thread(on_first_content, 20)
        await asyncio.sleep(0)
        return on_first_content

    try:
        on_first_content = asyncio.run(scenario())
        on_first_content(30)  # and after the run's loop has closed
    finally:
        profiler.request_executor.shutdown()

    assert errors == []
    [record] = [json.loads(line) for line in output.read_text().splitlines()]
    assert record == {
        "schema_version": 1,
        "event_type": "infer.dispatch",
        "session_id": profiler.session.session_id,
        "request_id": "r0",
        "x_request_id": f"stormlog-{profiler.run_id}-r0",
        "case_id": case.case_id,
        "phase": "measured",
        "intended_at_ns": 5,
        "started_at_ns": 10,
        "timestamp_ns": 10,
    }
