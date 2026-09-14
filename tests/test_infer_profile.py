import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from stormlog.infer.analysis import analyze_inference_events
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.openai_client import (
    ChatCompletionResult,
    OpenAIChatCompletionsClient,
)
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.samplers import NvidiaSmiSampler


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "fake-model"}]}).encode("utf-8")
            self._send_json(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if "always-fail" in str(payload.get("model")):
            body = json.dumps({"error": {"message": "forced failure"}}).encode("utf-8")
            self._send_json(body, status=503)
            return
        if "no-usage" in str(payload.get("model")):
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "hello fallback",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode("utf-8")
            self._send_json(body)
            return
        if payload.get("stream"):
            self._send_stream(
                include_usage=payload.get("stream_options") == {"include_usage": True}
            )
            return
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello world"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            }
        ).encode("utf-8")
        self._send_json(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    def _send_json(self, body: bytes, *, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, *, include_usage: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        final_chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {"content": " world"},
                    "finish_reason": "stop",
                }
            ],
        }
        if include_usage:
            final_chunk["usage"] = {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            }
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "hello"}}]},
            final_chunk,
        ]
        for chunk in chunks:
            time.sleep(0.01)
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@contextlib.contextmanager
def _fake_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class InferenceProfileTests(unittest.TestCase):
    def test_stream_parser_preserves_content_timing_and_done_boundary(self) -> None:
        client = OpenAIChatCompletionsClient(
            endpoint="http://localhost/v1/chat/completions",
            model="test",
            timeout_seconds=1,
        )
        response = iter(
            [
                b": heartbeat\n",
                b"\n",
                b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n',
                b'data: {"choices": [{"delta": {"content": "hello"}}]}\n',
                b'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}], "usage": {"total_tokens": 7}}\n',
                b"data: [DONE]\n",
                b"data: invalid trailing record\n",
            ]
        )
        with (
            mock.patch(
                "stormlog.infer.openai_client.time.perf_counter",
                side_effect=[1.0, 2.0, 3.0, 4.0],
            ),
            mock.patch("stormlog.infer.openai_client.time.time_ns", return_value=500),
        ):
            result = client._read_streaming_response(
                response=response, started_at_ns=100, started_perf=0.0
            )
        self.assertEqual(
            result,
            ChatCompletionResult(
                text="hello world",
                started_at_ns=100,
                ended_at_ns=500,
                e2e_latency_ms=4000.0,
                ttft_ms=2000.0,
                first_chunk_latency_ms=1000.0,
                chunk_interarrival_ms=[1000.0],
                usage={"total_tokens": 7},
                finish_reason="stop",
            ),
        )
        self.assertEqual(next(response), b"data: invalid trailing record\n")

    def test_profile_streaming_endpoint_writes_request_events(self) -> None:
        with _fake_server() as endpoint:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "infer.jsonl"
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = infer_main(
                        [
                            "profile",
                            "--endpoint",
                            str(endpoint),
                            "--model",
                            "fake-model",
                            "--concurrency",
                            "2",
                            "--input-tokens",
                            "8",
                            "--output-tokens",
                            "4",
                            "--requests",
                            "2",
                            "--warmup-requests",
                            "1",
                            "--system-sampler",
                            "none",
                            "--tokenizer",
                            "none",
                            "--output",
                            str(output),
                        ]
                    )

                self.assertEqual(exit_code, 0)
                records = [
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                ]
                measured = [
                    record
                    for record in records
                    if record.get("event_type") == "infer.request"
                    and record.get("phase") == "measured"
                ]
                warmup = [
                    record
                    for record in records
                    if record.get("event_type") == "infer.request"
                    and record.get("phase") == "warmup"
                ]
                self.assertEqual(len(measured), 2)
                self.assertEqual(len(warmup), 1)
                first = measured[0]
                self.assertEqual(first["status"], "ok")
                self.assertEqual(first["timestamp_ns"], first["started_at_ns"])
                self.assertGreater(first["ttft_ms"], 0)
                self.assertGreater(first["e2e_latency_ms"], first["ttft_ms"])
                self.assertEqual(first["output_tokens"], 2)
                self.assertEqual(first["output_token_source"], "server_usage")
                self.assertTrue(first["output_token_exact"])
                self.assertTrue(
                    any(
                        record.get("event_type") == "infer.summary"
                        for record in records
                    )
                )
                self.assertTrue(
                    any(
                        record.get("event_type") == "infer.session"
                        and record.get("status") == "completed"
                        for record in records
                    )
                )

    def test_stream_usage_can_be_disabled_for_compatibility(self) -> None:
        with _fake_server() as endpoint:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "infer.jsonl"
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = infer_main(
                        [
                            "profile",
                            "--endpoint",
                            str(endpoint),
                            "--model",
                            "fake-model",
                            "--input-tokens",
                            "8",
                            "--output-tokens",
                            "4",
                            "--requests",
                            "1",
                            "--no-stream-usage",
                            "--system-sampler",
                            "none",
                            "--tokenizer",
                            "none",
                            "--output",
                            str(output),
                        ]
                    )

                self.assertEqual(exit_code, 0)
                request = next(
                    record
                    for record in (
                        json.loads(line)
                        for line in output.read_text(encoding="utf-8").splitlines()
                    )
                    if record.get("event_type") == "infer.request"
                    and record.get("phase") == "measured"
                )
                self.assertEqual(request["output_token_source"], "estimated")

    def test_analyze_reports_latency_and_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "infer.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "schema_version": 1,
                                "event_type": "infer.request",
                                "session_id": "s1",
                                "request_id": "r1",
                                "case_id": "c1_in8_out4",
                                "phase": "measured",
                                "started_at_ns": 1_000_000_000,
                                "ended_at_ns": 2_000_000_000,
                                "status": "ok",
                                "e2e_latency_ms": 1000.0,
                                "ttft_ms": 100.0,
                                "first_chunk_latency_ms": 80.0,
                                "output_tokens": 4,
                                "total_tokens": 12,
                                "output_token_source": "server_usage",
                            }
                        ),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            report = analyze_inference_events(path)

            self.assertEqual(report["summary"]["total_requests"], 1)
            case = report["cases"]["c1_in8_out4"]
            self.assertEqual(case["latency_ms"]["e2e_p50"], 1000.0)
            self.assertEqual(case["throughput"]["output_tokens_per_second"], 4.0)

    def test_analyze_filters_system_samples_to_case_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "infer.jsonl"
            records = [
                {
                    "schema_version": 1,
                    "event_type": "infer.system_sample",
                    "session_id": "s1",
                    "timestamp_ns": 500_000_000,
                    "device_used_bytes": 700,
                },
                {
                    "schema_version": 1,
                    "event_type": "infer.request",
                    "session_id": "s1",
                    "request_id": "low",
                    "case_id": "c1_in8_out4",
                    "phase": "measured",
                    "started_at_ns": 1_000_000_000,
                    "ended_at_ns": 2_000_000_000,
                    "status": "ok",
                    "e2e_latency_ms": 1000.0,
                    "output_tokens": 4,
                    "total_tokens": 12,
                    "output_token_source": "server_usage",
                },
                {
                    "schema_version": 1,
                    "event_type": "infer.system_sample",
                    "session_id": "s1",
                    "timestamp_ns": 1_500_000_000,
                    "device_used_bytes": 100,
                },
                {
                    "schema_version": 1,
                    "event_type": "infer.request",
                    "session_id": "s1",
                    "request_id": "high",
                    "case_id": "c2_in8_out4",
                    "phase": "measured",
                    "started_at_ns": 4_000_000_000,
                    "ended_at_ns": 5_000_000_000,
                    "status": "ok",
                    "e2e_latency_ms": 1000.0,
                    "output_tokens": 4,
                    "total_tokens": 12,
                    "output_token_source": "server_usage",
                },
                {
                    "schema_version": 1,
                    "event_type": "infer.system_sample",
                    "session_id": "s1",
                    "timestamp_ns": 4_500_000_000,
                    "device_used_bytes": 900,
                },
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            report = analyze_inference_events(path)

            self.assertEqual(
                report["cases"]["c1_in8_out4"]["memory"]["peak_device_used_bytes"],
                100,
            )
            self.assertEqual(
                report["cases"]["c2_in8_out4"]["memory"]["peak_device_used_bytes"],
                900,
            )

    def test_analyze_reports_process_rss_and_ignores_incomplete_time_bounds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "infer.jsonl"
            records = [
                {
                    "schema_version": 1,
                    "event_type": "infer.request",
                    "session_id": "s1",
                    "request_id": "valid",
                    "case_id": "c1_in8_out4",
                    "phase": "measured",
                    "started_at_ns": 1_000_000_000,
                    "ended_at_ns": 2_000_000_000,
                    "status": "ok",
                    "e2e_latency_ms": 1000.0,
                    "output_tokens": 4,
                    "total_tokens": 12,
                    "output_token_source": "server_usage",
                },
                {
                    "schema_version": 1,
                    "event_type": "infer.request",
                    "session_id": "s1",
                    "request_id": "missing-start",
                    "case_id": "c1_in8_out4",
                    "phase": "measured",
                    "ended_at_ns": 10_000_000_000,
                    "status": "ok",
                    "e2e_latency_ms": 1000.0,
                    "output_tokens": 4,
                    "total_tokens": 12,
                    "output_token_source": "server_usage",
                },
                {
                    "schema_version": 1,
                    "event_type": "infer.system_sample",
                    "session_id": "s1",
                    "timestamp_ns": 1_200_000_000,
                    "process_rss_bytes": 200,
                },
                {
                    "schema_version": 1,
                    "event_type": "infer.system_sample",
                    "session_id": "s1",
                    "timestamp_ns": 1_500_000_000,
                    "process_rss_bytes": 500,
                },
                {
                    "schema_version": 1,
                    "event_type": "infer.system_sample",
                    "session_id": "s1",
                    "timestamp_ns": 5_000_000_000,
                    "process_rss_bytes": 900,
                },
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            report = analyze_inference_events(path)

            case = report["cases"]["c1_in8_out4"]
            self.assertEqual(case["throughput"]["duration_seconds"], 1.0)
            self.assertIsNone(case["memory"]["peak_device_used_bytes"])
            self.assertEqual(case["memory"]["peak_process_rss_bytes"], 500)

    def test_profile_non_streaming_without_usage_records_estimated_tokens(
        self,
    ) -> None:
        with _fake_server() as endpoint:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "infer.jsonl"
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = infer_main(
                        [
                            "profile",
                            "--endpoint",
                            str(endpoint),
                            "--model",
                            "no-usage-model",
                            "--input-tokens",
                            "8",
                            "--output-tokens",
                            "4",
                            "--requests",
                            "1",
                            "--no-stream",
                            "--system-sampler",
                            "none",
                            "--tokenizer",
                            "none",
                            "--output",
                            str(output),
                        ]
                    )

                self.assertEqual(exit_code, 0)
                request = next(
                    record
                    for record in (
                        json.loads(line)
                        for line in output.read_text(encoding="utf-8").splitlines()
                    )
                    if record.get("event_type") == "infer.request"
                )
                self.assertIsNone(request["ttft_ms"])
                self.assertEqual(request["output_token_source"], "estimated")
                self.assertFalse(request["output_token_exact"])

    def test_profile_returns_error_when_all_measured_requests_fail(self) -> None:
        with _fake_server() as endpoint:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "infer.jsonl"
                stderr = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(stderr):
                        exit_code = infer_main(
                            [
                                "profile",
                                "--endpoint",
                                str(endpoint),
                                "--model",
                                "always-fail-model",
                                "--input-tokens",
                                "8",
                                "--output-tokens",
                                "4",
                                "--requests",
                                "2",
                                "--system-sampler",
                                "none",
                                "--tokenizer",
                                "none",
                                "--output",
                                str(output),
                            ]
                        )

                self.assertEqual(exit_code, 1)
                self.assertIn(
                    "no measured inference requests succeeded",
                    stderr.getvalue(),
                )
                records = [
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                ]
                measured = [
                    record
                    for record in records
                    if record.get("event_type") == "infer.request"
                    and record.get("phase") == "measured"
                ]
                self.assertEqual(len(measured), 2)
                self.assertTrue(all(record["status"] == "error" for record in measured))

    def test_profile_marks_session_incomplete_when_analysis_fails(self) -> None:
        with _fake_server() as endpoint:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "infer.jsonl"
                profiler = InferenceProfiler(
                    ProfileConfig(
                        endpoint=endpoint,
                        model="fake-model",
                        concurrency=(1,),
                        input_tokens=(8,),
                        output_tokens=(4,),
                        request_count=1,
                        output_path=str(output),
                        system_sampler="none",
                        tokenizer="none",
                    )
                )

                with mock.patch(
                    "stormlog.infer.profile.analyze_inference_events",
                    side_effect=ValueError("analysis failed"),
                ):
                    with self.assertRaisesRegex(ValueError, "analysis failed"):
                        profiler.run()

                records = [
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                ]
                sessions = [
                    record
                    for record in records
                    if record.get("event_type") == "infer.session"
                ]
                self.assertEqual(sessions[0]["status"], "running")
                self.assertEqual(sessions[-1]["status"], "incomplete")
                self.assertFalse(
                    any(
                        record.get("event_type") == "infer.summary"
                        for record in records
                    )
                )

    def test_requested_concurrency_sets_request_executor_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "infer.jsonl"
            profiler = InferenceProfiler(
                ProfileConfig(
                    endpoint="http://127.0.0.1:1/v1/chat/completions",
                    model="fake-model",
                    concurrency=(34,),
                    input_tokens=(8,),
                    output_tokens=(4,),
                    request_count=34,
                    output_path=str(output),
                    stream=False,
                    system_sampler="none",
                    tokenizer="none",
                )
            )
            client = _BlockingClient(target_active=34)
            profiler.client = client  # type: ignore[assignment]

            profiler.run()

            self.assertEqual(client.max_active, 34)

    def test_analyze_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "infer.jsonl"
            report_path = Path(directory) / "report.json"
            artifact.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "event_type": "infer.request",
                        "session_id": "s1",
                        "request_id": "r1",
                        "case_id": "c1_in8_out4",
                        "phase": "measured",
                        "started_at_ns": 1_000_000_000,
                        "ended_at_ns": 2_000_000_000,
                        "status": "ok",
                        "e2e_latency_ms": 1000.0,
                        "ttft_ms": 100.0,
                        "first_chunk_latency_ms": 80.0,
                        "output_tokens": 4,
                        "total_tokens": 12,
                        "output_token_source": "server_usage",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = infer_main(
                    [
                        "analyze",
                        str(artifact),
                        "--format",
                        "json",
                        "--output",
                        str(report_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["total_requests"], 1)

    def test_infer_cli_suppresses_broken_pipe_errors(self) -> None:
        stderr = io.StringIO()
        with mock.patch(
            "stormlog.infer.cli.cmd_analyze",
            side_effect=BrokenPipeError,
        ):
            with contextlib.redirect_stderr(stderr):
                exit_code = infer_main(["analyze", "artifact.jsonl"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr.getvalue(), "")

    def test_openai_client_rejects_non_http_endpoints(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "endpoint must use http:// or https://"
        ):
            OpenAIChatCompletionsClient(
                endpoint="file:///tmp/secret",
                model="fake-model",
                timeout_seconds=1.0,
                api_key="secret",
            )

    def test_nvidia_sampler_returns_none_when_command_raises(self) -> None:
        sampler = NvidiaSmiSampler()
        with mock.patch(
            "stormlog.infer.samplers.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertIsNone(sampler.sample(session_id="s1"))

    def test_nvidia_sampler_selects_matching_device_id(self) -> None:
        sampler = NvidiaSmiSampler(device_id=1)
        result = SimpleNamespace(
            returncode=0,
            stdout=(
                "3, GPU-3, 40000, 1000, 39000, 7\n" "1, GPU-1, 20000, 3000, 17000, 42\n"
            ),
        )
        with mock.patch(
            "stormlog.infer.samplers.subprocess.run",
            return_value=result,
        ):
            sample = sampler.sample(session_id="s1")

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.device_id, 1)
        self.assertEqual(sample.device_name, "GPU-1")
        self.assertEqual(sample.device_used_bytes, 3000 * 1024 * 1024)
        self.assertEqual(sample.gpu_utilization_percent, 42.0)


class _BlockingClient:
    def __init__(self, *, target_active: int) -> None:
        self.target_active = target_active
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.release = threading.Event()

    def complete(
        self,
        *,
        prompt: str,
        output_tokens: int,
        stream: bool,
        stream_include_usage: bool,
    ) -> ChatCompletionResult:
        started_at_ns = time.time_ns()
        started_perf = time.perf_counter()
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= self.target_active:
                self.release.set()
        self.release.wait(timeout=2)
        with self.lock:
            self.active -= 1
        ended_at_ns = time.time_ns()
        return ChatCompletionResult(
            text="hello world",
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
            e2e_latency_ms=(time.perf_counter() - started_perf) * 1000.0,
            ttft_ms=None,
            first_chunk_latency_ms=None,
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
            finish_reason="stop",
        )


if __name__ == "__main__":
    unittest.main()
