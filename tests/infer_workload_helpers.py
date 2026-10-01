"""A fake chat client and a profiler runner for workload-control tests."""

import json
import threading
import time
from pathlib import Path
from typing import Any

from stormlog.infer.config import ProfileConfig
from stormlog.infer.openai_client import ChatCompletionResult
from stormlog.infer.profile import InferenceProfiler


class SleepingClient:
    """Answers every request after a fixed delay and tracks concurrency."""

    def __init__(self, latency_seconds: float) -> None:
        self.latency_seconds = latency_seconds
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def complete(
        self,
        *,
        prompt: str,
        output_tokens: int,
        stream: bool,
        stream_include_usage: bool,
    ) -> ChatCompletionResult:
        started_at_ns = time.time_ns()
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.latency_seconds)
        with self.lock:
            self.active -= 1
        return ChatCompletionResult(
            text="ok",
            started_at_ns=started_at_ns,
            ended_at_ns=time.time_ns(),
            e2e_latency_ms=self.latency_seconds * 1000.0,
            ttft_ms=None,
            first_chunk_latency_ms=None,
            usage={"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            finish_reason="stop",
        )


def run_profile_with_fake_client(
    tmp_path: Path, latency_seconds: float, **changes: Any
) -> tuple[list[dict[str, Any]], dict[str, Any], SleepingClient]:
    output = tmp_path / "infer.jsonl"
    values: dict[str, Any] = {
        "endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "model": "fake-model",
        "concurrency": (1,),
        "input_tokens": (8,),
        "output_tokens": (4,),
        "output_path": str(output),
        "stream": False,
        "system_sampler": "none",
        "tokenizer": "none",
    }
    values.update(changes)
    profiler = InferenceProfiler(ProfileConfig(**values))
    client = SleepingClient(latency_seconds)
    profiler.client = client  # type: ignore[assignment]
    report = profiler.run()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    requests = [r for r in records if r.get("event_type") == "infer.request"]
    return requests, report, client
