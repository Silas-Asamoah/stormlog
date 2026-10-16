"""Shared helpers for the fake vLLM engine's tests."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from examples.qualification.fake_engine import FakeEngine
from stormlog.infer.config import ProfileConfig
from stormlog.infer.correlation_codec import read_inference_records
from stormlog.infer.profile import InferenceProfiler


def chat(
    engine: FakeEngine,
    prompt: str,
    *,
    max_tokens: int = 4,
    stream: bool = True,
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    sampling: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """One chat completion; the streamed chunks, or the single JSON body.
    ``sampling`` adds parameters such as ``top_p`` to the body."""
    body = {
        "model": engine.config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
        **(sampling or {}),
    }
    sent = {"Content-Type": "application/json", **(headers or {})}
    if request_id is not None:
        sent["X-Request-Id"] = request_id
    request = urllib.request.Request(
        engine.endpoint, data=json.dumps(body).encode(), headers=sent, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode()
    if not stream:
        return [json.loads(raw)]
    chunks = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[len("data: ") :]))
    return chunks


def in_threads(calls: list[Callable[[], Any]]) -> list[Any]:
    """Run each call on its own thread and return the results in order."""
    results: list[Any] = [None] * len(calls)

    def runner(index: int, call: Callable[[], Any]) -> None:
        results[index] = call()

    threads = [
        threading.Thread(target=runner, args=(index, call), daemon=True)
        for index, call in enumerate(calls)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return results


def chats_in_background(
    engine: FakeEngine, prompts: list[str], *, max_tokens: int = 4
) -> list[threading.Thread]:
    """Start one streamed chat per prompt; join the returned threads."""
    threads = [
        threading.Thread(
            target=chat,
            args=(engine, prompt),
            kwargs={"max_tokens": max_tokens},
            daemon=True,
        )
        for prompt in prompts
    ]
    for thread in threads:
        thread.start()
    return threads


def join_all(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(timeout=60)


def get(url: str, *, timeout: float = 10.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()


def post(url: str, body: bytes = b"", *, timeout: float = 30.0) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()


def wait_until(predicate: Callable[[], bool], *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def words(count: int, tag: str) -> str:
    """``count`` distinct whitespace tokens, unique to ``tag``."""
    return " ".join(f"{tag}{index}" for index in range(count))


def run_profile(engine: FakeEngine, output: Path, **changes: Any) -> dict[str, Any]:
    """An ``infer profile`` run against the fake engine; returns its report."""
    values: dict[str, Any] = {
        "endpoint": engine.endpoint,
        "model": engine.config.model,
        "concurrency": (2,),
        "input_tokens": (32,),
        "output_tokens": (4,),
        "output_path": str(output),
        "request_count": 4,
        "tokenizer": "none",
        "system_sampler": "none",
        "run_id": "run-1",
        "prompt_mode": "unique",
    }
    values.update(changes)
    return InferenceProfiler(ProfileConfig(**values)).run()


def records(path: Path) -> list[dict[str, Any]]:
    """Read semantic inference records, expanding artifact-local contexts."""
    return read_inference_records(path)


def of_type(items: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [item for item in items if item.get("event_type") == event_type]
