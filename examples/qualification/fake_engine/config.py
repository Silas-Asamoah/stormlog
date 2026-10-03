"""Settings of the fake engine: fixed at start, and switches flipped at run time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VLLM_VERSION = "0.30.0"


@dataclass(frozen=True)
class FakeEngineConfig:
    """What the fake engine is, fixed for its lifetime.

    The defaults make a small, fast engine for CPU tests; every cost is a
    simulated duration, not computation. Token counts are whitespace words of
    the prompt plus a fixed chat-template prefix.
    """

    model: str = "fake/qwen-0.5b"
    host: str = "127.0.0.1"
    port: int = 0
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 256
    num_gpu_blocks: int = 256
    block_size: int = 16
    enable_prefix_caching: bool = True
    # One step costs step_seconds plus these per scheduled token.
    step_seconds: float = 0.002
    prefill_token_seconds: float = 0.00002
    decode_token_seconds: float = 0.0002
    # vLLM 0.30 suffixes each request ID with 8 random hex characters unless
    # VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set.
    request_id_randomization: bool = True
    # The execution hook's STORMLOG_VLLM_HOOK_DIR; None writes no raw log.
    hook_dir: Path | None = None
    # Seal hook segments this often, so a reader sees records soon.
    hook_seal_seconds: float = 60.0
    # vLLM's torch_profiler_dir; None means the profiler is not configured.
    trace_dir: Path | None = None
    # An OTLP/HTTP traces endpoint (``http://host:port/v1/traces``); None
    # exports no spans.
    spans_endpoint: str | None = None
    span_export_seconds: float = 0.25
    # "protobuf", as vLLM's OpenTelemetry exporter sends spans (it needs the
    # infer-otlp extra), or "json" for OTLP/JSON.
    span_encoding: str = "protobuf"
    # Only a subprocess may honour the kill switch.
    allow_kill: bool = False
    device_uuid: str = "GPU-00000000-0000-4000-8000-00000000fa4e"


@dataclass
class Controls:
    """Switches a test or a harness flips while the engine runs.

    Read without a lock: each is one attribute, and a reader seeing the old
    value for one more step is harmless.
    """

    # Profiler routes. A status other than 200 answers both routes with it,
    # as a server started without --profiler-config does.
    profiler_status: int = 200
    start_pause_seconds: float = 0.0
    stop_pause_seconds: float = 0.05
    trace_write_delay_seconds: float = 0.0
    # (a) The start runs, then the connection closes without an answer.
    drop_start_response: bool = False
    # (b) The stop answers 200 but writes no trace.
    stop_writes_trace: bool = True
    # (c) Stop by itself after this many profiled steps, like max_iterations.
    profiler_max_iterations: int | None = None
    # /metrics: "ok", "slow" (answer after metrics_delay_seconds) or "fail" (500).
    metrics_mode: str = "ok"
    metrics_delay_seconds: float = 0.0
    # (e) Hold each span this long past its request's end, and/or send every
    # batch twice.
    span_delay_seconds: float = 0.0
    span_duplicates: bool = False


__all__ = ["VLLM_VERSION", "Controls", "FakeEngineConfig"]
