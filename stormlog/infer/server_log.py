"""What a vLLM 0.30.0 server's log says it chose at start-up.

Some settings are only decided once the engine runs: which attention
backend it selected, how much KV cache fit, and which CUDA graphs it
captured. vLLM logs them; the configuration it reports does not hold all of
them. The patterns here are vLLM 0.30.0's log statements
(``platforms/cuda.py``, ``v1/core/kv_cache_utils.py``,
``v1/worker/gpu_model_runner.py``); another version may word them
differently, and then a field is simply not found.

A log file can hold several start-ups. Only the last one counts: each
"Initializing a V1 LLM engine" line starts over.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PATTERNS_VERSION = "vllm_0_30_0"

_START = re.compile(r"Initializing a V1 LLM engine \(v([^)]+)\) with config:")
_BACKEND = re.compile(
    r"Using (\S+) attention backend out of potential backends: \[([^\]]*)\]"
)
# A backend asked for by name; only cuda.py words it this way.
_FORCED_BACKEND = re.compile(r"\[cuda\.py:\d+\] Using (\S+) backend\.$")
_KV_CACHE = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) "
    r"tokens per request: ([\d.]+)x"
)
_BLOCKS_OVERRIDE = re.compile(
    r"Overriding num_gpu_blocks=(\d+) with num_gpu_blocks_override=(\d+)"
)
# gpu_model_runner.py names the batch kind; Model Runner V2 only the mode.
_CAPTURE = re.compile(
    r"Capturing CUDA graphs \((?:(decode|mixed prefill-decode), )?(\w+)\)"
)
_CAPTURED = re.compile(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB")


@dataclass
class _Startup:
    vllm_version: str | None = None
    backends: list[str] = field(default_factory=list)
    candidates: list[list[str]] = field(default_factory=list)
    kv_cache_tokens: list[int] = field(default_factory=list)
    max_concurrency: list[float] = field(default_factory=list)
    blocks_override: list[int] = field(default_factory=list)
    captures: list[str] = field(default_factory=list)
    capture_seconds: list[int] = field(default_factory=list)
    capture_gib: list[float] = field(default_factory=list)


def read_server_log(path: Path) -> dict[str, Any]:
    """The last start-up's choices, from a vLLM server log file."""
    with path.open(errors="replace") as handle:
        return parse_server_log(handle)


def parse_server_log(lines: Iterable[str]) -> dict[str, Any]:
    """The last start-up's choices, from log lines."""
    startup = _Startup()
    startups = 0
    for raw in lines:
        line = raw.rstrip("\n")
        started = _START.search(line)
        if started is not None:
            startup = _Startup(vllm_version=started.group(1))
            startups += 1
            continue
        _read_line(startup, line)
    return _record(startup, startups)


def _read_line(startup: _Startup, line: str) -> None:
    if match := _BACKEND.search(line):
        startup.backends.append(match.group(1))
        startup.candidates.append(_names(match.group(2)))
    elif match := _FORCED_BACKEND.search(line):
        # Logged as the enum: AttentionBackendEnum.FLASH_ATTN.
        startup.backends.append(match.group(1).rsplit(".", 1)[-1])
    elif match := _KV_CACHE.search(line):
        startup.kv_cache_tokens.append(int(match.group(1).replace(",", "")))
        startup.max_concurrency.append(float(match.group(3)))
    elif match := _BLOCKS_OVERRIDE.search(line):
        startup.blocks_override.append(int(match.group(2)))
    elif match := _CAPTURE.search(line):
        kind, mode = match.group(1), match.group(2)
        startup.captures.append(f"{kind}:{mode}" if kind else mode)
    elif match := _CAPTURED.search(line):
        startup.capture_seconds.append(int(match.group(1)))
        startup.capture_gib.append(float(match.group(2)))


def _record(startup: _Startup, startups: int) -> dict[str, Any]:
    issues: list[str] = []
    record = {
        "patterns": PATTERNS_VERSION,
        "startups": startups,
        "vllm_version": startup.vllm_version,
        "attention_backend": _one(startup.backends, "attention_backend", issues),
        "attention_candidates": startup.candidates[0] if startup.candidates else None,
        "kv_cache_size_tokens": _one(startup.kv_cache_tokens, "kv_cache", issues),
        "max_concurrency": _one(startup.max_concurrency, "max_concurrency", issues),
        "num_gpu_blocks_override": _one(startup.blocks_override, "override", issues),
        "cudagraph_captures": sorted(set(startup.captures)),
        # With CUDA graph memory profiling on (vLLM's default), a first
        # capture only measures memory; the last one is what the engine kept.
        "graph_capture_gib": startup.capture_gib[-1] if startup.capture_gib else None,
    }
    if startups == 0:
        issues.append("no vLLM V1 engine start-up in the log")
    record["issues"] = issues
    return record


def _one(values: list[Any], name: str, issues: list[str]) -> Any:
    """The value every process logged; a disagreement is kept as an issue."""
    distinct = sorted(set(values), key=str)
    if len(distinct) > 1:
        issues.append(f"{name} differs across processes: {distinct}")
        return distinct
    return distinct[0] if distinct else None


def _names(listing: str) -> list[str]:
    return [name.strip().strip("'\"") for name in listing.split(",") if name.strip()]


__all__ = ["PATTERNS_VERSION", "parse_server_log", "read_server_log"]
