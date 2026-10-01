"""The workload record: what traffic a run sent, so it can be repeated.

The digest covers what decides the requests a run sends: the cases,
arrivals, prompts, warmup, decoding settings, seed, tokenizer and requested
cache state. It leaves out the endpoint, model, timeouts and reset URL, so
the same workload sent to two engine configurations has the same digest.
The API key is never recorded, and the reset URL is recorded without its
credentials or query string.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .cache_state import redact_url
from .config import ProfileConfig
from .prompts import GENERATOR_VERSION, REPEAT, PromptSpec
from .tokens import TokenCounter


def workload_record(
    config: ProfileConfig,
    *,
    session_id: str,
    counter: TokenCounter,
    prompt_spec: PromptSpec,
) -> dict[str, Any]:
    """The ``infer.workload`` record written at the start of a run."""
    spec = workload_spec(config, counter=counter, prompt_spec=prompt_spec)
    return {
        "schema_version": 1,
        "event_type": "infer.workload",
        "session_id": session_id,
        "workload_digest": _digest(_traffic(spec, open_loop=_open_loop(config))),
        **spec,
        "chat_template": chat_template_identity(counter),
    }


def workload_spec(
    config: ProfileConfig, *, counter: TokenCounter, prompt_spec: PromptSpec
) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "generator": {"name": "stormlog.synthetic", "version": GENERATOR_VERSION},
        "cases": [
            {
                "case_id": case.case_id,
                "input_tokens": case.input_tokens,
                "output_tokens": case.output_tokens,
                "concurrency": case.concurrency,
                "arrival": case.arrival.to_record(),
            }
            for case in config.cases()
        ],
        "measurement": {
            "request_count": config.request_count,
            "duration_seconds": config.duration_seconds,
            "max_in_flight": config.max_in_flight,
            "overflow": config.overflow,
            "drain_timeout_seconds": config.drain_timeout_seconds,
            "timeout_seconds": config.timeout_seconds,
        },
        "prompts": prompt_spec.to_record(),
        "warmup": {
            "requests": config.warmup_requests,
            # Warmup prompts share nothing with measured ones, except when
            # one prompt is repeated for the whole case.
            "prompts": "measured" if prompt_spec.mode == REPEAT else "separate",
        },
        "decoding": {
            "stream": config.stream,
            "stream_include_usage": config.stream_include_usage,
            "max_tokens_field": config.max_tokens_field,
            "extra_body": config.extra_body or {},
            "other_settings": "server defaults",
        },
        "tokenizer": tokenizer_identity(counter),
        "cache": {
            "requested": config.cache_state,
            "reset_url": redact_url(config.cache_reset_url),
        },
    }


def _open_loop(config: ProfileConfig) -> bool:
    return any(case.arrival.open_loop for case in config.cases())


def _traffic(spec: dict[str, Any], *, open_loop: bool) -> dict[str, Any]:
    """The parts of the workload that decide which requests are sent, and when.

    Timeouts, the drain deadline and the reset URL describe how a run was
    measured and where, not what it sent. The in-flight limit and overflow
    policy only shape traffic for open-loop arrivals.
    """
    measurement = spec["measurement"]
    shaping = ["request_count", "duration_seconds"]
    if open_loop:
        shaping += ["max_in_flight", "overflow"]
    return {
        "seed": spec["seed"],
        "generator": spec["generator"],
        "cases": spec["cases"],
        "measurement": {field: measurement[field] for field in shaping},
        "prompts": spec["prompts"],
        "warmup": spec["warmup"],
        "decoding": _numbers_as_values(spec["decoding"]),
        "tokenizer": spec["tokenizer"],
        "cache_requested": spec["cache"]["requested"],
    }


def _numbers_as_values(value: Any) -> Any:
    """Make 0 and 0.0 the same, as a server reading the JSON would."""
    if isinstance(value, dict):
        return {key: _numbers_as_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numbers_as_values(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def tokenizer_identity(counter: TokenCounter) -> dict[str, Any]:
    """Which tokenizer sized the prompts and counted tokens without usage."""
    identity = getattr(counter, "identity", None)
    if callable(identity):
        return dict(identity())
    return {"source": counter.source, "exact": counter.exact}


def chat_template_identity(counter: TokenCounter) -> dict[str, Any]:
    """The server applies the chat template; Stormlog cannot see which one.

    When a local transformers tokenizer is available, the digest of its
    template is recorded as a hint, not as what the server used.
    """
    digest = getattr(counter, "chat_template_digest", None)
    return {
        "applied_by": "server",
        "client_template_digest": digest() if callable(digest) else None,
    }


def _digest(spec: dict[str, Any]) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
