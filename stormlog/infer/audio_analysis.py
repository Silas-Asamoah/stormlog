"""Offline audio cohorts, duration throughput and latency/RTF distributions."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, cast

from .audio_events import AUDIO_FAMILIES, validate_audio_record
from .populations import populations_from_records
from .quantiles import QuantileEstimate, penalized_quantile, successful_quantile
from .report_stats import is_number

METRICS = {
    "e2e_latency_ms": "ms",
    "first_response_byte_ms": "ms",
    "first_audio_ms": "ms",
    "first_transcript_ms": "ms",
    "rtf": "elapsed_seconds_per_audio_second",
}


def analyze_audio_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    audio, workloads, task, engine = _contract(records)
    requests = _measured(audio, "infer.audio.request")
    windows = _measured(records, "infer.phase_window")
    populations = populations_from_records(requests, windows, workloads[0])
    cases = [
        _case(
            case_id,
            [record for record in requests if record["case_id"] == case_id],
            population,
            task,
        )
        for case_id, population in populations.items()
    ]
    return {
        "report_type": "stormlog.infer.audio",
        "analysis_version": 1,
        "engine": engine,
        "task": task,
        "summary": _summary(requests, records, len(cases)),
        "cases": cases,
        "workload": workloads[0],
        "telemetry": {
            "client_observation_scope": "client_http",
            "server_execution_attribution": "unavailable",
            "native_allocator": "unavailable_without_server_evidence",
            "streaming_semantics": (
                "upload_then_output" if task == "stt" else "encoded_pcm_segments"
            ),
        },
    }


def _contract(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    audio = [record for record in records if record.get("event_type") in AUDIO_FAMILIES]
    for record in audio:
        validate_audio_record(record)
    task, engine = _identity(audio, records)
    workloads = [
        record for record in audio if record["event_type"] == "infer.audio.workload"
    ]
    if len(workloads) != 1:
        raise ValueError("Audio artifact requires exactly one workload record")
    return audio, workloads, task, engine


def _measured(records: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("event_type") == family and record.get("phase") == "measured"
    ]


def _identity(
    audio: list[dict[str, Any]], records: list[dict[str, Any]]
) -> tuple[str, str]:
    identities = {
        (record["task"], record["engine"], record["session_id"]) for record in audio
    }
    if len(identities) != 1 or any(
        record.get("event_type") == "infer.request" for record in records
    ):
        raise ValueError(
            "Mixed text/audio or multiple audio task/session identities are unsupported"
        )
    task, engine, session = next(iter(identities))
    return task, engine


def _summary(
    requests: list[dict[str, Any]], records: list[dict[str, Any]], case_count: int
) -> dict[str, Any]:
    statuses = Counter(record["status"] for record in requests)
    sessions = [
        record for record in records if record.get("event_type") == "infer.session"
    ]
    return {
        "total_requests": len(requests),
        "successful_requests": statuses["ok"],
        "failed_requests": len(requests) - statuses["ok"],
        "failures_by_status": dict(statuses - Counter(ok=statuses["ok"])),
        "failure_rate": (
            (len(requests) - statuses["ok"]) / len(requests) if requests else None
        ),
        "case_count": case_count,
        "session_status": sessions[-1].get("status") if sessions else "unknown",
    }


def _estimate(estimate: QuantileEstimate) -> dict[str, Any]:
    def neutral(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key.removesuffix("_ms"): neutral(item) for key, item in value.items()
            }
        return value

    return cast(dict[str, Any], neutral(asdict(estimate)))


def metric_summary(
    requests: list[dict[str, Any]], field: str, unit: str
) -> dict[str, Any]:
    successful = [record for record in requests if record["status"] == "ok"]
    values = [
        float(record[field]) for record in successful if is_number(record.get(field))
    ]
    missing = len(successful) - len(values)
    failures = len(requests) - len(successful)
    return {
        "unit": unit,
        "provenance": "client_monotonic_http",
        "n": len(values),
        "successful_missing": missing,
        "mean": sum(values) / len(values) if values else None,
        "successful": {
            f"p{round(p * 100)}": _estimate(successful_quantile(values, p))
            for p in (0.5, 0.9, 0.95, 0.99)
        },
        "failure_penalized": {
            f"p{round(p * 100)}": _estimate(
                penalized_quantile(values, failures, p, successful_missing=missing)
            )
            for p in (0.5, 0.9, 0.95, 0.99)
        },
    }


def _case(
    case_id: str, requests: list[dict[str, Any]], population: Any, task: str
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "population": population.population.to_record(),
        "intervals": population.intervals.to_record(),
        "metrics": {
            field: metric_summary(requests, field, unit)
            for field, unit in METRICS.items()
        },
        "unsuccessful": _unsuccessful(requests),
        "throughput": _throughput(requests, population, task),
    }


def _unsuccessful(requests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        status: {"n": len(group), "observed_e2e": _observed(group)}
        for status in sorted({record["status"] for record in requests} - {"ok"})
        for group in [[record for record in requests if record["status"] == status]]
    }


def _throughput(
    requests: list[dict[str, Any]], population: Any, task: str
) -> dict[str, Any]:
    duration_field = "output_audio_seconds" if task == "tts" else "input_audio_seconds"
    duration = sum(
        float(record[duration_field]) for record in requests if record["status"] == "ok"
    )
    interval = population.intervals.rate
    seconds = interval.seconds if interval is not None else None
    valid = population.population.cohort_valid
    available = bool(valid and seconds)
    return {
        "unit": "audio_seconds_per_second",
        "audio_duration_scope": duration_field.split("_")[0],
        "successful_audio_seconds": duration,
        "interval_seconds": seconds,
        "interval": interval.to_record() if interval else None,
        "audio_seconds_per_second": _per_second(duration, seconds, valid),
        "unavailable_reason": (
            None
            if available
            else population.intervals.rate_reason or "invalid_cohort_or_interval"
        ),
    }


def _observed(requests: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(record["e2e_latency_ms"])
        for record in requests
        if is_number(record.get("e2e_latency_ms"))
    ]
    return {
        "unit": "ms",
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
        "meaning": "elapsed_until_observed_failure_not_success_latency",
    }


def format_audio_analysis(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"Audio inference: {report['engine']} / {report['task']}",
        f"Requests: {summary['total_requests']}; successful: {summary['successful_requests']}; failed: {summary['failed_requests']}",
    ]
    for case in report["cases"]:
        lines.append(
            f"{case['case_id']}: {case['throughput']['audio_seconds_per_second']} audio seconds / second"
        )
        for metric, result in case["metrics"].items():
            lines.append(
                f"  {metric}: mean={result['mean']}, p90={result['successful']['p90']['value']}, n={result['n']} ({result['unit']})"
            )
    return "\n".join(lines)


def _per_second(duration: float, seconds: float | None, valid: bool) -> float | None:
    return duration / seconds if valid and seconds else None
