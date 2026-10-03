"""Analysis helpers for inference profiling artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .arrival_report import arrival_lines, arrival_summary, latency_from_intended_ms
from .cache_state import cache_lines, cache_summary
from .correlation_accounting import AlignedTimestamp
from .errors import InferInputError
from .host_clock import is_boot_qualified
from .report_stats import int_value as _int_value
from .report_stats import is_number as _is_number
from .report_stats import number_values as _number_values
from .report_stats import percentile as _percentile
from .server_clock import (
    AMBIGUOUS,
    UNCOVERED,
    SampleAlignment,
    ServerClock,
    align_samples,
    artifact_alignments,
    build_server_clock,
    client_clock_domain,
)
from .server_group import members as group_members
from .server_group import membership_issue
from .telemetry import ServerIdentity, TelemetrySample, load_telemetry
from .vllm_analysis import (
    load_external_spans,
    vllm_case_lines,
    vllm_lines,
    vllm_report,
)
from .vllm_execution_report import execution_lines, execution_report
from .workload_report import (
    length_summary,
    prompt_lines,
    prompt_summary,
    workload_lines,
    workload_summary,
)


def analyze_inference_events(
    path: str | Path,
    *,
    server_telemetry_paths: Iterable[str | Path] = (),
    direct_server: bool = False,
    clock_offset_ns: int | None = None,
    clock_uncertainty_ns: int | None = None,
    vllm_span_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Analyze an inference profiling JSONL artifact."""
    records = _load_jsonl(path)
    requests, samples = _partition_inference_records(records)
    server_samples = _load_server_samples(server_telemetry_paths)
    vllm = _vllm_telemetry(records, vllm_span_paths)
    join, members = _server_join(
        records,
        server_samples,
        direct_server=direct_server,
        clock_offset_ns=clock_offset_ns,
        clock_uncertainty_ns=clock_uncertainty_ns,
    )
    timelines = [(member, _member_timeline(member)) for member in members]
    ok_requests = [record for record in requests if record.get("status") == "ok"]
    cases = _case_reports(records, requests, samples, timelines, "group" in join)
    if timelines:
        join["case_coverage"] = _coverage_counts(cases)
    failed = [record for record in requests if record.get("status") != "ok"]
    return {
        "summary": {
            "total_requests": len(requests),
            "successful_requests": len(ok_requests),
            "failed_requests": len(failed),
            "failure_rate": (len(failed) / len(requests)) if requests else 0.0,
            "failures_by_status": _failures_by_status(failed),
            "case_count": len(cases),
        },
        "cases": cases,
        "workload": workload_summary(records),
        "telemetry": {
            "client_observation_scope": "client_local",
            "server_join": join,
            "server_targets": _server_targets(server_samples),
            "vllm": vllm,
            "execution": execution_report(records),
        },
    }


def _vllm_telemetry(
    records: list[dict[str, Any]], span_paths: Iterable[str | Path]
) -> dict[str, Any]:
    """The vLLM block; a span file or record that cannot be read is an input error."""
    try:
        return vllm_report(records, load_external_spans(records, span_paths))
    except (OSError, ValueError) as exc:
        raise InferInputError(f"vLLM telemetry: {_reason(exc)}") from exc


def _case_reports(
    records: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    timelines: list[tuple[_Member, _ServerTimeline]],
    grouped: bool,
) -> dict[str, dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in requests:
        by_case.setdefault(str(record.get("case_id", "unknown")), []).append(record)
    windows = _measured_windows(records)
    cache_states = _case_records(records, "infer.cache_state")
    cases = {}
    for case_id, case_requests in sorted(by_case.items()):
        cases[case_id] = _case_report(
            case_requests, samples, timelines, grouped, windows.get(case_id)
        )
        cases[case_id]["cache"] = cache_summary(cache_states.get(case_id))
    return cases


def _case_report(
    case_requests: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    timelines: list[tuple[_Member, _ServerTimeline]],
    grouped: bool,
    window: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize one case: latency from its completed requests, arrivals from all."""
    ok = [record for record in case_requests if record.get("status") == "ok"]
    report = _summarize_requests(ok, samples=_samples_for_request_window(samples, ok))
    report["arrivals"] = arrival_summary(case_requests, window)
    report["prompts"] = prompt_summary(case_requests, window)
    report["lengths"] = length_summary(ok)
    report["memory"].update(_server_case_memory(timelines, ok, grouped))
    return report


def _case_records(
    records: list[dict[str, Any]], event_type: str
) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("case_id")): record
        for record in records
        if record.get("event_type") == event_type
    }


def _measured_windows(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("case_id")): record
        for record in records
        if record.get("event_type") == "infer.phase_window"
        and record.get("phase") == "measured"
    }


def _failures_by_status(failed: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record.get("status")) for record in failed)
    return dict(sorted(counts.items()))


def format_analysis_text(report: dict[str, Any]) -> str:
    """Render an inference analysis report as text."""
    summary = report.get("summary", {})
    lines = [
        "Inference Profile Analysis",
        "-" * 28,
        f"Total requests: {summary.get('total_requests', 0)}",
        f"Successful requests: {summary.get('successful_requests', 0)}",
        f"Failed requests: {summary.get('failed_requests', 0)}"
        + _failure_breakdown(summary.get("failures_by_status")),
        f"Failure rate: {float(summary.get('failure_rate', 0.0)):.2%}",
    ]
    cases = report.get("cases", {})
    telemetry = report.get("telemetry", {})
    join = telemetry.get("server_join", {})
    vllm = telemetry.get("vllm")
    vllm_cases = vllm.get("cases", {}) if isinstance(vllm, dict) else {}
    lines.extend(workload_lines(report.get("workload")))
    lines.append("Memory observations: client-local")
    lines.extend(_server_status_lines(join))
    lines.extend(vllm_lines(vllm))
    lines.extend(execution_lines(telemetry.get("execution")))
    if isinstance(cases, dict) and cases:
        lines.append("")
        lines.append("Cases:")
        for case_id, case in cases.items():
            lines.extend(_case_lines(case_id, case))
            lines.extend(vllm_case_lines(vllm_cases.get(case_id)))
    return "\n".join(lines)


def _failure_breakdown(by_status: Any) -> str:
    if not isinstance(by_status, dict) or not by_status:
        return ""
    parts = ", ".join(f"{status} {count}" for status, count in by_status.items())
    return f" ({parts})"


def _server_status_lines(join: dict[str, Any]) -> list[str]:
    if join.get("status") == "joined":
        return _joined_status_lines(join)
    if join.get("status") not in {None, "not_configured"}:
        return [f"Server telemetry: unjoined ({join.get('reason')})"]
    return []


def _joined_status_lines(join: dict[str, Any]) -> list[str]:
    lines = [
        "Server telemetry: joined to case windows (declared direct route, "
        f"{join.get('clock_alignment_evidence')} clock evidence)"
    ]
    counts = join.get("case_coverage")
    if isinstance(counts, dict):
        lines.append(
            f"Server coverage: {counts.get('observed', 0)} observed, "
            f"{counts.get('partial', 0)} partial, {counts.get('empty', 0)} empty"
        )
    group = join.get("group")
    if isinstance(group, dict):
        lines.append(
            f"Server group: {group.get('group_id')} "
            f"({group.get('world_size')} members)"
        )
    for label, invalidation in _invalidations(join):
        lines.append(
            f"Server identity ended{label} ({invalidation.get('detail')}); "
            "case windows after the last confirmed poll are not joined"
        )
    return lines


def _invalidations(join: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    group = join.get("group")
    if not isinstance(group, dict):
        invalidation = join.get("invalidation")
        return [("", invalidation)] if isinstance(invalidation, dict) else []
    return [
        (f" for rank {member.get('rank')}", member["invalidation"])
        for member in group.get("members", [])
        if isinstance(member.get("invalidation"), dict)
    ]


def _case_lines(case_id: str, case: Any) -> list[str]:
    latency = case.get("latency_ms", {}) if isinstance(case, dict) else {}
    throughput = case.get("throughput", {}) if isinstance(case, dict) else {}
    lines = [
        f"- {case_id}: "
        f"p50 E2E={_fmt(latency.get('e2e_p50'))} ms, "
        f"p95 E2E={_fmt(latency.get('e2e_p95'))} ms, "
        f"p50 TTFT={_fmt(latency.get('ttft_p50'))} ms, "
        f"output={_fmt(throughput.get('output_tokens_per_second'))} tok/s, "
        f"requests={_fmt(throughput.get('requests_per_second'))} req/s"
    ]
    if isinstance(case, dict):
        lines.extend(arrival_lines(case.get("arrivals"), case.get("latency_ms")))
        lines.extend(prompt_lines(case.get("prompts")))
        lines.extend(cache_lines(case.get("cache")))
    memory = case.get("memory", {}) if isinstance(case, dict) else {}
    lines.extend(_server_case_lines(memory))
    return lines


def _server_case_lines(memory: Any) -> list[str]:
    if not isinstance(memory, dict):
        return []
    if isinstance(memory.get("server_members"), list):
        return [
            line
            for member in memory["server_members"]
            for line in _server_member_lines(member)
        ]
    coverage = memory.get("server_coverage") or {}
    if coverage.get("status") == "empty":
        return [f"  server telemetry: none ({coverage.get('reason')})"]
    lines = []
    if coverage.get("status") == "partial":
        lines.append(f"  server telemetry: partial ({coverage.get('reason')})")
    for metric, observation in (memory.get("server_observations") or {}).items():
        lines.append(_server_metric_line(metric, observation))
    return lines


def _server_member_lines(member: dict[str, Any]) -> list[str]:
    where = member.get("device_uuid") or member.get("host")
    label = f"  rank {member.get('rank')} ({where})"
    coverage = member.get("coverage") or {}
    if coverage.get("status") == "empty":
        return [f"{label}: none ({coverage.get('reason')})"]
    lines = []
    if coverage.get("status") == "partial":
        lines.append(f"{label}: partial ({coverage.get('reason')})")
    for metric, observation in (member.get("observations") or {}).items():
        lines.append(_server_metric_line(metric, observation, prefix=f"{label} "))
    return lines


def _server_metric_line(
    metric: str, observation: dict[str, Any], prefix: str = "  "
) -> str:
    label = f"{prefix}{metric} ({observation.get('observation_scope')})"
    value = observation.get("maximum_recorded_bytes")
    if value is None:
        missing = observation.get("missing_samples", 0)
        return f"{label}: no valid samples ({missing} missing)"
    return (
        f"{label}: max recorded {value} bytes, "
        f"{observation.get('valid_samples')} samples"
    )


# Every profile writes a session record first; requests follow.
_ARTIFACT_EVENT_TYPES = frozenset({"infer.session", "infer.request"})


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read an inference artifact; a file that cannot be read is invalid input."""
    try:
        records = _read_jsonl(Path(path))
    except (OSError, ValueError) as exc:
        raise InferInputError(f"{path}: {_reason(exc)}") from exc
    if not any(record.get("event_type") in _ARTIFACT_EVENT_TYPES for record in records):
        raise InferInputError(
            f"{path}: not an inference artifact (no infer.session or "
            "infer.request records)"
        )
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            records.append(payload)
    return records


def _reason(exc: Exception) -> str:
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return str(exc)


def _summarize_requests(
    requests: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    e2e = _number_values(requests, "e2e_latency_ms")
    from_intended = latency_from_intended_ms(requests)
    ttft = _number_values(requests, "ttft_ms")
    first_chunk = _number_values(requests, "first_chunk_latency_ms")
    output_tokens = sum(_int_value(record.get("output_tokens")) for record in requests)
    total_tokens = sum(_int_value(record.get("total_tokens")) for record in requests)
    request_window = _request_time_window(requests)
    duration_seconds = (
        max(request_window[1] - request_window[0], 0) / 1_000_000_000
        if request_window is not None
        else 0.0
    )
    request_count = len(requests)
    output_tps = output_tokens / duration_seconds if duration_seconds > 0 else 0.0
    total_tps = total_tokens / duration_seconds if duration_seconds > 0 else 0.0
    request_rate = request_count / duration_seconds if duration_seconds > 0 else 0.0
    peak_device_used = _peak_sample_value(samples, "device_used_bytes")
    peak_process_rss = _peak_sample_value(samples, "process_rss_bytes")
    return {
        "request_count": request_count,
        "latency_ms": {
            "e2e_p50": _percentile(e2e, 50),
            "e2e_p95": _percentile(e2e, 95),
            "e2e_p99": _percentile(e2e, 99),
            "e2e_from_intended_p50": _percentile(from_intended, 50),
            "e2e_from_intended_p95": _percentile(from_intended, 95),
            "e2e_from_intended_p99": _percentile(from_intended, 99),
            "ttft_p50": _percentile(ttft, 50),
            "ttft_p95": _percentile(ttft, 95),
            "ttft_p99": _percentile(ttft, 99),
            "first_chunk_p50": _percentile(first_chunk, 50),
            "first_chunk_p95": _percentile(first_chunk, 95),
        },
        "throughput": {
            "duration_seconds": duration_seconds,
            "requests_per_second": request_rate,
            "output_tokens_per_second": output_tps,
            "total_tokens_per_second": total_tps,
        },
        "tokens": {
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "output_token_sources": sorted(
                {
                    str(record.get("output_token_source", "unknown"))
                    for record in requests
                }
            ),
        },
        "memory": {
            "observation_scope": "client_local",
            "peak_device_used_bytes": peak_device_used,
            "peak_process_rss_bytes": peak_process_rss,
        },
    }


def _samples_for_request_window(
    samples: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    request_window = _request_time_window(requests)
    if request_window is None:
        return []

    start_ns, end_ns = request_window
    return [
        sample
        for sample in samples
        if _is_number(sample.get("timestamp_ns"))
        and start_ns <= _int_value(sample.get("timestamp_ns")) <= end_ns
    ]


def _request_time_window(requests: list[dict[str, Any]]) -> tuple[int, int] | None:
    bounds: list[tuple[int, int]] = []
    for record in requests:
        started_at = record.get("started_at_ns")
        ended_at = record.get("ended_at_ns")
        if not _is_number(started_at) or not _is_number(ended_at):
            continue
        bounds.append((_int_value(started_at), _int_value(ended_at)))
    if not bounds:
        return None
    return min(start for start, _end in bounds), max(end for _start, end in bounds)


def _peak_sample_value(samples: list[dict[str, Any]], field: str) -> int | None:
    return max(
        (
            _int_value(sample.get(field))
            for sample in samples
            if _is_number(sample.get(field))
        ),
        default=None,
    )


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "-"


def _partition_inference_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests = [
        record
        for record in records
        if record.get("event_type") == "infer.request"
        and record.get("phase") == "measured"
    ]
    samples = [
        record
        for record in records
        if record.get("event_type") == "infer.system_sample"
    ]
    return requests, samples


def _load_server_samples(paths: Iterable[str | Path]) -> list[TelemetrySample]:
    """Load every artifact; drop exact duplicates, e.g. a file passed twice."""
    return list(
        dict.fromkeys(sample for path in paths for sample in _server_samples(path))
    )


def _server_samples(path: str | Path) -> list[TelemetrySample]:
    try:
        return load_telemetry(path)
    except (OSError, ValueError) as exc:
        raise InferInputError(f"--server-telemetry {path}: {_reason(exc)}") from exc


@dataclass(frozen=True)
class _Member:
    """One joined server identity, its clock and its samples on the client clock."""

    identity: ServerIdentity
    samples: list[TelemetrySample]
    clock: ServerClock
    placed: SampleAlignment


def _server_join(
    records: list[dict[str, Any]],
    samples: list[TelemetrySample],
    *,
    direct_server: bool,
    clock_offset_ns: int | None,
    clock_uncertainty_ns: int | None,
) -> tuple[dict[str, Any], list[_Member]]:
    """Return a case-window join for one server or one declared group.

    The second value lists the joined members, ordered by rank.
    """
    if not samples:
        return {"status": "not_configured"}, []
    artifact = _artifact_record(records)
    if artifact is None:
        return _unjoined("missing_run_identity"), []
    issue = _run_id_issue(artifact, samples) or _identity_issue(samples, direct_server)
    if issue is not None:
        return issue, []
    by_identity = group_members(samples)
    clocks = _member_clocks(
        artifact, records, list(by_identity), clock_offset_ns, clock_uncertainty_ns
    )
    members = clocks if isinstance(clocks, str) else _place(by_identity, clocks)
    if isinstance(members, str):
        return _unjoined(members), []
    return _joined(members), members


def _place(
    by_identity: dict[ServerIdentity, list[TelemetrySample]],
    clocks: dict[ServerIdentity, ServerClock],
) -> list[_Member] | str:
    """Place each member's samples; every member needs at least one placed."""
    members = [
        _Member(
            identity, found, clocks[identity], align_samples(found, clocks[identity])
        )
        for identity, found in by_identity.items()
    ]
    for member in members:
        if not member.placed.aligned:
            ambiguous = member.placed.unaligned[AMBIGUOUS]
            return f"clock_alignment_{'ambiguous' if ambiguous else 'uncovered'}"
    return members


def _unjoined(reason: str) -> dict[str, Any]:
    return {"status": "unjoined", "reason": reason}


def _identity_issue(
    samples: list[TelemetrySample], direct_server: bool
) -> dict[str, Any] | None:
    membership = membership_issue(samples)
    if membership is not None:
        return _unjoined(membership)
    route_issue = _route_issue(samples, direct_server)
    return _unjoined(route_issue) if route_issue else None


def _member_clocks(
    artifact: dict[str, Any],
    records: list[dict[str, Any]],
    identities: list[ServerIdentity],
    offset_ns: int | None,
    uncertainty_ns: int | None,
) -> dict[ServerIdentity, ServerClock] | str:
    """Build one clock per server domain; the flags describe one remote domain."""
    client_domain = client_clock_domain(artifact)
    if client_domain is None:
        return "missing_client_clock_domain"
    domains = sorted({identity.clock_domain for identity in identities})
    flags_domain = _flags_domain(domains, client_domain, offset_ns)
    if flags_domain is None:
        return "clock_flags_ambiguous"
    run_id = str(artifact["context"].get("run_id"))
    recorded = artifact_alignments(records, run_id)
    clocks: dict[str, ServerClock] = {}
    for domain in domains:
        flags = (offset_ns, uncertainty_ns) if domain == flags_domain else (None, None)
        clock = build_server_clock(
            run_id=run_id,
            server_domain=domain,
            client_domain=client_domain,
            recorded=recorded,
            offset_ns=flags[0],
            uncertainty_ns=flags[1],
        )
        if isinstance(clock, str):
            return clock
        clocks[domain] = clock
    return {identity: clocks[identity.clock_domain] for identity in identities}


def _flags_domain(
    domains: list[str], client_domain: str, offset_ns: int | None
) -> str | None:
    """The server domain the clock flags describe, or None if that is unclear.

    Members on the client's host and boot share its clock; the flags then
    belong to the single remote domain. A member with the client's hostname
    but no boot ID counts as remote, because only the flags can align it. With
    several remote hosts, each needs its own ``infer.clock_alignment`` record.
    """
    remote = [
        domain
        for domain in domains
        if domain != client_domain or not is_boot_qualified(domain)
    ]
    if offset_ns is not None and len(remote) > 1:
        return None
    return remote[0] if remote else client_domain


def _joined(members: list[_Member]) -> dict[str, Any]:
    return {
        "status": "joined",
        "route_evidence": "operator_declared_direct",
        **_membership_report(members),
        **_clock_report(members),
        "attribution": "case_window_observation_only",
    }


def _membership_report(members: list[_Member]) -> dict[str, Any]:
    first = members[0].identity
    if first.group_id is None:
        return {
            "identity": asdict(first),
            "invalidation": _invalidation(members[0].samples),
        }
    return {
        "group": {
            "group_id": first.group_id,
            "world_size": first.world_size,
            "members": [
                {
                    "rank": member.identity.rank,
                    "identity": asdict(member.identity),
                    "clock_alignment_evidence": member.clock.evidence,
                    "invalidation": _invalidation(member.samples),
                }
                for member in members
            ],
        }
    }


def _clock_report(members: list[_Member]) -> dict[str, Any]:
    applied = _merge_applied(
        [item for member in members for item in member.placed.applied]
    )
    offsets = {item["offset_ns"] for item in applied}
    report: dict[str, Any] = {
        "clock_alignment_evidence": _evidence(members),
        # One offset when a single alignment placed every joined sample.
        "clock_offset_ns": next(iter(offsets)) if len(offsets) == 1 else None,
        "clock_uncertainty_ns": _max_uncertainty(members),
        "clock_alignments": applied,
        "unaligned_samples": _unaligned(members),
    }
    for key, field in (
        ("overridden_clock_alignments", "overridden"),
        ("ignored_clock_alignments", "ignored"),
    ):
        event_ids = _record_ids(members, field)
        if event_ids:
            report[key] = event_ids
    return report


def _record_ids(members: list[_Member], field: str) -> list[str]:
    """Event IDs a member's clock replaced or ignored, across the whole join."""
    return sorted({item for member in members for item in getattr(member.clock, field)})


def _max_uncertainty(members: list[_Member]) -> int:
    return max(
        placed.uncertainty_ns
        for member in members
        for placed in member.placed.aligned.values()
    )


def _unaligned(members: list[_Member]) -> dict[str, int]:
    return {
        key: sum(member.placed.unaligned[key] for member in members)
        for key in (UNCOVERED, AMBIGUOUS)
    }


def _evidence(members: list[_Member]) -> str:
    evidence = sorted({member.clock.evidence for member in members})
    return evidence[0] if len(evidence) == 1 else "mixed"


def _merge_applied(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine one alignment used by several members into one entry."""
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in items:
        key = tuple(value for name, value in item.items() if name != "aligned_samples")
        if key in merged:
            merged[key]["aligned_samples"] += item["aligned_samples"]
        else:
            merged[key] = dict(item)
    return list(merged.values())


def _run_id_issue(
    artifact: dict[str, Any], samples: list[TelemetrySample]
) -> dict[str, Any] | None:
    """A different run ID leaves the server data out; the client report stays."""
    artifact_run_id = artifact["context"].get("run_id")
    telemetry_run_ids = sorted({sample.run_id for sample in samples})
    if telemetry_run_ids == [artifact_run_id]:
        return None
    return {
        "status": "unjoined",
        "reason": "run_id_mismatch",
        "artifact_run_id": artifact_run_id,
        "telemetry_run_ids": telemetry_run_ids,
    }


def _route_issue(samples: list[TelemetrySample], direct_server: bool) -> str | None:
    if not direct_server:
        return "route_not_declared"
    if not any(sample.state == "valid" for sample in samples):
        return "no_valid_server_samples"
    return None


def _invalidation(samples: list[TelemetrySample]) -> dict[str, Any] | None:
    """Locate where the observed process or GPU stopped being the original one.

    Samples before the first ``invalid`` sample describe the original identity,
    so an invalidation only affects case windows that could reach past the last
    poll that still confirmed it.
    """
    first = min(
        (sample for sample in samples if sample.state == "invalid"),
        key=lambda sample: sample.observed_at_ns,
        default=None,
    )
    if first is None:
        return None
    confirmed = [
        sample.observed_at_ns
        for sample in samples
        if sample.observed_at_ns < first.observed_at_ns
    ]
    return {
        "observed_at_ns": first.observed_at_ns,
        "detail": first.detail,
        "last_confirmed_at_ns": max(confirmed, default=None),
    }


def _artifact_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    artifacts = [r for r in records if r.get("event_type") == "infer.artifact"]
    if len(artifacts) != 1 or not isinstance(artifacts[0].get("context"), dict):
        return None
    return artifacts[0]


def _server_targets(samples: list[TelemetrySample]) -> list[dict[str, Any]]:
    grouped: dict[ServerIdentity, list[TelemetrySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.identity, []).append(sample)
    targets: list[dict[str, Any]] = []
    for identity in sorted(
        grouped,
        key=lambda item: (
            item.host,
            item.pid,
            item.process_start_ns,
            item.device_uuid or "",
        ),
    ):
        selected = grouped[identity]
        states = Counter(s.state for s in selected)
        targets.append(
            {
                "identity": asdict(identity),
                "metrics": sorted({s.metric for s in selected}),
                "sample_states": {
                    state: states[state]
                    for state in ("valid", "missing", "stale", "invalid")
                },
            }
        )
    return targets


@dataclass(frozen=True)
class _ServerTimeline:
    """A joined collector's polls in client time, and how long it can be trusted.

    Each sample keeps the uncertainty of the alignment that placed it, so one
    imprecise alignment does not widen the margins of every other sample.
    """

    placed: dict[TelemetrySample, AlignedTimestamp]
    # Samples no single alignment placed, and the offsets that placed others.
    unplaced: dict[TelemetrySample, str]
    offsets: tuple[int, ...]
    slack_ns: int
    first_poll_ns: int
    last_poll_ns: int
    invalidated: bool
    trusted_until_ns: int | None


def _member_timeline(member: _Member) -> _ServerTimeline:
    aligned = member.placed.aligned
    values = [placed.value_ns for placed in aligned.values()]
    invalidation = _invalidation(member.samples)
    return _ServerTimeline(
        placed=aligned,
        unplaced=member.placed.unplaced,
        offsets=tuple(sorted({item["offset_ns"] for item in member.placed.applied})),
        slack_ns=max(sample.interval_ms for sample in member.samples) * 1_000_000,
        first_poll_ns=min(values),
        last_poll_ns=max(values),
        invalidated=invalidation is not None,
        trusted_until_ns=_trusted_until(aligned, invalidation),
    )


def _trusted_until(
    aligned: dict[TelemetrySample, AlignedTimestamp],
    invalidation: dict[str, Any] | None,
) -> int | None:
    """Client time by which the last confirming poll had certainly happened."""
    if invalidation is None:
        return None
    confirmed = [
        # A poll could have happened up to its own uncertainty earlier.
        placed.value_ns - placed.uncertainty_ns
        for sample, placed in aligned.items()
        if sample.observed_at_ns < invalidation["observed_at_ns"]
    ]
    return max(confirmed) if confirmed else None


def _server_case_memory(
    timelines: list[tuple[_Member, _ServerTimeline]],
    requests: list[dict[str, Any]],
    declared_group: bool,
) -> dict[str, Any]:
    """Server values for one case: one server's, or one entry per group member.

    Values from different members are never combined: separate collectors
    sample at different instants, so no per-case total would be well defined.
    """
    if not timelines:
        return {"server_observations": {}, "server_coverage": {"status": "not_joined"}}
    views = []
    for member, timeline in timelines:
        observations, coverage = _server_case_view(member.samples, requests, timeline)
        views.append((member, observations, coverage))
    if not declared_group:
        _member, observations, coverage = views[0]
        return {"server_observations": observations, "server_coverage": coverage}
    return {
        "server_observations": {},
        "server_coverage": _group_coverage([coverage for _, _, coverage in views]),
        "server_members": [
            _member_view(member, observations, coverage)
            for member, observations, coverage in views
        ],
    }


def _member_view(
    member: _Member, observations: dict[str, Any], coverage: dict[str, Any]
) -> dict[str, Any]:
    identity = member.identity
    return {
        "rank": identity.rank,
        "host": identity.host,
        "pid": identity.pid,
        "device_uuid": identity.device_uuid,
        "gpu_instance_id": identity.gpu_instance_id,
        "observations": observations,
        "coverage": coverage,
    }


def _group_coverage(coverages: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {coverage["status"] for coverage in coverages}
    if statuses == {"observed"}:
        return {"status": "observed", "reason": None}
    if statuses == {"empty"}:
        return {"status": "empty", "reason": "no_member_observed"}
    return {"status": "partial", "reason": "some_members_not_fully_observed"}


def _server_case_view(
    samples: list[TelemetrySample],
    requests: list[dict[str, Any]],
    timeline: _ServerTimeline,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize server samples in one case window and say how well it is covered.

    A sample counts only if its own clock uncertainty cannot move it outside
    the window, so a short window or a collector that was not running can
    leave a joined case without server values; the coverage record says which.
    """
    window = _request_time_window(requests)
    if window is None:
        return {}, {"status": "empty", "reason": "no_request_window"}
    start_ns, end_ns = window
    in_window = [
        sample
        for sample, placed in timeline.placed.items()
        if start_ns + placed.uncertainty_ns
        <= placed.value_ns
        <= end_ns - placed.uncertainty_ns
    ]
    margin = _case_margin(timeline, start_ns, end_ns)
    low, high = start_ns + margin, end_ns - margin
    gap = _unplaced_reason(timeline, start_ns, end_ns)
    coverage = _case_coverage(end_ns, low, high, bool(in_window), gap, timeline)
    if coverage["status"] == "empty":
        return {}, coverage
    return _metric_summaries(samples, in_window), coverage


def _case_margin(timeline: _ServerTimeline, start_ns: int, end_ns: int) -> int:
    """The clock uncertainty that describes one case window's coverage.

    It is the smallest uncertainty among samples placed inside the window, or
    among all samples when none are, so the counted window is the widest any
    sample could qualify for.
    """
    inside = [
        placed.uncertainty_ns
        for placed in timeline.placed.values()
        if start_ns <= placed.value_ns <= end_ns
    ]
    return min(inside or [placed.uncertainty_ns for placed in timeline.placed.values()])


def _unplaced_reason(
    timeline: _ServerTimeline, start_ns: int, end_ns: int
) -> str | None:
    """Say why a case has no samples when unplaced polls probably fell inside it.

    Unplaced samples have no client time; any offset that placed other samples
    gives the best estimate of where they would land.
    """
    near = [
        reason
        for sample, reason in timeline.unplaced.items()
        if any(
            start_ns <= sample.observed_at_ns + offset <= end_ns
            for offset in timeline.offsets
        )
    ]
    if not near:
        return None
    return f"clock_alignment_{AMBIGUOUS if AMBIGUOUS in near else UNCOVERED}"


def _case_coverage(
    end_ns: int,
    low: int,
    high: int,
    has_samples: bool,
    unplaced_reason: str | None,
    timeline: _ServerTimeline,
) -> dict[str, Any]:
    empty_reason = _empty_reason(
        end_ns, low, high, has_samples, unplaced_reason, timeline
    )
    if empty_reason is not None:
        return {
            "status": "empty",
            "reason": empty_reason,
            "counted_window_ns": [low, high] if high >= low else None,
        }
    partial_reason = _partial_reason(low, high, timeline)
    return {
        "status": "observed" if partial_reason is None else "partial",
        "reason": partial_reason,
        "counted_window_ns": [low, high],
    }


def _empty_reason(
    end_ns: int,
    low: int,
    high: int,
    has_samples: bool,
    unplaced_reason: str | None,
    timeline: _ServerTimeline,
) -> str | None:
    if high < low:
        return "window_shorter_than_uncertainty"
    if timeline.invalidated and (
        timeline.trusted_until_ns is None or end_ns > timeline.trusted_until_ns
    ):
        return "identity_invalidated"
    if not has_samples:
        return unplaced_reason or "no_collector_coverage"
    return None


def _partial_reason(low: int, high: int, timeline: _ServerTimeline) -> str | None:
    if timeline.first_poll_ns > low + timeline.slack_ns:
        return "collector_started_after_window_start"
    if timeline.last_poll_ns < high - timeline.slack_ns:
        return "collector_stopped_before_window_end"
    return None


def _coverage_counts(cases: dict[str, Any]) -> dict[str, int]:
    statuses = Counter(
        case["memory"]["server_coverage"]["status"] for case in cases.values()
    )
    return {status: statuses[status] for status in ("observed", "partial", "empty")}


def _metric_summaries(
    samples: list[TelemetrySample], in_window: list[TelemetrySample]
) -> dict[str, Any]:
    by_metric: dict[str, TelemetrySample] = {}
    for sample in samples:
        by_metric.setdefault(sample.metric, sample)
    window_by_metric: dict[str, list[TelemetrySample]] = {}
    for sample in in_window:
        window_by_metric.setdefault(sample.metric, []).append(sample)
    return {
        metric: _summarize_server_metric(first, window_by_metric.get(metric, []))
        for metric, first in sorted(by_metric.items())
    }


def _summarize_server_metric(
    first: TelemetrySample,
    window_samples: list[TelemetrySample],
) -> dict[str, Any]:
    """Describe one metric using only the samples inside the counted window."""
    states = Counter(sample.state for sample in window_samples)
    valid_values = [
        sample.value_bytes
        for sample in window_samples
        if sample.state == "valid" and sample.value_bytes is not None
    ]
    return {
        "observation_scope": first.scope,
        "counter_owner": first.counter_owner,
        "provenance": sorted({sample.provenance for sample in window_samples}),
        "sources": sorted({sample.source for sample in window_samples}),
        "maximum_recorded_bytes": max(valid_values, default=None),
        "valid_samples": len(valid_values),
        "missing_samples": states["missing"],
        "stale_samples": states["stale"],
        "invalid_samples": states["invalid"],
        "intervals_ms": sorted({sample.interval_ms for sample in window_samples}),
    }
