"""The run manifest: which server a run measured, before, after and declared.

An artifact carries ``infer.manifest`` records, appended and never changed:

- ``before``: a server description taken before the run, given to
  ``infer profile --describe-server``;
- ``after``: one taken after the run, attached by ``infer attach-manifest``;
- ``declared``: what the operator states and nothing observed, given to
  ``infer profile --declare``.

An ``after`` description must be of the same server lifetime as the
``before`` one (same host boot, same API server PID and start), of the same
run when it names one, and taken once the last measured phase had ended;
otherwise it is refused. Between the two, a setting that identifies the
server (GPU settings, driver, model files, launch arguments, packages,
start-up choices) must not change; one that drifts (SM clock, temperature,
clock event reasons) is reported as drift.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .describe_server import load_description
from .errors import InferInputError

MANIFEST_EVENT = "infer.manifest"
BEFORE = "before"
AFTER = "after"
DECLARED = "declared"
ROLES = (BEFORE, AFTER, DECLARED)
DECLARED_FORMAT = "stormlog.infer.declared"
DECLARED_VERSION = 1
IDENTITY_CHANGED = "identity_changed"

_MEASURED = "measured"


def description_record(
    description: Mapping[str, Any], *, role: str, session_id: str, run_id: str
) -> dict[str, Any]:
    """An ``infer.manifest`` record that carries a whole server description."""
    if role not in (BEFORE, AFTER):
        raise ValueError(f"a description is a {BEFORE} or {AFTER} manifest")
    host = description.get("host") or {}
    server = description.get("server") or {}
    return {
        "event_type": MANIFEST_EVENT,
        "role": role,
        "session_id": session_id,
        "run_id": run_id,
        "observed_at_ns": description.get("observed_at_ns"),
        "sha256": description.get("sha256"),
        "host": {"hostname": host.get("hostname"), "boot_id": host.get("boot_id")},
        "server": {"pid": server.get("pid"), "start_ticks": server.get("start_ticks")},
        "gpu_uuids": (description.get("gpus") or {}).get("server_uuids", []),
        "description": dict(description),
    }


def load_declarations(path: Path) -> dict[str, Any]:
    """A ``stormlog.infer.declared`` v1 file: field names and their values."""
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise InferInputError(f"--declare {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("format") != DECLARED_FORMAT:
        raise InferInputError(f"--declare {path}: not {DECLARED_FORMAT}")
    if document.get("version") != DECLARED_VERSION:
        raise InferInputError(
            f"--declare {path}: version {document.get('version')!r} is not "
            f"{DECLARED_VERSION}"
        )
    fields = document.get("fields")
    if not isinstance(fields, dict) or not all(isinstance(k, str) for k in fields):
        raise InferInputError(f"--declare {path}: fields must be an object")
    return document


def declared_record(
    declarations: Mapping[str, Any], *, session_id: str, run_id: str
) -> dict[str, Any]:
    return {
        "event_type": MANIFEST_EVENT,
        "role": DECLARED,
        "session_id": session_id,
        "run_id": run_id,
        "provenance": "declared",
        "fields": dict(declarations.get("fields") or {}),
    }


def manifests(
    records: Iterable[Mapping[str, Any]]
) -> dict[str, list[Mapping[str, Any]]]:
    """The artifact's manifest records by role, in artifact order."""
    found: dict[str, list[Mapping[str, Any]]] = {role: [] for role in ROLES}
    for record in records:
        if record.get("event_type") == MANIFEST_EVENT and record.get("role") in found:
            found[str(record["role"])].append(record)
    return found


def after_refusals(
    records: Sequence[Mapping[str, Any]], description: Mapping[str, Any], *, run_id: str
) -> list[str]:
    """Why an ``after`` description cannot be attached; empty when it can."""
    by_role = manifests(records)
    if not by_role[BEFORE]:
        return [
            "the artifact has no before description; profile with --describe-server"
        ]
    refusals = _lifetime_refusals(
        description, by_role[BEFORE][-1]["description"], run_id
    )
    ended = last_measured_end_ns(records)
    observed = description.get("observed_at_ns")
    if ended is not None and (not isinstance(observed, int) or observed < ended):
        refusals.append("it was taken before the last measured phase ended")
    if any(
        record.get("sha256") == description.get("sha256") for record in by_role[AFTER]
    ):
        refusals.append("it is already attached")
    return refusals


def _lifetime_refusals(
    description: Mapping[str, Any], before: Mapping[str, Any], run_id: str
) -> list[str]:
    """The same run, host boot and server process as the before description."""
    refusals = []
    named_run = description.get("run_id")
    if named_run is not None and named_run != run_id:
        refusals.append(f"it describes run {named_run}, not {run_id}")
    if _host(description) != _host(before):
        refusals.append("it was taken on another host or boot than the before one")
    if _lifetime(description) != _lifetime(before):
        refusals.append(
            "its server process (PID and start time) is not the before one's: "
            "the server was restarted"
        )
    return refusals


def last_measured_end_ns(records: Iterable[Mapping[str, Any]]) -> int | None:
    """When the last measured phase's drain ended, by the client's clock."""
    ends = [
        int(record["drained_at_ns"])
        for record in records
        if record.get("event_type") == "infer.phase_window"
        and record.get("phase") == _MEASURED
        and isinstance(record.get("drained_at_ns"), int)
    ]
    return max(ends, default=None)


def attach_manifest(artifact: Path, description_path: Path) -> dict[str, Any]:
    """Append ``description_path`` to ``artifact`` as its ``after`` manifest."""
    description = load_description(description_path)
    records = _read_records(artifact)
    identity = next(
        (r for r in records if r.get("event_type") == "infer.artifact"), None
    )
    if identity is None:
        raise InferInputError(f"{artifact}: no infer.artifact record")
    context = identity.get("context") or {}
    run_id, session_id = str(context.get("run_id")), str(context.get("session_id"))
    refusals = after_refusals(records, description, run_id=run_id)
    if refusals:
        raise InferInputError(
            f"{description_path} cannot be attached to {artifact}: "
            + "; ".join(refusals)
        )
    record = description_record(
        description, role=AFTER, session_id=session_id, run_id=run_id
    )
    try:
        with artifact.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        raise InferInputError(f"{artifact}: cannot append ({exc})") from exc
    return record


def compare_descriptions(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """What identifies the server and changed, and what drifted."""
    first, second = identity_fields(before), identity_fields(after)
    changes = [
        {"field": name, "before": first.get(name), "after": second.get(name)}
        for name in sorted(set(first) | set(second))
        if first.get(name) != second.get(name)
    ]
    return {"identity_changes": changes, "drift": _drift(before, after)}


def identity_fields(description: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of a description that must not change within a run."""
    gpus = description.get("gpus") or {}
    model = description.get("model") or {}
    runtime = description.get("runtime") or {}
    log = description.get("log") or {}
    fields: dict[str, Any] = {
        "gpu.driver_version": gpus.get("driver_version"),
        "gpu.cuda_driver_version": gpus.get("cuda_driver_version"),
        "gpu.server_uuids": gpus.get("server_uuids"),
        "model.resolved_snapshot": model.get("resolved_snapshot"),
        "model.weights_digest": model.get("weights_digest"),
        "model.chat_template_digest": model.get("chat_template_digest"),
        "server.launch": (description.get("server") or {}).get("launch"),
        "runtime.python": runtime.get("python"),
        "runtime.packages": runtime.get("packages"),
        "log.attention_backend": log.get("attention_backend"),
        "log.kv_cache_size_tokens": log.get("kv_cache_size_tokens"),
    }
    for device in _server_devices(description):
        for name, value in (device.get("settings") or {}).items():
            fields[f"gpu.{device.get('uuid')}.{name}"] = value
    return fields


def manifest_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The report's ``manifest`` block, or None when the run has none."""
    by_role = manifests(records)
    if not any(by_role.values()):
        return None
    before = by_role[BEFORE][-1] if by_role[BEFORE] else None
    after = by_role[AFTER][-1] if by_role[AFTER] else None
    summary: dict[str, Any] = {
        role: [_reference(record) for record in by_role[role]]
        for role in (BEFORE, AFTER)
    }
    summary[DECLARED] = by_role[DECLARED][-1]["fields"] if by_role[DECLARED] else None
    if before is not None and after is not None:
        compared = compare_descriptions(before["description"], after["description"])
        summary.update(compared)
        summary["protocol_failure"] = (
            IDENTITY_CHANGED if compared["identity_changes"] else None
        )
    return summary


def manifest_lines(summary: Any) -> list[str]:
    """The text report's lines for a ``manifest`` block."""
    if not isinstance(summary, dict):
        return []
    roles = [f"{role} {len(summary.get(role) or [])}" for role in (BEFORE, AFTER)]
    roles.append(f"declared {'yes' if summary.get(DECLARED) else 'no'}")
    lines = ["Server manifest: " + ", ".join(roles)]
    changes = summary.get("identity_changes") or []
    if changes:
        names = ", ".join(str(change.get("field")) for change in changes)
        lines.append(f"  identity changed during the run (protocol failure): {names}")
    return lines + _drift_lines(summary.get("drift") or {})


def _drift_lines(drift: Mapping[str, Any]) -> list[str]:
    lines = []
    for uuid, series in sorted(drift.items()):
        moved = [
            f"{name} {values.get('before')} -> {values.get('after')}"
            for name, values in series.items()
            if values.get("before") != values.get("after")
        ]
        if moved:
            lines.append(f"  drift on {uuid}: " + ", ".join(moved))
    return lines


def _reference(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sha256": record.get("sha256"),
        "observed_at_ns": record.get("observed_at_ns"),
        "server": record.get("server"),
        "gpu_uuids": record.get("gpu_uuids"),
    }


def _drift(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    first = {d.get("uuid"): d.get("series") or {} for d in _server_devices(before)}
    drift: dict[str, Any] = {}
    for device in _server_devices(after):
        earlier = first.get(device.get("uuid"))
        if earlier is None:
            continue
        later = device.get("series") or {}
        drift[str(device.get("uuid"))] = {
            name: {"before": earlier.get(name), "after": later.get(name)}
            for name in sorted(set(earlier) | set(later))
        }
    return drift


def _server_devices(description: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    devices = (description.get("gpus") or {}).get("devices") or []
    return [device for device in devices if device.get("server_pids")]


def _host(description: Mapping[str, Any]) -> tuple[Any, Any]:
    host = description.get("host") or {}
    return host.get("hostname"), host.get("boot_id")


def _lifetime(description: Mapping[str, Any]) -> tuple[Any, Any]:
    server = description.get("server") or {}
    return server.get("pid"), server.get("start_ticks")


def _read_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as exc:
        raise InferInputError(f"{path}: {exc}") from exc
    return [record for record in records if isinstance(record, dict)]


__all__ = [
    "AFTER",
    "BEFORE",
    "DECLARED",
    "DECLARED_FORMAT",
    "IDENTITY_CHANGED",
    "MANIFEST_EVENT",
    "ROLES",
    "after_refusals",
    "attach_manifest",
    "compare_descriptions",
    "declared_record",
    "description_record",
    "identity_fields",
    "last_measured_end_ns",
    "load_declarations",
    "manifest_lines",
    "manifest_summary",
    "manifests",
]
