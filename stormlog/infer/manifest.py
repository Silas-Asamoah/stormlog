"""The run manifest: which server a run measured, before, after and declared.

An artifact carries ``infer.manifest`` records, appended and never changed:

- ``before``: a server description taken before the run, given to
  ``infer profile --describe-server``;
- ``after``: one taken after the run, attached by ``infer attach-manifest``;
- ``declared``: what the operator states and nothing observed, given to
  ``infer profile --declare``.

A ``before`` description must not name another run. An ``after`` one must
be of the same server lifetime as the ``before`` one (same host boot, same
API server PID and start), of the same run when it names one, another
document than any already recorded, and taken once the last measured phase
had ended: later than the ``before`` one by at least as long as the
measured phases took, each interval on its own clock. Otherwise it is
refused. Between the two, a setting that identifies the server (GPU
settings, driver, model files, launch arguments, packages, start-up
choices) must not change; one that drifts (SM clock, temperature, clock
event reasons) is reported as drift. A setting only one of them could read
is unverified, not a change.

The ``before`` description is also checked against what the server told
the probe at the start of the run: the model it serves, its vLLM version
and the GPU driver. A mismatch means the description is of another server,
a protocol failure.
"""

from __future__ import annotations

import email.utils
import ipaddress
import json
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .describe_server import load_description
from .errors import InferInputError

MANIFEST_EVENT = "infer.manifest"
# The experiment runner's record of weights it verified before it launched
# a server: the only evidence that binds weights to what the server loaded.
MODEL_IDENTITY_EVENT = "infer.model_identity"
BEFORE = "before"
AFTER = "after"
DECLARED = "declared"
ROLES = (BEFORE, AFTER, DECLARED)
DECLARED_FORMAT = "stormlog.infer.declared"
DECLARED_VERSION = 1
IDENTITY_CHANGED = "identity_changed"
DESCRIPTION_MISMATCH = "description_mismatch"

_MEASURED = "measured"
# Clocks tick at slightly different rates; this much shortfall is allowed.
_ELAPSED_TOLERANCE_NS = 1_000_000_000
# An HTTP Date header's resolution: whole seconds.
_DATE_RESOLUTION_NS = 1_000_000_000


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


def model_identity_record(
    model: Mapping[str, Any],
    *,
    session_id: str,
    run_id: str,
    server: Mapping[str, Any],
    boot_id: str | None,
) -> dict[str, Any]:
    """An ``infer.model_identity`` record, bound to the server it launched.

    ``model`` is the verified model section (weights digest, snapshot,
    evidence); ``server`` the launched API server's ``pid`` and
    ``start_ticks``, on the boot ``boot_id``.
    """
    return {
        "event_type": MODEL_IDENTITY_EVENT,
        "session_id": session_id,
        "run_id": run_id,
        "server": {"pid": server.get("pid"), "start_ticks": server.get("start_ticks")},
        "boot_id": boot_id,
        "model": dict(model),
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


def before_refusals(description: Mapping[str, Any], *, run_id: str) -> list[str]:
    """Why a description cannot be a run's ``before`` one; empty when it can."""
    named_run = description.get("run_id")
    if named_run is not None and named_run != run_id:
        return [f"it describes run {named_run}, not {run_id}"]
    return []


def after_refusals(
    records: Sequence[Mapping[str, Any]], description: Mapping[str, Any], *, run_id: str
) -> list[str]:
    """Why an ``after`` description cannot be attached; empty when it can."""
    by_role = manifests(records)
    if not by_role[BEFORE]:
        return [
            "the artifact has no before description; profile with --describe-server"
        ]
    before = by_role[BEFORE][-1]["description"]
    refusals = _lifetime_refusals(description, before, run_id)
    refusals.extend(_document_refusals(description, by_role))
    refusals.extend(_timing_refusals(records, description, before))
    return refusals


def _document_refusals(
    description: Mapping[str, Any], by_role: Mapping[str, list[Mapping[str, Any]]]
) -> list[str]:
    sha = description.get("sha256")
    if any(record.get("sha256") == sha for record in by_role[BEFORE]):
        return ["it is the before description"]
    if any(record.get("sha256") == sha for record in by_role[AFTER]):
        return ["it is already attached"]
    return []


def _timing_refusals(
    records: Sequence[Mapping[str, Any]],
    description: Mapping[str, Any],
    before: Mapping[str, Any],
) -> list[str]:
    """Taken after the run: by the client's end, by the server's own clock,
    and by each clock's interval.

    The last two need no agreement between the clocks: the server stamped
    the after probe, asked once the run had ended, with its own Date; and
    the server's interval between the descriptions must cover the client's
    measured run.
    """
    observed = description.get("observed_at_ns")
    if not isinstance(observed, int):
        return ["it has no observation time"]
    ended = last_measured_end_ns(records)
    if ended is not None and observed < ended:
        return ["it was taken before the last measured phase ended"]
    answered = _after_probe_answered_ns(records)
    if answered is not None and observed < answered - _DATE_RESOLUTION_NS:
        return [
            "it was taken before the server answered the after probe, by the "
            "server's own clock"
        ]
    began = before.get("observed_at_ns")
    if not isinstance(began, int):
        return []
    if observed <= began:
        return ["it was taken before the before description"]
    span = run_span_ns(records)
    if span is not None and observed - began < span - _ELAPSED_TOLERANCE_NS:
        return [
            f"it was taken {(observed - began) / 1e9:.1f} s after the before "
            f"description, but the run's measured phases took {span / 1e9:.1f} s"
        ]
    return []


def _after_probe_answered_ns(records: Iterable[Mapping[str, Any]]) -> int | None:
    """The earliest Date the server stamped on the after probe's answers."""
    stamps = [
        _http_date_ns(answer.get("date"))
        for record in records
        if record.get("event_type") == "infer.server_probe"
        and record.get("phase") == "after"
        for answer in (record.get("answers") or {}).values()
        if isinstance(answer, Mapping)
    ]
    known = [stamp for stamp in stamps if stamp is not None]
    return min(known) if known else None


def _http_date_ns(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return int(email.utils.parsedate_to_datetime(value).timestamp()) * 10**9
    except (TypeError, ValueError):
        return None


def run_span_ns(records: Iterable[Mapping[str, Any]]) -> int | None:
    """How long the measured run took, by the client's clock.

    The measured phases' start to their last drain end; without phase
    windows (an interrupted run), the measured requests' first start to
    their last end. Requests lie inside their windows, so both count.
    """
    bounds = [
        found
        for found in (_bounds(r) for r in records if r.get("phase") == _MEASURED)
        if found is not None
    ]
    if not bounds:
        return None
    return max(end for _start, end in bounds) - min(start for start, _end in bounds)


_SPAN_FIELDS = {
    "infer.phase_window": ("started_at_ns", "drained_at_ns"),
    "infer.request": ("started_at_ns", "ended_at_ns"),
}


def _bounds(record: Mapping[str, Any]) -> tuple[int, int] | None:
    fields = _SPAN_FIELDS.get(str(record.get("event_type")))
    if fields is None:
        return None
    start, end = record.get(fields[0]), record.get(fields[1])
    if isinstance(start, int) and isinstance(end, int):
        return start, end
    return None


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


def attach_manifest(
    artifact: Path, description_path: Path, *, run_id: str | None = None
) -> dict[str, Any]:
    """Append ``description_path`` to ``artifact`` as its ``after`` manifest.

    ``run_id`` is the run the description must be of: by default the
    artifact's own. The experiment runner, which names its descriptions and
    the ``before`` manifest it attaches after its own run, passes that name.
    """
    description = load_description(description_path)
    records = _read_records(artifact)
    identity = next(
        (r for r in records if r.get("event_type") == "infer.artifact"), None
    )
    if identity is None:
        raise InferInputError(f"{artifact}: no infer.artifact record")
    context = identity.get("context") or {}
    session_id = str(context.get("session_id"))
    if run_id is None:
        run_id = str(context.get("run_id"))
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
    """What identifies the server and changed, what only one side could read,
    and what drifted.

    A field one description could not read (taken without ``--server-log``,
    ``--python``, or a GPU field NVML did not answer) is unverified: missing
    evidence is no change.
    """
    first, second = identity_fields(before), identity_fields(after)
    changes: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for name in sorted(set(first) | set(second)):
        a, b = first.get(name), second.get(name)
        if a == b:
            continue
        item = {"field": name, "before": a, "after": b}
        (changes if _read_value(a) and _read_value(b) else unverified).append(item)
    return {
        "identity_changes": changes,
        "identity_unverified": unverified,
        "drift": _drift(before, after),
    }


def _read_value(value: Any) -> bool:
    return value is not None and not (
        isinstance(value, Mapping) and "unavailable" in value
    )


def description_mismatches(
    description: Mapping[str, Any], records: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Where the before description disagrees with what the server told the probe.

    The model it serves, its vLLM version and the GPU driver: a description
    of another server, or a stale one, disagrees in at least one of them.
    A twin server, the same model on another GPU, differs in the port it
    listens on, compared when the probe reached it on this host's loopback
    in the describer's own network namespace, where no port is mapped.
    """
    probe = next(
        (
            r
            for r in records
            if r.get("event_type") == "infer.server_probe" and r.get("phase") == BEFORE
        ),
        None,
    )
    if probe is None:
        return []
    port = _port_mismatch(description, probe)
    found = _field_mismatches(description, probe.get("answers") or {})
    return found + ([port] if port is not None else [])


def _field_mismatches(
    description: Mapping[str, Any], answers: Mapping[str, Any]
) -> list[dict[str, Any]]:
    checks = [
        ("model", _described_model(description), _served_models(answers)),
        ("engine.version", _described_vllm(description), _probed_vllm(answers)),
        ("gpu.driver_version", _described_driver(description), _probed_driver(answers)),
    ]
    return [
        {"field": name, "description": described, "server": sorted(served)}
        for name, described, served in checks
        if described is not None and served and described not in served
    ]


def _port_mismatch(
    description: Mapping[str, Any], probe: Mapping[str, Any]
) -> dict[str, Any] | None:
    server = description.get("server") or {}
    ports = server.get("listen_ports")
    if not ports or server.get("shares_network_namespace") is not True:
        return None
    origin = urllib.parse.urlsplit(str(probe.get("origin") or ""))
    if not _loopback(origin.hostname):
        return None
    port = origin.port or (443 if origin.scheme == "https" else 80)
    if port in ports:
        return None
    return {"field": "server.port", "description": list(ports), "server": [port]}


def _loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def _described_model(description: Mapping[str, Any]) -> Any:
    return ((description.get("server") or {}).get("launch") or {}).get("model")


def _described_vllm(description: Mapping[str, Any]) -> Any:
    return ((description.get("runtime") or {}).get("packages") or {}).get("vllm")


def _described_driver(description: Mapping[str, Any]) -> Any:
    return (description.get("gpus") or {}).get("driver_version")


def _body(answers: Mapping[str, Any], route: str) -> Mapping[str, Any]:
    body = (answers.get(route) or {}).get("body")
    return body if isinstance(body, Mapping) else {}


def _served_models(answers: Mapping[str, Any]) -> set[str]:
    served: set[str] = set()
    for item in _body(answers, "/v1/models").get("data") or []:
        if isinstance(item, Mapping):
            served.update(str(item[key]) for key in ("id", "root") if item.get(key))
    info = _body(answers, "/server_info?config_format=json")
    model = ((info.get("vllm_config") or {}).get("model_config") or {}).get("model")
    if model:
        served.add(str(model))
    return served


def _probed_vllm(answers: Mapping[str, Any]) -> set[str]:
    version = _body(answers, "/version").get("version")
    return {str(version)} if version else set()


def _probed_driver(answers: Mapping[str, Any]) -> set[str]:
    info = _body(answers, "/server_info?config_format=json")
    driver = (info.get("system_env") or {}).get("nvidia_driver_version")
    return {str(driver)} if driver else set()


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
    if before is not None:
        summary.update(_checks(before["description"], after, records))
    return summary


def _checks(
    before: Mapping[str, Any],
    after: Mapping[str, Any] | None,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The before description against the probe, then against the after one."""
    mismatches = description_mismatches(before, records)
    checks: dict[str, Any] = {
        "description_mismatches": mismatches,
        "protocol_failure": DESCRIPTION_MISMATCH if mismatches else None,
    }
    if after is not None:
        checks.update(compare_descriptions(before, after["description"]))
        if checks["identity_changes"]:
            checks["protocol_failure"] = IDENTITY_CHANGED
    return checks


def manifest_lines(summary: Any) -> list[str]:
    """The text report's lines for a ``manifest`` block."""
    if not isinstance(summary, dict):
        return []
    roles = [f"{role} {len(summary.get(role) or [])}" for role in (BEFORE, AFTER)]
    roles.append(f"declared {'yes' if summary.get(DECLARED) else 'no'}")
    lines = ["Server manifest: " + ", ".join(roles)]
    for key, heading in _FIELD_LISTS:
        names = ", ".join(str(item.get("field")) for item in summary.get(key) or [])
        if names:
            lines.append(f"  {heading}: {names}")
    return lines + _drift_lines(summary.get("drift") or {})


_FIELD_LISTS = (
    ("identity_changes", "identity changed during the run (protocol failure)"),
    (
        "description_mismatches",
        "the before description does not match the probed server " "(protocol failure)",
    ),
    ("identity_unverified", "read on one side only, unverified"),
)


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
    "DESCRIPTION_MISMATCH",
    "IDENTITY_CHANGED",
    "MANIFEST_EVENT",
    "MODEL_IDENTITY_EVENT",
    "ROLES",
    "after_refusals",
    "attach_manifest",
    "before_refusals",
    "compare_descriptions",
    "declared_record",
    "description_mismatches",
    "description_record",
    "identity_fields",
    "last_measured_end_ns",
    "load_declarations",
    "manifest_lines",
    "manifest_summary",
    "manifests",
    "model_identity_record",
    "run_span_ns",
]
