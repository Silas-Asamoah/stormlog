"""Whether two runs measured the same thing, field by field.

A run's fields come from its artifact: the ``before`` server description,
what the server reported to the probe, the operator's declarations and the
workload record. Each field keeps its value, its source and its
provenance: ``observed`` by Stormlog, ``reported`` by the server,
``inferred`` from evidence that cannot show it (a model digest taken after
launch), or ``declared`` by the operator.

``compatible(a, b)`` compares two runs:

- ``incompatible``: an identity field (or an unclassified configuration
  leaf) differs and is not allowed;
- ``unverified``: a required field is unknown on either or both sides, or
  an identity field (or an unclassified leaf) is unknown on one side; a
  redacted, inferred or declared value is unknown, and two redacted values
  are never equal;
- ``compatible``: otherwise.

A ``null`` in vLLM's configuration is a setting (no quantization, no
speculative decoding), not missing evidence. Where both runs read a source
(``/server_info``'s configuration or environment, the server's process
environment), a field only one of them has is a difference; where one did
not read it, the field is unknown.

Launch fields are covariates and never block. Observation fields may
differ in ``overhead`` and ``incremental`` comparisons only. Labels are
ignored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config_classes import (
    CONFIG_CLASSES_VERSION,
    IDENTITY,
    LABEL,
    LAUNCH,
    OBSERVATION,
    UNCLASSIFIED,
    config_class,
    field_class,
    vllm_env_class,
)
from .manifest import BEFORE, DECLARED, MODEL_IDENTITY_EVENT, manifests
from .server_privacy import is_redacted
from .server_probe import SERVER_INFO, VERSION
from .workload import workload_digests

COMPATIBLE = "compatible"
UNVERIFIED = "unverified"
INCOMPATIBLE = "incompatible"
CONFIG = "config"
OVERHEAD = "overhead"
INCREMENTAL = "incremental"
MODES = (CONFIG, OVERHEAD, INCREMENTAL)

OBSERVED = "observed"
REPORTED = "reported"
INFERRED = "inferred"
DECLARED_PROVENANCE = "declared"

VLLM_CONFIG = "vllm_config"
VLLM_ENV = "vllm_env"

# What must be known on both sides for two runs to be shown comparable.
REQUIRED_V1 = (
    "model.weights_digest",
    "engine.version",
    "gpu.name",
    "gpu.driver_version",
    "workload.spec_digest",
    VLLM_CONFIG,
)
# Canonical names for the configuration leaves people name most.
ALIASES: Mapping[str, str] = {
    "engine.dtype": "/model_config/dtype",
    "engine.max_model_len": "/model_config/max_model_len",
    "engine.quantization": "/model_config/quantization",
    "engine.max_num_seqs": "/scheduler_config/max_num_seqs",
    "engine.max_num_batched_tokens": "/scheduler_config/max_num_batched_tokens",
    "engine.chunked_prefill": "/scheduler_config/enable_chunked_prefill",
    "engine.async_scheduling": "/scheduler_config/async_scheduling",
    "engine.stream_interval": "/scheduler_config/stream_interval",
    "engine.prefix_caching": "/cache_config/enable_prefix_caching",
    "engine.kv_cache_dtype": "/cache_config/cache_dtype",
    "engine.block_size": "/cache_config/block_size",
    "engine.gpu_memory_utilization": "/cache_config/gpu_memory_utilization",
    "engine.tensor_parallel_size": "/parallel_config/tensor_parallel_size",
    "engine.pipeline_parallel_size": "/parallel_config/pipeline_parallel_size",
    "engine.data_parallel_size": "/parallel_config/data_parallel_size",
    "engine.compilation_mode": "/compilation_config/mode",
    "engine.cudagraph_mode": "/compilation_config/cudagraph_mode",
    "engine.attention_backend": "/attention_config/backend",
    "engine.speculative": "/speculative_config",
}
_UNKNOWN_PROVENANCE = frozenset({INFERRED, DECLARED_PROVENANCE})
# Marker fields naming each source a run read, so that a field it lacks is
# known to be absent rather than unknown.
SCOPE_VLLM_CONFIG = "scope.vllm_config"
SCOPE_VLLM_ENV = "scope.vllm_env"
SCOPE_ENVIRON = "scope.environ"
_SCOPES = (
    (VLLM_CONFIG + "/", SCOPE_VLLM_CONFIG),
    (VLLM_ENV + "/", SCOPE_VLLM_ENV),
    ("environ.", SCOPE_ENVIRON),
)
# What cannot be told apart from its counterpart is not verified.
_MUST_BE_KNOWN = frozenset({IDENTITY, UNCLASSIFIED})


@dataclass(frozen=True)
class RunField:
    """One field of a run: its value, where it came from, and how it is known.

    ``null_is_value`` marks a field whose ``None`` is a setting, as in vLLM's
    configuration, rather than evidence that could not be read.
    """

    value: Any
    source: str
    provenance: str
    null_is_value: bool = False

    @property
    def known(self) -> bool:
        return (
            (self.value is not None or self.null_is_value)
            and not is_redacted(self.value)
            and not _unavailable(self.value)
            and self.provenance not in _UNKNOWN_PROVENANCE
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Difference:
    name: str
    field_class: str
    a: Any
    b: Any
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "field": self.name,
            "class": self.field_class,
            "a": self.a,
            "b": self.b,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Compatibility:
    status: str
    mode: str
    blocking: tuple[Difference, ...] = ()
    unverified: tuple[Difference, ...] = ()
    allowed: tuple[Difference, ...] = ()
    covariates: tuple[Difference, ...] = ()
    observation: tuple[Difference, ...] = ()
    unknown: tuple[str, ...] = field(default=())

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "classes": CONFIG_CLASSES_VERSION,
            "blocking": [item.to_record() for item in self.blocking],
            "unverified": [item.to_record() for item in self.unverified],
            "allowed": [item.to_record() for item in self.allowed],
            "covariates": [item.to_record() for item in self.covariates],
            "observation": [item.to_record() for item in self.observation],
            "unknown": list(self.unknown),
        }


def run_fields(records: Sequence[Mapping[str, Any]]) -> dict[str, RunField]:
    """Every comparable field of one run's artifact."""
    fields: dict[str, RunField] = {}
    by_role = manifests(records)
    if by_role[BEFORE]:
        description = by_role[BEFORE][-1]["description"]
        fields.update(_description_fields(description))
        fields.update(_launch_bound_fields(records, description))
    # What Stormlog observed outranks what the server reported.
    for name, item in _probe_fields(records).items():
        fields.setdefault(name, item)
    fields.update(_workload_fields(records))
    fields.update(_observer_fields(records))
    if by_role[DECLARED]:
        for name, value in (by_role[DECLARED][-1].get("fields") or {}).items():
            fields.setdefault(
                str(name), RunField(value, "declared", DECLARED_PROVENANCE)
            )
    return fields


def compatible(
    a: Mapping[str, RunField],
    b: Mapping[str, RunField],
    *,
    allowed: Iterable[str] = (),
    mode: str = CONFIG,
    required: Sequence[str] = REQUIRED_V1,
) -> Compatibility:
    """Whether runs ``a`` and ``b`` measured the same thing."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    allowances = [_allowance(name) for name in allowed]
    groups: dict[str, list[Difference]] = {
        "blocking": [],
        "unverified": [],
        "allowed": [],
        "covariates": [],
        "observation": [],
    }
    unknown: list[str] = []
    for name in sorted(set(a) | set(b)):
        difference = _one_sided(name, a, b) or _compared(
            name, classify(name), a.get(name), b.get(name)
        )
        _sort(difference, mode, allowances, groups, unknown)
    groups["unverified"].extend(
        _unknown_required(a, b, required, {d.name for d in groups["unverified"]})
    )
    status = (
        INCOMPATIBLE
        if groups["blocking"]
        else UNVERIFIED if groups["unverified"] else COMPATIBLE
    )
    return Compatibility(
        status=status,
        mode=mode,
        **{key: tuple(items) for key, items in groups.items()},
        unknown=tuple(unknown),
    )


def _unknown_required(
    a: Mapping[str, RunField],
    b: Mapping[str, RunField],
    required: Sequence[str],
    shown: set[str],
) -> list[Difference]:
    """Required fields not known on both sides, other than those already shown."""
    return [
        Difference(name, IDENTITY, _value(a, name), _value(b, name), "unknown")
        for name in required
        if name not in shown
        and not (_required_known(name, a) and _required_known(name, b))
    ]


def classify(name: str) -> str:
    """The class of any run field name."""
    if name.startswith(VLLM_CONFIG + "/"):
        return config_class(name[len(VLLM_CONFIG) :])
    if name.startswith(VLLM_ENV + "/"):
        return vllm_env_class(name[len(VLLM_ENV) + 1 :])
    return field_class(name)


def _sort(
    difference: Difference | None,
    mode: str,
    allowances: list[str],
    groups: dict[str, list[Difference]],
    unknown: list[str],
) -> None:
    """File a difference by its reason and class; labels are ignored."""
    if difference is None or difference.field_class == LABEL:
        return
    if difference.reason == "unknown":
        if difference.field_class in _MUST_BE_KNOWN:
            groups["unverified"].append(difference)
        else:
            unknown.append(difference.name)
    elif difference.reason == "differs_unverified":
        groups["unverified"].append(difference)
    else:
        name, kind = difference.name, difference.field_class
        groups[_group(name, kind, mode, allowances)].append(difference)


def _one_sided(
    name: str, a: Mapping[str, RunField], b: Mapping[str, RunField]
) -> Difference | None:
    """A field one run has and the other, which read its source, lacks.

    When the value that is there is no evidence (redacted, inferred or
    declared), the two differ but the difference is not verified.
    """
    first, second = a.get(name), b.get(name)
    if first is None and second is not None and _read(a, name):
        reason = "only_in_b" if second.known else "differs_unverified"
        return Difference(name, classify(name), None, second.value, reason)
    if second is None and first is not None and _read(b, name):
        reason = "only_in_a" if first.known else "differs_unverified"
        return Difference(name, classify(name), first.value, None, reason)
    return None


def _read(fields: Mapping[str, RunField], name: str) -> bool:
    """Whether the run read the source a field comes from."""
    scope = next(
        (marker for prefix, marker in _SCOPES if name.startswith(prefix)), None
    )
    return scope is not None and _known(fields.get(scope))


def _compared(
    name: str, kind: str, first: RunField | None, second: RunField | None
) -> Difference | None:
    """None when equal or absent on both sides; else how the two differ."""
    a_value, b_value = _value_of(first), _value_of(second)
    if _known(first) and _known(second):
        if a_value == b_value:
            return None
        reason = "unclassified" if kind == UNCLASSIFIED else "differs"
        return Difference(name, kind, a_value, b_value, reason)
    if a_value is None and b_value is None:
        return None
    # Neither value can be trusted; when they disagree, say so.
    disagree = a_value is not None and b_value is not None and a_value != b_value
    reason = "differs_unverified" if disagree else "unknown"
    return Difference(name, kind, a_value, b_value, reason)


def _value_of(item: RunField | None) -> Any:
    return None if item is None else item.value


def _known(item: RunField | None) -> bool:
    return item is not None and item.known


def _group(name: str, kind: str, mode: str, allowances: list[str]) -> str:
    """Where a known difference goes: allowed, a covariate, or blocking."""
    if any(_covers(allowance, name) for allowance in allowances):
        return "allowed"
    if kind == LAUNCH:
        return "covariates"
    if kind == OBSERVATION and mode != CONFIG:
        return "observation"
    return "blocking"


def _required_known(name: str, fields: Mapping[str, RunField]) -> bool:
    if name == VLLM_CONFIG:
        # Only the server's own answer counts; a declared leaf does not.
        return _known(fields.get(SCOPE_VLLM_CONFIG))
    item = fields.get(name)
    return item is not None and item.known


def _value(fields: Mapping[str, RunField], name: str) -> Any:
    item = fields.get(name)
    return None if item is None else item.value


def _allowance(name: str) -> str:
    """A canonical name, an alias, or a ``vllm_config`` JSON pointer."""
    if name in ALIASES:
        return VLLM_CONFIG + ALIASES[name]
    if name.startswith("/"):
        return VLLM_CONFIG + name
    return name


def _covers(allowance: str, name: str) -> bool:
    return (
        name == allowance
        or name.startswith(allowance + "/")
        or name.startswith(allowance + ".")
    )


def _unavailable(value: Any) -> bool:
    return isinstance(value, Mapping) and "unavailable" in value


# ---------------------------------------------------------------- sources


def _description_fields(description: Mapping[str, Any]) -> dict[str, RunField]:
    values = {
        **_gpu_values(_section(description, "gpus")),
        **_runtime_values(_section(description, "runtime")),
        **_log_values(_section(description, "log")),
        **_host_values(description),
    }
    fields = {
        name: RunField(value, "describe-server", OBSERVED)
        for name, value in values.items()
    }
    if isinstance(_section(description, "server").get("environ"), Mapping):
        fields[SCOPE_ENVIRON] = RunField(True, "describe-server", OBSERVED)
    fields.update(_model_fields(_section(description, "model")))
    return fields


def _gpu_values(gpus: Mapping[str, Any]) -> dict[str, Any]:
    devices = [d for d in gpus.get("devices") or [] if d.get("server_pids")]
    values: dict[str, Any] = {
        "gpu.driver_version": gpus.get("driver_version"),
        "gpu.cuda_driver_version": gpus.get("cuda_driver_version"),
        "gpu.uuids": sorted(str(d.get("uuid")) for d in devices) or None,
        "gpu.count": len(devices) if gpus else None,
    }
    values.update(_gpu_settings([_section(device, "settings") for device in devices]))
    return values


def _gpu_settings(settings: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Each setting across the server's GPUs: one value, or each distinct one."""
    names = sorted({key for setting in settings for key in setting})
    return {
        f"gpu.{name}": _one_or_all([setting.get(name) for setting in settings])
        for name in names
    }


def _runtime_values(runtime: Mapping[str, Any]) -> dict[str, Any]:
    values = {"runtime.python": runtime.get("python")}
    for package, version in _section(runtime, "packages").items():
        values[f"runtime.{package}"] = version
    return values


def _log_values(log: Mapping[str, Any]) -> dict[str, Any]:
    names = ("attention_backend", "kv_cache_size_tokens", "cudagraph_captures")
    return {f"effective.{name}": log[name] for name in names if name in log}


def _host_values(description: Mapping[str, Any]) -> dict[str, Any]:
    server = _section(description, "server")
    values = {
        "host.start_method": _section(server, "start_method").get("configured"),
        "host.nproc": _section(description, "host").get("nproc"),
    }
    for name, value in _section(server, "environ").items():
        if not name.startswith("VLLM_"):
            values[f"environ.{name}"] = value
    return values


def _section(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    return value if isinstance(value, Mapping) else {}


def _model_fields(model: Mapping[str, Any]) -> dict[str, RunField]:
    """Model fields as a description gives them: read after the server started.

    Whatever evidence a description names, its digests are inferred: only
    the runner's launch-bound record (``_launch_bound_fields``) shows what
    the server loaded.
    """
    evidence = model.get("identity_evidence")
    provenance = INFERRED
    source = f"describe-server ({evidence})"
    return {
        "model.configured": RunField(model.get("configured"), source, OBSERVED),
        "model.configured_revision": RunField(
            model.get("configured_revision"), source, OBSERVED
        ),
        "model.resolved_snapshot": RunField(
            model.get("resolved_snapshot"), source, provenance
        ),
        "model.weights_digest": RunField(
            model.get("weights_digest"), source, provenance
        ),
        "model.identity_evidence": RunField(evidence, source, OBSERVED),
        "server.chat_template_digest": RunField(
            model.get("chat_template_digest"), source, provenance
        ),
        "server.generation_config": RunField(
            _digest(model.get("generation_config")), source, provenance
        ),
    }


# Only a launch the experiment runner controlled binds weights to a server.
VERIFIED_EVIDENCE = frozenset({"pinned_commit_verified", "staged_snapshot_verified"})


def _launch_bound_fields(
    records: Sequence[Mapping[str, Any]], description: Mapping[str, Any]
) -> dict[str, RunField]:
    """The model fields the runner verified before launching this server.

    The runner's ``infer.model_identity`` record counts only for the server
    the before description shows (same boot, PID and start time), and only
    when the description's own digest, where it has one, agrees. Then the
    weights, the snapshot, and the chat template and generation defaults
    read from that snapshot are observed.
    """
    record = next(
        (
            r
            for r in reversed(records)
            if r.get("event_type") == MODEL_IDENTITY_EVENT and _binds(r, description)
        ),
        None,
    )
    model = _section(record or {}, "model")
    evidence = model.get("identity_evidence")
    described = _section(description, "model")
    if evidence not in VERIFIED_EVIDENCE or not _agrees(model, described):
        return {}
    source = f"experiment runner ({evidence})"
    values = {
        "model.weights_digest": model.get("weights_digest"),
        "model.resolved_snapshot": model.get("resolved_snapshot"),
        "server.chat_template_digest": described.get("chat_template_digest"),
        "server.generation_config": _digest(described.get("generation_config")),
        "model.identity_evidence": evidence,
    }
    return {name: RunField(value, source, OBSERVED) for name, value in values.items()}


def _binds(record: Mapping[str, Any], description: Mapping[str, Any]) -> bool:
    server = _section(description, "server")
    bound = _section(record, "server")
    lifetime = (server.get("pid"), server.get("start_ticks"))
    return (
        None not in lifetime
        and (bound.get("pid"), bound.get("start_ticks")) == lifetime
        and record.get("boot_id") == _section(description, "host").get("boot_id")
    )


def _agrees(model: Mapping[str, Any], described: Mapping[str, Any]) -> bool:
    """The description's digests, where it has them, match the verified ones."""
    return all(
        described.get(name) in (None, model.get(name))
        for name in ("weights_digest", "resolved_snapshot")
    )


def _probe_fields(records: Sequence[Mapping[str, Any]]) -> dict[str, RunField]:
    answers = _section(_before_probe(records) or {}, "answers")
    values: dict[str, Any] = {}
    version = _section(answers, VERSION).get("body")
    if isinstance(version, Mapping):
        values["engine.version"] = (version.get("version"), "/version")
    fields = {
        name: RunField(value, source, REPORTED)
        for name, (value, source) in values.items()
    }
    info = _section(answers, SERVER_INFO).get("body")
    if isinstance(info, Mapping):
        fields.update(_server_info_fields(info))
    return fields


def _before_probe(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The before probe that answered ``/server_info``, else the first.

    A runner probes ``/server_info`` once before measuring, and the
    workload's own probe asks only the basic routes, so that no collector
    runs beside it; the configuration comes from the runner's.
    """
    before = [
        r
        for r in records
        if r.get("event_type") == "infer.server_probe" and r.get("phase") == "before"
    ]
    answered = [
        r for r in before if _section(_section(r, "answers"), SERVER_INFO).get("body")
    ]
    found = answered or before
    return found[0] if found else None


def _server_info_fields(info: Mapping[str, Any]) -> dict[str, RunField]:
    """Configuration and environment, where null is a setting, and versions."""
    source = "/server_info"
    fields: dict[str, RunField] = {}
    if isinstance(info.get("vllm_config"), Mapping):
        fields[SCOPE_VLLM_CONFIG] = RunField(True, source, REPORTED)
        for pointer, value in _leaves(info.get("vllm_config"), ""):
            fields[VLLM_CONFIG + pointer] = _setting(value, source)
    if isinstance(info.get("vllm_env"), Mapping):
        fields[SCOPE_VLLM_ENV] = RunField(True, source, REPORTED)
        for name, value in _section(info, "vllm_env").items():
            fields[f"{VLLM_ENV}/{name}"] = _setting(value, source)
    packages = _section(_section(info, "system_env"), "packages")
    for package, version in packages.items():
        fields[f"runtime.{package}"] = RunField(version, source, REPORTED)
    return fields


def _setting(value: Any, source: str) -> RunField:
    return RunField(value, source, REPORTED, null_is_value=True)


def _workload_fields(records: Sequence[Mapping[str, Any]]) -> dict[str, RunField]:
    workload = next(
        (r for r in records if r.get("event_type") == "infer.workload"), None
    )
    if workload is None:
        return {}
    digests = workload_digests(dict(workload))
    measurement = workload.get("measurement") or {}
    source = "infer.workload"
    return {
        "workload.spec_digest": RunField(digests["spec_digest"], source, OBSERVED),
        "workload.realization_digest": RunField(
            digests["realization_digest"], source, OBSERVED
        ),
        "workload.timeout_seconds": RunField(
            measurement.get("timeout_seconds"), source, OBSERVED
        ),
        "workload.drain_timeout_seconds": RunField(
            measurement.get("drain_timeout_seconds"), source, OBSERVED
        ),
    }


def _observer_fields(records: Sequence[Mapping[str, Any]]) -> dict[str, RunField]:
    session = next((r for r in records if r.get("event_type") == "infer.session"), None)
    config = (session or {}).get("config") or {}
    source = "infer.session"
    return {
        f"observer.{name}": RunField(config.get(key), source, OBSERVED)
        for name, key in (
            ("system_sampler", "system_sampler"),
            ("vllm_metrics", "vllm_metrics"),
            ("vllm_spans", "vllm_spans"),
            ("trace", "trace"),
            ("execution", "vllm_execution_dir"),
        )
        if key in config
    }


def _leaves(value: Any, pointer: str) -> Iterable[tuple[str, Any]]:
    """Every leaf of a JSON document, by RFC 6901 pointer.

    A redacted marker is a leaf: its content is not configuration.
    """
    if isinstance(value, Mapping) and value and not is_redacted(value):
        for key, item in value.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            yield from _leaves(item, f"{pointer}/{token}")
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _leaves(item, f"{pointer}/{index}")
    else:
        yield pointer, value


def _one_or_all(values: list[Any]) -> Any:
    distinct = [value for i, value in enumerate(values) if value not in values[:i]]
    return distinct[0] if len(distinct) == 1 else distinct or None


def _digest(value: Any) -> str | None:
    if value is None:
        return None
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "ALIASES",
    "COMPATIBLE",
    "CONFIG",
    "INCOMPATIBLE",
    "INCREMENTAL",
    "MODES",
    "OVERHEAD",
    "REQUIRED_V1",
    "SCOPE_ENVIRON",
    "SCOPE_VLLM_CONFIG",
    "SCOPE_VLLM_ENV",
    "UNVERIFIED",
    "VERIFIED_EVIDENCE",
    "Compatibility",
    "Difference",
    "RunField",
    "classify",
    "compatible",
    "run_fields",
]
