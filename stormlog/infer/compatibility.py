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
- ``unverified``: a required field is unknown on either or both sides; a
  redacted, inferred or declared value is unknown, and two redacted values
  are never equal;
- ``compatible``: otherwise.

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
from .manifest import BEFORE, DECLARED, manifests
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


@dataclass(frozen=True)
class RunField:
    """One field of a run: its value, where it came from, and how it is known."""

    value: Any
    source: str
    provenance: str

    @property
    def known(self) -> bool:
        return (
            self.value is not None
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
        fields.update(_description_fields(by_role[BEFORE][-1]["description"]))
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
        _judge(name, a.get(name), b.get(name), mode, allowances, groups, unknown)
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


def _judge(
    name: str,
    first: RunField | None,
    second: RunField | None,
    mode: str,
    allowances: list[str],
    groups: dict[str, list[Difference]],
    unknown: list[str],
) -> None:
    kind = classify(name)
    if kind == LABEL:
        return
    difference = _compared(name, kind, first, second)
    if difference is None:
        return
    if difference.reason == "unknown":
        unknown.append(name)
    elif difference.reason == "differs_unverified":
        groups["unverified"].append(difference)
    else:
        groups[_group(name, kind, mode, allowances)].append(difference)


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
        return any(key.startswith(VLLM_CONFIG + "/") for key in fields)
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
    """Model fields; a digest the run did not bind to its launch is inferred."""
    evidence = model.get("identity_evidence")
    provenance = OBSERVED if evidence in VERIFIED_EVIDENCE else INFERRED
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


def _probe_fields(records: Sequence[Mapping[str, Any]]) -> dict[str, RunField]:
    probe = next(
        (
            r
            for r in records
            if r.get("event_type") == "infer.server_probe"
            and r.get("phase") == "before"
        ),
        None,
    )
    answers = _section(probe or {}, "answers")
    values: dict[str, Any] = {}
    version = _section(answers, VERSION).get("body")
    if isinstance(version, Mapping):
        values["engine.version"] = (version.get("version"), "/version")
    info = _section(answers, SERVER_INFO).get("body")
    if isinstance(info, Mapping):
        values.update(_server_info_values(info))
    return {
        name: RunField(value, source, REPORTED)
        for name, (value, source) in values.items()
    }


def _server_info_values(info: Mapping[str, Any]) -> dict[str, tuple[Any, str]]:
    source = "/server_info"
    values = {
        VLLM_CONFIG + pointer: (value, source)
        for pointer, value in _leaves(info.get("vllm_config"), "")
    }
    for name, value in _section(info, "vllm_env").items():
        values[f"{VLLM_ENV}/{name}"] = (value, source)
    packages = _section(_section(info, "system_env"), "packages")
    for package, version in packages.items():
        values[f"runtime.{package}"] = (version, source)
    return values


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
    "UNVERIFIED",
    "VERIFIED_EVIDENCE",
    "Compatibility",
    "Difference",
    "RunField",
    "classify",
    "compatible",
    "run_fields",
]
