"""Shared, research-only data contracts for native probe experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class ExperimentMode(str, Enum):
    """Profiler modes compared by the issue #118 experiment."""

    OFF = "off"
    PUBLIC_PYTORCH = "public-pytorch"
    PUBLIC_ENGINE = "public-engine"
    PROTON = "proton"
    TRUSTED = "trusted"
    EBPF_SEMANTIC = "ebpf-semantic"
    DIRECT_CUPTI = "direct-cupti"
    HYBRID_CUPTI_EBPF = "hybrid-cupti-ebpf"
    PROGRAMMABLE = "programmable"
    AMD_ROCPROFILER = "amd-rocprofiler"
    DETAILED_COUNTER = "detailed-counter"


class WorkloadId(str, Enum):
    """Canonical workload identifiers accepted by every experiment layer."""

    W1_EAGER = "w1-eager"
    W2_OVERLAP = "w2-overlap"
    W2_SERIALIZED = "w2-serialized"
    W3_GRAPH = "w3-graph"
    W4_STRESS = "w4-stress"
    VLLM = "vllm"


class ProcessRole(str, Enum):
    """Resource-accounting roles which must not be silently combined."""

    TARGET = "target"
    PROFILER_WRAPPER = "profiler_wrapper"
    HELPER_AGENT = "helper_agent"
    POSTPROCESSOR = "postprocessor"
    SYSTEM = "system"


class ResultStatus(str, Enum):
    """Outcome of a capability check or measured trial."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNTESTED = "untested"
    TIMEOUT = "timeout"


class EvidenceStatus(str, Enum):
    """Confidence vocabulary for capability matrix cells."""

    CONFIRMED_BY_PRIMARY_SOURCE = "CONFIRMED_BY_PRIMARY_SOURCE"
    PAPER_RESULT_NOT_REPRODUCED = "PAPER_RESULT_NOT_REPRODUCED"
    STORMLOG_VALIDATED = "STORMLOG_VALIDATED"
    INFERENCE = "INFERENCE"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"
    VERSION_DEPENDENT = "VERSION_DEPENDENT"


@dataclass(frozen=True)
class CommandSpec:
    """An argv-only command used for one experiment mode."""

    argv: tuple[str, ...]
    environment: Mapping[str, str]
    timeout_seconds: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CommandSpec":
        """Parse and validate a command specification."""
        raw_argv = value.get("argv")
        raw_environment = value.get("environment", {})
        raw_timeout = value.get("timeout_seconds", 300.0)
        if not isinstance(raw_argv, Sequence) or isinstance(raw_argv, (str, bytes)):
            raise ValueError("command argv must be an array")
        argv = tuple(_nonempty_string(item, "command argv item") for item in raw_argv)
        if not argv:
            raise ValueError("command argv must not be empty")
        if not isinstance(raw_environment, Mapping):
            raise ValueError("command environment must be an object")
        environment = {
            _nonempty_string(key, "environment key"): _string(
                value, "environment value"
            )
            for key, value in raw_environment.items()
        }
        if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
            raise ValueError("timeout_seconds must be a number")
        timeout_seconds = float(raw_timeout)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return cls(argv=argv, environment=environment, timeout_seconds=timeout_seconds)


@dataclass(frozen=True)
class ArtifactExpectation:
    """One output a mode is expected to produce during a trial."""

    artifact_id: str
    kind: str
    relative_path: str
    producer: str
    format: str
    required: bool = True
    sensitive: bool = True
    loss_metadata_expected: bool = False


@dataclass(frozen=True)
class ProcessRoleSpec:
    """How a process role is discovered without guessing after execution."""

    role: ProcessRole
    discovery: str
    argv_contains: str | None = None


@dataclass(frozen=True)
class TrialSpec:
    """One immutable workload and profiler-mode trial definition."""

    trial_id: str
    configuration_id: str
    workload_id: WorkloadId
    mode: ExperimentMode
    repetition: int
    command: CommandSpec
    expected_artifacts: tuple[ArtifactExpectation, ...] = ()
    process_roles: tuple[ProcessRoleSpec, ...] = ()
    measurement_range_id: str = "stormlog-native-probe-measured"


def _nonempty_string(value: object, field: str) -> str:
    parsed = _string(value, field)
    if not parsed.strip():
        raise ValueError(f"{field} must not be empty")
    return parsed


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value
