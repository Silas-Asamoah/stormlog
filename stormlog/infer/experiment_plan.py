"""An experiment plan: arms, blocks, their order, and every command a run runs.

A plan is ``stormlog.infer.experiment_plan`` v1, in JSON. It names one server
command, the arms (each with its server arguments, its treatments and its
workload steps), how many blocks to run, and in which order. The runner
(``experiment``) launches a fresh server for every run.

Commands are templates. ``{name}`` placeholders are filled per run:
``{run_dir}``, ``{run_id}``, ``{label}``, ``{experiment_id}``, ``{arm}``,
``{block}``, ``{position}``, ``{attempt}``, ``{block_seed}``, ``{server_pid}``,
``{base_url}``, ``{python}`` and ``{model}``. A placeholder the runner does
not know is refused when the plan is loaded, not when a run reaches it.

Every arm of a block shares one ``{block_seed}``, derived from the plan's
seed, so matched arms send identical prompts at identical times.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import InferInputError

PLAN_FORMAT = "stormlog.infer.experiment_plan"
PLAN_VERSION = 1
PLACEHOLDERS = frozenset(
    {
        "run_dir",
        "run_id",
        "label",
        "experiment_id",
        "arm",
        "block",
        "position",
        "attempt",
        "block_seed",
        "server_pid",
        "base_url",
        "python",
        "model",
    }
)
ORDER_KINDS = ("random", "williams", "explicit")
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SAME_AS = "same_as:"


@dataclass(frozen=True)
class Step:
    """One command a run (or a block prelude) runs, and what it must leave."""

    name: str
    command: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    cpu_affinity: str | None = None
    timeout_s: float = 900.0
    expect_exit: tuple[int, ...] = (0,)
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Treatment:
    """A process that runs beside the workload, such as a watcher."""

    name: str
    command: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    cpu_affinity: str | None = None
    ready_file: str | None = None
    ready_timeout_s: float = 60.0
    stop_signal: int = int(signal.SIGTERM)
    stop_timeout_s: float = 30.0
    expect_exit: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class Prelude:
    """A command run once before a block's first run, with an arm's server or none."""

    step: Step
    server_arm: str | None = None


@dataclass(frozen=True)
class ServerSpec:
    command: tuple[str, ...]
    base_url: str
    env: Mapping[str, str] = field(default_factory=dict)
    cpu_affinity: str | None = None
    start_timeout_s: float = 600.0
    stop_timeout_s: float = 30.0
    model: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Arm:
    name: str
    server_args: tuple[str, ...] = ()
    server_env: Mapping[str, str] = field(default_factory=dict)
    workload: tuple[Step, ...] = ()
    treatments: tuple[Treatment, ...] = ()


@dataclass(frozen=True)
class ExperimentPlan:
    """A validated plan, and the SHA-256 of the document it came from."""

    experiment_id: str
    seed: int
    blocks: int
    order: Mapping[str, Any]
    server: ServerSpec
    arms: Mapping[str, Arm]
    preludes: tuple[Prelude, ...] = ()
    describe: Mapping[str, bool] = field(default_factory=dict)
    secret_env: tuple[str, ...] = ()
    affinity_disjoint: bool = False
    prereg: Mapping[str, Any] | None = None
    digest: str = ""

    @property
    def prereg_digest(self) -> str | None:
        if self.prereg is None:
            return None
        return _digest(self.prereg)


@dataclass(frozen=True)
class RunOrder:
    """The arms of each block in run order, and whether positions balance."""

    blocks: list[list[str]]
    kind: str
    balanced: bool
    position_counts: dict[str, list[int]]

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "balanced": self.balanced,
            "blocks": self.blocks,
            "position_counts": self.position_counts,
        }


# ----------------------------------------------------------------- loading


def load_plan(path: str | Path) -> ExperimentPlan:
    """Read and check a plan file; InferInputError for anything wrong in it."""
    source = Path(path)
    try:
        document = json.loads(source.read_text())
    except (OSError, ValueError) as exc:
        raise InferInputError(f"experiment plan {source}: {exc}") from exc
    try:
        return plan_from_document(document)
    except (KeyError, TypeError, ValueError) as exc:
        raise InferInputError(f"experiment plan {source}: {exc}") from exc


def plan_from_document(document: Any) -> ExperimentPlan:
    """A plan from its parsed JSON; ValueError names what is wrong."""
    _check_header(document)
    blocks = document["blocks"]
    arms = _arms(document.get("arms"))
    plan = ExperimentPlan(
        experiment_id=_identifier(document.get("experiment_id"), "experiment_id"),
        seed=int(document.get("seed", 0)),
        blocks=blocks,
        order=_order(document.get("order") or {"kind": "random"}, blocks, arms),
        server=_server(document.get("server")),
        arms=arms,
        preludes=tuple(
            _prelude(item, arms) for item in document.get("block_prelude") or []
        ),
        describe=_describe(document.get("describe_server")),
        secret_env=tuple(str(name) for name in document.get("secret_env") or []),
        affinity_disjoint=bool(document.get("affinity_disjoint", False)),
        prereg=document.get("prereg"),
        digest=_digest(document),
    )
    _check_placeholders(plan)
    return plan


def _check_header(document: Any) -> None:
    if not isinstance(document, dict) or document.get("format") != PLAN_FORMAT:
        raise ValueError(f"not {PLAN_FORMAT}")
    if document.get("version") != PLAN_VERSION:
        raise ValueError(f"version {document.get('version')!r} is not {PLAN_VERSION}")
    blocks = document.get("blocks")
    if not isinstance(blocks, int) or blocks < 1:
        raise ValueError("blocks must be an integer >= 1")


def _describe(raw: Any) -> dict[str, bool]:
    """When to describe the server; every description is on unless turned off."""
    given = raw if isinstance(raw, dict) else {}
    return {
        key: bool(given.get(key, True)) for key in ("before", "after", "server_log")
    }


def _identifier(value: Any, what: str) -> str:
    if not isinstance(value, str) or _ID.match(value) is None:
        raise ValueError(f"{what} {value!r} must be a short identifier")
    return value


def _server(raw: Any) -> ServerSpec:
    if not isinstance(raw, dict):
        raise ValueError("server must be an object")
    return ServerSpec(
        command=_command(raw.get("command"), "server.command"),
        base_url=str(raw["base_url"]),
        env=_env(raw.get("env")),
        cpu_affinity=raw.get("cpu_affinity"),
        start_timeout_s=float(raw.get("start_timeout_s", 600.0)),
        stop_timeout_s=float(raw.get("stop_timeout_s", 30.0)),
        model=raw.get("model"),
    )


def _arms(raw: Any) -> dict[str, Arm]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("arms must be a non-empty object")
    arms: dict[str, Arm] = {}
    pending: dict[str, str] = {}
    for name, body in raw.items():
        arms[name], source = _arm(_identifier(name, "arm"), body)
        if source is not None:
            pending[name] = source
    for name, source in pending.items():
        if source not in arms or source in pending:
            raise ValueError(
                f"arm {name}: {_SAME_AS}{source} names no arm with its own steps"
            )
        arms[name] = replace(arms[name], workload=arms[source].workload)
    return arms


def _arm(name: str, body: Any) -> tuple[Arm, str | None]:
    """An arm, and the arm whose workload it reuses (``same_as:<arm>``), if any."""
    if not isinstance(body, dict):
        raise ValueError(f"arm {name}: must be an object")
    server = body.get("server") or {}
    workload, source = _workload(name, body.get("workload") or [])
    arm = Arm(
        name=name,
        server_args=tuple(str(a) for a in server.get("args") or []),
        server_env=_env(server.get("env")),
        workload=tuple(_step(item, f"arm {name}") for item in workload),
        treatments=tuple(
            _treatment(item, name) for item in body.get("treatments") or []
        ),
    )
    return arm, source


def _workload(name: str, raw: Any) -> tuple[list[Any], str | None]:
    """An arm's steps, or none and the arm named by ``same_as:<arm>``."""
    if not isinstance(raw, str):
        return list(raw), None
    if not raw.startswith(_SAME_AS):
        raise ValueError(f"arm {name}: workload must be a list or {_SAME_AS}<arm>")
    return [], raw[len(_SAME_AS) :]


def _step(raw: Any, where: str) -> Step:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a step must be an object")
    name = _identifier(raw.get("name"), f"{where} step name")
    return Step(
        name=name,
        command=_command(raw.get("command"), f"{where} step {name}"),
        env=_env(raw.get("env")),
        cpu_affinity=raw.get("cpu_affinity"),
        timeout_s=float(raw.get("timeout_s", 900.0)),
        expect_exit=tuple(int(c) for c in raw.get("expect_exit", [0])),
        artifacts=tuple(str(a) for a in raw.get("artifacts") or []),
    )


def _treatment(raw: Any, arm: str) -> Treatment:
    if not isinstance(raw, dict):
        raise ValueError(f"arm {arm}: a treatment must be an object")
    name = _identifier(raw.get("name"), f"arm {arm} treatment name")
    stop = raw.get("stop_signal", "SIGTERM")
    try:
        stop_signal = int(getattr(signal, stop)) if isinstance(stop, str) else int(stop)
    except AttributeError as exc:
        raise ValueError(
            f"arm {arm} treatment {name}: unknown signal {stop!r}"
        ) from exc
    return Treatment(
        name=name,
        command=_command(raw.get("command"), f"arm {arm} treatment {name}"),
        env=_env(raw.get("env")),
        cpu_affinity=raw.get("cpu_affinity"),
        ready_file=raw.get("ready_file"),
        ready_timeout_s=float(raw.get("ready_timeout_s", 60.0)),
        stop_signal=stop_signal,
        stop_timeout_s=float(raw.get("stop_timeout_s", 30.0)),
        expect_exit=tuple(int(c) for c in raw.get("expect_exit", [0])),
    )


def _prelude(raw: Any, arms: Mapping[str, Arm]) -> Prelude:
    if not isinstance(raw, dict):
        raise ValueError("a block prelude must be an object")
    server_arm = raw.get("server_arm")
    if server_arm is not None and server_arm not in arms:
        raise ValueError(f"block prelude: server_arm {server_arm!r} is not an arm")
    return Prelude(step=_step(raw, "block prelude"), server_arm=server_arm)


def _command(raw: Any, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(x, str) for x in raw):
        raise ValueError(f"{where}: command must be a non-empty list of strings")
    return tuple(raw)


def _env(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("env must be an object")
    return {str(key): str(value) for key, value in raw.items()}


def _check_placeholders(plan: ExperimentPlan) -> None:
    templates: list[str] = [*plan.server.command, plan.server.base_url]
    for arm in plan.arms.values():
        templates += arm.server_args
        for step in arm.workload:
            templates += [*step.command, *step.artifacts, *step.env.values()]
        for treatment in arm.treatments:
            templates += [*treatment.command, *treatment.env.values()]
            templates += [treatment.ready_file] if treatment.ready_file else []
    for prelude in plan.preludes:
        templates += [*prelude.step.command, *prelude.step.env.values()]
    unknown = {
        name for text in templates for name in _PLACEHOLDER.findall(text)
    } - PLACEHOLDERS
    if unknown:
        raise ValueError(f"unknown placeholders: {', '.join(sorted(unknown))}")


# ------------------------------------------------------------------- order


def _order(raw: Any, blocks: int, arms: Mapping[str, Arm]) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("kind") not in ORDER_KINDS:
        raise ValueError(f"order.kind must be one of {', '.join(ORDER_KINDS)}")
    if raw["kind"] != "explicit":
        return dict(raw)
    given = raw.get("blocks")
    if not isinstance(given, list) or len(given) != blocks:
        raise ValueError("an explicit order lists every block's arms")
    for row in given:
        if sorted(row) != sorted(arms):
            raise ValueError(f"an explicit block {row!r} must hold every arm once")
    return dict(raw)


def plan_order(plan: ExperimentPlan) -> RunOrder:
    """Each block's arms in the order they run."""
    names = list(plan.arms)
    kind = str(plan.order["kind"])
    if kind == "explicit":
        rows = [list(row) for row in plan.order["blocks"]]
        balanced = True
    elif kind == "williams":
        square = williams(len(names))
        rows = [[names[i] for i in square[b % len(square)]] for b in range(plan.blocks)]
        balanced = plan.blocks % len(square) == 0
    else:
        rng = random.Random(plan.seed)
        rows = []
        for _ in range(plan.blocks):
            row = list(names)
            rng.shuffle(row)
            rows.append(row)
        balanced = False
    return RunOrder(rows, kind, balanced, _position_counts(rows, names))


def williams(t: int) -> list[list[int]]:
    """A Williams design: each arm in each position, each ordered pair adjacent once.

    For an even number of arms it is t sequences; for an odd number, the
    square and its mirror, 2t sequences.
    """
    if t < 1:
        return []
    first, low, high = [0], 1, t - 1
    while len(first) < t:
        first.append(low)
        low += 1
        if len(first) < t:
            first.append(high)
            high -= 1
    square = [[(x + shift) % t for x in first] for shift in range(t)]
    if t % 2 == 1:
        square += [list(reversed(row)) for row in square]
    return square


def _position_counts(
    rows: Sequence[Sequence[str]], names: Sequence[str]
) -> dict[str, list[int]]:
    counts = {name: [0] * len(names) for name in names}
    for row in rows:
        for position, name in enumerate(row):
            counts[name][position] += 1
    return counts


def block_seed(plan: ExperimentPlan, block: int) -> int:
    """The seed every arm of a block shares: 31 bits of SHA-256(seed:block)."""
    digest = hashlib.sha256(f"{plan.seed}:{block}".encode()).hexdigest()
    return int(digest, 16) & 0x7FFFFFFF


def expand(template: str, values: Mapping[str, Any]) -> str:
    """A template with its placeholders filled; unknown ones are an error."""

    def fill(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"placeholder {{{key}}} has no value here")
        return str(values[key])

    return _PLACEHOLDER.sub(fill, template)


def _digest(document: Any) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "ORDER_KINDS",
    "PLACEHOLDERS",
    "PLAN_FORMAT",
    "PLAN_VERSION",
    "Arm",
    "ExperimentPlan",
    "Prelude",
    "RunOrder",
    "ServerSpec",
    "Step",
    "Treatment",
    "block_seed",
    "expand",
    "load_plan",
    "plan_from_document",
    "plan_order",
    "williams",
]
