"""``python -m examples.qualification inject``: run one injection run.

```bash
python -m examples.qualification inject --plan PLAN.json --out ROOT \\
  --base-url URL --model M --reference-channel HOOK_DIR \\
  [--api-server-pid PID] [--target role=PID ...] [--label q221-...] \\
  -- [extra infer profile arguments]
```

The server is launched by someone else (#213's ``run_plan``), with the
execution hook writing into ``HOOK_DIR``. ``--api-server-pid`` finds the
API server's EngineCore and TP workers by their vLLM titles; ``--target``
names any other process. Every target is bound to its start time here, at
startup, so a pid reused later is never signalled. Prints the published
run directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import psutil

from .inject import InjectionRun, Server
from .plan import PlanError, load_plan
from .pulser import Target, discover_roles
from .run_dir import RunDirectory, new_label


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m examples.qualification")
    commands = parser.add_subparsers(dest="command", required=True)
    inject = commands.add_parser("inject", help="Run one injection run")
    inject.add_argument("--plan", type=Path, required=True)
    inject.add_argument("--out", type=Path, required=True)
    inject.add_argument("--label", default=None, help="Default: a fresh q221- label")
    inject.add_argument("--base-url", required=True, help="The server's root URL")
    inject.add_argument("--model", required=True)
    inject.add_argument("--binding", default="vllm-0.30")
    inject.add_argument(
        "--reference-channel",
        type=Path,
        required=True,
        help="The server's STORMLOG_VLLM_HOOK_DIR, as this host sees it",
    )
    inject.add_argument(
        "--api-server-pid",
        type=int,
        default=None,
        help="The vLLM API server: its EngineCore and TP workers become targets",
    )
    inject.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="ROLE=PID",
        help="A process the plan may pulse, by role (engine_core, api_server, sidecar)",
    )
    return parser


def _targets(api_server_pid: int | None, entries: Sequence[str]) -> dict[str, Target]:
    """Every target, bound to its start time now.

    Raises:
        ValueError: for a malformed entry, a process that isn't there, or a
            server whose roles can't be told apart.
    """
    targets = {} if api_server_pid is None else discover_roles(api_server_pid)
    for entry in entries:
        role, _, pid = entry.partition("=")
        if not role or not pid.isdigit():
            raise ValueError(f"--target {entry!r} is not ROLE=PID")
        try:
            targets[role] = Target.of(int(pid), role)
        except psutil.Error as error:
            raise ValueError(f"--target {entry!r}: {error}") from error
    return targets


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    args = _parser().parse_args(argv[:split])
    try:
        plan = load_plan(args.plan)
        targets = _targets(args.api_server_pid, args.target)
    except (PlanError, ValueError, OSError, psutil.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if args.binding != plan.binding:
        print(f"Error: the plan's binding is {plan.binding}", file=sys.stderr)
        return 2
    server = Server(
        args.base_url.rstrip("/"), args.model, args.reference_channel, targets
    )
    directory = RunDirectory(args.out, args.label or new_label())
    run = InjectionRun(plan, directory, server, argv[split + 1 :])
    try:
        published = run.execute()
    except KeyboardInterrupt:
        print(f"Interrupted; the run so far is published as {directory.final}")
        return 130
    print(published)
    if run.failure is not None:
        print(f"Error: the run failed: {run.failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
