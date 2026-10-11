"""Run an experiment plan: matched arms, blocks, a fresh server per run.

The plan (``stormlog.infer.experiment_plan`` v1, see
``docs/inference_experiments.md``) names the server, the arms and their
workload steps, and the order. Each run's artifacts land in
``<output>/runs/<label>/``, ready for ``stormlog infer compare``::

    python -m examples.cli.infer_repeated_baseline --plan plan.json --output exp213

Run it again with ``--resume`` to continue an interrupted experiment, and
with ``--retry-incomplete`` to retry runs that ended in a protocol failure.
A resume refuses to go on while a run the runner was killed in is not
explained: give its cause with its evidence, which sets it aside and lets
``--retry-incomplete`` run it again, or mark it an outcome failure, kept::

    --external-cause t221-b03-p1-watch-a1=spot_preemption:"box paused at 03:12"
    --interrupted-as-outcome t221-b03-p1-watch-a1

Exit codes, for the whole experiment, not only the runs this invocation
ran: 0 when every planned run finished (completed, or an outcome failure,
which is data); 3 when some has not, its last attempt a protocol failure or
none finished (``not_run``), each named on stderr; 5 for a plan or output
directory it cannot use; 2 for a secret the plan names that is not set.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stormlog.exit_codes import ExitCode
from stormlog.infer.errors import InferInputError, InferUsageError
from stormlog.infer.experiment import ExternalCause, run_plan, unfinished_runs
from stormlog.infer.experiment_plan import load_plan


def _print(record: dict[str, Any]) -> None:
    reasons = ", ".join(record["reasons"]) or "-"
    print(f"{record['label']}: {record['state']} ({reasons})", flush=True)


def _causes(items: Sequence[str]) -> dict[str, ExternalCause]:
    """``LABEL=REASON:EVIDENCE`` for each attempt the runner was killed in."""
    causes = {}
    for item in items:
        label, _, rest = item.partition("=")
        reason, _, evidence = rest.partition(":")
        if not label or not rest:
            raise InferUsageError(
                f"--external-cause {item!r} is not LABEL=REASON:EVIDENCE"
            )
        causes[label] = ExternalCause(reason, evidence)
    return causes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument(
        "--external-cause", action="append", default=[], metavar="LABEL=REASON:EVIDENCE"
    )
    parser.add_argument(
        "--interrupted-as-outcome", action="append", default=[], metavar="LABEL"
    )
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        run_plan(
            plan,
            args.output,
            resume=args.resume,
            retry_incomplete=args.retry_incomplete,
            on_event=_print,
            external_causes=_causes(args.external_cause),
            interrupted_as_outcome=args.interrupted_as_outcome,
        )
    except InferUsageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return int(ExitCode.USAGE)
    except InferInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return int(ExitCode.INVALID_INPUT)
    unfinished = unfinished_runs(plan, args.output)
    if unfinished:
        runs = "run" if len(unfinished) == 1 else "runs"
        named = ", ".join(f"{label} ({state})" for label, state in unfinished.items())
        print(
            f"{len(unfinished)} planned {runs} not finished: {named}", file=sys.stderr
        )
        return int(ExitCode.FINDINGS)
    return int(ExitCode.OK)


if __name__ == "__main__":
    sys.exit(main())
