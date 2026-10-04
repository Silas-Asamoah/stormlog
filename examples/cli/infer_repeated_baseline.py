"""Run an experiment plan: matched arms, blocks, a fresh server per run.

The plan (``stormlog.infer.experiment_plan`` v1, see
``docs/inference_experiments.md``) names the server, the arms and their
workload steps, and the order. Each run's artifacts land in
``<output>/runs/<label>/``, ready for ``stormlog infer compare``::

    python -m examples.cli.infer_repeated_baseline --plan plan.json --output exp213

Run it again with ``--resume`` to continue an interrupted experiment, and
with ``--retry-incomplete`` to retry runs that ended in a protocol failure.
A run the runner was killed in is then an outcome failure, unless its cause
is given with its evidence::

    --external-cause t221-b03-p1-watch-a1=spot_preemption:"box paused at 03:12"

Exit codes: 0 when every planned run finished (completed, or an outcome
failure, which is data); 3 when some run ended in a protocol failure; 5 for
a plan or output directory it cannot use; 2 for a secret the plan names
that is not set.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stormlog.exit_codes import ExitCode
from stormlog.infer.errors import InferInputError, InferUsageError
from stormlog.infer.experiment import PROTOCOL_FAILURE, ExternalCause, run_plan
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
    args = parser.parse_args(argv)
    try:
        records = run_plan(
            load_plan(args.plan),
            args.output,
            resume=args.resume,
            retry_incomplete=args.retry_incomplete,
            on_event=_print,
            external_causes=_causes(args.external_cause),
        )
    except InferUsageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return int(ExitCode.USAGE)
    except InferInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return int(ExitCode.INVALID_INPUT)
    last = {}
    for record in records:
        last[(record["block"], record["arm"])] = record["state"]
    if any(state == PROTOCOL_FAILURE for state in last.values()):
        return int(ExitCode.FINDINGS)
    return int(ExitCode.OK)


if __name__ == "__main__":
    sys.exit(main())
