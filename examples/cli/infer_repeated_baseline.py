"""Run an experiment plan: matched arms, blocks, a fresh server per run.

The plan (``stormlog.infer.experiment_plan`` v1, see
``docs/inference_experiments.md``) names the server, the arms and their
workload steps, and the order. Each run's artifacts land in
``<output>/runs/<label>/``, ready for ``stormlog infer compare``::

    python -m examples.cli.infer_repeated_baseline --plan plan.json --output exp213

Run it again with ``--resume`` to continue an interrupted experiment, and
with ``--retry-incomplete`` to retry runs that ended in a protocol failure.

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
from stormlog.infer.experiment import PROTOCOL_FAILURE, run_plan
from stormlog.infer.experiment_plan import load_plan


def _print(record: dict[str, Any]) -> None:
    reasons = ", ".join(record["reasons"]) or "-"
    print(f"{record['label']}: {record['state']} ({reasons})", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    args = parser.parse_args(argv)
    try:
        records = run_plan(
            load_plan(args.plan),
            args.output,
            resume=args.resume,
            retry_incomplete=args.retry_incomplete,
            on_event=_print,
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
