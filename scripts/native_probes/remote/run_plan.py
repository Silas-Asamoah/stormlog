"""Run a v2 trial plan in order, resumably, on one GPU host.

Each trial gets a fresh server. A trial whose result.json exists is skipped, so
the plan can resume after a pause. Large trace files are hashed, then removed
unless the trial asks to keep them; every hash is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

KEEP_LIMIT_BYTES = 256 * 1024**2
MIN_FREE_BYTES = 20 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/home/v2"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    runs = args.root / "runs" / plan["plan_id"]
    runs.mkdir(parents=True, exist_ok=True)
    progress = (runs / "progress.log").open("a", encoding="utf-8")
    for trial in plan["trials"]:
        output = runs / trial["trial_id"]
        if (output / "result.json").exists():
            continue
        if shutil.disk_usage(args.root).free < MIN_FREE_BYTES:
            progress.write(
                f"{time.time():.0f} STOP low disk before {trial['trial_id']}\n"
            )
            return 3
        if output.exists():
            shutil.move(str(output), str(output) + f".aborted-{int(time.time())}")
        command = [
            str(args.root / "venv/bin/python"),
            "-m",
            "scripts.native_probes.workloads.vllm_window",
            "--output",
            str(output),
            "--helper",
            str(args.root / "libstormlog_cupti.so"),
            *trial["args"],
        ]
        started = time.time()
        progress.write(
            f"{started:.0f} START {trial['trial_id']} {' '.join(trial['args'])}\n"
        )
        progress.flush()
        completed = subprocess.run(
            command,
            cwd=args.root / "src",
            timeout=trial.get("timeout", 1800),
            capture_output=True,
            text=True,
        )
        (output / "runner-stdout.log").write_text(completed.stdout)
        (output / "runner-stderr.log").write_text(completed.stderr)
        hashes = []
        for path in sorted(p for p in output.rglob("*") if p.is_file()):
            hashes.append(
                f"{_sha256(path)}  {path.stat().st_size}  {path.relative_to(output)}"
            )
            if not trial.get("keep_traces") and path.stat().st_size > KEEP_LIMIT_BYTES:
                path.unlink()
                hashes[-1] += "  (removed after hashing)"
        (output / "artifacts.sha256").write_text("\n".join(hashes) + "\n")
        progress.write(
            f"{time.time():.0f} END {trial['trial_id']} exit={completed.returncode} "
            f"seconds={time.time() - started:.0f}\n"
        )
        progress.flush()
    progress.write(f"{time.time():.0f} PLAN COMPLETE\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
