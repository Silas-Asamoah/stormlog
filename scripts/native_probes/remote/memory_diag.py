"""Post-hoc diagnostic: where does the CUPTI helper's added RSS live?

The qualification run found server RSS growing about 17 bytes per record in
CUDA-graph mode while activity buffers stayed at 24 MiB. This runs one
vllm_window trial and, every few seconds, records each injected process's
smaps_rollup and every mapping holding more than 1 MiB, so growth can be
attributed to the heap, anonymous mappings (such as per-thread malloc arenas)
or file mappings. Not part of the declared plan.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/v2")
HELPER_NAME = "libstormlog_cupti"
INTERVAL_SECONDS = 10.0


def injected_pids() -> list[int]:
    pids = []
    for maps in Path("/proc").glob("[0-9]*/maps"):
        try:
            if HELPER_NAME in maps.read_text():
                pids.append(int(maps.parent.name))
        except OSError:
            continue
    return sorted(pids)


def rollup(pid: int) -> dict[str, int]:
    fields = {}
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()[1:]:
        name, value = line.split(":", 1)
        fields[name] = int(value.split()[0])
    return fields


def large_mappings(pid: int, minimum_kib: int = 1024) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in Path(f"/proc/{pid}/smaps").read_text().splitlines():
        head = line.split()
        if "-" in head[0] and not head[0].endswith(":"):
            current = {
                "range": head[0],
                "perms": head[1],
                "path": head[5] if len(head) > 5 else "",
            }
            mappings.append(current)
        elif current is not None and head[0] in ("Rss:", "Anonymous:"):
            current[head[0][:-1].lower() + "_kib"] = int(head[1])
    return [m for m in mappings if m.get("rss_kib", 0) >= minimum_kib]


def main() -> int:
    out = Path(sys.argv[1])
    mode = sys.argv[2]
    out.mkdir(parents=True, exist_ok=False)
    trial = subprocess.Popen(
        [
            str(ROOT / "venv/bin/python"),
            "-m",
            "scripts.native_probes.workloads.vllm_window",
            "--output",
            str(out / "trial"),
            "--helper",
            str(ROOT / "libstormlog_cupti_cu13_fixed.so"),
            "--mode",
            mode,
            "--window",
            "800",
            "--cuda-graphs",
        ],
        cwd=ROOT / "src",
    )
    with (out / "memory.jsonl").open("w", encoding="utf-8") as samples:
        while trial.poll() is None:
            for pid in injected_pids():
                try:
                    row = {
                        "at_ns": time.time_ns(),
                        "pid": pid,
                        "rollup_kib": rollup(pid),
                        "mappings": large_mappings(pid),
                    }
                except (OSError, ValueError):
                    continue
                samples.write(json.dumps(row) + "\n")
                samples.flush()
            time.sleep(INTERVAL_SECONDS)
    print(json.dumps({"diag": str(out), "trial_exit": trial.returncode}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
