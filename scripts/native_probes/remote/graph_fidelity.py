"""Compare cupti-min with Nsight on the vLLM eager and CUDA-graph trials.

vLLM batching differs between runs, so exact kernel counts cannot match across
separate processes. The comparison requires the same kernel name set, at least
99% launch correlation and, for graph mode, kernels attributed to graph nodes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.native_probes import cupti_trace


def main() -> int:
    runs = Path(sys.argv[1])
    report = {}
    for execution in ("eager", "graph"):
        cupti_dir = runs / f"{execution}-cupti-min-w200" / "cupti"
        nsys_dir = runs / f"{execution}-nsys-w200" / "nsys"
        sqlite = nsys_dir / "trace.sqlite"
        if not sqlite.exists():
            subprocess.run(
                [
                    "nsys",
                    "export",
                    "--type=sqlite",
                    f"--output={sqlite}",
                    str(nsys_dir / "trace.nsys-rep"),
                ],
                capture_output=True,
                check=False,
            )
        cupti = cupti_trace.cupti_summary(sorted(cupti_dir.rglob("activity.sclz")))
        nsys = cupti_trace.nsys_summary(sqlite)
        comparison = cupti_trace.compare(cupti, nsys, exact=False)
        if execution == "graph":
            comparison["graph_nodes_attributed"] = (
                cupti["graph_node_kernels"] > 0 and nsys["graph_node_kernels"] > 0
            )
            comparison["pass"] = (
                comparison["pass"] and comparison["graph_nodes_attributed"]
            )
        for summary in (cupti, nsys):
            summary.pop("kernel_counts_by_name")
        report[execution] = {"cupti": cupti, "nsys": nsys, "comparison": comparison}
    (runs / "graph_fidelity.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v["comparison"] for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
