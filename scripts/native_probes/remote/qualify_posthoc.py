"""Post-hoc re-checks after the runtime-check fix (not part of the declared plan).

The declared run found that helper 0.4.0 refused every process it was injected
into under Nsight Systems with "CUDA runtime version unavailable", so check D's
subscriber path was never reached. The fix compares versions when capture
starts and at finalize instead of at injection. This re-runs, with the old and
fixed builds where it shows the difference: the Nsight coexistence trial, the
same microbenchmark without Nsight, and checks A and C for regressions.
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

from scripts.native_probes.remote import micro_fidelity, qualify_cli

ROOT = Path("/home/v2")
OLD = ROOT / "libstormlog_cupti_cu13.so"
FIXED = ROOT / "libstormlog_cupti_cu13_fixed.so"
FIXED_SYSTEM = ROOT / "libstormlog_cupti_system_fixed.so"


def _with_helper(helper: Path, trial: Callable[[], Any]) -> Any:
    micro_fidelity.HELPER = str(helper)
    try:
        return trial()
    finally:
        micro_fidelity.HELPER = str(OLD)


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    checks: dict[str, Callable[[], Any]] = {}
    for label, helper in (("old", OLD), ("fixed", FIXED)):
        micro = partial(micro_fidelity.cupti_trial, "w1-eager", out / f"micro-{label}")
        nsys = partial(
            micro_fidelity.nsys_trial,
            "w1-eager",
            out / f"nsys-{label}",
            with_cupti=True,
        )
        checks[f"micro_cupti_only_{label}"] = partial(_with_helper, helper, micro)
        checks[f"nsys_coexistence_{label}"] = partial(_with_helper, helper, nsys)
    checks["A_fixed"] = lambda: qualify_cli.check_a(out / "a-fixed", FIXED)
    checks["C_fixed"] = lambda: qualify_cli.check_c(out / "c-fixed", FIXED_SYSTEM)
    report: dict[str, Any] = {}
    for name, check in checks.items():
        try:
            report[name] = check()
        except Exception as exc:  # recorded, so one failure does not hide others
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
        (out / "posthoc.json").write_text(
            json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
        )
    print(json.dumps({"posthoc": str(out / "posthoc.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
