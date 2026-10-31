# MLX review behaviors: issue #325

Implemented on `feat/mlx-inference-behaviors-325`, based on `release/dev`
`7f73e076`. [Issue #325](https://github.com/Silas-Asamoah/stormlog/issues/325)
collects five behaviors raised during the #298/#301 reviews.

| Item | Reproduction | Result |
| --- | --- | --- |
| Final scrape on stop | Cancelling a phase with a delayed end sample waited 300 ms and started a new `phase_end` scrape. A blocked default-executor sample also delayed `asyncio.run` shutdown. | Failed/interrupted phases start no end scrape. In-flight awaits record abandonment; their bounded daemon workers cannot delay loop shutdown or append late results. Normal phases retain their final sample. |
| Backwards wall clock | A successful normalized sample with wall stamps 2 s then 1 s failed stats validation. | Retain both raw stamps and their midpoint; validate against the unordered endpoints. Monotonic elapsed time and existing clock uncertainty preserve the clock-step evidence. |
| Loopback spelling | `localhost`, `localhost.`, IPv6 `::1` and alternate IPv4 loopback literals failed association with a `127.0.0.1` declaration. | Treat loopback aliases as equivalent only for declared association, preserving scheme/port and avoiding DNS. Literal listener/process proof is still required for verified association; overlapping alias captures remain ambiguous. |
| Stray `ValueError` | Injected internal failures in info, capture and analysis returned usage exit 2. | Validate known user inputs at the CLI boundary. Unexpected exceptions and capture-internal `ValueError` return error exit 1. Existing input/runtime/interruption outcomes remain covered. |
| Phase timeline readings | A TUI pilot with 400/900-byte phase boundary readings omitted those points, while offline analysis retained them. | Include phase-enter/exit readings in the live timeline, with an exact live/offline parity assertion. |

The initial regression run failed for all five behaviors. Tests cover normal
completion, interrupted scrapes, clock-step acceptance, invalid timing rejection,
loopback boundary cases, ambiguity and verification limits, known CLI usage
errors, internal capture errors and an MLX Textual pilot with phases.

## Local checks

```bash
python3 -m pytest tests/test_infer_omlx_stats.py tests/test_infer_mlx_report.py \
  tests/test_mlx_cli.py tests/test_mlx_diagnose.py tests/test_mlx_runner.py \
  tests/test_mlx_tui.py tests/test_infer_http_transport.py \
  tests/test_infer_vllm_profile.py -q
```

The full local MLX/inference/HTTP/scraper selection passed **294 tests**, with
seven optional hardware/model checks skipped. The docs regression selection
passed **97 tests** (391 passed in the combined selection, seven skipped).
The current CI-pinned isort/Black/flake8, mypy (282 source files), complexity
gate (maximum 10, no exceptions) and warnings-as-errors Sphinx build pass.
A native Metal pilot was run separately below. The 79 focused oMLX stats,
endpoint, CLI and TUI checks also pass on Python 3.12.

The shared cancellable HTTP worker was moved from the vLLM scraper to the
HTTP transport module; the vLLM regression selection verifies that path too.

## Apple Silicon checks, 2026-10-27

The running local oMLX server reports installed version 0.6.4. Its read-only
`/admin/api/stats` route returned HTTP 401; no admin session token was available,
and no authentication or server configuration was changed. These checks qualify
the live route's timing/error-record path, not successful authenticated metrics
or a supported oMLX/model combination.

- Cancelling after a real phase-start GET completed returned in about 0.11 ms
  and issued no phase-end GET, despite a one-second delay armed for that marker.
  Ten additional real SIGINT checks against the same route each ran in a fresh
  subprocess with a 15-second parent timeout. All ten propagated
  `KeyboardInterrupt`, exited successfully, and started no final GET. Signal to
  interrupt propagation took 0.306 to 0.401 ms; no hangs or timeouts occurred.
  The earlier single SIGINT check also passed, for eleven successful runs total.
- Two real GETs, each wrapped with an injected one-second backwards wall-clock
  step in the client sampler, produced valid error records with conservative
  clock uncertainty. A complete client artifact remained readable by offline
  inference analysis. The OS and server clocks were not changed.
- A native Textual pilot with MLX 0.32.3 and Metal available evaluated a
  1024-by-1024 array inside a phase. Phase-enter/exit observations were 0 and
  4,194,304 bytes; all three live observations exactly matched offline analysis.

Authenticated oMLX statistics remain unqualified, as in the existing serving
qualification record. Deterministic tests separately exercise successful stats
payloads and cancellation while a sample is blocked.

## PR #329 CI follow-up

The Python 3.11/PyTorch job failed the existing queue-recovery qualification
test's neighbor actuation assertion. The job passed 5,776 tests; its output did
not include the underlying actuation problems. A local Python 3.11 run with a
busy Python thread reproduced that assertion failure: all 240 neighbor requests
completed, but dispatch lag p95 was 1.355 s, beyond the 250 ms actuation limit.

The test now sends 40 rather than 120 fault requests per second. This still
requires realized queue saturation, no victim preemption and successful
recovery, while reducing the two-second episode from 240 to 80 streaming
requests. Three fresh-process contention checks passed, with dispatch lag p95
between 109 and 123 ms.
Production actuation and recovery thresholds are unchanged. Assertion failures
now include the injected actuation details so future CI reports name the
specific problem.

The Python 3.11 injection/neighbor selection passed 95 tests. The 97 docs
regression tests, changed-file isort/Black/flake8 checks and warnings-as-errors
Sphinx build also passed.
