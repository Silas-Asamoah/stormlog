---
orphan: true
---

# Issue #259 implementation validation

Implementation base: `7c94fb01c90e6dd9c48257aea8f3fdc53f91bcd6`.
Branch: `feat/compact-inference-contexts`. The measurements below were recorded
on the implementation worktree before committing.

## Contracts and integration coverage

| Contract/path | Evidence |
| --- | --- |
| All eight event classes and span/interval activities | `test_infer_context_codec.py::test_all_event_types_schema_and_typed_roundtrip`; validates physical v4 rows and typed/embedded equality. Existing embedded-schema fixtures remain. |
| Full context identity, aliases, collisions, optional defaults | Codec parametrization, alias append test, injected hash collision test; definition conflicts are fatal. |
| Legacy/mixed records and reader lifetime | Codec tests for semantic ordering, legacy null-interval preservation, unrelated writer inputs, independent files and module imports. |
| Invalid transport and physical error lines | Codec tests for IDs, versions/types, missing/both context forms, definition fields, invalid intervals, unknown IDs, blanks, and typed failures after definitions. |
| Append reuse and failure before mutation | `test_infer_capture.py`: repeated appends, embedded prefix preservation, corrupt existing registry, unserializable metadata, newline and permissions tests. |
| Deduplication, conflicts, multiple runs and device/clock separation | Codec alias accounting test and `test_infer_accounting.py::test_compact_accounting_keeps_conflicts_runs_and_device_clock_dimensions`; production accounting code unchanged. |
| Kineto graph/overlap at launch and kernel detail | Parametrized `test_capture_records_the_import_summary_and_registers_the_trace` compares original production capture events/accounting with loaded compact events. |
| Trace re-import in compact/mixed artifacts | Parametrized `test_importing_the_same_trace_twice_skips_it` checks unchanged physical line count. |
| Nsight PID/device/correlation identity and accounting | `test_compact_capture_preserves_process_device_identity_and_accounting`; original capture compared with loaded events. |
| Execution high-water and re-import | Parametrized `test_a_second_import_adds_only_what_became_final`: compact and embedded seeds, unchanged mark on re-import, only newly finalized iterations added. Existing foreign-ID scheme tests run on production compact append output. |
| Profile, automatic trace capture and execution profile | Existing production profile integration suites exercise the shared live writer and append paths; profile test explicitly asserts definitions and expands before checking run/session/clock values. |
| Multi-process/device execution contexts | Existing execution device fixtures plus full-context codec parametrization and multi-process Nsight round-trip. |
| Direct execution report input | Compact/embedded report equality, missing definition fails before skipped-record handling. Existing malformed embedded-record tests retained. |
| Clock alignment | Compact generator test expands before filtering, deduplicates own alignments and preserves foreign alignments. |
| Public telemetry analysis | Compact alignment join equals embedded baseline; missing definition produces `InferInputError`. |
| Hook admission and direct vLLM analysis | Compact request hook remains server admission evidence; direct compact report equals embedded report with joined span. |
| Incident output and limits | Loaded definition excluded, one semantic identity; reservation byte count equals actual encoded bundle bytes. Existing budget/cleanup tests retained. |
| Documentation/schema examples | `test_compact_correlation_schema_and_examples_match` validates and expands the documented v4 example and checks links. |
| Measurement workflow | Offline scenario tests at both detail levels use the existing deterministic graph/overlap fixture and production import API; compare original imported events, paired files, graphs and accounting. This is script validation, not the real audit benchmark. |

## Reader audit

The typed loader and capture preflight share the codec's physical-line iterator.
File analysis uses `read_inference_records`. `execution_report`, `vllm_report`,
and `artifact_alignments` expand complete input sequences before filtering.
Single-record helpers (including client clock-domain lookup) receive expanded
values. The fake-engine qualification test helper also reads expanded records,
so its profiler/hook/span integrations inspect semantic contexts. `_hook_request_ids` retains its embedded-v2 check, so v1 client records
cannot become server admission evidence. Trace and execution import helpers
already use the typed loader. Terminal profile appends write v1 records.

The remaining independent JSON readers consume different contracts:
`telemetry.load_telemetry` reads server samples; `vllm_spans` reads OTLP exports;
`vllm_execution_log` reads worker hook logs; watch history/ledger/config/store
read scrapes, control records and manifests. Query/catalog/TUI use memory
telemetry and attachment metadata rather than parsing inference correlation
records themselves. JSON response, arrivals and SLO readers retain their
existing separate inputs. No registry logic was added to those paths.

## Compatibility detail

The legacy model historically accepts `metadata.intervals: null` in an embedded
v3 record. Its interpretation remains unchanged. The codec rejects null in v4
rather than allowing it to become a plausible GPU span. Encoding that legacy
edge case retains its embedded representation; normal new correlation events
use compact v4. Unrelated legacy writer dictionaries still pass through.

## Checks and limitations

Checks on the implementation worktree:

| Check | Result |
| --- | --- |
| Final inference, documentation and qualification suites | 2,088 passed in 273.83 seconds; rerun after all implementation and publication edits. |
| Full inference suite | 1,854 passed, 2 skipped in 199.63 s. Later-added coverage was also run in the focused checks below. |
| Latest codec/correlation/accounting/capture suite | 81 passed. |
| Latest codec and documentation regression tests | 150 passed. |
| Broader affected integration selection | 424 passed, 1 skipped (before the final additional capture assertion). |
| PyTorch trace tests after installing the optional runtime | 6 passed. |
| Full core suite | 3,328 passed, 18 skipped, 19 deselected; two failures. The compact-format qualification helper was then fixed and all 37 affected qualification tests passed. The other failure (`test_vllm_hook.py::test_a_record_json_cannot_serialize_is_an_error_and_takes_no_number`) reproduces on the base archive under Python 3.14. Full core was not rerun after the helper fix. |
| Repository isort and flake8 | Passed. |
| Changed-file Black | Passed. Repository Black reports six unchanged baseline files; the base archive reproduces them. |
| Complexity | Passed; maximum 10, no baseline exceptions added. |
| Mypy | One pre-existing unused `type: ignore` in `vllm_spans.py:1168`; reproduced on the base archive with the same environment. No errors in new/changed code. |
| Clean Sphinx with warnings treated as errors | HTML builds, but check fails with 18 warnings: 16 reproduced on the base archive plus the two untracked planning documents outside the toctree. No warnings introduced by the new format/benchmark docs. |
| `git diff --check` | Passed. |

The Black baseline files are `stormlog/tui/builders.py`,
`tests/test_import_hardening.py`, `tests/test_infer_trace_torch.py`,
`tests/test_jax_import_hardening.py`, `tests/test_profiler_regressions.py`, and
`tests/test_vllm_hook.py`. They were not reformatted. The Sphinx baseline
warnings are ambiguous cross-references to existing `bytes` members.

Raw command logs were saved under `/tmp/stormlog-*.log` during this run. The
base archive used for comparison was produced with `git archive HEAD` into a
fresh temporary directory. No baseline files in the worktree were replaced.

The hook baseline failure arises because the Python 3.14 JSON encoder accepts
the deliberately adversarial `Changing` dictionary subclass in that test,
while its assertion expects rejection and no sequence number. This inference
storage change does not modify the hook writer or that test.

Use `.venv/bin/python` or `uv run --no-project --python .venv/bin/python python`.
The local environment uses Python 3.14.7; supported CI versions still need CI
verification. Dependencies are environment-only; no dependency files changed.

The maintainer authorized a fresh small-model Modal capture in place of waiting
for the contributor's original audit. The [real vLLM trace benchmark](issue-259-context-compaction.md)
now records 51,496 GPU events, paired launch/kernel reductions of 28.68%/28.88%,
exact semantic/graph/accounting equality and unchanged per-device/clock timing.
The original audit remains unreproduced. The standard-profiler capture has no
Stormlog iteration hooks, so all activities retain unresolved attribution.
The bounded capture script passed Black, isort, flake8 and syntax checks; its
highest callable complexity is 6. CUDA image checks and the real GPU run passed.
After adding the capture report, codec/documentation regressions again passed
all 150 tests. A fresh Sphinx build retained the same 18 baseline/planning-file
warnings; the new report and JSON results introduced none. The review bundle's
24 evidence files matched its manifest, and downloading the persisted Modal
bundle reproduced the local SHA-256 exactly.
