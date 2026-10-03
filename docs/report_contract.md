[← Back to docs](index.md)

# Report and Exit-Code Contract

Automated consumers such as CI jobs, agents, and the TUI command runner need
two things from a Stormlog command: a process exit code they can branch on
without parsing output, and a machine-readable report they can read later,
even when the process exit code was lost (for example from an artifact
uploaded with `if: always()`). This page is the contract for both.

- The exit-code table lives in `stormlog.exit_codes.ExitCode`.
- The report envelope is `stormlog.report` v1, published as
  `docs/schemas/stormlog_report_v1.schema.json`.

## Exit codes

| Code | Name | Meaning | Verdict status |
| --- | --- | --- | --- |
| 0 | `OK` | The command completed, no findings reached failure severity, and every configured gate passed. | `pass` |
| 1 | `ERROR` | An unexpected failure. Any output written may be incomplete. This is Python's own value for an uncaught exception. | `error` |
| 2 | `USAGE` | The command line is invalid, or this installation cannot serve the request (a missing framework runtime or optional extra, an output path that is not a directory). `argparse` owns this value. | `usage` |
| 3 | `FINDINGS` | The command completed and the measurement is sound, but it detected memory risk, OOM events, or leak findings. | `findings` |
| 4 | `GATE_FAILED` | The command completed, but a configured budget, tolerance, or threshold was exceeded. | `gate_failed` |
| 5 | `INVALID_INPUT` | An input artifact or asset is missing, unreadable, or has an unsupported schema or version. | `invalid_input` |
| 130 | `INTERRUPTED` | The command was stopped by SIGINT before it completed (the shell convention of 128 + 2). | `interrupted` |

How to read the table from a pipeline:

- `0`: proceed.
- `3` or `4`: the tool worked; the result is the problem. Read the report
  for the findings or the failed checks. Block or warn according to policy.
- `2` or `5`: fix the invocation, the environment, or the producing step.
  Retrying the same command will not help.
- `1`: treat as a tool failure and keep the output for a bug report.
- `130`: the run was cut short. `monitor` and `track` finalise their
  artifact with session status `interrupted`; `diagnose` leaves a partial
  bundle with neither `manifest.json` nor `report.json`, so treat the
  directory as incomplete.

`FINDINGS` and `GATE_FAILED` are separate because a gate is a deterministic
comparison against a threshold you configured, while a finding is a
heuristic detection by the tool. CI policy usually differs between the two.

### Rules

- Codes are never renumbered or reused. A new outcome takes the next free
  value below 128; values from 128 upward stay reserved for signals.
- Every code pairs with exactly one verdict status string. The pairing is
  published in `stormlog.exit_codes` and enforced by the report schema.
- A command may use a subset of the table but may not give a code another
  meaning.
- Ctrl+C during `monitor`, `track`, or `stormlog infer collect-server` is the
  documented way to end a capture: the artifact is finalised and the process
  exits 0. An interrupt anywhere else exits 130.
- `analyze` commands are read-only investigations. They exit 0 even when
  they surface findings; verdicts come from `diagnose` and from the gates.
  Read the analysis report for findings instead of the exit code.

### Where each command stands

| Command | 0 | 2 | 3 | 4 | 5 | 1 | 130 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gpumemprof diagnose` | no risk | bad `--duration`/`--interval`, `--native-history` off CUDA, PyTorch not installed, missing extra, `--output` that is (or is under) a file | risk flag raised | - | - | bundle not writable; any other unexpected error (report and manifest say `error`/1) | Ctrl+C (partial bundle, no report) |
| `tfmemprof diagnose`, `jaxmemprof diagnose` | no risk | as above, with TensorFlow/JAX not installed | risk flag raised | - | - | as above | as above |
| `gpumemprof analyze`, `tfmemprof analyze`, `jaxmemprof analyze` | done, including a JSON document with no telemetry events (a note is printed) | `tfmemprof analyze` without `--input` | - | - | missing file; unparsable JSON; (tf/jax) JSON that is not an object; requested session id not found | unexpected error | Ctrl+C |
| `gpumemprof monitor`/`track` | done, or Ctrl+C inside the capture loop; without PyTorch they fall back to the CPU tracker | missing W&B/MLflow extra | - | - | - | unexpected error | Ctrl+C outside the capture loop |
| `tfmemprof`/`jaxmemprof monitor`/`track` | done, or Ctrl+C inside the capture loop | TensorFlow/JAX not installed; missing extra | - | - | - | unexpected error | Ctrl+C outside the capture loop |
| `stormlog query ...` | done | argparse error; `--csv` on a query that cannot emit CSV | - | - | - | unexpected error | Ctrl+C |
| `examples.cli.benchmark_harness` | gates passed, or no `--check` | argparse error; regression defaults outside the `pr` profile | - | a budget or regression gate failed under `--check` | budget, baseline, or tolerance asset missing, unparsable, not an object, wrong version, non-numeric, or missing a metric; baseline config mismatch (file problems are checked before any scenario runs; a missing metric after) | unexpected error; `--artifact-root` or `--output` not writable | Ctrl+C |
| `stormlog infer profile` | done, with at least one measured request succeeding | argparse error; a setting it cannot use, checked before anything is sent; a requested tokenizer that is not installed, or `--strict-token-counts` with no tokenizer backend installed; an `--arrival-trace` case the trace does not have, or no `--arrival-trace-case` when it has several | no measured request succeeded | - | `--arrival-trace` missing, unparsable, or with invalid offsets | unexpected error | Ctrl+C (the artifact ends with an `interrupted` session record) |
| `stormlog infer analyze` | done, including an artifact in which every request failed | argparse error | - | - | artifact, `--server-telemetry` or `--vllm-spans` file missing, unparsable, or invalid, including an artifact with no `infer.session` or `infer.request` records | unexpected error; `--output` not writable | Ctrl+C |
| `stormlog infer collect-server` | duration elapsed, Ctrl+C or SIGTERM, or the server process ended | argparse error; options it cannot use; a `--pid` with no running process; a `--device-index` or `--device-uuid` the host does not have; no NVML library without `--no-gpu` | the GPU identity changed mid-run | - | - | unexpected error | - |
| `stormlog infer import-trace` | traces imported, including GPU activity left unresolved or unmeasured (the summary says how much) | argparse error; a `--device-uuid` that is not `INDEX=UUID` or names one device twice | - | - | artifact without an `infer.artifact` record, or from another run or session; a trace file missing, unparsable, or not a Kineto trace | unexpected error; artifact or envelope not writable | Ctrl+C |

### Changes from earlier releases

These are breaking changes for callers that matched the old numbers:

- `gpumemprof`, `tfmemprof`, and `jaxmemprof diagnose` exit `3` for memory
  risk. They used to exit `2`, which could not be told apart from an
  `argparse` usage error from the same command. The bundle manifest's
  `exit_code` field follows.
- `gpumemprof`, `tfmemprof`, and `jaxmemprof analyze` exit `5` for a
  missing or unusable input. They used to exit `1`.
- Invalid diagnose options and a missing runtime or extra exit `2`. They
  used to exit `1`.
- `gpumemprof` exits `130` when interrupted outside a capture loop. It used
  to exit `0`. `tfmemprof` and `jaxmemprof` print "Operation cancelled by
  user" instead of a traceback (the code was already `130`).
- All three `diagnose` commands exit `2` when `--output` is, or sits under,
  an existing file. `gpumemprof` and `tfmemprof` used to exit `1`.
- `gpumemprof` exits `2` when PyTorch is not installed and a command needs
  it. It used to exit `1`.
- `examples.cli.benchmark_harness --check` exits `4` for a failed gate and
  `5` for an unusable asset. Both used to exit `1`. An unwritable
  `--artifact-root` or `--output` exits `1` with a message instead of a
  traceback. CI only checks for a non-zero code, so the workflow is
  unchanged.
- The W&B export of a diagnose bundle logs the manifest's `exit_code` as
  the `stormlog_exit_code` metric, so dashboards keyed on `2` for memory
  risk now see `3`.
- `stormlog infer` follows the table. Each of these used to exit `1`:
  - `infer profile` exits `3` when no measured request succeeds, `2` for a
    setting it cannot use, and `5` for an `--arrival-trace` it cannot read.
  - `infer analyze` exits `5` for an artifact or `--server-telemetry` file
    it cannot read.
  - `infer collect-server` exits `3` when the GPU identity changes, and `2`
    for options it cannot use, a `--pid` with no running process, a GPU the
    host does not have, or a host without NVML.

## The report envelope

A report is one JSON object with `format: "stormlog.report"` and
`schema_version: 1`. It carries the verdict paired with the exit code,
findings with evidence pointers, flat metrics, and pointers to the versioned
artifacts the command wrote. Tool-specific detail goes in `payload`, which
the producing command owns and versions, so the envelope stays small and
strict (`additionalProperties: false` at the top level).

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | `1` |
| `format` | yes | `stormlog.report` |
| `report_kind` | yes | Payload family, documented per command. Today: `diagnose`. |
| `generated_at_utc` | yes | ISO 8601 timestamp in UTC. |
| `tool` | yes | `name` (console script), `command` (subcommand), optional `version` and `argv`. |
| `verdict` | yes | `status`, `exit_code`, and a one-line `summary`. `status` and `exit_code` must pair as in the table above; the schema enforces it. |
| `findings` | yes | A list, possibly empty. Each finding has a stable `id`, a `kind`, a `severity` (`info`, `warning`, `critical`), a `title`, an optional `message`, optional `metrics`, and a list of `evidence` pointers. |
| `metrics` | no | Flat `name -> number or null`. Numbers are finite: JSON has no NaN or Infinity, so the schema cannot name them; `stormlog.report` rejects them on write and on load, and a producer records a value it cannot represent as `null`. |
| `artifacts` | no | Files the command wrote or relied on: `kind`, `path`, optional `format` and `schema_version`. |
| `recommendations` | no | Short actions, one string each. |
| `session_id`, `run_id` | no | Identity for joins with sessions and [run envelopes](run_envelopes.md). |
| `payload` | no | Tool-specific object. Not constrained by this schema. |

Evidence pointers reuse the vocabulary of `stormlog query correlate` rows:
`kind`, optional `path` (relative to the directory that holds the report),
optional `pointer` (a JSON pointer inside that file), `session_id`,
`record_id`, `start_ns`, `end_ns`, and a `description`.

`tool.argv` is optional and must not carry secrets: a producer that records
it redacts credentials (for example a tracking URI with a token) or omits
the field. The diagnose producer does not record `argv`; the bundle
manifest's pre-existing `command_line` field is the place that does.

A bundle the command could not finish still gets a report: its verdict is
`error`/`1` with the summary `Bundle incomplete: <reason>` and no findings,
so a report never claims a verdict the process did not return.

### Example: a diagnose bundle with findings

Every `diagnose` bundle now contains `report.json` next to `manifest.json`.
This example is `tests/fixtures/reports/diagnose_findings.json`, trimmed:

```json
{
  "schema_version": 1,
  "format": "stormlog.report",
  "report_kind": "diagnose",
  "generated_at_utc": "2026-10-01T12:00:00Z",
  "tool": {"name": "gpumemprof", "command": "diagnose", "version": "0.4.0"},
  "verdict": {
    "status": "findings",
    "exit_code": 3,
    "summary": "Memory risk detected: oom_occurred, high_utilization"
  },
  "findings": [
    {
      "id": "diagnose.oom_occurred",
      "kind": "oom",
      "severity": "critical",
      "title": "The allocator recorded out-of-memory events",
      "metrics": {"num_ooms": 1},
      "evidence": [
        {
          "kind": "diagnose_summary",
          "path": "diagnostic_summary.json",
          "pointer": "/risk_flags/oom_occurred"
        },
        {"kind": "session", "session_id": "9f1c2a4e-5b6d-4c7e-8f90-1a2b3c4d5e6f"}
      ]
    }
  ],
  "metrics": {"utilization_ratio": 0.9, "num_ooms": 1},
  "artifacts": [
    {"kind": "diagnose_manifest", "path": "manifest.json", "schema_version": 2},
    {"kind": "diagnose_summary", "path": "diagnostic_summary.json"}
  ],
  "recommendations": ["Reduce batch size or enable activation checkpointing."],
  "session_id": "9f1c2a4e-5b6d-4c7e-8f90-1a2b3c4d5e6f",
  "payload": {"risk_flags": {"oom_occurred": true, "high_utilization": true}}
}
```

Diagnose findings map one-to-one onto the `risk_flags` in
`diagnostic_summary.json`: `oom_occurred` (critical), `high_utilization`
(warning, with the observed ratio and the 0.85 threshold), and
`fragmentation_warning` (warning, with the observed ratio and the 0.3
threshold; always off for TensorFlow and JAX). `recommendations` are the
summary's `suggestions`.

### Reading and validating reports

```python
from stormlog.report import load_report

report = load_report(bundle_dir / "report.json")  # raises ValueError if invalid
if report["verdict"]["status"] in {"findings", "gate_failed"}:
    for finding in report["findings"]:
        print(finding["severity"], finding["title"])
```

`stormlog.report.validate_report` applies the same rules as the published
schema without a JSON Schema dependency. The contract tests check both
against the same fixtures so they cannot drift.

### Compatibility and versioning

- Every object in the envelope is closed (`additionalProperties: false`),
  so any change to its shape is a new `schema_version`, including adding an
  optional field. A consumer that validates with the schema matching the
  `schema_version` it reads therefore never sees an unknown field, and a
  v1 consumer can reject a v2 report by its `schema_version` alone.
- `payload` is versioned by the producing command and documented under its
  `report_kind`. A change to a payload is not a change to the envelope.
- `verdict.exit_code` always equals the exit code the producing process
  returned. A report never claims a verdict the process did not return.
- Exit-code numbers follow the rules above and never move between
  meanings. Adding a code is a documented change to this page, to
  `stormlog.exit_codes`, and to the `verdict` pairing in the schema.
- Commands that do not emit a report yet (`analyze`, the benchmark harness,
  `stormlog infer analyze`) keep their current JSON shapes. When they
  adopt the envelope, their current output becomes the `payload`.

## Related pages

- [Command Line Guide](cli.md)
- [CI and Release Qualification](cookbook/ci_release.md)
- [Benchmark Harness](benchmark_harness.md)
- [TelemetryEvent v4 Schema](telemetry_schema.md)
- [Run Envelopes](run_envelopes.md)
