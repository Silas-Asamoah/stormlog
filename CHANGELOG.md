# Changelog

All notable changes to Stormlog are documented in this file. The project was
published as `gpu-memory-profiler` (repository
`Silas-Asamoah/gpu-memory-profiler`) through 0.2.2; from 0.2.3 the package and
the TUI command are `stormlog`, and the repository is
`Silas-Asamoah/stormlog`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Release dates are the GitHub release publication dates.

## [Unreleased]

Inference workload control for
[#212](https://github.com/Silas-Asamoah/stormlog/issues/212)
([#248](https://github.com/Silas-Asamoah/stormlog/pull/248)), the exit-code
and report contract for
[#33](https://github.com/Silas-Asamoah/stormlog/issues/33)
([#249](https://github.com/Silas-Asamoah/stormlog/pull/249)),
`stormlog infer` adopting the same exit-code table
([#250](https://github.com/Silas-Asamoah/stormlog/pull/250)), and a fix for
the flaky benchmark memory gates
([#252](https://github.com/Silas-Asamoah/stormlog/pull/252)).

### Added

- `stormlog.scrub`, shared scrubbing primitives for what Stormlog records or
  sends elsewhere: `redact_url` (moved from `stormlog.infer.cache_state`,
  which still exports it) with a new `origin_only` mode; `KnownSecrets`,
  which redacts the exact credentials Stormlog was given in their raw,
  percent-encoded, JSON-escaped and base64 forms; `scrub_text` for free text
  an exporter has consent to send; `truncate_utf8`; and
  `is_forbidden_key_name`. Documented in `docs/scrubbing.md`; the wider
  artifact policy stays with #111. (#220)
- `stormlog infer analyze` gives each case:
  - a `latency` block: p50/p90/p95/p99 of each client latency metric, and of
    vLLM's own TTFT, end-to-end and queue time when spans are joined;
  - for each quantile, its value over the successful requests and with
    failures ranked worst, its sample count, whether the case has enough
    requests, and its order-statistic confidence interval. The second has
    no value, with a reason, when a successful request lacks the metric. The
    block's `rule` states what the intervals assume;
  - for each status other than `ok`, the count and the elapsed time observed
    before the request ended (for a cancelled request, its send to its
    cancellation), kept apart from the latency quantiles;
  - a `streaming` block of chunk-level figures, never called inter-token
    latency. (#213)
- `stormlog.infer.slo`: SLO policies whose criteria name their boundary
  (`client.ttft`, `server.ttft`, ...), so client and server latency are never
  mixed. Policies come from a versioned JSON file (`stormlog.infer.slo` v1),
  `KEY:MS` flags, or an artifact's `infer.slo` record.
  - `evaluate_request` judges an `infer.request` record as met, missed or
    unknown, with a reason per criterion.
  - `evaluate_span` judges a vLLM span on server criteria only, with success
    unverified, since vLLM emits spans for failed requests too.
  - A negative or non-finite value is unknown, never a pass, and the latency
    quantiles leave it out.
  - There is no client inter-token latency; `docs/inference_slo.md` explains
    why. (#213)
- `stormlog infer compare --baseline ... --candidate ...`: compares two arms
  of runs case by case (goodput, attainment, throughput, failure fraction and
  latency quantiles), paired by block when runs are labelled (a block whose
  runs sent different workload realizations is set aside). A run that did
  not finish, or a case a run lacks, is an outcome, compared rather than
  set aside, and a value it lost fails the candidate's gate
  (`outcome_unrecoverable`), as does an outcome a runner records in
  `infer.run_state`; only a protocol failure, such as an external cause a
  runner records there (listed with its evidence), sets aside a block, both
  arms, and a gate's pre-registered `min_complete_blocks` says how many may
  go. `--allow` names what may differ between the arms; a field that may
  differ between one arm's runs takes `--allow-within-arm`, and each such
  field is named with each run's value (`diagnostics.allowed_within_arm`). A baseline outcome
  failure leaves the case's contrasts
  `not_evaluable: control_failed`, so a broken baseline never passes the
  candidate. A retried block keeps its last attempt, but a retry never
  replaces an outcome failure. An SLO gate over runs judged by
  different policies is refused unless `--slo` judges them all. Gates are
  `--gate METRIC=RULE:BUDGET` with `non-inferiority`, `significant` or
  `demonstrated`, and a budget that can never fail is a usage error;
  `--min-attainment` judges the share of runs that meet a target, a run
  whose SLO was not judged counting as not meeting it;
  `--family any_regression` applies Holm. Modes `config`, `overhead`
  and `incremental` check what may differ and the observers' contract. Exit
  0, 4 (a gate failed or could not be evaluated, unless
  `--allow-not-evaluable`) or 5 (not comparable); `--format json` and
  `--report` give a `stormlog.report` v1 envelope (`inference_comparison`)
  whose findings point into the `stormlog.infer.comparison` v1 payload, also
  when the comparison exits 5. `docs/report_contract.md` now says how an
  evidence pointer resolves in a JSON file, a JSONL file, or the report
  itself. (#213)
- `stormlog infer profile --experiment ID --arm NAME --block LABEL`
  (with `--position` and `--attempt`) records the run's labels, and
  `stormlog.infer.run_summary.summarize_run` reads a run back for
  comparison: its report, labels, comparable fields, and protocol failures
  (unfinished session, identity change, incomplete probe, invalid cohort,
  unacknowledged cold-cache reset). (#213)
- `python -m examples.cli.infer_repeated_baseline --plan PLAN --output DIR`
  runs an experiment plan from the command line, with `--resume` and
  `--retry-incomplete`; it exits 3 when a run ended in a protocol
  failure. (#213)
- Launch-bound model identity for the runner (`server.model`): a
  `pinned_hub` snapshot is resolved to a commit and every blob checked
  against its name, then served with `--revision <commit>` and
  `HF_HUB_OFFLINE=1`; a `staged` directory is hashed into a read-only,
  content-addressed copy. Files are rechecked after each run
  (`model_changed`), and the runner's server descriptions carry
  `pinned_commit_verified` or `staged_snapshot_verified`, the only evidence
  that verifies a model's identity in a comparison. (#213)
- `stormlog.infer.experiment.run_plan`: the experiment runner. Each run
  starts the arm's server in a process group of its own, checks it holds
  only vLLM's processes, describes it before and after, runs the treatments
  and workload steps, stops the whole group and verifies nothing is left
  (through `/proc` on Linux, so a process that left with `setsid` is found),
  then checks artifacts and labels and writes `SHA256SUMS`. It probes each
  server before measuring, and a `/server_info` that does not answer
  (`probe_incomplete`) is retried once on a fresh server. Each run ends
  `completed`, `outcome_failure` (kept as data) or `protocol_failure` (set
  aside, retryable), recorded in `index.jsonl`; resumes refuse a changed
  plan or pre-registration, and secrets reach commands without being
  written down. `sanitize_bundle` scans the finished bundle for the plan's
  secret values and for bearer, `hf_` and `sk-` token shapes
  (`sanitizer.json`). The runner appends an `infer.treatments` record to
  each artifact, and observers and comparisons see each treatment as an
  observer, `treatment:<name>`. (#213)
- `stormlog.infer.experiment_plan`: experiment plans
  (`stormlog.infer.experiment_plan` v1) with arms, workload steps (shared with
  `same_as:<arm>`), treatments, block preludes, a seeded `random`, `williams`
  or `explicit` order, one seed per block shared by every arm, and command
  templates whose placeholders are checked when the plan loads. (#213)
- Segments: `infer compare --segment NAME=START:END` (and
  `analyze_inference_events(segments=...)`) compare a slice of each case's
  measured phase as a case of its own, `<case>/<segment>`, with its own
  population, rates, latency and SLO. A segment by overlap is
  length-biased in rates and quantiles alike, so it is compared as
  diagnostics and never gated, and has no rates. (#213)
- `examples/analysis/simulation_study.py` validates the comparison's rules
  with 20,000 replications per cell (results in
  `examples/analysis/simulation_results.json`, tables in
  `docs/inference_comparison.md`): non-inferiority's false-safe rate at a true
  change equal to the budget stays at or below 2.83% for normal and t3 noise,
  with the effect varying by block, for a higher-is-better metric and for the
  independent design; missing outcomes only make it more careful, the
  independent design covers at least 0.958, the run-level attainment gate
  holds at the supremum of a false claim and under correlated failures where
  pooled requests do not, and the skew screen flags 5% of normal noise. A
  run-gate cell the gate cannot pass at all is reported as one, with the
  fewest runs that could, never as a pass rate of 0. A fraction's floored
  paired t stays within 2.83% when failures are independent (1.3–1.9% at
  6–10 blocks of 300 requests). Strong right skew (8–12%), pooled requests,
  and a fraction's intervals under failures correlated within a run
  (beta-binomial: 14–25% for the floored t, 25–49% for pooled requests) are
  published as limits. (#213)
- `tests/fixtures/infer/comparison_contract_v1.json`: the comparison's
  units contract, 27 cases whose expected effects, intervals and gate
  outcomes are computed from the formulas and the gate rules by
  `examples/analysis/comparison_contract.py`, including the run-level claim
  on fractions (the attainment budget boundary, 6 of 6, 8 of 8, 7 of 8 and
  9 of 10, an all-zero failure fraction, too few requests a run), a
  baseline outcome failure (`control_failed`) and the independent design's
  gate. (#213)
- `stormlog.infer.comparison_stats`: one metric compared between a
  baseline and a candidate arm of runs. Paired t on block log ratios (or
  differences) when runs carry block labels, Welch t with df = min(nA, nB) − 1
  otherwise, and a one-sample t for `absolute` metrics; Fieller and a
  bootstrap (from 10 pairs) as unguarded cross-checks. Effects carry their
  scale and unit (`relative` for log ratios). Per-run values may be
  `(lower, upper)` bounds: non-inferiority gates use the worst case,
  `significant` and `demonstrated` the best. A fraction (failure fraction,
  attainment) is gated on a claim about runs, exact however failures
  cluster within a run: k of n candidate runs within the budget of their
  block's baseline, passing iff the one-sided 97.5% Clopper–Pearson lower
  bound of k/n is at least 0.5 (6 of 6, 8 of 8, 9 of 10), with `3 / b`
  requests a run, and only with blocks (`fraction_needs_blocks`) and a measured baseline
  (`baseline_unmeasured`), the mean
  change beside it (`mean_exceeds_budget`, and a warning in the text
  verdict, when that is beyond the budget); its paired t, with the standard error floored at the
  pooled binomial one, and the pooled requests' bounds are reported,
  labelled, never gated. Zero rules (`candidate_zero`, `undefined_in_arm`,
  pre-registered fallback budgets), degenerate constant metrics (never
  passed on a run-level bound alone), a skew screen against a pinned normal
  table, leave-one-out decision flips, at least 3 pairs per gate and
  pre-registered `min_complete_blocks`. Also `clopper_pearson` with its
  model, `run_pass_gate`, `blocks_for_precision` and `holm`. (#213)
- `stormlog infer analyze` gains an `observers` block: for the system
  sampler, metrics scraper, span receiver, profiler traces and execution
  hook, whether each was requested, configured, active and healthy, judged
  in every measured phase (enough samples, ok start and end scrapes without
  long gaps and a resolved window, spans for 99% of accepted requests,
  traces stopped and imported, no dropped hook records, and the hook's
  heartbeat covering the phase where the import records it). The hook
  counts as requested when the server's environment enables it. What the artifact
  cannot show is `null` with a reason. The `infer.session` config now
  records `sample_interval_seconds` and `trace`, and the system sampler
  keeps a fixed grid, so slow samples no longer lower its rate. (#213)
- `stormlog.infer.compatibility`: whether two runs measured the same
  thing. Each run's fields (from its descriptions, probe, workload,
  observers and declarations) keep their value, source and provenance;
  every field is `identity`, `launch`, `observation` or `label`, with vLLM
  0.30.0's configuration classified by JSON pointer (`config_classes_v1`)
  and an unclassified difference blocking. `compatible(a, b)` gives
  `incompatible`, `unverified` (a required field unknown on either side, or
  an identity field unknown on one; redacted, inferred and declared values
  never verify one) or `compatible`. A `null` setting is a value, and a
  field only one run has, where both read its source, is a difference. The
  weights are verified only by the experiment runner's
  `infer.model_identity` record, bound to the server it launched, never by
  a description alone. The `infer.workload` record gains `spec_digest`,
  its digest without the seed. (#213)
- The run manifest: `infer profile --describe-server FILE` records a
  description taken before the run, refusing (exit 5) one that names another
  run, and `--declare FILE` the operator's declarations, as append-only
  `infer.manifest` records. `stormlog infer attach-manifest ARTIFACT FILE`
  appends the description taken after the run and refuses (exit 5) one from
  another run, host or boot, a restarted server, the before description
  itself, or one taken before the last measured phase ended or sooner after
  the before one than the run lasted. The report gains a `manifest` block:
  identity changes between before and after are a `protocol_failure:
  identity_changed`, a before description that disagrees with the probed
  server's model, vLLM version, driver or listening port is a
  `description_mismatch`, settings
  only one side could read are `identity_unverified`, and GPU clock and
  temperature changes are drift. `describe-server --run-id` ties a description
  to a run. (#213)
- `stormlog infer profile --server-probe {auto,basic,none}` (default
  `auto`) asks the server about itself before the first case and after the
  last: `/version`, `/v1/models` and, on a loopback or private host (or
  with `--allow-remote-probe`), vLLM's dev-mode `/server_info`, kept
  redacted as `infer.server_probe` records. Answers are capped at 4 MiB with
  no redirects, and each deadline bounds the whole exchange.
  `/server_info` gets one 120 s deadline and no retry; if it does not
  answer, or the server drops the request unanswered, the profile exits 5
  before measuring, since vLLM's environment collector may still be
  running. (#213)
- `stormlog infer describe-server --pid PID --output FILE`: one
  description of a running vLLM server from the host that serves it
  (`stormlog.infer.server_description` v1): its process tree from `/proc`,
  its GPUs from NVML, its model files, its start-up log (`--server-log`),
  its Python and package versions (`--python`, by default the server's own
  interpreter, found on the server's `PATH`), and the kept environment
  (vLLM's, NCCL's, CUDA's and PyTorch's settings, and those that change
  performance from outside them, such as `LD_PRELOAD`, `OMP_NUM_THREADS`
  and `TORCHINDUCTOR_`, `TRITON_` and `CUBLAS_`), with no credentials and a
  SHA-256 of its own content. (#213)
- `stormlog.infer.server_log`: the choices a vLLM 0.30.0 server logs at
  start-up (attention backend, KV cache size, CUDA graph captures), from the
  last start-up in its log, with disagreements between workers kept as
  issues. (#213)
- `stormlog.infer.server_model`: the model files a vLLM server was started
  with, from its command line and its Hugging Face cache (each file's
  algorithm, digest and size: SHA-256 for LFS blobs, git SHA-1 for the
  rest, named by the blob the snapshot links to even when that blob links
  on into a shared store, and none for a copied file a blob name does not
  vouch for) or, for
  a local directory, cached SHA-256 digests. It names its
  evidence (`pinned_commit`, `inferred`, `post_launch_digest`, `size_only`)
  and never claims the weights were the ones loaded. (#213)
- `stormlog.infer.server_gpu`: the GPUs a vLLM server uses, found by NVML's
  compute processes rather than an index, with each device's settings
  (power limits, application clocks, persistence, ECC, MIG, compute mode)
  and a reading of what drifts (SM clock, temperature, clock event
  reasons). An unreadable field records NVML's code. (#213)
- `stormlog.infer.server_process`: a vLLM server's processes from Linux
  `/proc`, each identified by PID and start time and given a role from vLLM
  0.30.0's process titles (`api_server`, `engine_core`, `worker`, helpers,
  `other`), with group, session and CPU affinity. It lists what is left of
  a group or session, and which remembered processes still run, including
  one that left its session. (#213)
- `stormlog.infer.server_privacy`: schema-aware redaction for server
  descriptions. Credential fields of vLLM 0.30.0's configuration
  (`credential_paths_v1`) and environment names with a secret word are
  replaced by a `{"redacted": true, "path": ...}` marker, URLs keep only
  their scheme, host and port and a short digest of their path (a token can
  sit in a path), and `system_env` keeps only allowlisted scalars and four
  runtime package versions. The credential fields include the free-form
  plugin configs, the model's `hf_overrides` and
  `override_generation_config`, and Ray's runtime environment, and
  free-form `OTEL_` settings such as resource attributes are removed.
  `max_num_batched_tokens`, `long_prefill_token_threshold` and vLLM's
  `*_TOKEN_THRESHOLD` knobs survive. (#213)
- `--slo KEY:MS` (repeatable) and `--slo-file FILE` on `stormlog infer
  profile` and `stormlog infer analyze`. `profile` records the policy in the
  artifact as `infer.slo`; `analyze` judges by the flags, or else by the
  policy the artifact recorded. The report gains a top-level `slo` block
  (name, digest, source, policy; the digest does not depend on the order of
  the criteria) and, per case, SLO attainment and SLO goodput at the offered
  load as lower and upper bounds with evidence coverage, `null` with a reason
  when the policy cannot be judged, and whether the case's cohort is valid. A
  malformed flag, or both options at once, exits 2; a missing or invalid
  policy file, including one with a key given twice or a number too large for
  a float, exits 5, as does a policy with a sliding interval, which an online
  watcher judges. `profile` warns before sending when a criterion cannot be
  judged per request in the run. When `analyze` options replace the policy
  the artifact recorded, `slo.overrides` keeps its name and digest, and a
  warning says so. (#213)
- `stormlog infer watch --root DIR --base-url URL` runs beside a vLLM server
  and records its incidents. Once per tick it scrapes `/metrics` into a
  bounded history and evaluates metric, #218 signal and scrape-health
  triggers; a trigger fires only once its condition has held for the hold
  time, with data gaps and outages handled as documented, so a violation
  shorter than `F - W - Δ` never fires. A firing opens an incident whose pre-
  and post-windows of scrapes are sealed into a bundle under
  `DIR/incidents`, published in generations under disk limits and
  retention. Its records follow the frozen `stormlog.infer.watch/1` schema
  (`tests/fixtures/watch/`) and go to a bounded ledger, and its own health
  is kept as `stormlog_watch_*` counters. It writes a
  `stormlog.report` with `report_kind: inference_watch` and exits 3 when a
  counting incident was detected, 1 when the watch was unsound, else 0
  (`docs/incident_capture.md`). One watcher owns a root at a time; a second
  exits 2. A health trigger may read only the scrapes the history holds:
  the default failed-scrape share reads its last 60, so with the default
  600 s history a `tick_seconds` over 10 s exits 2 until `history.seconds`
  is raised. Deep capture and SLO triggers come later.
  ([#219](https://github.com/Silas-Asamoah/stormlog/issues/219))
- `stormlog infer import-execution ARTIFACT DIR` reduces the vLLM execution
  hook's raw log (`docs/vllm_execution.md`) into `infer.iteration`,
  `infer.membership`, `infer.request` and `infer.clock_alignment` records:
  only final steps, each written once, with requests bound to the run by the
  recorded `X-Request-Id` and other clients' requests kept under keyed
  pseudonyms. `stormlog infer profile --vllm-execution-dir DIR` imports the
  log when the run ends, before the report, and now imports traces before
  the report too; `import-trace --vllm-execution-dir DIR` takes each
  trace's GPU UUID from the hook's worker hellos. `infer analyze` gains
  `telemetry.execution`, a coverage block of per-device unions: linkage,
  membership, ownership, measurement and capture loss, with non-additive
  case figures labelled. (#217)
- vLLM native telemetry for `stormlog infer profile`. `--vllm-metrics [URL]`
  scrapes vLLM's Prometheus metrics just before each phase's first send,
  after its drain and every `--vllm-metrics-interval` seconds between, as
  one `infer.vllm_scrape` record per scrape that keeps every series under
  its native name with native histogram boundaries. `--vllm-spans-listen
  [HOST:PORT]` runs an OTLP/HTTP receiver for the run and keeps each request
  span as an `infer.vllm_span` record (protobuf exports need the new
  `infer-otlp` extra, `opentelemetry-proto>=1.20`, and a server started with
  `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf`, since vLLM exports
  over gRPC by default); `infer analyze --vllm-spans FILE` loads spans
  collected elsewhere. Every request now sends
  `X-Request-Id: stormlog-<run_id>-<request_id>`, recorded as
  `x_request_id`, so spans join to requests by the recorded value. The
  report gains `telemetry.vllm`: per case and per engine label, counter and
  histogram deltas, gauge summaries, token rates, prefix-cache hit ratio,
  logical KV occupancy and the MFU estimates, with resets, restarts, missing
  and retired series left unresolved with a reason instead of zero, plus a
  capability record naming what vLLM 0.30.0 exposes and that
  `--collect-detailed-traces` never fills the forward and execute fields.
  ([#263](https://github.com/Silas-Asamoah/stormlog/pull/263))
- Open-loop arrivals for `stormlog infer profile`. `--arrival fixed-rate`,
  `poisson`, `burst` or `replay` sends requests on a seeded schedule that is
  fixed before the run starts (`--rate`, `--burst-size`, `--burst-interval`,
  `--arrival-trace`), bounded by `--max-in-flight` (default 128). When every
  slot is busy, `--overflow wait` (the default) holds the arrival and
  `--overflow drop` records it as `dropped` without sending it. `closed`
  stays the default and takes `--concurrency` as before; the open-loop modes
  reject it. ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- Latency measured from the intended arrival. Every request records
  `intended_at_ns`, `dispatch_lag_ms`, `held_for_slot` and
  `in_flight_at_dispatch`. Each case's report gains an `arrivals` block
  (offered, sent, completed, dropped and held counts, failures by status,
  peak in-flight, offered rate and dispatch-lag percentiles) and
  `latency_ms.e2e_from_intended_*` percentiles that include time spent
  waiting for a slot; the text report prints that p95 for every open-loop
  case. ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- `--prompt-mode repeat`, `unique` or `shared-prefix` (with `--prefix-groups`
  and `--shared-prefix-ratio`) controls how much prompt text requests share,
  so prefix-cache hits are chosen on purpose. Nonces derive from the seed,
  case and phase, so a run repeats exactly. Requests record `prompt_mode`,
  `prompt_id`, `prefix_group`, `shared_prefix_tokens` and `prompt_digest`;
  each phase window records a `prompts_digest`.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- `--cache-state cold` and `--cache-reset-url`, which is POSTed before each
  case with the API key when one is set. Each case writes an
  `infer.cache_state` record with the requested state, the reset's outcome,
  a verification status (always `unverified` for now, with the reason) and a
  `run_kind` of `cold_start`, `steady_state` or `unspecified`.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- Phase windows. Each phase has a window in which requests arrive and a
  drain after it, bounded by `--drain-timeout` (default: `--timeout`) from
  the window's end. At the deadline, a request still running is recorded as
  `cancelled`, and an open-loop arrival still waiting for a slot is recorded
  as `dropped`. Every phase writes an `infer.phase_window` record, and the
  report adds `window_seconds` and `drain_seconds` per case.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- Request failure statuses `timeout` (client timeout), `rejected` (HTTP 429
  or 503), `dropped` and `cancelled`, with `http_status` on every failure
  that has one. Previously every failure was recorded as `error`.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- An `infer.workload` record at the start of every run holding the seed,
  prompt generator version, cases and arrival shapes, measurement and
  warmup settings, decoding settings, requested cache state and tokenizer
  identity, plus a `workload_digest` that leaves out the endpoint, model,
  timeouts, reset URL and where a replay trace came from, so the same
  workload sent to two engines shares a digest. The API key is never recorded, and the reset URL is stored
  without credentials or query string.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- `--extra-body`, a JSON object merged into every request (for example
  `{"temperature": 0, "ignore_eos": true}`), recorded with the decoding
  settings. It cannot replace the fields Stormlog sets itself.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- Run-time warnings from `stormlog infer profile` when a cold cache is
  requested without a reset URL, when a reset fails, when deterministic
  prompts are sent to a server nothing resets, and when `--input-tokens` is
  below 32 with `unique` or `shared-prefix` prompts.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- A shared exit-code table in `stormlog.exit_codes.ExitCode`: `OK` (0),
  `ERROR` (1), `USAGE` (2), `FINDINGS` (3), `GATE_FAILED` (4),
  `INVALID_INPUT` (5) and `INTERRUPTED` (130), each paired with one verdict
  status. ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- The `stormlog.report` v1 envelope, published as
  `docs/schemas/stormlog_report_v1.schema.json`, with `build_report()`,
  `write_report()`, `load_report()` and `validate_report()`. It carries the
  verdict paired with the exit code, findings with evidence pointers, flat
  metrics, artifact pointers, recommendations and a tool-specific payload.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- `report.json` in every `gpumemprof`, `tfmemprof` and `jaxmemprof diagnose`
  bundle, next to `manifest.json`: one finding per raised risk flag with its
  severity, observed value and threshold, evidence pointers into
  `diagnostic_summary.json`, and the summary's suggestions as
  recommendations. A bundle the command could not finish gets an `error`/1
  report instead of a stale verdict.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- `docs/report_contract.md`, documenting the exit-code table, where each
  command stands, and the report envelope.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))

### Changed

- **Breaking:** `stormlog infer analyze` JSON is `analysis_version: 2`.
  - Each case's throughput divides by its rate interval. For an open loop
    that is the schedule's own window; for a closed loop, the phase start to
    the drain end. It used to be the span of the case's successful requests,
    which shrank when the last requests failed.
  - `throughput.duration_seconds` is replaced by `interval_seconds`,
    `interval_kind` and `numerator_cohort`. The rate keys keep their names
    with the new denominator, so a consumer must check `analysis_version`
    before reading any rate:

    | Version 1 | Version 2 |
    | --- | --- |
    | `throughput.duration_seconds`: first successful start to last successful end | `interval_seconds`, `interval_kind`, `numerator_cohort` |
    | `requests_per_second`, `output_tokens_per_second`, `total_tokens_per_second` over `duration_seconds` | The same keys over `interval_seconds` |
    | A rate over an empty span: `0.0` | `null`, with `intervals.rate_reason` |
  - A rate over an empty interval is `null`, not `0.0`, and so is every rate
    of an open loop with no known endpoint, such as a replay without
    `--duration` (`rate_reason: endpoint_undeclared`).
  - Each case gains a `population` block (offered, sent, accepted,
    successful, failed, timed out, cancelled and the rest, with cohort checks)
    and an `intervals` block. A case whose phase was cut short, such as by
    Ctrl+C, has an invalid cohort (`phase_window_missing`, with its missing
    scheduled arrivals) and no rates, and `summary.session_status` says how
    the run ended. (#213)
- `stormlog infer profile` tells apart where a failed request stopped. Two
  new request statuses:
  - `unreachable`: the connection failed before any byte was sent. A connect
    timeout used to be `timeout`, and a refused connection `error`.
  - `delivery_unknown`: sending failed after the connection completed, or
    the connection closed before the response's status line, so the server
    may have received the request. A reset after the status line is the
    answer the server gave: an `error` (or `rejected`) with its status.

  Inference requests and cache resets no longer follow HTTP redirects.
  urllib re-sent a redirected POST as a GET to another address, so a 3xx is
  now recorded as `error` with its status. They also ignore proxies set in
  the environment, through which an unreachable server read as the proxy's
  HTTP 502, and the session config records that they did
  (`environment_proxies`). (#213)
- **Breaking:** `gpumemprof`, `tfmemprof` and `jaxmemprof diagnose` exit 3
  for memory risk. They used to exit 2, which could not be told apart from
  an `argparse` usage error from the same command. The bundle manifest's
  `exit_code` field follows.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- **Breaking:** `gpumemprof`, `tfmemprof` and `jaxmemprof analyze` exit 5
  for a missing or unusable input. They used to exit 1.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- **Breaking:** invalid `diagnose` options (`--duration`, `--interval`,
  `--native-history` off CUDA) and a missing framework runtime or optional
  extra exit 2. They used to exit 1. In particular, `gpumemprof` exits 2
  instead of 1 when PyTorch is not installed and a command needs it.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- **Breaking:** all three `diagnose` commands exit 2 when `--output` is, or
  sits under, an existing file. `gpumemprof` and `tfmemprof` used to exit 1.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- **Breaking:** `gpumemprof` exits 130 when interrupted outside a
  `monitor`/`track` capture loop. It used to exit 0. `tfmemprof` and
  `jaxmemprof` print "Operation cancelled by user" instead of a traceback
  (their code was already 130). Ctrl+C inside a capture loop still
  finalizes the artifact and exits 0.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- **Breaking:** `examples.cli.benchmark_harness --check` exits 4 for a
  failed gate and 5 for an unusable budget, baseline or tolerance asset.
  Both used to exit 1. An unwritable `--artifact-root` or `--output` exits 1
  with a message instead of a traceback, and asking for regression defaults
  outside the `pr` profile is a usage error (2) instead of a `ValueError`
  traceback. CI only checks for a non-zero code, so the workflow is
  unchanged. ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- **Breaking:** the W&B export of a diagnose bundle logs the manifest's
  `exit_code` as the `stormlog_exit_code` metric, so dashboards keyed on 2
  for memory risk now see 3.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- **Breaking:** `stormlog infer` follows the same table. Each of these
  used to exit 1:
  - `infer profile` exits 3 when no measured request succeeds. It exits 2
    for a setting it cannot use, checked before anything is sent, or a
    requested tokenizer that is not installed, and 5 for an
    `--arrival-trace` it cannot read.
  - `infer analyze` exits 5 for a missing or unreadable artifact or
    `--server-telemetry` file, including an artifact with no inference
    records. It still exits 0 when every request in the artifact failed.
  - `infer collect-server` exits 3 when the GPU identity changes mid-run. It
    exits 2 for options it cannot use, a `--pid` with no running process, a
    GPU the host does not have, or a host without NVML.
  ([#250](https://github.com/Silas-Asamoah/stormlog/pull/250))
- A `stormlog infer profile` run stopped by Ctrl+C exits 130, records the
  requests still in flight as `cancelled`, and ends the artifact with an
  `infer.session` record whose status is `interrupted`; a run that fails for
  another reason ends with `incomplete`. Previously the artifact kept its
  `running` session record and Ctrl+C printed a raw asyncio traceback.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- `stormlog infer analyze` lists every case, including cases in which no
  request succeeded, so drops and failures stay visible.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- The vLLM execution hook takes about 60–65% less time on vLLM's engine thread
  per scheduler step. Each record's fields are now serialized once, when the
  record is queued. Before, they were walked in Python to bound their size,
  then serialized again by the writer thread. On an A30 serving
  Qwen2.5-0.5B with vLLM 0.30.0, the hook now costs 1.3% of throughput at
  concurrency 32 (it was 9.7%) and 7.7% at concurrency 256 (it was 12.0%).
  The records written are byte-identical. The queue counts each record at
  its exact JSON size, so:
  - the memory it holds is now its 32 MiB of JSON at most, plus about 113
    bytes a record: about 34 MiB in all at the 20,000-record cap. The live
    records it held before could take twice the 32 MiB;
  - fewer records are dropped as oversized: only a record whose JSON is over
    4 MiB is, where the old estimate, about 1.65 times the JSON, also
    dropped some that fit.

  A record the queue cannot take at any size, because it is full or the disk
  cap has stopped record writing, is now dropped before it is serialized, and
  so is a record whose request IDs, which clients choose, alone pass 4 MiB.
  (#217)

### Fixed

- `stormlog infer analyze` no longer keeps the first of two different
  deliveries of a request's vLLM span. That request, and any request with more
  than one span, is quarantined: its spans are left out of the case's span
  statistics and counted under `quarantined_requests`. (#213)
- `stormlog infer profile --cache-reset-url` no longer counts an HTTP 200 as a
  reset. vLLM's `/reset_prefix_cache` answers 200 with `{"success": false}`
  while blocks are still held. The answer is now read and recorded as
  `acknowledged`, `refused` or `accepted_unverified`. A refused reset is
  retried for up to `--cache-reset-timeout` seconds (default 10), then
  counts as a failed reset, so the case is not labelled a cold start. A
  `success` field with any value other than `true` is a refusal too, and no
  retry starts after the timeout. `infer.cache_state` records and the
  report's `cache` block gain `attempted` and `acknowledged`, and the reset's
  `success`, `answer`, `attempts` and `answered_at_ns`, when the recorded
  answer came back. (#213)
- The benchmark harness's memory gates no longer fail on runner noise:
  - The soak's RSS checks (`max_rss_delta_bytes`, `rss_growth_per_24h_equiv`)
    now read memory inside the sample loop, after a warmup. Before, they
    read it after the session closed, when building rollups left an
    allocator-dependent residue of 40–70 MB. That residue is now reported
    as `finalization_rss_delta_bytes` but not gated.
  - A new `--overhead-scratch-root` runs the overhead trials on RAM-backed
    storage. CI passes `/dev/shm`, so runner disk latency no longer
    inflates `runtime_overhead_pct`.

  Soak RSS values from earlier reports were measured after finalization,
  so they are not comparable.
  ([#252](https://github.com/Silas-Asamoah/stormlog/pull/252))

- `stormlog infer profile` rejects a case matrix in which two cases would
  share one ID, for example `--concurrency 1,1` or a repeated
  `--input-tokens` value. Such cases were previously reported as one.
  ([#248](https://github.com/Silas-Asamoah/stormlog/pull/248))
- `tfmemprof analyze` and `jaxmemprof analyze` report an unparsable or
  non-object JSON input with a message instead of a traceback.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- The benchmark harness validates its budget, baseline and tolerance files
  before any scenario runs, so a typo in `--budgets` no longer costs a full
  run, and a budgets file that is valid JSON but not an object no longer
  crashes with an `AttributeError` after the run.
  ([#249](https://github.com/Silas-Asamoah/stormlog/pull/249))
- `stormlog infer profile --trace vllm-torch` now stops the profiler after a
  `/start_profile` that answered 5xx, timed out, or lost its reply. vLLM runs
  the start before it replies, so such a server could be left profiling, its
  memory growing, until the next stop. Each `infer.trace_window` record now
  carries `start_outcome` (`acknowledged`, `rejected` or `unknown`) and the
  times the start request was sent and answered. Only a 401, 403, 404, 405
  or 407, which come before vLLM's handler runs, is left unstopped. A stop that fails is warned about
  and recorded, not retried, and the record says the profiler may still be
  running.
  ([#219](https://github.com/Silas-Asamoah/stormlog/issues/219))
- `stormlog infer profile`'s vLLM collectors bound what an endpoint can make
  the client hold. A `/metrics` response is read at most 8 MiB far and
  parsed one line at a time, refused at a line over 64 Ki characters or
  past 20,000 series, so a scrape holds at most about twenty-five times
  what it read. The OTLP span receiver admits at most 8 connections and
  10 s for each whole request, headers and a protobuf body's wire scan
  included; charges the exports in flight at most 128 MiB, each decoding
  step from a measured estimate before it runs (a protobuf parse per
  message, counted on the wire first); refuses a body of more than 10,000
  spans; and queues at most 100,000 spans and 64 MiB of what they hold.
  Before, the scrape read any response whole, and the receiver accepted
  any number of connections into an unbounded queue. Refusals are counted
  in the capability records (`docs/vllm_telemetry.md`, "Ingestion
  limits").
  ([#219](https://github.com/Silas-Asamoah/stormlog/issues/219))
- The append-only telemetry sink cuts a segment back to where it stood when
  a write fails partway, for example on a full disk, and, if that cut-back
  fails too, before the segment is written again. Before, the next
  successful flush extended the partial line into a corrupt record. A new
  opt-in bound, `TelemetrySinkConfig(max_buffer_bytes=...)`, keeps a sink on
  a failing disk from growing memory: a full buffer is flushed first, and
  records that still do not fit are dropped and counted; failed flushes,
  manifest writes and segment deletions are counted and retried with backoff
  instead of raising. `failure_diagnostics()` reports the counters.
  ([#219](https://github.com/Silas-Asamoah/stormlog/issues/219))

## [0.3.10] - 2026-10-01

### Added

- Capability-aware PyTorch device memory sampling. `DeviceMemoryCapabilities`
  declares per-counter and allocator-feature support, `DeviceMemorySample`
  counters are nullable, and `MemoryTracker(..., collector=...)` accepts an
  injected collector for device-only runtimes. Device-only tracking emits
  canonical samples, device usage timelines and one capability warning, and
  never fabricates allocator, fragmentation, history or attribution
  findings. CUDA, ROCm and MPS sampling is unchanged.
  ([#229](https://github.com/Silas-Asamoah/stormlog/pull/229))
- Canonical `TelemetryEventV4` exports with nullable memory counters,
  complete capability metadata under `metadata.memory_capabilities`, a
  published v4 JSON Schema, and conservative v2/v3/legacy upgrades; existing
  artifacts continue to load.
  ([#229](https://github.com/Silas-Asamoah/stormlog/pull/229))
- Explicit device-only CLI/TUI diagnostics: device-used memory is plotted
  when allocator memory is unavailable, allocator fields show as `N/A`, and
  unsupported fragmentation, gap and attribution analyses report a reason
  instead of a finding.
  ([#229](https://github.com/Silas-Asamoah/stormlog/pull/229))
- Execution correlation contracts for inference artifacts. A profile can
  relate a request to shared server iterations and GPU activities through
  membership records, with iteration elapsed time, summed activity duration
  and merged GPU interval time reported separately; new profiles carry a v2
  run identity alongside their v1 client records. No backend-specific
  collector is included yet.
  ([#233](https://github.com/Silas-Asamoah/stormlog/pull/233))
- Server telemetry for inference profiles. Endpoint-only reports now label
  their memory scope `client_local`. The new `stormlog infer collect-server`
  command records server process RSS (psutil) and whole-device or MIG memory
  (NVML) on the serving host, and `stormlog infer analyze` joins those
  samples to case windows when the run ID, route, server identity and clocks
  match, reporting per-case `memory.server_coverage` and a specific
  `unjoined` reason otherwise. A process exit, replacement or GPU change
  invalidates only the windows after it.
  ([#240](https://github.com/Silas-Asamoah/stormlog/pull/240))
- `infer.clock_alignment` records. Clock domains are named
  `{host}/{boot_id}/unix_epoch_ns`; samples from other hosts are placed on
  the client clock through alignment records in the artifact or through
  `--clock-offset-ns` and `--clock-uncertainty-ns`, and the report lists
  every alignment it applied.
  ([#243](https://github.com/Silas-Asamoah/stormlog/pull/243))
- Server groups for tensor-parallel servers. Each collector declares
  `--group-id`, `--rank` and `--world-size`; a group joins only when every
  rank appears exactly once, and the report lists values per member under
  `memory.server_members` rather than summing them.
  ([#244](https://github.com/Silas-Asamoah/stormlog/pull/244))

### Fixed

- The published `inference_telemetry_v1` and `telemetry_event_v4` JSON
  Schemas now reject what the loader rejects: group members without
  `world_size`, malformed clock-domain segments, counters that are declared
  unsupported but non-null, and whitespace-only identifiers. Appending
  capture records to an artifact whose last line has no trailing newline no
  longer corrupts the file, and capture rewrites keep the file's permissions
  instead of narrowing them to owner-only.
  ([#246](https://github.com/Silas-Asamoah/stormlog/pull/246))

### Maintenance

- Every function in `stormlog/` now meets a cyclomatic-complexity budget of
  10, enforced in CI ([#230](https://github.com/Silas-Asamoah/stormlog/pull/230)).
  Dependabot: docs protobuf >=7.36.1
  ([#231](https://github.com/Silas-Asamoah/stormlog/pull/231)),
  codecov-action 7.1.1 ([#238](https://github.com/Silas-Asamoah/stormlog/pull/238)),
  gh-action-pypi-publish 1.14.2 ([#205](https://github.com/Silas-Asamoah/stormlog/pull/205)),
  actions/checkout 7.0.1 ([#203](https://github.com/Silas-Asamoah/stormlog/pull/203)).

## [0.3.9] - 2026-09-01

### Added

- MLflow export, mirroring the Weights & Biases exporter: `--mlflow`,
  `--mlflow-tracking-uri`, `--mlflow-experiment`, `--mlflow-run-id`,
  `--mlflow-run-name`, `--mlflow-group`, `--mlflow-job-type`,
  `--mlflow-log-artifacts` and `--mlflow-log-attribution` on the
  `gpumemprof`, `tfmemprof` and `jaxmemprof` `track` and `diagnose`
  commands, plus the `stormlog.mlflow_integration` API. Session metrics and
  tags, alert and timeline tables, dashboard HTML, timeline plots, the
  attribution preview and output artifacts are logged to the run. Requires
  the new `stormlog[mlflow]` extra (`mlflow>=2.10.0`), included in
  `stormlog[all]`.
  ([#206](https://github.com/Silas-Asamoah/stormlog/pull/206),
  [#207](https://github.com/Silas-Asamoah/stormlog/pull/207))

## [0.3.8] - 2026-07-16

### Added

- Run envelopes and attachment catalogs: a `stormlog_run.json` v1 schema for
  explicit runs; `run_id`, `storage`, `source_namespace` and `source_ref`
  fields in `stormlog_attachments.json`; implicit runs synthesized from
  sessions that share a `job_id`; and `stormlog query runs` and
  `stormlog query attachments` with table, JSON and CSV output.
  ([#196](https://github.com/Silas-Asamoah/stormlog/pull/196))
- JAX dataset replay: `profile_training` accepts zero-argument dataset
  factories, caps one-shot iterator materialization with `steps_per_epoch`,
  and raises a clear error when a later epoch is empty.
  ([#187](https://github.com/Silas-Asamoah/stormlog/pull/187))
- `jaxmemprof analyze` gains leak detection, optimization, visualization and
  report options; JAX heatmaps, dashboards, CSV/JSON exports and bulk plot
  generation; named device selectors; Apple Silicon/Metal detection.
  ([#197](https://github.com/Silas-Asamoah/stormlog/pull/197))

### Changed

- **Breaking:** PyTorch tensor tracking is now disabled by default to avoid
  full-GC scans during profiling.
  Callers that need tensor counts must now pass `track_tensors=True`.
  ([#198](https://github.com/Silas-Asamoah/stormlog/pull/198))
- JAX device-memory statistics the runtime cannot provide are marked
  unavailable instead of reported as zero, and no longer produce samples,
  alerts, analysis or plots; process RSS stays available separately.
  ([#197](https://github.com/Silas-Asamoah/stormlog/pull/197))
- Continuous monitoring has a configurable retention bound for snapshots,
  and `stop_tracking()` waits for the tracking worker to finish.
  ([#198](https://github.com/Silas-Asamoah/stormlog/pull/198))
- `stormlog[all]` requires TensorFlow 2.21 or newer so it can coexist with
  protobuf 6.31. ([#198](https://github.com/Silas-Asamoah/stormlog/pull/198))

### Fixed

- `MemoryTracker.get_events()`, `get_memory_timeline()`, `get_statistics()`
  and `get_alerts()` no longer raise `deque mutated during iteration` while
  the sampling thread appends events.
  ([#195](https://github.com/Silas-Asamoah/stormlog/pull/195))
- Analyzer and visualizer edge cases: zero-duration samples no longer crash
  performance reports, a leak is reported only for consistently positive
  growth, CSV export returns a path that exists, mixed monitoring and
  profiling timelines are sorted before plotting, tracker timelines
  aggregate in one pass and honor CPU intervals, and negative byte values
  use the normal unit ladder.
  ([#198](https://github.com/Silas-Asamoah/stormlog/pull/198))
- `stormlog.jax` imports safely when JAX or protobuf is missing, and
  `JAXMemoryProfiler` raises a clear `ImportError` instead of failing
  mid-initialization.
  ([#186](https://github.com/Silas-Asamoah/stormlog/pull/186))
- The optional JAX pprof schema import is deferred, so an older protobuf
  runtime no longer aborts collection of the whole test suite.
  ([#198](https://github.com/Silas-Asamoah/stormlog/pull/198))

### Maintenance

- JAX CI hardening and shared JAX test helpers
  ([#186](https://github.com/Silas-Asamoah/stormlog/pull/186)). Dependabot:
  black 26.5.1 ([#171](https://github.com/Silas-Asamoah/stormlog/pull/171)),
  codecov-action 7.0.0 ([#185](https://github.com/Silas-Asamoah/stormlog/pull/185)),
  actions/checkout 7.0.0 ([#188](https://github.com/Silas-Asamoah/stormlog/pull/188)),
  docs protobuf >=7.35.1 ([#189](https://github.com/Silas-Asamoah/stormlog/pull/189)).

## [0.3.7] - 2026-06-22

### Added

- `rollups.json` v1 sidecars for append-only telemetry sinks, computed on
  sink close and recovery, so long monitor sessions can be summarized
  without reading every event. The query layer answers exact built-in
  summaries from a fresh sidecar and falls back to raw events otherwise.
  ([#176](https://github.com/Silas-Asamoah/stormlog/pull/176))
- `stormlog query correlate` and `QueryStore.correlate(...)`, which gather
  telemetry events, timeline markers, alerts, OOM bundles, diagnose bundles,
  rollup windows and attachments around a timestamp (`--at-ns`) or a
  telemetry `--record-id`, each with a confidence reason. External evidence
  such as profiler traces and runbooks is discovered through
  `stormlog_attachments.json` sidecars.
  ([#182](https://github.com/Silas-Asamoah/stormlog/pull/182))

### Changed

- JAX `memory_growth_rate` reports MB/second to match the analyzer's
  thresholds; it previously reported bytes/second.
  ([#181](https://github.com/Silas-Asamoah/stormlog/pull/181))

### Fixed

- JAX pprof visualization assigns flat memory to the leaf frame,
  single-epoch generator datasets are no longer materialized during training
  profiling, `jaxmemprof monitor` honors `--interval`, and the JAX CLI
  module can be patched when JAX is not installed.
  ([#181](https://github.com/Silas-Asamoah/stormlog/pull/181))
- Malformed `rollups.json` sidecars are rejected rather than trusted, rollup
  write failures include their traceback, and `jaxmemprof monitor` handles a
  zero duration.

## [0.3.6] - 2026-06-13

### Added

- Inference profiling for OpenAI-compatible Chat Completions endpoints:
  `stormlog infer profile` and `stormlog infer analyze` measure latency,
  throughput, memory and token accounting, recording request events, session
  summaries, optional system samples, tokenizer provenance and the analysis
  in a JSONL artifact. The top-level `stormlog` command now dispatches to
  the TUI (default), `tui`, `query` and `infer`, and `python -m stormlog`
  works. Tokenizer packages come with the new `stormlog[infer-tokenizers]`
  extra, also included in `all`.
  ([#177](https://github.com/Silas-Asamoah/stormlog/pull/177),
  [#180](https://github.com/Silas-Asamoah/stormlog/pull/180))
- Durable issue fingerprinting finalized: grouped issue rows carry hit
  counts, first and last seen timestamps, affected sessions, representative
  evidence and a state (`open`, `resolved`, `ignored` or `regressed`) with
  explicit state constants.
  ([#175](https://github.com/Silas-Asamoah/stormlog/pull/175))

## [0.3.5] - 2026-06-04

### Added

- JAX support. The `stormlog.jax` package provides `MemoryTracker`,
  `JAXMemoryProfiler`, `MemoryAnalyzer`, `profile_function` and
  `profile_context`, `run_diagnose`, `MemoryVisualizer`, a pprof parser and
  an attributed HTML call-graph view, and the `jaxmemprof` CLI offers
  `info`, `track`, `monitor`, `diagnose` and `analyze`. Device memory is
  read through `jax.Device.memory_stats()` after an XLA sync, and the OOM
  flight recorder attaches `.prof` artifacts to its bundles.
  ([#169](https://github.com/Silas-Asamoah/stormlog/pull/169),
  [#174](https://github.com/Silas-Asamoah/stormlog/pull/174))
- A local query layer. `stormlog.query.open(paths)` returns a `QueryStore`
  with `list_sessions`, `query_events`, `list_oom_bundles` and `summarize`,
  backed by manifest-first discovery of sink directories, flat telemetry
  files, diagnose bundles and OOM bundles, with no database. The
  `stormlog query sessions`, `events`, `ooms` and `summary` commands expose
  it with `--table`, `--json` and `--csv` output and filters; `stormlog`
  with no arguments still launches the TUI.
  ([#164](https://github.com/Silas-Asamoah/stormlog/pull/164))
- Recurring issue grouping: `stormlog.issues` (`IssueFingerprint`,
  `IssueEvidenceLink`, `StormlogIssue`), `QueryStore.list_issues()` and
  `stormlog query issues` group OOMs, collector degradation, alerts and
  hidden-memory anomalies by deterministic fingerprint, with evidence links
  back to the raw artifacts.
  ([#165](https://github.com/Silas-Asamoah/stormlog/pull/165))

## [0.3.4] - 2026-05-09

### Added

- `stormlog.derived_fields`: `compute_event_fields()`,
  `compute_session_fields()` and `enrich_event()` compute allocator gap,
  utilization ratio, fragmentation ratio, degraded-collector detection and
  session interruption in one place, now shared by diagnose and gap
  analysis. ([#158](https://github.com/Silas-Asamoah/stormlog/pull/158))
- A canonical, backend-neutral telemetry projection:
  `CanonicalTelemetryRecord` in `stormlog.telemetry_model`, exposed through
  `LoadedTelemetrySession.telemetry_records()`, `.resources()` and
  `.correlations()` and `TrackerSession.telemetry_records()`. Persisted
  artifact and sink formats are unchanged.
  ([#160](https://github.com/Silas-Asamoah/stormlog/pull/160))

### Maintenance

- Dependabot: black 26.3.1 ([#161](https://github.com/Silas-Asamoah/stormlog/pull/161)).

## [0.3.3] - 2026-05-05

### Added

- Timeline markers: a derived `TimelineMarker` API for lifecycle, collector,
  alert, OOM and phase telemetry, exposed through artifact diagnostics and
  rendered as compact per-rank summaries in the TUI distributed diagnostics
  view. ([#157](https://github.com/Silas-Asamoah/stormlog/pull/157))

### Changed

- The contributor guide describes the `stormlog` package layout, editable
  installs, extras and the `release/dev` workflow.
  ([#156](https://github.com/Silas-Asamoah/stormlog/pull/156))

### Maintenance

- First Dependabot updates: codecov-action 6.0.0
  ([#130](https://github.com/Silas-Asamoah/stormlog/pull/130)), pygments
  2.20.0 ([#104](https://github.com/Silas-Asamoah/stormlog/pull/104)),
  actions/checkout 6.0.2 ([#133](https://github.com/Silas-Asamoah/stormlog/pull/133)),
  sphinx-rtd-theme 3.1.0 ([#136](https://github.com/Silas-Asamoah/stormlog/pull/136),
  [#145](https://github.com/Silas-Asamoah/stormlog/pull/145)), isort 8.0.1
  ([#144](https://github.com/Silas-Asamoah/stormlog/pull/144)), myst-parser
  4.0.1 ([#141](https://github.com/Silas-Asamoah/stormlog/pull/141)), rich
  15.0.0 ([#154](https://github.com/Silas-Asamoah/stormlog/pull/154)), flake8
  7.3.0 ([#146](https://github.com/Silas-Asamoah/stormlog/pull/146)).

## [0.3.2] - 2026-04-23

### Added

- Structured workload phases. Trackers gain `phase(name)`,
  `enter_phase(name)` and `PhaseHandle.close()`, emit `phase_enter` and
  `phase_exit` telemetry, and phase attribution is carried through gap
  findings, collective attribution, cross-rank first-cause suspects, the
  CLI and the TUI. Overlapping phases from different threads are marked
  ambiguous rather than guessed.
  ([#148](https://github.com/Silas-Asamoah/stormlog/pull/148),
  [#149](https://github.com/Silas-Asamoah/stormlog/pull/149))
- Optional Weights & Biases export for `track` and `diagnose` outputs:
  metrics, summaries, tables and artifact bundles are logged to a W&B run,
  with a compact inline attribution preview. Requires the optional
  `wandb>=0.19.0` dependency.
  ([#149](https://github.com/Silas-Asamoah/stormlog/pull/149),
  [#151](https://github.com/Silas-Asamoah/stormlog/pull/151))
- Reference workflows validated on real GPUs: a phase-tracking demo, a
  single-node `torchrun` DDP example
  (`examples.scenarios.torchrun_ddp_reference`) and a W&B training scenario,
  with matching cookbook recipes.
  ([#149](https://github.com/Silas-Asamoah/stormlog/pull/149))

### Fixed

- Trackers emit `sample` events during normal tracking, not only in degraded
  fallback paths, so TUI Diagnostics no longer shows `Samples 0` next to
  tracker alerts. ([#149](https://github.com/Silas-Asamoah/stormlog/pull/149))
- Phase replay tolerates malformed events, the W&B export keeps sampled
  alert rows, and TensorFlow analysis tolerates legacy attribution objects.

## [0.3.1] - 2026-04-10

### Fixed

- PyTorch `FutureWarning` noise is suppressed during CUDA native attribution
  scans. ([#127](https://github.com/Silas-Asamoah/stormlog/pull/127))
- Release publishing works again: the PyPI publish action was updated to
  accept Core Metadata 2.4, and a rerun reuses the tag already at `HEAD`
  instead of consuming a new version.
  ([#147](https://github.com/Silas-Asamoah/stormlog/pull/147))

### Security

- Release and CI workflows hardened. Releases use PyPI Trusted Publishing
  (short-lived OIDC credentials) from a manual `workflow_dispatch` on `main`
  gated by the `pypi` environment and refuse to mutate existing tags;
  actions are pinned by SHA with least-privilege permissions and no
  credential persistence; a `zizmor` audit job runs in CI; Dependabot is
  configured with 7-day cooldowns.
  ([#128](https://github.com/Silas-Asamoah/stormlog/pull/128),
  [#129](https://github.com/Silas-Asamoah/stormlog/pull/129))

## [0.3.0] - 2026-04-08

### Added

- An append-only JSONL telemetry sink for always-on tracking, with rollover,
  retention and manifest-backed segment discovery, fed by the PyTorch, CPU,
  TensorFlow and TUI tracker paths. Loaders and `analyze` read sink
  directories, and torn tails are truncated on resume.
  ([#101](https://github.com/Silas-Asamoah/stormlog/pull/101))
- Session lifecycle. Telemetry schema v3 adds a top-level `session_id`, sink
  manifests (v2) keep a session ledger, and diagnose and OOM bundles point
  at their owning session. `gpumemprof analyze --session-id` targets a
  session, the TUI can switch sessions, and the default selection is the
  newest completed, then interrupted, then incomplete session.
  ([#102](https://github.com/Silas-Asamoah/stormlog/pull/102))
- Benchmark harness v0.4 for always-on operability: `--profile {pr,nightly}`
  and `--mode {overhead,soak,all}`, per-runtime `gpumemprof_cpu` and
  `tfmemprof_cpu` results, accelerated 6h- and 24h-equivalent soaks,
  versioned budget, baseline and tolerance assets, and a nightly CI gate.
  ([#105](https://github.com/Silas-Asamoah/stormlog/pull/105))
- A production cookbook under `docs/cookbook/` for always-on tracking,
  PyTorch and TensorFlow incident response, distributed diagnostics and
  CI/release qualification, linked from `gpumemprof` and `tfmemprof` help.
  ([#120](https://github.com/Silas-Asamoah/stormlog/pull/120))
- Clearer CUDA debug attribution HTML views.
  ([#121](https://github.com/Silas-Asamoah/stormlog/pull/121))

### Changed

- Always-on tracking survives collector failures. Partial failures still
  produce samples, core failures pause sampling without stopping tracking,
  recovery uses bounded retry and backoff, and the tracker reports a health
  state (`healthy`, `degraded`, `unhealthy`) with `collector_degraded` and
  `collector_recovered` events instead of synthetic zero samples. The CLI
  and TUI show collector health, retry timing and the last error, and
  TensorFlow tracking behaves the same way.
  ([#100](https://github.com/Silas-Asamoah/stormlog/pull/100))
- TensorFlow tracking keeps a bounded recent window with explicit truncation
  metadata, and sinks report rollover and prune counters alongside
  `history_retained_*` and `history_dropped_*` diagnostics.
  ([#105](https://github.com/Silas-Asamoah/stormlog/pull/105))

### Fixed

- Telemetry sinks restart across sessions, session and manifest helpers are
  hardened, and the benchmark harness finalizes runtime sessions on failure.

### Maintenance

- Release automation computes the next patch version correctly past `.10`.

## [0.2.10] - 2026-03-27

### Added

- Native CUDA memory attribution debug mode. `gpumemprof diagnose
  --native-history` records allocator history and writes PyTorch snapshot
  artifacts with best-effort pointer-to-tensor attribution into the bundle,
  and `MemoryTracker.capture_oom()` can append the same native snapshots to
  an OOM flight-recorder bundle. Opt-in and CUDA-only; the flag is rejected
  on other runtimes, and blocks without history fall back to address and
  size attribution. ([#98](https://github.com/Silas-Asamoah/stormlog/pull/98))
- A CI memory regression gate: `examples.cli.benchmark_harness --gate-mode
  regression` compares current metrics against checked-in v0.3 baseline and
  tolerance assets, and a dedicated CI job runs it.
  ([#95](https://github.com/Silas-Asamoah/stormlog/pull/95))

### Changed

- `gpumemprof info` reports detected GPU hardware separately from the
  supported PyTorch runtime, so a machine without an active runtime no
  longer looks GPU-less, and identical GPU model names are kept as separate
  entries. ([#96](https://github.com/Silas-Asamoah/stormlog/pull/96))

## [0.2.9] - 2026-03-13

### Changed

- `stormlog[all]` now includes the visualization and TUI extras as well as
  PyTorch and TensorFlow.
  ([#87](https://github.com/Silas-Asamoah/stormlog/pull/87))

### Maintenance

- CI installs the built wheel in a fresh virtual environment and runs the
  documented `gpumemprof` smoke path against it.
  ([#87](https://github.com/Silas-Asamoah/stormlog/pull/87))

## [0.2.8] - 2026-03-13

### Changed

- Documentation separates PyPI installs from source-checkout-only example
  workflows, and pip-hostile `examples.*` commands are replaced with CLI and
  Python equivalents. ([#84](https://github.com/Silas-Asamoah/stormlog/pull/84))
- Documentation links use human-readable labels, and the README shows the
  live CI badge. ([#86](https://github.com/Silas-Asamoah/stormlog/pull/86))

### Removed

- `RELEASE_CHECKLIST.md` and `PROJECT_STATUS.md`.
  ([#86](https://github.com/Silas-Asamoah/stormlog/pull/86))

### Maintenance

- Releases are automated from successful CI runs on `main`: the next patch
  tag is computed from existing tags, built with the matching
  setuptools_scm version, published to PyPI and attached to a GitHub
  Release. ([#86](https://github.com/Silas-Asamoah/stormlog/pull/86))

## [0.2.7] - 2026-03-09

### Changed

- **Breaking:** the Python sources moved under `stormlog/`, so the import
  path matches the install name: `import stormlog` replaces
  `import gpumemprof`, and `stormlog.tensorflow` replaces `tfmemprof`. The
  `gpumemprof`, `tfmemprof` and `stormlog` console scripts are unchanged,
  and the TUI can also be launched with `python -m stormlog.tui`. Docs,
  examples and tests were rewritten for the new namespace.
  ([#83](https://github.com/Silas-Asamoah/stormlog/pull/83))

### Maintenance

- The docs build uses the Stormlog Read the Docs canonical URL.

## [0.2.6] - 2026-03-06

Packaging and docs only: the stale README overview GIF was replaced with the
current Overview-tab screenshot so the PyPI page matches the shipped TUI, with
a regression check.

## [0.2.5] - 2026-03-06

Packaging and docs only: the README PyPI version badge reads from the live
PyPI JSON API, with a regression check.

## [0.2.4] - 2026-03-06

### Changed

- Documentation rewritten around the current CLI, API and TUI surfaces, with
  regenerated TUI screenshots and README demo media. The TUI needs
  `stormlog[tui,torch]`, not `stormlog[tui]` alone.
  ([#81](https://github.com/Silas-Asamoah/stormlog/pull/81))

### Fixed

- The PyPI project page renders correctly: README images and documentation
  links in the package long description use absolute URLs, with a regression
  test.

## [0.2.3.post1] - 2026-03-06

Packaging only: the final release under the old `gpu-memory-profiler` PyPI
name, published as a deprecated alias that directs users to
`pip install stormlog`.

## [0.2.3] - 2026-03-06

### Added

- Distributed identity in telemetry: exports carry `job_id`, `rank`,
  `local_rank` and `world_size`, inferred from common launcher environment
  variables, with explicit `gpumemprof` and `tfmemprof` overrides.
- Cross-rank timeline merge and a ranked first-cause spike detector for
  multi-rank telemetry in `MemoryAnalyzer` and `gpumemprof analyze`, with a
  static `cross_rank_timeline.png` from `--visualization`.
  ([#73](https://github.com/Silas-Asamoah/stormlog/pull/73))
- Collective memory attribution heuristics, surfaced in analyzer reports and
  TUI diagnostics.
- A TUI Diagnostics tab that loads distributed artifacts, keeps mixed
  artifacts separated by rank, and renders per-rank timelines.

### Changed

- **Breaking:** the package is published on PyPI as `stormlog`
  (`pip install stormlog`); `gpu-memory-profiler` is no longer updated. The
  Python imports stay `gpumemprof` and `tfmemprof` in this release.
- **Breaking:** the Textual TUI launcher command is now `stormlog`
  (old: `gpu-profiler`). Use `stormlog` instead of `gpu-profiler` when
  launching the TUI.
- `gpumemprof analyze` loads TelemetryEvent v2 exports directly, generates
  an optimization report including hidden-memory gap findings, writes JSON
  or a text summary, and falls back to a lightweight file summary for
  non-telemetry JSON. It returns a non-zero exit code on failure.
  ([#73](https://github.com/Silas-Asamoah/stormlog/pull/73))

### Fixed

- `MemoryTracker` and `CPUMemoryTracker` reject invalid `sampling_interval`
  and `max_events` values, and `CPUMemoryProfiler.start_monitoring` rejects
  a non-positive interval.
  ([#78](https://github.com/Silas-Asamoah/stormlog/pull/78))
- `tfmemprof`: `detect_memory_leaks` no longer divides by zero when the
  baseline is zero, `profile_function(name=...)` honors the custom name, and
  the tracker rejects a non-positive interval.
  ([#79](https://github.com/Silas-Asamoah/stormlog/pull/79))
- TUI: missing visualization dependencies produce a clear install message,
  PNG export avoids pyplot, single-point timelines export, and startup and
  diagnostics loads are hardened.

### Maintenance

- Timing-sensitive tests use bounded polling instead of fixed sleeps.
  ([#80](https://github.com/Silas-Asamoah/stormlog/pull/80))

## [0.2.2] - 2026-02-22

### Changed

- **Breaking:** `torch` and `tensorflow` are optional extras. Install
  `gpu-memory-profiler[torch]`, `[tf]` or `[all]`; a bare install no longer
  pulls either framework. Torch-dependent `gpumemprof` surfaces load lazily,
  and the CPU fallback tracker no longer imports torch.
- The TUI welcome banner is branded "Stormlog", the first appearance of the
  name.
- TUI: negative memory deltas are formatted with scaled units, system info
  lookup failures fall back cleanly, and the TensorFlow summary formatter
  guards missing fields.

### Fixed

- `pyproject.toml` parses again for editable installs, and the release
  workflow enforces clean tag-derived versions for PyPI uploads.
- Example scripts skip framework-specific demos when the framework is not
  installed, and the capability matrix tolerates an optional torch or
  TensorFlow.

## [0.2.1] - 2026-02-19

### Fixed

- The release workflow fetches tags so setuptools_scm builds the tagged
  version (the v0.2.0 publish had produced a dev version that PyPI
  rejected), and the docs point at the live PyPI page.
  ([#64](https://github.com/Silas-Asamoah/stormlog/pull/64))
- README media use direct links.

## [0.2.0] - 2026-02-18

The first GitHub release and the first PyPI publish, as
`gpu-memory-profiler`.

### Added

- TelemetryEvent v2, a canonical event model with a published JSON Schema.
  GPU, CPU and TensorFlow tracker exports emit `schema_version: 2` records,
  legacy v1 records convert, and `gpumemprof.telemetry` loads, validates and
  serializes events. ([#43](https://github.com/Silas-Asamoah/stormlog/pull/43))
- Cross-backend device memory collectors for CUDA, ROCm and MPS behind a
  `DeviceMemoryCollector` contract. `gpumemprof info` reports the detected
  backend, `monitor` and `track` use the MPS collector on Apple Silicon
  instead of the CPU fallback, and a compatibility matrix is published.
  ([#48](https://github.com/Silas-Asamoah/stormlog/pull/48))
- An opt-in OOM flight recorder: `MemoryTracker(enable_oom_flight_recorder=True, ...)`,
  `handle_exception()` and `capture_oom()`, and
  `gpumemprof track --oom-flight-recorder` with dump directory, buffer and
  retention options. Dumps are bundles with `manifest.json`, `events.json`,
  `metadata.json` and `environment.json`.
  ([#49](https://github.com/Silas-Asamoah/stormlog/pull/49))
- A `diagnose` command for `gpumemprof` and `tfmemprof` that writes one
  portable bundle (`environment.json`, `telemetry_timeline.json`,
  `diagnostic_summary.json`, `manifest.json`) and exits 0 with no risk, 2
  with memory risk and 1 on failure.
  ([#50](https://github.com/Silas-Asamoah/stormlog/pull/50))
- Hidden-memory gap analysis: `analyze_memory_gaps()` classifies the gap
  between device-reported and allocator-reported memory as
  `transient_spike`, `persistent_drift` or `fragmentation_like`, with
  severity, confidence and framework-specific remediation, and feeds the
  optimization reports. ([#51](https://github.com/Silas-Asamoah/stormlog/pull/51))
- A reproducible benchmark harness with runtime, CPU, sampling and
  artifact-growth metrics and versioned v0.2 budgets checked by `--check`.
  ([#55](https://github.com/Silas-Asamoah/stormlog/pull/55))
- Launch QA scenario modules under `examples/scenarios/` for CPU telemetry,
  MPS telemetry, OOM flight recorder coverage and TensorFlow end-to-end
  telemetry and diagnose checks, and a capability matrix orchestrator
  (`python -m examples.cli.capability_matrix`) with smoke and full modes,
  target selection (`auto|cpu|mps|both`), OOM mode controls and
  machine-readable reports. ([#56](https://github.com/Silas-Asamoah/stormlog/pull/56))
- TensorFlow backend detection (CUDA, MPS or CPU) with Apple Silicon
  diagnostics and C++ log suppression.
  ([#12](https://github.com/Silas-Asamoah/stormlog/pull/12))

### Changed

- **Breaking:** Python 3.8 and 3.9 are no longer supported; the minimum
  supported runtime is Python 3.10. Users on 3.8 or 3.9 should upgrade or
  pin `gpu-memory-profiler<0.2.0`.
  ([#37](https://github.com/Silas-Asamoah/stormlog/pull/37))
- Docs and API examples refreshed to match the current CLI and profiler
  behavior, with a versioned v0.2 compatibility matrix linked from the
  top-level docs and a regression guard against stale snippets.
  ([#52](https://github.com/Silas-Asamoah/stormlog/pull/52),
  [#54](https://github.com/Silas-Asamoah/stormlog/pull/54),
  [#62](https://github.com/Silas-Asamoah/stormlog/pull/62))
- Benchmark harness defaults stabilized at `--iterations 200`, and the TUI
  CLI/Playbook quick actions highlight the diagnose, OOM scenario and
  capability matrix workflows.
  ([#56](https://github.com/Silas-Asamoah/stormlog/pull/56))
- Previously suppressed errors are logged, and a missing optional library
  raises an explicit `ImportError` with guidance instead of silently
  disabling a feature. ([#40](https://github.com/Silas-Asamoah/stormlog/pull/40))

### Fixed

- `@profile_function` runs its target once and returns its result,
  `TensorTracker.count_tensors()` no longer mutates allocator state, and
  visualization imports are lazy.
  ([#10](https://github.com/Silas-Asamoah/stormlog/pull/10))
- `CPUMemoryTracker` is thread-safe: events and stats are guarded by a lock,
  removing races and duplicate peak events.
  ([#39](https://github.com/Silas-Asamoah/stormlog/pull/39))
- `examples.basic.tensorflow_demo` matches the current TensorFlow profiler
  constructor and API.
- Stale docs references to unsupported CLI options and non-existent
  profiler APIs removed.
- TUI import errors and the CI matrix's fail-fast cancellation.
  ([#13](https://github.com/Silas-Asamoah/stormlog/pull/13))

### Maintenance

- Deterministic Textual pilot, snapshot and PTY smoke tests for the TUI,
  gated in CI. ([#19](https://github.com/Silas-Asamoah/stormlog/pull/19),
  [#38](https://github.com/Silas-Asamoah/stormlog/pull/38),
  [#41](https://github.com/Silas-Asamoah/stormlog/pull/41),
  [#42](https://github.com/Silas-Asamoah/stormlog/pull/42))

## [0.1.0] - 2024-12-19

The initial GPU Memory Profiler release, brought into this repository by
[#3](https://github.com/Silas-Asamoah/stormlog/pull/3). There is no git tag
or GitHub release for it; the date is the one its original changelog
recorded.

### Added

- `gpumemprof`, a PyTorch profiler with real-time GPU memory monitoring,
  statistical leak detection, function decorators and context managers,
  configurable alerts, a watchdog, CSV and JSON export and a CLI.
- `tfmemprof`, a TensorFlow profiler with the same workflow plus Keras,
  mixed-precision and multi-GPU strategy coverage, and its own CLI.
- Visualization and analysis: matplotlib and plotly timelines, function
  comparisons, heatmaps, dashboards, fragmentation analysis and
  optimization scoring.
- CPU memory profiling for machines without a GPU.
- Documentation, testing guides, contributing guidelines, a code of conduct,
  a security policy and the MIT license.

---

## Contributing to the Changelog

A pull request that changes something users rely on adds an entry under
**[Unreleased]**, grouped under Added, Changed, Deprecated, Removed, Fixed or
Security, with a link to the pull request. Anything that changes an exit
code, a default, a schema or a public name must be listed there and marked
**Breaking**, so it is not lost before the release. Write entries for users,
not reviewers: say what changed and what they need to do. At release time,
the Unreleased entries become the new version's section, together with
anything gathered from the release's pull requests.

---

**For more information about this project, see the [README](README.md) and
[Documentation](docs/index.md).**

[Unreleased]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.10...HEAD
[0.3.10]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.9...v0.3.10
[0.3.9]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.8...v0.3.9
[0.3.8]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Silas-Asamoah/stormlog/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.10...v0.3.0
[0.2.10]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.3.post1...v0.2.4
[0.2.3.post1]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.3...v0.2.3.post1
[0.2.3]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Silas-Asamoah/stormlog/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Silas-Asamoah/stormlog/releases/tag/v0.2.0
[0.1.0]: https://github.com/Silas-Asamoah/stormlog/pull/3
