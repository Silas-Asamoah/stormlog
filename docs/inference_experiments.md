[← Back to docs](index.md)

# Inference experiments

A comparison is only as good as the runs behind it: matched arms, run in a
balanced order, each on a fresh server, with every command recorded. An
experiment plan says what to run; the runner (`stormlog.infer.experiment`)
runs it and leaves one directory per run for `stormlog infer compare`.

## The plan

A plan is `stormlog.infer.experiment_plan` version 1, in JSON:

```json
{"format": "stormlog.infer.experiment_plan", "version": 1,
 "experiment_id": "q213", "seed": 213, "blocks": 8,
 "order": {"kind": "williams"},
 "server": {"command": ["vllm", "serve", "{model}", "--port", "8000",
                        "--shutdown-timeout", "10"],
            "env": {"VLLM_SERVER_DEV_MODE": "1"}, "cpu_affinity": "0-3",
            "base_url": "http://127.0.0.1:8000/v1"},
 "arms": {
   "off": {"workload": [
     {"name": "c1",
      "command": ["{python}", "-m", "stormlog", "infer", "profile", "--seed", "{block_seed}",
                  "--experiment", "{experiment_id}", "--arm", "{arm}", "--block", "{block}",
                  "--output", "{run_dir}/c1.jsonl"],
      "cpu_affinity": "8-11", "timeout_s": 900, "expect_exit": [0, 3],
      "artifacts": ["{run_dir}/c1.jsonl"]}]},
   "watch": {"workload": "same_as:off",
             "treatments": [{"name": "watcher",
               "command": ["{python}", "-m", "watcher", "--ready-file", "{run_dir}/watch.ready"],
               "ready_file": "{run_dir}/watch.ready", "stop_signal": "SIGTERM"}]}}}
```

| Field | Meaning |
| --- | --- |
| `experiment_id` | A short identifier; every run is labelled with it |
| `seed`, `blocks` | The plan's seed, and how many blocks to run; each block runs every arm once |
| `order` | `random` (seeded shuffles), `williams`, or `explicit` with each block's arms listed |
| `server` | The server command, its environment, CPU set, base URL, and start and stop timeouts |
| `arms` | Each arm's server arguments and environment, its workload steps, and its treatments |
| `block_prelude` | Steps run once before each block's first run, with an arm's server (`server_arm`) or none |
| `describe_server` | Whether to describe the server `before` and `after` each run, with its log (`server_log`); all on by default |
| `secret_env` | Environment variables passed to the commands but never written down |
| `affinity_disjoint` | The server's CPUs and the commands' must not overlap |
| `prereg` | The pre-registration: what will be gated and how (see below) |

A step has a `name`, a `command`, and optionally an `env`, a `cpu_affinity`,
a `timeout_s` (900), the exit codes it may end with (`expect_exit`, `[0]`)
and the `artifacts` it must leave. An arm whose `workload` is
`"same_as:<arm>"` runs the same steps as that arm, so matched arms cannot
drift apart by copy and paste. A treatment runs beside the workload: it has
a command, a `ready_file` it writes when ready, a `stop_signal` and timeouts.

### Placeholders

Commands are templates. The runner fills, for each run:

| Placeholder | Value |
| --- | --- |
| `{run_dir}`, `{run_id}`, `{label}` | The run's directory, ID and label |
| `{experiment_id}`, `{arm}`, `{block}`, `{position}`, `{attempt}` | Its place in the plan |
| `{block_seed}` | The seed every arm of the block shares |
| `{server_pid}`, `{base_url}` | The run's server |
| `{python}` | The interpreter running the runner |
| `{model}` | The model the server loads, as resolved for this launch |

A placeholder the runner does not know is refused when the plan is loaded.

### Order and seeds

- `williams`: a Williams design. Each arm runs in each position equally
  often, and each arm follows each other arm equally often, so neither the
  position in a block nor the previous run can favor an arm. It takes as
  many blocks as arms (twice that for an odd number of arms); with another
  block count the order says `balanced: false` and gives each arm's position
  counts.
- `random`: each block's arms shuffled, from the plan's seed.
- `explicit`: each block's arms as listed; each must hold every arm once.

Every arm of a block gets the same `{block_seed}`, the first 31 bits of
SHA-256 of `<seed>:<block>`, so matched arms send identical prompts at
identical times.

## Running a plan

`stormlog.infer.experiment.run_plan(plan, output_dir)` runs each block in
the plan's order. Before a block's first run it runs the block's preludes;
a prelude that fails marks the block's runs `prelude_failed`. Each run then:

1. starts the arm's server (the plan's command plus the arm's arguments and
   environment), waits for `/health`, probes it once (`server-probe.json`,
   see [Inference Profiling](inference.md)), and checks the server's process
   tree holds only vLLM's own processes (`api_server`, `engine_core`,
   `worker` and Python's helpers), waiting up to 10 s for anything else,
   such as the `pip` the probe's collector ran, to leave;
2. describes the server (`describe-before.json`);
3. starts the arm's treatments, and waits for each one's ready file;
4. runs the workload steps in order, each to its exit code or its timeout;
5. checks each treatment is still running, then stops it with its signal;
6. describes the server again (`describe-after.json`), and gives each
   `infer` artifact the run made both descriptions: the `before` one, unless
   its workload step passed it already with `--describe-server`, then the
   `after` one; and, with a verified model, the `infer.model_identity`
   record (see Model identity);
7. stops the server's whole process group and checks nothing is left;
8. checks the promised artifacts exist and are labelled with this run's
   experiment, arm and block, writes `commands.sh`, `run.json` and, last,
   `SHA256SUMS`, and renames `runs/<label>.partial` to `runs/<label>`.

A run's label is `<experiment>-b<block>-p<position>-<arm>-a<attempt>`.

### Treatments are observers

After the workload, the runner appends an `infer.treatments` record to each
of the run's artifacts: every treatment that ran beside it, the digest of
its command template, its CPUs, and whether it was ready and stayed up. A
comparison sees each one as an observer named `treatment:<name>`: an
`overhead` baseline must have none, and an `incremental` candidate declares
the ones it adds (`--added-observers treatment:watcher`).

### How a run ends

Every run ends in exactly one state, written to `index.jsonl` with its
reasons:

| State | When | In a comparison |
| --- | --- | --- |
| `completed` | Every step exited as expected, every artifact is there and labelled, and every treatment held up | Compared |
| `outcome_failure` | The server exited (`server_exited`); a step failed or timed out; an artifact is missing; a treatment was not ready, stopped before the workload ended (`treatment_unhealthy`, even with exit 0), or exited unexpectedly | Compared: outcomes are data, and a retry never replaces them |
| `protocol_failure` | A server that never became healthy; a probe whose `/server_info` did not answer in 120 s (`probe_incomplete`); processes other than vLLM's; affinity not applied or overlapping; a failed prelude; an artifact labelled for another run; a cleanup that left processes (`collector_cleanup_unverified`, `treatment_cleanup_unverified:<name>`) | Set aside with its block, both arms, with the reason; may be retried |

When both kinds apply, the outcome wins, unless the protocol fault came
before the first workload step started. The runner appends the state to
each of the run's artifacts as an `infer.run_state` record (`state`,
`reasons`, `before_treatment`), and `infer compare` reads it: an outcome
failure is compared and counted against its arm (`runner:<reason>`), even
when the profile's session finished, and a protocol failure is the external
cause that sets aside its block (`external:<reason>`). A cleanup that left processes stops
the block: no server starts beside them. A `probe_incomplete` run is run
again at once on a fresh server, after its group is verified gone, and both
attempts are kept.

### The bundle

| Path | Content |
| --- | --- |
| `plan.json` | The plan's and the pre-registration's SHA-256 |
| `prereg.json` | The pre-registration, when the plan has one |
| `order.json` | Each block's arms in run order, whether positions balance, and each arm's position counts |
| `index.jsonl` | One line per attempt: state, reasons, every process with its PID, times, exit code and affinity, the server's cleanup |
| `runs/<label>/` | The run: its artifacts, `describe-*.json`, the logs of the server, every step and treatment, `commands.sh`, `run.json`, `SHA256SUMS` |
| `preludes/` | Each block's preludes and their logs |
| `sanitizer.json` | Whether the bundle is publishable, and any secret found, by file and line |

`commands.sh` holds each command exactly as run. A `secret_env` variable is
passed to the commands from the runner's environment and written only as
`NAME=${NAME}`.

When the plan has run, every file of the bundle is scanned
(`sanitizer.json`) for the plan's secret values and for the shapes
credentials take: an `Authorization: Bearer` value, a Hugging Face `hf_`
token, an `sk-` key. A hit names the file and line, never the value, and
makes the bundle not `publishable`.

### Resuming

`run_plan(..., resume=True)` continues an experiment in the same directory.
It refuses a plan or pre-registration that changed. A run is skipped when
its last attempt's `SHA256SUMS` verifies and it ended `completed` or
`outcome_failure`. With `retry_incomplete=True`, a run that ended in a
protocol failure (or whose digests do not verify) gets another attempt; the
earlier attempt stays on disk and in the index, and the new one records
`order_broken: true`, since it runs later than planned.

A run the runner itself was killed in (a preempted box, an operator's
abort) leaves its attempt in `runs/<label>.partial`, with no state. A resume
finishes that attempt in place, keeps its number, and indexes it with
`interrupted: true`. It is an `outcome_failure` (`runner_interrupted`),
kept and never retried, since what stopped the runner may have been the
treatment, unless its cause is given:

```python
run_plan(plan, out, resume=True, retry_incomplete=True,
         external_causes={"t221-b03-p1-watch-a1": ExternalCause(
             "spot_preemption", "box found paused without a release at 03:12")})
```

The reason must be `spot_preemption`, `operator_abort` or `infra_fault`,
with non-empty evidence, and the label must name an attempt that was
interrupted: a finished attempt keeps the state it recorded, so an outcome
failure can never become a set-aside. The attempt is then a
`protocol_failure` with that reason, its `run.json`, index line and
artifacts (`infer.run_state`) carry the cause and evidence, and
`retry_incomplete` runs it again. `infer compare` sets aside the attempt
(`external:spot_preemption`), lists the evidence, and keeps the retry in
its place.

## Running from the command line

```bash
python -m examples.cli.infer_repeated_baseline --plan plan.json --output exp213
python -m examples.cli.infer_repeated_baseline --plan plan.json --output exp213 --resume
python -m examples.cli.infer_repeated_baseline --plan plan.json --output exp213 \
  --resume --retry-incomplete \
  --external-cause 't213-b01-p0-off-a1=spot_preemption:box paused at 03:12'
```

It prints one line per run, with its state and reasons. It exits `0` when
every planned run finished (completed, or an outcome failure, which is
data), `3` when some run ended in a protocol failure (retry it with
`--resume --retry-incomplete`), `5` for a plan or output directory it
cannot use, and `2` for a secret the plan names that is not set or an
`--external-cause` it refuses. Then
compare the arms:

```bash
stormlog infer compare --baseline exp213/runs/*-off-a*/c1.jsonl \
  --candidate exp213/runs/*-watch-a*/c1.jsonl --gate 'client.e2e.p95=non-inferiority:0.05'
```

## Model identity

A digest of a model path taken after a server started says nothing about
what the server loaded: the path may have changed in between. So the runner
fixes the weights before it launches anything, and points every server at
exactly them. The plan's `server.model` says how:

```json
"model": {"route": "pinned_hub", "repo": "Qwen/Qwen2.5-0.5B-Instruct",
          "revision": "main", "hub_cache": "/home/.cache/huggingface/hub"}
```

```json
"model": {"route": "staged", "source": "/models/qwen-0.5b", "store": "/home/model-store"}
```

- **`pinned_hub`**: the revision is resolved to a commit in the cache, and
  every file of that snapshot is hashed and checked against its blob's name
  (SHA-256 for a file in LFS, git's SHA-1 for the rest). Each server gets
  `--revision <commit> --tokenizer-revision <commit>` and `HF_HUB_OFFLINE=1`,
  so it cannot load anything else. `{model}` is the repository.
- **`staged`**: every file of a local directory is hashed, and the directory
  is hard-linked (or copied) into `<store>/<weights_digest>/`, read-only. The
  server loads that directory: its name is its content. `{model}` is its
  path.

After each run the files are checked again, by size, modification time and
inode; a change makes the run `protocol_failure: model_changed`. Each run's
`model_identity.json` records the files and their digests, and the runner
appends an `infer.model_identity` record to each artifact: the verified
model, with its evidence (`pinned_commit_verified` or
`staged_snapshot_verified`), bound to the API server it launched by boot,
PID and start ticks, read as it started. Only this record verifies a model's
identity in a comparison, and only for the server the run's `before`
description shows; a description file alone never does, whatever evidence
it names. Off Linux the start ticks cannot be read, so the record is not
written and the identity stays unverified.

Without `server.model`, `{model}` is the plan's `server.model.name` (or
empty), and the model's identity stays unverified.

## Processes

Every process the runner starts (the server, each step, each treatment) gets
a session and process group of its own, so the runner owns everything it
starts, children included: vLLM's engine and workers, and the `pip` and
`nvidia-smi` its `/server_info` collector runs. A `cpu_affinity` pins the
process, and its children inherit the pin; the runner checks the pin took.

Stopping a process signals its whole group, then SIGKILL after a timeout.
The runner then checks that nothing it started is left: no process in the
group or the session, none of the server's processes it remembered by PID
and start time, which finds one that left the group with `setsid`, and no
process carrying the launch's mark. Every launch puts a fresh
`STORMLOG_RUN_MARK` in its environment, which every descendant inherits, so
a process forked after the tree was remembered (a late collector child) and
then moved to a session of its own is found too. A survivor is killed by
PID; one that outlives that is a failed cleanup (`collector_cleanup_unverified`
for the server, `treatment_cleanup_unverified:<name>` for a treatment), and
the runner does not start the next server beside it. On Linux these checks
read `/proc`; elsewhere they use `psutil`, which reads the mark too, and
remembered processes are not tracked. The mark differs from run to run and
is a label to comparisons.

## Python API

```python
from pathlib import Path

from stormlog.infer.experiment_plan import block_seed, load_plan, plan_order

plan = load_plan("plan.json")       # InferInputError for a plan it cannot run
order = plan_order(plan)            # each block's arms, balanced, position counts
seed = block_seed(plan, block=3)

from stormlog.infer.experiment import run_plan

records = run_plan(plan, Path("exp213-pilot"), resume=False)
```

## Related pages

- [Comparing inference runs](inference_comparison.md)
- [Inference server descriptions](inference_server.md)
- [Inference Profiling](inference.md)
