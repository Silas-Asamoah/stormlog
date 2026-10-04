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

## Processes

Every process the runner starts (the server, each step, each treatment) gets
a session and process group of its own, so the runner owns everything it
starts, children included: vLLM's engine and workers, and the `pip` and
`nvidia-smi` its `/server_info` collector runs. A `cpu_affinity` pins the
process, and its children inherit the pin; the runner checks the pin took.

Stopping a process signals its whole group, then SIGKILL after a timeout.
The runner then checks that nothing it started is left: no process in the
group or the session, and none of the server's processes it remembered by
PID and start time, which finds one that left the group with `setsid`. A
survivor is killed by PID; one that outlives that is a failed cleanup, and
the runner does not start the next server beside it. On Linux these checks
read `/proc`; elsewhere they use `psutil`, and remembered processes are not
tracked.

## Python API

```python
from stormlog.infer.experiment_plan import block_seed, load_plan, plan_order

plan = load_plan("plan.json")       # InferInputError for a plan it cannot run
order = plan_order(plan)            # each block's arms, balanced, position counts
seed = block_seed(plan, block=3)
```

## Related pages

- [Comparing inference runs](inference_comparison.md)
- [Inference server descriptions](inference_server.md)
- [Inference Profiling](inference.md)
