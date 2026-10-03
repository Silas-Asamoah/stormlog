[← Back to main docs](index.md)

# Testing and Validation Guide

This guide documents the test layers that currently exist in the repo and the commands maintainers actually run.

## Install test dependencies

```bash
python3 -m pip install -e ".[test]"
```

Add framework extras when you want the full PyTorch, TensorFlow, or JAX paths:

```bash
python3 -m pip install -e ".[torch]"
python3 -m pip install -e ".[tf]"
python3 -m pip install -e ".[jax]"
python3 -m pip install -e ".[all]"
```

`.[all]` now includes every runtime extra: PyTorch, TensorFlow, visualization,
and TUI dependencies.

If you plan to build the Sphinx docs locally, install the docs extra as well:

```bash
python3 -m pip install -e ".[docs]"
```

## Core local checks

### Pre-commit hooks

Install the hooks once per checkout so local edits run the same formatting and
static checks that coding agents should use before pushing:

```bash
python3 -m pre_commit install
```

Run the full hook set manually before handing off a branch:

```bash
python3 -m pre_commit run --all-files
```

### Full pytest run

```bash
python3 -m pytest -v
```

### PyTorch-oriented test slice

```bash
python3 -m pytest tests/ --ignore-glob="tests/test_tf*.py" -v -m "not tui_pilot and not tui_snapshot and not tui_pty"
```

### TensorFlow-oriented test slice

```bash
python3 -m pytest tests/ -o "python_files=test_tf*.py" -v -m "not tui_pilot and not tui_snapshot and not tui_pty"
```

### JAX-oriented test slice

```bash
python3 -m pytest tests/ -o "python_files=test_jax*.py" -v -m "not tui_pilot and not tui_snapshot and not tui_pty"
```

## TUI test layers

The TUI suite is intentionally split into three layers:

- `tui_pilot`: interaction-level tests for button clicks and navigation
- `tui_snapshot`: deterministic visual layout checks using exported SVG renders
- `tui_pty`: end-to-end smoke in a real terminal session

### Run the layers individually

```bash
python3 -m pytest tests/tui/test_app_pilot.py -m tui_pilot -v
python3 -m pytest tests/tui/test_app_snapshots.py -m tui_snapshot -v
python3 -m pytest tests/e2e/test_tui_pty.py -m tui_pty -v
```

## Example and scenario validation

> **Source checkout only.** The commands in this section require the repository
> `examples/` package.

### CLI smoke

```bash
python3 -m examples.cli.quickstart
```

### Capability matrix

```bash
python3 -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

### Scenario modules

```bash
python3 -m examples.scenarios.cpu_telemetry_scenario
python3 -m examples.scenarios.mps_telemetry_scenario
python3 -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
python3 -m examples.scenarios.tf_end_to_end_scenario
python3 -m examples.scenarios.wandb_training_smoke --device cuda --wandb-mode offline
python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=2 -m examples.scenarios.torchrun_ddp_reference
```

### CLI-only validation (pip install)

If you installed from PyPI and do not have the `examples/` package, use this
sequence instead:

```bash
gpumemprof info
gpumemprof track --duration 10 --interval 0.5 --output track.json --format json
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof diagnose --duration 0 --output ./diag
tfmemprof info
tfmemprof diagnose --duration 0 --output ./tf_diag
```

Optional MLflow export check (requires `pip install 'stormlog[mlflow]'`):

```bash
gpumemprof track --duration 10 --interval 0.5 --output track.json --format json \
  --mlflow --mlflow-experiment stormlog-smoke --mlflow-tracking-uri sqlite:///mlflow.db
```

## Fake vLLM engine

> **Source checkout only.** `examples.qualification.fake_engine` is not
> shipped in the PyPI package.

The inference tests run against a CPU-only stand-in for vLLM 0.30.0 instead of
a GPU server. It simulates continuous batching, so each mechanism moves the
series vLLM exports:

- **Scheduling.** Running requests are scheduled first, then waiting ones in
  arrival order, up to `max_num_seqs` and the token budget, with chunked
  prefill.
- **KV blocks.** When they run out, the newest running request is preempted
  and recomputed, as vLLM's scheduler does.
- **Prefix cache.** Freed blocks keep their hashes in an LRU queue, so a shared
  prefix is reused until other traffic evicts it.

Nothing is computed: each step sleeps for a simulated cost. Prompts are counted
as whitespace words plus a fixed four-token chat template.

```python
from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig

config = FakeEngineConfig(max_num_seqs=4, num_gpu_blocks=64, hook_dir=hook, trace_dir=traces)
with FakeEngine(config) as engine:
    ...  # engine.endpoint, engine.metrics_url, engine.base_url
```

To run it as its own process, so that signals reach it:

```bash
python -m examples.qualification.fake_engine --port 0 --step-seconds 0.001
```

It prints `FAKE_ENGINE_URL=<url> PID=<pid>` once it serves.
`FakeEngineProcess` starts it from a test and always continues it before
stopping it.

What it serves, as vLLM 0.30.0 does:

| Route | Behaviour |
| --- | --- |
| `POST /v1/chat/completions` | Streamed or whole, with usage. A request is named `chatcmpl-<X-Request-Id>`, plus vLLM's random suffix unless `request_id_randomization=False` |
| `GET /metrics` | vLLM's series names, labels and histogram buckets. The page is taken after each step, so it holds the last step's values while the loop is paused |
| `POST /reset_prefix_cache` | Runs on the step loop between steps, as vLLM's utility calls do. `{"success": false}` while blocks are held. With `reset_running_requests=true`, it preempts every running request first, and the next step lists them in its `preempted` |
| `POST /start_profile`, `/stop_profile` | Run between steps, with a configurable stop pause. Repeated calls answer 200. The stop writes a gzipped `rank0.*.pt.trace.json.gz` whose iteration ranges name the hook's steps. Without `trace_dir` both answer 404 |
| `GET /server_info`, `/version`, `/v1/models`, `/health` | Descriptive answers |

**Optional outputs:**

- `hook_dir` writes the execution hook's raw log through Stormlog's own
  writer, in the `stormlog.vllm_hook/1` format of
  [vLLM execution hook](vllm_execution.md). `infer profile --vllm-execution-dir`
  imports it, and `hook_seal_seconds` makes segments visible quickly.
- `spans_endpoint` exports each request's `llm_request` span as OTLP/HTTP JSON.
  A `traceparent` header makes the span its child.

**Fault controls,** over `/_fault/` routes that answer even while the front end
is held:

| Control | Effect |
| --- | --- |
| `pause?target=engine` / `frontend` (`&seconds=S`), `resume` | Hold the step loop, or every API answer while the engine keeps stepping |
| `controls` (JSON body) | Set any switch in `Controls`: profiler status, start and stop pauses, a lost start answer, a stop without a trace, a `max_iterations` self-stop, a delayed trace write, a failing or slow `/metrics`, late or duplicated spans |
| `foreign_trace`, `span_body?kind=oversized` / `gzip_bomb` | Drop a trace outside any window; send a span body the receiver must refuse |
| `state` | Steps, queue, preemptions |
| `kill` | Exit at once, like SIGKILL (subprocess mode only) |

**Not modeled:**

- async scheduling;
- tensor parallelism above 1;
- speculative decoding;
- real GPU timing.

New hook fields reach the fake engine in the change that adds them to the hook.

## CI behavior in this repo

The current CI workflow at `.github/workflows/ci.yml` runs:

- `push` on `main` and `develop`
- `pull_request` on `main` and `release/v0.2-readiness`
- a nightly scheduled run

### Current CI lanes

- framework matrix tests across supported Python versions
- built-wheel CLI smoke in a fresh environment
- source-checkout example smoke for `examples/` modules
- TUI pilot and snapshot tests on pull requests and `develop`
- TUI PTY smoke on `main` pushes and scheduled runs
- lint, docs, and package build checks in separate jobs

## Documentation checks

Documentation source is built from `docs/`, not from checked-in generated HTML.

### Rebuild docs locally

```bash
python3 -m sphinx -W --keep-going -b html docs docs/_build/html
```

This step requires the docs dependencies shown above.

### Run docs regression checks

```bash
python3 -m pytest tests/test_docs_regressions.py -v
```

The `docs/_build/` directory is build output. It may exist locally after a Sphinx build, but it is not maintained as source documentation.

## CPU-only validation

For laptop or CI environments without CUDA:

```bash
export CUDA_VISIBLE_DEVICES=
gpumemprof info
gpumemprof track --duration 10 --interval 0.5 --output cpu_track.json --format json
gpumemprof analyze cpu_track.json --format txt --output cpu_analysis.txt
gpumemprof diagnose --duration 0 --output ./cpu_diag
```

On Windows shells, clear `CUDA_VISIBLE_DEVICES` with the platform-appropriate syntax before running the same commands.

On Apple Silicon, clearing `CUDA_VISIBLE_DEVICES` disables CUDA but
`gpumemprof info` may still report the `mps` backend. Treat this as a
non-CUDA smoke test rather than a strict CPU-only force.

If you have a source checkout, you can also run `python3 -m pytest tests/test_utils.py -v`.
Pip installs do not include the `tests/` package.

## Recommended release-validation sequence

Use this when you need a compact but meaningful confidence pass:

```bash
python3 -m examples.cli.quickstart
python3 -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
python3 -m pytest tests/tui/test_app_pilot.py -m tui_pilot -v
python3 -m pytest tests/tui/test_app_snapshots.py -m tui_snapshot -v
python3 -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Related guides

- [Examples Guide](examples.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Example Test Guides](examples/test_guides/README.md)

---

[← Back to main docs](index.md)
