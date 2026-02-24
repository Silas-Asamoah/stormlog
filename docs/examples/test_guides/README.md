# Testing Guides (Markdown Edition)

These guides replace the legacy Python scripts that used to live in
`examples/test_guides/`. Instead of running massive ad-hoc scripts, you can now
follow these concise workflows and reuse the curated example modules.

## CPU-Only Sanity Check

Even without a GPU, you can verify the installation, inspect system info, and
exercise the CLI.

```bash
# Show system + GPU summary (falls back to CPU)
gpumemprof info

# Run the CLI quickstart (also part of CI)
python -m examples.cli.quickstart
```

For more CPU-focused tips, see `docs/cpu_compatibility.md`.

### CPU Smoke Test (Automated)

Use this duo whenever CUDA isn’t available:

```bash
# Step 1: force CPU mode and run the CLI walkthrough
set CUDA_VISIBLE_DEVICES=    # PowerShell/CMD
# or: export CUDA_VISIBLE_DEVICES=   # macOS/Linux shells
python -m examples.cli.quickstart

# Step 2: verify system-info fallbacks
pytest tests/test_utils.py
```

Both steps run quickly and unblock CI jobs that lack GPUs.

## Launch Matrix (Recommended)

Run one command to validate current launch capabilities:

```bash
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

This writes a machine-readable report under:

`artifacts/examples/capability_matrix/<timestamp>/report.json`

On macOS, `--target both` exercises CPU + MPS paths together.

## PyTorch GPU Checklist

```bash
# Basic profiling walkthrough
python -m examples.basic.pytorch_demo

# Advanced tracker/watchdog demo
python -m examples.advanced.tracking_demo
```

Both scripts emit summaries, alert counts, and export artifacts under
`artifacts/advanced_demo/`.

## TensorFlow GPU Checklist

```bash
# Minimal TensorFlow profiling run
python -m examples.basic.tensorflow_demo
```

The demo configures memory growth automatically and prints peak/average memory
statistics gathered via `TFMemoryProfiler`.

## CLI Smoke Test (PyTorch + TensorFlow)

```bash
python -m examples.cli.quickstart
```

This runs the same commands exercised in CI (`gpumemprof --help`, `info`,
optional `monitor`, and the equivalent `tfmemprof` commands).

### Scenario Modules (Telemetry + OOM + Diagnose)

```bash
python -m examples.scenarios.cpu_telemetry_scenario
python -m examples.scenarios.mps_telemetry_scenario
python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
python -m examples.scenarios.tf_end_to_end_scenario
```

Prefer an interactive dashboard? Launch the Textual TUI:

```bash
pip install "gpu-memory-profiler[tui]"
stormlog
```

## Enabling the CUDA Path

Ready to profile on GPUs? Follow `docs/gpu_setup.md` for driver + framework
installation instructions (PyTorch CUDA wheels, TensorFlow GPU wheel, and
verification commands). Once complete, unset `CUDA_VISIBLE_DEVICES` (or point
it to a real GPU) and rerun the PyTorch/TensorFlow checklists above.

## Automation Tips

- Integrate the demos into your workflow by adding `python -m examples.basic.*`
  steps to release or pre-commit scripts.
- Use `pytest tests/test_utils.py` for a lightweight regression check when GPUs
  are not available.
- On CI, we already run `examples.cli.quickstart`; feel free to add extra jobs
  that execute the PyTorch/TensorFlow demos on GPU-enabled runners.

## Related Documentation

- `docs/pytorch_testing_guide.md`
- `docs/tensorflow_testing_guide.md`
- `docs/testing.md`
