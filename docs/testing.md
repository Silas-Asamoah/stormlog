[← Back to main docs](index.md)

# Testing and Validation Guide

This guide documents the test layers that currently exist in the repo and the commands maintainers actually run.

## Install test dependencies

```bash
python3 -m pip install -e ".[test]"
```

Add framework extras when you want the full PyTorch or TensorFlow paths:

```bash
python3 -m pip install -e ".[torch]"
python3 -m pip install -e ".[tf]"
python3 -m pip install -e ".[all]"
```

If you plan to build the Sphinx docs locally, install the docs extra as well:

```bash
python3 -m pip install -e ".[docs]"
```

## Core local checks

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

## CI behavior in this repo

The current CI workflow at `.github/workflows/ci.yml` runs:

- `push` on `main` and `develop`
- `pull_request` on `main` and `release/v0.2-readiness`
- a nightly scheduled run

### Current CI lanes

- framework matrix tests across supported Python versions
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
