[← Back to main docs](index.md)

# Command Line Interface (CLI) Guide

This guide covers the current v0.2 command flows for:

- `gpumemprof` (PyTorch + backend-aware collector stack)
- `tfmemprof` (TensorFlow profiler tooling)

## Install and Verify

```bash
pip install -e .

gpumemprof --help
tfmemprof --help
```

## Canonical CLI Workflow

Use this sequence for reproducible diagnostics:

```bash
# 1) Inspect environment
gpumemprof info
tfmemprof info

# 2) Capture short telemetry sample
gpumemprof track --duration 2 --interval 0.5 --output /tmp/gpumemprof_track.json --format json --watchdog
gpumemprof analyze /tmp/gpumemprof_track.json --format txt --output /tmp/gpumemprof_analysis.txt

# 3) Produce diagnostic bundles
gpumemprof diagnose --duration 0 --output /tmp/gpumemprof_diag
tfmemprof diagnose --duration 0 --output /tmp/tf_diag
```

## `gpumemprof` Commands

### `info`

```bash
gpumemprof info
gpumemprof info --device 0 --detailed
```

### `monitor`

```bash
gpumemprof monitor --duration 30 --interval 0.5 --output monitor.csv --format csv
gpumemprof monitor --duration 30 --interval 0.5 --output monitor.json --format json
```

### `track`

```bash
gpumemprof track --duration 30 --interval 0.5 --output track.json --format json --watchdog
gpumemprof track --warning-threshold 75 --critical-threshold 90 --output alerts.csv
gpumemprof track --job-id train-42 --rank 1 --local-rank 1 --world-size 8 --output rank1.json --format json
```

OOM flight-recorder options are available for stress workflows:

```bash
gpumemprof track \
  --oom-flight-recorder \
  --oom-dump-dir ./oom_dumps \
  --oom-max-dumps 10 \
  --oom-max-total-mb 1024 \
  --output track.json --format json
```

### `analyze`

```bash
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof analyze track.json --visualization --plot-dir plots
```

### `diagnose`

```bash
gpumemprof diagnose --duration 5 --interval 0.5 --output ./diag_bundle
gpumemprof diagnose --duration 0 --output ./diag_bundle_quick
```

Exit codes:

- `0`: success, no memory risk
- `1`: runtime/argument failure
- `2`: success with risk detected

## `tfmemprof` Commands

### `info`

```bash
tfmemprof info
```

### `monitor`

```bash
tfmemprof monitor --interval 0.5 --duration 30 --output tf_monitor.json
```

### `track`

```bash
tfmemprof track --interval 0.5 --threshold 4096 --output tf_track.json
tfmemprof track --interval 0.5 --threshold 4096 --job-id train-42 --rank 3 --local-rank 1 --world-size 8 --output tf_rank3.json
```

### `analyze`

```bash
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize --report tf_report.txt
```

### `diagnose`

```bash
tfmemprof diagnose --duration 5 --interval 0.5 --output ./tf_diag
tfmemprof diagnose --duration 0 --output ./tf_diag_quick
```

## Launch QA Commands

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

Use `--mode full` for exhaustive examples coverage.

---

[← Back to main docs](index.md)
