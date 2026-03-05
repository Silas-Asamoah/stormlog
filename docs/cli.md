[← Back to main docs](index.md)

# Command Line Guide

Stormlog currently exposes three console scripts:

- `gpumemprof`
- `tfmemprof`
- `stormlog`

Use `gpumemprof` and `tfmemprof` for automation. Use `stormlog` when you want the Textual TUI.

## Verify the installed commands

```bash
gpumemprof --help
tfmemprof --help
```

If you installed the current TUI dependencies:

```bash
stormlog
```

## `gpumemprof`

The current command groups are:

- `info`
- `monitor`
- `track`
- `analyze`
- `diagnose`

### Inspect environment

```bash
gpumemprof info
gpumemprof info --device 0 --detailed
```

### Capture a bounded monitoring window

```bash
gpumemprof monitor --duration 30 --interval 0.5 --output monitor.csv --format csv
gpumemprof monitor --duration 30 --interval 0.5 --output monitor.json --format json
```

### Track events over time

```bash
gpumemprof track --duration 30 --interval 0.5 --output track.json --format json
gpumemprof track --warning-threshold 75 --critical-threshold 90 --output alerts.csv
gpumemprof track --job-id train-42 --rank 1 --local-rank 1 --world-size 8 --output rank1.json --format json
```

Optional OOM flight-recorder support:

```bash
gpumemprof track \
  --oom-flight-recorder \
  --oom-dump-dir ./oom_dumps \
  --oom-max-dumps 10 \
  --oom-max-total-mb 1024 \
  --output track.json --format json
```

### Analyze saved telemetry

```bash
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof analyze track.json --visualization --plot-dir plots
```

`gpumemprof analyze` uses a positional input file. If you add `--visualization`, plots are written to the directory passed via `--plot-dir` or to `plots/` by default.

### Produce a diagnose bundle

```bash
gpumemprof diagnose --duration 5 --interval 0.5 --output ./diag_bundle
gpumemprof diagnose --duration 0 --output ./diag_bundle_quick
```

Use `--duration 0` when you want a fast artifact bundle without a new tracking window.

## `tfmemprof`

The current command groups are:

- `info`
- `monitor`
- `track`
- `analyze`
- `diagnose`

### Inspect environment

```bash
tfmemprof info
```

### Monitor TensorFlow memory usage

```bash
tfmemprof monitor --interval 0.5 --duration 30 --output tf_monitor.json
tfmemprof monitor --interval 0.5 --duration 30 --threshold 4096 --device /GPU:0 --output tf_monitor_threshold.json
```

### Track TensorFlow memory usage

```bash
tfmemprof track --interval 0.5 --threshold 4096 --output tf_track.json
tfmemprof track --interval 0.5 --threshold 4096 --job-id train-42 --rank 3 --local-rank 1 --world-size 8 --output tf_rank3.json
```

### Analyze TensorFlow results

```bash
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize --visualize --report tf_report.txt
```

Unlike `gpumemprof analyze`, the TensorFlow analyzer uses `--input`.

### Produce a diagnose bundle

```bash
tfmemprof diagnose --duration 5 --interval 0.5 --output ./tf_diag
tfmemprof diagnose --duration 0 --output ./tf_diag_quick
```

## TUI launch

```bash
stormlog
```

Inside the TUI, the `CLI & Actions` tab exposes quick actions for:

- `gpumemprof info`
- `gpumemprof monitor`
- `tfmemprof monitor`
- `gpumemprof diagnose`
- sample workloads
- OOM scenario runner
- capability matrix smoke run

## Release-validation shortcuts

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

## Choosing the right command

### Use `monitor` when

- you want a bounded sample window
- you only need a simple CSV or JSON output

### Use `track` when

- you want event streams and alert thresholds
- you want later exports or distributed identity fields

### Use `analyze` when

- you already have saved telemetry
- you want a report or plot output

### Use `diagnose` when

- you need a portable artifact bundle to archive or share
- you plan to inspect the output later in the TUI diagnostics flow

---

[← Back to main docs](index.md)
