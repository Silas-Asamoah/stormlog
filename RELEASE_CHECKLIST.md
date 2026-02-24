# Release Checklist (v0.2)

This checklist is the source of truth for shipping `v0.2.0`.

## 1. Branch and Version Readiness

- [ ] Branch is based on latest `origin/main`
- [ ] Release work is isolated on `release/v0.2-readiness` (or final release branch)
- [ ] `CHANGELOG.md` Unreleased section is updated with v0.2 launch changes
- [ ] Version references in docs/readme are aligned with v0.2 launch candidate language

## 2. Core QA Gate (Conda: `tensor-torch-profiler`)

Run each command and capture outputs in release notes.

```bash
conda run -n tensor-torch-profiler gpumemprof --help
conda run -n tensor-torch-profiler gpumemprof info
conda run -n tensor-torch-profiler gpumemprof track --duration 2 --interval 0.5 --output /tmp/gpumemprof_track.json --format json --watchdog
conda run -n tensor-torch-profiler gpumemprof analyze /tmp/gpumemprof_track.json --format txt --output /tmp/gpumemprof_analyze.txt
conda run -n tensor-torch-profiler gpumemprof diagnose --duration 0 --output /tmp/gpumemprof_diag

conda run -n tensor-torch-profiler tfmemprof --help
conda run -n tensor-torch-profiler tfmemprof info
conda run -n tensor-torch-profiler tfmemprof monitor --interval 0.5 --duration 2 --output /tmp/tf_monitor.json
conda run -n tensor-torch-profiler tfmemprof analyze --input /tmp/tf_monitor.json --detect-leaks --optimize
conda run -n tensor-torch-profiler tfmemprof diagnose --duration 0 --output /tmp/tf_diag
```

## 3. Scenario and Example Gate

```bash
conda run -n tensor-torch-profiler python -m examples.cli.quickstart
conda run -n tensor-torch-profiler python -m examples.basic.pytorch_demo
conda run -n tensor-torch-profiler python -m examples.basic.tensorflow_demo
conda run -n tensor-torch-profiler python -m examples.advanced.tracking_demo
conda run -n tensor-torch-profiler python -m examples.cli.capability_matrix --mode full --target both --oom-mode simulated
```

- [ ] CPU telemetry scenario validated
- [ ] MPS telemetry scenario validated (or explicit SKIP reason captured)
- [ ] OOM flight-recorder scenario validated (simulated default)
- [ ] TensorFlow end-to-end telemetry + diagnose scenario validated
- [ ] Capability matrix report saved under `artifacts/examples/capability_matrix/<timestamp>/report.json`

## 4. Benchmark Gate

```bash
conda run -n tensor-torch-profiler python -m examples.cli.benchmark_harness --check --budgets docs/benchmarks/v0.2_budgets.json
```

- [ ] Budget status PASS
- [ ] Report saved to `artifacts/benchmarks/latest.json`

## 5. TUI Gate

- [ ] TUI launches successfully: `conda run -n tensor-torch-profiler stormlog`
- [ ] PTY smoke start/quit (`q`) passes
- [ ] CLI & Actions tab includes diagnose + OOM + capability-matrix launch actions

## 6. CI Gate

- [ ] CI test matrix passes on `main` PR checks
- [ ] Docs job (`sphinx -W`) passes in GitHub Actions
- [ ] TUI PR gate (`tui_pilot` + `tui_snapshot`) passes
- [ ] CLI job validates quickstart + capability matrix smoke
- [ ] Build and `twine check` pass

## 7. Hosted Docs Gate (Read the Docs)

- [ ] Read the Docs project is created and connected to this repository
- [ ] `latest` docs build succeeds from `release/v0.2-readiness`
- [ ] `stable` points to tag `v0.2.0` after release tag publish
- [ ] Hosted docs URL in `README.md` resolves and is reachable

## 8. Known Risk Tracking

- [ ] Local `pytest` segfault in `tensor-torch-profiler` documented as environment-specific risk
- [ ] Release decision explicitly uses CI + command-level QA evidence for v0.2 sign-off

## 9. Release Execution

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

- [ ] GitHub release created with v0.2 notes
- [ ] PyPI publication completed
- [ ] Post-release smoke install verified
