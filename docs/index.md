[← Back to README](../README.md)

# Stormlog Documentation

Stormlog ships four surfaces that should be treated as one workflow:

- Python APIs for profiling or tracking inside code
- CLI commands for telemetry capture and artifact generation
- OpenAI-compatible inference endpoint profiling
- a Textual TUI for live monitoring, visualization export, and diagnostics

Use the guides below based on the job you are doing, not based on package internals.

## Important: Pip vs Source Checkout

**If you installed Stormlog via `pip install stormlog`** (from PyPI):

- The `examples/` package is **not** included. Commands like `python -m examples.cli.quickstart` will fail with `ModuleNotFoundError`.
- Use the CLI commands and Python snippets in this documentation instead. The `gpumemprof`, `tfmemprof`, and `jaxmemprof` CLIs, `stormlog query`, `stormlog infer`, and the public Python APIs work with a pip install.
- Install `stormlog[tui,torch]` if you want `stormlog` or `stormlog tui` to launch the TUI from a pip install.
- The TUI **Capability Matrix** and **OOM scenario** buttons run example modules. If those fail, use the inline command runner in the TUI with the equivalent CLI commands from this guide.

**If you cloned the repository** and installed with `pip install -e .`:

- You have access to the `examples/`, `tests/`, and `docs/` source trees. The example modules and scenario runners will work.

```{toctree}
:maxdepth: 2
:caption: Guides

installation
usage
cli
inference
inference_correlation
native_probe_audit
native_probe_source_review
native_probe_experiment_specification
native_probe_security_deployment
native_probe_followups
tui
cookbook/index
cookbook/always_on
cookbook/pytorch
cookbook/tensorflow
cookbook/jax
cookbook/distributed
cookbook/incidents
cookbook/ci_release
examples
testing
troubleshooting
cpu_compatibility
compatibility_matrix
benchmark_harness
telemetry_schema
telemetry_projection
query_layer
run_envelopes
issue_fingerprinting
correlation
pytorch_testing_guide
tensorflow_testing_guide
jax_testing_guide
article
architecture
api
reference/index
gpu_setup
examples/test_guides/README
```

## Suggested reading order

### New user

1. [Installation](installation.md)
2. [Usage](usage.md)
3. [CLI](cli.md)

### Debugging a real run

1. [CLI](cli.md)
2. [Inference Profiling](inference.md)
3. [Inference execution correlation](inference_correlation.md)
4. [TUI](tui.md)
5. [Production Cookbook](cookbook/index.md)
6. [Troubleshooting](troubleshooting.md)

### Release or CI validation

1. [Testing](testing.md)
2. [CI and Release Qualification](cookbook/ci_release.md)
3. [Examples](examples.md)
4. [Benchmark Harness](benchmark_harness.md)

### Framework-specific workflows

- [PyTorch guide](pytorch_testing_guide.md)
- [TensorFlow guide](tensorflow_testing_guide.md)
- [JAX guide](jax_testing_guide.md)
- [Production Cookbook](cookbook/index.md)

## Notes

- `docs/_build/` is generated output and not maintained as source documentation.
- When docs and code disagree, treat the code and `--help` output as the source of truth and update the docs.

## Related links

- [Repository root](../README.md)
- [Contributing](../CONTRIBUTING.md)
