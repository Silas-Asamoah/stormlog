[← Back to README](../README.md)

# Stormlog Documentation

Stormlog ships three surfaces that should be treated as one workflow:

- Python APIs for profiling or tracking inside code
- CLI commands for telemetry capture and artifact generation
- a Textual TUI for live monitoring, visualization export, and diagnostics

Use the guides below based on the job you are doing, not based on package internals.

## Important: Pip vs Source Checkout

**If you installed Stormlog via `pip install stormlog`** (from PyPI):

- The `examples/` package is **not** included. Commands like `python -m examples.cli.quickstart` will fail with `ModuleNotFoundError`.
- Use the CLI commands and Python snippets in this documentation instead. The `gpumemprof` and `tfmemprof` CLIs, the `stormlog` TUI entrypoint, and the public Python APIs work with a pip install.
- The TUI **Capability Matrix** and **OOM scenario** buttons run example modules. If those fail, use the inline command runner in the TUI with the equivalent CLI commands from this guide.

**If you cloned the repository** and installed with `pip install -e .`:

- You have access to the `examples/`, `tests/`, and `docs/` source trees. The example modules and scenario runners will work.

```{toctree}
:maxdepth: 2
:caption: Guides

installation
usage
cli
tui
examples
testing
troubleshooting
cpu_compatibility
compatibility_matrix
benchmark_harness
telemetry_schema
pytorch_testing_guide
tensorflow_testing_guide
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
2. [TUI](tui.md)
3. [Troubleshooting](troubleshooting.md)

### Release or CI validation

1. [Testing](testing.md)
2. [Examples](examples.md)
3. [Benchmark Harness](benchmark_harness.md)

### Framework-specific workflows

- [PyTorch guide](pytorch_testing_guide.md)
- [TensorFlow guide](tensorflow_testing_guide.md)

## Notes

- `docs/_build/` is generated output and not maintained as source documentation.
- When docs and code disagree, treat the code and `--help` output as the source of truth and update the docs.

## Related links

- [Repository root](../README.md)
- [Contributing](../CONTRIBUTING.md)
