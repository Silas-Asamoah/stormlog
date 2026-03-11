[← Back to README](../README.md)

# Stormlog Documentation

Welcome to the official documentation for Stormlog! Here you'll find everything you need to install, use, and contribute to the project.

## Important: Pip vs Source Checkout

**If you installed Stormlog via `pip install stormlog`** (from PyPI):

- The `examples/` package is **not** included. Commands like `python -m examples.cli.quickstart` will fail with `ModuleNotFoundError`.
- Use the CLI commands and Python snippets in this documentation instead. The `gpumemprof` and `tfmemprof` CLIs, the `stormlog` TUI entrypoint, and the public Python APIs work with a pip install.
- The TUI **Capability Matrix** and **OOM scenario** buttons run example modules. If those fail, use the inline command runner in the TUI with the equivalent CLI commands from this guide.

**If you cloned the repository** and installed with `pip install -e .`:

- You have access to the `examples/`, `tests/`, and `docs/` source trees. The example modules and scenario runners will work.

```{toctree}
:maxdepth: 2
:caption: Documentation

installation
usage
cli
api
reference/index
examples
examples/test_guides/README
cpu_compatibility
compatibility_matrix
benchmark_harness
telemetry_schema
testing
troubleshooting
architecture
gpu_setup
tui
pytorch_testing_guide
tensorflow_testing_guide
article
```

## About

Stormlog is an open source tool for real-time GPU memory monitoring, leak detection, and optimization in deep learning workflows. It supports both PyTorch and TensorFlow, and provides both a Python API and CLI tools.

## Additional Links

- [Contributing](../CONTRIBUTING.md)

---

[← Back to README](../README.md)
