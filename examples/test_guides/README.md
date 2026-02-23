# Legacy Test Guides

The executable guides that previously lived in this directory have been
migrated to Markdown-based documentation under `docs/examples/test_guides/`.

Use the new docs for step-by-step instructions, and run the curated example
modules instead:

- `python -m examples.basic.pytorch_demo`
- `python -m examples.basic.tensorflow_demo`
- `python -m examples.advanced.tracking_demo`
- `python -m examples.cli.quickstart`

These cover the same scenarios as the removed scripts (CPU-only validation,
PyTorch GPU profiling, TensorFlow GPU profiling, and CLI walkthroughs) while
keeping the repository lint-clean.
