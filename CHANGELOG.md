# Changelog

All notable changes to GPU Memory Profiler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

- Launch QA scenario modules under `examples/scenarios/` for CPU telemetry, MPS telemetry, OOM flight recorder coverage, and TensorFlow end-to-end telemetry/diagnose checks.
- Capability matrix orchestrator (`python -m examples.cli.capability_matrix`) with smoke/full modes, target selection (`auto|cpu|mps|both`), OOM mode controls, and machine-readable reports.
- Scenario smoke tests (`tests/test_examples_scenarios.py`) and updated TUI pilot coverage for launch quick actions.
- Updated TUI snapshot baselines for intentional CLI & Actions tab changes.

### Changed

- Drop support for Python 3.8 and 3.9; minimum supported runtime is now Python 3.10.
- Migration note: users on Python 3.8/3.9 should upgrade to Python 3.10+ or pin `gpu-memory-profiler<0.2.0`.
- Breaking change: the Textual TUI launcher command is now `stormlog` (old: `gpu-profiler`).
- Migration note: use `stormlog` instead of `gpu-profiler` when launching the TUI.
- Refresh docs/API examples to match current CLI and profiler behavior.
- Publish a versioned compatibility matrix for v0.2 and link it from top-level docs.
- Stabilize benchmark harness defaults (`--iterations 200`) and align benchmark/testing documentation to this baseline.
- Expand TUI CLI/Playbook guidance and quick actions to highlight diagnose, OOM scenario, and capability matrix workflows.
- Refresh release-facing docs (`README`, examples guides, `RELEASE_CHECKLIST`, `PROJECT_STATUS`) for v0.2 launch readiness.

### Deprecated

- [Future deprecations will be listed here]

### Removed

- [Future removals will be listed here]

### Fixed

- Remove stale docs references to unsupported CLI options and non-existent profiler APIs.
- Fix `examples.basic.tensorflow_demo` constructor/API mismatch so the demo runs against the current TensorFlow profiler implementation.

### Security

- [Future security fixes will be listed here]

## [0.1.0] - 2024-12-19

### Added

- **Core PyTorch Profiler (`gpumemprof`)**

  - Real-time GPU memory monitoring with configurable sampling intervals
  - Memory leak detection using statistical analysis and pattern recognition
  - Interactive visualizations with matplotlib and plotly support
  - Context-aware profiling with function decorators and context managers
  - Command-line interface for standalone usage
  - Advanced analytics including pattern detection and fragmentation analysis
  - Alert system with configurable thresholds
  - Data export capabilities (CSV, JSON)
  - Automatic memory management with watchdog system

- **Core TensorFlow Profiler (`tfmemprof`)**

  - Real-time TensorFlow GPU memory monitoring
  - TensorFlow-specific memory leak detection
  - Integration with TensorFlow's memory management system
  - Support for TensorFlow sessions and graph execution
  - Keras model profiling capabilities
  - Mixed precision profiling support
  - Multi-GPU strategy profiling
  - Command-line interface for TensorFlow workflows

- **Visualization & Analysis**

  - Memory timeline plots with interactive features
  - Function comparison charts
  - Memory usage heatmaps
  - Interactive dashboards with Plotly
  - Memory fragmentation analysis
  - Performance correlation analysis
  - Optimization scoring and recommendations

- **Command Line Tools**

  - `gpumemprof` CLI for PyTorch profiling
  - `tfmemprof` CLI for TensorFlow profiling
  - System information display
  - Real-time monitoring capabilities
  - Background tracking with alerts
  - Results analysis and visualization

- **CPU Compatibility**

  - CPU memory profiling for systems without GPU
  - Cross-platform compatibility
  - CPU-based model training profiling
  - System RAM monitoring capabilities

- **Testing & Documentation**
  - Comprehensive test suite for both GPU and CPU environments
  - PyTorch testing guide with examples
  - TensorFlow testing guide with examples
  - CPU compatibility guide
  - Complete API documentation
  - Usage examples and tutorials
  - Troubleshooting guides

### Technical Features

- Modular architecture with 7 core components
- Thread-safe profiling with background monitoring
- Configurable sampling intervals and alert thresholds
- Support for multiple GPU devices
- Memory snapshot capture and analysis
- Tensor lifecycle tracking (PyTorch)
- Graph execution monitoring (TensorFlow)
- Export capabilities for further analysis

### Documentation

- Comprehensive documentation in `/docs/` directory
- Quick start guides for both PyTorch and TensorFlow
- API reference with examples
- CLI usage guide
- CPU compatibility guide
- Testing guides for both frameworks
- In-depth technical article
- Contributing guidelines and code of conduct

### Infrastructure

- Open source project structure
- MIT License
- Contributing guidelines (CONTRIBUTING.md)
- Code of Conduct (CODE_OF_CONDUCT.md)
- Security policy (SECURITY.md)
- Changelog tracking
- Development setup instructions

---

## Version History

- **0.1.0** (2024-12-19): Initial release with full PyTorch and TensorFlow support

## Release Notes

### Version 0.1.0

This is the initial release of GPU Memory Profiler, providing comprehensive memory profiling capabilities for both PyTorch and TensorFlow deep learning frameworks. The release includes:

- Complete PyTorch profiler with real-time monitoring, leak detection, and visualization
- Complete TensorFlow profiler with TensorFlow-specific optimizations
- Command-line interfaces for both frameworks
- CPU compatibility for systems without GPU support
- Comprehensive documentation and testing guides
- Open source project structure ready for community contributions

### Breaking Changes

None - this is the initial release.

### Known Issues

- Some visualization features may require additional dependencies (PyQt5, tkinter)
- TensorFlow CLI may have dependency conflicts with certain typing-extensions versions
- CPU profiling is limited compared to GPU profiling capabilities

### Migration Guide

N/A - this is the initial release.

---

## Contributing to the Changelog

When contributing to the project, please update this changelog by adding entries under the appropriate version section. Follow the format:

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for security vulnerability fixes

Use the present tense ("Add" not "Added") and imperative mood ("Move cursor to..." not "Moves cursor to...").

---

**For more information about this project, see the [README](README.md) and [Documentation](docs/index.md).**
