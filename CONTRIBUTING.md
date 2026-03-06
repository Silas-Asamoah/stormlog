# Contributing to Stormlog

Thank you for your interest in contributing to Stormlog. This document provides
guidelines and information for contributors.

## Table of Contents

-   [Code of Conduct](#code-of-conduct)
-   [How Can I Contribute?](#how-can-i-contribute)
-   [Development Setup](#development-setup)
-   [Code Style](#code-style)
-   [Testing](#testing)
-   [Pull Request Process](#pull-request-process)
-   [Reporting Bugs](#reporting-bugs)
-   [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

-   Use the GitHub issue tracker
-   Include a clear and descriptive title
-   Provide detailed steps to reproduce the bug
-   Include system information (OS, Python version, PyTorch/TensorFlow version, CUDA version)
-   Include error messages and stack traces
-   Describe the expected behavior

### Suggesting Enhancements

-   Use the GitHub issue tracker
-   Provide a clear description of the enhancement
-   Explain why this enhancement would be useful
-   Include mockups or examples if applicable

### Pull Requests

-   Fork the repository
-   Create a feature branch (`git checkout -b feature/amazing-feature`)
-   Make your changes
-   Add tests for new functionality
-   Ensure all tests pass
-   Update documentation
-   Submit a pull request

## Development Setup

### Prerequisites

-   Python 3.10 or higher
-   Git
-   pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Silas-Asamoah/stormlog.git
cd stormlog

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Development Dependencies

```bash
pip install pytest pytest-cov black flake8 mypy
```

## Code Style

### Python Code

We use:

-   **Black** for code formatting
-   **flake8** for linting
-   **mypy** for type checking

```bash
# Format code
black gpumemprof/ tfmemprof/ tests/ examples/

# Check linting
flake8 gpumemprof/ tfmemprof/ tests/ examples/

# Type checking
mypy gpumemprof/ tfmemprof/
```

### Documentation

-   Use Markdown for documentation
-   Follow the existing documentation structure
-   Include code examples
-   Update the table of contents when adding new pages

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:

-   `feat`: New feature
-   `fix`: Bug fix
-   `docs`: Documentation changes
-   `style`: Code style changes
-   `refactor`: Code refactoring
-   `test`: Adding or updating tests
-   `chore`: Maintenance tasks

## Testing

### Running Tests

```bash
# Run all tests
python3 -m pytest

# Run with coverage
python3 -m pytest --cov=gpumemprof --cov=tfmemprof --cov-report=html

# Run specific test file
python3 -m pytest tests/test_core_profiler.py -v

# Run GPU tests only (if CUDA available)
python3 -m pytest -m "gpu"

# Run CPU tests only
python3 -m pytest -m "cpu"
```

### Writing Tests

-   Write tests for all new functionality
-   Use descriptive test names
-   Test both success and failure cases
-   Mock external dependencies
-   Ensure tests work on both GPU and CPU systems

### Test Structure

```
tests/
├── conftest.py                    # Test configuration and fixtures
├── test_core_profiler.py          # Core profiling functionality
├── test_cpu_profiler.py           # CPU-mode profiler tests
├── test_cli_diagnose.py           # gpumemprof diagnose CLI tests
├── test_cli_info.py               # gpumemprof info CLI tests
├── test_device_collectors.py      # Backend device collector tests
├── test_gap_analysis.py           # Gap analysis tests
├── test_oom_flight_recorder.py    # OOM flight-recorder tests
├── test_profiler.py               # Profiler integration tests
├── test_telemetry_v2.py           # Telemetry schema tests
├── test_utils.py                  # Utility function tests
├── tui/                           # TUI component tests
└── e2e/                           # End-to-end smoke tests
```

## Pull Request Process

1. **Fork and clone** the repository
2. **Create a feature branch** from `main`
3. **Make your changes** following the code style guidelines
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run tests** and ensure they pass
7. **Commit your changes** with conventional commit messages
8. **Push to your fork** and create a pull request
9. **Wait for review** and address any feedback

### Pull Request Guidelines

-   Provide a clear description of the changes
-   Include any relevant issue numbers
-   Add screenshots for UI changes
-   Update documentation if needed
-   Ensure all CI checks pass

## Reporting Bugs

### Before Submitting

1. Check if the bug has already been reported
2. Try to reproduce the bug with the latest version
3. Check if the bug is related to your system configuration

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**System Information**

-   OS: [e.g. Ubuntu 20.04]
-   Python version: [e.g. 3.9.0]
-   PyTorch version: [e.g. 1.12.0]
-   TensorFlow version: [e.g. 2.9.0]
-   CUDA version: [e.g. 11.6]

**Additional context**
Add any other context about the problem here.
```

## Feature Requests

### Before Submitting

1. Check if the feature has already been requested
2. Consider if the feature aligns with the project's goals
3. Think about the implementation complexity

### Feature Request Template

```markdown
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## Getting Help

-   **GitHub Issues**: For bug reports and feature requests
-   **GitHub Discussions**: For questions and general discussion
-   **Documentation**: Check the [docs](docs/) for usage information

## License

By contributing to GPU Memory Profiler, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to GPU Memory Profiler! 🚀
