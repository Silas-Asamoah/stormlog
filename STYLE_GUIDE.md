# Python Style Guide

This document defines the coding standards for the GPU Memory Profiler project.

## Automated Tooling

We use automated tools to enforce consistent code style. All tools are configured in `pyproject.toml` and `.flake8`.

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **black** | Code formatting | `pyproject.toml` |
| **isort** | Import sorting | `pyproject.toml` (profile=black) |
| **flake8** | Linting | `.flake8` |
| **mypy** | Type checking | `pyproject.toml` |

## Quick Start

### Installing Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This installs hooks that automatically check your code before each commit.

### Running Style Checks Manually

```bash
# Format code (modifies files)
black gpumemprof/ tfmemprof/ tests/ examples/
isort gpumemprof/ tfmemprof/ tests/ examples/

# Check without modifying
black --check gpumemprof/ tfmemprof/ tests/ examples/
isort --check gpumemprof/ tfmemprof/ tests/ examples/
flake8 gpumemprof/ tfmemprof/ tests/ examples/
mypy gpumemprof/ tfmemprof/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Code Style Guidelines

### Line Length

Maximum line length is **88 characters** (Black default). This is enforced by Black for code, while long strings are allowed to exceed this limit for readability.

### Imports

Imports are organized by isort with the "black" profile:

1. `__future__` imports
2. Standard library imports
3. Third-party imports
4. Local application imports

Each group is separated by a blank line. Example:

```python
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from gpumemprof.profiler import GPUMemoryProfiler
from gpumemprof.utils import format_bytes
```

### Conditional Imports

For optional dependencies (torch, tensorflow), use try/except patterns:

```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

### Type Hints

All public functions and methods should have type hints:

```python
def process_data(data: List[float], threshold: float = 0.5) -> Dict[str, Any]:
    """Process input data and return results."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """Short description of the function.

    Longer description if needed, explaining the function's purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.
    """
```

### Naming Conventions

- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Private members**: `_leading_underscore`
- **Unused variables**: `_variable` (underscore prefix)

Avoid ambiguous single-letter names like `l`, `O`, `I`. Use descriptive names instead.

### F-strings

Use f-strings for string formatting, but only when placeholders are present:

```python
# Good
name = "World"
print(f"Hello, {name}!")

# Bad - no placeholders, use regular string
print(f"Hello, World!")  # Should be: print("Hello, World!")
```

## Special Patterns

### Lazy Loading in `__init__.py`

The `gpumemprof/__init__.py` uses lazy loading via `__getattr__` for performance. This pattern is intentional - do not convert to eager imports.

### Environment Setup Before Imports

Some files set environment variables before imports:

```python
from .tf_env import configure_tensorflow_logging

configure_tensorflow_logging()

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
```

This pattern is necessary for proper TensorFlow configuration and is documented via per-file-ignores in `.flake8`.

## Suppressing Lint Errors

When suppression is necessary, use inline comments with an explanation:

```python
import unused_but_needed  # noqa: F401  # Re-exported in __all__
```

Per-file ignores are configured in `.flake8` for legitimate patterns like:
- `__init__.py` files with re-exports (F401, E402)
- Example files with conditional imports (E402, F401, F811)
- Test files with side-effect variables (F841)

## CI Integration

The CI pipeline (`.github/workflows/ci.yml`) runs the following checks:

1. **isort** - Import sorting check
2. **black** - Code formatting check
3. **flake8** - Linting check
4. **mypy** - Type checking

All checks must pass for PRs to be merged.

## Resources

- [Black documentation](https://black.readthedocs.io/)
- [isort documentation](https://pycqa.github.io/isort/)
- [flake8 documentation](https://flake8.pycqa.org/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
