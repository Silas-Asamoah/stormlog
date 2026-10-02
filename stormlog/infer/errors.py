"""Errors that tell the inference CLI which contract exit code a failure gets.

Both subclass ``ValueError``, so code that catches ``ValueError`` from the
library keeps working.
"""

from __future__ import annotations


class InferUsageError(ValueError):
    """A setting the command cannot use; the CLI exits ``USAGE`` (2)."""


class InferInputError(ValueError):
    """An input file the command cannot read; the CLI exits ``INVALID_INPUT`` (5)."""


class TokenizerUnavailableError(InferUsageError, RuntimeError):
    """Strict token counts were requested but no tokenizer backend loads.

    A usage error for the CLI (exit 2); it is still a ``RuntimeError``, which
    is what ``build_token_counter`` raised before, so existing callers that
    catch that keep working.
    """
