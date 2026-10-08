"""Pytest markers more than one test file needs."""

import os

import pytest

# Permissions do not bind root: a directory chmodded read-only is still
# writable to it, as in a CI container that runs as uid 0, so a test of
# that refusal cannot fail there.
needs_non_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root can write a read-only directory",
)
