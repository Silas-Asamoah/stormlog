"""vLLM general-plugin entry point for Stormlog's execution hook.

vLLM imports every installed general plugin in its engine-core and worker
processes, so this module must stay light: it imports nothing beyond the
standard library until ``STORMLOG_VLLM_HOOK_DIR`` is set, and it never raises
into vLLM. See ``docs/vllm_execution.md``.
"""

from __future__ import annotations

import os
import sys


def register() -> None:
    """Install the hook when ``STORMLOG_VLLM_HOOK_DIR`` is set; else do nothing."""
    if not os.environ.get("STORMLOG_VLLM_HOOK_DIR"):
        return
    try:
        from stormlog.infer.vllm_hook import install

        install()
    except Exception as exc:  # the hook must never stop vLLM from starting
        print(f"stormlog vLLM hook disabled: {exc!r}", file=sys.stderr)
