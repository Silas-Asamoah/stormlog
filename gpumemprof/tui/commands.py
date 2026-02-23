"""Asynchronous CLI command runner for the Textual TUI."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional


class CLICommandRunner:
    """Runs shell commands asynchronously while streaming output."""

    def __init__(self) -> None:
        self._process: Optional[asyncio.subprocess.Process] = None
        self._io_tasks: list[asyncio.Task] = []

    @property
    def is_running(self) -> bool:
        return self._process is not None

    async def run(
        self,
        command: str,
        callback: Callable[[str, str], Awaitable[None]],
    ) -> int:
        """Run a command and stream stdout/stderr via callback."""
        if self._process:
            raise RuntimeError("Another command is already running.")

        self._process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def pump(stream: Optional[asyncio.StreamReader], label: str) -> None:
            if not stream:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                await callback(
                    label, line.decode("utf-8", errors="replace").rstrip("\n")
                )

        stdout_task = asyncio.create_task(pump(self._process.stdout, "stdout"))
        stderr_task = asyncio.create_task(pump(self._process.stderr, "stderr"))
        self._io_tasks = [stdout_task, stderr_task]

        return_code = await self._process.wait()
        await asyncio.gather(*self._io_tasks, return_exceptions=True)
        self._cleanup()
        return return_code

    async def cancel(self) -> bool:
        """Attempt to terminate the running command."""
        if not self._process:
            return False

        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=3)
        except asyncio.TimeoutError:
            self._process.kill()
            await self._process.wait()

        await asyncio.gather(*self._io_tasks, return_exceptions=True)
        self._cleanup()
        return True

    def _cleanup(self) -> None:
        self._process = None
        self._io_tasks = []
