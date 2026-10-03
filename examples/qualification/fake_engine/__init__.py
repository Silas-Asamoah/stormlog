"""A fault-injectable fake vLLM 0.30.0 server for CPU tests.

It simulates continuous batching (a waiting queue, ``max_num_seqs``, chunked
prefill, KV blocks with recompute preemption, a prefix cache) and serves the
routes and records Stormlog reads from a real server. See
``docs/testing.md`` ("Fake vLLM engine").
"""

from .config import Controls, FakeEngineConfig
from .engine import Engine, EngineObserver, FakeRequest, ScheduledMember, Step
from .process import FakeEngineProcess
from .server import FakeEngine

__all__ = [
    "Controls",
    "Engine",
    "EngineObserver",
    "FakeEngine",
    "FakeEngineConfig",
    "FakeEngineProcess",
    "FakeRequest",
    "ScheduledMember",
    "Step",
]
