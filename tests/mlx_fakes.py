"""Small deterministic native-API fake; no framework dependencies."""

from collections import deque
from types import SimpleNamespace

from stormlog.mlx.runtime import MLXRuntime


class FakeCore:
    gpu = "gpu"

    def __init__(self):
        self.calls = []
        self.values = {
            "active_memory": 100,
            "cache_memory": 20,
            "peak_memory": 1000,
            "memory_limit": 2000,
        }
        self.queued = {}
        self.metal = SimpleNamespace(is_available=lambda: True)
        self.outputs = []
        self.sync_error = None

    def _get(self, name):
        self.calls.append(name)
        queue = self.queued.get(name)
        value = queue.popleft() if queue else self.values[name]
        if isinstance(value, BaseException):
            raise value
        return value

    def queue(self, name, values):
        self.queued[name] = deque(values)

    def get_active_memory(self):
        return self._get("active_memory")

    def get_cache_memory(self):
        return self._get("cache_memory")

    def get_peak_memory(self):
        return self._get("peak_memory")

    def get_memory_limit(self):
        return self._get("memory_limit")

    def device_info(self, device):
        self.calls.append(("device_info", device))
        return {"device_name": "Fake Metal", "max_recommended_working_set_size": 5000}

    def default_stream(self, device):
        self.calls.append(("default_stream", device))
        return "default"

    def eval(self, *roots):
        self.calls.append("eval")
        # Do not retain objects; real profiling must not retain them either.
        self.outputs.append(tuple(id(root) for root in roots))
        self.values["active_memory"] = 500
        self.values["peak_memory"] = max(500, self.values["peak_memory"])

    def synchronize(self, stream):
        self.calls.append(("sync", stream))
        if self.sync_error:
            raise self.sync_error

    def reset_peak_memory(self):
        self.calls.append("reset")
        self.values["peak_memory"] = 0


def make_runtime(core=None):
    return MLXRuntime(
        core=core or FakeCore(),
        runtime_version="0.32.3",
        platform_system="Darwin",
        platform_machine="arm64",
    )
