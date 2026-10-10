import json

import pytest

from stormlog.mlx.oom import classify_oom_exception
from stormlog.mlx.tracker import MemoryTracker
from tests.mlx_fakes import make_runtime


@pytest.mark.parametrize(
    "message",
    [
        "Metal compiler failed",
        "[metal::malloc] Resource limit (10) exceeded.",
        "[metal::malloc] Attempting to allocate 999 bytes which is greater than the maximum allowed buffer size of 10 bytes.",
        "generic out of memory",
        "[malloc] Unable to allocate 3 bytes. unrelated",
    ],
)
def test_non_oom_signatures(message):
    assert not classify_oom_exception(RuntimeError(message)).is_oom


def test_known_failure_and_bounded_bundle(tmp_path):
    tracker = MemoryTracker(
        runtime=make_runtime(),
        sampling_interval=10,
        enable_oom_flight_recorder=True,
        oom_dump_dir=str(tmp_path),
        oom_buffer_size=2,
        oom_max_dumps=1,
    )
    tracker.start_tracking()
    error = RuntimeError("[malloc] Unable to allocate 4096 bytes.")
    assert classify_oom_exception(error).is_oom
    path = tracker.record_exception(error)
    assert path is not None
    tracker.record_exception(error)
    tracker.stop_tracking()
    bundles = list(tmp_path.iterdir())
    assert len(bundles) == 1
    events = json.loads((bundles[0] / "events.json").read_text())
    assert len(events) <= 2
    metadata = json.loads((bundles[0] / "metadata.json").read_text())
    assert metadata["exception_message"] == str(error)
