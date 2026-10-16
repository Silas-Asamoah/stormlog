import json
import subprocess
import sys

from stormlog.mlx.collector import MLXCollector
from stormlog.session import create_session_summary
from tests.mlx_fakes import make_runtime


def test_package_classes_offline_and_decorators_never_import_frameworks(tmp_path):
    profile = tmp_path / "profiles.json"
    profile.write_text(
        json.dumps(
            {"format": "stormlog.mlx.profile", "schema_version": 1, "profiles": []}
        )
    )
    telemetry = tmp_path / "telemetry.json"
    collector = MLXCollector(make_runtime())
    record = collector.telemetry_record(
        collector.capture_snapshot(), create_session_summary(source="test")
    )
    telemetry.write_text(json.dumps([record]))
    code = """
import importlib.abc
import sys
attempts = []
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'jax', 'tensorflow'}:
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockFrameworks())
import stormlog
import stormlog.mlx as package
for name in package.__all__:
    getattr(package, name)
from stormlog.mlx.profile_artifact import load_profiles
assert load_profiles(sys.argv[1])['profiles'] == []
from stormlog.telemetry import load_telemetry_events
assert load_telemetry_events(sys.argv[2])[0].metadata['framework'] == 'mlx'
from stormlog.mlx.context_profiler import profile_function
@profile_function
def decorated():
    return None
from stormlog.mlx.tracker import MemoryTracker
from stormlog.oom_flight_recorder import OOMFlightRecorder
from stormlog.system_info import get_system_info
get_system_info()
from tests.mlx_fakes import make_runtime
tracker = MemoryTracker(runtime=make_runtime(), sampling_interval=10,
                        enable_oom_flight_recorder=True, oom_dump_dir=sys.argv[3])
tracker.start_tracking()
assert tracker.record_exception(RuntimeError('[malloc] Unable to allocate 4096 bytes.'))
tracker.stop_tracking()
assert not attempts, attempts
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(profile),
            str(telemetry),
            str(tmp_path / "oom"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
