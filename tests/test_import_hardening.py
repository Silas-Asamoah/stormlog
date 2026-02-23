import subprocess
import sys
import textwrap


def test_imports_are_hardened_when_torch_is_missing() -> None:
    code = textwrap.dedent(
        """
        import builtins
        from types import SimpleNamespace

        original_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        import gpumemprof
        import gpumemprof.cli as gpumemprof_cli
        import gpumemprof.utils as gpumemprof_utils

        # Import should remain usable for non-torch helpers.
        assert gpumemprof.format_bytes(1024) == "1.00 KB"
        system_info = gpumemprof_utils.get_system_info()
        assert system_info["cuda_available"] is False
        assert system_info["detected_backend"] == "cpu"

        gpumemprof_cli.cmd_info(SimpleNamespace(device=None, detailed=False))
        gpumemprof_cli.cmd_monitor(
            SimpleNamespace(
                device=None,
                duration=0.0,
                interval=0.01,
                output=None,
                format="json",
            )
        )
        gpumemprof_cli.cmd_track(
            SimpleNamespace(
                device=None,
                duration=1e-9,
                interval=0.01,
                output=None,
                format="json",
                watchdog=False,
                warning_threshold=80.0,
                critical_threshold=95.0,
                oom_flight_recorder=False,
                oom_dump_dir="oom_dumps",
                oom_buffer_size=None,
                oom_max_dumps=5,
                oom_max_total_mb=256,
            )
        )

        try:
            gpumemprof_utils.get_gpu_info()
        except ImportError as exc:
            assert "gpu-memory-profiler[torch]" in str(exc)
        else:
            raise AssertionError("Expected get_gpu_info to fail lazily without torch")

        try:
            _ = gpumemprof.GPUMemoryProfiler
        except ImportError as exc:
            assert "gpu-memory-profiler[torch]" in str(exc)
        else:
            raise AssertionError("Expected GPUMemoryProfiler symbol load to fail lazily without torch")

        print("ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
