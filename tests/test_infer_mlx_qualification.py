"""Explicit opt-in live matrix: requested qualification fails missing setup."""

import json
import os
import platform
from pathlib import Path

import pytest

from examples.scenarios.mlx_inference_scenario import parser, run

pytestmark = pytest.mark.infer_mlx_qualification


def _configuration(request):
    path = os.environ.get("STORMLOG_INFER_MLX_QUALIFICATION_CONFIG")
    if not path:
        if "infer_mlx_qualification" in request.config.option.markexpr:
            pytest.fail(
                "Requested MLX inference qualification requires STORMLOG_INFER_MLX_QUALIFICATION_CONFIG; no simulated fallback"
            )
        pytest.skip("Live endpoint qualification is explicitly opt in")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.fail("Native qualification requires Apple Silicon macOS")
    from stormlog.mlx.runtime import MLXRuntime

    MLXRuntime()  # Prerequisite failure is a failure, not an all-skipped pass.
    configuration = json.loads(Path(path).read_text())
    if set(configuration) != {"omlx", "mlx-lm", "mlx-audio"}:
        pytest.fail("Provide all three explicitly provisioned endpoint configurations")
    return configuration


@pytest.mark.parametrize("engine", ["omlx", "mlx-lm", "mlx-audio"])
def test_live_engine_matrix(request, engine):
    configuration = _configuration(request)
    arguments = configuration[engine]
    if not isinstance(arguments, list) or not all(
        isinstance(arg, str) for arg in arguments
    ):
        pytest.fail(
            "Each engine configuration must be a list of scenario CLI arguments"
        )
    args = parser().parse_args(["--engine", engine, *arguments])
    result = run(args)
    for row in result["results"]:
        if row["name"].endswith("-rate"):
            assert row["summary"]["successful_requests"] > 0
        else:
            assert row["summary"]["successful_requests"] == args.requests * 2
            assert row["summary"]["failed_requests"] == 0
        if row["name"].endswith("-stats"):
            stats = row["telemetry"]["mlx"]["omlx_stats"]
            assert stats["samples"] >= 2 and stats["failed_samples"] == 0
            assert any(
                metric["valid_samples"] > 0 for metric in stats["metrics"].values()
            )
    if engine == "omlx" and not args.omlx_stats_url:
        pytest.fail(
            "oMLX qualification requires an explicit read-only stats endpoint and separately supplied normal admin credential"
        )
