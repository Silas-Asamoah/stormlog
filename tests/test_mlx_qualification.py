"""Qualification distinguishes synthetic contracts, native gates and optional targets."""

import json

import pytest

from examples.cli.capability_matrix import run_matrix
from examples.scenarios import mlx_memory_scenario as scenario
from stormlog.mlx.runtime import MLXMetalUnavailableError


def test_synthetic_qualification_roundtrips_all_report_outcomes(tmp_path):
    result = scenario.run_scenario(output_dir=tmp_path / "fake")
    assert result["hardware"] is False
    assert "measurements" not in result
    assert result["roundtrip"]["total_samples"] > 3
    assert result["roundtrip"]["retained_samples"] == 3
    assert [r["exit_code"] for r in result["failure_roundtrips"]] == [1, 1, 3, 130]
    assert result["failure_roundtrips"][-1]["session_status"] == "interrupted"
    saved = json.loads((tmp_path / "fake/qualification.json").read_text())
    assert saved["source"]["source_sha256"]


def test_required_hardware_gate_fails_when_metal_is_unavailable(tmp_path, monkeypatch):
    def unavailable():
        raise MLXMetalUnavailableError("Metal absent")

    monkeypatch.setattr(scenario, "MLXRuntime", unavailable)
    with pytest.raises(MLXMetalUnavailableError, match="Metal absent"):
        scenario.run_scenario(output_dir=tmp_path / "native", hardware=True)
    assert not (tmp_path / "native/qualification.json").exists()


@pytest.mark.parametrize("outcome", ["PASS", "SKIP", "FAIL"])
def test_optional_matrix_has_truthful_capability_status(tmp_path, monkeypatch, outcome):
    def run(**kwargs):
        assert kwargs["hardware"] is True
        if outcome == "SKIP":
            raise MLXMetalUnavailableError("Metal absent")
        if outcome == "FAIL":
            raise AssertionError("broken contract")
        return {"status": "PASS"}

    monkeypatch.setattr(scenario, "run_scenario", run)
    report = run_matrix(
        mode="smoke",
        target="mlx",
        oom_mode="simulated",
        artifacts_root=tmp_path,
        skip_tui=True,
    )
    assert report["checks"][0]["status"] == outcome
    if outcome == "SKIP":
        assert "Metal absent" in report["checks"][0]["details"]["reason"]
