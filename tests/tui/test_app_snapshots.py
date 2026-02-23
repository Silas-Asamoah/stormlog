from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("textual")
from textual.widgets import Header as TextualHeader
from textual.widgets import TabbedContent, TabPane

from gpumemprof.tui import app as appmod
from gpumemprof.tui.app import GPUMemoryProfilerTUI

pytestmark = pytest.mark.tui_snapshot

TERMINAL_SIZE = (140, 44)


def _configure_snapshot_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appmod, "WELCOME_MESSAGES", ["GPU Memory Profiler"])
    monkeypatch.setattr(appmod, "Figlet", None)
    monkeypatch.setattr(
        appmod,
        "get_system_info",
        lambda: {
            "platform": "Darwin-23.6.0-arm64",
            "python_version": "3.10.17",
            "cuda_available": False,
            "cuda_version": "N/A",
            "cuda_device_count": 0,
        },
    )
    monkeypatch.setattr(
        appmod,
        "_safe_get_gpu_info",
        lambda: {
            "device_name": "Snapshot GPU",
            "total_memory": 16 * (1024**3),
            "allocated_memory": 2 * (1024**3),
            "reserved_memory": 3 * (1024**3),
            "max_memory_allocated": 4 * (1024**3),
        },
    )
    monkeypatch.setattr(
        appmod,
        "_safe_get_tf_system_info",
        lambda: {"tensorflow_version": "2.15.0"},
    )
    monkeypatch.setattr(
        appmod,
        "_safe_get_tf_gpu_info",
        lambda: {
            "devices": [
                {
                    "name": "TF Snapshot GPU",
                    "current_memory_mb": 512.0,
                    "peak_memory_mb": 1024.0,
                }
            ],
            "total_memory": 2048.0,
        },
    )
    monkeypatch.setattr(appmod, "fetch_pytorch_profiles", lambda limit=15: [])
    monkeypatch.setattr(appmod, "fetch_tensorflow_profiles", lambda limit=15: [])
    monkeypatch.setattr(
        appmod.GPUMemoryProfilerTUI, "set_interval", lambda *args, **kwargs: None
    )

    class SnapshotHeader(TextualHeader):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["show_clock"] = False
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(appmod, "Header", SnapshotHeader)


@pytest.fixture(autouse=True)  # type: ignore[misc, unused-ignore]
def _deterministic_snapshot_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_snapshot_overrides(monkeypatch)


def _pane_title(pane: TabPane) -> str:
    title = getattr(pane, "title", None)
    if title is None:
        title = getattr(pane, "_title", "")
    if hasattr(title, "plain"):
        title = title.plain  # type: ignore[union-attr, unused-ignore]
    return str(title)


def _tab_id_by_title(app: GPUMemoryProfilerTUI, title: str) -> str:
    for pane in app.query(TabPane):
        if _pane_title(pane) == title:
            assert pane.id is not None
            return str(pane.id)
    raise AssertionError(f"Tab not found: {title}")


def _assert_svg_rendered(contents: str) -> None:
    assert "<svg" in contents
    assert len(contents) > 1000


def _capture_tab_svg(
    *,
    tab_title: str,
    visible_selectors: tuple[str, ...],
) -> None:
    async def scenario() -> None:
        app = GPUMemoryProfilerTUI()
        async with app.run_test(headless=True, size=TERMINAL_SIZE) as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = _tab_id_by_title(app, tab_title)
            await pilot.pause()
            for selector in visible_selectors:
                assert app.query_one(selector).display is True
            screenshot = app.export_screenshot()
            await pilot.pause()
            _assert_svg_rendered(screenshot)

    asyncio.run(scenario())


def test_snapshot_overview_tab() -> None:
    _capture_tab_svg(
        tab_title="Overview",
        visible_selectors=("#overview-welcome",),
    )


def test_snapshot_pytorch_tab() -> None:
    _capture_tab_svg(
        tab_title="PyTorch",
        visible_selectors=("#pytorch-profile-controls",),
    )


def test_snapshot_tensorflow_tab() -> None:
    _capture_tab_svg(
        tab_title="TensorFlow",
        visible_selectors=("#tensorflow-profile-controls",),
    )


def test_snapshot_monitoring_tab() -> None:
    _capture_tab_svg(
        tab_title="Monitoring",
        visible_selectors=(
            "#monitor-controls-row1",
            "#monitor-controls-row2",
            "#monitor-controls-row3",
        ),
    )


def test_snapshot_visualizations_tab() -> None:
    _capture_tab_svg(
        tab_title="Visualizations",
        visible_selectors=("#visual-buttons",),
    )


def test_snapshot_cli_actions_tab() -> None:
    _capture_tab_svg(
        tab_title="CLI & Actions",
        visible_selectors=("#cli-buttons-row1", "#cli-buttons-row2"),
    )
