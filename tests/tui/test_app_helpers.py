from __future__ import annotations

import pytest

pytest.importorskip("textual")

from gpumemprof.tui import app as appmod


def test_build_system_markdown_uses_system_info_fallback_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _raise_system_info() -> dict[str, object]:
        raise RuntimeError("system info unavailable")

    def _fake_build_system_markdown(
        *,
        system_info: dict[str, object],
        gpu_info: dict[str, object],
        tf_system_info: dict[str, object],
        tf_gpu_info: dict[str, object],
    ) -> str:
        captured["system_info"] = system_info
        captured["gpu_info"] = gpu_info
        captured["tf_system_info"] = tf_system_info
        captured["tf_gpu_info"] = tf_gpu_info
        return "ok"

    monkeypatch.setattr(appmod, "get_system_info", _raise_system_info)
    monkeypatch.setattr(appmod, "_safe_get_gpu_info", lambda: {})
    monkeypatch.setattr(appmod, "_safe_get_tf_system_info", lambda: {})
    monkeypatch.setattr(appmod, "_safe_get_tf_gpu_info", lambda: {})
    monkeypatch.setattr(
        appmod.tui_builders, "build_system_markdown", _fake_build_system_markdown
    )

    assert appmod._build_system_markdown() == "ok"
    assert captured["system_info"] == {}
