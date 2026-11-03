"""The TUI builds mdurl's URL tables before Textual parses Markdown in threads.

mdurl stores each lazily built table in a module-level dict before filling
it, so two Markdown widgets parsing at once in Textual's thread pool could
read a half-built table. These checks are deterministic: they look at which
tables exist, not at thread timing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

import mdurl._decode
import mdurl._encode
from markdown_it import MarkdownIt

from stormlog.tui.app import GPUMemoryProfilerTUI
from stormlog.tui.markdown_warmup import warm_markdown_url_caches

# A document that sends every link form through markdown-it's normalizers.
LINKS = (
    "[a link](https://example.com/a b?q=%3B) and ![an image](img/x y.png)\n\n"
    "<https://example.com/%3B%2F> and <mailto:someone@example.com>\n\n"
    "[ref]\n\n[ref]: https://example.com/ref 'title'\n"
)


@pytest.fixture
def cold_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mdurl._encode, "encode_cache", {})
    monkeypatch.setattr(mdurl._decode, "decode_cache", {})


def _tables() -> dict[str, dict[str, list[str]]]:
    return {
        "encode": dict(mdurl._encode.encode_cache),
        "decode": dict(mdurl._decode.decode_cache),
    }


@pytest.mark.usefixtures("cold_caches")
def test_the_warm_up_builds_every_table_markdown_parsing_reads() -> None:
    warm_markdown_url_caches()
    warmed = _tables()
    # Textual's default parser, on links, images, autolinks and references.
    MarkdownIt("gfm-like").parse(LINKS)
    # Parsing found every table it needed already built, and complete.
    assert _tables() == warmed
    assert set(warmed["encode"]) == {mdurl.ENCODE_DEFAULT_CHARS}
    assert set(warmed["decode"]) == {mdurl.DECODE_DEFAULT_CHARS + "%"}
    assert all(len(table) == 128 for kind in warmed.values() for table in kind.values())


@pytest.mark.usefixtures("cold_caches")
@pytest.mark.parametrize("framework", ["pytorch", "mlx"])
def test_the_app_builds_the_tables_before_it_mounts_anything(framework: str) -> None:
    GPUMemoryProfilerTUI(framework=framework)
    tables = _tables()
    assert set(tables["encode"]) == {mdurl.ENCODE_DEFAULT_CHARS}
    assert set(tables["decode"]) == {mdurl.DECODE_DEFAULT_CHARS + "%"}
