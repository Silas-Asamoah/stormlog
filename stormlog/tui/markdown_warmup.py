"""Build markdown-it's URL caches before Textual parses Markdown in threads.

Textual parses each ``Markdown`` widget's document in the default thread
pool, so panels that update together parse concurrently. markdown-it
normalizes link URLs with ``mdurl.encode`` and autolink text with
``mdurl.decode``. mdurl 0.1.2 builds the lookup table for each ``exclude``
string lazily and stores the empty list in its module-level cache before
filling it, so a second thread can index a half-built table (``IndexError``
in ``mdurl._encode``) or read a decode table whose excluded characters are
not patched in yet (a wrongly decoded link).

Calling markdown-it's own normalizers once on the main thread builds the
tables for exactly the ``exclude`` values markdown-it passes; afterwards
every thread only reads them. They are reached through ``MarkdownIt``'s
public methods, which every parser calls, so no private module is imported.
"""

from __future__ import annotations

from markdown_it import MarkdownIt


def warm_markdown_url_caches() -> None:
    """Fill mdurl's encode and decode tables that markdown-it reads."""
    parser = MarkdownIt()
    parser.normalizeLink("")
    parser.normalizeLinkText("")
