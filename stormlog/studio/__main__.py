"""``python -m stormlog.studio``: the same as ``stormlog studio``."""

from __future__ import annotations

import sys

from stormlog.studio.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
