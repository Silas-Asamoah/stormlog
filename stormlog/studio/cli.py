"""``stormlog studio``: the launcher (design §2.4).

``stormlog studio [PATH]`` binds Studio to loopback, prints the root and a
one-time sign-in URL, optionally opens a browser on it, and serves until
SIGINT or SIGTERM. ``stormlog studio url [ROUTE]`` prints a fresh sign-in
URL for a Studio already running, using the admin token in its state file.

Exit codes: 0 a normal stop; 2 bad flags, no free port, no running
instance, or an unsupported platform (Windows, decision 10); 5 a ``PATH``
missing or unreadable; 1 anything unexpected.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import html
import importlib
import json
import os
import signal
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType
from typing import Any, TextIO
from urllib.parse import quote

from stormlog.studio.contracts import (
    Composer,
    LaunchOptions,
    Limits,
    Route,
    StudioContext,
)
from stormlog.studio.httpd import StudioServer
from stormlog.studio.security import (
    BOOTSTRAP_TTL_S,
    SessionTokens,
    launch_url,
    refuse_broad_root,
)
from stormlog.studio.static_files import StaticFiles

__all__ = [
    "DEFAULT_PORT",
    "PORT_FALLBACKS",
    "LOOPBACK_HOSTS",
    "EXIT_OK",
    "EXIT_ERROR",
    "EXIT_USAGE",
    "EXIT_INPUT",
    "StudioUsageError",
    "main",
    "state_directory",
    "build_options",
]

DEFAULT_PORT = 8790
# Decision 15: 8790, then the next 9.
PORT_FALLBACKS = 9
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INPUT = 5
_URL_TIMEOUT_S = 10.0
# A health or bootstrap document is a few hundred bytes.
_URL_MAX_BYTES = 64 * 2**10
# C4's composition root, imported only when no composer is given.
APP_MODULE = "stormlog.studio.app"
_WINDOWS_MESSAGE = (
    "stormlog studio runs on macOS and Linux only; on Windows, run it in WSL."
)


class StudioUsageError(Exception):
    """A launch that cannot go ahead; carries the exit code."""

    def __init__(self, message: str, code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.code = code


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        raise StudioUsageError(f"{self.prog}: {message}")


def main(
    argv: Sequence[str],
    *,
    compose: Composer | None = None,
    stop: threading.Event | None = None,
    open_browser: Callable[[str], object] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run ``stormlog studio`` with ``argv`` (without the ``studio`` word).

    ``compose`` builds the context and routes from the launch options; by
    default it is ``stormlog.studio.app``'s, imported only here. Tests pass
    a stub, ``stop`` to end the serve loop and ``open_browser`` to see what
    would open.
    """
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    if sys.platform.startswith("win"):
        print(_WINDOWS_MESSAGE, file=err)
        return EXIT_USAGE
    try:
        if list(argv[:1]) == ["url"]:
            return _url_main(list(argv[1:]), out)
        options = build_options(_parse(argv))
        return _serve(options, compose, stop, open_browser, out, err)
    except StudioUsageError as exc:
        print(str(exc), file=err)
        return exc.code


# --- Arguments and options ---------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="stormlog studio",
        description=(
            "Serve Stormlog Studio on loopback for the artifacts under one "
            "directory. `stormlog studio url [ROUTE]` prints a fresh sign-in "
            "link for a running Studio."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="an artifact to open, or the directory to serve (default: .)",
    )
    parser.add_argument("--root", help="the workspace root; required for / or ~")
    parser.add_argument("--host", default="127.0.0.1", help="a loopback address")
    parser.add_argument("--port", type=_port, default=None, help="0 picks a port")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--analysis-max-bytes", type=_positive, default=None)
    parser.add_argument("--log-requests", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    return _parser().parse_args(list(argv))


def _port(text: str) -> int:
    if not text.isdigit() or int(text) > 65535:
        raise argparse.ArgumentTypeError(f"not a port: {text!r}")
    return int(text)


def _positive(text: str) -> int:
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"not a positive integer: {text!r}")
    return int(text)


def state_directory() -> Path:
    """``$XDG_STATE_HOME/stormlog/studio``, else ``~/.local/state/...``."""
    base = os.environ.get("XDG_STATE_HOME") or ""
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "stormlog" / "studio"


def build_options(args: argparse.Namespace) -> LaunchOptions:
    """The launch options for parsed ``args``; StudioUsageError when refused."""
    if args.host not in LOOPBACK_HOSTS:
        raise StudioUsageError(
            f"--host must be one of {', '.join(LOOPBACK_HOSTS)}; "
            "for remote use, tunnel with ssh -L (see docs/studio.md)"
        )
    root, open_path = _root_and_path(args.path, args.root)
    limits = Limits()
    if args.analysis_max_bytes is not None:
        limits = dataclasses.replace(limits, analysis_bytes=args.analysis_max_bytes)
    return LaunchOptions(
        root=root,
        open_path=open_path,
        host=args.host,
        port=DEFAULT_PORT if args.port is None else args.port,
        open_browser=not args.no_browser,
        state_dir=state_directory(),
        log_requests=args.log_requests,
        debug=args.debug,
        limits=limits,
    )


def _root_and_path(path: str | None, root_flag: str | None) -> tuple[Path, str | None]:
    target = Path(path).expanduser() if path else None
    if target is not None and not os.access(target, os.R_OK):
        raise StudioUsageError(f"cannot read {path}", EXIT_INPUT)
    root = Path(os.path.realpath(_chosen_root(target, root_flag)))
    if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
        raise StudioUsageError(f"cannot read the directory {root}", EXIT_INPUT)
    refused = refuse_broad_root(root, named=root_flag is not None)
    if refused is not None:
        raise StudioUsageError(
            f"refusing to serve {refused} ({root}): every JSONL file under it "
            f"would be readable through Studio. Pass --root {root} to mean it."
        )
    return root, _open_path(root, target)


def _chosen_root(target: Path | None, root_flag: str | None) -> Path:
    """``--root``; else an artifact's directory, a directory, or the cwd."""
    if root_flag is not None:
        return Path(root_flag).expanduser()
    if target is None:
        return Path.cwd()
    return target if target.is_dir() else target.parent


def _open_path(root: Path, target: Path | None) -> str | None:
    if target is None:
        return None
    real = Path(os.path.realpath(target))
    if real == root:
        return None
    try:
        relative = real.relative_to(root)
    except ValueError:
        raise StudioUsageError(f"{target} is not under the root {root}") from None
    if real.is_dir():
        raise StudioUsageError(f"{target} is a directory under --root; omit it")
    return relative.as_posix()


# --- Serving -----------------------------------------------------------------


def _default_compose(options: LaunchOptions) -> tuple[StudioContext, list[Route]]:
    try:
        app = importlib.import_module(APP_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != APP_MODULE:
            raise
        raise StudioUsageError(
            "this build of Stormlog has no Studio API yet (stormlog.studio.app)",
            EXIT_ERROR,
        ) from None
    ctx: StudioContext = app.build_context(options)
    routes: list[Route] = app.routes(ctx)
    return ctx, routes


def _serve(
    options: LaunchOptions,
    compose: Composer | None,
    stop: threading.Event | None,
    opener: Callable[[str], object] | None,
    out: TextIO,
    err: TextIO,
) -> int:
    state = _ensure_private_dir(options.state_dir)
    ctx, routes = (compose or _default_compose)(options)
    stop = stop or threading.Event()
    # Before the state file and the "Open" line: a caller may signal as
    # soon as it reads that line, and must get a clean stop, not a kill.
    restore = _install_signals(stop)
    server: StudioServer | None = None
    files: list[Path] = []
    try:
        server = _bind(options, ctx, routes, err)
        files.append(_write_state(state, server, ctx, options))
        bootstrap = ctx.tokens.issue_bootstrap(ttl_s=BOOTSTRAP_TTL_S)
        url = launch_url(server.base_url, bootstrap, _open_route(options.open_path))
        print(f"Stormlog Studio is serving {options.root}", file=out)
        print(f"Open {url}", file=out, flush=True)
        launch = _launch(state, ctx, url, options, opener or webbrowser.open)
        if launch is not None:
            files.append(launch)
        _wait(stop, ctx, bootstrap, launch)
    finally:
        restore()
        if server is not None:
            server.stop()
        _close_context(ctx, compose is None)
        for path in files:
            _remove(path)
    return EXIT_OK


def _bind(
    options: LaunchOptions,
    ctx: StudioContext,
    routes: list[Route],
    err: TextIO,
) -> StudioServer:
    """Start a server on the port, or on one of the next 9 for the default."""
    tries = PORT_FALLBACKS + 1 if options.port == DEFAULT_PORT else 1
    for port in range(options.port, options.port + tries):
        server = StudioServer(
            routes,
            ctx.tokens,
            StaticFiles(),
            host=options.host,
            port=port,
            limits=options.limits,
            log_requests=options.log_requests,
            debug=options.debug,
            stderr=err,
        )
        try:
            server.start()
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
            continue
        return server
    last = options.port + tries - 1
    span = f"{options.port}-{last}" if tries > 1 else str(options.port)
    raise StudioUsageError(f"no free port in {span} on {options.host}")


def _open_route(open_path: str | None) -> str | None:
    if open_path is None:
        return None
    return "/requests?artifact=" + quote(open_path, safe="")


def _ensure_private_dir(path: Path) -> Path:
    """Create ``path`` 0700; refuse one another user owns."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise StudioUsageError(f"{path} is not a directory you own")
    if stat.S_IMODE(info.st_mode) != 0o700:
        path.chmod(0o700)
    return path


def _write_private(path: Path, text: str) -> Path:
    """Write ``text`` to a new 0600 file, replacing an older one.

    The temporary name holds the PID, so one left by a crash is stale once
    that PID is reused by a new launcher: it is removed (never followed,
    and only in Studio's own 0700 directories) and the write goes ahead.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        fd = os.open(temporary, flags, 0o600)
    except FileExistsError:
        temporary.unlink()
        fd = os.open(temporary, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)
    return path


def _write_state(
    state: Path, server: StudioServer, ctx: StudioContext, options: LaunchOptions
) -> Path:
    record = {
        "pid": os.getpid(),
        "host": options.host,
        "port": server.bound_port,
        "base_url": server.base_url,
        "root": str(options.root),
        "instance_id": ctx.instance_id,
        "admin_token": ctx.tokens.issue_admin(),
    }
    return _write_private(state / f"{server.bound_port}.json", json.dumps(record))


def _launch(
    state: Path,
    ctx: StudioContext,
    url: str,
    options: LaunchOptions,
    opener: Callable[[str], object],
) -> Path | None:
    """Open the browser on a 0600 redirect file, not on the URL itself.

    A URL on a command line is visible in process listings; the file is
    readable only by its owner and is removed once its bootstrap is spent
    or expired, and at exit.
    """
    if not options.open_browser:
        return None
    directory = _ensure_private_dir(state / "launch")
    escaped = html.escape(url, quote=True)
    page = (
        '<!doctype html><meta charset="utf-8">'
        '<meta name="referrer" content="no-referrer">'
        f'<meta http-equiv="refresh" content="0;url={escaped}">'
        f'<title>Stormlog Studio</title><a href="{escaped}">Open Stormlog Studio</a>\n'
    )
    path = _write_private(directory / f"{ctx.instance_id}.html", page)
    try:
        opener(path.as_uri())
    except Exception:  # a missing browser must not stop the server
        pass
    return path


def _wait(
    stop: threading.Event,
    ctx: StudioContext,
    bootstrap: str,
    launch: Path | None,
) -> None:
    """Serve until ``stop`` is set (``_serve``'s SIGINT/SIGTERM handlers)."""
    deadline = time.monotonic() + BOOTSTRAP_TTL_S
    while not stop.wait(0.5):
        if launch is not None and _launch_spent(ctx, bootstrap, deadline):
            _remove(launch)
            launch = None


def _launch_spent(ctx: StudioContext, bootstrap: str, deadline: float) -> bool:
    if time.monotonic() >= deadline:
        return True
    tokens = ctx.tokens
    return isinstance(tokens, SessionTokens) and not tokens.bootstrap_pending(bootstrap)


def _install_signals(stop: threading.Event) -> Callable[[], None]:
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    def on_signal(signum: int, frame: FrameType | None) -> None:
        stop.set()

    previous = {
        signum: signal.signal(signum, on_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def restore() -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    return restore


def _close_context(ctx: StudioContext, composed_by_app: bool) -> None:
    """Cancel builds, kill the analyzer and stop the pool (§2.4)."""
    if composed_by_app:
        app = importlib.import_module(APP_MODULE)

        app.shutdown(ctx)
        return
    ctx.indexes.close()
    ctx.analysis.close()
    ctx.pool.shutdown(wait=False, cancel_futures=True)


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# --- stormlog studio url -----------------------------------------------------


def _url_main(argv: list[str], out: TextIO) -> int:
    parser = _ArgumentParser(
        prog="stormlog studio url",
        description="Print a fresh sign-in URL for a running Studio.",
    )
    parser.add_argument("route", nargs="?", help="a page, such as /requests?...")
    parser.add_argument("--port", type=_port, default=None)
    args = parser.parse_args(argv)
    record = _running_instance(state_directory(), args.port)
    url = _request_bootstrap(record, args.route)
    print(url, file=out)
    return EXIT_OK


def _running_instance(state: Path, port: int | None) -> dict[str, Any]:
    records = _live_records(state)
    if port is not None:
        records = [r for r in records if r.get("port") == port]
    if not records:
        raise StudioUsageError("no Studio instance is running; start `stormlog studio`")
    if len(records) > 1:
        ports = ", ".join(str(r.get("port")) for r in records)
        raise StudioUsageError(f"several Studio instances run ({ports}); pass --port")
    return records[0]


def _live_records(state: Path) -> list[dict[str, Any]]:
    names = sorted(state.glob("*.json")) if state.is_dir() else []
    return [r for r in (_read_state(p) for p in names) if r is not None]


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not _alive(record.get("pid")):
        return None
    return record


def _alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _request_bootstrap(record: dict[str, Any], route: str | None) -> str:
    """A fresh sign-in URL from the instance ``record`` names.

    A live PID is not proof the instance still runs (the PID may be
    reused), so before the admin token is sent, ``GET /api/v1/health``
    must answer with the record's ``instance_id``; and the URL printed
    must be on the record's ``base_url``. Another server on the port
    receives no token and cannot have a link of its choosing printed.
    """
    base = str(record.get("base_url", ""))
    # Never through a proxy: the admin token goes only to loopback.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    health = _fetch_json(opener, urllib.request.Request(base + "/api/v1/health"), base)
    instance = record.get("instance_id")
    if not isinstance(instance, str) or _field(health, "instance_id") != instance:
        raise StudioUsageError(
            f"the server at {base} is not the Studio instance its state file "
            "names; start `stormlog studio` again"
        )
    request = urllib.request.Request(
        base + "/api/v1/admin/bootstrap",
        data=json.dumps({"route": route}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {record.get('admin_token', '')}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    url = _field(_fetch_json(opener, request, base), "url")
    if not isinstance(url, str) or not url.startswith(base + "/#/"):
        raise StudioUsageError(
            f"the server at {base} answered with a URL that is not its own; "
            "not printing it"
        )
    return url


def _fetch_json(
    opener: urllib.request.OpenerDirector, request: urllib.request.Request, base: str
) -> Any:
    try:
        with opener.open(request, timeout=_URL_TIMEOUT_S) as response:
            return json.loads(response.read(_URL_MAX_BYTES).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise StudioUsageError(f"Studio refused: HTTP {exc.code}") from None
    except (OSError, ValueError, RecursionError):
        raise StudioUsageError(f"no Studio answers at {base}") from None


def _field(document: Any, name: str) -> Any:
    return document.get(name) if isinstance(document, dict) else None
