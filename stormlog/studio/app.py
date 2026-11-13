"""Studio's composition root (design §10.1, component C4).

This is the only module that imports concrete classes across Studio's
components. :func:`build_context` puts C3's workspace root and tokens, C1's
index registry and C4's scanner, analyzer and caches into one
:class:`~stormlog.studio.contracts.StudioContext`, around one builder pool.
:func:`routes` is C3's auth routes plus C4's resource routes, and
:func:`compose` is the default ``Composer`` that ``stormlog studio``
(``cli.main``) uses. :func:`shutdown` stops what :func:`build_context`
started.

The context is built before any socket is bound, and the launcher may
bind a fallback port, so nothing here reads ``LaunchOptions.port`` as the
port Studio serves on (C3's note). :func:`serve`, for tests and embedders,
reports the port the server bound (``StudioServer.bound_port``).
"""

from __future__ import annotations

import logging
import secrets
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from stormlog import __version__
from stormlog.studio import api, selection
from stormlog.studio.analysis_cache import AnalyzerRunner
from stormlog.studio.cache import Documents
from stormlog.studio.contracts import (
    IndexRegistry,
    IndexSnapshot,
    LaunchOptions,
    Route,
    StudioContext,
    StudioError,
    publish_hook,
)
from stormlog.studio.httpd import StudioServer
from stormlog.studio.index import ArtifactIndexes
from stormlog.studio.security import RootDirectory, SessionTokens, auth_routes
from stormlog.studio.static_files import StaticFiles
from stormlog.studio.workspace import ArtifactScanner

__all__ = [
    "BUILDER_THREADS",
    "build_context",
    "compose",
    "routes",
    "serve",
    "shutdown",
    "warm",
]

_LOG = logging.getLogger(__name__)

# §2.1: one builder thread builds and refreshes indexes and scans.
BUILDER_THREADS = 1


def build_context(options: LaunchOptions) -> StudioContext:
    """Build every component for ``options``, sharing one builder pool.

    It also registers :func:`warm` as a publish listener, so a selection's
    case facts are computed on the builder pool when a snapshot lands,
    not by the first request that reads it (Fable's F-7); and it starts
    the index of the artifact the launcher opens, and a first workspace
    scan, so both are under way before the browser asks.
    """
    instance_id = secrets.token_hex(16)
    pool = ThreadPoolExecutor(
        max_workers=BUILDER_THREADS, thread_name_prefix="stormlog-studio-builder"
    )
    root = RootDirectory(options.root)
    limits = options.limits
    indexes = ArtifactIndexes(root, pool, limits)
    ctx = StudioContext(
        instance_id=instance_id,
        version=str(__version__),
        options=options,
        root=root,
        tokens=SessionTokens(),
        indexes=indexes,
        scanner=ArtifactScanner(root, pool, limits),
        analysis=AnalyzerRunner(
            root, options.state_dir, limits, instance_id=instance_id
        ),
        cache=Documents(limits),
        pool=pool,
    )
    add_warm_listener(indexes)
    _start(ctx)
    return ctx


def add_warm_listener(indexes: IndexRegistry) -> bool:
    """Register :func:`warm` on ``indexes``; False when it takes no listener."""
    hook = publish_hook(indexes)
    if hook is None:
        return False
    hook.add_publish_listener(warm)
    return True


def warm(snapshot: IndexSnapshot) -> None:
    """Compute C2's per-snapshot facts for a snapshot just published.

    ``selection.warm(snapshot)`` is C2's entry point for this. Until a C2
    build provides it, this calls ``selection.case_facts(snapshot)``, the
    cost F-7 measured (about 2.5 s at 500,000 rows), which is the one
    ``warm`` covers. A failed index has no rows to warm.
    """
    if snapshot.status == "failed":
        return
    warmer: Any = getattr(selection, "warm", None)
    if callable(warmer):
        warmer(snapshot)
    else:
        selection.case_facts(snapshot)


def _start(ctx: StudioContext) -> None:
    """Queue the opened artifact's first build, then a first scan."""
    if ctx.options.open_path is not None:
        try:
            ctx.indexes.snapshot(ctx.options.open_path)
        except StudioError as error:
            _LOG.warning("cannot open %s: %s", ctx.options.open_path, error)
    try:
        ctx.scanner.scan()
    except StudioError as error:
        _LOG.warning("no first workspace scan: %s", error)


def routes(ctx: StudioContext) -> list[Route]:
    """Every slice-1 route: C3's auth routes, then C4's resource routes."""
    return [*auth_routes(ctx), *api.resource_routes(ctx)]


def compose(options: LaunchOptions) -> tuple[StudioContext, list[Route]]:
    """The default ``Composer``: the context for ``options`` and its routes."""
    ctx = build_context(options)
    return ctx, routes(ctx)


def serve(ctx: StudioContext, route_list: list[Route] | None = None) -> StudioServer:
    """Start a :class:`StudioServer` for ``ctx`` on ``options.host`` and port.

    For tests and embedders; ``stormlog studio`` binds through ``cli``,
    which also tries fallback ports. Read the port from the returned
    server's ``bound_port`` (``options.port`` may be 0).
    """
    options = ctx.options
    server = StudioServer(
        routes(ctx) if route_list is None else route_list,
        ctx.tokens,
        StaticFiles(),
        host=options.host,
        port=options.port,
        limits=options.limits,
        log_requests=options.log_requests,
        debug=options.debug,
    )
    server.start()
    return server


def shutdown(ctx: StudioContext) -> None:
    """Cancel builds, kill the analyzer and stop the builder pool (§2.4).

    Indexes close first, so a build stops at its next batch; then the
    analyzer subprocess is killed and its directory removed; then the
    pool drops queued work. Each step runs even when an earlier one fails.
    """
    try:
        ctx.indexes.close()
    finally:
        try:
            ctx.analysis.close()
        finally:
            ctx.pool.shutdown(wait=False, cancel_futures=True)
