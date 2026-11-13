[← Back to main docs](index.md)

# Stormlog Studio

Stormlog Studio is a local web workspace for inference artifacts (issue
#227). One Python process reads the artifacts under one directory, computes
every number with Stormlog's own code, and serves a browser frontend that
only formats and draws what the server sent.

This page covers starting Studio, signing in, and the security model. The
frontend and the artifact API are still being built: in this release the
server, its security layer and a placeholder page ship, and the routes that
read artifacts arrive with the next components.

## Platforms

Studio runs on macOS and Linux. On Windows, `stormlog studio` exits with
status 2 and a message; run it under WSL instead.

Studio needs no extra: the server uses only the standard library, and its
static files ship in the base wheel. Only `stormlog studio` imports it.

## Starting Studio

```bash
stormlog studio [PATH] [--root DIR] [--host 127.0.0.1] [--port N] [--no-browser]
                [--analysis-max-bytes N] [--log-requests] [--debug]
stormlog studio url [ROUTE] [--port N]
```

- `PATH` is an artifact or a directory. For an artifact, its directory is
  the workspace root and the browser opens on that artifact. For a
  directory, it is the root. With neither, the root is the current
  directory. `python -m stormlog.studio` is the same command.
- `--root DIR` names the root explicitly. A `PATH` given with it must be a
  file under it.
- Studio refuses `/` and your home directory as the root unless `--root`
  names them, because every JSONL file under the root becomes readable
  through Studio.
- `--host` takes a loopback address only: `127.0.0.1` (the default), `::1`
  or `localhost`. Anything else exits 2. There are no remote-access flags.
- `--port` defaults to 8790. When 8790 is taken, Studio tries 8791 to 8799,
  then exits 2. An explicit `--port N` is used as given, with no fallback,
  so it stays where an SSH tunnel expects it. `--port 0` picks any free
  port.
- `--no-browser` prints the sign-in URL without opening a browser.
- `--analysis-max-bytes N` caps the artifact size the out-of-process
  analyzer accepts (128 MiB by default).
- `--log-requests` logs each request's method, route, status and duration
  to stderr. It never logs query strings, headers or bodies. Logging is off
  by default.
- `--debug` prints tracebacks of internal errors to stderr. Responses never
  carry them.

Studio prints the root and a sign-in URL:

```text
Stormlog Studio is serving /home/me/runs
Open http://127.0.0.1:8790/#/auth?t=...
```

SIGINT (Ctrl+C) or SIGTERM stops Studio: it stops accepting connections,
cancels index builds, stops the analyzer, removes its state files and exits
0.

Exit codes: 0 for a normal stop; 2 for bad flags, no free port, no running
instance (`studio url`) or Windows; 5 for a `PATH` or root that is missing
or unreadable; 1 for anything unexpected.

## Signing in

The sign-in URL carries a one-time bootstrap token in its fragment (the
part after `#`), which browsers never send in a request line, a log or a
`Referer` header. The page trades it once for a session token and removes
it from the address bar.

- A bootstrap token works once, for 10 minutes.
- A session token is kept in the browser's `localStorage` for that Studio
  instance, so new tabs, bookmarks and pasted deep links work. It stops
  working after 12 hours without a request, and when that Studio process
  exits.
- To get a new sign-in URL for a running Studio, run
  `stormlog studio url`. Give a page to open, such as
  `stormlog studio url '/requests?artifact=runs%2F042.jsonl'`. With several
  instances running, pick one with `--port`.

When Studio opens the browser itself, it opens a private redirect file
rather than putting the token on a command line, where other users could
see it in the process list. The file is readable only by you and is removed
once the token is used or expires.

### State files

Studio keeps its state in `$XDG_STATE_HOME/stormlog/studio`, or
`~/.local/state/stormlog/studio` when `XDG_STATE_HOME` is unset (on macOS
too). The directory is created with mode 0700. While Studio runs it holds:

- `<port>.json` (mode 0600): the process ID, port, root, instance ID and
  the admin token that `stormlog studio url` uses;
- `launch/<instance id>.html` (mode 0600): the browser redirect file.

Both are removed when Studio stops. A file left by a killed Studio is
ignored by `stormlog studio url`.

## Using Studio on another machine

Studio listens on loopback only. To use it on a remote host, forward a
local port over SSH:

```bash
ssh -L 8790:127.0.0.1:8790 gpu-box
# on gpu-box:
stormlog studio --port 8790 --no-browser ~/runs
```

Then open the printed URL on your own machine. Any local port works
(`ssh -L 9000:127.0.0.1:8790` and `http://127.0.0.1:9000/#/auth?t=...`),
because Studio accepts a loopback host name with any port.

**Shared machines are not supported in this release.** On a host where
other people have accounts, any local user can connect to Studio's
loopback port. Every API route still needs a token, and the tokens and
state files are private to you, so other users cannot read artifacts
through Studio without one. But the threat model of a multi-user box
(another user who reads your browser profile, or binds a port Studio
falls back to) is not one this release defends against. Run Studio on a
machine only you use, or tunnel from one.

## Security model

Studio is a local server reachable from every page your browser opens, so
it defends against web pages as well as other local processes.

Every request passes these checks, in order:

1. **Host.** The host name must be `127.0.0.1`, `localhost` or `[::1]`,
   with any port. Anything else gets 403, which stops DNS rebinding. This
   applies to static files too.
2. **Origin.** If the browser sends an `Origin`, it must be `http://`
   followed by the request's `Host`. A page served from another local port
   fails this check.
3. **Fetch Metadata.** Under `/api/`, `Sec-Fetch-Site` must be
   `same-origin` or `none` when present. The page itself (`/` and
   `/static/`) accepts any value, so links from documents and issues open
   Studio.
4. **Auth.** Every API route needs a session token in an
   `Authorization: Bearer` header, except `GET /api/v1/health` and
   `POST /api/v1/session`. Downloads and raw-line tabs, which cannot send
   a header, use a single-use ticket valid for 30 seconds, bound to one
   URL. The admin route that `stormlog studio url` calls refuses any
   request that comes from a browser.
5. **POST bodies** must be `application/json` and, except on the admin
   route, carry an `Origin`.

Studio sends no CORS headers, uses no cookies and answers `OPTIONS` with
405. Every response carries `X-Content-Type-Options: nosniff` and
`Referrer-Policy: no-referrer`. API responses carry
`Cache-Control: no-store`. The page carries a strict
`Content-Security-Policy` with no inline scripts, and static files carry an
ETag with `Cache-Control: no-cache`, so the browser revalidates them on
each load.

Paths in the API are relative to the workspace root. Studio refuses
absolute paths, `..` segments, backslashes, NUL bytes and symlinks that
lead out of the root, and it checks that the file it opens is the file it
resolved. It opens a file one path component at a time from the root,
never following a symlink, so a directory swapped for a symlink while
Studio opens a file under it is refused too.

One case is not refused: a **hard link** inside the root to a file
outside it. A hard link is the file itself, not a pointer to a path, so
Studio serves it like any other file under the root. Making one needs
write access to a directory under the root and read access to the
target, so it lets nobody read what they could not read already. But if
other people can write under the root (a shared `runs/` directory),
whatever they link there is shown, analysed and exported by Studio as
part of the root. Serve a directory only you can write to.

Studio bounds what any client can hold:

| Bound | Default | On reaching it |
| --- | --- | --- |
| Connections | 16 | 503 `busy` |
| Request line and headers | 16 KiB | 431 `head_too_large` |
| Request body | 64 KiB | 413 `body_too_large` |
| Time per connection | 30 s from accept, then the route's own deadline (600 s for exports and raw lines) | the connection is closed |

Errors come back as `stormlog.studio.error` documents
(`docs/schemas/studio_error_v1.schema.json`). Their messages are scrubbed
of credentials and cut to 512 bytes.
