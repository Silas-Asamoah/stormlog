[← Back to docs](index.md)

# Scrubbing

Stormlog keeps a local artifact for every run, and it can also send data
elsewhere. The scrubbing primitives in `stormlog.scrub` are shared, so every
surface redacts the same way. They are building blocks, not a policy. An
exporter decides what it sends from its own allowlist of fields, and documents
that list. These helpers are the layers underneath the allowlist.

This page covers what each helper guarantees and what it does not. A wider
policy for every artifact Stormlog writes is tracked in
[#111](https://github.com/Silas-Asamoah/stormlog/issues/111). Until that
lands, two fields of the local inference artifact stay sensitive:

- `infer.request.error_message` holds the server's whole error body, which
  can echo the request, prompt included.
- `endpoint` is recorded as given, credentials and query string included.

## URLs

`redact_url(url)` keeps the scheme, host, port and path of a URL. It drops
any user name and password, and replaces a query string with
`?<redacted>`, since either can carry a token:

```python
from stormlog.scrub import redact_url

redact_url("https://user:hunter2@host:8443/flush_cache?token=abc")
# 'https://host:8443/flush_cache?<redacted>'
```

The path is kept, so a URL whose path holds a secret is not safe to share
after `redact_url` alone. The host is lower-cased, and an IPv6 host keeps
its brackets. `redact_url(None)` is `None`.

`stormlog.infer.cache_state.redact_url` is the same function, kept for
callers that imported it from there.
