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
after `redact_url` alone. `redact_url(url, origin_only=True)` drops the path
too and keeps only the scheme, host and port. Use it wherever the path is not
known to be safe, as an exporter does for every URL it reports:

```python
redact_url("https://host:8443/v1/sk-secret/chat?k=v", origin_only=True)
# 'https://host:8443'
```

The host is lower-cased, and an IPv6 host keeps its brackets.
`redact_url(None)` is `None`.

`stormlog.infer.cache_state.redact_url` is the same function, kept for
callers that imported it from there.

## Credentials Stormlog was given

`KnownSecrets` holds the exact values Stormlog was handed as credentials, such
as an API key, a collector's header values, or a password in a URL. It
redacts them wherever they appear. An exporter's allowlist already keeps these
fields out of what it sends. `KnownSecrets` is the backstop for a value that
turns up somewhere unexpected, for example echoed inside an error message.

Each value is matched in the forms it most often travels in:

| Form | Example of where it appears |
| --- | --- |
| as given | a header value, a log line |
| percent-encoded (`quote` and `quote_plus`) | a URL |
| JSON-escaped (ASCII and UTF-8) | a string inside a JSON body |
| base64 (standard and URL-safe, with and without padding) | an encoded token |

A Basic authorization header encodes `user:password` as one string, so
`url_secrets(url)` returns that pair alongside the password itself. It also
returns a user name given without a password, which is often a token, and
every query value:

```python
from stormlog.scrub import KnownSecrets, url_secrets

secrets = KnownSecrets([api_key, *url_secrets(endpoint)])
secrets.redact(text)    # each form of each value becomes <redacted>
secrets.found_in(text)  # True if any form is present
```

Values shorter than 8 characters (`MIN_SECRET_LENGTH`) are skipped and
counted in `skipped_short`. Replacing a short value everywhere would erase
ordinary text such as case names and numbers, and a credential that short is
not protected by redaction anyway. When one value contains another, the
longer is replaced first, so no tail of it is left behind.
