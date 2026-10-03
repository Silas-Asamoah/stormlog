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

Each value is matched in the spellings it most often travels in:

| Spelling | Example of where it appears |
| --- | --- |
| as given | a header value, a log line |
| percent-encoded, with hex digits in either case, any characters left plain (as `quote` leaves `/`), and `+` for a space | a URL |
| JSON-escaped: `\uXXXX` in either case, surrogate pairs for characters outside the BMP, an escaped `/`, and the short escapes such as `\"` | a string inside a JSON body |
| base64 of the value on its own, standard and URL-safe, with and without padding | an encoded token, a Basic header |

Each character can be spelled differently from the next, so a value that is
partly encoded is still found. These are not covered:
- double encoding, such as `%252F`;
- base64 of a longer string that contains the value, where it starts at
  another offset;
- HTML entities and other character encodings, such as UTF-16.

Finding every spelling costs more than a plain search: about 30 ms per
megabyte for each value. Exporters redact bounded strings, and `scrub_text`
bounds its input first.

A Basic authorization header encodes `user:password` as one string, so
`url_secrets(url)` returns that pair alongside the password itself, decoded
and also as written in the URL, for a client that built the header without
decoding it. It also returns the user name on its own, whatever the
password, since a user name is often a token, and every query value. A URL's
fragment is never sent to a server, so it is not read:

```python
from stormlog.scrub import KnownSecrets, url_secrets

secrets = KnownSecrets([api_key, *url_secrets(endpoint)])
secrets.redact(text)    # each form of each value becomes <redacted>
secrets.found_in(text)  # True if any form is present
```

Values shorter than 8 characters (`MIN_SECRET_LENGTH`) are skipped and
counted in `skipped_short`. Replacing a short value everywhere would erase
ordinary text such as case names and numbers, and a credential that short is
not protected by redaction anyway. Every occurrence of every value is found
in the original text before anything is replaced, and overlapping
occurrences are merged into one redaction, so no part of one value is left
behind when another that contains it, or crosses it, is replaced.

## Free text

`scrub_text(text, max_bytes=None, secrets=None)` removes the common shapes of
a credential from free text, then cuts the result to at most `max_bytes` bytes
of UTF-8. Use it only on text an exporter has consent to send, such as an
error message. It is a third layer under the allowlist and `KnownSecrets`, not
a guarantee: a pattern catches what a credential usually looks like, and an
opaque secret with no recognisable shape gets through. That is why the
exporters send no free text unless you opt in.

What it removes:

| Shape | Example | Result |
| --- | --- | --- |
| an `Authorization` or `Proxy-Authorization` header, to the end of its line | `Authorization: Bearer abc…` | `Authorization: <redacted>` |
| a Bearer or Basic token | `bearer abc.def-123` | `bearer <redacted>` |
| URL user information and query strings | `https://u:p@host/x?k=v` | `https://<redacted>@host/x?<redacted>` |
| a JSON member whose key contains a word from `SECRET_KEY_WORDS` (below) | `"api_key": "abc"` | `"api_key": "<redacted>"` |
| `key=value` or `key: value` with such a key, the value bare or in single or double quotes | `client_secret=abc`, `password="a b"` | `client_secret=<redacted>`, `password="<redacted>"` |
| known key formats | `sk-…`, `hf_…`, `AKIA…`/`ASIA…`, `ghp_…` and the other GitHub token prefixes, `github_pat_…`, `xox?-…`, three-part JWTs | `<redacted>` |
| private key blocks, including one cut before its `END` line | `-----BEGIN PRIVATE KEY-----…` | `<redacted>` |

The steps run in a fixed order:
1. The text is first cut to `max_bytes` + 4,096 characters
   (`INPUT_MARGIN_CHARS`), so the work is bounded by `max_bytes` whatever
   the body's size. Every pattern also runs in time linear in its input:
   each can start only where a run of its characters starts, and a key's
   words are checked on the whole key, never by backtracking.
2. Every match is found in that text: each form of each value in
   `secrets`, and each pattern's span.
3. Overlapping spans are merged, and each is replaced once with
   `<redacted>`. No replacement happens before every match is found, so a
   marker inserted for one match can never hide another, such as a known
   secret in a URL's path hiding the URL's query.
4. The cut to `max_bytes` comes last.

Only the first `max_bytes` characters of the input can reach the output. The
margin is lookahead: it lets a match that starts within the kept length be
seen whole, and text from it is never copied, even when redactions before it
leave room in the output. So the cut never leaves part of a secret that a
whole match would have removed, as long as the secret is shorter than the
margin, however much earlier redactions shrink the text.
The patterns err on the side of removing too much. For example, a
`max_tokens="128"` field loses its value, and a key run together with the
word before it is still removed. A value containing spaces loses only its
first word, which is one reason consent is needed.

`truncate_utf8(text, max_bytes)` returns the longest prefix whose UTF-8
encoding fits `max_bytes`, and never splits a character. A lone surrogate,
which UTF-8 cannot encode, becomes `?`.

## Key names

`is_forbidden_key_name(name)` says whether a key's name suggests its value may
be a credential. An exporter refuses such a key from any outside source, such
as an environment variable or a flag, even when an operator names it
explicitly, because the value cannot be checked. The test is a substring
match, ignoring case, against the words in `SECRET_KEY_WORDS`:

`pass`, `pwd`, `secret`, `token`, `key`, `auth`, `bearer`, `cred`, `cookie`,
`session`, `signature`, `private`

So `api-key`, `API_KEY` and `apikey` are all caught. A few innocent names are
caught too, which is the safe mistake. `scrub_text` uses the same words for
the keys in JSON members and `key=value` pairs.
