[← Stormlog Studio](studio.md)

# Stormlog Studio API (slice 1)

This page lists every route `stormlog studio` serves: its auth, its
parameters, the document it answers with, its errors, and an example. The
design is issue #227's server design (§5); each document's JSON Schema in
`docs/schemas/studio_<name>_v1.schema.json` is its one definition, and
`docs/schemas/examples/` holds an example of each.

## Conventions

- Base path `/api/v1/`. JSON bodies are UTF-8, `application/json`.
- Every JSON body names its shape: `"format": "stormlog.studio.<name>"`,
  `"schema_version": 1`.
- An artifact is named by its path relative to the workspace root, in
  `artifact=`. Every document about one artifact carries the `artifact`
  block (`studio_artifact_block_v1`): path, index ID, generation, status
  (`building`, `ready`, `partial`, `failed`), size, bytes read, SHA-256,
  growth, session, `origin_ns`, residency, the limits hit and the `cut`.
- Integers under a key ending in `_ns`, and any integer beyond 2^53 − 1,
  are decimal strings. NaN and infinities are never sent.
- A query parameter the route does not take is `400 unknown_parameter`, so
  a typo never widens a selection. A single-valued parameter given twice
  is `400 bad_parameter`.
- Responses carry `Cache-Control: no-store` and no ETag.
- Every route also passes the checks of [the security model](studio.md):
  the Host name, the Origin, Fetch Metadata on `/api/`, auth, and for a
  `POST` a JSON body with an Origin.

### Auth

| Auth | How the request authenticates |
| --- | --- |
| public | nothing |
| bearer | `Authorization: Bearer <session token>` |
| bearer_or_ticket | the bearer token, or `ticket=<ticket>` from `POST /api/v1/tickets` for exactly this route and query; a ticket is single-use and lasts 30 s |
| admin | `Authorization: Bearer <admin token>` from the state file, with no `Origin` and no `Sec-Fetch-*` header |

Without a valid credential every route but the two public ones answers
`401 unauthorized`. A spent or expired ticket answers `401` with a short
`text/plain` note instead of JSON, because a ticket URL is a navigation.

### Errors

Every error is a `stormlog.studio.error` document (`studio_error_v1`):

```json
{"format": "stormlog.studio.error", "schema_version": 1,
 "error": {"code": "bad_parameter", "message": "status: unknown value 'eror'",
           "http_status": 400, "verdict_status": "usage", "field": "status",
           "details": {"allowed": ["cancelled", "delivery_unknown", "dropped", "error", "ok",
                                   "rejected", "timeout", "unreachable"]}}}
```

| HTTP | Codes |
| --- | --- |
| 400 | `bad_parameter`, `unknown_parameter`, `limit_exceeded`, `malformed_request` |
| 401 | `unauthorized`, `bootstrap_invalid`, `ticket_invalid` |
| 403 | `host_not_allowed`, `origin_not_allowed`, `fetch_site_not_allowed`, `path_not_allowed` |
| 404 | `artifact_not_found`, `request_not_found`, `line_out_of_range`, `route_not_found` |
| 405 | `method_not_allowed` |
| 409 | `cursor_mismatch` (`details.reason`: `index_rebuilt`, `other_filter`, `other_sort`), `request_ambiguous` (lists the lines), `stale_line` |
| 413, 415, 431 | `body_too_large`, `unsupported_media_type`, `head_too_large` |
| 422 | `not_an_inference_artifact`, `unsupported_artifact` (`details.reason`: `audio`, `multiple_sessions`), `unreadable`, `report_too_large` |
| 500 | `internal_error`, with a generic message |
| 503 | `busy` |

**Failed indexes.** `GET /api/v1/artifact`, its capabilities, sources and
analysis, and the line and evidence routes answer for any index state, so
the page can show why an artifact cannot be served. Every other artifact
route answers a failed index with its 422: `unreadable`,
`not_an_inference_artifact`, or `unsupported_artifact` with
`details.reason` `audio` or `multiple_sessions`. An index that failed on
an internal error is `500 internal_error`.

**While an index builds** every artifact route answers 200 with the
documents of a zero-row snapshot whose `artifact.status` is `building`.
The page re-reads `GET /api/v1/artifact` every 2 s.

**202.** Only `GET /api/v1/artifact/analysis` answers 202, while the
analysis is `pending` or `running`: the body is the whole analysis
document, with `Retry-After`.

### Filters

The selection routes (`requests`, `summary`, `distribution`, `points`,
`export/preview`, `export`, and `request` for `in_selection`) take the same
filter parameters and pick rows the same way, so their `selection` blocks
agree for one query.

| Parameter | Values | Default |
| --- | --- | --- |
| `phase` | `measured`, `warmup`, `all` | `measured` |
| `case` | repeatable | every case |
| `status` | repeatable: `ok`, `timeout`, `rejected`, `error`, `dropped`, `cancelled`, `unreachable`, `delivery_unknown` | every status |
| `stream` | `true`, `false` | both |
| `token_source` | repeatable, on `output_token_source` | every source |
| `q` | text, at most 256 characters | none |
| `window` | `START_NS,END_NS`, half-open, client clock | none |
| `window_ms` | `START,END` in ms from `origin_ns`, at most 6 decimals; not with `window` | none |
| `window_membership` | `arrival`, `overlap` | `arrival` |
| `completed_by_ns`, `completed_by_ms` | one of the two: keeps `ended_at_ns <= t` | none |
| `latency` | `METRIC:MIN:MAX`, half-open; an empty `MAX` is unbounded | none |

Every selection document carries the `selection` block: the canonical
filter and `params`, a `fingerprint`, the `size`, the cases pooled, the
whole case if any, the cohort, the rate scope and the searched fields.

## Routes

The examples use `$T` for the session token and `run.jsonl` for the
artifact; `curl -g` keeps the brackets and commas as written.

| Method and path | Auth | Document |
| --- | --- | --- |
| `POST /api/v1/session` | public | `session` |
| `GET /api/v1/health` | public | `health` |
| `POST /api/v1/tickets` | bearer | `ticket` |
| `POST /api/v1/admin/bootstrap` | admin | `bootstrap` |
| `GET /api/v1/server` | bearer | `server` |
| `GET /api/v1/capabilities` | bearer | `capabilities` |
| `GET /api/v1/catalog/metrics` | bearer | `metric_catalog` |
| `GET /api/v1/workspace` | bearer | `workspace` |
| `POST /api/v1/workspace/scan` | bearer | `workspace` |
| `GET /api/v1/workspace/items` | bearer | `items` |
| `GET /api/v1/artifact` | bearer | `artifact` |
| `GET /api/v1/artifact/capabilities` | bearer | `artifact_capabilities` |
| `GET /api/v1/artifact/sources` | bearer | `sources` |
| `GET /api/v1/artifact/cases` | bearer | `cases` |
| `GET /api/v1/artifact/requests` | bearer | `requests` |
| `GET /api/v1/artifact/summary` | bearer | `summary` |
| `GET /api/v1/artifact/distribution` | bearer | `distribution` |
| `GET /api/v1/artifact/points` | bearer | `points` |
| `GET /api/v1/artifact/samples` | bearer | `samples` |
| `GET /api/v1/artifact/request` | bearer | `request` |
| `GET /api/v1/artifact/request/chunks` | bearer | `chunks` |
| `GET /api/v1/artifact/line` | bearer_or_ticket | the raw line, `text/plain` |
| `GET /api/v1/artifact/evidence` | bearer | `evidence` |
| `GET /api/v1/artifact/export/preview` | bearer | `export_preview` |
| `GET /api/v1/artifact/export` | bearer_or_ticket | NDJSON, streamed |
| `GET /api/v1/artifact/analysis` | bearer | `analysis` |
| `GET /api/v1/report` | bearer | `report` |

Every route has a 30 s deadline except `line` and `export`, which have
600 s.

### Sign-in and tickets

**`POST /api/v1/session`** (public). Body `{"bootstrap": "<token>"}`, the
one-time token from the launch URL's fragment. Answers `session` with
`session_token` and `instance_id`. Errors: `401 bootstrap_invalid` (spent,
expired or unknown), `400 bad_parameter`, `415`.

```bash
curl -s -X POST -H 'Origin: http://127.0.0.1:8790' -H 'Content-Type: application/json' \
  -d '{"bootstrap": "..."}' http://127.0.0.1:8790/api/v1/session
```

**`GET /api/v1/health`** (public). Answers `health`: `{"ok": true,
"instance_id": ...}`. A new `instance_id` means Studio restarted and the
stored token is dead.

**`POST /api/v1/tickets`** (bearer). Body `{"route": "/api/v1/...",
"query": "<query string>"}`. Answers `ticket` with `ticket`, `url` (the
route and query with `ticket=` added) and `expires_in_s` (30). Only the
`line` and `export` routes redeem one.

```bash
curl -s -X POST -H "Authorization: Bearer $T" -H 'Origin: http://127.0.0.1:8790' \
  -H 'Content-Type: application/json' \
  -d '{"route": "/api/v1/artifact/export", "query": "artifact=run.jsonl&status=ok"}' \
  http://127.0.0.1:8790/api/v1/tickets
```

**`POST /api/v1/admin/bootstrap`** (admin). Body `{"route": "/requests?..."}`
or `{}`. Answers `bootstrap` with a fresh sign-in `url`. `stormlog studio
url` calls it; a browser cannot (any `Origin` or `Sec-Fetch-*` is 403).

### Server and workspace

**`GET /api/v1/server`** (bearer). No parameters. Answers `server`: the
Stormlog `version`, `api_version` 1, `instance_id`, the root's real path
and every bound of design §8.1 (`limits`). The port is not in it: read it
from the URL you used.

**`GET /api/v1/capabilities`** (bearer). No parameters. Answers
`capabilities`: jobs `unavailable` (`slice_2`), the analyzer's size and
time bounds, the export profile `original_bytes`, `remote: false`.

**`GET /api/v1/catalog/metrics`** (bearer). No parameters. Answers
`metric_catalog`: one entry per displayed value with its label, unit,
scope, source fields, formula, missing rule and notes, and the panel states.

**`GET /api/v1/workspace`** (bearer). No parameters. Answers `workspace`:
the last scan's state, counts by kind and the scan bounds. Studio starts a
first scan when it starts.

**`POST /api/v1/workspace/scan`** (bearer). Body `{}`. Queues a scan and
answers 200 with the `workspace` document (`scan.running` true until it
ends). Errors: `503 busy` while a scan runs; `400 unknown_parameter` for a
body other than `{}`.

**`GET /api/v1/workspace/items`** (bearer). `kind` (`inference_artifact`,
`audio_artifact`, `report`, `run_envelope`), `q` (substring of the path,
at most 256 characters), `cursor`, `limit` (1 to 1000, default 100).
Answers `items`: path, kind, size, `mtime_ns`, session, status, report
kind and verdict, sorted by path, with `next_cursor`. Errors: `400
bad_parameter` or `limit_exceeded`; `409 cursor_mismatch` for a cursor
made with another filter.

```bash
curl -s -H "Authorization: Bearer $T" \
  'http://127.0.0.1:8790/api/v1/workspace/items?kind=inference_artifact&limit=50'
```

**`GET /api/v1/report`** (bearer). `path`: a report envelope under the
root. Answers `report`: the `stormlog.report` v1 envelope, validated, with
its size and SHA-256. Errors: `404 artifact_not_found`, `403
path_not_allowed`, `422 report_too_large` (over 16 MiB), `422 unreadable`
with `details.reason` `not_a_report`.

### One artifact

**`GET /api/v1/artifact`** (bearer). `artifact`. Answers `artifact`:
session, index state and progress, inventory, problems, cases (planned and
observed, measured requests by status) and warnings, for any index state.
`problems.by_code` counts the problem lines by code; `unsupported_version`
there counts client request lines at another schema version, which are
never rows. `warnings` holds the banner's text: `index_cut` (a budget
stopped the index; `artifact.cut` names the first line left out),
`unsupported_version`, `problem_lines`, `growing`, `no_terminal_record`,
`gaps_not_resident`, `error_messages_not_resident`. Errors: `400`, `403
path_not_allowed`, `404 artifact_not_found`.

```bash
curl -s -H "Authorization: Bearer $T" 'http://127.0.0.1:8790/api/v1/artifact?artifact=run.jsonl'
```

```json
{"format": "stormlog.studio.artifact", "schema_version": 1,
 "artifact": {"path": "run.jsonl", "status": "partial", "cut": {"line": 412, "offset": 309120, ...}, ...},
 "problems": {"count": 1, "by_code": {"unsupported_version": 1}, "first": [...]},
 "warnings": [{"code": "index_cut", "message": "the index stopped at line 412 ..."}], ...}
```

**`GET /api/v1/artifact/capabilities`** (bearer). `artifact`. Answers
`artifact_capabilities`: each panel's state (`available`, `empty`,
`pending`, `partial`, `unavailable`, `not_indexed`,
`in_artifact_panel_later`, `unsupported`) with a reason and collector, and
the analyzer's observers once its report is ready. It starts the analysis
if none ran. Never cached, because the analysis changes state without a
new generation.

**`GET /api/v1/artifact/sources`** (bearer). `artifact`. Answers `sources`:
the inventory, the small records kept whole (`written` against `kept`, so
a record over the 64 KiB limit is disclosed), each panel's availability,
the analysis status and the observers when ready.

**`GET /api/v1/artifact/analysis`** (bearer). `artifact`. Answers
`analysis`: `pending`, `running`, `ready` (with the `infer analyze`
report), `unavailable` (`over_size_cap`, `growing`, `empty`, `changed`,
the index's failure, ...) or `failed` (`timeout`, `nonzero_exit`,
`bad_output`, `output_too_large`, `internal`, ...). While `pending` or
`running` it answers **202** with the same document and `Retry-After`.

```bash
curl -s -i -H "Authorization: Bearer $T" 'http://127.0.0.1:8790/api/v1/artifact/analysis?artifact=run.jsonl'
# HTTP/1.1 202 Accepted ... Retry-After: 2, then later HTTP/1.1 200 OK
```

**`GET /api/v1/artifact/cases`** (bearer). `artifact`. Answers `cases`: one
summary per measured case, planned cases included,
`comparison_scope: "full_measured_case"`, no deltas.

### Selection routes

**`GET /api/v1/artifact/requests`** (bearer). `artifact`, the filters,
`sort` (`artifact` default, `started_at`, `client.e2e`, `client.ttft`,
`first_chunk`, `output_tokens`, `status`, `case`, `request_id`), `order`
(`asc` default, `desc`), `cursor`, `limit` (1 to 1000, default 100).
Answers `requests`: the selection block, one page of rows, `next_cursor`,
`total` and `generation_moved` (true when the cursor came from an older
generation of the same index). Errors: `400`; `409 cursor_mismatch` for a
cursor from a rebuilt index (`index_rebuilt`) or another filter or sort.

```bash
curl -s -g -H "Authorization: Bearer $T" \
  'http://127.0.0.1:8790/api/v1/artifact/requests?artifact=run.jsonl&status=error&status=timeout&sort=client.e2e&order=desc&limit=50'
```

**`GET /api/v1/artifact/summary`** (bearer). `artifact` and the filters.
Answers `summary`: counts, failure share, latency quantiles with n and
sufficiency, streaming gaps, token coverage, intervals and throughput for a
rate-eligible selection, client memory and freshness.

```bash
curl -s -g -H "Authorization: Bearer $T" \
  'http://127.0.0.1:8790/api/v1/artifact/summary?artifact=run.jsonl&case=c8_in4096_out128&window_ms=0,30000'
```

**`GET /api/v1/artifact/distribution`** (bearer). The filters, `metric` (a
client metric such as `client.e2e`), and `bins` (1 to 200, default 20) or
`edges` (`e0,...,eN`, at most 201, finite and increasing), not both.
Answers `distribution`: half-open bins with counts and their exact
`latency=` parameter, `below`, `above` and the denominator.

**`GET /api/v1/artifact/points`** (bearer). The filters, `x` (`start`
default, `end`, `arrival`), `y` (a client metric, `client.e2e` default;
`first_chunk`, `output_tokens`, `chunk_count`), `max_points` (2 to 5000,
default 2000), `keep` (repeatable request IDs always drawn). Answers
`points`: `x_ms` from `origin_ns` with the exact `x_ns`, `y`, `drawn`,
`total`, `without_value` and the downsampling method.

**`GET /api/v1/artifact/samples`** (bearer). `artifact`, `window` or
`window_ms` (with `window_membership`), `max_points` (3 to 5000, default
1000). Answers `samples`: client system samples per `(sampler,
device_id)`, bucketed with min, max and last, and the gaps.

**`GET /api/v1/artifact/request`** (bearer). `request_id`, optional `line`
(when the ID repeats), and the filters, for `in_selection`. Answers
`request`: the record (non-finite values nulled and listed), client values
with reasons, `elapsed_ms` for a failure, token provenance, gap spread and
the first 128 gaps, and its dispatch and first-content lines. Errors: `404
request_not_found`; `409 request_ambiguous` with the lines.

**`GET /api/v1/artifact/request/chunks`** (bearer). `artifact`,
`request_id`, optional `line`, `offset` (default 0), `limit` (1 to 10000,
default 1000). Answers `chunks`: the gaps in written order, `total`.

### Raw lines, evidence and export

**`GET /api/v1/artifact/line`** (bearer_or_ticket). `artifact`, `line`
(0-based, blank lines counted), optional `sha256`. Answers the line's bytes
as written, without its newline, `text/plain; charset=utf-8`, with
`X-Stormlog-Line` and `X-Stormlog-Line-Sha256`. Studio reads only lines
its index holds a reference for: every client request row, the first 1,000
lines of each event type, problem lines, dispatch and first-content lines,
and the cut. Errors: `404 line_out_of_range` (`details.reason` `past_end`
or `not_indexed`); `409 stale_line` when `sha256` differs or the file's
bytes changed.

```bash
curl -s -H "Authorization: Bearer $T" \
  'http://127.0.0.1:8790/api/v1/artifact/line?artifact=run.jsonl&line=842'
```

**`GET /api/v1/artifact/evidence`** (bearer). `artifact`, `line`, optional
`record_id` and `sha256`. Answers `evidence`: `resolved`, or `unresolved`
with `line_out_of_range`, `hash_mismatch` or `record_id_mismatch`, and the
line found.

**`GET /api/v1/artifact/export/preview`** (bearer). `artifact` and the
filters. Answers `export_preview`: the count and byte size of the export,
the selection block, the source file's path, size and SHA-256, the
Stormlog version, fields that may be sensitive with counts, and the
limitations ("not a redaction").

**`GET /api/v1/artifact/export`** (bearer_or_ticket). `artifact` and the
filters. Answers the selection's client request lines byte for byte, in
file order, as `application/x-ndjson`, streamed with chunked encoding,
with `Content-Disposition: attachment`, `X-Stormlog-Export-Count`,
`X-Stormlog-Selection` and `X-Stormlog-Source-Sha256` (or the prefix
headers while the file is not fully read). A line whose bytes changed
during the stream ends it without its final chunk, so the download fails
instead of saving a mixed file. The browser starts it with a ticket:

```bash
URL=$(curl -s -X POST -H "Authorization: Bearer $T" -H 'Origin: http://127.0.0.1:8790' \
  -H 'Content-Type: application/json' \
  -d '{"route": "/api/v1/artifact/export", "query": "artifact=run.jsonl&status=ok"}' \
  http://127.0.0.1:8790/api/v1/tickets | python3 -c 'import json,sys; print(json.load(sys.stdin)["url"])')
curl -s -o selection.jsonl "http://127.0.0.1:8790$URL"
```
