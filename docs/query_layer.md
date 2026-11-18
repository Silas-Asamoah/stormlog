[← Back to docs](index.md)

# Local Query Layer

Stormlog exposes a local, file-backed query surface for telemetry sessions and
artifact bundles through `stormlog.query` and `stormlog query`. The v1 design is
pure Python and in-process: it reads existing artifact files directly and does
not require a database server, hosted service, or persistent index.

## Engine Choice

The query layer starts with a pure Python row model instead of adopting DuckDB
as a required runtime dependency.

DuckDB is still a good future acceleration path. The current DuckDB Python
docs show that `read_json` can auto-detect regular JSON and newline-delimited
JSON files, infer object schemas, and read multiple files. The JSON extension is
also shipped by default or autoloaded on first use. Those traits make DuckDB a
strong candidate once Stormlog has enough large multi-session datasets to need a
columnar execution engine.

For v1, the harder product problem is not SQL execution. It is stable artifact
discovery, schema projection, session linkage, and a public contract that CLI,
TUI, notebooks, and automation can share. Keeping the first version in Python
lets Stormlog define that contract without designing around one backend too
early.

References:

- [DuckDB Python data ingestion](https://duckdb.org/docs/current/clients/python/data_ingestion.html)
- [DuckDB JSON loading](https://duckdb.org/docs/current/data/json/loading_json)
- [DuckDB JSON extension loading](https://duckdb.org/docs/current/data/json/installing_and_loading.html)

## Catalog and Projection

`ArtifactCatalog` discovers available artifacts before loading every event row.
It recognizes:

- explicit `stormlog_run.json` run envelopes
- append-only telemetry sink directories and JSONL segments
- flat JSON, JSONL, and CSV telemetry files
- diagnose bundles with session manifests
- OOM dump bundles with stable manifests and metadata
- `stormlog_attachments.json` sidecars

Discovery is manifest-first. Sink manifests provide session rows, segment paths,
and event counts without parsing all segment records. Diagnose and OOM manifests
provide bundle/session linkage without loading timelines or OOM event buffers.
Flat telemetry files are loaded only when a query needs their session or event
rows because they do not have a separate manifest ledger.

Run discovery is envelope-first. When `stormlog_run.json` exists, Stormlog uses
that explicit run membership. Otherwise, it synthesizes run rows from session
metadata: sessions sharing a non-null `job_id` become one distributed run, while
sessions without `job_id` remain separate even when they share a sink directory.

The row contracts are explicit and automation-friendly:

- `RunRow`: explicit or synthesized run identity, source provenance, grouped
  session ids, ranks, and attachment count
- `RunAttachmentRow`: local artifacts, distributed rank artifacts, and external
  references indexed by run/session/job/rank/kind/source namespace
- `SessionRow`: session identity, status, rank metadata, source provenance,
  event count when known, warning count, and linked OOM bundle count
- `EventRow`: canonical `TelemetryEvent` fields plus `source_path`,
  `source_kind`, and `session_status`
- `OOMBundleRow`: bundle path, creation time, backend, reason, event count,
  session linkage, and exception type/module
- `ExternalAttachment`: local attachment sidecar rows that link URLs or local
  paths to session/job/rank/time metadata
- `CorrelationEvidence`: derived evidence rows with confidence and reasons for
  investigation pivots around a timestamp or projected telemetry record
- `StormlogIssue`: grouped issue fingerprint, state, hit count, first/last
  seen timestamps, affected sessions, representative evidence, and evidence
  links back to raw sessions/events/bundles
- `SummaryRow`: built-in metric results with session/rank/status grouping

The canonical telemetry schema remains the event contract. Query rows add
provenance but do not mutate persisted telemetry records.

See [Run Envelopes](run_envelopes.md) for the run/session distinction,
`stormlog_run.json` schema, attachment storage model, and source namespace
rules.

`QueryStore.correlate(...)` is the first correlation-oriented query surface. It
collects telemetry events, derived timeline markers, alerts, OOM bundles,
diagnose manifests, fresh rollup windows, and `stormlog_attachments.json`
sidecars into one evidence list. Correlation requires a timestamp anchor or a
projected telemetry `record_id`; rows are ranked by identifier match, time-window
overlap, and distance from the anchor. See [Correlation Workflow](correlation.md)
for the matching contract.

Issue grouping is also derived. `QueryStore.list_issues()` groups OOMs,
collector degradation, alerts, and hidden-memory anomalies using deterministic
fingerprints. The current implementation computes these rows during query/load;
a future sidecar can persist issue state by fingerprint id without changing raw
telemetry artifacts. See [Durable Issue Fingerprinting](issue_fingerprinting.md).

## Caching and Loading

`QueryStore` caches loaded sessions in memory per source path and source kind.
There is no persistent index in v1. This keeps discovery cheap, avoids stale
cache invalidation rules, and makes the behavior easy to reason about for local
artifact directories that may still be written by long-running jobs.

Event materialization happens only for:

- `query_events(...)`
- summaries that require raw telemetry rows
- flat telemetry files whose sessions cannot be listed from a manifest

Session listing from a sink manifest and OOM bundle listing do not require JSONL
segment materialization.

## Built-In Summaries

The v1 query layer intentionally provides a small set of summaries instead of a
custom aggregation language:

- session count by status
- peak allocator allocated/reserved bytes
- peak device used bytes
- alert count by session or rank
- collector degradation transitions
- interrupted sessions with linked OOM bundles
- hidden-memory gap growth using `device_used_bytes - allocator_reserved_bytes`

These cover the immediate operational questions while leaving room for a DuckDB
adapter or richer aggregation API later.

To ask which fields separate one rank, window, session or grouped issue from
a baseline, use `QueryStore.compare` or `stormlog query compare`; see
[Outlier Comparison](outlier_comparison.md).

## Follow-On Tasks

- Add an optional DuckDB adapter behind the same row model for very large JSONL
  sink directories.
- Reuse `ArtifactCatalog` in TUI and distributed diagnostics loaders where it
  can replace bespoke discovery logic.
- Add notebook examples that use `stormlog.query.open([...])` directly.
- Add persistent indexing only after measuring real multi-session directory
  costs and defining invalidation behavior.
- Add automation-specific schemas on top of query rows rather than changing the
  artifact contract.
- Persist grouped issue state in an artifact-level `issues.json` sidecar after
  the fingerprint schema has been exercised by CLI/TUI users.
