[← Back to main docs](index.md)

# Outlier Comparison

Stormlog can already tell you that a rank drifted, a window grew a hidden gap,
a session hit an OOM, or a collector degraded. Outlier comparison answers the
next question: **what makes this subset different from the rest?** You pick a
selection and a baseline, and Stormlog ranks the telemetry fields whose
distributions differ most between them.

```bash
stormlog query compare rank ./live_sink --rank 3
stormlog query compare window ./live_sink --session run-42 \
  --time-start-ns 1715500000000000000 --time-end-ns 1715500060000000000
stormlog query compare session ./runs --session run-oom --baseline-session run-ok
stormlog query compare fingerprint ./live_sink --fingerprint-id issue:3f2a...
```

The idea follows Honeycomb's
[BubbleUp](https://docs.honeycomb.io/investigate/analyze/identify-outliers),
which compares a selection with everything outside it, field by field.
Honeycomb does not publish its scoring, so the ranking below is Stormlog's own.

## Selection And Baseline

Comparison always reads raw telemetry events (`TelemetryEvent` v4) through the
[query layer](query_layer.md). Rollup sidecars are not used, because they keep
peaks and counts, not the value distributions a comparison needs.

| Scope | Selection | Baseline |
| --- | --- | --- |
| `rank` | events of `--rank` | events of every other rank; `--session` and `--job-id` narrow both |
| `window` | events of `--session` in `[--time-start-ns, --time-end-ns)` | the rest of that session; `--rank` narrows both |
| `session` | events of `--session` | events of each `--baseline-session`, or of every other loaded session; `--job-id` narrows both |
| `fingerprint` | events behind one grouped issue from `stormlog query issues` | other events of the same session within `--window-ns` (default 60 s) of an issue hit |

In Python, the same comparison is `QueryStore.compare`:

```python
import stormlog.query

store = stormlog.query.open(["./live_sink"])
result = store.compare(stormlog.query.ComparisonFilter(scope="rank", rank=3))
for row in result.fields[:5]:
    print(row.field, round(row.score, 3), row.confidence)
```

`stormlog.outliers.compare_events(selection, baseline, scope=...)` runs the
same ranking over any two event lists you already hold; the TUI uses it.

## How Fields Are Ranked

Every field gets a score from 0 (same distribution) to 1 (no overlap), so
categorical and numeric fields rank on one scale.

- **Categorical fields** score the total variation distance between the two
  value distributions, `0.5 * sum(|p_selection(v) - p_baseline(v)|)`. An event
  without the field counts as the value `(missing)`. The result lists the
  values whose share changed most, with both shares.
- **Numeric fields** score the two-sample Kolmogorov-Smirnov statistic: the
  largest gap between the two empirical distribution functions of the values
  present. When a field is missing far more often on one side, the change in
  missing ratio is used instead, so a counter that disappears under a degraded
  collector still ranks. The result lists the median and p95 of each side.

Both scores are rank-based or share-based, so a few huge values cannot
dominate them, and neither needs the data to follow any distribution.

Scores are effect sizes, **not significance tests**. Telemetry samples arrive
as an autocorrelated time series: consecutive samples of one rank are not
independent draws, which every chi-square or KS p-value assumes. With that
assumption broken, a long session makes every difference look significant and
a short one hides real ones. Stormlog ranks by how large a difference is and
states how much data supports it.

## Eligible Fields

| Kind | Fields |
| --- | --- |
| Categorical | `event_type`, `collector`, `backend`, `severity`, `is_alert`, `collector_health_status`, `collector_last_error_stem`, `is_degraded_collector`, `context`, `host`, `device_id`, `rank`, `local_rank`, `world_size` |
| Numeric | `allocator_allocated_bytes`, `allocator_reserved_bytes`, `allocator_active_bytes`, `allocator_inactive_bytes`, `allocator_change_bytes`, `device_used_bytes`, `device_free_bytes`, `hidden_gap_bytes`, `allocator_gap_bytes`, `utilization_ratio`, `fragmentation_ratio`, `allocator_cache_ratio`, `collector_consecutive_failures` |

`backend` and `severity` use the same normalization as
[issue fingerprints](issue_fingerprinting.md). `collector_last_error_stem` is
the collector's last error reduced to its stable stem, with numbers and hex
removed. `hidden_gap_bytes` is `device_used_bytes - allocator_reserved_bytes`,
as in the rollups and the hidden-gap summary; the ratio fields come from
`stormlog.derived_fields`.

Identifiers are never compared: `session_id`, `job_id`, `pid`, `timestamp_ns`
and record ids are unique by design and would only add noise. The fields that
define a scope are left out of it, so `rank` and `local_rank` are not compared
in rank scope, nor is `device_id`, since each rank usually owns its own
device. `--field` (repeatable) restricts a comparison to chosen fields.

## Reading The Output

Rank 3 of a four-rank run whose collector degraded for most of the run, with
reserved memory about 3 GB higher than the other ranks:

```text
$ stormlog query compare rank ./live_sink --rank 3 --limit 6
Selection: 300 events; baseline: 900 events
Field                      Kind         Score  Confidence  Difference                     Missing    Reasons
-------------------------  -----------  -----  ----------  -----------------------------  ---------  -------
allocator_allocated_bytes  numeric      0.800  high        median 7.07 GB vs 4.80 GB      0% vs 0%
allocator_reserved_bytes   numeric      0.800  high        median 8.97 GB vs 6.00 GB      0% vs 0%
collector_health_status    categorical  0.800  high        degraded: 80% vs 0%            0% vs 0%
utilization_ratio          numeric      0.800  high        median 0.295 vs 0.2            0% vs 0%
device_used_bytes          numeric      0.571  high        median 9.11 GB vs 6.30 GB      53% vs 0%
hidden_gap_bytes           numeric      0.533  high        median 307.20 MB vs 307.20 MB  53% vs 0%
```

`device_used_bytes` and `hidden_gap_bytes` rank through their missing ratio:
the degraded collector dropped the device counter on about half of rank 3's
samples, while the gap itself, where present, did not change.

`--json` prints the full result: the `criteria` that produced it, the
sessions, ranks and time span of each side, every ranked field with its value
shares or quantiles, the skipped fields, and warnings. The payload carries
`"format": "stormlog.outlier_comparison"` and `"schema_version": 1`.

## Failure Modes

**Tiny or heavily filtered baseline.** A comparison still runs, but when
either side has fewer than `--min-events` events (default 30), every field is
marked `low` confidence with `small_selection` or `small_baseline`, and a
warning says so. A numeric field present in fewer than `--min-events` rows on
either side gets `sparse_field`. An empty side yields no fields and a warning,
not an error, as with other empty queries. Widen the window, drop a filter, or
use session scope with more baseline sessions to raise support.

**Sparse data.** Fields absent from both sides are listed under `skipped` as
`no_data`, and fields with one value everywhere as `constant`, so the ranking
only shows fields that can separate the two sets.

**High-cardinality values.** A categorical field with more than
`--max-values` distinct values (default 50) is skipped as `high_cardinality`
instead of being scored: with that many values almost every value is rare on
both sides, and the distance between two scatters of rare values measures
sampling noise more than a real difference. Free-text errors are stemmed
before counting for the same reason.

**Degraded sessions.** A degraded collector usually drops counters rather
than reporting wrong ones. Missing values stay visible (as `(missing)` for
categorical fields and through the missing ratio for numeric ones), so a
degraded rank surfaces through `collector_health_status`,
`is_degraded_collector` and the missing ratios of the counters it lost,
instead of silently comparing only the samples that survived.

**Clock skew.** Window and fingerprint scopes use producer timestamps. Stormlog
does not rewrite cross-host clocks; see [Correlation](correlation.md).

## Connection To Issues And Summaries

- **Issues.** Fingerprint scope takes a `fingerprint_id` from
  `stormlog query issues` and compares the issue's hits with the events around
  them, which answers "what is different when this issue fires?".
- **Sessions and summaries.** `stormlog query summary` and
  `stormlog query sessions` point at the rank, session or window worth
  comparing; the comparison result names the sessions, ranks and time span on
  each side, so you can pivot to `stormlog query events` or
  `stormlog query correlate` with the same identifiers.

## TUI

In the Diagnostics tab, load live telemetry or artifacts, select a rank row,
and press `Compare Rank`. The comparison table ranks the fields that separate
that rank from the other ranks shown, and the log reports both sides' sizes,
the top field and any low-confidence warning. The rank filter limits the
baseline to the ranks it keeps.
