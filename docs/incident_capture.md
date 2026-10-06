[← Back to docs](index.md)

# Inference incident capture

`stormlog infer watch` keeps a bounded record of a vLLM server's recent past
beside it. When a condition has been bad for long enough, it seals what it has
into an incident bundle, and can open one bounded profiler window. This page
grows with the command (#219). Today the watcher scrapes `/metrics`, evaluates
metric, signal and health triggers, and seals metrics-only incidents. Deep
capture, SLO triggers and OTLP spans come in later steps.

## Running the watcher

```bash
stormlog infer watch --root ./watch --base-url http://127.0.0.1:8000
```

The watcher runs beside the server until Ctrl+C or SIGTERM, or for
`--duration` seconds. vLLM only sees `GET /metrics`, once per tick. Nothing
the watcher does can block it: the ledger and the incident store write on
their own threads, and a stalled disk makes them drop and count records
instead of piling them up.

| Option | Meaning |
| --- | --- |
| `--root DIR` | Where the watcher keeps everything (required). |
| `--config FILE` | A `stormlog.infer.watch_config` file; see below. |
| `--base-url URL` | The server. Overrides `server.base_url`. |
| `--metrics-url URL` | Its `/metrics`, when that is not at the server's origin. |
| `--interval SECONDS` | The tick Δ. Overrides `tick_seconds` (default 1). |
| `--duration SECONDS` | Stop after this long. |
| `--ready-file FILE` | Written, with the session id, after the first successful scrape. |
| `--api-key-env NAME` | An environment variable holding the server's bearer token. |
| `--test-trigger every=SECONDS` or `file` | For qualification only; see below. |

Under the root:

```text
watch/
  .watch.lock    held by the watcher that owns the root, for its lifetime
  ledger/        every record the watcher wrote, as an append-only sink
  incidents/     one bundle per incident (see "Incident bundles")
  report.json    the stormlog.report written when the watch ends
  test-trigger   with --test-trigger file: create it to fire a test incident
```

One watcher owns a root at a time. A second one on the same root, which
would interleave its ledger and bundles with the first's, exits 2 before it
writes anything; so does a watcher whose `incidents/` another process holds.

### Each tick

1. Scrape `/metrics`. One scrape is in flight at a time; a tick that comes
   due during a slow scrape is skipped and counted. A response over 8 MiB or
   20,000 series is refused and counted as oversized. A scrape has
   `scrape_timeout_seconds` in all, not per read, so an answer that trickles
   in cannot hold the watch: past the timeout the scrape is counted failed
   and its fetch is left to finish on its own thread. No other fetch starts
   until it has; the ticks meanwhile are counted missed, but still evaluate
   the triggers and seal incidents. SIGINT, SIGTERM or the end of
   `--duration` cut a scrape short at once.
2. Keep the scrape in the bounded history (`history.seconds`, 600 by
   default, and `history.bytes`, 32 MiB; see "The recent past in memory").
   A scrape larger than the whole bound is counted as oversized too, and
   no trigger judges it.
3. Seal incidents whose post-window has ended, so they hold no open slot
   against this tick's firings.
4. Evaluate every trigger (see "Triggers and what "sustained" means").
5. Open an incident for a trigger that fires, or join it to one still
   collecting its post-window.
6. Append an `infer.watch_health` record.

At start, and then every 60 ticks, retention removes the bundles over the
store's limits, and records each in an `infer.incident_pruned` record. A
seal the store cannot hold first removes the oldest sealed bundles to make
room (see "Disk limits"), and each of those is recorded the same way,
before the incident.

An exporter restart between two scrapes is recorded as a health incident at
once, without a sustain.

### Configuration

Every key is optional except `server.base_url`, which the command line can
give instead. An unknown key, a wrong type, a number that is not finite
(JSON readers accept `NaN` and `Infinity`), a share outside (0, 1], a tick
under 0.05 s, a tick or scrape timeout over an hour, a history, incident
window or trigger time over seven days, a `max_age_hours` over ten years,
a server URL that is not http or https with a host, or
contradictory settings (a hold shorter than its window, a window longer
than the history, a health trigger that reads more scrapes than the
history holds at the tick, or `pre_seconds` plus `post_seconds` longer
than the history, which the seal reads both windows from) are refused with
exit 2,
before anything is created under the root; a file that cannot be read, or
is not a version-1 watch config, exits 5.

```json
{
  "format": "stormlog.infer.watch_config",
  "version": 1,
  "server": {"base_url": "http://127.0.0.1:8000", "metrics_url": "auto"},
  "tick_seconds": 1,
  "scrape_timeout_seconds": 1,
  "history": {"seconds": 600, "bytes": 33554432},
  "incident": {
    "pre_seconds": 120,
    "post_seconds": 60,
    "max_open_incidents": 2,
    "max_incidents_per_hour": 30
  },
  "store": {
    "max_total_bytes": 4294967296,
    "max_incident_bytes": 1073741824,
    "max_incidents": 50,
    "max_age_hours": 72
  },
  "triggers": [
    {
      "id": "waiting",
      "kind": "metric",
      "window_seconds": 30,
      "hold_seconds": 60,
      "gauge": {"family": "vllm:num_requests_waiting", "at_least": 32}
    }
  ]
}
```

A trigger has an `id`, a `kind` (`metric`, `signal` or `health`), optional
`window_seconds` (`W`), `hold_seconds` (`F`), `clear_seconds` (`C`),
`counts_toward_exit` and `engine`, and exactly one predicate.
`counts_toward_exit` is a JSON boolean, true by default for metric and
signal triggers; a health trigger never counts toward the exit code, and
setting it true is refused.

| Key | Settings |
| --- | --- |
| `gauge` | `family`, `at_least`, optional `share` (1.0) and `min_samples` (2) |
| `counter_rate` | `family`, `at_least_per_s` |
| `histogram_share` | `family`, `above`, `share`, optional `min_samples` (20) |
| `signal` | one of `queue_saturation`, `kv_preemption_pressure`, `prefix_cache_loss` |
| `scrape_failures` | optional `consecutive` (3) |
| `scrape_failure_share` | optional `share` (0.05) and `scrapes` (60) |
| `frozen_exporter` | optional `ticks` (5) |

Without `triggers`, the defaults (`watch_defaults/1`) watch the
`queue_saturation` and `kv_preemption_pressure` signals, three consecutive
failed scrapes, a share of failed scrapes (at least 5% of the last 60, held
for 60 s), and a frozen exporter. Failures too sparse to come three in a row
leave every window trigger judged on fewer scrapes; the share reports them.
A config that lists its own `triggers` keeps the three health triggers
beside them, so the watch never loses sight of its own scraper: give a
trigger the same id to replace one, or set `"default_health_triggers":
false` to drop them. The share needs `history.seconds` of at least 60,
and of at least 60 ticks: with the default 600 s history, a tick over 10 s
needs a longer history.
The `export` section belongs to the exporter (#220) and is passed through.

A server running several engines (vLLM's data parallelism) labels each
engine's series apart, and a trigger judges one engine. Name it with
`server.engine` for every trigger, or with a trigger's own `engine`, as the
`engine` label gives it (`"0"`, `"1"`). A trigger that names none, on such a
server, is never judged: every window is a data gap with the reason
`engine_required`, and the watch ends unsound (exit 1). To watch every
engine, give each its own trigger.

The session record in the ledger holds the resolved configuration, every
trigger as resolved (its `W`, `F`, `C`, `G`, clear tolerance, policy and
predicate), its SHA-256 digest, so two configs that watch the same way have
the same digest, and for each trigger the shortest violation that can fire
it (`F − W − Δ`, at least 0) and its detection bound in seconds. The `export` section, which may hold
credentials, is #220's and is left out of both. The watcher evaluates its triggers as
each scrape returns, so an evaluation can run up to `scrape_timeout_seconds`
after its tick: the bound counts that lateness (`j` in "A lasting violation
fires on time" below), and a window's end scrape, or a health trigger's
newest, may be a tick plus that timeout old.

### Incidents

An incident's pre-window reaches back from the firing to the start of the
trigger's first violating window, at most `pre_seconds`; an exporter restart
or a test trigger, which has no window, reaches back the whole
`pre_seconds`. Its post-window runs `post_seconds` past the firing. Firings
within the post-window, its last instant included, join it, and widen its
pre-window to their own: at most 16 triggers in one incident. A joining
trigger that counts toward the exit code makes the incident count, and a
metric or signal trigger joining a health incident takes it out of
`health_only`. A test trigger never joins an incident and is never joined.

When the post-window ends, the incident is sealed: its windows, the scrapes
inside them and its `infer.incident` record go into `gen-0/incident.jsonl`
of its bundle, an inference artifact that
`stormlog.infer.correlation_events.load_inference_artifact` reads. The
record's `status` is `completed`, or `interrupted` when the watch stopped
first. Each window says how complete it is: `complete`, `partial` or
`missing` (no successful scrape). A window is partial when a scrape in it
failed, when the history began late, when it was cut short (an interrupted
post-window is judged against the whole `post_seconds`), or when ticks
were missed: fewer scrapes were attempted than one a tick, give or take
one. `fidelity_detail.scrapes` gives the counts: attempted, expected, ok
and failed.

The watcher decides on its monotonic clock and writes times on the wall
clock as it read it at the start, plus the time elapsed since: a wall-clock
step during the watch (NTP, a manual change) moves no window, but then the
bounds an incident records and the wall times its scrapes carry differ by
the size of the step.

The `infer.incident` record goes to the ledger, and into `report.json`, also
when its bundle could not be written (a full disk, say): its `bundle` is then
null and `bundle_error` says why. A watch whose every incident write failed
is unsound (exit 1).

At most `max_open_incidents` collect at once, and at most
`max_incidents_per_hour` open in any trailing hour, per lane. Metric and
signal triggers have one lane; health and test triggers share one side
lane, with the same limits. So an exporter that restarts over and over,
or a qualification run, never turns away an incident from a metric or
signal trigger, though health and test incidents use up each other's
budget: two health incidents in an hour with a limit of two turn the next
test trigger away. Up to twice each limit can be reached in all: the
session record states the totals as
`incident.max_open_incidents_total` and
`incident.max_incidents_per_hour_total`, which a consumer sizes by. A firing turned away is counted
by reason in `stormlog_watch_suppressed_total`: `rate_limit`, `open_limit`,
or `join_limit` when the incident it would join already holds 16 triggers. When the watch stops, open
incidents are sealed as `interrupted`.

### Records and health

The ledger holds the frozen `stormlog.infer.watch/1` records:
`infer.watch_session` (`started`, `ended`), `infer.trigger_state`,
`infer.incident_event`, `infer.incident`, `infer.incident_association`,
`infer.incident_finalized`, `infer.incident_pruned` and
`infer.watch_health`. Every value that can become a metric label comes from
a closed vocabulary, and an incident's `loss` holds a fixed set of integer
counters, null when their source was not running. One record of each type is
in `tests/fixtures/watch/records_v1.jsonl`, which a test keeps equal to what
the code writes.

An exporter (#220) observes every record as it is queued for the ledger,
the ones the ledger then drops included, so an exported record may be
missing from the ledger; the ledger's own drops are counted in
`ledger_dropped`. `infer.watch_session` records begin and end each watch,
with its resolved configuration.

The watcher's own health is held in memory as `stormlog_watch_*` gauges and
counters (`tests/fixtures/watch/watch_stats_v1.json`): history bytes and
age, scrapes by outcome, missed and frozen ticks, incidents by trigger kind
and capture status, windows by fidelity, suppressions, bundles removed by
reason (the `infer.incident_pruned` reasons, `disk_full` included), and
records each sink dropped. `report.json` carries a copy at the end of the
watch, with the store operations that raised on the store's worker (a
retention pass, say) counted in `metrics.store_operations_failed` and the
last one's error in `payload.store_last_error`. The families grow with the commands that fill them (deep capture,
the span ring, finalization), so a newer fixture has more; none is renamed
or removed within version 1.

The incident limits' trailing hour is kept in memory: a restarted watch
starts a new hour. Keeping it across restarts comes with deep capture,
which keeps its own state under the root.

### Exit codes

| Code | When |
| --- | --- |
| 0 | The watch ended and no incident counted toward the exit code. |
| 3 | At least one incident from a counting trigger (`metric` or `signal` by default) was recorded. A firing the limits turned away does not count: it left no incident to look at. |
| 1 | The watch was unsound: no scrape ever succeeded, a trigger needed an engine named, the ledger lost records, every incident write failed, the store or ledger did not finish within the shutdown deadline, or the report could not be written. `report.json`, when written, lists the reasons under `payload.unsound`. |
| 2 | A setting it cannot use, or a root another watcher is using. |
| 5 | A config file it cannot read. |

Ctrl+C and SIGTERM are the documented way to end a watch: the exit code is
still one of the above, never 130. On the first, the watch seals what is
open and waits for its writers within the shutdown deadline (30 s: two
thirds for the incident store, the rest for the ledger). A second cuts that
to 5 s, shared the same way: a writer, the store's or the ledger's, still
busy at the end of its share is left behind and the watch is unsound, and
the root stays locked until the process exits, since that writer may still
write there. Once `--duration` has begun the shutdown, the first signal
cuts it short at once and the second does nothing more. A third is the
default interrupt. From the
report's write on, further signals are ignored, so the process exits with
the code the report holds. A watch run from Python code through
`stormlog.infer.cli.main(["watch", ...])` gives the handlers back when it
returns. Run off the main thread, it installs no handlers and touches none,
so only `--duration` ends it.

### Test triggers

`--test-trigger every=SECONDS` fires a `test` incident on that period (at
most seven days, or the command exits 2);
`--test-trigger file` fires one whenever `<root>/test-trigger` appears,
records the file's mtime as `requested_at_ns`, and deletes it. Test
incidents follow the same limits as any other and never count toward the
exit code. They exist to qualify the capture path, not to watch a server.

## Incident bundles

Each incident is one directory under `<watch root>/incidents/`:

```text
inc-20261003T120000Z-0001-1f2e3d4c/
  .lock            readers, and the generation's writer, hold it shared;
                   deletion takes it exclusively
  gen-0/           written when the incident is sealed
  gen-1/           written by finalization, next to gen-0 until it is published
  manifest.json    names the current generation and lists its files
```

One process owns a store: it holds `incidents/.store.lock` exclusively for
its lifetime, and a second one on the same root fails at once
(`StoreInUse`). A generation is pinned while it is written, so no deletion
can take it from under its writer, and it is published only if every file
the writer put in it is still there.

### Publication

A bundle is written in generations, so a reader always sees one complete
generation:

1. The new generation is written whole, and its files and directories are
   fsynced.
2. `manifest.json` is replaced: written to a temporary file, fsynced, renamed
   over the old one, and the bundle directory fsynced. The manifest names the
   current generation (`current`), and lists each of its files with its size
   and SHA-256.
3. Only then is the previous generation deleted, and only when no reader
   holds the bundle. A later generation hard-links unchanged files, such as
   raw traces, from the previous one instead of copying them.

A crash leaves either the old manifest with its whole generation, or the new
one with its whole generation. When the watcher next starts, before it prunes
anything, it removes any generation the manifest does not name. A bundle a
crash left with no manifest is sealed as `interrupted`, with `complete:
false`, from whatever its `gen-0` holds. One that holds nothing is removed
after an hour.

An incident's membership is frozen when it is sealed: its manifest has
`membership_frozen: true`. A trigger that relates to it later is recorded in
the watcher's ledger, and the bundle does not change for it.

### Reading a bundle

From Python, pin the bundle for the whole read:

```python
from stormlog.infer.watch.store import open_incident_bundle

with open_incident_bundle("watch/incidents/inc-20261003T120000Z-0001-1f2e3d4c") as view:
    print(view.manifest.status, view.manifest.current)
    records = view.file("incident.jsonl").read_text().splitlines()
```

While the bundle is pinned, no generation it names is deleted. A tool that
cannot take the lock reads `manifest.json`, then the files it names. If one
of them has gone, a newer generation was published in between, so it reads
the manifest again: `read_manifest_snapshot` does this up to three times.
That only guards against a stale manifest, since a file can still go before
it is read; `read_bundle_file(bundle, "incident.jsonl")` reads one file
without the lock and retries when it vanishes.

### Disk limits

Every byte a bundle's generations hold is charged to the store's budget
before it is written:
- A raw trace is hard-linked in (copied, where no link can be made) and
  charged its actual size first. Its original name is removed only once the
  generation is published, so an abandoned generation leaves it where it
  was, and anything it grew by meanwhile is charged at publication. A trace
  in a directory the watcher cannot write is copied, not linked: its name
  could never be removed, and its producer could rewrite the bundle's copy
  through it.
- A file written by the watcher is charged chunk by chunk, before each chunk
  reaches the disk.

A write that would go over its allowance abandons the generation before
anything is published, and frees what it had charged. A file hard-linked
between generations is counted once.

The budget charges the bytes of files, not the disk they take. A file can
take up to one filesystem block more than its bytes, and has an entry in the
manifest; neither is charged. A generation holds at most 1,024 files, which
bounds both: about 4 MiB of blocks and 200 KiB of manifest per generation.

| Limit | Default | When it is reached |
| --- | --- | --- |
| `max_total_bytes` | 4 GiB | the oldest sealed bundles are removed to make room for a new bundle or generation, never one still open, being finalized or being read; one that could not fit even then is not written, and nothing is removed for it |
| `max_incident_bytes` | 1 GiB | one bundle's bytes across its generations, a file linked forward counted once: a reservation over it is refused, and a write or link that would take the bundle over it fails before anything is published |
| `max_incidents` | 50 | the oldest sealed bundles are removed |
| `max_age_hours` | 72 | bundles sealed earlier are removed |

Retention removes the oldest seal first, never a bundle that is still open
or being finalized. A bundle a reader holds is skipped, and removed once the
reader lets go. A bundle whose manifest cannot be read (corrupt, or written
by a newer Stormlog) is charged like any other and removed once its
directory is older than `max_age_hours`.

The budget counts bytes, not the disk: a disk with less free space than
`max_total_bytes` fills first. The watcher says so when it starts (in its
session record, and on stderr). A seal the disk refuses (ENOSPC) removes
the oldest sealed bundles not still open or being read, each recorded with
the reason `disk_full`, until the disk has the seal's bytes free, and is
tried again once, so the newest incidents are the ones kept. When removing
them all could not free that much, or the seal still fails once room is
made, something else is filling the disk: no bundle is removed for it, or
no more, and the incident is recorded without its bundle.

These limits apply only to what the watcher keeps under its root. vLLM
writes each profiler trace into its own trace directory before the watcher
can measure it, so nothing here bounds that write; see the watcher's
trace-volume settings.

### The recent past in memory

Between incidents the watcher holds its recent scrapes in memory, each
serialized and compressed: a vLLM 0.30.0 scrape takes about 5.7 KB this way.
The memory bound counts each one's compressed bytes plus 384 bytes for the
Python objects that hold it, so it bounds what is retained even when the
items are tiny. A scrape larger than the whole bound is refused and counted
(`oversized`). Only the last few scrapes, as many as the widest trigger
window or the longest health-trigger tail needs, are also held parsed (about
130 KB each), and only scrapes the memory holds, so a trigger never judges a
scrape its incident's bundle cannot contain. Each scrape's status and
expanded size are kept beside it, so sealing an incident reads none of them
on the watcher's loop; the bundle is written from the compressed copies on
the store's thread, expanded one at a time and never parsed.

## Triggers and what "sustained" means

A trigger asks a question of the server's recent `/metrics` scrapes once per
tick (Δ, 1 s by default) and only fires when the answer stays bad for long
enough. Each evaluation looks at a window of about the last `W` seconds and
is one of:

- **violating**;
- **clear**;
- **data gap**: the window cannot be judged. A scrape failed or is missing at
  either end, or too few samples arrived; or anywhere inside the window a
  counter went backwards or was recreated, a series changed its labels, two
  scrapes were out of order or at one instant, a histogram's step was not
  itself a histogram, or the exporter restarted. A scrape that failed inside
  the window only leaves fewer samples: the window is judged on the others,
  a counter is differenced across the failure, and the evaluation records
  how many failed (`failed_scrapes` in its detail);
- **masked**: what the evaluation read overlaps the watcher's own profiler
  start or stop and the recovery after it. That is the window from its
  first scrape, which can start up to a tick before `t - W`, and for a
  health trigger the scrapes it reads.

The window's end scrape must have finished within one tick plus the scrape
timeout of the evaluation, since a tick can land while a slow scrape is
still in flight, and its start scrape within one tick of `t - W`. The start
scrape is the latest that qualifies, so at a steady cadence of one scrape
per tick a window's samples span `W - Δ`: that is why the first full window
of the example below is in at 29 s, not 30 s.

An outage is therefore a data gap while it covers the window's end, and
again, `W` later, while it covers `t - W` (less a tick at either side). In
between, while the outage lies inside the window, the window is judged on
the scrapes around it. The failed-scrape share health trigger reports a
scraper that fails now and then.

| State | On | Next |
| --- | --- | --- |
| inactive | violating | pending, with nothing accumulated yet |
| pending | violating | pending, adding the time since the previous evaluation; fires once that reaches the hold time `F` |
| pending | masked; a data gap of at most `G` in all since the last judged evaluation; clear, with at most `clear_tolerance` of clear time in all since it went pending | pending, the clock paused |
| pending | more data gap than `G`, or more clear time than that | inactive (a reset, recorded with its reason) |
| firing | violating, masked or data gap | firing |
| firing | clear | resolving |
| resolving | violating | firing again: the same episode, counted as a re-entry |
| resolving | masked or data gap | resolving, its clock paused |
| resolving | `C` of clear in all, counted from the evaluation after the one that started resolving | inactive; the trigger can fire again |

Masked time never counts toward `G`. The defaults are `W` = 30 s, `F` =
60 s, `C` = `F`, `G` = `max(F / 2, 2Δ)` (never under two ticks, so one late
scrape and one refused cannot reset a trigger), and `clear_tolerance` =
`min(2Δ, F / 10)`; `F` must be at least `W`.

What this guarantees is about the predicate the watcher evaluates, not about
the fault behind it. A 20 s fault can keep a queue observably saturated for
much longer.
- **A violation shorter than `F` never fires.** Accumulation starts at zero
  on the first violating evaluation, so the accumulated time is at most the
  span from the first violating tick to the last. A window's first scrape is
  the latest to return within one tick of the window's start, so it may
  have returned up to `Δ` before it. A window predicate therefore stays true
  for at most `d + W + Δ` after an observable violation of length `d`, from
  the start of the first violating scrape to the response of the last, and
  `d < F - W - Δ` never fires: with the defaults and 1 s ticks, anything
  under 29 s.
- **A lasting violation fires on time.** If the predicate turns violating at
  `a` and stays so, the trigger fires by `a + Δ + ⌈(F + j)/Δ⌉·Δ + j`, plus any
  time it spent paused, where ticks are scheduled every `Δ` and each runs at
  most `j` late. The first violating evaluation comes within `Δ + j`, a late
  first tick shortens the accumulated time by up to `j`, and the firing tick
  can itself be late. With ticks on time and `F` a multiple of `Δ`, that is
  `a + Δ + F`. For a persistent change that a predicate sees only once its
  window is full, `a` is at most the onset plus `W`, so detection takes at
  most `W` more. A tick that runs early counts in `j` too: with ticks up to
  `e` early and `l` late, `j` is `e + l`; on time to the millisecond with
  `j` = 0, one tick 2 ms early can fire a tick later than the bound. The
  session record states each trigger's bound, with `j` the scrape timeout,
  since the watcher evaluates as each scrape returns.
- **Resets restart the count.** After a reset, the bound counts again from
  the next violating tick. Data gaps longer than `G` that keep coming back
  leave no bound at all. A scrape-health trigger reports them: consecutive
  failures, or a share of failed scrapes, since isolated failures leave each
  window judged on fewer samples.

Worked example, with `W` = 30, `F` = 60 and `G` = 30. The queue is saturated
from 0 s, and every scrape fails from 89 s to 121 s:
1. The first full window is in at 29 s, and the trigger goes pending.
2. By 88 s it has 59 s of the 60 s it needs. The outage at 89 s stops it.
3. At 119 s more than 30 s of data gap have passed, and it resets.
4. Windows can be judged again once their start scrape, at 122 s, follows the
   outage: at 151 s. The trigger goes pending again and fires at 211 s.

A test reproduces this from scrapes.

### Predicates

Every figure is engine-wide: it covers all of the server's traffic. A
server running several engines (vLLM's data parallelism) labels each
engine's series apart, and a predicate reads one engine's: without an
engine named, every window over such a server is a data gap, with the
reason `engine_required`.

| Predicate | Fires when | Judged on |
| --- | --- | --- |
| a gauge at or above a value | every sample in the window, or a chosen share of them, is at or above it | the window's minimum, or the share |
| a counter's rate | the counter rises at least as fast as the threshold | the lower bound of its rate, from the window's timing uncertainty, timed on the watcher's monotonic clock so a wall-clock step cannot bend it |
| a histogram's share above a value | more than the chosen share of the window's observations are above the value | the lower bucket bound: a value between two bucket bounds gives an interval `[lo, hi]`, and only `lo` can fire |
| a #218 signal (`queue_saturation`, `kv_preemption_pressure`, `prefix_cache_loss`) | the signal exceeds its threshold in #218's shared table | the signal's own window rule; the incident says the mechanism is *suspected* |
| scrape failures | the last `k` scrapes all failed; a failed scrape is evidence here, not a gap | the scrapes |
| failed-scrape share | at least a chosen share of the last `n` scrapes failed, consecutive or not | the scrapes |
| a frozen exporter | over the last `k + 1` scrapes, requests run or wait and no progress counter moves | the gauges and the generation and prompt token counters |

A health trigger reads the last scrapes, whenever they finished. When none
has finished for a tick plus the scrape timeout (a wedged scraper), its last
verdict is not today's: the scrape-failure triggers count that as
violating, and the frozen exporter as a data gap (`no_recent_scrape`).

A trigger on a histogram vLLM records when a request completes (e2e, TPOT,
the `request_*` families) has its masked window widened by the completion
horizon. Requests delayed by a capture's pause finish up to one request
lifetime later.
