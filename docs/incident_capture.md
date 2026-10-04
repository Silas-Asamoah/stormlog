[← Back to docs](index.md)

# Inference incident capture

`stormlog infer watch` keeps a bounded record of a vLLM server's recent past
beside it. When a condition has been bad for long enough, it seals what it has
into an incident bundle, and can open one bounded profiler window. This page
grows with the command (#219). It starts with the bundle format, which the
rest builds on.

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

These limits apply only to what the watcher keeps under its root. vLLM
writes each profiler trace into its own trace directory before the watcher
can measure it, so nothing here bounds that write; see the watcher's
trace-volume settings.

### The recent past in memory

Between incidents the watcher holds its recent scrapes in memory, each
serialized and compressed: a vLLM 0.30.0 scrape takes about 5.7 KB this way.
The memory bound counts each one's compressed bytes plus 320 bytes for the
Python objects that hold it, so it bounds what is retained even when the
items are tiny. A scrape larger than the whole bound is refused and counted
(`oversized`). Only the last few scrapes, as many as the widest trigger
window needs, are also held parsed (about 130 KB each), and only scrapes the
memory holds, so a trigger never judges a scrape its incident's bundle
cannot contain. Scrapes are parsed back one at a time when a bundle is
written.

## Triggers and what "sustained" means

A trigger asks a question of the server's recent `/metrics` scrapes once per
tick (Δ, 1 s by default) and only fires when the answer stays bad for long
enough. Each evaluation looks at a window of the last `W` seconds and is one
of:

- **violating**;
- **clear**;
- **data gap**: the window cannot be judged. A scrape failed or is missing at
  either end, or too few samples arrived; or anywhere inside the window a
  counter went backwards or was recreated, a series changed its labels, two
  scrapes were out of order or at one instant, a histogram's step was not
  itself a histogram, or the exporter restarted. A scrape that failed inside
  the window only leaves fewer samples;
- **masked**: what the evaluation read overlaps the watcher's own profiler
  start or stop and the recovery after it. That is the window from its
  first scrape, which can start up to a tick before `t - W`, and for a
  health trigger the scrapes it reads.

The window's end scrape must have finished within one tick of the
evaluation, and its start scrape within one tick of `t - W`. After an
outage a window is a data gap until its start scrape follows the outage.

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
60 s, `C` = `F`, `G` = `F / 2`, and `clear_tolerance` = `min(2Δ, F / 10)`;
`F` must be at least `W`.

What this guarantees is about the predicate the watcher evaluates, not about
the fault behind it. A 20 s fault can keep a queue observably saturated for
much longer.
- **A violation shorter than `F` never fires.** Accumulation starts at zero
  on the first violating evaluation, so the accumulated time is at most the
  span from the first violating tick to the last. A window predicate stays
  true for at most `d + W` after an observable violation of length `d`, so
  `d < F - W` never fires: with the defaults, anything under 30 s.
- **A lasting violation fires on time.** If the predicate turns violating at
  `a` and stays so, the trigger fires by `a + Δ + ⌈(F + j)/Δ⌉·Δ + j`, plus any
  time it spent paused, where ticks are scheduled every `Δ` and each runs at
  most `j` late. The first violating evaluation comes within `Δ + j`, a late
  first tick shortens the accumulated time by up to `j`, and the firing tick
  can itself be late. With ticks on time and `F` a multiple of `Δ`, that is
  `a + Δ + F`. For a persistent change that a predicate sees only once its
  window is full, `a` is at most the onset plus `W`, so detection takes at
  most `W` more. The session record states each trigger's bound, with `j`
  the scrape timeout, since the watcher evaluates as each scrape returns.
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
