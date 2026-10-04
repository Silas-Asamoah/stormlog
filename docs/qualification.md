# Qualifying diagnosis and capture

Stormlog's diagnosis of a vLLM server is qualified by injecting known faults
and benign workload changes into a running server and scoring what the
diagnoser reports against what was injected. `stormlog.infer.qualify` holds the
parts of that protocol that are library code, so the harness, the scorer and
other issues' fixtures share one definition.

## Exact bounds

Every gated claim is a one-sided bound at 95%, computed exactly. The
exploratory ones (DX-ON, TP2) are reported with the two-sided exact 95%
interval, `clopper_pearson_interval(k, n)`: 6 of 6 gives [0.54, 1] and 8 of 8
gives [0.63, 1].

| Claim | Bound | Function |
| --- | --- | --- |
| Accuracy in a stratum | Clopper–Pearson lower bound on the share of episodes diagnosed correctly | `clopper_pearson_lower(k, n)` |
| False-positive rate | Clopper–Pearson upper bound on the share of negative runs with a false claim | `clopper_pearson_upper(k, n)` |
| False claims per negative hour | the exact Poisson upper bound on the rate | `poisson_rate_upper(k, hours)` |
| Victim impact of an episode | one-sided Fisher exact test of SLO violations in the effect window against the baseline | `fisher_greater(...)` |

For scale: 15 correct episodes out of 15 give a lower bound of 0.819, and 14
out of 15 give 0.721. No false claim in 60 negative runs bounds the rate at
0.0487, and none in 4.9 negative hours bounds it at 0.61 per hour.

## Ground truth: `stormlog.qualify.injection/1`

The harness writes one record per attempted episode into a run's
`truth/injections.jsonl`, which only the scorer reads. A record says:

- **what was injected:** `episode_type` (the catalog ID, such as `F2`), the
  `cause_class` (`fault`, `workload_change`, `instrumentation`, `placebo` or
  `none`), and the method, target and dose;
- **what a correct diagnosis is:** `expects`, a kind at a location, by
  default a primary fault claim at `warning`;
- **which other findings are neutral:**
  - `secondary`: each one names the edge from #218's table that leads to it,
    such as `kv_preemption_pressure->queue_saturation`;
  - `allows`: findings neither credited nor counted as false;
- **when:** `times`, on the victim's clock (`clock_domain`):
  - the action's onset and end;
  - the effect's onset and end, taken from the reference channel at event
    time, and the basis for each;
  - when recovery held;
  - for the short twins (S-x), how long the observable predicate held;
- **whether it counts:** `validity` in four layers, and a `status`.

```json
{"format": "stormlog.qualify.injection/1", "episode_id": "q221-0f3a9c1b2d4e5f60",
 "episode_type": "F2", "cause_class": "fault",
 "injected": {"method": "neighbor_traffic", "dose": {"concurrency": 8, "input_tokens": 2048, "output_tokens": 1024}},
 "expects": [{"kind": "kv_preemption_pressure", "component": "kv_cache", "rank": null, "engine": null,
              "role": "primary", "cause": "fault", "min_severity": "warning"}],
 "secondary": [{"kind": "queue_saturation", "component": "scheduler", "rank": null, "engine": null,
                "edge": "kv_preemption_pressure->queue_saturation"}],
 "allows": [{"kind": "load_increase", "component": "workload", "rank": null, "engine": null}],
 "times": {"action_onset_ns": 0, "action_end_ns": 0, "effect_onset_ns": 0, "effect_end_ns": 0,
           "effect_basis": "reference_hook_preempted_victim", "first_observation_ns": 0,
           "recovery_held_at_ns": 0, "predicate_duration_ns": null, "priming_check": {"passed": true}},
 "clock_domain": "<the victim artifact's clock domain>",
 "actions": [{"kind": "neighbor_phase", "at_wall_ns": 0, "at_mono_ns": 0, "result": "ok"}],
 "validity": {"actuation": "ok", "realization": "realized", "observation": "complete",
              "impact": {"status": "impact", "reason": null, "p_value": 0.001,
                         "effect": {"violations": 9, "met": 31, "unknown": 0},
                         "baseline": {"violations": 4, "met": 131, "unknown": 0}},
              "realized_mechanisms": ["kv_preemption_pressure"], "checks": []},
 "status": "valid"}
```

A null location field means "not specified". Kinds, components, causes and
severities are #218's closed vocabulary.

`parse_injection` refuses a record that would be scored wrongly, listing
every problem:
- a fault episode expects exactly one finding, of cause `fault` at
  `warning`; any other episode expects no fault;
- a workload kind is expected only as `workload_change` at `info`, as #218
  claims it;
- times are integers or null, and an effect doesn't end before it begins;
- a `valid` episode has actuation `ok`, realization `realized`, and both
  its effect onset and end.

`load_injections` also refuses an episode written twice.

**The run** (`stormlog.qualify.run/1`, `truth/run.json`). Each episode names
its `run_id`, and one run record holds what the harness measured on the
victim's clock: the measured window, and the priming, baseline and
final-recovery windows, any run-level protocol failure (a failed priming
check), and the run's actions. The scorer derives the run's negative
exposure from it and the run's episodes (`negative_exposure`). Every
sub-window must lie inside the measured one, and `score_run` refuses an
episode on another clock than its run's: the two are compared. `score_run`
also checks a run and its episodes built in memory as if they had been read
from files, and raises `GroundTruthError` on what parsing would refuse. It
refuses an episode outside the run's measured window, and `summarize` a run
scored twice. A run record needs its `run_id`, its `clock_domain` and a
measured window with length.

```json
{"format": "stormlog.qualify.run/1", "run_id": "q221-dxoff-b03-r07",
 "clock_domain": "<the victim artifact's clock domain>",
 "measured": {"start_ns": 0, "end_ns": 0}, "priming": {"start_ns": 0, "end_ns": 0},
 "baseline": {"start_ns": 0, "end_ns": 0}, "final_recovery": {"start_ns": 0, "end_ns": 0},
 "protocol_failure": null, "actions": []}
```

**The four validity layers:**

1. **Actuation:** the action took place. Signals were delivered and the
   stopped state confirmed; the neighbor's rate held; a configuration took
   effect; a kill was confirmed.
2. **Realization:** the mechanism occurred, on the victim's own requests
   where it can be attributed to them.
3. **Observation completeness** of the diagnosed configuration's capture.
   This is reported, never used to drop an episode.
4. **Victim impact:** whether the episode raised the victim's SLO violations.
   `assess_impact` gives `impact` when the effect window has at least three
   violations, and a one-sided Fisher exact test against the baseline gives
   p < 0.05. A missed SLO, an unreachable request included, is a violation;
   an unknown outcome is not. It gives `partial` (reason
   `slo_evidence_coverage`) when outcomes are known for less than 0.9 of the
   window's requests, and (reason `no_baseline_outcomes`) when no baseline
   outcome is known to compare with.

**Status.** `decide_status` returns the first failure, in this order:

| Status | Failure |
| --- | --- |
| `protocol_failure` | a failure such as a failed priming check |
| `incomparable` | the victim's clock domain differs |
| `not_actuated` | the action didn't happen |
| `invalid_alignment` | the episode isn't aligned (below) |
| `not_realized` | the mechanism didn't occur |
| `recovery_incomplete` | recovery timed out |

Otherwise the status is `valid`.

An episode is aligned (`is_aligned`) when its action and effect lie wholly
inside the victim's measured window, with at least 30 s of clean time before
the action, counted from when the window opened or the previous episode
recovered. Every attempted episode is published; accuracy is computed over
the `valid` ones.

## Effect timing, realization and recovery

The harness keeps a reference channel beside every diagnosed configuration:
the execution hook, a tailer, and scrapes every second. `recovery` reads it as
time series on the victim's clock (`Signals`). Those are the victim's own
admission waits and cached fractions, the preemptions of its requests, the
waiting and KV-usage gauges, hook step starts, the victim's chunk gaps, and
the intervals during which a victim request was in flight. The injector's own
times are `Actions`. Nothing here depends on the diagnosed
configuration's capture.

`Baseline.measure` takes these from the baseline segment:
- the p95 wait;
- the range of the waiting count;
- the maximum KV usage;
- the busy step gaps' and the chunk gaps' count, mean, p95, p99 and p99.9,
  and how many were over twice the p99. A step
  gap counts only its busy part: from the later of its first step and the
  moment the victim's current in-flight interval opened, to its second step.
  Idle time measures the traffic, not the engine, but a request that arrived
  in idle time and waited for a step felt that wait. A gap whose second step
  falls in idle time is dropped. `in_flight` has
  no default: a caller that doesn't know the intervals says None, and then
  no gap is busy and cadence recovery never holds;
- the median cached fraction.

`effect_timing(episode_type, context)` gives each mechanism's onset, its end
(the start of the first interval over which its recovery criteria hold) and
when recovery held. The earliest such interval is found exactly: what an
interval sees changes only where a sample leaves it or enters it, so every
such start is tried.

| Episodes | Effect onset | Recovered when, for 10 s |
| --- | --- | --- |
| F1 / T1 | the first 5 s window whose median victim wait exceeds the baseline p95 (F1); the neighbor's first send (T1) | over at least 20 waits and 5 waiting counts, the first of each back in band, no more waits above the baseline p95, and no more counts outside the baseline's range, than chance allows (`MostlyWithin`, the same Binomial allowance as cadence); and none far out: no wait above twice the baseline's p99, no count above twice its highest (or 2); and not often near those bounds: the hold's mean wait at most 1.25 times the baseline's (the cadence rule's 20% rate tolerance), its mean count at most 1.25 times the baseline's, and at least 1 for a near-empty queue. So recurring saturation bursts don't pass as chance, whether far out or just under the bounds: rev-220-b's bursts one second in five, under both bounds, recovered with bursts to come in 10–17 of 20 runs, now 0–2. Bursts one second in ten still pass in 11–18 of 20: a 10 s hold catches about 3 of the burst's 30 waits and one of its 10 scrapes, within chance and the 1.25× means |
| F2 / T2 | the first victim preemption (F2); the neighbor's first admission (T2) | no victim preemption, and KV usage at most the baseline maximum + 0.05 |
| F3 / T3 / T3b | the first 5 s window whose median victim cached fraction is below 0.5 (F3); the neighbor's first send (T3, T3b) | the first sample, the median of the hold's first 5 s (the window the onset is found in) and the median over the hold are all at least 0.9. A hold that began inside a dip, or just before one shorter than half the hold, would end the effect before the dip did |
| F4a / F4b / H0 / P | the first stop confirmed (state `T`) | from the last `SIGCONT`, for 10 s, the busy step gaps look like the baseline's (below); for F4b, the victim's chunk gaps too |
| W1 | the neighbor's first send | the queue and KV criteria |
| F5 / R0 | the first stop confirmed | as F4a |
| I1 | the stop request | at the stop's return plus #219's drain |
| N | its scheduled slot's start (`Actions.slot_ns`) | at the slot's end |

A short twin (`S-<x>`) follows its fault's rules. A workload twin (T1, T2,
T3, T3b) leaves the signals alone, so its recovery holds at once; its effect,
the benign change, lasts until its action ends (`Actions.action_end_ns`), as
N's slot does, so its scoring window is not just the grace.

**Cadence** (`CadenceWithin`) is judged like with like, over at least 20
busy gaps in the hold:
- their mean is within 20% of the baseline's rate (a long gap weighs by its
  length, so a slow minority shows);
- no more of them are longer than twice the baseline's p99 than the
  baseline's own share of such gaps allows by chance (the 99% point of
  Binomial(n, share); none when the baseline had none), and none is longer
  than twice its p99.9;
- no more of them lie above the baseline's p95 than chance allows: the 99%
  point of Binomial(n, 0.05).

**A gap still open at the hold's end counts.** A gap is known only when its
next step arrives, so the time from the hold's last step to its end is a gap
still open: its busy part (since the last step, or since a victim request
arrived after it) joins the hold's gaps for the long-gap and mean rules. An
engine that resumes for a moment and then hangs has a long gap at every
hold's end until it steps again, and a live poll during a hang finds no
recovery. A hold found earlier, in an idle stretch, can still be complete
when a request sent since is stuck, so `engine_stalled(context, now)` reads
the present: whether the busy part of the gap still open at `now` is longer
than twice the baseline's p99. The harness says START only while it isn't.

**A baseline too thin to compare with never recovers.** A criterion whose
baseline has fewer samples than it needs in a hold (20 busy gaps, 20 waits,
5 waiting counts) never holds, so the episode times out as
`recovery_incomplete` instead of recovering against infinite thresholds.
`Baseline` records each count. `recovery_blocked(episode_type, context)`
names each series that is too thin, and by how much (`baseline_too_thin: 3
waiting counts of the 5 a hold needs`), so that a timeout's record can say
why recovery never held.

An engine back at its baseline recovers at once; one still degraded doesn't.
The tests hold both: engines with 40–51% of steps 10× slow, every step 2× or
3× slow, bimodal stalls, slow steps among idle gaps, pulses the injector
never recorded, a slow resume, and a minority 1.6× slow, against jittered
engines that must recover in every seed.

**The rule's resolution.** The search tries every start, so a stretch of a
degraded engine that looks normal for a whole hold is found. Over the 10 s
cadence hold, of 20 seeded engines degraded for 60 s, a uniform 1.15× slow
engine never recovers more than 10 s early; one with 1% of gaps 20× slow
(a 400 ms stall every 2 s or so) does in 1, and one with 5% of gaps 3× slow
in 4. A 5 s hold let 11–13 of 20 through. Milder or rarer degradation than
that can end an effect early, and the thresholds are refrozen from
`dev_v1` with this in view. A healthy jittered engine recovers at the last
`SIGCONT` in 39 of 40 seeds, and within 15 s in all of them.

**What the minimum assumes of the victim.** Twenty busy gaps in the 10 s
hold need about 2 busy steps a second: a victim busy 5% of the time with
25 ms steps falls short, and every pulse episode, H0 and P, then times out
as `recovery_incomplete` (rev-220-b measured 2 of 20 recovering at a 5%
busy share against a 5 s hold). #221's victims keep a request in flight
nearly all the time, and G0 records each victim's busy share so a light one
is caught before a campaign.

**Rare long steps are normal, up to the smallest dose.** When fewer than 1%
of steps are prefill steps, twice the p99 falls below one. A hold may then
have as many gaps over twice the p99 as the baseline's own share of them
predicts: the 99% point of Binomial(n, share), about 2.8 times the
baseline's rate over a 10 s hold. No gap may be longer than twice the
baseline's p99.9, nor than `long_gap_tolerance_cap_s`, the smallest
F4a/F4b dose (60 ms). Where twice the p99 is already longer than that,
nothing is tolerated beyond it.

With 5 ms decode steps and 25 ms prefill steps (under the cap, as G0
checks real engines are), 20 seeds each:
- engines with 0.4–1.5% prefill steps recover at the last `SIGCONT` in
  19–20 of 20 (one seed at 1% took 6.2 s, a prefill step in the tail past
  the cap). Without the tolerance, 12–17 of 20 recovered within 150 s at
  0.6–1.0%, after a median of 19–35 s;
- the blind spot is stalls shorter than the cap. At 0.8% prefill, 30–50 ms
  stalls every 1–2 s after the last pulse are missed in 20 of 20, and
  70 ms, 100 ms or 300 ms stalls in none.

The cap also stops a contaminated baseline from widening the tolerance.
Twice the p99.9 rests on a baseline's few largest gaps. Without the cap,
three 1 s pauses in a 20 ms engine's baseline let 1–1.5 s stalls every
10 s recover early in 16 of 20; with it, in none.

An engine whose prefill steps are longer than the cap falls back to the
strict rule, where each prefill step ends a hold. In rev-220-b's probe,
0.8% prefill steps of 250 ms recover at the `SIGCONT` in 3 of 20. G0
rules this out by checking every dose against the measured step times. If
G0 fails that check, step kinds go into `Signals`, and prefill steps are
left out of the hold instead. Where twice the p99 exceeds 300 ms (2%
prefill steps of 250 ms), 300 ms stalls every 2 s pass in 15 of 20, as
they did before the tolerance.

**Cadence is blind while the victim is idle.** Only busy gaps count, so a
stall that falls wholly in victim idle time (about a fifth of the time at
3 requests per second) is not seen by recovery. That recovery after a pulse
train is real rests on actuation: every pulse's stop and continue are
confirmed, and pulses the injector never recorded are among the tests.

`realization(episode_type, context, timing)` applies the catalog's checks:

| Episode | Realized when |
| --- | --- |
| F1 | its onset was reached, with no victim preemption |
| F2 | at least one victim request was preempted |
| F3 | the victim's cached fraction fell below 0.5 |
| F4a | no hook step started during a pulse |
| F4b | the stopped state was confirmed. Whether the engine kept stepping in every pulse with a victim request in flight is recorded but doesn't gate: if it didn't, `added_mechanisms` adds `host_stall@engine_core` to the realized set, as A.4 says |
| F5 | the stopped state was confirmed, and the peer rank's NCCL wait lengthened (`Actions.peer_wait_extended`, from Nsight) |
| T1 | the waits stayed within the baseline (no window's median wait above its p95; the check's value is the highest window median over the p95, so a twin that just crossed it is told from one that saturated) |
| T2 | nothing was preempted |
| T3 | the cached fraction stayed at 0.9 or more |
| T3b | as T3, and the engine-wide prefix hit ratio fell at least 0.05 below the baseline's median |
| H0, R0 | the stopped state was confirmed |
| I1 | the capture started and stopped |
| W1, P, N | A.4's column is empty: realized when the action took place |

Each check is judged over the effect and the whole action (to
`Actions.action_end_ns`): a twin leaves the signals alone, so its effect
window is empty, but its neighbor runs on. Each check is recorded with its
value, and whether it gates.

**Two checks look only at the victim's median or the victim's requests.**
- **The cache checks use the median.** F3 is realized when the median cached
  fraction falls below 0.5. T3 is benign while that median stays at 0.9 or
  more, and F3's recovery holds once it is back at 0.9. Up to half of the
  victim's requests can therefore still miss. In rev-220-b's probe, F3
  recovered at the neighbor's stop in 20 of 20 runs with 20–45% of requests
  still missing. A T3 whose neighbor evicts 40% of the victim's prefixes is
  still a valid negative, and a correct cache-loss claim there counts as
  false. The design defines F3 and T3 this way. A share bound like the
  queue's would be stricter.
- **F1 counts only the victim's preemptions.** A dose that fills the KV
  cache and preempts the neighbor's requests is still a clean
  queue-saturation episode. G0's calibration keeps F1 below preemption,
  since the fake engine and vLLM preempt the newest request, which may be
  the victim's.

**A missing reference signal leaves its check incomplete.** When the
reference channel has no samples for a check (no scraped hit ratio in the
baseline or the episode, say), the check neither passes nor fails; it is
recorded `incomplete`, and `observation_of` makes the episode's observation
`incomplete`. The other checks decide. An episode whose every gating check is
incomplete is not realized: nothing was judged. A type with no
rule, a typo such as `F4A` or the outages X1–X3 (judged by C.6's own
criteria), raises `KeyError` rather than passing vacuously.

Two functions drive a run:
- `priming_check` is the run's precondition: the victim's median cached
  fraction over the last 10 s of priming is at least 0.9. Otherwise the run
  is a protocol failure.
- `next_episode` lets the next episode start once the previous one's recovery
  has held, and no sooner than 60 s after its action ended. Recovery that
  holds only after 150 s, or not by then, is a timeout, whenever the harness
  asks, and the run's remaining episodes are skipped.

The thresholds are frozen from `dev_v1` in `Thresholds`; the defaults are the
design's.

## Scoring: `score_v1`

`score_run(run, injections, diagnosis, config)` scores a run's episodes
against the run's diagnosis: a `stormlog.report` from `diagnose_artifact`,
or its payload. `score_episode` scores one episode alone. `summarize(runs,
config)` turns the runs into the claims. The rules are frozen before any
evaluation data is drawn.

**One finding, one episode.** A finding that qualifies for several episodes
of a run goes to the one whose effect began latest at or before its start
(`assign_findings`), so it is never credited twice.

**The candidate set.** These are every finding, of any kind, subject or role,
that passes the temporal rule, less the neutral secondaries. The set is
ranked by #218's total `rank`, and top-1 and top-3 are taken over it.

- **The temporal rule.** Take
  `S = [effect_onset − pre_grace, effect_end + grace(kind)]`. Here
  `pre_grace` is the finding's `window.resolution_ns`, at most
  `ScoreConfig.max_resolution_ns` (30 s: #218's resolution is its first
  flagged window's span, and it joins 1 s windows up to its 30 s span cap),
  plus its `window.uncertainty_ns`, at most `ScoreConfig.max_uncertainty_ns`
  (5 s; #218 emits none on one host). Both bounds are fixed, frozen with
  `score_v1`, never taken from a claim. A run's problems count, separately,
  the findings that claim more resolution and those that claim more
  uncertainty. `grace` is frozen per finding kind (`ScoreConfig.grace_ns`). A finding qualifies
  when it starts inside `S` and at least half of its window lies inside. A finding with no window, or a run-wide one, never does, nor
  does one whose window ends before it starts or names a clock domain
  other than the victim's. A finding with a severity or role outside #218's
  vocabulary is refused (`ValueError`, naming it).
- **A neutral secondary** names an upstream in `secondary_to` that passes
  four checks:
  - it is an eligible candidate;
  - it matches one of the label's `expects`, `secondary` or `allows` entries;
  - one of #218's edges joins the two, with each end at a component the edge
    allows. The table (`diagnosis_edges_v1`) is #218's PR 2 table, E1–E6,
    copied edge for edge until that PR lands below this one, then imported:
    E1 is `capture_pause@profiler` → `host_stall@engine_core`;
  - the secondary's window lies inside the upstream's window ± grace.

  Any other secondary is scored as if it were primary.

**A match** has all of these:
- the label's kind;
- `role: primary`;
- an eligible claim;
- the label's cause;
- at least the label's severity;
- the label's location. At L1 that is the component; at L2 it also needs the
  rank and engine where the label names them. They are read from #218's
  location block: the engine is `location.engine_producer`, the hook
  producer that names the engine, and the rank `location.rank`. A label
  names the engine by the reference hook hello's producer.

A secondary of the right kind never matches, because the diagnoser said the
mechanism followed from something else. Nor is it a false claim, because it
names the true mechanism.

**False claims.** A false claim is a fault claim among the candidates that
has all of these properties:
- it is eligible, has cause `fault`, and is at `warning`;
- it is not neutral;
- it is not of the label's kind at the label's component;
- it is not in `allows`. A `secondary` entry exempts a finding only by
  making it neutral, through its valid edge in time and place (A.4); a
  declared kind in any other role, or without that edge, counts.

In a negative run it is a false positive; in a fault episode it is counted as
spurious.

**Misses**, each counted against accuracy:

| Label | The episode |
| --- | --- |
| `outranked` | has a match, but below the gated top-k |
| `mismatch` | has an eligible primary of the label's kind with the wrong cause, severity or location |
| `secondary_only` | has the label's kind only as secondaries |
| `ineligible` | has the label's kind only where #218 made it `claim: observation` |
| `coverage_gap` | has none, and #218 didn't assess the kind at the label's component |
| `no_finding` | has none, and the kind was assessed there |

The labels apply in that order. A kind counts as assessed at a component
as #218 says per component (`coverage.<kind>.components`) for a kind that
spans components, and by its own status, `assessed`, for any other: #218
PR 1b reports `host_stall` assessed at `api_server` but not at `engine_core`
or `worker`, so an F4b miss at `api_server` is `no_finding`, and an F4a miss
a coverage gap.

**The claims.**

| Claim | Population | Gate |
| --- | --- | --- |
| Accuracy per episode type (top-1 at L2) | `valid` fault episodes, one stratum for every type the support matrix (`ScoreConfig.supported_types`, required) declares. A declared stratum with no valid episode has no bound and fails; each records its excluded episodes by status | Clopper–Pearson lower bound ≥ 0.78 in every stratum |
| False-positive rate | negative runs: a run holding exactly one `valid` episode of C.5's eight negative types (T1, T2, T3, T3b, H0, W1, P, N). A run's false claims are counted over its whole negative exposure: the measured window less the priming and the span of every attempted fault or instrumentation episode, from its onset to its effect end plus the longest grace of any kind (a finding of any kind is assigned to the episode up to its own grace). A claim's window is clipped to the measured window and placed when at least half of what is left lies in the exposure, wherever it starts: a claim over the whole run is the plainest false positive. A claim with no window counts unless one of the run's injected faults expects its kind; then it may be about that fault, and is reported as unplaced (never credited to it either). A claim on another clock than the run's can't be placed, so it counts (failing closed), and the run's problems say how many findings were off its clock. I1 and outages are not negatives; a run with two negatives is reported, not counted. A unit needs at least `ScoreConfig.min_exposure_ns` (60 s) of exposure, and its negative episode's own window wholly inside it. Every run with a negative that isn't a unit is counted by why (its episode's status, `protocol_failure`, `several_negatives`, `exposure_below_minimum` or `negative_outside_exposure`) in `excluded_negative_runs`, since each one shrinks the denominator: 0 of 52 bounds the rate at 0.056 | upper bound ≤ 0.05 |
| False claims per negative hour | the same claims over the same exposure, summed over the negative runs | descriptive: the exact Poisson bound |
| Incident attribution | fault episodes with victim impact | descriptive |
| Condition localization | fault episodes. An eligible finding of the label's kind at its location counts, in any role, cause or severity | descriptive |

Spurious claims, duplicate matches, the miss labels and the secondary
error rate (secondaries that weren't neutral, over all the fault episodes'
secondaries) are reported alongside. `Summary.to_record(config)` records
everything the score froze: its version, the edge table's version, the
gated metric and level, the targets and confidence, the grace per kind
(the default shown for a kind without its own), the support matrix and the
negative types.

## The harness

> **Source checkout only.** `examples.qualification` is not shipped in the
> PyPI package.

### The reference channel

Every run keeps a reference channel beside the configuration being
diagnosed, and the harness reads only it to judge effect timing, realization
and recovery. `examples.qualification.reference` is the vLLM 0.30 binding's
reader:

- **`HookTailer`** reads the execution hook's raw log, under the run's
  `truth/reference/hook`, as it is written. It takes complete lines only and
  never reads a record twice; a sealed segment continues from where its
  `.part` was read. It notes when each record was first seen
  (`probes/hook-firstseen.jsonl`) and when each segment was sealed
  (`probes/seal-observations.jsonl`). The replay uses these times to cut the
  hook log to what an online analyzer could have read.
- **`VictimView`** keeps the victim's series from the engine's records. Victim
  requests are the ones whose `X-Request-Id` carries the victim's run prefix.
  - **Wait:** from its `alias` to the start of the step that first schedules
    it.
  - **Cached fraction:** that step's prefix-cache hit over the victim's
    shared prefix, at most 1.
  - **Preemption:** the request's ID in a step's `preempted`.
  - **Cadence:** every step's start is kept.
- **`scrape_metrics`** reads `/metrics` once: the waiting count summed over engines
  and the highest KV usage. `ReferenceChannel` takes one per poll, into
  `truth/reference/scrapes.jsonl`.
- **`chunk_gaps`** rebuilds the victim's gaps between streamed chunks from its
  client records.

`ReferenceChannel.signals()` returns them as the `Signals` that
`stormlog.infer.qualify.recovery` reads.

### The signal pulser

`examples.qualification.pulser` stops a serving process and continues it:
F4a and F4b pulse EngineCore and the API server, F5 a TP worker, and H0 its
5 ms twin. A pulse is `SIGSTOP`, a confirmed stop, a wait and `SIGCONT`:

- **The right process.** A target is its pid and its start time, checked
  before every signal, so a recycled pid is never signalled.
- **A confirmed stop.** After `SIGSTOP` the pulser polls until the process is
  stopped, within 1 s, and records the latency.
- **Always continued.** The target is continued:
  - in a `finally` around each pulse;
  - at `atexit`;
  - by `Pulser.close`;
  - by SIGTERM and SIGHUP handlers, which continue every target before the
    harness exits (their default action would skip `finally` and `atexit`);
  - by a watchdog (`examples.qualification.watchdog`) in a session of its
    own, ignoring SIGINT, SIGTERM and SIGHUP, so a signal to the harness's
    process group (Ctrl+C, a job's SIGTERM, an ssh disconnect) never reaches
    it. It reads a pipe from the harness: end of file means the harness is
    gone, however it died, and the target is continued at once. It also
    continues a target stopped more than 1 s past the longest pulse.
- **No stop without a watchdog.** The pulser waits for the watchdog to say
  it is ready before its first stop, replaces a watchdog that died before the
  next one, and refuses to pulse if it can't.
- **Caps.** A pulse lasts at most 2 s, at a duty cycle of at most 50%.

Each pulse's stop, confirmation and continue times are kept, so effect timing
can start from the first confirmed stop.

`discover_roles(api_server_pid)` names the processes under a vLLM API server by
the titles vLLM 0.30 gives them: `EngineCore`, `Worker_TP0`, `Worker_TP1`.
Each is returned as a target with its start time, so a later signal reaches
the same process.

### Neighbor traffic

`examples.qualification.neighbor` injects another tenant's load: F1, F2 and
F3, and their workload twins.

- **A neighbor** is an `infer profile` run of its own, on a thread of the
  harness. It has its own run ID, so the hook tells its requests from the
  victim's by their `X-Request-Id`. It runs with a high in-flight limit and
  writes its artifact under `truth/`.
- **Arrivals.** An open-loop neighbor arrives at a fixed rate, so its plan is
  a schedule, not a distribution. A closed-loop one runs a number of workers,
  as F2's eight concurrent long requests do.
- **Actuation** is judged from the neighbor's own artifact:
  - an open-loop neighbor must reach its planned rate within 5%, with no
    arrival held for a slot;
  - a closed-loop one must keep every worker busy;
  - any failed request is a problem.

  Its first send is the onset of the workload twins' effect.

### The capture pause (I1)

`examples.qualification.capture.capture_window` opens one profiler window and
closes it, stamping when each call was requested and when it returned. A start
that fails is never followed by a stop. vLLM writes the trace inside the stop
call while its step loop waits, so the stop's interval, plus #219's drain, is
I1's effect.

### The catalog and the plan

`examples.qualification.catalog` is A.4's catalog: each episode type's label
(`expects`, `secondary`, `allows`), its cause class, and how it is injected.

| Method | Types |
| --- | --- |
| Neighbor traffic | F1, F2, F3, T1, T2, T3, T3b, W1 |
| Pulses to a role's process | F4a, F4b, H0, P (P pulses a sidecar) |
| One profiler window | I1 |
| Nothing | N |

T2's mixed prefill is `allowed` rather than secondary, because no #218 edge
leads from a workload change to it. The short twins (S-x), the TP=2 types
(F5, R0, F6) and the outages (X1–X3) are refused for now: they need #219's
predicates, a second GPU and #220's tools.

A plan, `stormlog.qualify.plan/1`, holds:
- the profile and a seed;
- the binding (`vllm-0.30`);
- the victim's workload: rate, token shape, prefix groups, shared-prefix
  ratio, SLO;
- the timeline: priming, baseline, episode length, the 60 s minimum and
  150 s timeout for recovery, final recovery;
- the episodes in order, each dose filled from the catalog's defaults.

`load_plan` refuses a plan that can't be run, listing every problem: an
unknown type, a neighbor without a rate or a concurrency, a pulse past the
pulser's caps, a capture that isn't between 0 and 60 s.

```json
{"format": "stormlog.qualify.plan/1", "profile": "dx-off", "seed": 7,
 "victim": {"rate_per_second": 3.0, "input_tokens": 512, "output_tokens": 64,
            "prefix_groups": 4, "shared_prefix_ratio": 0.75},
 "episodes": [{"type": "F2"}, {"type": "N"},
              {"type": "F1", "dose": {"rate_per_second": 24, "input_tokens": 128, "output_tokens": 16}}]}
```

### The victim

`python -m examples.qualification.victim --probes DIR -- <infer profile
arguments>` runs the profile exactly as `stormlog infer profile` would, and
adds three probes in its own process:

- **Phase markers** in `DIR/markers/`, one file per phase start and end. The
  harness times its episodes against the measured window from them, while the
  run is still going.
- **The append-time probe:** each artifact line's index and when its append
  was flushed, in `DIR/append-times.jsonl`. A client record is ready for an
  analyzer then, and the replay cuts the client artifact by these times.
- **The client idle probe:** a 10 ms timer on its own thread. A tick that
  comes 20 ms late or more is noted in `DIR/client-idle.jsonl`, so a stall
  on the client's own host is seen.

### The inject command

```bash
python -m examples.qualification inject --plan PLAN.json --out ROOT \
  --base-url URL --model M --reference-channel HOOK_DIR \
  [--label q221-...] [--target engine_core=PID --target api_server=PID ...] \
  -- [extra infer profile arguments for the victim]
```

The harness never launches the server; #213's `run_plan` does. It is given
the server's URL, the pid of each role a plan may pulse, and the hook
directory the server writes (`STORMLOG_VLLM_HOOK_DIR`, as this host sees it).
One run goes:

1. **Start.** The victim starts, at the plan's rate with its shared-prefix
   prompts, and the reference channel is polled every second.
2. **Priming,** then the priming check. If it fails, every episode of the run
   is a `protocol_failure`, still published.
3. **Baseline,** measured for recovery's thresholds.
4. **Episodes, in the plan's order.** Each waits until the previous one's
   recovery has held, at least the minimum after its action, and its own
   clean time (30 s by default) after the previous effect ended. Then it is
   actuated by neighbor traffic, pulses, a profiler window or nothing, and
   its recovery is watched live. A recovery timeout skips the remaining
   episodes; they are published as `not_actuated`.
5. **Final recovery.** The victim is stopped, and each episode's
   `stormlog.qualify.injection/1` record is written to
   `truth/injections.jsonl` with:
   - its effect timing and realization checks;
   - its impact on the victim's SLO (when the plan sets one), counted by
     arrival in the effect window against the baseline;
   - its status.

The run is published atomically (see below).

**The run directory** is named by an opaque label, `q221-<16 hex>`, that says
nothing about its episodes:

```text
<root>/<label>/
  run/      victim.jsonl            the only path handed to the diagnoser
  truth/    injections.jsonl, episodes.json, plan.json, neighbor-<n>.jsonl, reference/
  probes/   markers/, append-times.jsonl, client-idle.jsonl, hook-firstseen.jsonl,
            seal-observations.jsonl, victim.log
  SHA256SUMS
```

It is written under `<root>/.<label>.partial`. Once `SHA256SUMS` is written
last, the directory is renamed into place, so a reader never sees half a
run. `run_dir.verify` checks a run against its sums.

**The victim's outcomes** for impact are a stand-in for #213's
`evaluate_request`, used until #213 lands, and the rule is the same:
- **Met:** a request that met the SLO.
- **Violation:** one that missed it, or failed, being timed out, rejected,
  in error, or never sent.
- **Unknown:** one that was cancelled.
