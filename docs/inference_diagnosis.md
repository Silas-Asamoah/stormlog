[← Back to main docs](index.md)

# Inference diagnosis

Stormlog's diagnoser explains a slow request or a slow window of an
inference run from the evidence the run captured: client latency, vLLM's own
metrics and spans, the scheduler steps of the
[execution hook](vllm_execution.md), and imported GPU traces. Each finding
names a mechanism with the observations and records behind it, how far it
can be trusted, the competing mechanisms and what became of each, and an
experiment that would confirm it; a class that cannot be checked says so,
with a reason. This version assesses queue saturation, KV preemption
pressure, prefix-cache loss, client admission, host stalls at the API
server, capture pauses and the four workload kinds. Mixed-prefill
interference, rank delay and transfer degradation are `unsupported` in the
coverage with the reason `not_assessed_by_this_version`, and `host_stall` is
`partial` because its engine-loop and worker forms are not assessed yet.
The page also documents the threshold table, the cheap signals an online
trigger evaluates over a window of `/metrics` scrapes, and engine-loop
stalls in the hook's raw records.

## Diagnose an artifact

```python
from stormlog.infer.diagnosis import DiagnoseOptions, diagnose_artifact

report = diagnose_artifact("infer.jsonl")  # automatic incident selection
report = diagnose_artifact("infer.jsonl", windows=[(start_ns, end_ns)])
report = diagnose_artifact(
    "infer.jsonl", options=DiagnoseOptions(request_ids=("c1_in8_out4_measured_0_7",))
)
```

`diagnose_artifact` reads only the artifact it is given, once, and returns a
validated [`stormlog.report` v1](report_contract.md) report with
`report_kind: inference_diagnosis`. Its exit code is 3 when any finding is a
`warning`, else 0. The same artifact and options give the same report; the
generation time can be fixed with `DiagnoseOptions(generated_at_ns=...)`.
Thresholds can be overridden by key (`DiagnoseOptions(thresholds=...)`); an
unknown key, or a value that is not a finite number, is refused.

The payload, `stormlog.inference_diagnosis` v1
([schema](schemas/inference_diagnosis_v1.schema.json)), holds:

| Block | What it says |
| --- | --- |
| `diagnoser` | the version, a SHA-256 over the thresholds and options that decide the result, and the generation time |
| `inputs` | the artifact's path, size, SHA-256 and line count, so a reader can tell whether it changed |
| `outcome` | `findings`, `no_findings`, or `inconclusive` when an incident has no eligible explanation or automatic selection could test no window. A declared subject whose TTFT and end-to-end excess intervals both reach zero was no slower than its reference: it is no incident to explain, and the summary ends "no excess in the declared window" |
| `join` | what was joined: client requests, dispatch and first-content records, engine executions, engines and their clocks |
| `selection` | every analysis window with its tests, and the subjects |
| `coverage` | per kind: `assessed`, `partial` or `unsupported`, with reasons, per subject |
| `findings_detail` | per finding ID: its claim, cause, role, eligibility, confidence, location, window, detection time and evidence, observations, alternatives, experiment, up to 8 display pointers and its full support |
| `thresholds` | the table version, the overridden keys and every value used |

The envelope's findings carry the verdict, flat metrics and up to 8
evidence pointers: `path` (relative to the report's directory when the
report is written to a file), `pointer` `/<line>`, and `record_id`.

## Finding kinds

Every finding names one kind, located at one component, with one cause. The
vocabulary is closed (`stormlog.infer.diagnosis_vocabulary`): a new kind is a
change to the diagnosis payload's schema.

| Kind | Component | What it means |
| --- | --- | --- |
| `queue_saturation` | `scheduler` | Requests wait to be scheduled because the engine is at capacity |
| `kv_preemption_pressure` | `kv_cache` | Running requests are preempted because KV blocks ran out, and are recomputed |
| `prefix_cache_loss` | `prefix_cache` | Requests that should reuse a cached prefix find less of it cached |
| `mixed_prefill_interference` | `scheduler` | Decode steps slow down because they share steps with other requests' prefill |
| `host_stall` | `engine_core`, `worker` or `api_server` | The process that should make progress does not: the engine loop, a worker's launches, or the API server |
| `rank_delay` | `worker` | One tensor-parallel rank arrives late, and the others wait in the collective |
| `transfer_degradation` | `interconnect` | Collective or copy time grows on every rank at the same work |
| `capture_pause` | `profiler` | A stall caused by Stormlog's own profiler window |
| `client_admission` | `client` | Stormlog's client held requests back (`--max-in-flight`) |
| `load_increase`, `longer_inputs`, `longer_outputs`, `prefix_sharing_drop` | `workload` | The workload asked for more; always reported at `info` |

A cause is one of `fault`, `workload_change`, `instrumentation` and
`undetermined`.

## The command

```bash
stormlog infer diagnose infer.jsonl                       # automatic incidents
stormlog infer diagnose infer.jsonl --window START,END    # a declared window, ns
stormlog infer diagnose infer.jsonl --request ID --request ID
stormlog infer diagnose infer.jsonl --case ID             # one case's incidents
stormlog infer diagnose infer.jsonl --output diagnosis.json --format json
stormlog infer diagnose --inspect diagnosis.json FINDING_ID [--all]
```

`--window` and `--request` declare subjects and may be repeated; declared
requests are compared, like a declared window's, with the unflagged requests
of their cases that arrived before the first of them, and a difference of
medians needs 20 in each arm, so one request alone cannot be explained.
`--case` (repeatable) keeps automatic selection to those cases' incidents;
`--window-seconds` sets the base window of automatic selection;
`--thresholds FILE` overrides entries of the threshold table from a JSON
object; `--metrics-from-engine` asserts that the scraped metrics exporter is
the engine whose hook log was imported. `--output` writes the validated
report; `--format` prints the text view (the default) or the report as JSON.

The text view gives the verdict and outcome, the coverage of every kind, and
each finding with its claim, cause, confidence per claim, observations,
competitors, experiment and up to 8 `path:line record_id` pointers.

`--inspect REPORT FINDING_ID` never diagnoses again. It reads the saved
report and prints the records the finding rests on, by physical line: its
display pointers, or with `--all` its whole support. It finds the artifact
where the report recorded it, else relative to the report, else beside it,
so a report moved with its artifact still resolves. When the artifact's
SHA-256 differs from the report's, each record is found by its ID and its
line's hash checked, with a warning; a record that changed is an error, and
a finding whose support was kept only as line ranges stops with
`support_unresolvable_after_modification`. A missing artifact or report
exits `5`.

| Code | When |
| --- | --- |
| 0 | the diagnosis completed with no `warning` finding, including `inconclusive` |
| 3 | at least one `warning` finding |
| 2 | a usage error: no artifact, a malformed `--window`, an unknown threshold key or one that is not a finite number |
| 5 | an unreadable artifact, report or threshold file |
| 1 | anything unexpected |

## Citing records

A diagnosis cites the artifact records behind each finding. It reads the
artifact once, by physical line: a line's number counts from 0 and includes
blank lines, and each line is kept with its SHA-256 (of its bytes without the
newline) and the record's ID. The ID says what the record is, so a reader can
find it again if the file changed; the whole file's size, SHA-256 and line
count are recorded with the diagnosis so a reader can tell.

| Record | `record_id` |
| --- | --- |
| any v2 record (`schema_version` 2 or 3) | its `event_id` |
| `infer.request`, `infer.dispatch`, `infer.first_content` | `<event_type>/<request_id>` |
| `infer.phase_start`, `infer.phase_window` | `<event_type>/<case_id>/<phase>` |
| `infer.trace_window` | `infer.trace_window/<case_id>/<phase>/<started_at_ns>` |
| `infer.vllm_span` | `infer.vllm_span/<trace_id>/<span_id>` |
| `infer.vllm_scrape` | `infer.vllm_scrape@<observed_at_ns>` |
| `infer.telemetry_sample` | `infer.telemetry_sample@<observed_at_ns>` |
| any other v1 record | `<event_type>@<timestamp_ns>` |

A record without the fields its ID needs has none, and is cited by line
alone.

## Where a request's time went

A request's TTFT, from the client's send to its first content, is split into
segments that tile it, and its end-to-end latency likewise:

| Segment | From | To | Clock |
| --- | --- | --- | --- |
| `send_to_ingress` | the client's send | the engine's admission (`alias`) | client to engine |
| `engine_ingress` | admission | entering the scheduler's queue (`enqueued`) | engine |
| `scheduler_wait` | entering the queue | the `schedule()` call that first ran it | engine |
| `prefill` | that call | the completion of the first step that kept an output token for it | engine |
| `first_token_delivery` (TTFT) | that completion | the client's first content | engine to client |
| `decode` (end-to-end) | that completion | the completion of the step it finished in | engine |
| `final_delivery` (end-to-end) | that completion | the client's end | engine to client |

On a hook log without `enqueued` records the second and third are one
segment, `engine_ingress_to_schedule`, never called a queue wait. A segment
on the engine's monotonic clock is exact. One that crosses between the
client's clock and the engine's is an interval: the engine's stamp is a
bracketed read of the same wall clock when the engine shares the client's
host and boot, so the segment is known to within that bracket. Otherwise it
is unknown, with a reason: `legacy_unbracketed_stamp` for a hook from before
the bracket, `cross_host_continuity_unknown` for an engine on another host,
and `wall_clock_discontinuity` when the wall clock may have jumped between
the two reads. The residual, the client's interval minus the segments, is an
interval too, and exists only when every segment does. A request with more
than one engine execution, or none, is not decomposed.

**Continuity.** The engine's `wall - mono` offset, from every step's
schedule entry and completion, splits its log into continuity segments. A
bracketed sample allows an interval of offsets, from its first wall read to
its second; a segment keeps the offsets all its samples allow, each widened
by 100 ppm of the time since (and never less than 10 us), which covers
NTP's normal slewing, and ends where a sample allows none of them. A thread
descheduled between its reads (19.77 us once in run 1's 55,668 samples, or
a whole SIGSTOP pulse) widens its own interval, which is no jump; a sample
without a second read allows its one offset. A client read pairs with an engine read only when it
lies within the wall span of the engine read's segment, or, before the
first sample or after the last, within one sample gap of it. Two jumps that
cancel between two samples cannot be seen, so a segment is monitored to
within its largest sample gap, not verified.

## What gets explained

A diagnosis explains subjects. A caller can declare them: a window (start
and end on the artifact's clock) or a list of requests; a declared subject
is an incident by declaration, and each class still applies its own tests.
A caller can also keep automatic selection to some cases. Without a
declared subject, the diagnosis selects incidents itself, per case, from
the measured phase:

- **Assignment.** A request belongs to the window of its intended arrival,
  or of its send when it had no intended time, so a window holds what
  arrived in it however long it took.
- **Windows.** Base windows of 1 s are joined forward until each holds 20
  requests, or spans 30 s; a quiet stretch that reaches 30 s is a window of
  its own. A window that holds 20 ends early at the next arrival in its base
  window, which starts the next window: a burst a batch job submits within
  one second still spans many windows, so the sustain rule sees how long it
  lasted. Each boundary is known when it passes, so a prefix of the run
  draws the same windows up to its end.
- **Reference.** A window is compared with every earlier window of its case
  that was not flagged itself, and only once those hold at least 114
  requests, the fewest that bound a p90 under the shared sufficiency rule;
  until then it is `insufficient_reference`.
- **Test.** For TTFT and for end-to-end latency separately, the threshold is
  the reference's p90 (nearest rank), and a one-sided Fisher's exact test on
  (window above, window below) against (reference above, reference below)
  flags the window at α = 0.01, if at least 3 of its requests are above. A
  failed request is above any threshold, on either side, but it is no
  latency: the p90 is taken over the reference's requests that succeeded,
  so a reference with over a tenth failed still has a threshold a window
  of failures can exceed.
- **Error rate.** α is per window and holds only if requests are
  independent. In a continuous-batching server consecutive requests share
  batch state: in a simulation with the selection's own test, 0.2% of
  healthy 30-window runs got a false incident with independent requests,
  11% with a lag-1 correlation of 0.5, and 80% with 0.9. So α is not the
  rate of false incidents, and `selection.error_rate` says so: the test,
  `alpha_per_window`, `assumes: independent_requests`, and
  `calibrated_false_incident_rate`, null until #221's healthy runs measure
  it. The classes' gates, not the test, are what keep a false incident from
  becoming a warning: a bursty run below capacity selects an incident and
  stays `inconclusive`.
- **Incident.** At least two consecutive flagged windows of a case, joined
  into one subject. One bad window alone is not an incident.
- **Onset.** A window joined forward from calm traffic can begin seconds
  before the burst that flagged it. An incident's `onset_ns` is when its
  degradation was first visible: the earliest instant a request of its
  first flagged window that was over a flagged threshold had run longer than
  that threshold (its send plus the threshold). A finding's `window` starts
  there, and its `resolution_ns` is the first flagged window's span, how
  finely selection placed the onset; the subject keeps the windows' own
  start.
- **Abstention.** A window with fewer than 20 requests of its own
  (`too_few_requests`) or too small a reference (`insufficient_reference`)
  is not tested, and the summary counts such windows as untested. When no
  window could be tested, as in a run too short to build a reference, no
  incident was ruled out either, so the outcome is `inconclusive`, not
  `no_findings`; the exit code stays 0.

Every judgement is causal: a window is judged at its evaluation time, the
end of the window after it, from what the artifact held by then. A request
still running then is censored at its elapsed time, and counts as above only
once that passes the threshold. In a growing artifact, a window whose
evaluation time is after the client's last record, with requests still
running, is `not_yet_evaluable`: judging it would read them as if that time
had passed. `first_detectable_ns` is when selection could first have chosen
the subject; a class's own evidence, such as the hook's records up to its
last heartbeat, can come later. A request is known from its send only when
the client wrote `infer.dispatch` records, and its TTFT before its end only
with `infer.first_content` records; without dispatch records, an incident
has no `first_detectable_ns`, and says `legacy_no_dispatch_records`; a
declared subject has none either, and says `declared`. A finding's
`detection_evidence` says what made its incident selectable then (basis
`selection_sustained/1`): the two windows the sustain rule needed, each
with its evaluation time, the metrics it was flagged on, and per metric its
requests above the reference p90 out of those known, the reference's, and
the p-value, read from the client records stamped by
`client_records_through_ns`, which is `first_detectable_ns`. It is null
when there is no detection time. A class's own evidence (hook records,
dated by their `source_seq_max`) can come later; neither field dates it.

**Location.** A finding's `location` names its `component`; an engine-side
finding adds `engine_producer`, and a finding located in one worker adds its
TP or PP `rank` and its `pid`. No kind this version assesses is located in
one worker, so none has a rank yet. An absent field is unspecified.

**Server-only artifacts.** An artifact with no client requests, as an
incident watcher captures one, can still be diagnosed over a declared
window: when no client request arrived in it, the subject is the engine's
executions admitted in it (by their admission's wall stamp, on the same host
and boot), against those admitted before it and in no other declared
window, so an earlier incident is no part of a later one's reference
(`basis: engine`). The queue and KV classes then work from the engine's own
segments, the queue with `partial/no_client_latency` and its contribution
judged against the engine's TTFT (admission to the first step that kept a
token); the classes that need the client report
`unsupported/no_client_requests`.

An incident watcher's bundle (#219) says where it looked: its
`infer.incident_window` records, a `pre` window before each detection and a
`post` window after it, each with `start_ns` and `end_ns` on the bundle's
wall clock. The import keeps every engine step inside either, as it does a
run's phases, and without a declared subject the diagnosis takes each
`post` window as one (`declared_by: incident_window`), against the
executions admitted before it, those of its `pre` window among them. A
bundle without these records keeps another client's steps only inside an
`infer.phase_start` that never ended, imported without `--server-stopped`.

With a declared SLO the test would compare violations instead; that waits
for the SLO policies of the comparison work (#213), and the threshold
version records which test ran.

## How a finding is graded

A finding names a kind at a location for one subject, with its
observations, the competing mechanisms and what became of each (`ruled_out`,
`contributing`, `untestable`, `not_ruled_out`, or `upstream` for a cause that
comes before it), and an experiment that would confirm it. A competitor
measured by the share of the excess it explains itself is `ruled_out` only
below a small floor (10%), not merely below the share that would make it
the explanation instead: between the two it is `contributing`, a real but
minor second cause.

- **Eligibility.** A kind may claim a fault only when its gates hold and
  every competitor indispensable to it is `ruled_out`; one that is
  `untestable` fails the gate as surely as one that is not ruled out. An
  ineligible finding is `claim: observation`, `cause: undetermined`, at
  `info`, and `eligibility.failed` lists why. An indispensable competitor
  that is `contributing` leaves the finding eligible but contested
  (`eligibility.contested`): it stays `claim: condition` at `info`, since
  the incident had a second cause. Other competitors only lower
  confidence.
- **Confidence** is ordinal and per claim. *Condition*, that the mechanism
  occurred, asks for direct evidence, enough samples, robustness to clock
  uncertainty and observed loss coverage. *Contribution*, that it explains
  the incident, asks for an excess whose interval excludes zero, competitors
  excluded and, where the kind needs one, a witness. A claim is `high` with
  everything met, `medium` with one miss or unknown coverage, and `low`
  otherwise or with known loss. `confidence.level` is the lower of the two,
  and a `partial` assessment is at most `medium`.
- **Severity and cause.** `warning` needs an eligible claim, an incident
  subject, a condition and a contribution each at least `medium`, and the
  one contribution criterion that says the mechanism explains the incident
  met: `confidence.contribution.explains` names it (the queue's
  `explains_ttft_excess`, KV's `explains_e2e_excess`, prefix loss's
  `ttft_rose`, client admission's `explains_intended_latency_excess`, and
  `explains_ttft_excess` for the API server and capture pauses). One unmet
  criterion lowers confidence to `medium`, but never that one: a queue that
  explains a sixth of the TTFT rise is not the fault. A warning's cause is
  then `fault`, and with `role: primary` that is the fault claim
  (`claim: fault`); for an instrumentation kind (`client_admission`,
  `capture_pause`) it is `instrumentation`, since Stormlog's own client or
  profiler caused it, and the claim is a condition. Either exits 3. An eligible finding at `info` is `claim: condition`, its
  cause undetermined until a driver says otherwise. Workload kinds are
  always `workload_change` at `info`; instrumentation kinds are
  `instrumentation`.
- **Roles.** Every finding is `primary` in this version, with an empty
  `secondary_to`: the edge table that makes a finding secondary to the
  cause upstream of it comes with the engine-loop class. Until then a
  competitor that is `upstream` (KV preemption, for the queue: it leaves
  requests waiting to resume and keeps new ones out) contests the finding
  like a contributing one: it stays eligible but claims a condition at
  `info`, since it may be the upstream cause's consequence. A cause is
  upstream only when the subject's finding of that kind is eligible; an
  observation establishes nothing, and leaves the competitor
  `not_ruled_out`.
- **Rank.** Findings are ordered by: primary before secondary, eligible
  before observation, confidence, the contribution's lower bound (in ms of
  latency for every kind, so kinds compare), kind,
  location, window start, and finally ID, so the order is total.
- **Identity.** A finding's ID is `diagnosis.<kind>.<12 hex>`, a hash of the
  run, the subject, the kind, the location and the window start: the same
  artifact gives the same IDs.
- **Support.** Every line a finding used is listed as `(line, record_id,
  sha256)` triples, up to 10,000; above that as line ranges, the count and a
  digest of the lines' hashes (`support_identity: ranges_only`).

Differences of medians come with a percentile bootstrap 95% interval
(B = 2000, fixed seed), and need at least 20 values in each arm. Each arm is
resampled in arrival order by a moving-block bootstrap, in runs of
consecutive values as long as the cube root of the arm's size (7 for 300):
waits within a burst rise one after another, and resampling single values
would treat them as independent and give an interval narrower than its
95%.

## Queue saturation

Requests waited to be scheduled because the engine was full. For one
subject, the class compares the median `scheduler_wait` of the subject's
requests with its reference's (a difference of medians with a bootstrap
interval); with no excess it reports `not_observed`.

| Gate | Holds when |
| --- | --- |
| `capacity_witness` | at least half of the subject's requests waited mostly through steps at capacity: of the busy steps scheduled while one waited (from its entry into the queue to the call that ran it, not counting that call), at least half ran the hello's `max_num_seqs` or scheduled `max_num_batched_tokens`. A request that waited through no step waited only for the next one to begin, and was not held by capacity. A step counts the slots the step before freed and it did not refill: under async scheduling vLLM plans a step before the last one's outputs are seen, so a slot freed by a request that reached `max_tokens` is refilled one step late (a request that ended by end of sequence was already planned into the next step, where it is discarded, so it is counted once) |
| `usable_timing` | the wait is `scheduler_wait`; on a log without `enqueued` records it is `engine_ingress_to_schedule`, labelled, and the gate fails |

| Competitor | Indispensable | Ruled out when |
| --- | --- | --- |
| `engine_stall` | yes | engine-loop stalls over their limit, found by the same rules as the online `engine_loop_gap`, during the waits or ending at most their own length before one began, last under 10% of the wait excess in all; up to half of it they are `contributing`. A stall holds a request back by no more than its own length, and one just before the waits counts because a request reaching the engine during a stall enters the queue only when the loop resumes, leaving a backlog that drains after it. A host gap while only queued requests exist, none running, is no engine-loop stall (nothing was ready), so this competitor cannot see it |
| `scheduler_paused` | yes | no pause transition overlaps the waits and the hook observes pauses with nothing lost over them; else, without pause records, the longest stretch without an admission while a subject's request waited (the longest pause that could hide there, since a paused scheduler admits nobody) is under 10% of the wait excess; a longer one is `untestable`, since a full engine admits nobody either. Both are judged over the stretches in which a subject's request waited, not the calm between them |
| `blocked_waiting` | yes | every waiting request's `enqueued` record says it used neither structured output nor streaming input |
| `engine_ingress` | yes | the `engine_ingress` excess is under 10% of the wait excess; from 10% to a quarter it is `contributing` (untestable on a log without `enqueued` records) |
| `kv_preemption_pressure` | no | the subject's own allocation preemptions held its waits for under 10% of the wait excess: the median request's time waiting behind the subject's preempted requests, which vLLM puts back at the head of the queue until they resume. Another client's preemptions, and a reset's, are no evidence of the subject's KV pressure. Up to half the excess it is `contributing`; above that it is `upstream` when the subject's KV finding is eligible, which contests the queue (no fault claim, a condition at `info`), and otherwise `not_ruled_out`, since an observation of KV pressure establishes nothing upstream |
| `client_admission` | no | no request was held at the client |
| `host_stall@api_server` | no | the `send_to_ingress` excess is under 10% of the wait excess; from 10% to a quarter it is `contributing` |

The excess is a median over the subject's requests, so the witness asks
about the requests, not the steps: bursts that overflow `max_num_seqs` by a
few leave every step run while someone waited full, yet the median request
waited only for the next step boundary, as it does when nothing overflows.
The metrics report the requests' share
(`requests_waiting_at_capacity_share`), the steps' share counting late
refills (`steps_at_capacity_share`), and the steps' share by their members
alone (`steps_at_max_num_seqs_share`). The contribution claim also asks that
the wait excess be at least half the TTFT excess. Without engine records the class is
`unsupported/no_server_queue_signal`; without a witness it is
`partial/no_capacity_witness`; with requests on several engines,
`unsupported/several_engines`.

**vLLM's metrics.** Scrapes describe the whole engine between two instants,
never a request. A scraped exporter is bound to the engine whose hook log
was imported only when the operator asserts it with `--metrics-from-engine`;
then `vllm:num_requests_waiting_by_reason{reason="capacity"}` above zero over
the subject's window can stand in for the hook's capacity witness, but only
where the hook measured none (no step ran while a placed wait lasted, or the
hello gave no capacity): engine-global metrics never overrule the steps the
requests waited through. It is a weaker witness than the hook's: any
waiting request counts, and vLLM's capacity reason also counts waits bound
by KV space. `detail.capacity_witness_source` says which held (`hook` or
`exporter`). Without
engine records, the queue and KV classes give at most a window-level
observation from the scrapes in the subject's window (the median waiting
count, or the preemption counter's increase, over its threshold):
`partial` with reasons `aggregate_only` and the missing hook evidence, never
a fault claim, and with `location.exporter_binding` `asserted` or
`exporter_scoped`.

## KV preemption pressure

Running requests were preempted because KV blocks ran out, waited to
resume, and recomputed what they had. The class reads the import's
`engine.preempted` stages over the subject's requests' engine lifetimes.
vLLM also preempts every running request when the prefix cache is reset, and
the import attributes those preemptions to the reset; so the class's gate,
`allocation_cause_established`, holds only when its indispensable
competitor, a reset, is ruled out: the hook records resets (`cache_reset` in
`observes`), none happened over the subject (as a stage, or as a dated fact
the import kept for a reset before any step was written), and the hook's
loss coverage spans it (every `dropped` count, `<kind>_oversized` included, and `errors`
unchanged, the writer not capped). Without that the class is
`partial/preemption_cause_unknown` and the finding is an observation that
preemption and recomputation happened.

Each affected request's cost is observed: the wait from the preempting
`schedule()` call to the call that resumed it
(`preemption_to_resume_entry`), and the positions it computed again below
the highest context it had reached. The contribution claim asks for an
end-to-end excess whose interval excludes zero, and resume waits adding up
to at least half of it over the subject's requests. Without engine records
the class is `unsupported/no_hook_preemption_data`.

## Prefix-cache loss

Requests that should have found their shared prefix cached found less of
it. The class reads the client's declared prefix groups (`prefix_group`,
`shared_prefix_tokens`) and each request's `cached_at_admission` from the
import. A request is *warm* when another request of its group had finished
prefilling the shared span (its `computed_after` reached it) before this
request entered the scheduler; requests of a group used for the first time
together are not warm. What a warm request should find is its group's own
experience: the median `cached_at_admission` of the reference's warm
requests of that group, never a length derived from a tokenizer. The
finding is the subject's warm requests falling short of it: a difference of
medians of the shortfall, with an interval.

Three competitors are indispensable: a prefix-cache reset between the
group warming and the requests entering (ruled out only where the hook
records resets and lost nothing); `prefix_sharing_drop`, ruled out when the
share of requests declaring a shared prefix and the median shared length
both stayed within 10% of the reference's; and
`prefix_working_set_growth`, ruled out when the distinct shared prefixes
the subject's requests declared (in tokens) are at most 1.1 times those of
as many of the latest reference requests. A cache that evicts the least
recently used loses old prefixes to a larger working set without any
fault. Fewer than 3 warm requests is
`too_few_warm_requests`; requests that declare no sharing are
`unsupported/no_declared_sharing`, and without engine records
`unsupported/no_per_request_cache_evidence`.

## Client admission, the API server and capture pauses

**`client_admission`** (cause `instrumentation`): Stormlog's open-loop
client held requests at its in-flight limit (`held_for_slot`) or dropped
them, so they went out late. The class counts held and dropped requests and
compares the subject's dispatch lag (send minus intended arrival) with the
reference's; the contribution claim asks that the lag excess be at least
half the excess of first content measured from the intended arrival. Its
gate `held_or_dropped` asks for a held or dropped request: lag alone says
the client sent late, not why (a starved client thread looks the same, and
no in-flight limit would help it), so such a finding is titled "The client
sent requests later than their arrivals", stays an observation, and its
experiment reruns the client on an idle host. A
closed loop has no intended arrivals: `unsupported/no_intended_arrivals`.

**`host_stall` at `api_server`** (`detail.form: frontend`): requests took
longer to reach the engine, in `send_to_ingress`, while the engine kept
stepping, so the time went in HTTP, the API server or its IPC. Its gates are
engine progress (a step completed inside at least half of the stalled
requests' send-to-admission intervals) and a bounded placement of
`send_to_ingress` for at least half the subject's requests; the metrics say
how many were placed (`send_to_ingress_placed` of `subject_requests`), since
an unplaced one, across a wall clock step say, is left out of the excess.
The client stamps a send before it connects and writes, so a client too
starved of CPU to write promptly also lengthens `send_to_ingress`, and no
competitor rules that out yet: a client on a busy host should check its own
CPU before giving the API server more. Two competitors are indispensable: a scheduler paused for
new requests, which looks the same from the client and is ruled out only by
pause records with nothing lost over every stalled request's interval (each
placed on the engine's clock, ending at its admission), and a capture
pause, ruled out when no profiler window overlaps the stalls. Without engine records it is
`unsupported/no_engine_progress_evidence`; without a placed
`send_to_ingress`, `unsupported/clock_alignment_required`. Host stalls in
the engine loop and the workers are not assessed yet, so `host_stall` is
`partial` in the coverage, whose `components` say so per component:
`api_server` as assessed, `engine_core` and `worker` `unsupported`.

**`capture_pause`** (cause `instrumentation`): a profiler stop, which
blocks the server while it writes the trace, lay across requests waiting for
their first content. A stop can hold a request back by no more than the
part of its wait for first content the stop overlapped, so the contribution
claim asks that the median of that overlap, over the subject's requests, be
at least half the TTFT excess, and the contribution's lower bound is that
median (or the excess's lower bound, when smaller): a 2 ms stop inside a
long queue explains 2 ms of it at most. It needs the stop request's own
stamp, `stop_requested_at_ns` on the `infer.trace_window` record; without it
the class is `unsupported/no_stop_request_stamp`, and a run without profiler
windows has nothing to assess (`no_trace_windows`).

## Workload changes

Four kinds say what the workload asked for, never what failed; they are
always `info` with cause `workload_change`, and an incident explained only
by them is still `inconclusive`. How much of the incident the demand
explains is a driver's question, which this version does not answer, so
their contribution is not assessed (`level: null`, unmet `not_determined`)
and their confidence is the condition's:

| Kind | Found when |
| --- | --- |
| `load_increase` | the subject's requests arrived faster than the reference's, the lower bound of the rate ratio's exact 95% interval (conditional binomial, Clopper-Pearson) at least 1.25 |
| `longer_inputs` | the median prompt grew, the difference of medians' interval above zero and at least 10% longer |
| `longer_outputs` | the same for outputs |
| `prefix_sharing_drop` | the share of requests declaring a shared prefix fell: its exact interval's upper bound at least 0.1 below the reference's share |

## Memory

`payload.memory` is a ledger of what each source says about the server's
memory, never a sum: its sources sample different instants, so categories
are never added or subtracted. Each entry gives its category, scope, source,
unit, status (`observed`, `not_collected`, `unsupported`, or
`exporter_scoped` for metrics not bound to the engine), and, when observed,
its provenance, cadence, clock and peak (`sampled_max` over the run).

| Category | Source | Status |
| --- | --- | --- |
| `physical_device`, `physical_mig_instance` | NVML through `infer collect-server` (`--server-telemetry`) | observed when given |
| `physical_process_gpu`, `host_process_rss` | the same collector | observed when given |
| `allocator_allocated`, `allocator_reserved` | the collector's allocator counters | observed when given; nothing in vLLM's server reports them yet |
| `runtime` (CUDA context, NCCL, workspaces) | none | unsupported |
| `cuda_graph_pools` | none | unsupported |
| `kv_blocks_allocated` | `vllm:kv_cache_usage_perc` times the hello's `num_gpu_blocks` | observed with `--metrics-from-engine`, else `exporter_scoped` |

The KV figure counts blocks held by running requests: in tokens these are
capacity slots, not live tokens, and cached blocks that are free count as
free. `nesting` says whether the categories nest: under the default caching
allocator (the hello's `enable_cumem_allocator` false), KV blocks lie within
the KV pool, within what the allocator reserved, within the process's GPU
memory, within the device, with runtime memory outside the allocator; with
any other allocator, or a hello that does not say, `holds` is null.

`stormlog infer diagnose ... --server-telemetry JSONL` (repeatable), or
`DiagnoseOptions(server_telemetry=...)`, names the collector artifacts to
read; they and the artifact are the only files a diagnosis reads.

## Thresholds

Online triggers and the diagnoser read thresholds from one versioned table,
`stormlog.infer.diagnosis_thresholds` (version `diagnosis_thresholds_v1`), so
they cannot disagree about what a threshold is. A caller may override an
entry; every result records the table version and whether it did. An override
of a key the table lacks, or with a value that is not a finite number (a NaN,
a string, a bool), is refused, as is a window floor (`min_scrapes`) below 2.
The values are provisional until they are read from real runs.

| Key | Value | Meaning |
| --- | --- | --- |
| `queue_saturation.median_waiting_requests` | 1 | requests waiting in the window's median scrape |
| `kv_preemption_pressure.preemptions` | 1 | preemptions counted in the window |
| `prefix_cache_loss.hit_ratio_drop` | 0.2 | fall of the prefix-cache hit ratio below the caller's reference |
| `prefix_cache_loss.min_queried_tokens` | 2048 | tokens the window must have queried the prefix cache for before its ratio decides |
| `prefix_cache_loss.working_set_ratio` | 1.1 | how much larger the subject's declared working set of shared prefixes may be than the reference's before it can explain the loss |
| `host_stall.stall_factor` | 10 | an engine-loop stall is at least this many times the median completion cadence before it |
| `host_stall.stall_floor_ns` | 50 ms | and at least this long |
| `host_stall.no_baseline_floor_ns` | 500 ms | or, with no earlier busy steps to compare with, at least this long |
| `host_stall.baseline_window_ns` | 30 s | the window of earlier busy steps the cadence is taken from |
| `host_stall.min_busy_steps` | 20 | busy steps at least as large as the stall's that window needs |
| `host_stall.matched_bin_min_steps` | 20 | steps of the stall's own work bucket (scheduled tokens within a factor of two) needed to compare it with steps of its size |
| `host_stall.heartbeat_grace_ns` | 3 s | how recently the hook's writer must have been heard from to judge a stall still going on |
| `queue_saturation.witness_request_share` | 0.5 | share of the subject's requests that must have waited mostly through steps at capacity for a witness |
| `queue_saturation.ttft_excess_share` | 0.5 | share of the TTFT excess the wait excess must reach to explain it |
| `queue_saturation.stall_excess_share` | 0.5 | share of the wait excess the engine stalls during or just before the waits must last to explain it instead |
| `queue_saturation.front_excess_share` | 0.25 | share of the wait excess an excess before the queue (engine ingress, the API server) must reach to explain it instead |
| `queue_saturation.kv_hold_share` | 0.5 | share of the wait excess the median request must have waited behind the subject's preempted requests for KV pressure to be upstream of the queue |
| `queue_saturation.competitor_floor_share` | 0.1 | share of the wait excess below which a competitor is ruled out; above it, up to the competitor's own share, it is contributing |
| `load_increase.arrival_rate_ratio` | 1.25 | lower bound of the arrival rate ratio for a load increase |
| `workload.length_ratio` | 1.1 | how much longer median prompts or outputs must be |
| `prefix_sharing_drop.share_drop` | 0.1 | how far the share declaring a shared prefix must fall |
| `selection.window_seconds` | 1 | base window of incident selection |
| `selection.span_cap_seconds` | 30 | longest a joined window may span |
| `selection.min_requests` | 20 | requests a window is joined until it holds |
| `selection.reference_min_requests` | 114 | reference requests before a window is tested |
| `selection.alpha` | 0.01 | level of the one-sided Fisher's exact test |
| `selection.min_above` | 3 | fewest requests above the reference p90 in a flagged window |

## Online signals

`stormlog.infer.diagnosis_signals.evaluate_signal(kind, scrapes, config)`
evaluates one kind over a window of consecutive scrapes the caller chose,
using the [scrape windows](vllm_telemetry.md#windows-of-scrapes) rules. It
returns a `SignalValue`:

| Field | Meaning |
| --- | --- |
| `value` | the kind's measure over the window, or `None` |
| `sufficient` | whether the window had enough data to decide |
| `reason` | the first reason it did not; all of them are in `detail["reasons"]` |
| `exceeds` | the verdict against the threshold; `None` whenever `sufficient` is False |
| `threshold`, `thresholds_version`, `threshold_overridden` | which threshold decided |
| `detail` | supporting figures; `scope: engine_global`; the window's successful `scrapes`, `failed_scrapes` inside it, `window_seconds` and `placement` |

| Kind | `value` | Supporting detail |
| --- | --- | --- |
| `queue_saturation` | median of `vllm:num_requests_waiting` | its maximum; the median per waiting reason (`capacity`, `deferred`); bounds on the window's p90 queue time |
| `kv_preemption_pressure` | increase of `vllm:num_preemptions_total` | the preemption rate; the maximum KV usage |
| `prefix_cache_loss` | prefix-cache hits per queried token | the hit and query counts; the drop below `config.reference` |

`/metrics` describes everything an engine served, from every client, so a
signal over its threshold means a mechanism is *suspected* in the engine's
traffic; it does not say whose requests it hurt. A prefix-cache signal in
particular cannot tell another client's prompts lowering the ratio from the
cache losing a victim's prefixes, and it gives no verdict without a reference
ratio (`requires_reference`); a reference outside 0-1 is refused. Nor does it
decide over fewer than `min_queried_tokens` queried tokens
(`too_few_queried_tokens`): vLLM counts every prompt token of a new request
as a query, so a quiet second with one short, unseen prompt has a ratio of
0 by construction. Every figure of one signal comes from one engine: behind an
exporter with several, `config.engine` names it, and without it the window is
`engine_required`.
Kinds that metrics alone cannot decide answer `requires_hook`,
`requires_trace` or `requires_client`.

## Engine-loop stalls

`stormlog.infer.diagnosis_loop.engine_loop_gap(records, config)` reads one
engine epoch's raw [execution hook](vllm_execution.md) records (the
`scheduled`, `completed`, `heartbeat` and, from hooks that record them,
`pause` records, in `seq` order) and returns a `SignalValue` for its stalls:
stretches in which the engine made no progress while it had work it could
run. `value` is the stall furthest over its own limit, or the longest when
none is over, so a long stall against a lenient limit can be reported below
a shorter one against a strict limit. Only stalls of at least the lowest
floor are compared with a baseline, which keeps a window of fast steps
linear in its steps; a window in which every step is that slow takes time in
proportion to its steps times the steps in a baseline window.
It needs no import, so an online trigger can run it on the records it tails;
the diagnoser runs the same rules on imported steps. `LoopGapConfig` refuses
a threshold override with a key the table lacks, a value that is not a
finite number, or a loop threshold that is not positive. A trigger that
evaluates a later window of a long log, such as its tail, passes the epoch's
hello as `config.hello`, or prepends it: the hello says whether pauses are
recorded, and is neither a sequence gap nor a zero point for that window's
coverage.

Work is *ready* during a stretch when a request ran in the step before it
and in the step after it. The step before a gap between steps is the one
whose completion starts it, without the request finishing there: under
async scheduling a request's second step is scheduled before its first
completes, so the step scheduled just before it may come after an idle
stretch in which nothing was ready. A memberless, zero-token step (which
vLLM schedules to send finished IDs) runs nothing and is passed over: the gap
runs from the step before it. A request prefilled in chunks is ready
between them. A streaming-input request (`resumable`) is ready within a
turn, where it decodes like any other, but not across a gap after which its
prompt grew: then it was waiting for its client's next input. A
stretch the scheduler spent paused with
`PAUSED_ALL` (from the hook's `pause` records), or one the caller excludes
with `exclude_wall` (for example its own profiler stop), has no ready work;
only the part of a stall such an interval covers is removed, and what
remains on either side is still a stall. Work waiting to be admitted is not
ready: a host gap while only queued requests exist, with none running, is
not a stall by this rule, so the signal cannot see it.
Where a stall sits decides what it can be blamed on:

| `detail["locus"]` | Stretch | `detail["attribution"]` |
| --- | --- | --- |
| `between_steps` | a step's completion to the next `schedule()` entry | `host` |
| `in_schedule` | inside `schedule()` | `host` |
| `within_step` | a step's own time after `schedule()` returned (or after the previous completion, under async scheduling) | `host_or_gpu`: without a GPU trace the two cannot be told apart |

A stall exceeds when it is at least `stall_factor` times the median
completion cadence of the busy steps that completed in the `baseline_window`
before it, and at least `stall_floor_ns`. Steps are compared with steps of
their own size: the cadence is taken over earlier steps whose scheduled
tokens lie within a factor of two of the stall's step when enough share it
(`detail["baseline"]` is `matched`), so a step running a long prefill is not
measured against decode-only steps. Otherwise it is taken over the earlier
busy steps at least as large as the stall's (`unmatched`), when there are
`min_busy_steps` of them, so a step is never measured against smaller ones;
otherwise it is replaced by `no_baseline_floor_ns` (`floor`). A second long
prefill a few seconds after the first therefore meets the floor, not the
decode cadence. Only earlier steps count, so the decision never depends on
what happened after the stall. With `config.now_wall_ns`, a stall still
going on counts from the last completion (`detail["ongoing"]`), while a
request of that step is still running: not one that finished there (by a
finish reason, or discarded after its end of sequence under async
scheduling), was ended since (a `terminal` record, as for a cancel), or
belongs to an epoch that said `goodbye`.

A stall is judged only where the records are known to be whole: between
two heartbeats (the hello counting as one with nothing lost) whose drop
counts and errors did not change, one at or before the stall's start and one
at or after its end. A stall still going on also needs a heartbeat since it
began, the last within `heartbeat_grace_ns` (3 s: the writer beats once a
second, but under load its beats slip, 2.3 s apart on a real vLLM 0.30.0
run) of the evaluation time. A capped or killed writer stops writing records
and heartbeats alike, so the engine running on unrecorded looks like a stall
with no heartbeat around it. A host stall that holds Python's GIL stops the
writer's thread too, so while it lasts it is no verdict, and it is judged
once it ends and the heartbeats resume. A stall over its limit
outside that coverage gives no verdict (`hook_coverage_unknown`) instead of
exceeding; `detail["covered"]` says which the reported stall was.

The records also give no verdict (`sufficient` is False) when they come from
two epochs (`epoch_changed`), skip a `seq` or show drop counts (of any kind,
oversized records included) rising between heartbeats
(`hook_records_dropped`), show write errors rising between heartbeats
(`hook_writer_errors`), hold no completed step (`too_few_steps`) or are
absent (`requires_hook`). Write errors also count failed seals and
`status.json` writes, which lose nothing, so this abstains more than it
must. Only the epoch's `status.json`, passed as `config.status`, says the
writer is capped (`hook_capped`): no heartbeat ever does.
`detail["pause_capability"]` says whether the hook records pauses. Without
it, a pause of every running request (vLLM's `PAUSED_ALL`, as for an RL
weight sync) looks like a stall with ready work wherever a pause can act: a
gap between steps, or, under async scheduling, a step scheduled before the
one before it completed, whose output waits for an engine step() call that
a paused scheduler skips. Such a stall over its limit is no verdict
(`pause_state_unknown`). A pause changes only what schedule() returns and
whether the engine steps, so a long schedule() call or a step run within
one call keeps its verdict. A pause can only remove stalls, so records with
no stall over its limit still say none exceeded. On a log from #217's hook,
which records no pauses, spec5's 1,315.5 ms first step still exceeds.
