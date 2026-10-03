# Qualifying diagnosis and capture

Stormlog's diagnosis of a vLLM server is qualified by injecting known faults
and benign workload changes into a running server and scoring what the
diagnoser reports against what was injected. `stormlog.infer.qualify` holds the
parts of that protocol that are library code, so the harness, the scorer and
other issues' fixtures share one definition.

## Exact bounds

Every claim is a one-sided bound at 95%, computed exactly:

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
   window's requests.

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
