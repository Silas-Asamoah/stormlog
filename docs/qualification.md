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
