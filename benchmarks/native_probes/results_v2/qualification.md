# Qualification of `stormlog infer profile --trace cupti` (helper 0.4.0)

Declared before any of these runs in `../qualification_cupti_cli.json` (SHA-256 `85322696…befd9`). The host recorded that hash before the first trial (`qualify.log`).

- **Setup:** one JarvisLabs L4 (driver 580.126.20), the same software as v2: vLLM 0.30.0, torch 2.13.0+cu130, Qwen2.5-0.5B-Instruct at the v2 revision, and Nsight Systems 2026.3.2.
- **Run:** 2026-10-30.
- **Helper builds:**
  - linked to the venv's CUDA 13 CUPTI: SHA-256 `182e8bb1…`
  - the mismatch control, built with the image's CUDA 12.6.77 toolkit: `2b40be90…`
- **Per-trial records:** `qualification_trials.jsonl`. Each record carries the SHA-256 of its full `result.json`.
- **Raw files:** outside Git, in the maintainer's `qualification-results.tgz`, `memdiag.tgz` and `memdiag-kernel.tgz`.

## Declared checks

| Check | Prediction | Result | Verdict |
|---|---|---|---|
| A: CLI end to end, CUDA graphs | one window acknowledged and stopped by time bound; traces imported; event_loss all zero | window acknowledged, stopped at 20 s (`time_bound`); 1,165,513 GPU events (1,023,647 on graph nodes) imported as 144,904 records; 4.0M helper records, 0 drops, 175 kernels without timestamps, reported in `event_loss` | **Pass** (the loss prediction was wrong, and the loss was reported) |
| B: bounded consumer, eager | peak buffers ≤ 32 MiB; W=800 RSS below v2's 234 MiB; refusals > 0 with a delayed writer | buffers peaked at the 32 MiB cap in both trials; RSS 218 MiB at W=800 and 50 MiB at W=200 with the delay; refusals counted in both | **Pass**, but at W=800 the writer fell behind: 35.8M of 38.0M records (94%) were dropped, all counted |
| C: CUDA runtime mismatch | helper refuses; window rejected with the reason | `helper built against CUDA 12060 but the target's CUDA runtime is 13000`; no control socket; window `rejected` with that message; nothing imported | **Pass** |
| D1: coexistence with vLLM's torch profiler | helper initializes; Kineto loses or degrades GPU activity | helper initialized and holds CUPTI; Kineto logged `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` and wrote 2.2M CPU events and no GPU events; the import added 0 records **without a note** | **Pass** as declared ("recorded as observed"); Kineto's failure is silent to the Stormlog user |
| D2: coexistence under Nsight Systems | helper reports another CUPTI client | helper refused in all 4 injected processes, but with `CUDA runtime version unavailable`; the subscriber check was not reached; Nsight counted 3,000 kernels | **Pass** on the declared rule (no clean empty finalize). The predicted mechanism was not exercised |
| E: CUDA-graph window scaling | p95 within +2%, added RSS under 150 MiB, stop under 2 s at every W | see table: p95 +1.0%, +1.6%, **+5.2%**; RSS 56, 240, **666 MiB**; stop ≤ 0.56 s | **Fail** on p95 at W=2200 and on memory at W ≥ 800 |

### Check E: CUDA-graph window scaling

Each cell is the median of 3 repetitions, with [min, max], computed with `analyze_v2.py` exactly as in v2. The p95 delta is against the median `off` p95 over the same request indices: 327.3, 328.1 and 328.3 ms.

| mode | W | ok/n | added RSS MiB | stop s | window p95 Δ | artifact MiB | records | dropped | CUPTI buf MiB | reqs >2s/failed |
|---|---|---|---|---|---|---|---|---|---|---|
| cupti-min | 200 | 3/3 | 56 [56, 58] | 0.53 [0.53, 0.56] | +1.0% [+0.8%, +1.4%] | 114.5 [114.2, 114.6] | 3.80e+06 | 0 | 24 | 0 |
| cupti-min | 800 | 3/3 | 240 [240, 248] | 0.29 [0.28, 0.30] | +1.6% [+1.5%, +2.5%] | 456.5 [455.5, 457.8] | 1.51e+07 | 0 | 24 | 0 |
| cupti-min | 2200 | 3/3 | 666 [662, 670] | 0.28 [0.28, 0.33] | +5.2% [+5.1%, +7.4%] | 1240 [1229, 1242] | 4.11e+07 | 0 | 24 | 0 |
| off | 200 | 3/3 | 0 | – | −0.1% [−0.3%, +0.0%] | – | – | – | – | 0 |
| off | 800 | 3/3 | 0 | – | +0.0% [−0.2%, +0.0%] | – | – | – | – | 0 |
| off | 2200 | 3/3 | 0 | – | −0.1% [−0.4%, +0.0%] | – | – | – | – | 0 |

Against the graph-mode adoption criteria declared in the plan:
- **p95 ≤ 5% at every W: not met.** W=2200 gave +5.2% [+5.1%, +7.4%], which is 345 ms against 328 ms.
- **Memory at W=2200 ≤ 1.5× W=200: not met.** It was 11.9×: 666 MiB against 56 MiB.
- **Loss: met.** Every trial finalized, with 0 drops.

## Post-hoc (not part of the declared plan)

### Runtime-version check moved to start and finalize

Check D2 raised the possibility that the injection-time version check refuses processes that load the CUDA runtime late. The check was therefore changed to compare versions when capture starts and again at finalize. Re-runs with the old (`182e8bb1…`) and fixed (`7d0659c2…`; mismatch control `d4805329…`) builds gave:

| Re-check | Old build | Fixed build |
|---|---|---|
| microbenchmark W1, helper only | 3,000 kernels, 100% correlation, runtime 13000 found | same |
| W1 under Nsight Systems | refused: runtime version unavailable | refused: runtime version unavailable (found neither at start nor at finalize) |
| check A | – | pass: 146,974 records imported, runtime 13000 |
| check C | – | pass: refused with the mismatch message |

The old check did not refuse ordinary processes; the D2 failure is specific to running under Nsight. The fix causes no regression, but it does not change the Nsight outcome. The subscriber check, for the case where another CUPTI client attaches first, remains **untested on hardware**.

### Where the graph-mode memory goes

Each row is one graph-mode trial at W=800, sampling the injected engine process's `smaps_rollup` and every mapping over 1 MiB every 10 s (`scripts/native_probes/remote/memory_diag.py`):

| mode | activities | helper records | added RSS | anonymous growth, start to stop | window p95 |
|---|---|---|---|---|---|
| cupti-min | runtime, driver, kernel; written | 15.2M | 246 MiB | 231 MiB | 332 ms |
| cupti-discard | runtime, driver, kernel; discarded unread | 0 (15.1M discarded) | 256 MiB | 252 MiB | 335 ms |
| cupti-kernel | kernel only; written | 5.6M | 244 MiB | 222 MiB | 489 ms |

- All of the growth was in the process's main `[heap]`, with no new anonymous mappings and no change to file-backed memory (470 MiB). It was not released after the stop.
- Discarding records left the growth unchanged, so in graph mode it is **not** the helper's decoding, encoding or writing.
- Dropping the API activities removed 63% of the records but left the growth unchanged. So it scales with kernels traced on CUDA-graph nodes, about 40–45 bytes per kernel. A strong inference is that it lives inside CUPTI or the CUDA driver. The diagnostic shows which heap grew, not which library allocated it.
- This corrects how the v2 decision record's consumer explanation applies. v2's eager discard ablation stayed at 8 MiB, so in eager mode the consumer dominated. In graph mode, CUPTI's own kernel tracing grows memory whatever the consumer does.
- The kernel-only trial's p95 of 489 ms is a single observation with no matching `off` trial. v2 measured kernel-only graph mode at +1.1% at W=200. It is reported here as unexplained, not as a cost of the mode.

## Not measured

- The subscriber check, when another CUPTI client attaches first.
- Graph-level activity (`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`) as a way to avoid per-kernel growth.
- Pinned host memory, overlapping streams, non-root users, and other GPUs.
