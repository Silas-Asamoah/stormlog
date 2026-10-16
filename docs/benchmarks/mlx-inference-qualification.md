# MLX inference qualification

Qualified locally on 2026-10-11 for the specific versions/models below. PR4
implements text/audio endpoint profiling, passive process-local server capture,
and framework-free offline evidence joins. **Authenticated oMLX statistics
remain unqualified at the user's request.** This is a limited qualification,
not blanket support for every model, server build or operating system.

## Environment and reproducible identity

Native Apple M5 Pro, arm64, macOS 27.0.1, Python 3.11.16, MLX/Metal 0.32.3.
Client regressions ran on Python 3.14.7. The temporary serving interpreter
preferred its own MLX 0.32.3 and reused the already installed server packages;
the user's older MLX 0.32.2 environment was not upgraded.

| Server | Installed distribution | Qualified models |
| --- | --- | --- |
| oMLX | 0.7.0.dev2, locally installed fork | Qwen2.5-0.5B-Instruct-4bit |
| MLX-LM | 0.31.3 | Qwen2.5-0.5B-Instruct-4bit |
| MLX-Audio | 0.4.6 | Kokoro-82M-bf16 TTS; Parakeet TDT 0.6B v3 4-bit STT |

The [machine-readable results](mlx-inference-qualification.json) record actual
entrypoint SHA-256 hashes, Stormlog source hashes, model snapshot revisions,
fixture digests, PID, counts, latencies and memory scope. Installed versions
are not equated to the design-source commits in the implementation plan.
Full package versions are in the local raw qualification artifact.

Model snapshots:

- Qwen: `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
- Kokoro: `a71e4d38b236d968966a2002c4c895dbd12b1c3c`.
- Parakeet: `ab4003325092525ed02350a981dede8122fb8a4e`, from
  [transcriber-app/parakeet-tdt-0.6b-v3-mlx-4bit](https://huggingface.co/transcriber-app/parakeet-tdt-0.6b-v3-mlx-4bit).

TTS uses two authored paragraphs, voice `af_heart`, speed 1.0. Each successful
response contains two WAV segments totalling 6.9 seconds. STT uses the authored
sentence “The quick brown fox jumps over the lazy dog.” synthesized explicitly
with macOS `say`, then converted with `afconvert` to mono 16-bit PCM/16 kHz.
Input duration is 2.67075 seconds. Parakeet returned the exact sentence in the
separately retained known-speech check. Client artifacts retain measurements,
not the sentence, transcript or audio; the explicit qualification fixture is
saved separately. Preparation is explicit; the client never resamples inputs.

## Protocol and native gates

Both text engines passed chat/raw completion, buffered JSON and SSE, usage
requested/omitted streaming, concurrency 1 and 2. Each closed-loop workload
has five measured requests per concurrency plus one excluded warmup. Separate
server response IDs are retained; no sent-ID echo or scheduler ownership is
inferred. TTS passed buffered/streamed concatenated WAV and first-playable-frame
timing. STT passed buffered JSON and native NDJSON streaming with known speech.
All tasks passed fixed-rate 100 requests/s with maximum one in flight and drop
overflow; drops remain in offered populations and successful duration excludes
dropped requests. Rates are short stress cases, not sustained throughput claims.

All three launchers executed model calls in the sampled serving PID. The
qualification-only server example installs bounded read-only model-call
checkpoints in both telemetry modes. It reads active/cache/lifetime peak and
process RSS, never evaluates arrays, synchronizes streams, clears caches or
resets peaks. Forward-call boundaries may precede lazy device execution;
generator yields are observations, not request spans. Native sampling starts
after bootstrap imports and before model loading in this harness.

Each on-run has 128 checkpoint observations; all 128 lifetime peak readings
fall between adjacent native samples within a 150 ms bracket. Active/cache
can change between sequential getters and samples and are not required to
match at arbitrary busy boundaries. Full-session native lifetime peaks cover
at least the checkpoint peaks. Finalized native sidecars passed digest,
identity and clock validation after copying. All three offline joins report
`verified` process association and complete captures, with request ownership,
kernel and prefill/decode attribution explicitly unavailable.

## Matched native sampling off/on observations

Off then on ran sequentially per engine with the same installed dependencies,
models, manifests, seed 0, warmup/count/concurrency/order, 16-word synthetic
text target, eight output tokens and temperature 0. Fresh server processes
started with local weights; oMLX used separate fresh state and `--no-cache`.
Model loading/cache warmup occurred normally in each process, with no client
reset. Audio dependencies and the English spaCy model were prepared before
these final pairs. Native on sampling interval was 50 ms. Both modes used
identical checkpoint instrumentation, bounded to the first 128 boundaries.

The table selects concurrency 1 streamed workloads; all buffered and
concurrency 2 measurements are in the results JSON. Each row has **n=5**.
p90 is descriptive and fails the report's statistical sufficiency threshold.
A regression suite and ordinary host activity ran concurrently. These are
observed timings for protocol qualification, **not a causal instrumentation
overhead estimate or a model speed ranking**.

| Engine | Workload | Native sampling | Mean E2E ms | p90 E2E ms | Successful throughput |
| --- | --- | --- | ---: | ---: | ---: |
| mlx-lm | chat-stream | off | 86.45 | 88.45 | 11.52 req/s |
| mlx-lm | completion-stream | off | 86.48 | 90.18 | 11.52 req/s |
| mlx-lm | chat-stream | on | 101.07 | 110.65 | 9.86 req/s |
| mlx-lm | completion-stream | on | 99.40 | 101.13 | 10.03 req/s |
| omlx | chat-stream | off | 82.67 | 86.01 | 12.05 req/s |
| omlx | completion-stream | off | 77.26 | 78.13 | 12.89 req/s |
| omlx | chat-stream | on | 84.13 | 91.88 | 11.84 req/s |
| omlx | completion-stream | on | 80.10 | 83.61 | 12.43 req/s |
| mlx-audio | tts-stream | off | 219.69 | 221.56 | 31.36 audio s/s |
| mlx-audio | stt-stream | off | 28.22 | 28.64 | 93.64 audio s/s |
| mlx-audio | tts-stream | on | 218.53 | 220.09 | 31.54 audio s/s |
| mlx-audio | stt-stream | on | 28.68 | 28.94 | 92.23 audio s/s |

Text throughput is successful requests divided by measured-span seconds;
audio throughput is successful output seconds (TTS) or input seconds (STT)
divided by that span. HTTP latency includes upload, queueing and delivery.

| Engine | Native sampling | Checkpoint RSS max bytes | Checkpoint lifetime peak bytes | Session sampled active max bytes | Session observed lifetime peak bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| mlx-lm | off | 813170688 | 352783560 | unavailable | unavailable |
| mlx-lm | on | 815349760 | 352783560 | 305763564 | 352783560 |
| omlx | off | 1076674560 | 352094404 | unavailable | unavailable |
| omlx | on | 1057587200 | 352061636 | 336033926 | 352061636 |
| mlx-audio | off | 1415299072 | 1332963147 | unavailable | unavailable |
| mlx-audio | on | 1417199616 | 1332963147 | 1846351379 | 1846351435 |

Checkpoint RSS/peak includes warmup and only the first 128 model-call boundaries;
full native session maxima include startup, loading and all tasks. These
populations differ and must not be compared as request peaks. RSS, active,
cache and lifetime peak are overlapping process observations, never added as
disjoint memory or presented as total/free GPU capacity. Server code itself
may initialize allocator limits; Stormlog's passive collector never does.

## Limits and unsupported combinations

- Authenticated `/admin/api/stats` returned 401 without a normally issued
  admin cookie. Inference continued successfully and stats failures remained
  evidence loss. Cookie/bearer credential separation, GET-only sampling,
  redaction, epoch handling and parser bounds pass fixture tests. Successful
  authenticated live stats and its overhead are **unqualified**; no login,
  authentication changes, signing-secret token generation or bypass was used.
- Whisper-tiny-asr-fp16 on this MLX-Audio build failed inside upstream
  `mx.async_eval` with a missing GPU stream in the worker thread. It is
  unsupported by this qualification; Parakeet passed instead.
- The cached Limite model failed upstream MLX-LM model-type resolution. A
  Kokoro snapshot-directory hash could not be resolved by upstream MLX-Audio;
  its explicit HF ID passed. Stormlog passes all model IDs unchanged.
- Kokoro required explicitly prepared Misaki/English spaCy dependencies and
  Parakeet required explicitly provisioned weights. Missing dependencies and
  model resolution failures are not successful endpoint qualification.
- Workers, reload, active distributed inference, realtime audio, reference
  audio cloning, compressed audio, embeddings and vision remain outside this
  contract. Live audio watch and audio token/SLO flags are rejected.
- The installed oMLX fork rewrote its generated global worker-interpreter shim
  despite the dedicated base path. After stopping only the qualification
  processes, that generated shim was restored to the installed user oMLX
  interpreter. No existing inference service was restarted.

## Artifacts and reproduction

Raw files are saved locally under `artifacts/mlx-inference-pr4/` (Git-ignored),
including each off/on client matrix, copied native bundles, checkpoints,
redacted launch configuration, fixtures, known-speech response, joined reports
and `qualification.json`. Unauthenticated 401 stats cases are retained
separately, outside the matched off/on pairs. `files.json` records SHA-256 and size for every saved
file. Its digest is in the committed results JSON. Server settings, credentials,
model weights and raw admin responses are not in this archive. Copy the raw
directory when sharing evidence; a source checkout alone contains the summary.

From a source checkout, explicitly launch a dedicated server with
`examples/scenarios/mlx_inference_server.py`, then point
`examples/scenarios/mlx_inference_scenario.py` at that endpoint. Both expose
`--help`; neither client nor offline analysis starts/downloads/resets a server.
Use matching `--run-id`, explicit version/models/manifests and `--telemetry-state`.
After stopping the owned server, analyze the copied client file with
`--mlx-server-evidence <capture>/native`. See the [inference guide](../mlx_inference.md)
for public profile/capture commands.

The live pytest gate and manual self-hosted Apple Silicon job require an
explicit prepared endpoint configuration for all three engines and normally
issued admin credentials for successful stats qualification. Selecting the
gate without prerequisites fails; default ordinary tests skip its three
unconfigured live cases. The local limited qualification here deliberately
leaves authenticated stats unqualified rather than making that full gate green.

Initial protocol validation passed 1,980 inference tests with three unconfigured
live tests skipped; 44 focused follow-up tests passed after the distributed-settings
guard. Copied text/audio joins and CLI help passed in an installed-wheel client
with MLX, all server packages, PyTorch, TensorFlow and JAX absent.

## Publication checks (2026-10-11)

- Full locally supported CI suite: **3,567 passed, 25 skipped**, using four
  workers on Python 3.14.7. TensorFlow/JAX matrix tests were excluded because
  those frameworks are not installed locally; their remote jobs remain separate.
- TUI pilot/snapshot suite: 18 passed. Terminal smoke: one passed. Explicit
  native Apple Silicon Metal gate: four passed. Final affected vLLM tests:
  141 passed, one optional benchmark skipped. Documentation contracts: 96 passed.
- Repository-wide Black 24.3.0, isort 8.0.1 and flake8 7.3.0 passed using the
  pinned Python 3.10 CI environment. Mypy 1.8.0 (CI) and 2.4.0 (local Python
  3.14) both passed all 209 source files without disabling unused-ignore checks.
- Complexity <=10 and the workflow audit passed. Source and wheel distributions
  built in isolated environments and passed `twine check`.
- The exact tracked documentation, copied to a temporary source directory,
  passed a clean `sphinx -E -W --keep-going` HTML build with no warnings.
  Built-in `bytes` type references are explicitly qualified, including dataclass
  signatures, instead of resolving to unrelated properties named `bytes`.
  User-owned untracked planning pages remain untouched and outside that published
  source set; building every local page still reports their orphan warnings.

The broad run initially exposed a pre-existing fixture assumption: Python 3.14
can JSON-encode deeply nested lists iteratively. Its serialization-failure test
now uses invalid mapping keys, retaining error/sequence assertions on supported
versions. Reader type-ignore codes accommodate the differing stdlib `readinto`
annotations. These changes preserve runtime behavior. The full rerun passed.
Authenticated live oMLX stats remain unqualified as explicitly requested.


## Checks after the base-branch merge (2026-10-11)

Pulled the conflict-resolution merge `d6bf8ff` from `release/dev`. Removed
duplicate profiler imports and the duplicate text-client endpoint validator,
retaining the shared bounded-transport validation alongside the incoming
export/trace-header and HTTP-status support. Regression cases cover missing
hosts, invalid ports, URL credentials/fragments and unsupported schemes before
transport construction, for chat and raw completion clients.

CI-pinned Black/isort/flake8 passed the full source/test/example trees, mypy
passed all 232 source files, and the complexity gate passed. The targeted MLX
text/audio, HTTP transport, CLI portability, profiler and export integration
suite passed **176 tests**. The live model artifacts above identify their
pre-merge source digests; these post-merge checks cover portable integration.

## Upstream integration verification (2026-10-16)

Merged `release/dev` revision `330f7fe` into `mlx-integration4`. The merged
CLI retains upstream comparison/manifest/diagnosis commands and MLX audio/server
commands. Bounded text transport forwards send/first-content callbacks; the
shared scheduler honors stop files; workload spec and realization digests retain
the text task while preserving legacy chat digests. Server probes use the shared
HTTP transport. Comparison, manifest and diagnosis readers expand compact
contexts while diagnosis retains physical evidence hashes and line numbers.

The focused serving selection passed 158 tests, all 140 framework-independent
MLX contracts passed, and the CLI/TUI/telemetry/query/session/docs/hook selection
passed 330 tests with one optional skip. The compact-reader and diagnosis
fixture regressions passed; CI-pinned repository formatting, lint, mypy across
282 source files and the complexity gate passed. These checks cover portable
integration; the native live measurements above keep their original source
digests and authenticated statistics remain unqualified.

The final complete inference regression suite on Python 3.11 passed
**3,152 tests, with 8 skipped**, using four workers. Diagnosis fixtures remove
engine evidence from decoded records so the same checks apply to embedded and
compact artifacts. No native framework or server package was needed for this
portable suite.
