# MLX inference profiling

PR4 extends endpoint profiling for oMLX, MLX-LM and MLX-Audio. The implementation
and selected live model/task combinations are recorded in
[the qualification results](benchmarks/mlx-inference-qualification.md).
Authenticated oMLX statistics remain unqualified at the user's request.

Text profiling retains the existing chat defaults. Select `--engine omlx`
or `--engine mlx-lm` and `--task completion` for raw text completions:

```bash
stormlog infer profile --engine mlx-lm --task completion \
  --base-url http://127.0.0.1:8080/v1 --model MODEL_ID \
  --tokenizer none --input-tokens 32 --output-tokens 16 \
  --requests 10 --concurrency 1,2 --output /tmp/mlx-completion.jsonl
```

Model IDs remain opaque. `--endpoint` is a full URL; `--base-url` gains only
the task route. JSON and SSE share the existing no-redirect, no-proxy,
no-retry transport. `--timeout` bounds the total request, including trickling
responses; `--max-response-bytes` bounds cumulative response data. Sent
request IDs and server-generated completion IDs are recorded independently.
Unknown server versions and execution attribution remain unavailable.

Text and audio scheduling share arrival indices, in-flight admission, drops,
phase windows and bounded drain. Calls still running after an asyncio drain
remain tracked; the next phase waits for them before starting. The text
extraction passed 148 existing scheduling, profiler, trace and vLLM tests.

## Audio fixtures and measurements

`profile-audio` accepts one task per run. TTS uses a JSONL text manifest;
STT uses a JSONL manifest of PCM WAV files relative to the manifest directory.
Every fixture is validated before traffic or artifact creation. Model and
endpoint are excluded from the workload digest; input bytes, settings, seed,
selection order, arrival policy and measurement bounds are included.

```json
{"schema_version":1,"id":"tts-001","task":"tts","text":"This is a short test.","settings":{"speed":1.0}}
```

```json
{"schema_version":1,"id":"stt-001","task":"stt","audio":"fixtures/speech.wav","settings":{"language":"en"}}
```

```bash
stormlog infer profile-audio --task tts --engine mlx-audio \
  --base-url http://127.0.0.1:8001/v1 --model TTS_MODEL_ID \
  --manifest tts.jsonl --stream --concurrency 1,2 --requests 10 \
  --output /tmp/mlx-audio/tts-run.jsonl

stormlog infer profile-audio --task stt --engine mlx-audio \
  --base-url http://127.0.0.1:8001/v1 --model STT_MODEL_ID \
  --manifest stt.jsonl --stream --concurrency 1,2 --requests 10 \
  --output /tmp/mlx-audio/stt-run.jsonl

stormlog infer analyze /tmp/mlx-audio/tts-run.jsonl --format json
```

Audio defaults to buffered responses. TTS requires WAV; STT requires JSON
when buffered and NDJSON when streamed. The stream is upload-then-output,
not microphone streaming. Voice-cloning references, compressed codecs,
text token-length flags, token SLOs and live audio watch are unsupported.
Unknown fixture settings are rejected rather than ignored.

The response parser counts all concatenated PCM WAV segments. First-audio
latency requires a complete playable frame; first-body-byte latency also
includes headers/container delivery preceding those frames. STT text deltas
are accumulated, while full records carrying segments/language replace the
deltas under the pinned MLX-Audio wire contract. An error record after HTTP
200 still makes the request fail.

RTF divides client HTTP elapsed seconds by successful output duration for
TTS or validated input duration for STT. Audio throughput divides successful
audio seconds by the declared case interval. These include upload, network
and delivery time. Reports retain arrival cohorts, failures, drops, timeouts,
cancellations, warmup exclusion, sample counts and quantile sufficiency;
failed partial responses do not contribute successful audio duration.

Normal event rows contain fixture IDs/digests and measurements, without raw
text, transcripts or audio. `--retain-media-dir` opts into bounded response
media and STT input WAV retention within the artifact directory. Set
`--max-retained-bytes` to cap total retention. Retention failures are recorded
separately from inference success. Response/fixture bytes, total fixture bytes
and audio duration also have explicit configurable limits.

The [audio v1 schema](schemas/inference_audio_v1.schema.json) covers request,
workload, summary and manifest variants. Native MLX telemetry remains in a
separate sidecar; it never enters the inference context codec as wire v4.

## Read-only oMLX statistics

Opt in with `--engine omlx --omlx-stats-url
http://127.0.0.1:8000/admin/api/stats`. Add `?scope=alltime` for lifetime
counters or `?model=MODEL_ID` for model counters. The default is session/server
scope. Only the documented token/request counters, uptime, average serving
rates, cache efficiency and active/waiting request gauges are retained.
The source response can contain credentials; its raw body is never recorded.

The inspected oMLX admin route uses a normally issued `omlx_admin_session`
cookie: `--omlx-stats-cookie-env ENV_NAME` supplies that separate token.
`--omlx-stats-api-key-env ENV_NAME` supplies a bearer key for deployments
whose admin gateway explicitly supports it. Choose one. Sampling never logs
in or changes authentication. Inference credentials are never reused for
admin requests, even on the same origin.
A missing/unauthorized stats endpoint is evidence loss, not an inference
failure. Counter decreases and uptime restarts mark epoch changes; outage
and sampling scope are explicit. Aggregate statistics can include other
clients and do not prove request ownership, allocator usage or pure compute
time. Sampling uses bounded GET requests to `/admin/api/stats` only.

## Same-process server evidence

Run the server in an explicitly selected interpreter with its server package
and MLX 0.32.3 or newer. The launcher candidates are oMLX 0.7.0.dev2,
MLX-LM 0.31.3 and MLX-Audio 0.4.6. These version gates identify inspected
entrypoints; consult the qualification results for supported model/task
combinations. Use the same run ID for server capture and client profiles.

```bash
stormlog infer mlx-serve --engine mlx-lm --run-id local-run \
  --output /tmp/server-evidence --interval 0.1 -- \
  --model /path/to/preprovisioned-model --host 127.0.0.1 --port 8080

stormlog infer analyze /tmp/client-run.jsonl \
  --mlx-server-evidence /tmp/server-evidence --format json
```

The foreground launcher passes server arguments unchanged and restores Python
arguments on exit. Workers, reload and distributed modes are rejected before
capture. oMLX requires an explicit `--base-path` and disabled cluster settings;
its own startup behavior remains that of the installed server distribution.
For custom entrypoints, use `stormlog.infer.mlx_server.capture_mlx_server`
around inference performed in that interpreter.

Sampling reads native allocator counters and independent process RSS. It never
evaluates model arrays, synchronizes streams or resets peak memory. Native
framework telemetry v4 stays under `telemetry/`; `mlx_evidence.json` records
file digests, capture bounds, host/boot/PID/process-start identity, version
provenance and redacted arguments. The
[evidence v1 schema](schemas/inference_mlx_evidence_v1.schema.json) also covers
the separately versioned oMLX statistics records.

Offline loading verifies size/record limits, paths, digests, schemas, session
identity and sample timestamps without importing MLX or server packages.
Repeat `--mlx-server-evidence` for separate captures. Identical copies are
deduplicated; conflicting identities or contents fail validation.

Matching run IDs and endpoints produce `declared_direct` association. A
controlled launcher with an observed literal-loopback listener owned by its
sampled PID and matching host/boot can establish `verified` process
association. Request execution ownership remains unavailable. Missing identity
is `unjoined`; overlapping endpoint process epochs are `ambiguous`.

Case summaries include only samples definitely within the case window after
clock uncertainty. Different clocks need recorded alignment or explicit
`--clock-offset-ns` and `--clock-uncertainty-ns`. Coverage records sample count,
cadence and observation gaps. Sampled allocator maxima, runtime lifetime peaks,
cache and RSS have distinct scopes. They are never summed across processes or
reported as total device capacity, request peaks or kernel/prefill/decode
attribution.
