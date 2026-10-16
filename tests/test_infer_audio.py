"""Audio transport and reporting qualification without MLX/server packages."""

from __future__ import annotations

import contextlib
import io
import json
import struct
import threading
import wave
from dataclasses import replace
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from stormlog.infer.analysis import analyze_inference_events, format_analysis_text
from stormlog.infer.audio_config import AudioProfileConfig
from stormlog.infer.audio_events import AudioRequestEvent
from stormlog.infer.audio_profile import AudioProfiler
from stormlog.infer.audio_stream import TranscriptStream, WavSegments
from stormlog.infer.audio_workload import AudioWorkload
from stormlog.infer.errors import InferInputError


def wav(frames=800, rate=8000):
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\0\0" * frames)
    return output.getvalue()


def config(tmp_path, task="tts", **kwargs):
    rows = [
        {
            "schema_version": 1,
            "id": "fixture-one",
            "task": task,
            "settings": {},
            "text" if task == "tts" else "audio": (
                "private sample text" if task == "tts" else "speech.wav"
            ),
        }
    ]
    (tmp_path / "speech.wav").write_bytes(wav())
    path = tmp_path / f"{task}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return AudioProfileConfig(
        endpoint="http://127.0.0.1:1/v1/audio/speech",
        model="opaque",
        manifest_path=str(path),
        output_path=str(tmp_path / "run.jsonl"),
        task=task,
        **kwargs,
    )


@contextlib.contextmanager
def server(task, *, stream=False, body=None, split=7, status=200):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            incoming = self.rfile.read(int(self.headers["Content-Length"]))
            if task == "tts":
                calls.append(json.loads(incoming))
            else:
                msg = BytesParser(policy=policy.default).parsebytes(
                    (
                        "Content-Type: " + self.headers["Content-Type"] + "\r\n\r\n"
                    ).encode()
                    + incoming
                )
                calls.append(
                    {
                        part.get_param(
                            "name", header="content-disposition"
                        ): part.get_payload(decode=True)
                        for part in msg.iter_parts()
                    }
                )
            output = (
                body
                if body is not None
                else (
                    wav(800) + wav(400)
                    if task == "tts"
                    else (
                        Path(
                            "tests/fixtures/mlx_inference/mlx_audio/transcript.ndjson"
                        ).read_bytes()
                        if stream
                        else b'{"text":"hi there"}'
                    )
                )
            )
            self.send_response(status)
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            for index in range(0, len(output), split):
                self.wfile.write(output[index : index + split])
                self.wfile.flush()

        def log_message(self, *_):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/custom", calls
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.parametrize("step", [1, 3, 17, 10000])
def test_wav_counts_every_segment_and_waits_for_playable_frames(step):
    data = wav(800) + wav(400)
    parser = WavSegments(len(data))
    parser.feed(data[:44], 1)
    assert parser.first_audio_at is None
    for start in range(44, len(data), step):
        parser.feed(data[start : start + step], 2)
    assert len(parser.finish()) == 2
    assert parser.duration_seconds == pytest.approx(0.15)
    assert parser.first_audio_at == 2


@pytest.mark.parametrize("cut", [1, 12, 43, 100, -1])
def test_truncated_wav_is_never_success(cut):
    data = wav()
    parser = WavSegments(len(data))
    parser.feed(data[:cut], 1)
    with pytest.raises(ValueError):
        parser.finish()


def test_wav_validates_codec_padding_and_container_bounds():
    base = wav()
    junk = b"JUNK" + struct.pack("<I", 1) + b"x\0"
    data = base[:12] + junk + base[12:]
    data = data[:4] + struct.pack("<I", len(data) - 8) + data[8:]
    parser = WavSegments(len(data))
    parser.feed(data, 1)
    assert parser.finish()[0]["frames"] == 800
    corrupt = bytearray(base)
    corrupt[20:22] = struct.pack("<H", 3)
    with pytest.raises(ValueError, match="PCM"):
        WavSegments(len(base)).feed(bytes(corrupt), 1)


def test_ndjson_split_utf8_full_record_and_errors():
    data = '{"text":"é"}\n{"text":"cole"}\n{"text":"école","language":"fr"}\n'.encode()
    parser = TranscriptStream()
    for byte in data:
        parser.feed(bytes([byte]), 2)
    assert parser.finish(3) == "école"
    assert parser.first_transcript_at == 2
    with pytest.raises(ValueError, match="error record"):
        parser.feed(b'{"error":"private transcript"}\n', 4)
    with pytest.raises(ValueError, match="byte limit"):
        TranscriptStream(4).feed(b"12345", 1)


@pytest.mark.parametrize("task", ["tts", "stt"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("concurrency", [1, 2])
def test_audio_protocol_and_report(tmp_path, task, stream, concurrency):
    cfg = config(
        tmp_path,
        task,
        stream=stream,
        concurrency=(concurrency,),
        request_count=3,
        warmup_requests=1,
    )
    with server(task, stream=stream) as (url, calls):
        report = AudioProfiler(replace(cfg, endpoint=url)).run()
    assert len(calls) == 4
    assert report["report_type"] == "stormlog.infer.audio"
    assert (
        report["summary"]["total_requests"]
        == report["summary"]["successful_requests"]
        == 3
    )
    assert report["summary"]["session_status"] == "completed"
    case = report["cases"][0]
    assert case["population"]["cohort_valid"]
    assert case["throughput"]["successful_audio_seconds"] == pytest.approx(
        0.45 if task == "tts" else 0.3
    )
    assert case["metrics"]["rtf"]["n"] == 3
    assert case["metrics"]["rtf"]["successful"]["p99"]["sufficient"] is False
    assert "Audio inference" in format_analysis_text(report)
    assert analyze_inference_events(cfg.output_path)["summary"] == report["summary"]
    artifact = Path(cfg.output_path).read_text()
    assert "private sample text" not in artifact and "hi there" not in artifact
    assert "prompt_tokens" not in artifact and "output_tokens" not in artifact
    for row in map(json.loads, artifact.splitlines()):
        if row["event_type"] == "infer.audio.request":
            assert AudioRequestEvent.from_record(row).to_record() == row
    if task == "tts":
        assert calls[0]["input"] == "private sample text"
        assert calls[0]["response_format"] == "wav"
    else:
        assert calls[0]["file"] == wav()
        assert calls[0]["model"] == b"opaque"
        assert calls[0]["response_format"] == (b"ndjson" if stream else b"json")


@pytest.mark.parametrize(
    "task,body", [("tts", b"RIFFbroken"), ("stt", b'{"error":"private"}')]
)
def test_invalid_responses_keep_failure_inclusive_cohort(tmp_path, task, body):
    cfg = config(tmp_path, task)
    with server(task, body=body) as (url, _):
        report = AudioProfiler(replace(cfg, endpoint=url)).run()
    assert report["summary"]["failed_requests"] == 1
    case = report["cases"][0]
    assert case["throughput"]["successful_audio_seconds"] == 0
    assert case["metrics"]["rtf"]["n"] == 0
    assert case["metrics"]["rtf"]["failure_penalized"]["p90"]["value"] is None
    assert "private" not in Path(cfg.output_path).read_text()


def test_manifest_failure_before_artifact_or_traffic(tmp_path):
    cfg = config(tmp_path)
    path = Path(cfg.manifest_path)
    path.write_text(path.read_text() + "\n" + path.read_text())
    with pytest.raises(InferInputError):
        AudioProfiler(cfg)
    assert not Path(cfg.output_path).exists()


def test_digest_is_portable_and_excludes_model_endpoint(tmp_path):
    cfg = config(tmp_path, "stt")
    first = AudioWorkload(cfg).record("session")["workload_digest"]
    assert (
        AudioWorkload(
            replace(cfg, model="another", endpoint="http://elsewhere/custom")
        ).record("different")["workload_digest"]
        == first
    )
    (tmp_path / "speech.wav").write_bytes(wav(100))
    assert AudioWorkload(cfg).record("session")["workload_digest"] != first


@pytest.mark.parametrize(
    "mode,kwargs",
    [
        ("fixed-rate", {"rates": (30,)}),
        ("burst", {"burst_size": 2, "burst_interval_seconds": 0.01}),
    ],
)
def test_audio_open_loop_cohorts(tmp_path, mode, kwargs):
    cfg = config(
        tmp_path, arrival_mode=mode, request_count=6, max_in_flight=2, **kwargs
    )
    with server("tts") as (url, _):
        report = AudioProfiler(replace(cfg, endpoint=url)).run()
    assert report["summary"]["total_requests"] == 6
    assert report["cases"][0]["population"]["cohort_valid"]
    assert report["cases"][0]["throughput"]["interval"]["kind"] == "scheduled_window"


def test_schema_round_trip_and_unknown_version(tmp_path):
    import jsonschema

    schema = json.loads(Path("docs/schemas/inference_audio_v1.schema.json").read_text())
    cfg = config(tmp_path)
    with server("tts") as (url, _):
        AudioProfiler(replace(cfg, endpoint=url)).run()
    records = list(map(json.loads, Path(cfg.output_path).read_text().splitlines()))
    for row in records:
        if row["event_type"].startswith("infer.audio."):
            jsonschema.validate(row, schema)
    request = next(row for row in records if row["event_type"] == "infer.audio.request")
    request["schema_version"] = 99
    with pytest.raises(ValueError, match="version"):
        AudioRequestEvent.from_record(request)
    Path(cfg.output_path).write_text("\n".join(json.dumps(row) for row in records))
    with pytest.raises(InferInputError, match="version"):
        analyze_inference_events(cfg.output_path)


def test_retention_bound_does_not_turn_inference_into_failure(tmp_path):
    cfg = config(
        tmp_path, retain_media_dir=str(tmp_path / "media"), max_retained_bytes=1
    )
    with server("tts") as (url, _):
        report = AudioProfiler(replace(cfg, endpoint=url)).run()
    assert report["summary"]["successful_requests"] == 1
    row = next(
        row
        for row in map(json.loads, Path(cfg.output_path).read_text().splitlines())
        if row["event_type"] == "infer.audio.request"
    )
    assert row["retained_media"]["response"]["status"] == "limit_exceeded"
    assert not (tmp_path / "media").exists()


def test_stt_stream_error_after_successful_delta_is_failure(tmp_path):
    cfg = config(tmp_path, "stt", stream=True)
    with server("stt", body=b'{"text":"private"}\n{"error":"oops"}\n') as (url, _):
        report = AudioProfiler(replace(cfg, endpoint=url)).run()
    assert report["summary"]["failed_requests"] == 1
    row = next(
        row
        for row in map(json.loads, Path(cfg.output_path).read_text().splitlines())
        if row["event_type"] == "infer.audio.request"
    )
    assert row["first_transcript_ms"] is not None
    assert row["rtf"] is None and row["output_audio_seconds"] is None
    assert "private" not in Path(cfg.output_path).read_text()


def test_audio_replay_and_fixture_selection_are_deterministic(tmp_path):
    from stormlog.infer.arrivals import ArrivalTrace, load_arrival_trace

    cfg = config(
        tmp_path,
        arrival_mode="replay",
        arrival_trace=ArrivalTrace((0, 0.01, 0.02), "fixture"),
        request_count=None,
    )
    with server("tts") as (url, _):
        report = AudioProfiler(replace(cfg, endpoint=url)).run()
    assert report["summary"]["total_requests"] == 3
    assert report["cases"][0]["throughput"]["audio_seconds_per_second"] is None
    assert load_arrival_trace(cfg.output_path).offsets_seconds == (0, 0.01, 0.02)
    path = Path(cfg.manifest_path)
    path.write_text(
        path.read_text()
        + "\n"
        + json.dumps(
            {"schema_version": 1, "id": "two", "task": "tts", "text": "second"}
        )
    )
    workload = AudioWorkload(replace(cfg, seed=1))
    assert [workload.select(index).fixture_id for index in range(4)] == [
        "two",
        "fixture-one",
        "two",
        "fixture-one",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        dict(response_format="json"),
        dict(timeout_seconds=float("nan")),
        dict(max_fixture_bytes=0),
        dict(task="chat"),
    ],
)
def test_invalid_config_before_output(tmp_path, bad):
    cfg = config(tmp_path)
    with pytest.raises(ValueError):
        AudioProfiler(replace(cfg, **bad))
    assert not Path(cfg.output_path).exists()


@pytest.mark.parametrize(
    "flag", ["--input-tokens", "--slo", "--cache-reset-url", "--trace"]
)
def test_audio_cli_rejects_text_only_flags(tmp_path, flag):
    from stormlog.infer.cli import build_parser

    cfg = config(tmp_path)
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(
            [
                "profile-audio",
                "--task",
                "tts",
                "--model",
                "model",
                "--endpoint",
                cfg.endpoint,
                "--manifest",
                cfg.manifest_path,
                "--output",
                cfg.output_path,
                flag,
                "1",
            ]
        )
    assert raised.value.code == 2
    assert not Path(cfg.output_path).exists()


def test_audio_cli_runs_with_separate_defaults(tmp_path, capsys):
    from stormlog.infer.cli import main

    cfg = config(tmp_path)
    with server("tts") as (url, calls):
        assert (
            main(
                [
                    "profile-audio",
                    "--task",
                    "tts",
                    "--model",
                    "model",
                    "--endpoint",
                    url,
                    "--manifest",
                    cfg.manifest_path,
                    "--output",
                    cfg.output_path,
                ]
            )
            == 0
        )
    assert calls[0]["stream"] is False
    assert "Audio inference" in capsys.readouterr().out
