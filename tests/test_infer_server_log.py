"""What a vLLM 0.30.0 server's log says it chose at start-up."""

from __future__ import annotations

from pathlib import Path

from stormlog.infer.server_log import parse_server_log, read_server_log

START = (
    "(EngineCore_DP0 pid=200) INFO 10-03 12:00:01 [core.py:124] Initializing a V1 "
    "LLM engine (v0.30.0) with config: model='Qwen/Qwen2.5-0.5B-Instruct', ..."
)
BACKEND = (
    "(Worker_TP{rank} pid=30{rank}) INFO 10-03 12:00:05 [cuda.py:539] Using "
    "FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', "
    "'FLEX_ATTENTION']."
)
KV = (
    "(EngineCore_DP0 pid=200) INFO 10-03 12:00:20 [kv_cache_utils.py:2395] GPU KV "
    "cache size: 1,234,560 tokens, Maximum concurrency for 32,768 tokens per "
    "request: 37.68x"
)
CAPTURE = (
    "(Worker_TP0 pid=300) Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): "
    "100%|██████████| 67/67 [00:03<00:00, 20.1it/s]"
)
CAPTURE_DECODE = (
    "(Worker_TP0 pid=300) Capturing CUDA graphs (decode, FULL): 100%| 35/35"
)
CAPTURED = (
    "(Worker_TP0 pid=300) INFO 10-03 12:00:30 [gpu_model_runner.py:6930] Graph "
    "capturing finished in 7 secs, took 0.45 GiB"
)


def _startup(version: str = "0.30.0", backend: str = "FLASH_ATTN") -> list[str]:
    return [
        START.replace("v0.30.0", f"v{version}"),
        BACKEND.format(rank=0).replace("Using FLASH_ATTN", f"Using {backend}"),
        BACKEND.format(rank=1).replace("Using FLASH_ATTN", f"Using {backend}"),
        KV,
        CAPTURE,
        CAPTURE_DECODE,
        CAPTURED,
        "(APIServer pid=100) INFO 10-03 12:01:00 [loggers.py:1] Engine 000: ...",
    ]


def test_the_choices_of_a_start_up_are_read() -> None:
    facts = parse_server_log(_startup())
    assert facts["vllm_version"] == "0.30.0"
    assert facts["attention_backend"] == "FLASH_ATTN"
    assert facts["attention_candidates"] == ["FLASH_ATTN", "FLEX_ATTENTION"]
    assert facts["kv_cache_size_tokens"] == 1_234_560
    assert facts["max_concurrency"] == 37.68
    assert facts["cudagraph_captures"] == [
        "decode:FULL",
        "mixed prefill-decode:PIECEWISE",
    ]
    assert facts["graph_capture_gib"] == 0.45
    assert facts["issues"] == []


def test_only_the_last_start_up_counts(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("\n".join(_startup(backend="TRITON_ATTN") + _startup()) + "\n")
    facts = read_server_log(log)
    assert facts["startups"] == 2
    assert facts["attention_backend"] == "FLASH_ATTN"


def test_processes_that_disagree_are_kept_as_an_issue() -> None:
    lines = _startup()
    lines[2] = lines[2].replace("Using FLASH_ATTN", "Using TRITON_ATTN")
    facts = parse_server_log(lines)
    assert facts["attention_backend"] == ["FLASH_ATTN", "TRITON_ATTN"]
    assert any("attention_backend differs" in issue for issue in facts["issues"])


def test_a_backend_chosen_by_name_is_read_without_its_enum_prefix() -> None:
    forced = (
        "(Worker_TP0 pid=300) INFO 10-03 12:00:05 [cuda.py:478] Using "
        "AttentionBackendEnum.FLASHINFER backend."
    )
    other = "(APIServer pid=100) INFO [serving.py:1] Using xgrammar backend."
    facts = parse_server_log([START, forced, other])
    assert facts["attention_backend"] == "FLASHINFER"


def test_a_log_without_a_start_up_says_so() -> None:
    facts = parse_server_log(["something else"])
    assert facts["startups"] == 0
    assert facts["attention_backend"] is None
    assert facts["issues"] == ["no vLLM V1 engine start-up in the log"]


# As vLLM 0.30.0's Model Runner V2 logged a start-up on the A30 box: the
# capture progress bars name only the mode, and a first capture for memory
# profiling comes before the KV cache is sized.
MODEL_RUNNER_V2 = [
    "(EngineCore pid=739) INFO 10-03 22:00:51 [core.py:123] Initializing a V1 LLM "
    "engine (v0.30.0) with config: model='/home/.cache/huggingface/hub/x'",
    "(EngineCore pid=739) INFO 10-03 22:00:55 [cuda.py:538] Using FLASH_ATTN "
    "attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', "
    "'TRITON_ATTN', 'FLEX_ATTENTION'].",
    "(EngineCore pid=739) Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51 "
    "[00:00<?, ?it/s]\rCapturing CUDA graphs (PIECEWISE):   4%|▍         | 2/51",
    "(EngineCore pid=739) Capturing CUDA graphs (FULL):   0%|          | 0/2 "
    "[00:00<?, ?it/s]\rCapturing CUDA graphs (FULL): 100%|██████████| 2/2",
    "(EngineCore pid=739) INFO 10-03 22:01:04 [model_runner.py:1066] Graph "
    "capturing finished in 2 secs, took 0.25 GiB",
    "(EngineCore pid=739) INFO 10-03 22:01:04 [kv_cache_utils.py:2395] GPU KV cache "
    "size: 890,960 tokens, Maximum concurrency for 4,096 tokens per request: 217.52x",
    "(EngineCore pid=739) Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51",
    "(EngineCore pid=739) Capturing CUDA graphs (FULL):   0%|          | 0/35",
    "(EngineCore pid=739) INFO 10-03 22:01:09 [model_runner.py:1066] Graph "
    "capturing finished in 3 secs, took 0.18 GiB",
]


def test_model_runner_v2_captures_are_read_and_the_last_one_counts() -> None:
    facts = parse_server_log(MODEL_RUNNER_V2)
    assert facts["attention_backend"] == "FLASH_ATTN"
    assert facts["kv_cache_size_tokens"] == 890_960
    assert facts["cudagraph_captures"] == ["FULL", "PIECEWISE"]
    # The first capture only measured memory; the second is the one kept.
    assert facts["graph_capture_gib"] == 0.18
