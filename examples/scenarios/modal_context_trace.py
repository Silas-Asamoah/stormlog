"""Capture a bounded real vLLM trace for the offline context-compaction scenario.

Run with the separately installed Modal CLI (not a Stormlog dependency)::

    modal run --profile YOUR_PROFILE examples/scenarios/modal_context_trace.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
VLLM_VERSION = "0.30.0"
VOLUME_NAME = "stormlog-issue-259-traces"
REMOTE_ROOT = Path("/captures")
CUDA_HOME = "/usr/local/lib/python3.12/site-packages/nvidia/cu13"

app = modal.App("stormlog-issue-259-qwen-trace")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
    # PyTorch pins 13.0 headers; unconstrained JIT tools otherwise resolve to 13.4.
    .pip_install(
        "nvidia-cuda-nvcc==13.0.88",
        "nvidia-cuda-crt==13.0.88",
        "nvidia-nvvm==13.0.88",
        "nvidia-cuda-runtime==13.0.96",
    )
    .run_commands(
        f"test -x {CUDA_HOME}/bin/nvcc",
        f"ln -s lib {CUDA_HOME}/lib64",
        f"ln -s libcudart.so.13 {CUDA_HOME}/lib/libcudart.so",
        f"{CUDA_HOME}/bin/nvcc --version",
        "printf '#include <cuda_runtime.h>\\n"
        "#include <cuda/std/type_traits>\\n"
        "__global__ void check() {}\\n"
        "int main() { return 0; }\\n' > /tmp/stormlog-cuda-check.cu",
        f"{CUDA_HOME}/bin/nvcc --cudart shared /tmp/stormlog-cuda-check.cu "
        "-I/usr/local/lib/python3.12/site-packages/flashinfer/data/cccl/"
        "libcudacxx/include -o /tmp/stormlog-cuda-check",
    )
    .env({"CUDA_HOME": CUDA_HOME})
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _prompts() -> list[str]:
    return [
        f"Request {index}. Explain the following chess position considerations. "
        + "Development, king safety, pawn structure and active pieces matter. "
        * (4 + index % 4 * 4)
        + "Give a detailed continuation:"
        for index in range(16)
    ]


def _pack(root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(root)))
    return buffer.getvalue()


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=16384,
    timeout=900,
    retries=0,
    max_containers=1,
    volumes={str(REMOTE_ROOT): volume},
)
def capture(run_id: str) -> bytes:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    root = REMOTE_ROOT / run_id
    trace_dir = root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=False)
    gpu_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    (root / "nvidia-smi.txt").write_text(gpu_csv + "\n")
    nvcc = subprocess.check_output([f"{CUDA_HOME}/bin/nvcc", "--version"], text=True)
    (root / "nvcc.txt").write_text(nvcc)
    (root / "packages.txt").write_text(
        subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    )
    profiler = {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir),
        "torch_profiler_use_gzip": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_with_flops": False,
    }
    engine_args = {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": 1024,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 1024,
        "gpu_memory_utilization": 0.5,
        "enable_prefix_caching": False,
        "seed": 259,
        "profiler_config": profiler,
    }
    sampling = {"temperature": 0.0, "max_tokens": 64, "ignore_eos": True}
    prompts = _prompts()
    _write_json(root / "workload.json", {"prompts": prompts, "sampling": sampling})
    metadata = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_toolkit_root": CUDA_HOME,
        "nvcc": nvcc,
        "python": platform.python_version(),
        "gpu_csv": gpu_csv,
        "engine_args": engine_args,
        "warmup_batches": 1,
        "profiled_batches": 2,
        "requests_per_batch": len(prompts),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "volume": VOLUME_NAME,
        "volume_directory": run_id,
    }
    _write_json(root / "capture.json", metadata)
    volume.commit()
    llm = LLM(**engine_args)
    params = SamplingParams(**sampling)
    llm.generate(prompts, params, use_tqdm=False)
    generated = []
    started = time.monotonic()
    llm.start_profile()
    try:
        for batch in range(2):
            outputs = llm.generate(prompts, params, use_tqdm=False)
            generated.extend(
                {
                    "batch": batch,
                    "request_id": output.request_id,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(output.outputs[0].token_ids),
                    "text": output.outputs[0].text,
                }
                for output in outputs
            )
    finally:
        llm.stop_profile()
    metadata["profiled_wall_seconds"] = time.monotonic() - started
    # Worker profiling writes may finish asynchronously after stop_profile.
    time.sleep(10)
    traces = sorted(trace_dir.rglob("*.pt.trace.json*"))
    if not traces:
        raise RuntimeError("vLLM did not write a profiler trace")
    metadata["traces"] = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in traces
    ]
    metadata["completed_requests"] = len(generated)
    metadata["generated_tokens"] = sum(row["output_tokens"] for row in generated)
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(root / "outputs.json", generated)
    _write_json(root / "capture.json", metadata)
    volume.commit()
    print(json.dumps(metadata, indent=2))
    return _pack(root)


@app.local_entrypoint()
def main(output_dir: str = "artifacts/issue-259") -> None:
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    root = Path(output_dir).resolve() / run_id
    root.mkdir(parents=True, exist_ok=False)
    _write_json(
        root / "local-provenance.json",
        {
            "run_id": run_id,
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True)
            ),
            "capture_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )
    payload = capture.remote(run_id)
    (root / "capture.tar.gz").write_bytes(payload)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(root, filter="data")
    print(f"Saved capture and provenance to {root}")
