#!/bin/sh
# Idempotent L4 setup for protocol v2. /home persists across pause; the root
# filesystem does not, so system packages (nsys) are reinstalled every session.
set -eu
ROOT=/home/v2
SRC=$ROOT/src
mkdir -p "$ROOT/env" "$ROOT/build"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$ROOT/env/setup.log"; }

if [ ! -x "$ROOT/venv/bin/python" ]; then
  log "creating venv"
  python3 -m venv "$ROOT/venv"
  "$ROOT/venv/bin/pip" install -q --upgrade pip
  "$ROOT/venv/bin/pip" install -q vllm==0.30.0 psutil cmake
fi

if ! command -v nsys >/dev/null 2>&1; then
  log "installing nsight-systems-cli"
  . /etc/os-release
  repo="ubuntu$(echo "$VERSION_ID" | tr -d .)"
  keyring=/tmp/cuda-keyring.deb
  wget -q -O "$keyring" "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i "$keyring" >/dev/null
  apt-get update -qq >/dev/null
  package=$(apt-cache search '^nsight-systems-20' | sort -V | tail -1 | cut -d' ' -f1)
  apt-get install -y -qq "$package" >/dev/null
  ln -sf "$(find /opt/nvidia/nsight-systems -path '*/bin/nsys' | sort -V | tail -1)" /usr/local/bin/nsys
fi

if [ ! -f /usr/include/zlib.h ]; then
  log "installing zlib headers"
  apt-get install -y -qq zlib1g-dev >/dev/null
fi

log "building CUPTI helper"
"$ROOT/venv/bin/cmake" -S "$SRC/native/cupti" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >"$ROOT/env/cmake-configure.log" 2>&1
"$ROOT/venv/bin/cmake" --build "$ROOT/build" -j >"$ROOT/env/cmake-build.log" 2>&1
# The toolkit build links the system CUPTI (12.x); kept only as a control,
# because CUPTI must match the target's CUDA runtime (torch cu13 here).
SYSTEM_HELPER=$(find "$ROOT/build" -name 'libstormlog_cupti_injection.so' | head -1)
test -n "$SYSTEM_HELPER"
ln -sf "$SYSTEM_HELPER" "$ROOT/libstormlog_cupti_system.so"
CU13=$("$ROOT/venv/bin/python" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))")
g++ -std=c++17 -O2 -shared -fPIC -Wall -Wextra -I"$CU13/include" \
  "$SRC/native/cupti/stormlog_cupti_injection.cpp" -o "$ROOT/libstormlog_cupti_cu13.so" \
  "$CU13/lib/libcupti.so.13" -lz -ldl -lpthread -Wl,-rpath,"$CU13/lib" >"$ROOT/env/g++-cu13.log" 2>&1
HELPER=$ROOT/libstormlog_cupti_cu13.so
ln -sf "$HELPER" "$ROOT/libstormlog_cupti.so"

log "prefetching model"
"$ROOT/venv/bin/python" -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-0.5B-Instruct', revision='c89bee90d9f811437d9735454613c35b4a3c4dc8')" >/dev/null

log "recording environment"
{
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
  nsys --version
  uname -a
  "$ROOT/venv/bin/python" -c "import torch, vllm; print('torch', torch.__version__, torch.version.cuda); print('vllm', vllm.__version__)"
  ldd "$HELPER" | grep -i cupti
  sha256sum "$HELPER" "$SYSTEM_HELPER"
} >"$ROOT/env/environment.txt" 2>&1
"$ROOT/venv/bin/pip" freeze >"$ROOT/env/pip-freeze.txt"
"$ROOT/venv/bin/python" - >"$ROOT/env/vllm-profiler-config.txt" 2>&1 <<'PY' || true
import dataclasses
from vllm.config import ProfilerConfig
print([f.name for f in dataclasses.fields(ProfilerConfig)])
PY
log "setup complete"
