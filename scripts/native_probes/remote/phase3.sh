#!/bin/sh
# Phase 3: vLLM eager and CUDA-graph fidelity, then microbenchmark fidelity,
# induced loss and coexistence. Each step logs its exit code and continues.
set -u
cd /home/v2/src || exit 2
PY=/home/v2/venv/bin/python
LOG=/home/v2/phase3.log
step() {
  name=$1
  shift
  echo "$(date -u +%FT%TZ) START $name" >> "$LOG"
  "$@" > "/home/v2/phase3-$name.out" 2>&1
  echo "$(date -u +%FT%TZ) END $name exit=$?" >> "$LOG"
}
step graph-plan $PY -m scripts.native_probes.remote.run_plan benchmarks/native_probes/plans_v2/graph.json
step graph-compare $PY -m scripts.native_probes.remote.graph_fidelity /home/v2/runs/graph
step micro $PY -m scripts.native_probes.remote.micro_fidelity /home/v2/runs/micro
echo "$(date -u +%FT%TZ) PHASE3 COMPLETE" >> "$LOG"
