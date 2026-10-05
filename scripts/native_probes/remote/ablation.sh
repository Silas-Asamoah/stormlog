#!/bin/sh
# Waits for Phase 3, switches to the ablation-capable helper, runs the ablations.
set -u
until grep -q "PHASE3 COMPLETE" /home/v2/phase3.log 2>/dev/null; do sleep 30; done
CU13=/home/v2/venv/lib/python3.10/site-packages/nvidia/cu13
g++ -std=c++17 -O2 -shared -fPIC -Wall -Wextra -I"$CU13/include" \
  /home/v2/src/native/cupti/stormlog_cupti_injection.cpp -o /home/v2/libstormlog_cupti_cu13_ablation.so \
  "$CU13/lib/libcupti.so.13" -lz -ldl -lpthread -Wl,-rpath,"$CU13/lib" > /home/v2/env/g++-ablation.log 2>&1
sha256sum /home/v2/libstormlog_cupti_cu13_ablation.so >> /home/v2/env/environment.txt
ln -sf /home/v2/libstormlog_cupti_cu13_ablation.so /home/v2/libstormlog_cupti.so
cd /home/v2/src && /home/v2/venv/bin/python -m scripts.native_probes.remote.run_plan benchmarks/native_probes/plans_v2/ablation.json > /home/v2/ablation.out 2>&1
echo "ABLATION COMPLETE $(date -u +%FT%TZ)" >> /home/v2/phase3.log
