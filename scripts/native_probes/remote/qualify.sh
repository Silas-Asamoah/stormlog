#!/bin/sh
# Qualification run for `--trace cupti` (issue #118), declared in
# benchmarks/native_probes/qualification_cupti_cli.json. Records the plan's
# hash before any trial, then runs checks A, C, D and the harness trials.
set -u
cd /home/v2/src || exit 2
LOG=/home/v2/qualify.log
PLAN=benchmarks/native_probes/qualification_cupti_cli.json
echo "$(date -u +%FT%TZ) PLAN $(sha256sum "$PLAN")" >> "$LOG"
echo "$(date -u +%FT%TZ) START checks" >> "$LOG"
/home/v2/venv/bin/python -m scripts.native_probes.remote.qualify_cli /home/v2/runs/qualify-checks \
  > /home/v2/qualify-checks.out 2>&1
echo "$(date -u +%FT%TZ) END checks exit=$?" >> "$LOG"
echo "$(date -u +%FT%TZ) START trials" >> "$LOG"
/home/v2/venv/bin/python -m scripts.native_probes.remote.run_plan "$PLAN" \
  > /home/v2/qualify-trials.out 2>&1
echo "$(date -u +%FT%TZ) END trials exit=$?" >> "$LOG"
echo "$(date -u +%FT%TZ) QUALIFY COMPLETE" >> "$LOG"
