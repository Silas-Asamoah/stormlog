[← Back to main docs](index.md)

# TelemetryEvent v2 Schema

`TelemetryEvent v2` is the canonical event format for tracker exports.

Schema file:

`docs/schemas/telemetry_event_v2.schema.json`

## Required fields

- `schema_version` (`2`)
- `timestamp_ns`
- `event_type`
- `collector`
- `sampling_interval_ms`
- `pid`
- `host`
- `device_id`
- `allocator_allocated_bytes`
- `allocator_reserved_bytes`
- `allocator_active_bytes`
- `allocator_inactive_bytes`
- `allocator_change_bytes`
- `device_used_bytes`
- `device_free_bytes`
- `device_total_bytes`
- `context`
- `metadata`

## Distributed identity fields

`TelemetryEvent v2` also recognizes these top-level distributed identity fields:

- `job_id` (`string | null`)
- `rank` (`integer`)
- `local_rank` (`integer`)
- `world_size` (`integer`)

New exports always emit these fields. For single-process runs, the defaults are:

- `job_id` -> `null`
- `rank` -> `0`
- `local_rank` -> `0`
- `world_size` -> `1`

`TelemetryEvent v2` validation is strict:
- Unknown top-level fields are rejected.
- `metadata` must be a JSON object (`dict` in Python terms).
- `rank` and `local_rank` must be >= `0`.
- `world_size` must be >= `1`.
- `rank` and `local_rank` must be < `world_size`.

## Collector values

- `gpumemprof.cuda_tracker`
- `gpumemprof.rocm_tracker`
- `gpumemprof.mps_tracker`
- `gpumemprof.cpu_tracker`
- `tfmemprof.memory_tracker`

## Backend capability metadata

Tracker exports may include backend capability hints under `metadata`:

- `backend`
- `supports_device_total`
- `supports_device_free`
- `sampling_source`

## Legacy v1 to v2 conversion defaults

Conversion is permissive by default in `gpumemprof.telemetry.telemetry_event_from_record`.
Legacy conversion is attempted only when `schema_version` is absent.

If `schema_version` is present:
- It must be an integer.
- It must be exactly `2`.
- Any other value is rejected (no legacy fallback).

- Missing `pid` -> `-1`
- Missing `host` -> `"unknown"`
- Missing `device_id` -> inferred from `device` if possible, otherwise `-1`
- Missing `allocator_reserved_bytes` -> `allocator_allocated_bytes`
- Missing `allocator_change_bytes` -> `0`
- Missing `device_used_bytes` -> `allocator_allocated_bytes`
- Missing `device_total_bytes` and `device_free_bytes` -> `null`
- Missing `event_type` -> `type` field if present, else `"sample"`
- Missing distributed identity -> `job_id: null`, `rank: 0`, `local_rank: 0`, `world_size: 1`
- Legacy `metadata_*` fields are folded into the v2 `metadata` object

If a legacy record is missing a valid timestamp, conversion fails.

## Distributed env inference

Tracker constructors can infer distributed identity from common launcher env vars:

- PyTorch / `torchrun`: `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `TORCHELASTIC_RUN_ID`
- Open MPI: `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_LOCAL_RANK`, `OMPI_COMM_WORLD_SIZE`
- Slurm: `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`, `SLURM_JOB_ID`

CLI and Python API callers can override these values explicitly.

## Python API

Use the public conversion/validation helpers in `gpumemprof.telemetry`:

```python
from gpumemprof.telemetry import (
    load_telemetry_events,
    telemetry_event_from_record,
    telemetry_event_to_dict,
    validate_telemetry_record,
)
```

- `load_telemetry_events(path, permissive_legacy=True, events_key=None)`
- `telemetry_event_from_record(record, permissive_legacy=True, ...)`
- `validate_telemetry_record(record)`

These APIs normalize legacy records to `schema_version: 2` and enforce required fields.
