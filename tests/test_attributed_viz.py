from __future__ import annotations

import stormlog.attributed_viz as attributed_viz


def test_snapshot_timeline_preserves_origin_and_free_completion_accounting() -> None:
    snapshot = {
        "segments": [],
        "device_traces": [
            [
                {"action": "segment_alloc", "time_us": 1000, "size": 1024},
                {"action": "alloc", "time_us": 2000, "addr": 100, "size": 64},
                {"action": "free_requested", "time_us": 3000, "addr": 100, "size": 64},
                {"action": "oom", "time_us": 4000, "addr": 999, "size": 1000},
                {"action": "free_completed", "time_us": 5000, "addr": 100, "size": 64},
            ]
        ],
    }
    index = {
        "storage_pointer_count": 1,
        "attributed_storage_pointers": [
            {
                "storage_ptr_int": 100,
                "names": ["layer.weight"],
                "tensors": [{"shape": [4, 4], "dtype": "torch.float32"}],
            }
        ],
    }

    payload = attributed_viz._process_snapshot(snapshot, index)

    assert [
        (event["action"], event["t"], event["cum"], event["addr"])
        for event in payload["events"]
    ] == [
        ("alloc", 1.0, 64, 100),
        ("free_requested", 2.0, 64, 100),
        ("oom", 3.0, 64, 0),
        ("free_completed", 4.0, 0, 100),
    ]
    assert payload["events"][0]["name"] == "layer.weight"
    assert payload["events"][0]["shape"] == "[4, 4]"
    assert payload["events"][0]["dtype"] == "torch.float32"
    assert payload["events"][2]["name"] == "OOM"
    assert payload["peak"] == 64
    assert payload["offenders"] == []


def test_snapshot_block_offsets_and_name_fallback_are_preserved() -> None:
    snapshot = {
        "segments": [
            {
                "address": 100,
                "total_size": 160,
                "blocks": [
                    {"size": 32, "state": "inactive"},
                    {
                        "size": 64,
                        "state": "active_allocated",
                        "frames": [
                            {"name": "forward", "filename": "linear.py", "line": 1}
                        ],
                    },
                    {"size": 64, "state": "active_allocated"},
                ],
            }
        ],
        "device_traces": [],
    }

    payload = attributed_viz._process_snapshot(snapshot, {})

    assert [block["address"] for block in payload["segments"][0]["blocks"]] == [
        100,
        132,
        196,
    ]
    assert [row["name"] for row in payload["active_table"]] == [
        "Activation (Linear)",
        "Unnamed Tensor",
    ]
    assert [row["pool"] for row in payload["active_table"]] == ["unknown", "unknown"]
    assert payload["segments"][0]["segment_type"] == "large"
    assert payload["segments"][0]["blocks"][0]["name"] == ""
    assert payload["peak"] == 128
    assert payload["peak_label"] == "Active Alloc"
    assert payload["events_display"] == "n/a"
