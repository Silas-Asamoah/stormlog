import gzip
from pathlib import Path
from typing import Any, Dict, List


def _is_protobuf_version_error(exc: Exception) -> bool:
    exception_type = type(exc)
    return (
        exception_type.__module__ == "google.protobuf.runtime_version"
        and exception_type.__name__ == "VersionError"
    )


def parse_jax_memory_profile(file_path: str) -> Dict[str, Any]:
    """Parse a JAX .prof (gzipped pprof protobuf) using the official protobuf schema."""
    profile_pb2 = _load_profile_schema()

    path = Path(file_path)
    try:
        with gzip.open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JAX memory profile not found: {path}") from exc
    except PermissionError as exc:
        raise PermissionError(f"JAX memory profile is not readable: {path}") from exc

    profile = profile_pb2.Profile()
    profile.ParseFromString(data)

    string_table = profile.string_table

    # Map function_id -> function name
    functions: Dict[int, str] = {}
    for func in profile.function:
        functions[func.id] = string_table[func.name]

    # Map location_id -> [function_names]
    locations: Dict[int, List[str]] = {}
    for loc in profile.location:
        names = []
        for line in loc.line:
            func_name = functions.get(line.function_id, "<unknown>")
            names.append(func_name)
        locations[loc.id] = names

    # Flatten samples
    samples = []
    for sample in profile.sample:
        # pprof puts innermost call first, so reverse to get root->leaf stack
        stack = []
        for loc_id in sample.location_id:
            loc_names = locations.get(loc_id, ["<unknown>"])
            # The line entries in a location are innermost-first too
            stack.extend(loc_names)

        stack.reverse()
        samples.append({"stack": stack, "values": list(sample.value)})

    return {"samples": samples}


def _load_profile_schema() -> Any:
    try:
        from . import profile_pb2
    except Exception as exc:
        if not isinstance(exc, ImportError) and not _is_protobuf_version_error(exc):
            raise
        raise ImportError(
            "JAX memory profile parsing requires protobuf>=6.31.1 for the "
            "bundled profile schema. Install stormlog[jax] in a compatible "
            "environment."
        ) from exc

    return profile_pb2
