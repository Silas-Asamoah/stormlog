"""Shared validation for immutable native-probe evidence documents."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

_SCHEMAS = Path(__file__).with_name("schemas")
_jsonschema = importlib.import_module("jsonschema")


def validate_document(value: object, schema_name: str) -> None:
    """Validate a document and report schema failures as contract errors."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{schema_name}: document is not valid JSON: {error}"
        ) from error

    with (_SCHEMAS / schema_name).open(encoding="utf-8") as source:
        schema: dict[str, Any] = json.load(source)
    try:
        _jsonschema.Draft202012Validator(schema).validate(value)
    except _jsonschema.ValidationError as error:
        raise ValueError(f"{schema_name}: {error.message}") from error
