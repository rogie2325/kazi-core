"""
JSON Schema export for KaziConfig.

Lets a coding LLM (or an IDE / YAML linter) validate a generated config file
before submitting it::

    python -m kazi config-schema > kazi.schema.json

    # YAML editor with the schema:
    # yaml-language-server: $schema=./kazi.schema.json
    llm:
      provider: openai     # ← autocomplete + validation against the enum
      model: gpt-4o

The schema is derived from the dataclass tree under ``KaziConfig`` — no
extra dependencies (no Pydantic) and no manual duplication.  When a new
dataclass field is added, the schema picks it up automatically.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import typing
from typing import Any


def _python_type_to_schema(tp: Any) -> dict:
    """Map a Python type annotation to a JSON Schema fragment."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    # Union / Optional → first non-None type
    if origin is typing.Union or (hasattr(tp, "__class__") and tp.__class__.__name__ == "UnionType"):
        non_none = [a for a in args if a is not type(None)]
        # Optional[T] → T schema with nullable
        schema = _python_type_to_schema(non_none[0])
        if type(None) in args:
            schema = {"oneOf": [schema, {"type": "null"}]}
        return schema

    if origin in (list, tuple, set, frozenset) or tp in (list, tuple, set, frozenset):
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_schema(item_type)}

    if origin is dict or tp is dict:
        value_type = args[1] if len(args) >= 2 else Any
        return {"type": "object", "additionalProperties": _python_type_to_schema(value_type)}

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return {"type": "string", "enum": [e.value for e in tp]}

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _dataclass_to_schema(tp)

    if tp is str:
        return {"type": "string"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is type(None):
        return {"type": "null"}

    # Fallback for Any / complex generics
    return {}


def _dataclass_to_schema(cls: type) -> dict:
    """Convert a dataclass type into a JSON Schema object definition."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")

    properties: dict[str, dict] = {}
    required: list[str] = []

    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {f.name: f.type for f in dataclasses.fields(cls)}

    for field in dataclasses.fields(cls):
        if field.name.startswith("_"):
            continue
        annotation = hints.get(field.name, field.type)
        field_schema = _python_type_to_schema(annotation)
        # Add the field's docstring-style description from the dataclass field metadata
        if field.metadata.get("description"):
            field_schema["description"] = field.metadata["description"]
        properties[field.name] = field_schema

        has_default = (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        if not has_default:
            required.append(field.name)

    out: dict[str, Any] = {
        "type": "object",
        "title": cls.__name__,
        "properties": properties,
    }
    if required:
        out["required"] = required
    return out


def kazi_config_schema() -> dict:
    """
    Return the full JSON Schema (draft 2020-12) for KaziConfig.

    Suitable for IDEs (YAML / JSON language servers), CI validation, and
    LLM-generated config validation before submission.
    """
    from kazi.core.config import KaziConfig
    schema = _dataclass_to_schema(KaziConfig)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "KaziConfig"
    schema["description"] = (
        "Top-level configuration for a kazi orchestrator. "
        "Wire LLM, RAG, MCP, A2A, security, and routing layers in one file."
    )
    return schema


def kazi_config_schema_json(*, indent: int = 2) -> str:
    """Return the KaziConfig JSON Schema serialised as a JSON string."""
    return json.dumps(kazi_config_schema(), indent=indent, default=str)
