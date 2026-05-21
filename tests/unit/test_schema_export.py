"""Tests for kazi.core.schema — JSON Schema export for KaziConfig."""
from __future__ import annotations

import json

from kazi.core.config import KaziConfig, LLMConfig
from kazi.core.schema import (
    _dataclass_to_schema,
    _python_type_to_schema,
    kazi_config_schema,
    kazi_config_schema_json,
)

# ── _python_type_to_schema primitives ─────────────────────────────────────────

def test_python_to_schema_primitive_str():
    assert _python_type_to_schema(str) == {"type": "string"}


def test_python_to_schema_primitive_int():
    assert _python_type_to_schema(int) == {"type": "integer"}


def test_python_to_schema_primitive_float():
    assert _python_type_to_schema(float) == {"type": "number"}


def test_python_to_schema_primitive_bool():
    assert _python_type_to_schema(bool) == {"type": "boolean"}


def test_python_to_schema_list():
    schema = _python_type_to_schema(list[str])
    assert schema["type"] == "array"
    assert schema["items"] == {"type": "string"}


def test_python_to_schema_dict():
    schema = _python_type_to_schema(dict[str, int])
    assert schema["type"] == "object"
    assert schema["additionalProperties"] == {"type": "integer"}


def test_python_to_schema_enum():
    from kazi.core.config import LLMProvider
    schema = _python_type_to_schema(LLMProvider)
    assert schema["type"] == "string"
    assert set(schema["enum"]) == {"openai", "anthropic", "google", "local"}


def test_python_to_schema_optional_int():
    schema = _python_type_to_schema(int | None)
    # Optional[int] → oneOf [{integer}, {null}]
    assert "oneOf" in schema
    types = {item.get("type") for item in schema["oneOf"]}
    assert types == {"integer", "null"}


# ── _dataclass_to_schema ──────────────────────────────────────────────────────

def test_dataclass_to_schema_llm_config_has_known_fields():
    schema = _dataclass_to_schema(LLMConfig)
    assert schema["type"] == "object"
    assert schema["title"] == "LLMConfig"
    props = schema["properties"]
    assert "provider" in props
    assert "model" in props
    assert "temperature" in props
    assert "seed" in props
    # provider should be an enum
    assert props["provider"]["type"] == "string"
    assert "enum" in props["provider"]


def test_dataclass_to_schema_skips_underscore_fields():
    schema = _dataclass_to_schema(LLMConfig)
    for name in schema["properties"]:
        assert not name.startswith("_")


# ── Top-level config schema ───────────────────────────────────────────────────

def test_kazi_config_schema_has_all_sections():
    schema = kazi_config_schema()
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert schema["title"] == "KaziConfig"
    props = schema["properties"]
    for section in ("llm", "rag", "mcp", "a2a", "memory", "security"):
        assert section in props, f"missing section: {section}"


def test_kazi_config_schema_is_valid_json():
    payload = kazi_config_schema_json()
    parsed = json.loads(payload)
    assert parsed["title"] == "KaziConfig"


def test_kazi_config_schema_to_json_schema_classmethod_works():
    """KaziConfig.to_json_schema() should delegate and return the same shape."""
    via_method = KaziConfig.to_json_schema()
    via_func = kazi_config_schema()
    assert via_method["title"] == via_func["title"]
    assert via_method["properties"].keys() == via_func["properties"].keys()
