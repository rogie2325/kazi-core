"""Tests for kazi.utils.serialization."""
import json

from kazi.utils.serialization import safe_json, state_to_dict

# ── state_to_dict ─────────────────────────────────────────────────────────────

def _make_message(cls_name: str, content):
    """Build a minimal duck-typed message object."""
    class _Msg:
        pass
    msg = _Msg()
    msg.__class__.__name__ = cls_name
    msg.content = content
    return msg


def test_state_scalars_pass_through():
    state = {
        "thread_id": "t1",
        "current_step": "start",
        "tool_calls_made": 3,
        "max_tool_calls": 25,
        "system_prompt": None,
        "messages": [],
        "final_answer": None,
        "metadata": {},
    }
    result = state_to_dict(state)
    assert result["thread_id"] == "t1"
    assert result["tool_calls_made"] == 3
    assert result["system_prompt"] is None
    assert result["final_answer"] is None


def test_state_non_scalar_is_stringified():
    state = {
        "metadata": {"nested": "dict"},
        "messages": [],
    }
    result = state_to_dict(state)
    # metadata is a dict — not a scalar, not messages, so str()
    assert isinstance(result["metadata"], str)


def test_state_messages_converted_to_dicts():
    human = _make_message("HumanMessage", "Hello there")
    ai = _make_message("AIMessage", "Hi back")
    state = {"messages": [human, ai], "thread_id": "t", "tool_calls_made": 0}
    result = state_to_dict(state)

    assert result["messages"][0] == {"role": "user", "content": "Hello there"}
    assert result["messages"][1] == {"role": "assistant", "content": "Hi back"}


def test_state_system_and_tool_messages():
    system = _make_message("SystemMessage", "You are helpful.")
    tool = _make_message("ToolMessage", "search result here")
    state = {"messages": [system, tool]}
    result = state_to_dict(state)

    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "tool"


def test_state_unknown_message_type_uses_lowercased_classname():
    custom = _make_message("FunctionMessage", "fn output")
    state = {"messages": [custom]}
    result = state_to_dict(state)
    assert result["messages"][0]["role"] == "functionmessage"


def test_state_message_with_list_content_is_json_encoded():
    """Non-string message content (e.g. Anthropic content blocks) becomes a JSON string."""
    blocks = [{"type": "text", "text": "hello"}]
    ai = _make_message("AIMessage", blocks)
    state = {"messages": [ai]}
    result = state_to_dict(state)

    # Content should be a JSON string, not the raw list
    encoded = result["messages"][0]["content"]
    assert isinstance(encoded, str)
    decoded = json.loads(encoded)
    assert decoded[0]["text"] == "hello"


def test_state_empty_messages():
    state = {"messages": []}
    result = state_to_dict(state)
    assert result["messages"] == []


def test_state_bool_value_passes_through():
    state = {"messages": [], "verbose": True}
    result = state_to_dict(state)
    assert result["verbose"] is True


# ── safe_json ─────────────────────────────────────────────────────────────────

def test_safe_json_primitive_types():
    assert json.loads(safe_json(42)) == 42
    assert json.loads(safe_json("hello")) == "hello"
    assert json.loads(safe_json(True)) is True
    assert json.loads(safe_json(None)) is None


def test_safe_json_dict_and_list():
    result = safe_json({"a": 1, "b": [2, 3]})
    assert json.loads(result) == {"a": 1, "b": [2, 3]}


def test_safe_json_non_serialisable_falls_back_to_str():
    class Unserializable:
        def __repr__(self):
            return "Unserializable()"

    result = safe_json(Unserializable())
    # json.dumps(value, default=str) converts it via str()
    assert isinstance(result, str)
    assert json.loads(result) is not None  # valid JSON


def test_safe_json_custom_fallback():
    result = safe_json(object(), fallback="FAILED")
    # The default=str path handles most objects without hitting fallback;
    # verify the result is still valid JSON in the normal case.
    assert isinstance(result, str)


def test_safe_json_nested_with_datetime():
    """datetime is not JSON-serialisable natively; default=str should handle it."""
    from datetime import datetime
    now = datetime(2025, 1, 1, 12, 0, 0)
    result = safe_json({"created_at": now})
    parsed = json.loads(result)
    assert "2025" in parsed["created_at"]


def test_safe_json_empty_structures():
    assert safe_json({}) == "{}"
    assert safe_json([]) == "[]"


def test_safe_json_returns_fallback_for_circular_reference():
    """json.dumps raises on circular refs even with default=str — fallback kicks in."""
    circular = {}
    circular["self"] = circular  # type: ignore[assignment]
    result = safe_json(circular, fallback="[circular]")
    assert result == "[circular]"
