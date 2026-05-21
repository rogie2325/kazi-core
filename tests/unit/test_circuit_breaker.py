"""
Unit tests for graph_builder._CircuitBreaker.

Covers all three states (CLOSED, OPEN, HALF_OPEN) and every
state transition: CLOSED→OPEN, OPEN→HALF_OPEN, HALF_OPEN→CLOSED,
HALF_OPEN→OPEN.  Pure Python — no LLM calls or external dependencies.
"""
import time

from kazi.brain.graph_builder import _CBState, _CircuitBreaker

# ── CLOSED state ──────────────────────────────────────────────────────────────

def test_closed_allows_requests():
    cb = _CircuitBreaker(threshold=3, cooldown=60.0)
    assert cb.allow() is True


def test_closed_stays_closed_on_success():
    cb = _CircuitBreaker(threshold=3, cooldown=60.0)
    cb.record_success()
    assert cb.state == _CBState.CLOSED
    assert cb.failure_count == 0


def test_closed_counts_failures():
    cb = _CircuitBreaker(threshold=3, cooldown=60.0)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    assert cb.state == _CBState.CLOSED  # threshold not reached yet


def test_closed_opens_at_threshold():
    cb = _CircuitBreaker(threshold=3, cooldown=60.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == _CBState.OPEN


def test_success_resets_failure_count():
    cb = _CircuitBreaker(threshold=5, cooldown=60.0)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.state == _CBState.CLOSED


# ── OPEN state ────────────────────────────────────────────────────────────────

def test_open_blocks_requests_before_cooldown():
    cb = _CircuitBreaker(threshold=1, cooldown=60.0)
    cb.record_failure()
    assert cb.state == _CBState.OPEN
    assert cb.allow() is False


def test_open_transitions_to_half_open_after_cooldown():
    cb = _CircuitBreaker(threshold=1, cooldown=0.01)
    cb.record_failure()
    assert cb.state == _CBState.OPEN
    time.sleep(0.02)
    result = cb.allow()
    assert result is True
    assert cb.state == _CBState.HALF_OPEN


def test_open_stores_opened_at_timestamp():
    cb = _CircuitBreaker(threshold=1, cooldown=60.0)
    before = time.monotonic()
    cb.record_failure()
    assert cb.opened_at >= before


# ── HALF_OPEN state ───────────────────────────────────────────────────────────

def test_half_open_allows_one_request():
    cb = _CircuitBreaker(threshold=1, cooldown=0.01)
    cb.record_failure()
    time.sleep(0.02)
    cb.allow()  # triggers CLOSED→OPEN→HALF_OPEN
    assert cb.state == _CBState.HALF_OPEN
    # HALF_OPEN allows the test request
    assert cb.allow() is True


def test_half_open_success_closes_breaker():
    cb = _CircuitBreaker(threshold=1, cooldown=0.01)
    cb.record_failure()
    time.sleep(0.02)
    cb.allow()  # → HALF_OPEN
    cb.record_success()
    assert cb.state == _CBState.CLOSED
    assert cb.failure_count == 0


def test_half_open_failure_reopens_breaker():
    cb = _CircuitBreaker(threshold=1, cooldown=0.01)
    cb.record_failure()
    time.sleep(0.02)
    cb.allow()  # → HALF_OPEN
    cb.record_failure()  # fail in HALF_OPEN → back to OPEN
    assert cb.state == _CBState.OPEN


# ── record_success in non-CLOSED state logs recovery ─────────────────────────

def test_record_success_from_half_open_logs(caplog):
    import logging
    cb = _CircuitBreaker(threshold=1, cooldown=0.01)
    cb.record_failure()
    time.sleep(0.02)
    cb.allow()  # → HALF_OPEN
    with caplog.at_level(logging.INFO, logger="kazi.brain.graph_builder"):
        cb.record_success()
    assert "recovered" in caplog.text or "CLOSED" in caplog.text


# ── _build_human_message (static method, no LLM) ─────────────────────────────

def test_build_human_message_text_only():
    from kazi.brain.graph_builder import GraphBrain
    msg = GraphBrain._build_human_message("Hello")
    from langchain_core.messages import HumanMessage
    assert isinstance(msg, HumanMessage)
    assert msg.content == "Hello"


def test_build_human_message_with_http_image_url():
    from kazi.brain.graph_builder import GraphBrain
    msg = GraphBrain._build_human_message(
        "What is in this image?",
        images=["https://example.com/img.jpg"],
    )
    content = msg.content
    assert isinstance(content, list)
    text_parts = [p for p in content if p.get("type") == "text"]
    img_parts = [p for p in content if p.get("type") == "image_url"]
    assert text_parts[0]["text"] == "What is in this image?"
    assert img_parts[0]["image_url"]["url"] == "https://example.com/img.jpg"


def test_build_human_message_with_bytes_image():
    from kazi.brain.graph_builder import GraphBrain
    fake_jpeg = b"\xff\xd8\xff" + b"\x00" * 10  # minimal fake JPEG bytes
    msg = GraphBrain._build_human_message("Describe this.", images=[fake_jpeg])
    content = msg.content
    assert isinstance(content, list)
    img_parts = [p for p in content if p.get("type") == "image_url"]
    assert len(img_parts) == 1
    assert "data:image/jpeg;base64," in img_parts[0]["image_url"]["url"]


def test_build_human_message_with_local_file_image(tmp_path):
    from kazi.brain.graph_builder import GraphBrain
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    msg = GraphBrain._build_human_message("Analyse.", images=[str(img_file)])
    content = msg.content
    img_parts = [p for p in content if p.get("type") == "image_url"]
    assert len(img_parts) == 1
    assert "base64," in img_parts[0]["image_url"]["url"]


def test_build_human_message_no_images_returns_simple_message():
    from kazi.brain.graph_builder import GraphBrain
    msg = GraphBrain._build_human_message("Simple", images=None)
    assert msg.content == "Simple"
