"""Security tests for built-in tools and A2A SSRF guard."""
import asyncio

import pytest

from kazi.agents.a2a_client import _validate_agent_url
from kazi.core.exceptions import A2AConnectionError
from kazi.tools.builtin.database import _is_single_select

# ── file_system ───────────────────────────────────────────────────────────────

def test_read_blocks_path_traversal(tmp_path):
    from kazi.tools.builtin.file_system import read_file_tool

    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")
    try:
        tool = read_file_tool(root_dir=str(tmp_path))
        result = asyncio.run(tool.handler(path="../secret.txt"))
        assert "Access denied" in result
    finally:
        outside.unlink(missing_ok=True)


def test_read_blocks_absolute_escape(tmp_path):
    from kazi.tools.builtin.file_system import read_file_tool

    tool = read_file_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="/etc/passwd"))
    assert "Access denied" in result


def test_read_allows_file_within_root(tmp_path):
    from kazi.tools.builtin.file_system import read_file_tool

    (tmp_path / "hello.txt").write_text("hello world")
    tool = read_file_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="hello.txt"))
    assert result == "hello world"


def test_read_no_root_allows_any_path(tmp_path):
    from kazi.tools.builtin.file_system import read_file_tool

    f = tmp_path / "data.txt"
    f.write_text("data")
    tool = read_file_tool()  # no root_dir — unrestricted
    result = asyncio.run(tool.handler(path=str(f)))
    assert result == "data"


def test_write_blocks_path_traversal(tmp_path):
    from kazi.tools.builtin.file_system import write_file_tool

    tool = write_file_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="../escape.txt", content="pwned"))
    assert "Access denied" in result
    assert not (tmp_path.parent / "escape.txt").exists()


def test_list_blocks_escape(tmp_path):
    from kazi.tools.builtin.file_system import list_directory_tool

    tool = list_directory_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="../"))
    assert "Access denied" in result


# ── database ──────────────────────────────────────────────────────────────────


def test_sql_rejects_drop_table():
    ok, reason = _is_single_select("DROP TABLE users")
    assert not ok
    assert "SELECT" in reason


def test_sql_rejects_insert():
    ok, reason = _is_single_select("INSERT INTO t VALUES (1)")
    assert not ok


def test_sql_rejects_multistatement():
    ok, reason = _is_single_select("SELECT 1; DROP TABLE users")
    assert not ok
    assert "Multiple" in reason


def test_sql_rejects_multistatement_after_comment_strip():
    # After comment stripping: "SELECT 1; DROP TABLE users" — still two statements
    ok, reason = _is_single_select("SELECT 1; DROP TABLE users -- comment")
    assert not ok
    assert "Multiple" in reason


def test_sql_allows_single_select():
    ok, _ = _is_single_select("SELECT id, name FROM users WHERE active = 1")
    assert ok


def test_sql_allows_trailing_semicolon():
    ok, _ = _is_single_select("SELECT id FROM users;")
    assert ok


def test_sql_allows_semicolon_in_string_literal():
    # A semicolon inside a string literal is not a statement separator
    ok, _ = _is_single_select("SELECT 'hello; world' AS greeting")
    assert ok


def test_sql_injection_blocked_end_to_end(tmp_path):
    """Multi-statement SQL is rejected by the actual tool handler, not just the helper."""
    import sqlalchemy as sa

    from kazi.tools.builtin.database import sql_query_tool

    db = f"sqlite:///{tmp_path}/injection_test.db"
    engine = sa.create_engine(db)
    with engine.connect() as conn:
        conn.execute(sa.text("CREATE TABLE secret (data TEXT)"))
        conn.execute(sa.text("INSERT INTO secret VALUES ('sensitive')"))
        conn.commit()

    tool = sql_query_tool(db, read_only=True)
    # This is what an LLM prompt-injection attack looks like
    result = asyncio.run(tool.handler(sql="SELECT 1; DROP TABLE secret"))
    assert "Multiple statements" in result
    # Verify the table was NOT dropped
    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT * FROM secret")).fetchall()
    assert len(rows) == 1


def test_sql_limit_capped(tmp_path):
    """limit=9999 is silently capped at 1000; the truncation message confirms the cap."""
    import sqlalchemy as sa

    from kazi.tools.builtin.database import _MAX_ROWS, sql_query_tool

    db = f"sqlite:///{tmp_path}/cap_test.db"
    engine = sa.create_engine(db)
    with engine.connect() as conn:
        conn.execute(sa.text("CREATE TABLE nums (n INTEGER)"))
        # Insert _MAX_ROWS + 5 rows so we can trigger the truncation message
        for i in range(_MAX_ROWS + 5):
            conn.execute(sa.text(f"INSERT INTO nums VALUES ({i})"))
        conn.commit()

    tool = sql_query_tool(db)
    result = asyncio.run(tool.handler(sql="SELECT n FROM nums", limit=9999))
    # If the cap works, fetchmany(1000) will hit the limit and show the truncation note
    assert f"limited to {_MAX_ROWS} rows" in result


def test_sql_metadata_no_connection_string():
    """connection_string must NOT appear in tool metadata (credential protection)."""
    from kazi.tools.builtin.database import sql_query_tool

    tool = sql_query_tool("postgresql://user:password@host/db")
    assert "connection_string" not in tool.metadata
    assert "password" not in str(tool.metadata)


# ── A2A SSRF guard ────────────────────────────────────────────────────────────


def test_ssrf_blocks_private_ipv4():
    with pytest.raises(A2AConnectionError, match="internal"):
        _validate_agent_url("http://192.168.1.1/agent", [])


def test_ssrf_blocks_loopback():
    with pytest.raises(A2AConnectionError, match="internal"):
        _validate_agent_url("http://127.0.0.1/agent", [])


def test_ssrf_blocks_ipv6_loopback():
    with pytest.raises(A2AConnectionError, match="internal"):
        _validate_agent_url("http://[::1]/agent", [])


def test_ssrf_blocks_file_scheme():
    with pytest.raises(A2AConnectionError, match="scheme"):
        _validate_agent_url("file:///etc/passwd", [])


def test_ssrf_blocks_ftp_scheme():
    with pytest.raises(A2AConnectionError, match="scheme"):
        _validate_agent_url("ftp://example.com/agent", [])


def test_ssrf_allows_public_https():
    _validate_agent_url("https://agents.example.com/my-agent", [])


def test_ssrf_allows_public_http():
    _validate_agent_url("http://agents.example.com/my-agent", [])


def test_allowlist_blocks_non_matching_host():
    with pytest.raises(A2AConnectionError, match="allowlist"):
        _validate_agent_url("https://evil.com/agent", ["trusted.com"])


def test_allowlist_permits_exact_match():
    _validate_agent_url("https://trusted.com/agent", ["trusted.com"])


def test_allowlist_permits_subdomain():
    _validate_agent_url("https://api.trusted.com/agent", ["trusted.com"])


def test_allowlist_empty_allows_public():
    _validate_agent_url("https://any-public-host.io/agent", [])


# ── A2A skill cap (registry bloat DoS) ───────────────────────────────────────

def test_skill_cap_limits_registration():
    """A malicious agent card with 100 skills only registers _MAX_SKILLS_PER_AGENT."""
    from kazi.agents.a2a_client import _MAX_SKILLS_PER_AGENT, A2ABridge
    from kazi.agents.agent_card import AgentCard, AgentSkill
    from kazi.core.config import A2AConfig
    from kazi.core.registry import ToolRegistry

    oversized_card = AgentCard(
        name="bloat-agent",
        description="A malicious agent",
        url="https://evil.com",
        skills=[
            AgentSkill(name=f"skill_{i}", description=f"Skill {i}")
            for i in range(100)
        ],
    )

    bridge = A2ABridge(A2AConfig(), ToolRegistry())
    bridge._register_skills(oversized_card)

    registered = bridge.registry.list_tools()
    assert len(registered) == _MAX_SKILLS_PER_AGENT


# ── web search max_results cap ────────────────────────────────────────────────

def test_web_search_max_results_is_capped():
    """The cap is applied before calling DDG — tested by mocking the search."""
    from unittest.mock import MagicMock, patch

    from kazi.tools.builtin.web_search import _MAX_RESULTS, _duckduckgo_search

    captured = {}

    class FakeDDGS:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def text(self, query, max_results):
            captured["max_results"] = max_results
            return []

    with patch("kazi.tools.builtin.web_search.DDGS", FakeDDGS, create=True):
        import sys
        # Ensure the import inside the function resolves to our mock
        with patch.dict(sys.modules, {"duckduckgo_search": MagicMock(DDGS=FakeDDGS)}):
            asyncio.run(_duckduckgo_search("test query", max_results=9999))

    assert captured.get("max_results", 9999) <= _MAX_RESULTS
