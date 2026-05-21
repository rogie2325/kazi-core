"""
Covers uncovered lines in the built-in tools:
  file_system: write success, list success, not-a-file, directory-not-found, empty dir
  database:    empty result set, query error, non-sqlite engine kwargs
  web_search:  missing package message, no results, result formatting
"""
import asyncio

import pytest

# ── file_system — write_file_tool ─────────────────────────────────────────────

def test_write_creates_file_within_root(tmp_path):
    from kazi.tools.builtin.file_system import write_file_tool

    tool = write_file_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="output.txt", content="hello world"))

    assert "Written" in result
    assert "11 chars" in result
    assert (tmp_path / "output.txt").read_text() == "hello world"


def test_write_creates_subdirectory_if_needed(tmp_path):
    from kazi.tools.builtin.file_system import write_file_tool

    tool = write_file_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="subdir/nested/file.txt", content="data"))

    assert "Written" in result
    assert (tmp_path / "subdir" / "nested" / "file.txt").exists()


def test_write_overwrites_existing_file(tmp_path):
    from kazi.tools.builtin.file_system import write_file_tool

    (tmp_path / "existing.txt").write_text("old content")
    tool = write_file_tool(root_dir=str(tmp_path))
    asyncio.run(tool.handler(path="existing.txt", content="new content"))

    assert (tmp_path / "existing.txt").read_text() == "new content"


def test_write_no_root_dir_writes_to_path(tmp_path):
    from kazi.tools.builtin.file_system import write_file_tool

    target = tmp_path / "free_write.txt"
    tool = write_file_tool()  # no root_dir
    result = asyncio.run(tool.handler(path=str(target), content="unrestricted"))

    assert "Written" in result
    assert target.read_text() == "unrestricted"


# ── file_system — read_file_tool (gap coverage) ───────────────────────────────

def test_read_returns_not_a_file_for_directory(tmp_path):
    from kazi.tools.builtin.file_system import read_file_tool

    subdir = tmp_path / "mydir"
    subdir.mkdir()
    tool = read_file_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="mydir"))

    assert "Not a file" in result


def test_read_returns_file_not_found(tmp_path):
    from kazi.tools.builtin.file_system import read_file_tool

    tool = read_file_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="ghost.txt"))

    assert "not found" in result.lower()


# ── file_system — list_directory_tool ─────────────────────────────────────────

def test_list_returns_entries(tmp_path):
    from kazi.tools.builtin.file_system import list_directory_tool

    (tmp_path / "file1.txt").write_text("a")
    (tmp_path / "file2.txt").write_text("bb")
    (tmp_path / "subdir").mkdir()

    tool = list_directory_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="."))

    assert "file1.txt" in result
    assert "file2.txt" in result
    assert "subdir" in result


def test_list_shows_file_sizes(tmp_path):
    from kazi.tools.builtin.file_system import list_directory_tool

    (tmp_path / "data.txt").write_text("hello")  # 5 bytes
    tool = list_directory_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="."))

    assert "5 bytes" in result


def test_list_empty_directory_returns_empty_marker(tmp_path):
    from kazi.tools.builtin.file_system import list_directory_tool

    empty = tmp_path / "empty_dir"
    empty.mkdir()
    tool = list_directory_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="empty_dir"))

    assert result == "(empty)"


def test_list_nonexistent_directory(tmp_path):
    from kazi.tools.builtin.file_system import list_directory_tool

    tool = list_directory_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="ghost_dir"))

    assert "not found" in result.lower()


def test_list_default_path_uses_dot(tmp_path):
    from kazi.tools.builtin.file_system import list_directory_tool

    (tmp_path / "readme.txt").write_text("hi")
    # root is tmp_path; default path "." resolves to tmp_path
    tool = list_directory_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="."))
    assert "readme.txt" in result


def test_list_shows_trailing_slash_for_dirs(tmp_path):
    from kazi.tools.builtin.file_system import list_directory_tool

    (tmp_path / "mysubdir").mkdir()
    tool = list_directory_tool(root_dir=str(tmp_path))
    result = asyncio.run(tool.handler(path="."))

    assert "mysubdir/" in result


# ── database — empty result and query error ───────────────────────────────────

def test_sql_returns_no_rows_message(tmp_path):
    import sqlalchemy as sa

    from kazi.tools.builtin.database import sql_query_tool

    db = f"sqlite:///{tmp_path}/empty.db"
    engine = sa.create_engine(db)
    with engine.connect() as conn:
        conn.execute(sa.text("CREATE TABLE items (id INTEGER)"))
        conn.commit()

    tool = sql_query_tool(db)
    result = asyncio.run(tool.handler(sql="SELECT * FROM items"))
    assert "no rows" in result.lower()


def test_sql_returns_query_error_on_bad_sql(tmp_path):
    from kazi.tools.builtin.database import sql_query_tool

    db = f"sqlite:///{tmp_path}/err.db"
    tool = sql_query_tool(db)
    result = asyncio.run(tool.handler(sql="SELECT * FROM nonexistent_table_xyz"))
    assert "error" in result.lower()


def test_sql_read_only_false_allows_non_select(tmp_path):
    """read_only=False should allow INSERT and not block it."""
    import sqlalchemy as sa

    from kazi.tools.builtin.database import sql_query_tool

    db = f"sqlite:///{tmp_path}/rw.db"
    engine = sa.create_engine(db)
    with engine.connect() as conn:
        conn.execute(sa.text("CREATE TABLE t (x INTEGER)"))
        conn.commit()

    tool = sql_query_tool(db, read_only=False)
    # INSERT should not be rejected (returns a query error since INSERT returns no rows, but not blocked)
    result = asyncio.run(tool.handler(sql="INSERT INTO t VALUES (1)"))
    # The query error or "no rows" is expected — key thing is it wasn't blocked by the SELECT filter
    assert "Only SELECT" not in result


def test_sql_engine_cached_for_same_connection_string(tmp_path):
    """_get_engine must return the same engine object for the same URL (cache hit)."""
    from kazi.tools.builtin.database import _get_engine

    db = f"sqlite:///{tmp_path}/cache_test.db"
    e1 = _get_engine(db)
    e2 = _get_engine(db)
    assert e1 is e2


# ── web_search — missing package and no results ───────────────────────────────

@pytest.mark.asyncio
async def test_web_search_missing_package_returns_install_message(monkeypatch):
    """When duckduckgo-search is not installed, return a helpful install message."""
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"duckduckgo_search": None}):
        import importlib

        from kazi.tools.builtin import web_search
        # Force reimport so the try/except block runs fresh
        importlib.reload(web_search)

        # Patch the import inside _duckduckgo_search
        import builtins
        real_import = builtins.__import__

        def block_ddg(name, *args, **kwargs):
            if name == "duckduckgo_search":
                raise ImportError("No module named 'duckduckgo_search'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", block_ddg):
            result = await web_search._duckduckgo_search("test query")

    assert "pip install" in result or "not installed" in result


@pytest.mark.asyncio
async def test_web_search_no_results_message():
    """When DDG returns zero results, return 'No results found.'"""
    from unittest.mock import MagicMock, patch

    from kazi.tools.builtin.web_search import _duckduckgo_search

    class EmptyDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, query, max_results): return []

    with patch("kazi.tools.builtin.web_search.DDGS", EmptyDDGS, create=True):
        import builtins
        real_import = builtins.__import__

        def inject_ddg(name, *args, **kwargs):
            if name == "duckduckgo_search":
                mod = MagicMock()
                mod.DDGS = EmptyDDGS
                return mod
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", inject_ddg):
            result = await _duckduckgo_search("obscure query with zero results")

    assert "No results" in result


@pytest.mark.asyncio
async def test_web_search_formats_results():
    """Each result should show title, URL, and body."""
    from unittest.mock import MagicMock, patch

    from kazi.tools.builtin.web_search import _duckduckgo_search

    fake_results = [
        {"title": "Kazi AI", "href": "https://kazi.ai", "body": "A great framework"},
        {"title": "Docs", "href": "https://docs.kazi.ai", "body": "Full documentation"},
    ]

    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, query, max_results): return fake_results

    import builtins
    real_import = builtins.__import__

    def inject_ddg(name, *args, **kwargs):
        if name == "duckduckgo_search":
            mod = MagicMock()
            mod.DDGS = FakeDDGS
            return mod
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", inject_ddg):
        result = await _duckduckgo_search("kazi ai")

    assert "Kazi AI" in result
    assert "https://kazi.ai" in result
    assert "A great framework" in result
    assert "Docs" in result


@pytest.mark.asyncio
async def test_web_search_enforces_min_results():
    """max_results=0 should be clamped to 1."""
    from unittest.mock import MagicMock, patch

    from kazi.tools.builtin.web_search import _duckduckgo_search

    received = {}

    class CaptureDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, query, max_results):
            received["max_results"] = max_results
            return []

    import builtins
    real_import = builtins.__import__

    def inject(name, *args, **kwargs):
        if name == "duckduckgo_search":
            mod = MagicMock()
            mod.DDGS = CaptureDDGS
            return mod
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", inject):
        await _duckduckgo_search("query", max_results=0)

    assert received["max_results"] == 1
