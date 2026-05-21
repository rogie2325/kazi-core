"""Built-in database tool — read-only SQL queries via SQLAlchemy."""
from __future__ import annotations

import functools
import re

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource

# Simplified SQL lexer patterns for structural analysis (not execution)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_STRING_LITERAL_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", re.DOTALL)

_MAX_ROWS = 1_000


@functools.lru_cache(maxsize=16)
def _get_engine(connection_string: str):
    """
    Return a cached SQLAlchemy engine for the given connection string.

    Engines are long-lived objects designed to be shared — creating one per
    query exhausts the connection pool. The cache is keyed by connection
    string; up to 16 distinct databases are cached simultaneously.

    pool_pre_ping=True validates connections before use, recovering silently
    from stale connections after database restarts or firewall timeouts.
    """
    import sqlalchemy as sa

    kwargs: dict = {"pool_pre_ping": True}
    if not connection_string.startswith("sqlite"):
        # SQLite does not support pool_size / max_overflow
        kwargs.update({"pool_size": 5, "max_overflow": 10})
    return sa.create_engine(connection_string, **kwargs)


def _strip_sql(sql: str) -> str:
    """Remove comments and string literals for safe structural analysis."""
    sql = _BLOCK_COMMENT_RE.sub(" ", sql)
    sql = _LINE_COMMENT_RE.sub(" ", sql)
    sql = _STRING_LITERAL_RE.sub("''", sql)
    return sql


# Patterns that are syntactically part of a SELECT but perform writes or
# filesystem access on MySQL/MariaDB.  Checked after stripping comments and
# string literals so obfuscation like SELECT/**/ doesn't bypass them.
_WRITE_CLAUSE_RE = re.compile(
    r"\bINTO\s+(OUTFILE|DUMPFILE)\b",
    re.IGNORECASE,
)


def _is_single_select(sql: str) -> tuple[bool, str]:
    """
    Return (True, "") if sql is a single SELECT statement.
    Return (False, reason) if it is not.

    Blocks:
      - Non-SELECT statements (INSERT, UPDATE, DROP, etc.)
      - Multi-statement injection: SELECT 1; DROP TABLE users
      - MySQL file-write clauses: SELECT … INTO OUTFILE / INTO DUMPFILE
    """
    stripped = _strip_sql(sql).strip()
    upper = stripped.upper()
    if not upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed in read-only mode."
    # A trailing semicolon is fine; a semicolon followed by more tokens is not
    without_trailing = stripped.rstrip(";").rstrip()
    if ";" in without_trailing:
        return False, "Multiple statements are not allowed — only a single SELECT query."
    # Block MySQL file-write clauses that masquerade as SELECT statements
    if _WRITE_CLAUSE_RE.search(stripped):
        return False, "SELECT INTO OUTFILE / DUMPFILE is not allowed in read-only mode."
    return True, ""


def sql_query_tool(connection_string: str, read_only: bool = True) -> ToolDefinition:
    """
    Return a ToolDefinition that runs SQL queries against a database.

    connection_string: SQLAlchemy URL e.g. "sqlite:///data.db"
    read_only: if True, rejects non-SELECT and multi-statement SQL (default True)

    Security notes:
      - In read_only mode, multi-statement injection (SELECT 1; DROP ...) is blocked.
      - The connection_string is NOT stored in tool metadata to avoid credential exposure.
      - Rows returned are capped at 1,000 regardless of the limit argument.
    """

    async def _run_query(sql: str, limit: int = 100) -> str:
        import asyncio

        import sqlalchemy as sa

        limit = max(1, min(limit, _MAX_ROWS))

        if read_only:
            ok, reason = _is_single_select(sql)
            if not ok:
                return reason

        def _execute() -> str:
            engine = _get_engine(connection_string)
            with engine.connect() as conn:
                try:
                    result = conn.execute(sa.text(sql))
                    rows = result.fetchmany(limit)
                    cols = list(result.keys())
                    if not rows:
                        return "Query returned no rows."
                    lines = [" | ".join(cols)]
                    lines.append("-" * len(lines[0]))
                    for row in rows:
                        lines.append(" | ".join(str(v) for v in row))
                    if len(rows) == limit:
                        lines.append(f"... (limited to {limit} rows)")
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Query error: {exc}"

        return await asyncio.to_thread(_execute)

    return ToolDefinition(
        name="sql_query",
        description="Run a SQL query against the connected database and return results.",
        parameters=[
            ToolParameter(
                name="sql",
                type="string",
                description="SQL statement to execute",
                required=True,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description=f"Max rows to return (default 100, hard cap {_MAX_ROWS})",
                required=False,
                default=100,
            ),
        ],
        source=ToolSource.NATIVE,
        handler=_run_query,
        # connection_string intentionally omitted — may contain credentials
        metadata={"read_only": read_only},
    )
