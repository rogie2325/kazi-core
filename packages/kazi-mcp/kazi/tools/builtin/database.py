"""Built-in database tool — read-only SQL queries via SQLAlchemy."""
from __future__ import annotations

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource


def sql_query_tool(connection_string: str, read_only: bool = True) -> ToolDefinition:
    """
    Return a ToolDefinition that runs SQL SELECT queries against a database.

    connection_string: SQLAlchemy URL e.g. "sqlite:///data.db"
    read_only: if True, rejects any non-SELECT statement (default True)
    """

    async def _run_query(sql: str, limit: int = 100) -> str:
        import sqlalchemy as sa

        if read_only and not sql.strip().upper().startswith("SELECT"):
            return "Only SELECT queries are allowed in read-only mode."

        engine = sa.create_engine(connection_string)
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

    return ToolDefinition(
        name="sql_query",
        description="Run a SQL query against the connected database and return results.",
        parameters=[
            ToolParameter(name="sql", type="string", description="SQL statement to execute", required=True),
            ToolParameter(name="limit", type="integer", description="Max rows to return (default 100)", required=False, default=100),
        ],
        source=ToolSource.NATIVE,
        handler=_run_query,
        metadata={"connection_string": connection_string, "read_only": read_only},
    )
