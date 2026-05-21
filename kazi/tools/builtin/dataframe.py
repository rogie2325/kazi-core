"""
Built-in dataframe tools — query CSV, Excel, Parquet, and JSON files.

Supports pandas and Polars interchangeably.  Polars is used when installed
(faster, lower memory), falls back to pandas automatically.

Tools exposed:
  data_query_tool   — load a file and filter/query rows
  data_summary_tool — load a file and return schema + descriptive stats
"""
from __future__ import annotations

import logging
from pathlib import Path

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource

logger = logging.getLogger(__name__)

_SUPPORTED = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}
_MAX_ROWS = 5_000   # hard cap — LLM context budget
_MAX_COLS = 50      # columns beyond this are dropped with a warning


# ── Engine detection ──────────────────────────────────────────────────────────

def _engine(preferred: str = "auto") -> str:
    if preferred == "polars":
        try:
            import polars  # noqa: F401
            return "polars"
        except ImportError:
            raise ImportError("polars not installed. Run: pip install polars")
    if preferred == "pandas":
        try:
            import pandas  # noqa: F401
            return "pandas"
        except ImportError:
            raise ImportError("pandas not installed. Run: pip install pandas")
    # auto: prefer polars for speed, fall back to pandas
    try:
        import polars  # noqa: F401
        return "polars"
    except ImportError:
        pass
    try:
        import pandas  # noqa: F401
        return "pandas"
    except ImportError:
        raise ImportError(
            "Neither polars nor pandas is installed. "
            "Run: pip install kazi-core[data]"
        )


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_pandas(path: Path, limit: int):
    import pandas as pd
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep, nrows=limit)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=limit)
    if ext == ".parquet":
        df = pd.read_parquet(path)
        return df.head(limit)
    if ext in (".json", ".jsonl"):
        return pd.read_json(path, lines=(ext == ".jsonl"), nrows=limit)
    raise ValueError(f"Unsupported file type: {ext}")


def _load_polars(path: Path, limit: int):
    import polars as pl
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        return pl.read_csv(path, separator=sep, n_rows=limit)
    if ext in (".xlsx", ".xls"):
        return pl.read_excel(path, read_csv_options={"n_rows": limit})
    if ext == ".parquet":
        return pl.read_parquet(path, n_rows=limit)
    if ext in (".json", ".jsonl"):
        return pl.read_json(path)[:limit]
    raise ValueError(f"Unsupported file type: {ext}")


def _load(path: Path, engine: str, limit: int):
    if engine == "polars":
        return _load_polars(path, limit)
    return _load_pandas(path, limit)


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt(df, engine: str, limit: int) -> str:
    """Convert a dataframe to a compact string for LLM consumption."""
    if engine == "polars":
        n_rows, n_cols = df.shape
        if n_cols > _MAX_COLS:
            df = df[:, :_MAX_COLS]
        if n_rows > limit:
            df = df.head(limit)
        lines = [f"Shape: {n_rows} rows × {n_cols} cols"]
        if n_cols > _MAX_COLS:
            lines.append(f"(showing first {_MAX_COLS} of {n_cols} columns)")
        lines.append(df.to_pandas().to_string(index=False, max_rows=limit))
    else:
        n_rows, n_cols = df.shape
        if n_cols > _MAX_COLS:
            df = df.iloc[:, :_MAX_COLS]
        if n_rows > limit:
            df = df.head(limit)
        lines = [f"Shape: {n_rows} rows × {n_cols} cols"]
        if n_cols > _MAX_COLS:
            lines.append(f"(showing first {_MAX_COLS} of {n_cols} columns)")
        lines.append(df.to_string(index=False, max_rows=limit))
    if n_rows > limit:
        lines.append(f"... (limited to {limit} rows)")
    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────────────────────

def data_query_tool(
    root_dir: str | None = None,
    engine: str = "auto",
    max_rows: int = 500,
) -> ToolDefinition:
    """
    Return a ToolDefinition that loads a tabular file and filters rows.

    root_dir    Confine file access to this directory (path traversal blocked).
    engine      "auto" | "pandas" | "polars"
    max_rows    Cap on rows returned to the LLM (default 500).

    The query parameter accepts pandas .query() syntax:
      "amount > 1000 and category == 'food'"
      "status == 'pending'"
    Leave query empty to return the first max_rows rows.
    """
    _root = Path(root_dir).resolve() if root_dir else None
    _eng = _engine(engine)
    _limit = min(max_rows, _MAX_ROWS)

    async def _run(file: str, query: str = "", columns: str = "") -> str:
        path = Path(file)
        if _root:
            try:
                path = (_root / file).resolve()
            except (OSError, ValueError) as exc:
                return f"Invalid path: {exc}"
            if not path.is_relative_to(_root):
                return "Access denied: path is outside the allowed directory."

        if not path.exists():
            return f"File not found: {file}"
        if path.suffix.lower() not in _SUPPORTED:
            return f"Unsupported file type: {path.suffix}. Supported: {', '.join(_SUPPORTED)}"

        try:
            df = _load(path, _eng, _limit * 2)  # load extra for post-filter
        except Exception as exc:
            return f"Failed to load {file}: {exc}"

        # Column selection
        if columns:
            col_list = [c.strip() for c in columns.split(",") if c.strip()]
            try:
                if _eng == "polars":
                    df = df.select([c for c in col_list if c in df.columns])
                else:
                    df = df[[c for c in col_list if c in df.columns]]
            except Exception as exc:
                return f"Column selection error: {exc}"

        # Row filter
        if query:
            try:
                if _eng == "polars":
                    import polars as pl
                    df = df.filter(pl.Expr.deserialize(query.encode(), format="json")) if False else df.filter(query) if hasattr(df, "sql_filter") else df.to_pandas().query(query)
                    # Simplification: convert to pandas for .query() compatibility
                    if hasattr(df, "to_pandas"):
                        df_pd = df.to_pandas()
                        df_pd = df_pd.query(query)
                        return _fmt_pandas(df_pd, _limit)
                else:
                    df = df.query(query)
            except Exception as exc:
                return f"Query error: {exc}. Use pandas .query() syntax (e.g. \"amount > 100 and status == 'paid'\")."

        return _fmt(df, _eng, _limit)

    return ToolDefinition(
        name="data_query",
        description=(
            "Load a CSV, Excel, Parquet, or JSON file and filter rows using pandas query syntax. "
            "Use for expense reports, sales data, user exports, or any tabular dataset."
        ),
        parameters=[
            ToolParameter(name="file", type="string", description="File path to load", required=True),
            ToolParameter(
                name="query", type="string",
                description="pandas .query() filter e.g. \"amount > 1000 and status == 'pending'\". Empty = all rows.",
                required=False, default="",
            ),
            ToolParameter(
                name="columns", type="string",
                description="Comma-separated column names to include. Empty = all columns.",
                required=False, default="",
            ),
        ],
        source=ToolSource.NATIVE,
        handler=_run,
        metadata={"engine": _eng, "root_dir": root_dir, "max_rows": _limit},
    )


def data_summary_tool(
    root_dir: str | None = None,
    engine: str = "auto",
) -> ToolDefinition:
    """
    Return a ToolDefinition that loads a file and returns schema + descriptive stats.

    Useful for letting the agent understand the shape of a dataset before querying it.
    """
    _root = Path(root_dir).resolve() if root_dir else None
    _eng = _engine(engine)

    async def _run(file: str) -> str:
        path = Path(file)
        if _root:
            try:
                path = (_root / file).resolve()
            except (OSError, ValueError) as exc:
                return f"Invalid path: {exc}"
            if not path.is_relative_to(_root):
                return "Access denied: path is outside the allowed directory."

        if not path.exists():
            return f"File not found: {file}"
        if path.suffix.lower() not in _SUPPORTED:
            return f"Unsupported file type: {path.suffix}"

        try:
            df = _load(path, _eng, _MAX_ROWS)
        except Exception as exc:
            return f"Failed to load {file}: {exc}"

        lines = [f"File: {path.name}"]

        if _eng == "polars":
            n_rows, n_cols = df.shape
            lines.append(f"Shape: {n_rows:,} rows × {n_cols} columns")
            lines.append("\nColumn types:")
            for col, dtype in zip(df.columns, df.dtypes):
                lines.append(f"  {col}: {dtype}")
            lines.append("\nDescriptive stats:")
            try:
                lines.append(df.describe().to_pandas().to_string())
            except Exception:
                pass
            null_counts = df.null_count()
            if null_counts.sum_horizontal().to_list()[0] > 0:
                lines.append("\nNull counts:")
                for col in df.columns:
                    n = df[col].null_count()
                    if n > 0:
                        lines.append(f"  {col}: {n}")
        else:
            n_rows, n_cols = df.shape
            lines.append(f"Shape: {n_rows:,} rows × {n_cols} columns")
            lines.append("\nColumn types:")
            for col, dtype in df.dtypes.items():
                lines.append(f"  {col}: {dtype}")
            lines.append("\nDescriptive stats:")
            try:
                lines.append(df.describe(include="all").to_string())
            except Exception:
                pass
            nulls = df.isnull().sum()
            if nulls.any():
                lines.append("\nNull counts:")
                for col, n in nulls[nulls > 0].items():
                    lines.append(f"  {col}: {n}")

        return "\n".join(lines)

    return ToolDefinition(
        name="data_summary",
        description=(
            "Load a tabular file and return its schema, column types, and descriptive statistics. "
            "Call this first to understand a dataset before querying it."
        ),
        parameters=[
            ToolParameter(name="file", type="string", description="File path to summarise", required=True),
        ],
        source=ToolSource.NATIVE,
        handler=_run,
        metadata={"engine": _eng, "root_dir": root_dir},
    )


def _fmt_pandas(df, limit: int) -> str:
    n_rows, n_cols = df.shape
    if n_cols > _MAX_COLS:
        df = df.iloc[:, :_MAX_COLS]
    if n_rows > limit:
        df = df.head(limit)
    lines = [f"Shape: {n_rows} rows × {n_cols} cols"]
    lines.append(df.to_string(index=False, max_rows=limit))
    if n_rows > limit:
        lines.append(f"... (limited to {limit} rows)")
    return "\n".join(lines)
