"""Built-in filesystem tools (read / write / list)."""
from __future__ import annotations

from pathlib import Path

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource

_ACCESS_DENIED = "Access denied: path is outside the allowed directory."


def _safe_resolve(path: str, root: Path | None) -> tuple[Path, str | None]:
    """
    Resolve `path`, optionally confining it to `root`.
    Returns (resolved_path, error_message_or_None).

    Uses Path.resolve() + is_relative_to() so that symlinks and ``..``
    components cannot escape the root, regardless of how the path is spelled.
    """
    if root is None:
        return Path(path), None
    try:
        resolved = (root / path).resolve()
    except (OSError, ValueError) as exc:
        return Path(path), f"Invalid path: {exc}"
    if not resolved.is_relative_to(root):
        return resolved, _ACCESS_DENIED
    return resolved, None


def read_file_tool(root_dir: str | None = None) -> ToolDefinition:
    """
    Return a ToolDefinition for reading files.

    root_dir
        When set, all paths are confined to this directory tree. ``..``
        traversal and absolute paths that escape it are blocked with an
        "Access denied" error. Strongly recommended for any deployment
        where the LLM can choose which file to read.

        Omit only in fully trusted, single-user environments.
    """
    _root = Path(root_dir).resolve() if root_dir else None

    async def _read(path: str) -> str:
        p, err = _safe_resolve(path, _root)
        if err:
            return err
        if not p.exists():
            return f"File not found: {path}"
        if not p.is_file():
            return f"Not a file: {path}"
        return p.read_text(encoding="utf-8", errors="replace")

    return ToolDefinition(
        name="read_file",
        description="Read the contents of a file from the filesystem.",
        parameters=[
            ToolParameter(name="path", type="string", description="File path to read", required=True),
        ],
        source=ToolSource.NATIVE,
        handler=_read,
        metadata={"root_dir": root_dir},
    )


_MAX_WRITE_BYTES = 10 * 1024 * 1024  # 10 MB hard cap per write


def write_file_tool(root_dir: str | None = None, max_bytes: int = _MAX_WRITE_BYTES) -> ToolDefinition:
    """
    Return a ToolDefinition for writing files.

    root_dir
        When set, all write paths are confined to this directory tree.
        Strongly recommended — without it the LLM can overwrite any file
        that the process has permission to write.
    max_bytes
        Maximum content size per write (default 10 MB).  Prevents a runaway
        LLM from filling the disk with a single tool call.
    """
    _root = Path(root_dir).resolve() if root_dir else None

    async def _write(path: str, content: str) -> str:
        encoded = content.encode("utf-8")
        if len(encoded) > max_bytes:
            return (
                f"Write rejected: content is {len(encoded):,} bytes "
                f"(max {max_bytes // (1024 * 1024)} MB)."
            )
        p, err = _safe_resolve(path, _root)
        if err:
            return err
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"

    return ToolDefinition(
        name="write_file",
        description=f"Write content to a file on the filesystem (max {max_bytes // (1024*1024)} MB).",
        parameters=[
            ToolParameter(name="path", type="string", description="File path to write", required=True),
            ToolParameter(name="content", type="string", description="Content to write", required=True),
        ],
        source=ToolSource.NATIVE,
        handler=_write,
        metadata={"root_dir": root_dir, "max_bytes": max_bytes},
    )


def list_directory_tool(root_dir: str | None = None) -> ToolDefinition:
    """
    Return a ToolDefinition for listing directory contents.

    root_dir
        When set, listings are confined to this directory tree.
    """
    _root = Path(root_dir).resolve() if root_dir else None

    async def _list(path: str = ".") -> str:
        p, err = _safe_resolve(path, _root)
        if err:
            return err
        if not p.exists():
            return f"Directory not found: {path}"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for e in entries:
            marker = "" if e.is_dir() else "  "
            size = f" ({e.stat().st_size} bytes)" if e.is_file() else "/"
            lines.append(f"{marker}{e.name}{size}")
        return "\n".join(lines) if lines else "(empty)"

    return ToolDefinition(
        name="list_directory",
        description="List files and directories at a given path.",
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Directory path (default: '.')",
                required=False,
                default=".",
            ),
        ],
        source=ToolSource.NATIVE,
        handler=_list,
        metadata={"root_dir": root_dir},
    )
