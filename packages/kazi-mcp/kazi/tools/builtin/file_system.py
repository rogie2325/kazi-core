"""Built-in filesystem tools (read / write / list)."""
from __future__ import annotations

import os
from pathlib import Path

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource


async def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    if not p.is_file():
        return f"Not a file: {path}"
    return p.read_text(encoding="utf-8", errors="replace")


async def _write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written {len(content)} chars to {path}"


async def _list_directory(path: str = ".") -> str:
    p = Path(path)
    if not p.exists():
        return f"Directory not found: {path}"
    entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
    lines = []
    for e in entries:
        marker = "" if e.is_dir() else "  "
        size = f" ({e.stat().st_size} bytes)" if e.is_file() else "/"
        lines.append(f"{marker}{e.name}{size}")
    return "\n".join(lines) if lines else "(empty)"


def read_file_tool() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read the contents of a file from the filesystem.",
        parameters=[ToolParameter(name="path", type="string", description="File path to read", required=True)],
        source=ToolSource.NATIVE,
        handler=_read_file,
    )


def write_file_tool() -> ToolDefinition:
    return ToolDefinition(
        name="write_file",
        description="Write content to a file on the filesystem.",
        parameters=[
            ToolParameter(name="path", type="string", description="File path to write", required=True),
            ToolParameter(name="content", type="string", description="Content to write", required=True),
        ],
        source=ToolSource.NATIVE,
        handler=_write_file,
    )


def list_directory_tool() -> ToolDefinition:
    return ToolDefinition(
        name="list_directory",
        description="List files and directories at a given path.",
        parameters=[ToolParameter(name="path", type="string", description="Directory path (default: '.')", required=False, default=".")],
        source=ToolSource.NATIVE,
        handler=_list_directory,
    )
