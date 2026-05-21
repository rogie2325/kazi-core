"""
Custom document parsers for non-standard formats.

Each parser returns a list of dicts with 'text' and optional 'metadata'.
"""
from __future__ import annotations

import json
from typing import Any


def parse_jsonl(path: str, text_field: str = "text") -> list[dict]:
    """Parse a JSONL file, extracting the named text field from each line."""
    docs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_field, "")
                if text:
                    meta = {k: v for k, v in obj.items() if k != text_field}
                    docs.append({"text": text, "metadata": {"line": i, **meta}})
            except json.JSONDecodeError:
                continue
    return docs


def parse_csv(
    path: str,
    text_columns: list[str],
    delimiter: str = ",",
) -> list[dict]:
    """Parse a CSV file, joining the specified columns into a single text field."""
    import csv

    docs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            parts = [str(row[col]) for col in text_columns if col in row]
            text = " | ".join(parts)
            if text.strip():
                meta = {k: v for k, v in row.items() if k not in text_columns}
                docs.append({"text": text, "metadata": {"row": i, **meta}})
    return docs


def parse_notion_export(path: str) -> list[dict]:
    """Parse a Notion markdown export directory."""
    from pathlib import Path

    docs = []
    for md_file in Path(path).rglob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        if text.strip():
            docs.append({
                "text": text,
                "metadata": {
                    "source": str(md_file),
                    "filename": md_file.name,
                },
            })
    return docs


def parse_slack_export(path: str) -> list[dict]:
    """Parse a Slack JSON export directory (one JSON file per channel per day)."""
    import json
    from pathlib import Path

    docs = []
    for json_file in Path(path).rglob("*.json"):
        try:
            messages = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        for msg in messages:
            text = msg.get("text", "").strip()
            if text and msg.get("type") == "message":
                docs.append({
                    "text": text,
                    "metadata": {
                        "channel": json_file.parent.name,
                        "ts": msg.get("ts"),
                        "user": msg.get("user"),
                    },
                })
    return docs
