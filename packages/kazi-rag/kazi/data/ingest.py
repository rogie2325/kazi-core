"""
High-level ingestion helpers that wrap IndexManager for common sources.
"""
from __future__ import annotations

from typing import Optional

from kazi.data.index_manager import IndexManager


async def ingest_pdf(
    manager: IndexManager,
    path: str,
    index_name: str = "default",
) -> None:
    """Ingest one or more PDF files into a named index."""
    await manager.ingest_directory(path, index_name=index_name, file_extensions=[".pdf"])


async def ingest_text_files(
    manager: IndexManager,
    path: str,
    index_name: str = "default",
    extensions: Optional[list[str]] = None,
) -> None:
    exts = extensions or [".txt", ".md", ".rst"]
    await manager.ingest_directory(path, index_name=index_name, file_extensions=exts)


async def ingest_web_pages(
    manager: IndexManager,
    urls: list[str],
    index_name: str = "default",
) -> None:
    """Fetch web pages and ingest their text content."""
    import httpx

    docs = []
    async with httpx.AsyncClient(timeout=30) as client:
        for url in urls:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                # Strip HTML tags naively; production code should use BeautifulSoup
                import re
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()
                docs.append({"text": text, "metadata": {"source": url}})
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Failed to fetch %s: %s", url, exc)

    if docs:
        await manager.ingest_documents(docs, index_name=index_name)


async def ingest_strings(
    manager: IndexManager,
    texts: list[str],
    index_name: str = "default",
    metadata: Optional[list[dict]] = None,
) -> None:
    """Ingest plain strings directly (useful for testing)."""
    meta = metadata or [{}] * len(texts)
    docs = [{"text": t, "metadata": m} for t, m in zip(texts, meta)]
    await manager.ingest_documents(docs, index_name=index_name)
