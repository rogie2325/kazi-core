"""
Query engine helpers — convenience wrappers over IndexManager.get_query_engine().
"""
from __future__ import annotations

from typing import Optional

from kazi.data.index_manager import IndexManager


async def query(
    manager: IndexManager,
    question: str,
    index_name: str = "default",
    top_k: Optional[int] = None,
) -> str:
    """Run a single RAG query and return the response as a string."""
    engine = manager.get_query_engine(index_name, similarity_top_k=top_k)
    response = engine.query(question)
    return str(response)


async def query_with_sources(
    manager: IndexManager,
    question: str,
    index_name: str = "default",
    top_k: Optional[int] = None,
) -> dict:
    """
    Run a RAG query and return both the answer and source node metadata.

    Returns: {"answer": str, "sources": [{"text": ..., "metadata": ...}]}
    """
    engine = manager.get_query_engine(index_name, similarity_top_k=top_k)
    response = engine.query(question)

    sources = []
    for node in getattr(response, "source_nodes", []):
        sources.append({
            "text": node.node.get_content()[:500],
            "metadata": node.node.metadata,
            "score": float(getattr(node, "score", 0.0)),
        })

    return {"answer": str(response), "sources": sources}
