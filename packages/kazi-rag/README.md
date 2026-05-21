# kazi-rag

LlamaIndex RAG integration for kazi. Handles document ingestion, vector indexing, and retrieval-augmented generation.

## Install

```bash
pip install kazi-rag[openai]
pip install kazi-rag[anthropic]
pip install kazi-rag[rerank]   # adds Cohere reranking
```

## What's included

- Document ingestion with automatic chunking and embedding
- Vector store backends: in-memory, Chroma, Pinecone, Qdrant, Weaviate
- Hybrid search: vector similarity + BM25 keyword retrieval
- Optional Cohere reranking

## Usage

This package is used automatically when you call `kazi.ingest()` via `kazi-core`. To use the RAG layer directly:

```python
from kazi import KaziConfig, LLMConfig, LLMProvider, RAGConfig, VectorStoreBackend

config = KaziConfig(
    llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"),
    rag=RAGConfig(
        vector_store=VectorStoreBackend.CHROMA,
        persist_dir="./kazi_index",
    ),
)
```

## License

MIT
