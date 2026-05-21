from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from kazi.core.config import LLMConfig, RAGConfig
from kazi.core.exceptions import IndexNotFoundError
from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manages LlamaIndex vector indices as the RAG backbone.

    Handles ingestion from directories or structured dicts, persists
    indices to disk, and wraps them as ToolDefinitions for the registry.
    """

    def __init__(self, rag_config: RAGConfig, llm_config: LLMConfig) -> None:
        self.config = rag_config
        self.llm_config = llm_config
        self._indices: dict[str, object] = {}
        self._query_engines: dict[str, object] = {}
        self._configured = False

    def _ensure_configured(self) -> None:
        if self._configured:
            return
        self._configure_llama_index()
        self._configured = True

    def _configure_llama_index(self) -> None:
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        Settings.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        provider = self.llm_config.provider.value

        if provider == "openai":
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding
                from llama_index.llms.openai import OpenAI

                Settings.embed_model = OpenAIEmbedding(
                    model=self.config.embedding_model,
                    api_key=self.llm_config.api_key,
                )
                Settings.llm = OpenAI(
                    model=self.llm_config.model,
                    temperature=self.llm_config.temperature,
                    api_key=self.llm_config.api_key,
                )
            except ImportError:
                logger.warning("llama-index-llms-openai not installed; using defaults")

        elif provider == "anthropic":
            try:
                from llama_index.llms.anthropic import Anthropic
                Settings.llm = Anthropic(
                    model=self.llm_config.model,
                    api_key=self.llm_config.api_key,
                )
            except ImportError:
                logger.warning("llama-index-llms-anthropic not installed; using defaults")

    async def ingest_directory(
        self,
        path: str,
        index_name: str = "default",
        file_extensions: Optional[list[str]] = None,
    ) -> object:
        self._ensure_configured()
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

        kwargs: dict = {"input_dir": path}
        if file_extensions:
            kwargs["required_exts"] = file_extensions

        documents = SimpleDirectoryReader(**kwargs).load_data()
        logger.info("Ingested %d documents from %s → index '%s'", len(documents), path, index_name)

        index = VectorStoreIndex.from_documents(documents)
        self._persist(index, index_name)
        self._indices[index_name] = index
        self._query_engines.pop(index_name, None)  # invalidate cached engine
        return index

    async def ingest_documents(
        self,
        documents: list[dict],
        index_name: str = "default",
    ) -> object:
        self._ensure_configured()
        from llama_index.core import Document, VectorStoreIndex

        llama_docs = [
            Document(text=d["text"], metadata=d.get("metadata", {}))
            for d in documents
        ]
        index = VectorStoreIndex.from_documents(llama_docs)
        self._persist(index, index_name)
        self._indices[index_name] = index
        self._query_engines.pop(index_name, None)
        return index

    def load_index(self, index_name: str = "default") -> object:
        if index_name in self._indices:
            return self._indices[index_name]

        persist_path = Path(self.config.persist_dir) / index_name
        if not persist_path.exists():
            raise IndexNotFoundError(f"No persisted index at {persist_path}")

        from llama_index.core import StorageContext, load_index_from_storage

        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage_context)
        self._indices[index_name] = index
        return index

    def get_query_engine(
        self,
        index_name: str = "default",
        similarity_top_k: Optional[int] = None,
    ) -> object:
        top_k = similarity_top_k or self.config.similarity_top_k
        cache_key = f"{index_name}@{top_k}"
        if cache_key in self._query_engines:
            return self._query_engines[cache_key]

        index = self._indices.get(index_name)
        if index is None:
            index = self.load_index(index_name)

        from llama_index.core.postprocessor import SimilarityPostprocessor
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import VectorIndexRetriever

        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.3)]

        if self.config.reranker:
            try:
                from llama_index.postprocessor.cohere_rerank import CohereRerank
                postprocessors.append(CohereRerank(top_n=top_k))
            except ImportError:
                logger.warning("Cohere reranker requested but llama-index-postprocessor-cohere-rerank not installed")

        engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=postprocessors)
        self._query_engines[cache_key] = engine
        return engine

    def as_tool_definition(
        self,
        index_name: str = "default",
        tool_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ToolDefinition:
        engine = self.get_query_engine(index_name)
        name = tool_name or f"search_{index_name}"
        desc = description or f"Search the '{index_name}' knowledge base for relevant information."

        async def query_handler(query: str) -> str:
            response = engine.query(query)
            return str(response)

        return ToolDefinition(
            name=name,
            description=desc,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The question or search query",
                    required=True,
                )
            ],
            source=ToolSource.RAG,
            handler=query_handler,
            metadata={"index_name": index_name},
        )

    def list_indices(self) -> list[str]:
        persisted: set[str] = set()
        persist_dir = Path(self.config.persist_dir)
        if persist_dir.exists():
            persisted = {p.name for p in persist_dir.iterdir() if p.is_dir()}
        return list(persisted | set(self._indices.keys()))

    def _persist(self, index, index_name: str) -> None:
        persist_path = Path(self.config.persist_dir) / index_name
        persist_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_path))
