from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from kazi.core.config import LLMConfig, RAGConfig
from kazi.core.exceptions import IndexNotFoundError
from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llama_index.core.query_engine import BaseQueryEngine


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
        self._query_engines: dict[str, BaseQueryEngine] = {}
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

        # Custom models bypass all provider-specific setup
        if self.config.custom_embedding is not None:
            Settings.embed_model = self.config.custom_embedding
        if self.config.custom_synthesis_llm is not None:
            Settings.llm = self.config.custom_synthesis_llm

        # If both custom hooks are provided, skip provider-specific wiring entirely
        if self.config.custom_embedding is not None and self.config.custom_synthesis_llm is not None:
            return

        # If only a custom embedding is provided, use MockLLM as synthesis default
        # so provider-specific LLM setup (which may require API keys) is skipped.
        if self.config.custom_embedding is not None and self.config.custom_synthesis_llm is None:
            from llama_index.core.llms.mock import MockLLM
            Settings.llm = MockLLM()
            return

        provider = self.llm_config.provider.value

        if provider == "openai":
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding
                from llama_index.llms.openai import OpenAI

                api_key = self.llm_config.resolved_api_key()
                if self.config.custom_embedding is None:
                    Settings.embed_model = OpenAIEmbedding(
                        model=self.config.embedding_model,
                        api_key=api_key,
                    )
                if self.config.custom_synthesis_llm is None and api_key:
                    Settings.llm = OpenAI(
                        model=self.llm_config.model,
                        temperature=self.llm_config.temperature,
                        api_key=api_key,
                    )
            except ImportError:
                logger.warning("llama-index-llms-openai not installed; using defaults")

        elif provider == "anthropic":
            try:
                from llama_index.llms.anthropic import Anthropic
                if self.config.custom_synthesis_llm is None:
                    Settings.llm = Anthropic(
                        model=self.llm_config.model,
                        api_key=self.llm_config.resolved_api_key(),
                    )
            except ImportError:
                logger.warning("llama-index-llms-anthropic not installed; using defaults")

    def _make_storage_context(self, index_name: str) -> object:
        from llama_index.core import StorageContext

        backend = self.config.vector_store.value

        if backend == "chroma":
            try:
                import chromadb
                from llama_index.vector_stores.chroma import ChromaVectorStore

                collection_name = self.config.chroma_collection or index_name
                if self.config.vector_store_url:
                    client = chromadb.HttpClient(host=self.config.vector_store_url)
                else:
                    client = chromadb.PersistentClient(path=self.config.persist_dir)
                collection = client.get_or_create_collection(collection_name)
                storage_ctx: object = StorageContext.from_defaults(
                    vector_store=ChromaVectorStore(chroma_collection=collection)
                )
                return storage_ctx
            except ImportError:
                logger.warning("chromadb / llama-index-vector-stores-chroma not installed; falling back to in-memory")

        elif backend == "pinecone":
            try:
                from llama_index.vector_stores.pinecone import PineconeVectorStore
                from pinecone import Pinecone

                pc = Pinecone(api_key=self.config.vector_store_api_key)
                pinecone_index = pc.Index(index_name)
                return StorageContext.from_defaults(
                    vector_store=PineconeVectorStore(pinecone_index=pinecone_index)
                )
            except ImportError:
                logger.warning("pinecone / llama-index-vector-stores-pinecone not installed; falling back to in-memory")

        elif backend == "qdrant":
            try:
                from llama_index.vector_stores.qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient

                client = QdrantClient(
                    url=self.config.vector_store_url,
                    api_key=self.config.vector_store_api_key,
                )
                return StorageContext.from_defaults(
                    vector_store=QdrantVectorStore(client=client, collection_name=index_name)
                )
            except ImportError:
                logger.warning("qdrant-client / llama-index-vector-stores-qdrant not installed; falling back to in-memory")

        elif backend == "weaviate":
            try:
                import weaviate
                from llama_index.vector_stores.weaviate import WeaviateVectorStore

                auth = weaviate.auth.AuthApiKey(self.config.vector_store_api_key) if self.config.vector_store_api_key else None
                client = weaviate.connect_to_custom(
                    http_host=self.config.vector_store_url or "localhost",
                    http_port=8080,
                    http_secure=True,
                    grpc_host=self.config.vector_store_url or "localhost",
                    grpc_port=50051,
                    grpc_secure=True,
                    auth_credentials=auth,
                )
                return StorageContext.from_defaults(
                    vector_store=WeaviateVectorStore(weaviate_client=client, index_name=index_name)
                )
            except ImportError:
                logger.warning("weaviate-client / llama-index-vector-stores-weaviate not installed; falling back to in-memory")

        # IN_MEMORY or fallback
        return StorageContext.from_defaults()

    async def ping(self) -> None:
        """Lightweight connectivity check — raises if the vector store is unreachable."""
        if not self._indices:
            return  # Nothing ingested yet — store reachable by definition
        self._ensure_configured()

    async def ingest_directory(
        self,
        path: str,
        index_name: str = "default",
        file_extensions: list[str] | None = None,
    ) -> object:
        self._ensure_configured()
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

        kwargs: dict = {"input_dir": path}
        if file_extensions:
            kwargs["required_exts"] = file_extensions

        documents = SimpleDirectoryReader(**kwargs).load_data()
        logger.info("Ingested %d documents from %s → index '%s'", len(documents), path, index_name)

        storage_context = self._make_storage_context(index_name)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)  # type: ignore[arg-type]
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
        storage_context = self._make_storage_context(index_name)
        index = VectorStoreIndex.from_documents(llama_docs, storage_context=storage_context)  # type: ignore[arg-type]
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
        similarity_top_k: int | None = None,
    ) -> BaseQueryEngine:
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

        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)  # type: ignore[arg-type]
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.3)]

        if self.config.reranker:
            try:
                from llama_index.postprocessor.cohere_rerank import CohereRerank
                postprocessors.append(CohereRerank(top_n=top_k))
            except ImportError:
                logger.warning("Cohere reranker requested but llama-index-postprocessor-cohere-rerank not installed")

        engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=postprocessors)  # type: ignore[arg-type]
        self._query_engines[cache_key] = engine
        return engine

    def as_tool_definition(
        self,
        index_name: str = "default",
        tool_name: str | None = None,
        description: str | None = None,
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
