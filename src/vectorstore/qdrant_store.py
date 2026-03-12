"""
Module 4 — Vector Database Layer

Wraps Qdrant via LangChain's QdrantVectorStore for storing and retrieving
limitation chunks.  Supports both in-memory mode (development) and a
persistent server (production).

Public API
----------
LimitationVectorStore.from_documents(docs, embeddings) -> LimitationVectorStore
LimitationVectorStore.similarity_search(query, k, filter) -> list[Document]
LimitationVectorStore.as_retriever(k) -> VectorStoreRetriever
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.logger import logger


class LimitationVectorStore:
    """
    Thin wrapper around LangChain's QdrantVectorStore that handles
    collection creation, document ingestion, and filtered search.
    """

    def __init__(
        self,
        store: QdrantVectorStore,
        client: QdrantClient,
        collection_name: str,
    ) -> None:
        self._store = store
        self._client = client
        self._collection_name = collection_name

    # ------------------------------------------------------------------ #
    # Factory constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def create_in_memory(
        cls,
        embeddings: Embeddings,
        collection_name: str = "research_limitations",
        vector_size: int = 1536,
    ) -> "LimitationVectorStore":
        """
        Create an in-memory Qdrant instance.
        Ideal for development / single-session runs.
        """
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        logger.info("Qdrant in-memory collection '{}' created", collection_name)
        return cls(store, client, collection_name)

    @classmethod
    def create_persistent(
        cls,
        embeddings: Embeddings,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: str = "research_limitations",
        vector_size: int = 1536,
    ) -> "LimitationVectorStore":
        """
        Connect to a running Qdrant server.
        Creates the collection if it does not exist.
        """
        client = QdrantClient(host=host, port=port, api_key=api_key)
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '{}'", collection_name)
        else:
            logger.info("Reusing existing Qdrant collection '{}'", collection_name)

        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        return cls(store, client, collection_name)

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and persist a list of Documents."""
        if not documents:
            logger.warning("add_documents called with empty list")
            return
        self._store.add_documents(documents)
        logger.info(
            "Added {} documents to collection '{}'",
            len(documents),
            self._collection_name,
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def similarity_search(
        self,
        query: str,
        k: int = 8,
        filter_category: Optional[str] = None,
    ) -> list[Document]:
        """
        Retrieve top-k most similar chunks.

        Parameters
        ----------
        query : str
            Natural language query.
        k : int
            Number of results to return.
        filter_category : str | None
            If set, restrict results to a category:
            'limitation', 'research_gap', 'future_work',
            'methodological_weakness', 'raw'.
        """
        search_kwargs: dict = {"k": k}
        if filter_category:
            search_kwargs["filter"] = {
                "must": [
                    {"key": "metadata.category", "match": {"value": filter_category}}
                ]
            }

        results = self._store.similarity_search(query, **search_kwargs)
        logger.debug(
            "similarity_search '{}' → {} results (filter={})",
            query[:60],
            len(results),
            filter_category,
        )
        return results

    def as_retriever(self, k: int = 8):
        """Return a LangChain-compatible retriever."""
        return self._store.as_retriever(search_kwargs={"k": k})

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """Return the total number of stored vectors."""
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0
