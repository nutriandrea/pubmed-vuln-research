"""
Main orchestrator — wires every module together into a single pipeline.

Usage (programmatic):
    from src.orchestrator import ResearchLimitationAnalyzer

    analyzer = ResearchLimitationAnalyzer()
    analyzer.ingest(topic="breast cancer", date_from=2020, date_to=2025,
                    paper_type="review", max_papers=10)
    report = analyzer.synthesize()
    answer = analyzer.ask("What are the main dataset limitations?")

The orchestrator also handles persistence:
  - Raw paper metadata saved to data/raw/<topic_slug>/
  - Processed limitations saved to data/processed/<topic_slug>/
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, List

from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from config.settings import settings
from src.extractor.section_extractor import LimitationExtractor
from src.extractor.models import ExtractedLimitations
from src.logger import logger
from src.processor.document_builder import DocumentBuilder
from src.rag.pipeline import LimitationRAGPipeline
from src.retriever.pubmed_client import PubMedClient, PubMedQueryParams
from src.retriever.models import PaperMetadata
from src.vectorstore.qdrant_store import LimitationVectorStore


def _slug(text: str) -> str:
    """Convert a topic string to a safe directory name."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class ResearchLimitationAnalyzer:
    """
    High-level facade for the full pipeline.

    Attributes
    ----------
    topic : str | None
        Set after calling ingest().
    rag : LimitationRAGPipeline | None
        Available after ingest() completes.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 8,
    ) -> None:
        self._model_name = model_name
        self._embeddings = OpenAIEmbeddings(model=embedding_model)
        self._top_k = top_k
        self.topic: Optional[str] = None
        self._vector_store: Optional[LimitationVectorStore] = None
        self.rag: Optional[LimitationRAGPipeline] = None
        self._n_papers: int = 0

    # ------------------------------------------------------------------ #
    # Ingestion pipeline
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        topic: str,
        date_from: int = 2020,
        date_to: int = 2025,
        paper_type: Optional[str] = None,
        max_papers: int = 10,
        method: Optional[str] = None,
        exclude_terms: Optional[List[str]] = None,
        reset_knowledge_base: bool = True,
    ) -> int:
        """
        Run the full ingestion pipeline:
          1. Search PubMed with title-only query and synonym expansion
          2. Extract limitations per paper
          3. Chunk and embed documents
          4. Store in Qdrant (clearing previous content if reset=True)

        Returns the number of documents indexed.
        """
        self.topic = topic
        slug = _slug(topic)
        raw_dir = settings.raw_data_dir / slug
        processed_dir = settings.processed_data_dir / slug
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # ---- Reset knowledge base if requested ----
        if reset_knowledge_base and self._vector_store is not None:
            logger.info("=== Resetting knowledge base ===")
            try:
                self._vector_store.clear()
                logger.info("Knowledge base cleared")
            except Exception as e:
                logger.warning(f"Could not clear vector store: {e}")

        # ---- Step 1: Retrieve from PubMed ----
        logger.info("=== STEP 1: PubMed retrieval (title-only with synonyms) ===")
        client = PubMedClient(
            email=settings.ncbi_email,
            api_key=settings.ncbi_api_key,
            cache_dir=raw_dir,
        )
        params = PubMedQueryParams(
            topic=topic,
            date_from=date_from,
            date_to=date_to,
            paper_type=paper_type,
            max_results=max_papers,
            method=method,
            exclude_terms=exclude_terms,
            use_synonym_expansion=True,
        )
        papers: list[PaperMetadata] = client.search(params)
        logger.info("Retrieved {} papers", len(papers))

        if not papers:
            logger.warning("No papers found. Check query or increase date range.")
            return 0

        # ---- Step 2: Extract limitations ----
        logger.info("=== STEP 2: Limitation extraction ===")
        extractor = LimitationExtractor(model_name=self._model_name)
        extracted_list: list[ExtractedLimitations] = []

        for paper in tqdm(papers, desc="Extracting limitations"):
            extracted = extractor.extract(paper)
            extracted_list.append(extracted)
            # Persist to disk
            out_path = processed_dir / f"{paper.pmid}.json"
            with open(out_path, "w") as f:
                json.dump(extracted.model_dump(), f, indent=2)

        logger.info(
            "Extraction complete: {} papers processed", len(extracted_list)
        )

        # ---- Step 3: Build document chunks ----
        logger.info("=== STEP 3: Document chunking ===")
        builder = DocumentBuilder(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        documents = builder.build(extracted_list)
        logger.info("Created {} document chunks", len(documents))

        # ---- Step 4: Index into Qdrant ----
        logger.info("=== STEP 4: Indexing into Qdrant ===")
        if settings.qdrant_in_memory:
            self._vector_store = LimitationVectorStore.create_in_memory(
                embeddings=self._embeddings,
                collection_name=settings.qdrant_collection,
            )
        else:
            self._vector_store = LimitationVectorStore.create_persistent(
                embeddings=self._embeddings,
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                collection_name=settings.qdrant_collection,
            )

        self._vector_store.add_documents(documents)
        self._n_papers = len(papers)

        # ---- Step 5: Build RAG pipeline ----
        logger.info("=== STEP 5: RAG pipeline ready ===")
        self.rag = LimitationRAGPipeline(
            vector_store=self._vector_store,
            model_name=self._model_name,
            top_k=self._top_k,
        )

        indexed = self._vector_store.count()
        logger.info(
            "Ingestion complete: {} papers, {} chunks indexed",
            self._n_papers,
            indexed,
        )
        return indexed

    # ------------------------------------------------------------------ #
    # Query interface
    # ------------------------------------------------------------------ #

    def ask(self, question: str) -> str:
        """Ask a free-form question about research limitations."""
        self._require_rag()
        return self.rag.ask(question)  # type: ignore

    def ask_with_sources(self, question: str) -> dict:
        """Ask a question and return answer + source references."""
        self._require_rag()
        return self.rag.ask_with_sources(question)  # type: ignore

    def synthesize(self) -> str:
        """Generate the full structured limitations report for the ingested topic."""
        self._require_rag()
        topic = self.topic if self.topic else ""
        return self.rag.synthesize(topic=topic, n_papers=self._n_papers)  # type: ignore

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _require_rag(self) -> None:
        if self.rag is None:
            raise RuntimeError(
                "No knowledge base loaded. Call ingest() first."
            )
