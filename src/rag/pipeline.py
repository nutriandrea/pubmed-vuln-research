"""
Module 5 — RAG Pipeline
Module 6 — LLM Reasoning Layer

Combines retrieval from the vector store with LLM synthesis to answer
research-limitation questions.

Public API
----------
LimitationRAGPipeline.ask(question)        -> str  (single Q&A)
LimitationRAGPipeline.synthesize(topic)    -> str  (full report)
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.logger import logger
from src.rag.prompts import RAG_ANSWER_PROMPT, SYNTHESIS_PROMPT
from src.vectorstore.qdrant_store import LimitationVectorStore


def _format_docs(docs: list[Document]) -> str:
    """
    Turn a list of retrieved Documents into a single context string,
    prepending each chunk with its source citation.
    """
    parts = []
    for doc in docs:
        meta = doc.metadata
        source = f"[{meta.get('paper_title', 'Unknown')} ({meta.get('year', '?')})] "
        category = meta.get("category", "")
        if category:
            source += f"[{category}] "
        parts.append(source + doc.page_content)
    return "\n\n---\n\n".join(parts)


class LimitationRAGPipeline:
    """
    End-to-end RAG pipeline for research limitation analysis.

    Parameters
    ----------
    vector_store : LimitationVectorStore
        Pre-populated vector store with limitation chunks.
    model_name : str
        OpenAI chat model to use for synthesis.
    top_k : int
        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        vector_store: LimitationVectorStore,
        model_name: str = "gpt-4o-mini",
        top_k: int = 8,
    ) -> None:
        self._store = vector_store
        self._top_k = top_k
        self._llm = ChatOpenAI(model=model_name, temperature=0)
        self._str_parser = StrOutputParser()
        self._build_chain()

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def ask(self, question: str) -> str:
        """
        Answer a single question about research limitations.

        The question is used both as the retrieval query and as the
        prompt input for the LLM.
        """
        logger.info("RAG ask: '{}'", question)
        retrieved = self._store.similarity_search(question, k=self._top_k)
        if not retrieved:
            return "No relevant limitations found in the knowledge base for this query."

        context = _format_docs(retrieved)
        response = self._qa_chain.invoke(
            {"context": context, "question": question}
        )
        logger.info("RAG answer generated ({} chars)", len(response))
        return response

    def synthesize(self, topic: str, n_papers: Optional[int] = None) -> str:
        """
        Generate a full structured limitations report for a topic.

        Retrieves a broader set of chunks and asks the LLM to synthesize
        them into a categorised report.

        Parameters
        ----------
        topic : str
            The research topic (e.g. "breast cancer detection with deep learning").
        n_papers : int | None
            Number of source papers (shown in the system prompt for context).
        """
        logger.info("Synthesizing limitations report for topic: '{}'", topic)
        # Use a broad query to surface all categories
        query = (
            f"limitations weaknesses research gaps methodology {topic}"
        )
        # Retrieve more chunks for synthesis
        retrieved = self._store.similarity_search(query, k=min(self._top_k * 2, 20))
        if not retrieved:
            return "No limitations have been indexed yet. Run the ingestion pipeline first."

        context = _format_docs(retrieved)
        n = n_papers or len({d.metadata.get("pmid") for d in retrieved})

        response = self._synthesis_chain.invoke(
            {"context": context, "topic": topic, "n_papers": n}
        )
        logger.info("Synthesis report generated ({} chars)", len(response))
        return response

    def ask_with_sources(self, question: str) -> dict:
        """
        Like ask(), but also returns the source documents used.

        Returns
        -------
        dict with keys:
            - 'answer': str
            - 'sources': list[dict]  (paper_title, year, pmid, category)
        """
        retrieved = self._store.similarity_search(question, k=self._top_k)
        if not retrieved:
            return {
                "answer": "No relevant limitations found.",
                "sources": [],
            }

        context = _format_docs(retrieved)
        answer = self._qa_chain.invoke({"context": context, "question": question})

        sources = []
        seen = set()
        for doc in retrieved:
            meta = doc.metadata
            key = (meta.get("pmid"), meta.get("category"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "paper_title": meta.get("paper_title"),
                    "year": meta.get("year"),
                    "pmid": meta.get("pmid"),
                    "journal": meta.get("journal"),
                    "category": meta.get("category"),
                    "pubmed_url": meta.get("pubmed_url"),
                })

        return {"answer": answer, "sources": sources}

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _build_chain(self) -> None:
        """Pre-build both LangChain LCEL chains."""
        # Q&A chain: context + question → answer
        self._qa_chain = (
            RAG_ANSWER_PROMPT
            | self._llm
            | self._str_parser
        )
        # Synthesis chain: context + topic + n_papers → full report
        self._synthesis_chain = (
            SYNTHESIS_PROMPT
            | self._llm
            | self._str_parser
        )
