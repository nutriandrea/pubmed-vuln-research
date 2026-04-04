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
from src.rag.prompts import (
    RAG_ANSWER_PROMPT,
    SYNTHESIS_PROMPT,
    RESEARCH_GRADE_PROMPT,
    INSIGHT_GENERATOR_PROMPT,
)
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
        severity = meta.get("severity", "")
        if category:
            source += f"[{category}] "
        if severity:
            source += f"[{severity}] "
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

    def ask(
        self,
        question: str,
        filter_category: Optional[str] = None,
        filter_type: Optional[str] = "limitation",
    ) -> str:
        """
        Answer a single question about research limitations.

        The question is used both as the retrieval query and as the
        prompt input for the LLM.
        
        Parameters
        ----------
        question : str
            The question to answer.
        filter_category : str | None
            Filter by category (dataset, methodology, evaluation, bias, etc.).
        filter_type : str | None
            Filter by type (default: "limitation").
        """
        logger.info("RAG ask: '{}'", question)
        retrieved = self._store.similarity_search(
            question,
            k=self._top_k,
            filter_type=filter_type,
            filter_category=filter_category,
        )
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

    def ask_with_sources(
        self,
        question: str,
        filter_category: Optional[str] = None,
        filter_type: Optional[str] = "limitation",
    ) -> dict:
        """
        Like ask(), but also returns the source documents used.
        
        Parameters
        ----------
        question : str
            The question to answer.
        filter_category : str | None
            Filter by category.
        filter_type : str | None
            Filter by type (default: "limitation").

        Returns
        -------
        dict with keys:
            - 'answer': str
            - 'sources': list[dict]  (paper_title, year, pmid, category, severity)
        """
        retrieved = self._store.similarity_search(
            question,
            k=self._top_k,
            filter_type=filter_type,
            filter_category=filter_category,
        )
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
                    "severity": meta.get("severity"),
                    "pubmed_url": meta.get("pubmed_url"),
                })

        return {"answer": answer, "sources": sources}

    def ask_research_grade(
        self,
        question: str,
        filter_category: Optional[str] = None,
    ) -> dict:
        """
        Answer with research-grade output including confidence levels and evidence.
        
        Returns:
            dict with 'answer', 'sources', 'confidence', 'key_limitations'
        """
        logger.info("Research-grade ask: '{}'", question)
        retrieved = self._store.similarity_search(
            question,
            k=self._top_k * 2,  # Get more for better analysis
            filter_type="limitation",
            filter_category=filter_category,
        )
        if not retrieved:
            return {
                "answer": "No relevant limitations found.",
                "sources": [],
                "confidence": "LOW",
                "key_limitations": [],
            }

        context = _format_docs(retrieved)
        answer = self._research_grade_chain.invoke(
            {"context": context, "question": question}
        )

        sources = []
        key_limitations = []
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
                    "category": meta.get("category"),
                    "severity": meta.get("severity"),
                })
                
                if meta.get("severity") == "high":
                    key_limitations.append({
                        "text": doc.page_content[:200],
                        "category": meta.get("category"),
                        "severity": meta.get("severity"),
                    })

        # Estimate confidence based on number of sources
        confidence = "HIGH" if len(sources) > 20 else "MEDIUM" if len(sources) > 5 else "LOW"

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "key_limitations": key_limitations[:5],
        }

    def generate_insights(self, n_papers: int = 0) -> str:
        """
        Automatically generate top insights without user query.
        
        Uses vulnerability data to create structured insights.
        """
        logger.info("Generating automatic insights")
        
        # Retrieve a broad sample of limitations
        query = "limitations weaknesses research gaps methodology"
        retrieved = self._store.similarity_search(query, k=self._top_k * 3)
        
        if not retrieved:
            return "No data available for insights. Run ingestion first."

        context = _format_docs(retrieved)
        n = n_papers or len({d.metadata.get("pmid") for d in retrieved})
        
        response = self._insight_chain.invoke(
            {"context": context, "n_papers": n}
        )
        
        logger.info("Insights generated ({} chars)", len(response))
        return response

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
        # Research-grade chain: enhanced answer with confidence + evidence
        self._research_grade_chain = (
            RESEARCH_GRADE_PROMPT
            | self._llm
            | self._str_parser
        )
        # Insight generator chain: automatic insights
        self._insight_chain = (
            INSIGHT_GENERATOR_PROMPT
            | self._llm
            | self._str_parser
        )
