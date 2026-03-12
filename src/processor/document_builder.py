"""
Module 3 — Knowledge Processing

Converts ExtractedLimitations objects into LangChain Documents with rich
metadata, then chunks them ready for embedding + vector storage.

Public API
----------
DocumentBuilder.build(extracted: list[ExtractedLimitations]) -> list[Document]
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import logger
from src.extractor.models import ExtractedLimitations


def _clean_text(text: str) -> str:
    """
    Normalise whitespace and remove artefacts from PMC extraction.
    Keeps sentence structure intact.
    """
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove very short orphan lines (page numbers, headers artefacts)
    lines = [l for l in text.splitlines() if len(l.strip()) > 3 or l.strip() == ""]
    text = "\n".join(lines)
    # Normalise whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class DocumentBuilder:
    """
    Converts structured ExtractedLimitations into chunked LangChain Documents.

    Each Document carries metadata so that every retrieved chunk can be
    traced back to its source paper.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "],
            length_function=len,
        )

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def build(self, extracted_list: list[ExtractedLimitations]) -> list[Document]:
        """
        Build a list of chunked Documents from all extracted papers.
        """
        all_docs: list[Document] = []
        for extracted in extracted_list:
            docs = self._process_one(extracted)
            all_docs.extend(docs)
            logger.debug(
                "PMID {} → {} document chunks", extracted.pmid, len(docs)
            )

        logger.info(
            "DocumentBuilder: {} papers → {} total chunks",
            len(extracted_list),
            len(all_docs),
        )
        return all_docs

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _process_one(self, ex: ExtractedLimitations) -> list[Document]:
        """Convert a single ExtractedLimitations into multiple Documents."""
        base_metadata = {
            "pmid": ex.pmid,
            "paper_title": ex.paper_title,
            "year": ex.year,
            "authors": "; ".join(ex.authors[:5]),
            "journal": ex.journal,
            "doi": ex.doi,
            "pubmed_url": ex.pubmed_url,
            "used_full_text": ex.used_full_text,
        }

        documents: list[Document] = []

        # Build one document per category with a category tag
        categories = {
            "limitation": ex.limitations,
            "research_gap": ex.research_gaps,
            "future_work": ex.future_work,
            "methodological_weakness": ex.methodological_weaknesses,
        }

        for category, items in categories.items():
            if not items:
                continue
            block = self._items_to_block(ex.paper_title, ex.year, items)
            block = _clean_text(block)
            chunks = self._splitter.split_text(block)
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                meta = {**base_metadata, "category": category, "chunk_index": i}
                documents.append(Document(page_content=chunk, metadata=meta))

        # If LLM found nothing but raw text exists, chunk the raw text
        if not documents and ex.raw_limitation_text.strip():
            logger.debug("Falling back to raw limitation text for PMID {}", ex.pmid)
            block = _clean_text(ex.raw_limitation_text)
            chunks = self._splitter.split_text(block)
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                meta = {**base_metadata, "category": "raw", "chunk_index": i}
                documents.append(Document(page_content=chunk, metadata=meta))

        return documents

    def _items_to_block(self, title: str, year: str, items: list[str]) -> str:
        """Format a list of bullet points as a text block."""
        header = f"From: {title} ({year})\n"
        body = "\n".join(f"- {item}" for item in items if item.strip())
        return header + body
