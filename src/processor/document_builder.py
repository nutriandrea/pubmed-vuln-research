"""
Module 3 — Knowledge Processing

Converts ExtractedLimitations objects into LangChain Documents with rich
metadata, then chunks them ready for embedding + vector storage.

ENHANCED: Now includes structured metadata from classification layer:
- type: "limitation" (always)
- category: dataset, methodology, evaluation, bias, other
- severity: low, medium, high
- normalized_text: simplified form for clustering

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


def _normalize_limitation(text: str) -> str:
    """
    Normalize limitation text for better clustering.
    Converts to lowercase, removes stop words, standardizes common terms.
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    replacements = {
        r'\bsmall\s+sample\s+size\b': 'small sample',
        r'\blimited\s+sample\s+size\b': 'small sample',
        r'\blimited\s+sample\b': 'small sample',
        r'\bsmall\s+dataset\b': 'small dataset',
        r'\blimited\s+data\b': 'limited data',
        r'\blimited\s+cohort\b': 'small sample',
        r'\blimited\s+number\s+of\s+patients\b': 'small sample',
        r'\black\s+of\s+': 'lack of ',
        r'\binsufficient\s+': 'insufficient ',
        r'\binadequate\s+': 'inadequate ',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
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
            "extraction_method": ex.extraction_method,
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
                meta = {
                    **base_metadata,
                    "type": "limitation",
                    "category": category,
                    "source_section": category,
                    "chunk_index": i,
                }
                documents.append(Document(page_content=chunk, metadata=meta))

        # Process classified limitations (from classification layer)
        if ex.classified_limitations:
            for classified in ex.classified_limitations:
                text = classified.text.strip()
                if not text:
                    continue
                
                # Create document for each classified limitation
                meta = {
                    **base_metadata,
                    "type": "limitation",
                    "category": classified.category,
                    "severity": classified.severity,
                    "normalized_text": _normalize_limitation(classified.text),
                    "source_section": "classified",
                    "chunk_index": 0,
                }
                documents.append(Document(page_content=text, metadata=meta))
                logger.debug(
                    "Added classified limitation for PMID {}: category={}, severity={}",
                    ex.pmid, classified.category, classified.severity
                )

        # If LLM found nothing but raw text exists, chunk the raw text
        if not documents and ex.raw_limitation_text.strip():
            logger.debug("Falling back to raw limitation text for PMID {}", ex.pmid)
            block = _clean_text(ex.raw_limitation_text)
            chunks = self._splitter.split_text(block)
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                meta = {**base_metadata, "type": "limitation", "category": "raw", "chunk_index": i}
                documents.append(Document(page_content=chunk, metadata=meta))

        return documents

    def _items_to_block(self, title: str, year: str, items: list[str]) -> str:
        """Format a list of bullet points as a text block."""
        header = f"From: {title} ({year})\n"
        body = "\n".join(f"- {item}" for item in items if item.strip())
        return header + body
