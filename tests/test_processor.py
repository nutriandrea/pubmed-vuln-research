"""
Unit tests for the knowledge processor (DocumentBuilder).
"""

from __future__ import annotations

import pytest
from src.processor.document_builder import DocumentBuilder, _clean_text
from src.extractor.models import ExtractedLimitations


def _make_extracted(pmid="11111", n_limitations=3, n_gaps=2) -> ExtractedLimitations:
    return ExtractedLimitations(
        pmid=pmid,
        paper_title=f"Test Paper {pmid}",
        year="2023",
        authors=["Author A", "Author B"],
        journal="Test Journal",
        limitations=[f"Limitation {i}" for i in range(n_limitations)],
        research_gaps=[f"Gap {i}" for i in range(n_gaps)],
        future_work=["Future direction one"],
        methodological_weaknesses=[],
    )


class TestCleanText:
    def test_removes_excess_blank_lines(self):
        text = "line1\n\n\n\nline2"
        result = _clean_text(text)
        assert "\n\n\n" not in result

    def test_collapses_spaces(self):
        text = "word1   word2\tword3"
        result = _clean_text(text)
        assert "  " not in result


class TestDocumentBuilder:
    def setup_method(self):
        self.builder = DocumentBuilder(chunk_size=500, chunk_overlap=50)

    def test_builds_documents_from_extracted(self):
        extracted = _make_extracted(n_limitations=3, n_gaps=2)
        docs = self.builder.build([extracted])
        assert len(docs) > 0

    def test_metadata_is_attached(self):
        extracted = _make_extracted(pmid="55555")
        docs = self.builder.build([extracted])
        for doc in docs:
            assert doc.metadata["pmid"] == "55555"
            assert doc.metadata["paper_title"] == "Test Paper 55555"
            assert "category" in doc.metadata

    def test_categories_present(self):
        extracted = _make_extracted()
        docs = self.builder.build([extracted])
        categories = {d.metadata["category"] for d in docs}
        assert "limitation" in categories
        assert "research_gap" in categories
        assert "future_work" in categories

    def test_fallback_to_raw_text_when_no_structured_items(self):
        extracted = ExtractedLimitations(
            pmid="00000",
            paper_title="Empty Paper",
            year="2022",
            raw_limitation_text="This study has many limitations that we discuss here.",
        )
        docs = self.builder.build([extracted])
        assert len(docs) > 0
        assert docs[0].metadata["category"] == "raw"

    def test_multiple_papers(self):
        papers = [_make_extracted(pmid=str(i)) for i in range(5)]
        docs = self.builder.build(papers)
        pmids = {d.metadata["pmid"] for d in docs}
        assert len(pmids) == 5
