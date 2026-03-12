"""
Unit tests for the text extraction module.
Uses mocked LLM calls — no real API usage.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.extractor.section_extractor import LimitationExtractor, _SECTION_PATTERNS
from src.retriever.models import PaperMetadata


FULL_TEXT_WITH_SECTIONS = """
Introduction
This paper presents a deep learning model for cancer detection.

Methods
We trained a CNN on 500 images.

Results
Accuracy was 87%.

Limitations
The dataset was limited to a single institution, which may reduce generalizability.
The model was not validated on external cohorts.
Class imbalance was present in the training data.

Future Work
Future studies should include multi-center datasets.
Prospective validation is recommended.

References
[1] Smith et al. 2020
"""


class TestSectionExtraction:
    def setup_method(self):
        self.extractor = LimitationExtractor.__new__(LimitationExtractor)

    def test_heuristic_finds_limitations_section(self):
        result = self.extractor._extract_sections_heuristic(FULL_TEXT_WITH_SECTIONS)
        assert "single institution" in result
        assert "LIMITATIONS" in result

    def test_heuristic_finds_future_work_section(self):
        result = self.extractor._extract_sections_heuristic(FULL_TEXT_WITH_SECTIONS)
        assert "multi-center" in result

    def test_heuristic_stops_at_references(self):
        result = self.extractor._extract_sections_heuristic(FULL_TEXT_WITH_SECTIONS)
        assert "Smith et al" not in result

    def test_heuristic_returns_empty_for_plain_abstract(self):
        abstract = "We propose a new model. Results are promising. Accuracy 90%."
        result = self.extractor._extract_sections_heuristic(abstract)
        assert result == ""


class TestLimitationExtractor:
    @patch("src.extractor.section_extractor.ChatOpenAI")
    def test_extract_uses_abstract_when_no_full_text(self, mock_chat):
        mock_llm_instance = MagicMock()
        mock_chat.return_value = mock_llm_instance
        mock_llm_instance.return_value = MagicMock(content='{"limitations":["small sample"],"research_gaps":[],"future_work":[],"methodological_weaknesses":[]}')

        extractor = LimitationExtractor()
        # Patch the chain to return structured data directly
        extractor._chain = MagicMock()
        extractor._chain.invoke.return_value = {
            "limitations": ["Small sample size limits generalizability"],
            "research_gaps": ["No external validation"],
            "future_work": ["Multi-center study needed"],
            "methodological_weaknesses": [],
        }

        paper = PaperMetadata(
            pmid="99999",
            title="Test Paper",
            year="2023",
            abstract="This study has limitations including small sample size.",
            authors=["Doe J"],
            journal="Test Journal",
        )

        result = extractor.extract(paper)
        assert result.pmid == "99999"
        assert len(result.limitations) > 0
        assert "Small sample size" in result.limitations[0]
        assert result.used_full_text is False

    @patch("src.extractor.section_extractor.ChatOpenAI")
    def test_extract_uses_full_text_when_available(self, mock_chat):
        extractor = LimitationExtractor()
        extractor._chain = MagicMock()
        extractor._chain.invoke.return_value = {
            "limitations": ["Single institution data"],
            "research_gaps": [],
            "future_work": [],
            "methodological_weaknesses": [],
        }

        paper = PaperMetadata(
            pmid="88888",
            title="Full Text Paper",
            year="2022",
            abstract="Short abstract.",
            full_text=FULL_TEXT_WITH_SECTIONS,
        )

        result = extractor.extract(paper)
        assert result.used_full_text is True
        assert result.pmid == "88888"
