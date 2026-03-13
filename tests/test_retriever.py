"""
Unit tests for the PubMed retrieval module.
No real network calls — uses mocked Entrez responses.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from src.retriever.pubmed_client import PubMedClient, PubMedQueryParams


# ------------------------------------------------------------------ #
# PubMedQueryParams
# ------------------------------------------------------------------ #

class TestPubMedQueryParams:
    def test_basic_query(self):
        p = PubMedQueryParams(topic="breast cancer", date_from=2020, date_to=2025)
        q = p.build_query()
        assert "breast cancer" in q
        # Date range is now passed as Entrez params, NOT embedded in the query string
        assert "2020" not in q or "[pdat]" not in q

    def test_paper_type_review(self):
        p = PubMedQueryParams(
            topic="diabetes", date_from=2021, date_to=2024, paper_type="review"
        )
        q = p.build_query()
        assert "Review[Publication Type]" in q

    def test_unknown_paper_type_does_not_raise(self):
        p = PubMedQueryParams(topic="x", paper_type="unknown_type")
        q = p.build_query()  # should not raise; just omits the type filter
        assert "x" in q


# ------------------------------------------------------------------ #
# PubMedClient — parsing
# ------------------------------------------------------------------ #

MINIMAL_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">12345678</PMID>
      <Article>
        <ArticleTitle>Deep Learning for Breast Cancer Detection: A Review</ArticleTitle>
        <Abstract>
          <AbstractText>This study has several limitations including small dataset size.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>Medical Imaging Journal</Title>
          <JournalIssue>
            <PubDate><Year>2022</Year></PubDate>
          </JournalIssue>
        </Journal>
        <PublicationTypeList>
          <PublicationType UI="D016454">Review</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestPubMedClientParsing:
    def setup_method(self):
        self.client = PubMedClient(email="test@test.com")

    def test_parse_minimal_xml(self):
        paper = self.client._parse_pubmed_xml("12345678", MINIMAL_XML)
        assert paper is not None
        assert paper.pmid == "12345678"
        assert "Breast Cancer" in paper.title or "breast cancer" in paper.title.lower()
        assert paper.year == "2022"
        assert "Smith John" in paper.authors
        assert paper.journal == "Medical Imaging Journal"
        assert "limitations" in paper.abstract.lower()

    def test_parse_returns_none_on_bad_xml(self):
        paper = self.client._parse_pubmed_xml("999", b"<not valid xml>>>")
        assert paper is None

    @patch("src.retriever.pubmed_client.Entrez.esearch")
    @patch("src.retriever.pubmed_client.Entrez.read")
    def test_esearch_returns_pmids(self, mock_read, mock_esearch):
        mock_esearch.return_value = MagicMock()
        mock_read.return_value = {"IdList": ["11111111", "22222222"]}
        params = PubMedQueryParams(topic="breast cancer", date_from=2020, date_to=2025, max_results=5)
        pmids = self.client._esearch(params)
        assert pmids == ["11111111", "22222222"]
        # Verify date is passed as Entrez params, not in the term string
        call_kwargs = mock_esearch.call_args.kwargs
        assert call_kwargs.get("mindate") == "2020"
        assert call_kwargs.get("maxdate") == "2025"
        assert call_kwargs.get("datetype") == "pdat"
        assert "2020" not in call_kwargs.get("term", "")
