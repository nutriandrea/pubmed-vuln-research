"""
Pydantic models for paper metadata and raw content.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Structured representation of a PubMed paper."""

    pmid: str
    title: str
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: str = ""
    abstract: str = ""
    publication_types: list[str] = Field(default_factory=list)
    # Full text sections when available from PMC
    full_text: Optional[str] = None
    pmc_id: Optional[str] = None
    doi: Optional[str] = None
    # URL for reference
    pubmed_url: str = ""

    @property
    def citation(self) -> str:
        authors_short = (
            ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        )
        return f"{authors_short} ({self.year}). {self.title}. {self.journal}. PMID:{self.pmid}"
