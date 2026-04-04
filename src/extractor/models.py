"""
Output models for the extraction step.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LimitationCategory(str, Enum):
    DATASET = "dataset"
    METHODOLOGY = "methodology"
    EVALUATION = "evaluation"
    BIAS = "bias"
    OTHER = "other"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClassifiedLimitation(BaseModel):
    """Singola limitazione classificata"""
    text: str = Field(description="The limitation text")
    category: str = Field(description="Category: dataset, methodology, evaluation, bias, other")
    severity: str = Field(description="Severity: low, medium, high")


class ExtractedLimitations(BaseModel):
    """
    Structured limitations extracted from a single paper.
    This is the unit stored in the processed data directory and
    later passed to the knowledge processor.
    """

    pmid: str
    paper_title: str
    year: str
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    doi: str = ""
    pubmed_url: str = ""

    # Raw section text pulled from the paper (truncated to 500 chars)
    raw_limitation_text: str = ""

    # LLM-structured output
    limitations: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    methodological_weaknesses: list[str] = Field(default_factory=list)

    # True when we used full PMC text; False means abstract only
    used_full_text: bool = False
    
    # Method used for extraction:
    # - "full_text_sections": Used regex-extracted sections from PMC full text
    # - "abstract": Used abstract only
    # - "full_text_fallback": Used beginning of full text (no good sections found)
    # - "no_text": No text available for this paper
    extraction_method: str = "unknown"

    # Classification layer output: filtered limitations with category and severity
    classified_limitations: list[ClassifiedLimitation] = Field(default_factory=list)

    @property
    def all_weakness_points(self) -> list[str]:
        """Flat list of every extracted weakness point."""
        return (
            self.limitations
            + self.research_gaps
            + self.future_work
            + self.methodological_weaknesses
        )

    def to_summary_text(self) -> str:
        """Human-readable summary block for a paper."""
        lines = [
            f"Paper: {self.paper_title} ({self.year})",
            f"PMID: {self.pmid}",
            "",
        ]
        if self.limitations:
            lines.append("Limitations:")
            lines += [f"  - {l}" for l in self.limitations]
        if self.research_gaps:
            lines.append("Research Gaps:")
            lines += [f"  - {g}" for g in self.research_gaps]
        if self.future_work:
            lines.append("Future Work:")
            lines += [f"  - {f}" for f in self.future_work]
        if self.methodological_weaknesses:
            lines.append("Methodological Weaknesses:")
            lines += [f"  - {w}" for w in self.methodological_weaknesses]
        return "\n".join(lines)
