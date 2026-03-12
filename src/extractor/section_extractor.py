"""
Module 2 — Text Extraction

Two-stage extraction strategy:
  1. Regex/heuristic extraction of named sections (Discussion, Limitations, etc.)
     applied when full text (PMC) is available.
  2. LLM-based extraction applied to abstract or full text when structure
     is ambiguous or only abstract is available.

Public API
----------
LimitationExtractor.extract(paper: PaperMetadata) -> ExtractedLimitations
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.logger import logger
from src.retriever.models import PaperMetadata
from src.extractor.models import ExtractedLimitations

# ------------------------------------------------------------------ #
# Section header patterns (case-insensitive)
# ------------------------------------------------------------------ #
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("limitations", re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:study\s+)?limitations?\s*\n",
        re.IGNORECASE
    )),
    ("discussion", re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?discussion\s*\n",
        re.IGNORECASE
    )),
    ("future_work", re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:future\s+(?:work|directions?|research|perspectives?))\s*\n",
        re.IGNORECASE
    )),
    ("conclusion", re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?conclusions?\s*\n",
        re.IGNORECASE
    )),
]

# Section terminators — stop collecting text when one of these is hit
_NEXT_SECTION = re.compile(
    r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:references?|bibliography|acknowledgements?|funding|"
    r"author\s+contributions?|conflicts?\s+of\s+interest|appendix|supplementary)\s*\n",
    re.IGNORECASE,
)

# ------------------------------------------------------------------ #
# LLM output schema
# ------------------------------------------------------------------ #
class _LLMExtractionSchema(BaseModel):
    limitations: list[str] = Field(
        default_factory=list,
        description="Explicit limitations of the study"
    )
    research_gaps: list[str] = Field(
        default_factory=list,
        description="Identified gaps in existing research"
    )
    future_work: list[str] = Field(
        default_factory=list,
        description="Suggested future research directions"
    )
    methodological_weaknesses: list[str] = Field(
        default_factory=list,
        description="Methodological flaws or weaknesses"
    )


_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a scientific literature analyst specializing in identifying
research weaknesses. Your task is to extract ONLY the following from the
provided paper text:
- Explicit limitations of the study
- Research gaps identified by the authors
- Suggested future work or research directions
- Methodological weaknesses or flaws

Rules:
- Do NOT summarize strengths or contributions.
- Do NOT invent limitations not present in the text.
- Be specific: quote or closely paraphrase the authors' own words.
- Each item in the lists should be a complete, self-contained sentence.
- Return valid JSON matching the schema exactly.

Schema:
{{
  "limitations": ["..."],
  "research_gaps": ["..."],
  "future_work": ["..."],
  "methodological_weaknesses": ["..."]
}}""",
    ),
    (
        "human",
        "Paper title: {title}\n\nText to analyze:\n{text}\n\nReturn JSON only.",
    ),
])


class LimitationExtractor:
    """
    Extracts limitations, research gaps, and methodological weaknesses
    from PaperMetadata objects.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self._llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._chain = _EXTRACTION_PROMPT | self._llm | JsonOutputParser()

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def extract(self, paper: PaperMetadata) -> ExtractedLimitations:
        """
        Main entry point: decide which text source to use, extract sections,
        then call the LLM.
        """
        base = ExtractedLimitations(
            pmid=paper.pmid,
            paper_title=paper.title,
            year=paper.year,
            authors=paper.authors,
            journal=paper.journal,
            doi=paper.doi or "",
            pubmed_url=paper.pubmed_url,
        )

        if paper.full_text:
            logger.debug("Using PMC full text for PMID {}", paper.pmid)
            section_text = self._extract_sections_heuristic(paper.full_text)
            # Fall back to full text if heuristic found nothing meaningful
            if len(section_text) < 200:
                section_text = paper.full_text[:6000]  # cap to save tokens
            base.used_full_text = True
        else:
            logger.debug("Using abstract for PMID {}", paper.pmid)
            section_text = paper.abstract

        base.raw_limitation_text = section_text

        if not section_text.strip():
            logger.warning("No text available for PMID {}", paper.pmid)
            return base

        structured = self._llm_extract(paper.title, section_text)
        base.limitations = structured.get("limitations", [])
        base.research_gaps = structured.get("research_gaps", [])
        base.future_work = structured.get("future_work", [])
        base.methodological_weaknesses = structured.get("methodological_weaknesses", [])

        logger.info(
            "PMID {} → {} limitations, {} gaps, {} future work items",
            paper.pmid,
            len(base.limitations),
            len(base.research_gaps),
            len(base.future_work),
        )
        return base

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _extract_sections_heuristic(self, text: str) -> str:
        """
        Pull named sections (Limitations, Discussion, etc.) from full text
        using regex. Returns concatenated section content.
        """
        collected: list[str] = []

        for section_name, pattern in _SECTION_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            start = match.end()
            # Find where the next major section begins
            end_match = _NEXT_SECTION.search(text, start)
            end = end_match.start() if end_match else start + 4000
            section_text = text[start:end].strip()
            if section_text:
                collected.append(f"[{section_name.upper()}]\n{section_text}")
                logger.debug(
                    "Heuristic found section '{}' ({} chars)", section_name, len(section_text)
                )

        return "\n\n".join(collected)

    def _llm_extract(self, title: str, text: str) -> dict:
        """
        Call GPT to extract structured limitations from text.
        Returns a dict matching _LLMExtractionSchema.
        """
        # Trim text to ~3500 words to stay within context
        words = text.split()
        if len(words) > 3500:
            text = " ".join(words[:3500])
            logger.debug("Text trimmed to 3500 words for PMID extraction")

        try:
            result = self._chain.invoke({"title": title, "text": text})
            # Validate that all expected keys exist
            for key in ("limitations", "research_gaps", "future_work", "methodological_weaknesses"):
                if key not in result:
                    result[key] = []
            return result
        except Exception as exc:
            logger.error("LLM extraction failed: {}", exc)
            return {
                "limitations": [],
                "research_gaps": [],
                "future_work": [],
                "methodological_weaknesses": [],
            }
