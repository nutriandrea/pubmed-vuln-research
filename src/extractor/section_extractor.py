"""
Module 2 — Text Extraction (Hybrid Approach Optimized)

Two-stage extraction strategy:
  1. Rule-based extraction of named sections (Discussion, Limitations, etc.)
     applied when full text (PMC) is available.
  2. LLM-based extraction applied to abstract or full text when structure
     is ambiguous or only abstract is available.

OPTIMIZATION: Text is truncated to ~2000 chars to reduce LLM tokens while
maintaining quality. Sections are prioritized: Limitations > Discussion > Abstract.

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
from src.extractor.limitation_classifier import LimitationClassifier

# ------------------------------------------------------------------ #
# Section header patterns (case-insensitive, comprehensive)
# ------------------------------------------------------------------ #
# Priority: higher priority sections are processed first
_SECTION_PATTERNS: list[tuple[int, str, re.Pattern]] = [
    # Priority 1: Limitations
    (1, "limitations", re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:study\s+)?(?:key\s+)?limitations?\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    (1, "strengths_and_limitations", re.compile(
        r"(?:^|\n)\s*(?:strengths?\s+and\s+limitations?)\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    (1, "limitations_of_the_study", re.compile(
        r"(?:^|\n)\s*(?:limitations?\s+of\s+(?:the\s+)?(?:study|research|analysis|method))\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    
    # Priority 2: Discussion (most papers have this)
    (2, "discussion", re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?discussion\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    (2, "results_and_discussion", re.compile(
        r"(?:^|\n)\s*(?:results?\s+(?:and|&)?\s*discussion)\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    (2, "discussion_and_conclusions", re.compile(
        r"(?:^|\n)\s*(?:discussion\s+(?:and|&)\s*conclusions?)\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    
    # Priority 3: Future work / perspectives
    (3, "future_work", re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:future\s+(?:work|directions?|research|perspectives?|implications?|recommendations?))\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    (3, "future_research", re.compile(
        r"(?:^|\n)\s*(?:(?:further|additional)\s+(?:research|study|work))\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    
    # Priority 4: Conclusion
    (4, "conclusion", re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?conclusions?\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    
    # Priority 5: Weaknesses / shortcomings
    (5, "weaknesses", re.compile(
        r"(?:^|\n)\s*(?:(?:methodological\s+)?weaknesses?|shortcomings?|criticisms?|challenges?)\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
    
    # Priority 6: Implications
    (6, "implications", re.compile(
        r"(?:^|\n)\s*(?:clinical\s+)?implications?\s*[:\-\.]?\s*\n",
        re.IGNORECASE
    )),
]

# Section terminators — stop collecting text when one of these is hit
_NEXT_SECTION = re.compile(
    r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:references?|bibliography|acknowledgements?|funding|"
    r"author\s+contributions?|conflicts?\s+of\s+interest|appendix|supplementary|"
    r"supplementary\s+(?:materials?|information|tables?|figures?)|"
    r"additional\s+(?:file|data)|"
    r"figure\s+(?:legends?|captions?))\s*[:\-\.]?\s*\n",
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


# Prompt for SHORT texts (optimized for ~500-2000 chars)
_SHORT_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a scientific literature analyst specializing in identifying
research weaknesses. The text provided is a SELECTED EXCERPT from a paper,
focused on Discussion/Limitations sections.

IMPORTANT: Extract ALL sentences that IMPLICATE a limitation, even if NOT explicitly stated.

Include these patterns (DO NOT require explicit keywords):
- Sample size: "small cohort", "limited number", "few patients", "relatively small"
- Design: "retrospective", "single-center", "observational", "single arm"
- Data: "lack of diversity", "limited dataset", "no validation", "heterogeneous"
- Follow-up: "short follow-up", "limited duration", "preliminary", "pilot"
- Methodology: "no control group", "underpowered", "limited statistical power"

Extract limitations even when expressed indirectly like:
- "This study was conducted on a relatively small cohort"
- "Only one center participated"
- "Data were collected retrospectively"
- "Further studies are needed"
- "Preliminary findings suggest"
- "This was a pilot study"

Extract from the provided text:
- Explicit limitations of the study
- Research gaps identified by the authors
- Suggested future work or research directions
- Methodological weaknesses or flaws

Rules:
- Be INCLUSIVE. Better to over-extract than miss relevant limitations.
- Extract implicit limitations that are clearly stated even without keywords.
- Quote or paraphrase the authors' own words when possible.
- Each item should be a complete, self-contained sentence.
- If a category has no explicit mentions, return an empty list.

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
        "Paper title: {title}\nPMID: {pmid}\n\nText to analyze:\n{text}\n\nReturn JSON only.",
    ),
])

# Prompt for LONG texts (fallback when excerpt is insufficient)
_LONG_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a scientific literature analyst specializing in identifying
research weaknesses. Your task is to extract from the provided paper text:

IMPORTANT: Extract ALL sentences that IMPLICATE a limitation, even if NOT explicitly stated.

Include these patterns (DO NOT require explicit keywords):
- Sample size: "small cohort", "limited number", "few patients", "relatively small"
- Design: "retrospective", "single-center", "observational", "single arm"
- Data: "lack of diversity", "limited dataset", "no validation", "heterogeneous"
- Follow-up: "short follow-up", "limited duration", "preliminary", "pilot"
- Methodology: "no control group", "underpowered", "limited statistical power"

Extract limitations even when expressed indirectly like:
- "This study was conducted on a relatively small cohort"
- "Only one center participated"
- "Data were collected retrospectively"
- "Further studies are needed"
- "Preliminary findings suggest"
- "This was a pilot study"

Extract from the provided text:
- Explicit limitations of the study
- Research gaps identified by the authors
- Suggested future work or research directions
- Methodological weaknesses or flaws

Rules:
- Be INCLUSIVE. Better to over-extract than miss relevant limitations.
- Focus on the DISCUSSION, LIMITATIONS, and CONCLUSION sections.
- Quote or paraphrase the authors' own words when possible.
- Each item should be a complete, self-contained sentence.
- If a category has no explicit mentions, return an empty list.

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
        "Paper title: {title}\nPMID: {pmid}\n\nText to analyze:\n{text}\n\nReturn JSON only.",
    ),
])


class LimitationExtractor:
    """
    Extracts limitations, research gaps, and methodological weaknesses
    from PaperMetadata objects using a hybrid approach.
    
    Optimization: Text is truncated to ~2000 chars to reduce LLM tokens.
    """

    # Max characters to pass to LLM (roughly 500 words)
    MAX_LLM_CHARS = 2000
    
    # Min characters to consider section as valid
    MIN_SECTION_CHARS = 50

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self._llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._short_chain = _SHORT_EXTRACT_PROMPT | self._llm | JsonOutputParser()
        self._long_chain = _LONG_EXTRACT_PROMPT | self._llm | JsonOutputParser()

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def extract(self, paper: PaperMetadata) -> ExtractedLimitations:
        """
        Main entry point: decide which text source to use, extract sections,
        then call the LLM with optimized text size.
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

        # Determine extraction method and text source
        extraction_text = ""
        extraction_method = "none"
        
        if paper.full_text:
            logger.debug("Using PMC full text for PMID {}", paper.pmid)
            sections = self._extract_sections_heuristic(paper.full_text)
            
            if sections and len(sections) >= self.MIN_SECTION_CHARS:
                # Good sections found - use them
                extraction_text = sections
                extraction_method = "full_text_sections"
            elif paper.abstract:
                # No good sections - use abstract
                extraction_text = paper.abstract
                extraction_method = "abstract"
            else:
                # No abstract either - use first part of full text
                extraction_text = paper.full_text[:3000]
                extraction_method = "full_text_fallback"
            
            base.used_full_text = True
        elif paper.abstract:
            logger.debug("Using abstract for PMID {}", paper.pmid)
            extraction_text = paper.abstract
            extraction_method = "abstract"
        else:
            logger.warning("No text available for PMID {}", paper.pmid)
            base.extraction_method = "no_text"
            return base
        
        base.extraction_method = extraction_method
        base.raw_limitation_text = extraction_text[:500]  # Store first 500 chars for reference
        
        # Truncate to max LLM chars
        if len(extraction_text) > self.MAX_LLM_CHARS:
            extraction_text = self._smart_truncate(extraction_text)
        
        logger.debug("=== EXTRACTION TEXT FOR PMID {} ({} chars, method: {}) ===\n{}",
                    paper.pmid, len(extraction_text), extraction_method, extraction_text[:500])

        if not extraction_text.strip():
            logger.warning("No text available for PMID {}", paper.pmid)
            return base

        structured = self._llm_extract(paper.title, paper.pmid, extraction_text, extraction_method)

        logger.debug("=== LLM EXTRACTED DATA FOR PMID {} ===\n{}",
                    paper.pmid, json.dumps(structured, indent=2))
        base.limitations = structured.get("limitations", [])
        base.research_gaps = structured.get("research_gaps", [])
        base.future_work = structured.get("future_work", [])
        base.methodological_weaknesses = structured.get("methodological_weaknesses", [])

        logger.info(
            "PMID {} [{}] → {} limitations, {} gaps, {} future work items",
            paper.pmid,
            extraction_method,
            len(base.limitations),
            len(base.research_gaps),
            len(base.future_work),
        )

        # Classification layer: filter and categorize limitations
        if extraction_text.strip():
            try:
                classifier = LimitationClassifier(model_name=self._llm.model_name)
                classified = classifier.classify(extraction_text)
                base.classified_limitations = classified.limitations
                logger.debug(
                    "Classified {} limitations for PMID {}",
                    len(classified.limitations),
                    paper.pmid
                )
            except Exception as e:
                logger.warning(f"Classification failed for PMID {paper.pmid}: {e}")

        return base

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _extract_sections_heuristic(self, text: str) -> str:
        """
        Pull named sections (Limitations, Discussion, etc.) from full text
        using regex. Returns concatenated section content, sorted by priority.
        """
        collected: list[tuple[int, str]] = []  # (priority, text)

        for priority, section_name, pattern in _SECTION_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            start = match.end()
            # Find where the next major section begins
            end_match = _NEXT_SECTION.search(text, start)
            end = end_match.start() if end_match else start + 3000
            section_text = text[start:end].strip()
            if section_text and len(section_text) >= self.MIN_SECTION_CHARS:
                collected.append((priority, f"[{section_name.upper()}]\n{section_text}"))
                logger.debug(
                    "Found section '{}' (priority={}, {} chars)", section_name, priority, len(section_text)
                )

        # Sort by priority and concatenate
        if not collected:
            return ""
        
        # Sort: lower priority number = higher importance
        collected.sort(key=lambda x: x[0])
        
        result = "\n\n".join(text for _, text in collected)
        logger.debug("Total extracted sections: {} chars", len(result))
        return result

    def _smart_truncate(self, text: str) -> str:
        """
        Smart truncation that tries to keep the most relevant parts.
        Prioritizes Limitations > Discussion > Conclusion sections.
        """
        # If text is short enough, return as-is
        if len(text) <= self.MAX_LLM_CHARS:
            return text
        
        # Try to find a good break point (end of sentence, paragraph)
        truncated = text[:self.MAX_LLM_CHARS]
        
        # Find the last sentence boundary
        last_period = truncated.rfind('. ')
        last_newline = truncated.rfind('\n\n')
        
        # Use the later of the two as break point
        break_point = max(last_period, last_newline)
        
        if break_point > self.MAX_LLM_CHARS * 0.7:  # Only use if we're at least 70% through
            truncated = truncated[:break_point + 1]
        
        logger.debug("Text truncated from {} to {} chars", len(text), len(truncated))
        return truncated

    def _llm_extract(self, title: str, pmid: str, text: str, method: str) -> dict:
        """
        Call GPT to extract structured limitations from text.
        Uses short prompt for section-based extraction, long prompt for fallback.
        """
        try:
            # Use appropriate chain based on method
            if method in ("full_text_sections", "abstract"):
                result = self._short_chain.invoke({
                    "title": title,
                    "pmid": pmid,
                    "text": text
                })
            else:
                # Full text fallback - use longer prompt
                # Further truncate to reduce tokens
                words = text.split()
                if len(words) > 800:
                    text = " ".join(words[:800])
                result = self._long_chain.invoke({
                    "title": title,
                    "pmid": pmid,
                    "text": text
                })
            
            # Validate that all expected keys exist
            for key in ("limitations", "research_gaps", "future_work", "methodological_weaknesses"):
                if key not in result:
                    result[key] = []
            return result
        except Exception as exc:
            logger.error("LLM extraction failed for PMID {}: {}", pmid, exc)
            return {
                "limitations": [],
                "research_gaps": [],
                "future_work": [],
                "methodological_weaknesses": [],
            }

    # ------------------------------------------------------------------ #
    # Batch Extraction (for optimized processing)
    # ------------------------------------------------------------------ #
    
    def extract_batch(self, papers: list[PaperMetadata], progress_callback=None) -> list[ExtractedLimitations]:
        """
        Extract limitations for a batch of papers using OPTIMIZED batch LLM.
        
        This method uses batch LLM calls to process papers 50 at a time,
        reducing API calls by 98% and time by 50x.
        
        Args:
            papers: List of PaperMetadata objects
            progress_callback: Optional callback(type, data) for progress updates
            
        Returns:
            List of ExtractedLimitations
        """
        from src.extractor.extraction_cache import ExtractionCache
        from config.settings import settings
        import re
        
        def slug(t): return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")
        
        # Get topic from first paper or use default
        topic = slug(papers[0].title[:50]) if papers else "batch"
        cache_dir = settings.processed_data_dir / topic
        
        cache = ExtractionCache(cache_dir)
        
        # Get cached and missing PMIDs
        pmids = [p.pmid for p in papers]
        cached, missing_pmids = cache.get_all(pmids)
        
        results = list(cached)  # Start with cached results
        
        if progress_callback:
            progress_callback("status", {
                "msg": f"💾 Cache: {len(cached)} hits, {len(missing_pmids)} to process with batch LLM"
            })
        
        if missing_pmids:
            # Build mapping for missing papers
            missing_papers = [p for p in papers if p.pmid in missing_pmids]
            
            # Use BATCH LLM extraction
            if progress_callback:
                progress_callback("progress", {
                    "percent": 0,
                    "msg": f"🚀 Starting batch LLM extraction ({len(missing_papers)} papers, ~{len(missing_papers)//50 + 1} batches)"
                })
            
            try:
                # Extract in batches using optimized batch LLM
                new_extractions = self.extract_batch_llm(missing_papers, batch_size=50)
                
                # Save to cache and add to results
                for extraction in new_extractions:
                    cache.set(extraction.pmid, extraction)
                    results.append(extraction)
                
                if progress_callback:
                    progress_callback("status", {
                        "msg": f"✅ Batch LLM extraction complete: {len(new_extractions)} papers processed"
                    })
                    
            except Exception as e:
                logger.error(f"Batch LLM failed: {e}. Using fallback individual extraction.")
                
                # Fallback to individual extraction
                for i, paper in enumerate(missing_papers):
                    extraction = self.extract(paper)
                    cache.set(paper.pmid, extraction)
                    results.append(extraction)
                    
                    if progress_callback and (i + 1) % 10 == 0:
                        percent = int(((i + 1) / len(missing_papers)) * 100)
                        progress_callback("progress", {
                            "percent": percent,
                            "msg": f"Extracted {i + 1}/{len(missing_papers)} papers (fallback mode)"
                        })
        
        if progress_callback:
            progress_callback("complete", {
                "total": len(results),
                "cached": len(cached),
                "new": len(results) - len(cached),
                "msg": f"✅ Complete: {len(results)} papers ({len(cached)} from cache)"
            })
        
        return results

    # ------------------------------------------------------------------ #
    # Batch LLM Extraction (HIGHLY OPTIMIZED)
    # ------------------------------------------------------------------ #
    
    def extract_batch_llm(self, papers: list[PaperMetadata], batch_size: int = 50) -> list[ExtractedLimitations]:
        """
        Extract limitations for multiple papers using BATCH LLM calls.
        
        This is the HIGHLY OPTIMIZED method for processing thousands of papers.
        Instead of 1 LLM call per paper, it processes 50 papers per call.
        
        Example:
            10,000 papers → 200 LLM calls (instead of 10,000)
            Time: ~30 minutes (instead of 14 hours)
        
        Args:
            papers: List of PaperMetadata objects
            batch_size: Number of papers per LLM call (default 50)
            
        Returns:
            List of ExtractedLimitations
        """
        results = []
        
        # Process in batches
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(papers) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} papers)")
            
            # Extract sections for batch (local, fast)
            batch_data = []
            for paper in batch:
                extraction_text = ""
                
                if paper.full_text:
                    sections = self._extract_sections_heuristic(paper.full_text)
                    if sections and len(sections) >= self.MIN_SECTION_CHARS:
                        extraction_text = sections
                    elif paper.abstract:
                        extraction_text = paper.abstract
                    else:
                        extraction_text = paper.full_text[:3000]
                elif paper.abstract:
                    extraction_text = paper.abstract
                
                # Truncate
                if len(extraction_text) > self.MAX_LLM_CHARS:
                    extraction_text = self._smart_truncate(extraction_text)
                
                batch_data.append({
                    "pmid": paper.pmid,
                    "title": paper.title,
                    "text": extraction_text
                })
            
            # Single LLM call for entire batch
            try:
                batch_result = self._llm_extract_batch(batch_data)
                results.extend(batch_result)
            except Exception as e:
                logger.error(f"Batch LLM extraction failed: {e}. Falling back to individual extraction.")
                # Fallback to individual extraction for this batch
                for paper in batch:
                    results.append(self.extract(paper))
        
        return results
    
    def _llm_extract_batch(self, batch_data: list[dict]) -> list[ExtractedLimitations]:
        """
        Call LLM once for multiple papers.
        
        Formats all papers into a structured prompt and extracts limitations for each.
        """
        # Build combined prompt
        prompt_parts = []
        for i, item in enumerate(batch_data):
            prompt_parts.append(f"""### PAPER {i+1}
PMID: {item['pmid']}
Title: {item['title']}

Text:
{item['text'][:1500]}

Extract limitations, gaps, future work, and weaknesses from this paper.""")

        combined_prompt = "\n".join(prompt_parts)
        
        # Create a simple extraction prompt
        batch_prompt = f"""You are analyzing {len(batch_data)} scientific papers to extract research limitations.

IMPORTANT: Extract ALL sentences that IMPLICATE a limitation, even if NOT explicitly stated.

Include these patterns (DO NOT require explicit keywords):
- Sample size: "small cohort", "limited number", "few patients", "relatively small"
- Design: "retrospective", "single-center", "observational", "single arm"
- Data: "lack of diversity", "limited dataset", "no validation", "heterogeneous"
- Follow-up: "short follow-up", "limited duration", "preliminary", "pilot"
- Methodology: "no control group", "underpowered", "limited statistical power"

Extract limitations even when expressed indirectly like:
- "This study was conducted on a relatively small cohort"
- "Only one center participated"
- "Data were collected retrospectively"
- "Further studies are needed"
- "Preliminary findings suggest"
- "This was a pilot study"

For each paper below, extract:
- limitations: Study limitations (explicit AND implicit)
- research_gaps: Gaps in the research identified by authors
- future_work: Suggested future research directions
- methodological_weaknesses: Methodological flaws (explicit AND implicit)

Be INCLUSIVE. Better to over-extract than miss relevant limitations.

Papers:
{combined_prompt}

Return a JSON array with {len(batch_data)} objects, one per paper:
[{{"pmid": "...", "limitations": [...], "research_gaps": [...], "future_work": [...], "methodological_weaknesses": [...]}}]

Return ONLY the JSON array, no other text."""

        try:
            # Call LLM once for entire batch
            response = self._llm.invoke(batch_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                extractions = json.loads(json_match.group(0))
                
                # Build ExtractedLimitations objects
                results = []
                pmid_to_paper = {item['pmid']: item for item in batch_data}
                
                for ext in extractions:
                    pmid = ext.get('pmid', '')
                    if pmid in pmid_to_paper:
                        paper_info = pmid_to_paper[pmid]
                        results.append(ExtractedLimitations(
                            pmid=pmid,
                            paper_title=paper_info['title'],
                            year="",
                            authors=[],
                            journal="",
                            doi="",
                            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            limitations=ext.get('limitations', []),
                            research_gaps=ext.get('research_gaps', []),
                            future_work=ext.get('future_work', []),
                            methodological_weaknesses=ext.get('methodological_weaknesses', []),
                            extraction_method="batch_llm"
                        ))
                
                return results
            else:
                logger.error("Failed to parse batch LLM response")
                return []
                
        except Exception as e:
            logger.error(f"Batch LLM call failed: {e}")
            raise
