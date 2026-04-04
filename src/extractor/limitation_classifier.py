"""
Classification Layer - Filtra il testo per tenere solo vere limitazioni

Input: testo estratto da Discussion/Limitations
Output: limitazioni classificate per categoria e severità

Categories:
- dataset: limitazioni relative ai dati (sample size, dataset bias, etc.)
- methodology: debolezze metodologiche (study design, controls, etc.)
- evaluation: problemi di valutazione/metriche
- bias: bias di selezione, pubblicazione, etc.
- other: altre limitazioni

Severity:
- low, medium, high
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.logger import logger


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


class ClassifiedLimitations(BaseModel):
    """Output della classificazione"""
    limitations: list[ClassifiedLimitation] = Field(default_factory=list, description="Filtered limitations")
    discarded: list[str] = Field(default_factory=list, description="Sentences that are not limitations")


_CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are a scientific research reviewer specialized in identifying limitations in academic papers.

Your task is to analyze the given text and extract ONLY sentences that describe REAL limitations, weaknesses, biases, or research gaps.

IGNORE (do NOT extract):
- Results and findings
- Interpretations of results
- General discussion points
- Background information
- Conclusions that don't mention limitations
- Positive statements about the study
- Suggestions that are not explicitly about limitations

CATEGORIES (assign one to each limitation):
- dataset: limitations related to data (sample size, data quality, data availability, representativeness)
- methodology: methodological weaknesses (study design, lack of controls, statistical issues, measurement problems)
- evaluation: evaluation limitations (missing validation, inadequate metrics, lack of comparison)
- bias: biases (selection bias, publication bias, confirmation bias, etc.)
- other: limitations that don't fit above categories

SEVERITY (assess impact):
- high: fundamental flaws that seriously affect validity
- medium: notable limitations that affect interpretation
- low: minor limitations or suggestions for future work

Return a JSON object with:
1. "limitations": array of objects with "text", "category", "severity"
2. "discarded": array of sentence strings that were NOT limitations

If no limitations are found, return an empty limitations array.

TEXT TO ANALYZE:
---
{text}
---

Respond with JSON only, no other text."""
)


class LimitationClassifier:
    """
    Classification layer che filtra il testo per tenere solo vere limitazioni.
    
    Usage:
        classifier = LimitationClassifier()
        result = classifier.classify(discussion_text)
        for lim in result.limitations:
            print(f"[{lim.category}] ({lim.severity}) {lim.text}")
    """

    MAX_TEXT_CHARS = 3000

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> None:
        self._llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._chain = _CLASSIFY_PROMPT | self._llm | JsonOutputParser()

    def classify(self, text: str) -> ClassifiedLimitations:
        """
        Classifica il testo e restituisce solo le vere limitazioni.
        
        Args:
            text: Testo da analysare (Discussion, Limitations, etc.)
            
        Returns:
            ClassifiedLimitations con limitazioni filtrate e categorizzate
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to classifier")
            return ClassifiedLimitations()

        # Truncate to max chars
        text_to_analyze = text[:self.MAX_TEXT_CHARS]
        
        logger.debug("Classifying text ({} chars)", len(text_to_analyze))

        try:
            result = self._chain.invoke({"text": text_to_analyze})
            
            if not result:
                logger.warning("Classifier returned empty result")
                return ClassifiedLimitations()

            # Validate and return
            classified = ClassifiedLimitations.model_validate(result)
            
            logger.info(
                "Classified: {} limitations, {} discarded",
                len(classified.limitations),
                len(classified.discarded)
            )
            
            return classified

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassifiedLimitations()

    def classify_batch(self, texts: list[str]) -> list[ClassifiedLimitations]:
        """
        Classifica multiple testi in batch.
        
        Args:
            texts: Lista di testi da classificare
            
        Returns:
            Lista di ClassifiedLimitations
        """
        results = []
        for text in texts:
            result = self.classify(text)
            results.append(result)
        return results


def get_category_stats(classified: ClassifiedLimitations) -> dict:
    """
    Statistiche sulle categorie di limitazioni trovate.
    
    Returns:
        Dict con count per categoria e severità
    """
    stats = {
        "by_category": {c.value: 0 for c in LimitationCategory},
        "by_severity": {s.value: 0 for s in Severity},
        "total": len(classified.limitations),
        "discarded": len(classified.discarded),
    }
    
    for lim in classified.limitations:
        cat = lim.category.lower()
        sev = lim.severity.lower()
        
        if cat in stats["by_category"]:
            stats["by_category"][cat] += 1
        if sev in stats["by_severity"]:
            stats["by_severity"][sev] += 1
    
    return stats
