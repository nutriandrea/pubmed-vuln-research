"""
Module for expanding search queries with synonyms and similar terms.

This module provides query expansion using:
1. A biomedical synonym dictionary
2. Simple LLM-based expansion (optional)
3. Manual synonym definitions

Usage:
    expander = SynonymExpander()
    expanded = expander.expand_query("breast cancer", field="Title")
    # Returns: ("breast cancer"[Title] OR "breast tumor"[Title] OR ...)
"""

from __future__ import annotations

import re
from typing import Optional, List, Tuple


class SynonymExpander:
    """
    Expands search queries with biomedical synonyms and similar terms.

    Uses a predefined dictionary of common biomedical synonyms.
    Can be extended with LLM-based expansion if needed.
    """

    def __init__(self) -> None:
        # Biomedical synonym dictionary
        # Maps common terms to their synonyms
        self._synonym_map: dict[str, list[str]] = {
            # Cancer terms
            "breast cancer": ["breast tumor", "mammary carcinoma", "breast neoplasm"],
            "cancer": ["tumor", "neoplasm", "carcinoma", "malignancy"],
            "lung cancer": ["lung tumor", "pulmonary carcinoma", "lung neoplasm"],
            "prostate cancer": ["prostate tumor", "prostatic carcinoma"],
            "colorectal cancer": ["colon cancer", "rectal cancer", "colorectal tumor"],

            # Disease terms
            "diabetes": ["diabetes mellitus", "type 2 diabetes", "type 1 diabetes"],
            "heart disease": ["cardiovascular disease", "cardiac disease"],
            "alzheimer": ["alzheimer's disease", "dementia"],

            # Technology terms
            "deep learning": ["neural network", "machine learning", "AI", "artificial intelligence"],
            "machine learning": ["statistical learning", "predictive modeling"],
            "neural network": ["deep learning", "CNN", "RNN", "transformer"],
            "artificial intelligence": ["AI", "machine learning", "deep learning"],

            # Research methodology terms
            "clinical trial": ["randomized controlled trial", "RCT", "clinical study"],
            "systematic review": ["meta-analysis", "evidence synthesis"],
            "randomized controlled trial": ["RCT", "clinical trial"],
            "cohort study": ["longitudinal study", "observational study"],

            # Imaging terms
            "mri": ["magnetic resonance imaging", "magnetic resonance"],
            "ct scan": ["computed tomography", "cat scan"],
            "ultrasound": ["sonography", "echography"],

            # Treatment terms
            "chemotherapy": ["chemo", "cytotoxic therapy"],
            "immunotherapy": ["immune therapy", "biological therapy"],
            "radiation therapy": ["radiotherapy", "radiation treatment"],
            
            # Eye/corneal terms
            "fuchs endothelial dystrophy": ["fuchs' endothelial dystrophy", "fuchs dystrophy"],
            "corneal endothelial decompensation": ["corneal endothelial failure"],
            "pseudophakic bullous keratopathy": ["pbk"],
        }

    def expand_query(
        self,
        query: str,
        field: str = "Title",
        use_llm: bool = False,
    ) -> str:
        """
        Expand a search query with synonyms.

        Parameters
        ----------
        query : str
            The search query (e.g., "breast cancer")
        field : str
            The PubMed field to search (e.g., "Title", "Abstract")
        use_llm : bool
            Whether to use LLM for expansion (not implemented yet)

        Returns
        -------
        str
            The expanded query in PubMed format, e.g.,
            ("breast cancer"[Title] OR "breast tumor"[Title] OR ...)
        """
        # Clean and normalize the query
        normalized_query = self._normalize_term(query)

        # Get synonyms for the query
        synonyms = self._get_synonyms(normalized_query)

        # Build the expanded query
        if synonyms:
            # Combine query and synonyms with OR
            all_terms = [normalized_query] + synonyms
            or_clauses = [f'"{term}"[{field}]' for term in all_terms]
            expanded = f"({' OR '.join(or_clauses)})"
        else:
            # No synonyms found, just use the original query
            expanded = f'"{normalized_query}"[{field}]'

        return expanded

    def expand_combined_query(
        self,
        main_topic: str,
        method: Optional[str] = None,
        field: str = "Title",
        exclude_terms: Optional[List[str]] = None,
    ) -> str:
        """
        Expand a combined search query with multiple topics and methods.

        Supports:
        - Single term: "cancer" -> "cancer"[Title] OR synonyms
        - OR terms: "term1 OR term2" -> (term1 OR term2)[Title]
        - Comma-separated: "term1, term2" -> (term1 OR term2)[Title]

        Parameters
        ----------
        main_topic : str
            The main research topic (e.g., "breast cancer" or "term1 OR term2")
        method : str, optional
            The method/technology (e.g., "deep learning")
        field : str
            The PubMed field to search
        exclude_terms : list[str], optional
            Terms to exclude with NOT operator

        Returns
        -------
        str
            The combined expanded query
        """
        # Parse the main topic for OR operators or comma-separated terms
        topic_expanded = self._parse_topic(main_topic, field=field)

        # Build the query parts
        query_parts = [topic_expanded]

        # Add method if provided
        if method:
            method_expanded = self._parse_topic(method, field=field)
            query_parts.append(method_expanded)

        # Add exclusions if provided
        if exclude_terms:
            for term in exclude_terms:
                normalized = self._normalize_term(term)
                query_parts.append(f'NOT "{normalized}"[{field}]')

        # Combine with AND
        return " AND ".join(query_parts)

    def _parse_topic(self, topic: str, field: str = "Title") -> str:
        """
        Parse a topic string and expand it appropriately.
        
        Handles:
        - Single term: "cancer" -> expands with synonyms
        - OR terms: "term1 OR term2" -> (term1 OR term2)
        - Comma-separated: "term1, term2" -> (term1 OR term2)
        """
        # Collect all unique terms to avoid duplicates
        all_terms = set()
        
        # Check if it contains OR operator
        if " OR " in topic.upper():
            # Split by OR (case-insensitive)
            terms = re.split(r'\s+OR\s+', topic, flags=re.IGNORECASE)
            for term in terms:
                term = term.strip()
                if term:
                    # Get the term and its synonyms
                    normalized = self._normalize_term(term)
                    all_terms.add(normalized)
                    synonyms = self._get_synonyms(normalized)
                    for syn in synonyms:
                        all_terms.add(syn)
        
        # Check if it contains comma (treated as OR)
        elif ',' in topic:
            # Split by comma
            terms = [t.strip() for t in topic.split(',') if t.strip()]
            for term in terms:
                normalized = self._normalize_term(term)
                all_terms.add(normalized)
                synonyms = self._get_synonyms(normalized)
                for syn in synonyms:
                    all_terms.add(syn)
        
        # Single term - expand with synonyms
        else:
            normalized = self._normalize_term(topic)
            all_terms.add(normalized)
            synonyms = self._get_synonyms(normalized)
            for syn in synonyms:
                all_terms.add(syn)
        
        # Build the OR clause with unique terms
        if all_terms:
            or_clauses = [f'"{term}"[{field}]' for term in sorted(all_terms)]
            return f"({' OR '.join(or_clauses)})"
        
        return f'"{topic}"[{field}]'

    def _normalize_term(self, term: str) -> str:
        """Normalize a search term."""
        # Remove extra whitespace
        term = " ".join(term.strip().split())
        # Convert to lowercase for lookup
        return term.lower()

    def _get_synonyms(self, term: str) -> List[str]:
        """
        Get synonyms for a term from the dictionary.

        Uses both exact matches and partial matches.
        """
        synonyms = []

        # Try exact match first
        if term in self._synonym_map:
            synonyms.extend(self._synonym_map[term])

        # Try partial matches (e.g., "cancer" in "breast cancer")
        for key, values in self._synonym_map.items():
            if key != term and key in term:
                synonyms.extend(values)

        # Remove duplicates and the original term
        synonyms = list(set(synonyms))
        synonyms = [s for s in synonyms if s.lower() != term.lower()]

        return synonyms

    def get_all_synonyms_for_term(self, term: str) -> List[str]:
        """
        Get all synonyms for a specific term.
        Useful for debugging and logging.
        """
        return self._get_synonyms(self._normalize_term(term))
