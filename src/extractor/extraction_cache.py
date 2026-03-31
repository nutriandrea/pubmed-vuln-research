"""
Extraction Cache Module

Provides caching for extracted limitations to avoid re-processing papers.
Cache is stored as JSON files in the data/processed directory.

Usage:
    cache = ExtractionCache(cache_dir)
    
    # Check cache for multiple PMIDs
    cached, missing = cache.get_all([pmid1, pmid2, pmid3])
    
    # Process missing papers
    new_extractions = extractor.extract_batch(missing)
    
    # Save to cache
    for ext in new_extractions:
        cache.set(ext.pmid, ext)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.logger import logger
from src.extractor.models import ExtractedLimitations


class ExtractionCache:
    """
    Cache for extracted limitations.
    
    Stores extracted data as JSON files to avoid re-processing papers.
    Thread-safe for concurrent access.
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "extraction_index.json"
        self._lock_file = self.cache_dir / ".lock"
    
    def get(self, pmid: str) -> Optional[ExtractedLimitations]:
        """
        Retrieve extracted limitations from cache.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            ExtractedLimitations if cached, None otherwise
        """
        cache_file = self._get_cache_file(pmid)
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                return ExtractedLimitations(**data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load cache for PMID {pmid}: {e}")
                return None
        return None
    
    def set(self, pmid: str, extraction: ExtractedLimitations):
        """
        Save extracted limitations to cache.
        
        Args:
            pmid: PubMed ID
            extraction: ExtractedLimitations object
        """
        cache_file = self._get_cache_file(pmid)
        try:
            with open(cache_file, 'w') as f:
                json.dump(extraction.model_dump(), f, indent=2)
            
            # Update index
            self._add_to_index(pmid, str(cache_file))
            logger.debug(f"Cached extraction for PMID {pmid}")
        except Exception as e:
            logger.error(f"Failed to save cache for PMID {pmid}: {e}")
    
    def get_all(self, pmids: list[str]) -> tuple[list[ExtractedLimitations], list[str]]:
        """
        Separate cached and missing PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            Tuple of (cached_extractions, missing_pmids)
        """
        cached = []
        missing = []
        
        for pmid in pmids:
            result = self.get(pmid)
            if result:
                cached.append(result)
            else:
                missing.append(pmid)
        
        logger.info(f"Cache: {len(cached)} hits, {len(missing)} misses")
        return cached, missing
    
    def has(self, pmid: str) -> bool:
        """Check if PMID is cached."""
        return self._get_cache_file(pmid).exists()
    
    def clear(self):
        """Clear all cached extractions."""
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name != "extraction_index.json":
                cache_file.unlink()
        if self._index_file.exists():
            self._index_file.unlink()
        logger.info("Extraction cache cleared")
    
    def size(self) -> int:
        """Return number of cached extractions."""
        return len(list(self.cache_dir.glob("*.json"))) - 1  # Exclude index
    
    def get_stats(self) -> dict:
        """Return cache statistics."""
        index = self._load_index()
        return {
            "total_cached": len(index),
            "cache_dir": str(self.cache_dir),
        }
    
    def _get_cache_file(self, pmid: str) -> Path:
        """Get cache file path for PMID."""
        return self.cache_dir / f"{pmid}.json"
    
    def _load_index(self) -> dict[str, str]:
        """Load cache index."""
        if self._index_file.exists():
            try:
                with open(self._index_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception):
                return {}
        return {}
    
    def _save_index(self, index: dict[str, str]):
        """Save cache index."""
        with open(self._index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _add_to_index(self, pmid: str, path: str):
        """Add entry to cache index."""
        index = self._load_index()
        index[pmid] = path
        self._save_index(index)
