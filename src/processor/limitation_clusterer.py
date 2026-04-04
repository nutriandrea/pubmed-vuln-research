"""
Semantic Deduplication Module

Cluster similar limitations together to identify common themes.
This transforms raw extractions into aggregated insights.

Output:
- cluster representative
- frequency count
- examples from different papers
- normalized text

Usage:
    clusterer = LimitationClusterer()
    clusters = clusterer.cluster(limitations, similarity_threshold=0.85)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.logger import logger


class LimitationCluster:
    """A cluster of similar limitations"""
    
    def __init__(
        self,
        cluster_id: int,
        representative: str,
        normalized: str,
        frequency: int = 0,
    ):
        self.cluster_id = cluster_id
        self.representative = representative
        self.normalized = normalized
        self.frequency = frequency
        self.examples: list[str] = []
        self.paper_ids: list[str] = []
        self.categories: dict[str, int] = {}
        self.severities: dict[str, int] = {}
    
    def add_example(
        self,
        text: str,
        pmid: str,
        category: str = "other",
        severity: str = "medium",
    ):
        self.examples.append(text)
        self.paper_ids.append(pmid)
        self.categories[category] = self.categories.get(category, 0) + 1
        self.severities[severity] = self.severities.get(severity, 0) + 1
    
    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "representative": self.representative,
            "normalized": self.normalized,
            "frequency": self.frequency,
            "examples": self.examples[:5],  # Keep top 5 examples
            "papers": list(set(self.paper_ids))[:10],  # Unique PMIDs
            "categories": self.categories,
            "severities": self.severities,
        }


class LimitationClusterer:
    """
    Cluster similar limitations using embedding + cosine similarity.
    
    Uses agglomerative clustering for better handling of varying cluster sizes.
    """
    
    SIMILARITY_THRESHOLD = 0.85
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.85,
    ):
        self._embeddings = OpenAIEmbeddings(model=embedding_model)
        self._threshold = similarity_threshold
    
    def cluster(
        self,
        limitations: list[dict],
        min_cluster_size: int = 2,
    ) -> list[LimitationCluster]:
        """
        Cluster limitations by semantic similarity.
        
        Args:
            limitations: List of dicts with keys: text, pmid, category, severity
            min_cluster_size: Minimum items to form a cluster
            
        Returns:
            List of LimitationCluster sorted by frequency
        """
        if not limitations:
            return []
        
        # Prepare texts for embedding
        texts = [self._normalize(l["text"]) for l in limitations]
        
        # Embed all texts
        logger.info("Embedding {} limitations for clustering", len(texts))
        embeddings = self._embeddings.embed_documents(texts)
        embeddings = np.array(embeddings)
        
        # Compute cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Agglomerative clustering
        clusters = self._agglomerative_cluster(
            similarity_matrix,
            min_cluster_size=min_cluster_size,
        )
        
        # Build cluster objects
        result = []
        for cluster_id, indices in enumerate(clusters):
            if len(indices) < min_cluster_size:
                continue
            
            # Find representative (most central text)
            cluster_embeddings = embeddings[indices]
            mean_embedding = np.mean(cluster_embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            
            # Find closest to mean
            similarities = np.dot(cluster_embeddings, mean_embedding)
            rep_idx = indices[np.argmax(similarities)]
            
            cluster = LimitationCluster(
                cluster_id=cluster_id,
                representative=texts[rep_idx],
                normalized=self._normalize(limitations[rep_idx]["text"]),
                frequency=len(indices),
            )
            
            # Add all examples
            for idx in indices:
                lim = limitations[idx]
                cluster.add_example(
                    text=lim["text"],
                    pmid=lim.get("pmid", ""),
                    category=lim.get("category", "other"),
                    severity=lim.get("severity", "medium"),
                )
            
            result.append(cluster)
        
        # Sort by frequency
        result.sort(key=lambda c: c.frequency, reverse=True)
        
        logger.info(
            "Clustered {} limitations into {} clusters",
            len(limitations),
            len(result)
        )
        
        return result
    
    def _agglomerative_cluster(
        self,
        similarity_matrix: np.ndarray,
        min_cluster_size: int = 2,
    ) -> list[list[int]]:
        """
        Simple agglomerative clustering using average linkage.
        """
        n = len(similarity_matrix)
        clusters = [[i] for i in range(n)]
        
        while len(clusters) > 1:
            # Find closest pair of clusters
            max_sim = -1
            merge = (0, 1)
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    sim = self._cluster_similarity(
                        similarity_matrix,
                        clusters[i],
                        clusters[j],
                    )
                    if sim > max_sim:
                        max_sim = sim
                        merge = (i, j)
            
            # Stop if below threshold
            if max_sim < self._threshold:
                break
            
            # Merge clusters
            i, j = merge
            clusters[i].extend(clusters[j])
            del clusters[j]
        
        # Filter small clusters
        return [c for c in clusters if len(c) >= min_cluster_size]
    
    def _cluster_similarity(
        self,
        sim_matrix: np.ndarray,
        cluster_a: list[int],
        cluster_b: list[int],
    ) -> float:
        """Compute average similarity between two clusters."""
        similarities = []
        for a in cluster_a:
            for b in cluster_b:
                similarities.append(sim_matrix[a, b])
        return np.mean(similarities) if similarities else 0
    
    def _normalize(self, text: str) -> str:
        """Normalize text for better matching."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Common normalizations
        replacements = {
            r'\bsmall\s+sample\s+size\b': 'small sample',
            r'\blimited\s+sample\s+size\b': 'small sample',
            r'\bsample\s+size\s+was\s+small\b': 'small sample',
            r'\bsmall\s+sample\s+cohort\b': 'small sample',
            r'\blimited\s+dataset\b': 'limited data',
            r'\bsmall\s+dataset\b': 'small dataset',
            r'\black\s+of\s+external\s+validation\b': 'no external validation',
            r'\black\s+of\s+validation\b': 'no validation',
            r'\bsingle\s+center\b': 'single center',
            r'\bsingle\s+institution\b': 'single center',
            r'\bretrospective\b': 'retrospective design',
            r'\bprospective\b': 'prospective design',
            r'\bselection\s+bias\b': 'selection bias',
            r'\bconfounding\b': 'confounding factors',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def save_clusters(
        self,
        clusters: list[LimitationCluster],
        output_path: Path,
    ):
        """Save clusters to JSON file."""
        data = [c.to_dict() for c in clusters]
        
        with open(output_path, 'w') as f:
            json.dump({
                "total_clusters": len(clusters),
                "total_mentions": sum(c.frequency for c in clusters),
                "clusters": data,
            }, f, indent=2)
        
        logger.info("Saved {} clusters to {}", len(clusters), output_path)


def extract_limitations_for_clustering(
    extracted_list: list,
) -> list[dict]:
    """
    Extract limitation data from ExtractedLimitations objects for clustering.
    
    Args:
        extracted_list: List of ExtractedLimitations
        
    Returns:
        List of dicts with text, pmid, category, severity
    """
    limitations = []
    
    for extracted in extracted_list:
        # From classified limitations (primary source)
        for classified in extracted.classified_limitations:
            limitations.append({
                "text": classified.text,
                "pmid": extracted.pmid,
                "paper_title": extracted.paper_title,
                "category": classified.category,
                "severity": classified.severity,
            })
        
        # From raw limitations (fallback)
        for text in extracted.limitations:
            limitations.append({
                "text": text,
                "pmid": extracted.pmid,
                "paper_title": extracted.paper_title,
                "category": "limitation",
                "severity": "medium",
            })
        
        # From methodological weaknesses
        for text in extracted.methodological_weaknesses:
            limitations.append({
                "text": text,
                "pmid": extracted.pmid,
                "paper_title": extracted.paper_title,
                "category": "methodology",
                "severity": "medium",
            })
    
    return limitations
