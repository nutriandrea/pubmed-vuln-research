"""
Module 1 — PubMed Data Retrieval

Uses Biopython's Entrez interface (same underlying API as LangChain's
PubMedAPIWrapper) for fine-grained control over queries, metadata
extraction, and PMC full-text retrieval.

Public API
----------
PubMedClient.search(query_params) -> list[PaperMetadata]
"""

from __future__ import annotations

import threading
import time
import json
from pathlib import Path
from typing import Optional, List

import xmltodict
import requests
from Bio import Entrez
from tenacity import retry, stop_after_attempt, wait_exponential

from src.logger import logger
from src.retriever.models import PaperMetadata
from src.retriever.synonym_expander import SynonymExpander

# Bio.Entrez uses module-level globals (email, api_key) which are NOT
# thread-safe.  This lock serialises all writes to those globals AND all
# Entrez network calls so concurrent ingestion sessions don't corrupt
# each other's state.
_ENTREZ_LOCK = threading.Lock()

# PubMed types accepted in --paper_type CLI flag
PUBMED_TYPE_MAP: dict[str, str] = {
    "review": "Review[Publication Type]",
    "clinical_trial": "Clinical Trial[Publication Type]",
    "meta_analysis": "Meta-Analysis[Publication Type]",
    "systematic_review": "Systematic Review[Publication Type]",
    "case_report": "Case Reports[Publication Type]",
    "randomized_controlled_trial": "Randomized Controlled Trial[Publication Type]",
}


class PubMedQueryParams:
    """Validated search parameters."""

    def __init__(
        self,
        topic: str,
        date_from: int = 2020,
        date_to: int = 2025,
        paper_type: Optional[str] = None,
        max_results: int = 20,
        method: Optional[str] = None,
        exclude_terms: Optional[List[str]] = None,
        use_synonym_expansion: bool = True,
    ) -> None:
        self.topic = topic
        self.date_from = date_from
        self.date_to = date_to
        self.paper_type = paper_type
        self.max_results = max_results
        self.method = method
        self.exclude_terms = exclude_terms or []
        self.use_synonym_expansion = use_synonym_expansion
        self._expander = SynonymExpander()

    def build_query(self) -> str:
            """
            Build the PubMed term string. 
            
            If the topic contains advanced PubMed syntax (like [Mesh], [tiab], or parentheses),
            it returns the query as-is to preserve professional search filtering.
            Otherwise, it applies title-only search and optional synonym expansion.
            
            Supports:
                - Advanced Query: "(term1[Mesh] OR term2[tiab])" -> Returns as-is
                - Simple query: "breast cancer" -> Becomes "breast cancer"[Title]
                - Combined with method: topic + method -> Becomes "topic"[Title] AND "method"[Title]
            """
            
            # Check if the user is using advanced PubMed tags or complex grouping.
            # If '[' or '(' is present, we bypass the automatic [Title] wrapping 
            # to allow specialized fields like [Mesh], [tiab], etc.
            if "[" in self.topic or "(" in self.topic:
                logger.info("Advanced query detected. Bypassing automatic title-wrapping.")
                query = self.topic
            else:
                # Standard logic for simple search terms
                if self.use_synonym_expansion:
                    # Expand synonyms but limit search to the Title field by default
                    query = self._expander.expand_combined_query(
                        main_topic=self.topic,
                        method=self.method,
                        field="Title",
                        exclude_terms=self.exclude_terms,
                    )
                else:
                    # Fallback to simple title search if expansion is disabled
                    parts = [f'"{self.topic}"[Title]']
                    if self.method:
                        parts.append(f'"{self.method}"[Title]')
                    query = " AND ".join(parts)
    
            # Append publication type filter (e.g., Review, Clinical Trial) if selected in UI
            if self.paper_type:
                pt_filter = PUBMED_TYPE_MAP.get(
                    self.paper_type.lower().replace(" ", "_"), None
                )
                if pt_filter:
                    query = f"({query}) AND {pt_filter}"
                else:
                    logger.warning(
                        "Unknown paper_type '{}'. Available: {}",
                        self.paper_type,
                        list(PUBMED_TYPE_MAP.keys()),
                    )
    
            return query

class PubMedClient:
    """
    Retrieves papers from PubMed and optionally full text from PMC.

    Parameters
    ----------
    email : str
        Required by NCBI.
    api_key : str | None
        NCBI API key for higher rate limits (10 req/s vs 3 req/s).
    cache_dir : Path | None
        If set, raw XML responses are cached here to avoid repeated calls.
    """

    _EFETCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    _PMC_BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._email   = email
        self._api_key = api_key
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        # Set module globals once under the lock
        with _ENTREZ_LOCK:
            Entrez.email = email
            if api_key:
                Entrez.api_key = api_key

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def search(self, params: PubMedQueryParams) -> list[PaperMetadata]:
        """
        Run a PubMed search and return enriched PaperMetadata objects.
        """
        query = params.build_query()
        logger.info(
            "PubMed query: {}  [date {}-{}]",
            query, params.date_from, params.date_to,
        )

        pmids = self._esearch(params)
        logger.info("Found {} PMIDs", len(pmids))

        papers: list[PaperMetadata] = []
        for pmid in pmids:
            paper = self._fetch_paper(pmid)
            if paper:
                papers.append(paper)
            time.sleep(0.35)  # stay within NCBI rate limits

        logger.info("Retrieved {} papers with metadata", len(papers))
        return papers

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _esearch(self, params: PubMedQueryParams) -> list[str]:
        """
        Return list of PMIDs matching the query.

        Date range is passed as Entrez mindate/maxdate/datetype parameters
        rather than embedded in the term string.  This avoids HTTP 400
        errors caused by range syntax encoding when usehistory is active,
        and is the form recommended by NCBI documentation.

        usehistory is omitted — we retrieve IDs directly (IdList) which
        is simpler and avoids the WebEnv/query_key state that can cause
        400s when module globals are mutated by concurrent threads.
        """
        query = params.build_query()
        with _ENTREZ_LOCK:
            # Re-assert globals inside the lock in case another thread
            # changed them between our __init__ and this call.
            Entrez.email = self._email
            if self._api_key:
                Entrez.api_key = self._api_key
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=params.max_results,
                mindate=str(params.date_from),
                maxdate=str(params.date_to),
                datetype="pdat",
            )
            record = Entrez.read(handle)
            handle.close()
        return record.get("IdList", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _fetch_paper(self, pmid: str) -> Optional[PaperMetadata]:
        """Fetch full metadata for a single PMID."""
        cache_path = self.cache_dir / f"{pmid}.json" if self.cache_dir else None

        if cache_path and cache_path.exists():
            logger.debug("Cache hit for PMID {}", pmid)
            with open(cache_path) as f:
                data = json.load(f)
            return PaperMetadata(**data)

        try:
            with _ENTREZ_LOCK:
                Entrez.email = self._email
                if self._api_key:
                    Entrez.api_key = self._api_key
                handle = Entrez.efetch(
                    db="pubmed", id=pmid, rettype="xml", retmode="xml"
                )
                raw = handle.read()
                handle.close()
        except Exception as exc:
            logger.error("Failed to fetch PMID {}: {}", pmid, exc)
            return None

        paper = self._parse_pubmed_xml(pmid, raw)

        # Try to enrich with PMC full text
        if paper and paper.pmc_id:
            paper.full_text = self._fetch_pmc_fulltext(paper.pmc_id)

        if paper and cache_path:
            with open(cache_path, "w") as f:
                json.dump(paper.model_dump(), f, indent=2)

        return paper

    def _parse_pubmed_xml(self, pmid: str, xml_bytes: bytes) -> Optional[PaperMetadata]:
        """Parse PubMed XML into PaperMetadata."""
        try:
            doc = xmltodict.parse(xml_bytes)
        except Exception as exc:
            logger.error("XML parse error for PMID {}: {}", pmid, exc)
            return None

        try:
            article_set = doc.get("PubmedArticleSet", {})
            pubmed_article = article_set.get("PubmedArticle", {})
            # Handle list vs single article
            if isinstance(pubmed_article, list):
                pubmed_article = pubmed_article[0]

            medline = pubmed_article.get("MedlineCitation", {})
            article = medline.get("Article", {})

            # Title
            title_raw = article.get("ArticleTitle", "")
            title = (
                title_raw if isinstance(title_raw, str)
                else title_raw.get("#text", str(title_raw))
            )

            # Abstract
            abstract_obj = article.get("Abstract", {})
            abstract_text = abstract_obj.get("AbstractText", "") if abstract_obj else ""
            if isinstance(abstract_text, list):
                abstract_text = " ".join(
                    (t.get("#text", str(t)) if isinstance(t, dict) else str(t))
                    for t in abstract_text
                )
            elif isinstance(abstract_text, dict):
                abstract_text = abstract_text.get("#text", "")

            # Authors
            author_list = article.get("AuthorList", {}).get("Author", [])
            if isinstance(author_list, dict):
                author_list = [author_list]
            authors = []
            for a in author_list:
                last = a.get("LastName", "")
                first = a.get("ForeName", "")
                if last:
                    authors.append(f"{last} {first}".strip())

            # Journal & year
            journal_info = article.get("Journal", {})
            journal_title = journal_info.get("Title", "")
            pub_date = (
                journal_info.get("JournalIssue", {})
                .get("PubDate", {})
            )
            year = pub_date.get("Year", "") or pub_date.get("MedlineDate", "")[:4]

            # Publication types
            pt_list = article.get("PublicationTypeList", {}).get("PublicationType", [])
            if isinstance(pt_list, (str, dict)):
                pt_list = [pt_list]
            pub_types = [
                (p if isinstance(p, str) else p.get("#text", "")) for p in pt_list
            ]

            # PMC ID and DOI
            article_id_list = (
                pubmed_article.get("PubmedData", {})
                .get("ArticleIdList", {})
                .get("ArticleId", [])
            )
            if isinstance(article_id_list, dict):
                article_id_list = [article_id_list]
            pmc_id = None
            doi = None
            for aid in article_id_list:
                id_type = aid.get("@IdType", "")
                id_val = aid.get("#text", "")
                if id_type == "pmc":
                    pmc_id = id_val
                elif id_type == "doi":
                    doi = id_val

            return PaperMetadata(
                pmid=pmid,
                title=title,
                authors=authors,
                journal=journal_title,
                year=str(year),
                abstract=abstract_text,
                publication_types=pub_types,
                pmc_id=pmc_id,
                doi=doi,
                pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )

        except Exception as exc:
            logger.error("Metadata parse error for PMID {}: {}", pmid, exc)
            return None

    def _fetch_pmc_fulltext(self, pmc_id: str) -> Optional[str]:
        """
        Attempt to retrieve full text from PubMed Central OA API.
        Returns plain text or None if unavailable.
        """
        url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmc_id}/unicode"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.debug("PMC full text not available for {}", pmc_id)
                return None
            data = resp.json()
            # BioC JSON structure: documents[0].passages[].text
            passages = []
            for doc in data if isinstance(data, list) else [data]:
                for passage in doc.get("documents", [{}])[0].get("passages", []):
                    text = passage.get("text", "").strip()
                    if text:
                        passages.append(text)
            full_text = "\n\n".join(passages)
            logger.debug(
                "PMC full text retrieved for {} ({} chars)", pmc_id, len(full_text)
            )
            return full_text or None
        except Exception as exc:
            logger.debug("PMC fetch error for {}: {}", pmc_id, exc)
            return None
