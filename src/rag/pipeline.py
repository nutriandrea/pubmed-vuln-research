"""
Module 5 — RAG Pipeline
Module 6 — LLM Reasoning Layer

Combines retrieval from the vector store with LLM synthesis to answer
research-limitation questions.

Public API
----------
LimitationRAGPipeline.ask(question)        -> str  (single Q&A)
LimitationRAGPipeline.synthesize(topic)    -> str  (full report)
"""

from __future__ import annotations

from typing import Optional
from collections import Counter, defaultdict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.logger import logger
from src.rag.prompts import (
    RAG_ANSWER_PROMPT,
    SYNTHESIS_PROMPT,
    RESEARCH_GRADE_PROMPT,
    INSIGHT_GENERATOR_PROMPT,
)
from src.vectorstore.qdrant_store import LimitationVectorStore


def _format_docs(docs: list[Document]) -> str:
    """
    Turn a list of retrieved Documents into a single context string,
    prepending each chunk with its source citation.
    """
    parts = []
    for doc in docs:
        meta = doc.metadata
        pmid = meta.get("pmid", "")
        pubmed_url = meta.get("pubmed_url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "")
        title = meta.get("paper_title", "Unknown")
        year = meta.get("year", "?")
        
        if pubmed_url:
            source = f"[{title} ({year})]({pubmed_url}) "
        else:
            source = f"[{title} ({year})] "
        
        category = meta.get("category", "")
        severity = meta.get("severity", "")
        if category:
            source += f"[{category}] "
        if severity:
            source += f"[{severity}] "
        parts.append(source + doc.page_content)
    return "\n\n---\n\n".join(parts)


def _analyze_limitations(docs: list[Document]) -> dict:
    """Pre-LLM statistical analysis of limitations with accurate counting."""
    stats = {
        'n_papers': set(),
        'by_category': Counter(),
        'by_severity': Counter(),
        'by_year': Counter(),
        'by_type': Counter(),
        'by_type_severity': defaultdict(Counter),
    }
    
    # Industry standards for comparison
    MIN_SAMPLE_SIZES = {
        'pilot_study': 20,
        'observational': 50,
        'clinical_trial': 100,
        'confirmatory': 300,
    }
    
    for doc in docs:
        meta = doc.metadata
        pmid = meta.get('pmid')
        category = meta.get('category', 'limitation')
        severity = meta.get('severity', 'medium')
        year = meta.get('year', 'unknown')
        
        if pmid:
            stats['n_papers'].add(pmid)
        
        stats['by_category'][category] += 1
        stats['by_severity'][severity] += 1
        if year and year != 'unknown':
            stats['by_year'][year] += 1
        
        # More accurate type detection from category/metadata, not just content
        category_lower = category.lower()
        
        # Map category to type
        if 'sample' in category_lower or 'cohort' in category_lower:
            type_name = 'sample_size'
        elif 'retrospective' in category_lower:
            type_name = 'retrospective'
        elif 'single' in category_lower or 'center' in category_lower:
            type_name = 'single_center'
        elif 'follow' in category_lower or 'duration' in category_lower:
            type_name = 'follow_up'
        elif 'validation' in category_lower:
            type_name = 'validation'
        elif 'control' in category_lower:
            type_name = 'control_group'
        elif 'population' in category_lower or 'diversity' in category_lower:
            type_name = 'population'
        elif 'randomized' in category_lower or 'rct' in category_lower:
            type_name = 'rct'
        elif 'method' in category_lower or 'design' in category_lower:
            type_name = 'methodology'
        elif 'data' in category_lower or 'dataset' in category_lower:
            type_name = 'data_limitation'
        else:
            # Fallback: detect from content
            content = doc.page_content.lower()
            if 'sample' in content and ('small' in content or 'limited' in content):
                type_name = 'sample_size'
            elif 'retrospective' in content:
                type_name = 'retrospective'
            elif 'single center' in content or 'monocentric' in content:
                type_name = 'single_center'
            elif 'follow-up' in content or 'duration' in content:
                type_name = 'follow_up'
            elif 'validation' in content:
                type_name = 'validation'
            elif 'control group' in content:
                type_name = 'control_group'
            else:
                type_name = 'other_limitation'
        
        stats['by_type'][type_name] += 1
        stats['by_type_severity'][type_name][severity] += 1
    
    # Calculate dynamic severity based on frequency and impact
    n_papers = len(stats['n_papers'])
    by_type_with_severity = {}
    
    for type_name, count in stats['by_type'].items():
        frequency = count / n_papers if n_papers > 0 else 0
        high_count = stats['by_type_severity'][type_name].get('high', 0)
        
        # Dynamic severity calculation
        if frequency > 0.5 or high_count > 3:
            dynamic_severity = 'HIGH'
        elif frequency > 0.25 or high_count > 1:
            dynamic_severity = 'MEDIUM'
        else:
            dynamic_severity = 'LOW'
        
        by_type_with_severity[type_name] = {
            'count': count,
            'percentage': round(frequency * 100, 1),
            'severity': dynamic_severity,
            'high_count': high_count,
        }
    
    # Sort by severity impact
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    sorted_types = sorted(
        by_type_with_severity.items(),
        key=lambda x: (severity_order.get(x[1]['severity'], 3), -x[1]['count'])
    )
    by_type_with_severity = dict(sorted_types)
    
    # Calculate temporal trends
    trends = _calculate_temporal_trends(stats['by_year'])
    
    stats['n_papers'] = n_papers
    stats['by_category'] = dict(stats['by_category'])
    stats['by_severity'] = dict(stats['by_severity'])
    stats['by_year'] = dict(sorted(stats['by_year'].items()))
    stats['by_type'] = by_type_with_severity
    stats['trends'] = trends
    
    # Add comparison standards
    stats['standards'] = {
        'sample_size': {
            'pilot': MIN_SAMPLE_SIZES['pilot_study'],
            'observational': MIN_SAMPLE_SIZES['observational'],
            'clinical_trial': MIN_SAMPLE_SIZES['clinical_trial'],
            'confirmatory': MIN_SAMPLE_SIZES['confirmatory'],
            'guideline': 'ICH E6 recommends minimum 300 for confirmatory trials',
        },
        'follow_up': {
            'transplant': 24,
            'chronic': 60,
            'guideline': 'AAO recommends 24-month minimum for transplant studies',
        },
        'control_groups': {
            'guideline': 'CONSORT statement requires control group for RCTs',
        },
    }
    
    return stats


def _calculate_temporal_trends(by_year: dict) -> dict:
    """Analyze how limitations change over years."""
    if not by_year or len(by_year) < 2:
        return {'summary': 'Insufficient data for trend analysis'}
    
    years = sorted(by_year.keys())
    if len(years) < 2:
        return {'summary': 'Insufficient data for trend analysis'}
    
    first_year_count = by_year.get(years[0], 0)
    last_year_count = by_year.get(years[-1], 0)
    
    if last_year_count > first_year_count * 1.5:
        trend = 'INCREASING'
    elif last_year_count < first_year_count * 0.7:
        trend = 'DECREASING'
    else:
        trend = 'STABLE'
    
    return {
        'summary': f'{trend} trend from {years[0]} to {years[-1]}',
        'trend': trend,
        'first_year': years[0],
        'last_year': years[-1],
        'first_count': first_year_count,
        'last_count': last_year_count,
    }



def _identify_research_streams(docs: list[Document]) -> list[dict]:
    """Identify research streams from documents based on content clustering."""
    # Group documents by keywords in paper_title or content
    stream_docs = defaultdict(list)
    
    # Merge similar streams to avoid duplicates
    STREAM_MERGES = {
        'Fuchs Dystrophy': 'Endothelial Disorders',
        'Endothelial Keratoplasty': 'Transplant Surgery',
        'Penetrating Keratoplasty': 'Transplant Surgery',
        'Lamellar Keratoplasty': 'Transplant Surgery',
        'Corneal Biology': 'Endothelial Disorders',
        'Imaging': 'Diagnostic Methods',
        'Diagnosis': 'Diagnostic Methods',
    }
    
    # Priority: higher priority streams are checked first
    STREAM_PRIORITY = {
        'Transplant Surgery': 1,
        'Endothelial Disorders': 2,
        'Clinical Outcomes': 3,
        'Immunology': 4,
        'Genetics': 5,
        'Diagnostic Methods': 6,
    }
    
    keywords_to_stream = {
        'clinical': 'Clinical Outcomes',
        'outcome': 'Clinical Outcomes',
        'survival': 'Clinical Outcomes',
        'graft': 'Transplant Surgery',
        'transplant': 'Transplant Surgery',
        'surgery': 'Transplant Surgery',
        'dmek': 'Transplant Surgery',
        'pkp': 'Transplant Surgery',
        'dalk': 'Transplant Surgery',
        'keratoplasty': 'Transplant Surgery',
        'immunology': 'Immunology',
        'immune': 'Immunology',
        'rejection': 'Immunology',
        'genetic': 'Genetics',
        'gene': 'Genetics',
        'mutation': 'Genetics',
        'fuchs': 'Endothelial Disorders',
        'endothelial': 'Endothelial Disorders',
        'diabetes': 'Diabetes Comorbidities',
        'diabetic': 'Diabetes Comorbidities',
        'pediatric': 'Pediatric',
        'child': 'Pediatric',
    }
    
    for doc in docs:
        content = (doc.page_content + ' ' + doc.metadata.get('paper_title', '')).lower()
        
        # Find matching stream with priority
        matched_stream = None
        for kw, stream in keywords_to_stream.items():
            if kw in content:
                # Apply merge rules
                canonical = STREAM_MERGES.get(stream, stream)
                if matched_stream is None:
                    matched_stream = canonical
                elif STREAM_PRIORITY.get(canonical, 99) < STREAM_PRIORITY.get(matched_stream, 99):
                    matched_stream = canonical
        
        if matched_stream:
            stream_docs[matched_stream].append(doc)
    
    # Build stream info with accurate severity calculation
    streams = []
    for stream, stream_docs_list in stream_docs.items():
        unique_pmids = set(d.metadata.get('pmid') for d in stream_docs_list if d.metadata.get('pmid'))
        
        # Count problems and calculate severity
        high_problems = sum(1 for d in stream_docs_list 
                         if d.metadata.get('severity') == 'high')
        medium_problems = sum(1 for d in stream_docs_list 
                           if d.metadata.get('severity') == 'medium')
        
        problems = sum(1 for d in stream_docs_list 
                     if d.metadata.get('category') in ['limitation', 'methodological_weakness'])
        
        # Calculate overall severity based on problem count and severity mix
        if high_problems > 3 or problems > 10:
            overall_severity = 'HIGH'
        elif high_problems > 1 or problems > 5:
            overall_severity = 'MEDIUM'
        else:
            overall_severity = 'LOW'
        
        # Calculate % of papers with problems
        problem_rate = problems / len(unique_pmids) * 100 if unique_pmids else 0
        
        streams.append({
            'name': stream,
            'n_papers': len(unique_pmids),
            'n_docs': len(stream_docs_list),
            'n_problems': problems,
            'problem_rate': round(problem_rate, 1),
            'high_count': high_problems,
            'medium_count': medium_problems,
            'severity_breakdown': {'high': high_problems, 'medium': medium_problems},
            'overall_severity': overall_severity,
        })
    
    # Sort by severity (HIGH > MEDIUM > LOW) then by problem count
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    streams.sort(key=lambda x: (severity_order.get(x['overall_severity'], 3), -x['n_problems']))
    
    return streams[:10]  # Return top 10 streams


# Specific recommendations based on stream and problem type
SPECIFIC_RECOMMENDATIONS = {
    'sample_size': {
        'clinical_trial': 'Enroll minimum 200-300 subjects per arm (ICH E6 guideline)',
        'pilot_study': 'Include minimum 20-30 subjects per arm',
        'rare_disease': 'Use adaptive trial design or registry-based approach',
    },
    'follow_up': {
        'transplant': 'Implement minimum 24-month follow-up (AAO recommendation)',
        'chronic': 'Plan for 5+ years longitudinal follow-up',
    },
    'retrospective': {
        'default': 'Consider prospective design or nested case-control within prospective cohort',
    },
    'control_group': {
        'intervention': 'Include RCT or matched control group (CONSORT requirement)',
        'diagnostic': 'Validate on independent cohort',
    },
    'single_center': {
        'default': 'Conduct multi-center study to improve generalizability',
    },
    'rct': {
        'default': 'Prioritize RCT where feasible, or use quasi-experimental design',
    },
}


class LimitationRAGPipeline:
    """
    End-to-end RAG pipeline for research limitation analysis.

    Parameters
    ----------
    vector_store : LimitationVectorStore
        Pre-populated vector store with limitation chunks.
    model_name : str
        OpenAI chat model to use for synthesis.
    top_k : int
        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        vector_store: LimitationVectorStore,
        model_name: str = "gpt-4o-mini",
        top_k: int = 8,
    ) -> None:
        self._store = vector_store
        self._top_k = top_k
        self._llm = ChatOpenAI(model=model_name, temperature=0)
        self._str_parser = StrOutputParser()
        self._build_chain()

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def ask(
        self,
        question: str,
        filter_category: Optional[str] = None,
        filter_type: Optional[str] = "limitation",
    ) -> str:
        """
        Answer a single question about research limitations.

        The question is used both as the retrieval query and as the
        prompt input for the LLM.
        
        Parameters
        ----------
        question : str
            The question to answer.
        filter_category : str | None
            Filter by category (dataset, methodology, evaluation, bias, etc.).
        filter_type : str | None
            Filter by type (default: "limitation").
        """
        logger.info("RAG ask: '{}'", question)
        retrieved = self._store.similarity_search(
            question,
            k=self._top_k,
            filter_type=filter_type,
            filter_category=filter_category,
        )
        if not retrieved:
            return "No relevant limitations found in the knowledge base for this query."

        context = _format_docs(retrieved)
        response = self._qa_chain.invoke(
            {"context": context, "question": question}
        )
        logger.info("RAG answer generated ({} chars)", len(response))
        return response

    def synthesize(self, topic: str, n_papers: Optional[int] = None) -> str:
        """
        Generate a full structured limitations report for a topic.

        Retrieves a broader set of chunks and asks the LLM to synthesize
        them into a categorised report with detailed analysis.

        Parameters
        ----------
        topic : str
            The research topic (e.g. "breast cancer detection with deep learning").
        n_papers : int | None
            Number of source papers (shown in the system prompt for context).
        """
        logger.info("Synthesizing limitations report for topic: '{}'", topic)
        
        # Use a broad query to surface all categories
        query = f"limitations weaknesses research gaps methodology {topic}"
        
        # Retrieve MORE chunks for detailed synthesis (increased from 20 to 150)
        retrieved = self._store.similarity_search(query, k=150)
        
        if not retrieved:
            return "No limitations have been indexed yet. Run the ingestion pipeline first."

        # Pre-LLM statistical analysis
        stats = _analyze_limitations(retrieved)
        streams = _identify_research_streams(retrieved)
        
        n = n_papers or stats['n_papers']
        
        # Format statistics for the prompt - IMPROVED
        type_lines = []
        for type_name, type_data in stats['by_type'].items():
            type_lines.append(f"- {type_name}: {type_data['count']} papers ({type_data['percentage']}%) [SEVERITY: {type_data['severity']}]")
        
        streams_lines = []
        for i, s in enumerate(streams[:10]):
            severity_emoji = '🚨' if s['overall_severity'] == 'HIGH' else '⚠️' if s['overall_severity'] == 'MEDIUM' else '✅'
            streams_lines.append(f"{i+1}. {s['name']} {severity_emoji}: {s['n_problems']} problems in {s['n_papers']} papers ({s['problem_rate']}% of papers), HIGH issues: {s['high_count']}")
        
        # Temporal trend info
        trend_info = stats.get('trends', {})
        
        # Standards comparison
        standards_info = f"""
INDUSTRY STANDARDS FOR COMPARISON:
- Sample Size (Pilot): >{stats['standards']['sample_size']['pilot']} subjects
- Sample Size (Observational): >{stats['standards']['sample_size']['observational']} subjects  
- Sample Size (Clinical Trial): >{stats['standards']['sample_size']['clinical_trial']} subjects
- Sample Size (Confirmatory): >{stats['standards']['sample_size']['confirmatory']} subjects
- Follow-up (Transplant): {stats['standards']['follow_up']['transplant']} months minimum
- Control Groups: Required for RCTs (CONSORT)
"""
        
        stats_str = f"""
QUANTITATIVE ANALYSIS:
- Total unique papers analyzed: {stats['n_papers']}
- Total document chunks: {len(retrieved)}

LIMITATION TYPE FREQUENCY (sorted by severity):
{chr(10).join(type_lines)}

SEVERITY DISTRIBUTION:
{chr(10).join(f"- {k}: {v}" for k, v in stats['by_severity'].items())}

TEMPORAL TREND: {trend_info.get('summary', 'N/A')}

RESEARCH STREAMS (Ranked by severity and problems):
{chr(10).join(streams_lines)}

{standards_info}
"""
        
        context = _format_docs(retrieved)
        
        response = self._synthesis_chain.invoke(
            {"context": context, "topic": topic, "n_papers": n, "stats": stats_str, "streams": streams}
        )
        logger.info("Synthesis report generated ({} chars)", len(response))
        return response

    def ask_with_sources(
        self,
        question: str,
        filter_category: Optional[str] = None,
        filter_type: Optional[str] = "limitation",
    ) -> dict:
        """
        Like ask(), but also returns the source documents used.
        
        Parameters
        ----------
        question : str
            The question to answer.
        filter_category : str | None
            Filter by category.
        filter_type : str | None
            Filter by type (default: "limitation").

        Returns
        -------
        dict with keys:
            - 'answer': str
            - 'sources': list[dict]  (paper_title, year, pmid, category, severity)
        """
        retrieved = self._store.similarity_search(
            question,
            k=self._top_k,
            filter_type=filter_type,
            filter_category=filter_category,
        )
        if not retrieved:
            return {
                "answer": "No relevant limitations found.",
                "sources": [],
            }

        context = _format_docs(retrieved)
        answer = self._qa_chain.invoke({"context": context, "question": question})

        sources = []
        seen = set()
        for doc in retrieved:
            meta = doc.metadata
            key = (meta.get("pmid"), meta.get("category"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "paper_title": meta.get("paper_title"),
                    "year": meta.get("year"),
                    "pmid": meta.get("pmid"),
                    "journal": meta.get("journal"),
                    "category": meta.get("category"),
                    "severity": meta.get("severity"),
                    "pubmed_url": meta.get("pubmed_url"),
                })

        return {"answer": answer, "sources": sources}

    def ask_research_grade(
        self,
        question: str,
        filter_category: Optional[str] = None,
    ) -> dict:
        """
        Answer with research-grade output including confidence levels and evidence.
        
        Returns:
            dict with 'answer', 'sources', 'confidence', 'key_limitations'
        """
        logger.info("Research-grade ask: '{}'", question)
        retrieved = self._store.similarity_search(
            question,
            k=self._top_k * 2,  # Get more for better analysis
            filter_type="limitation",
            filter_category=filter_category,
        )
        if not retrieved:
            return {
                "answer": "No relevant limitations found.",
                "sources": [],
                "confidence": "LOW",
                "key_limitations": [],
            }

        context = _format_docs(retrieved)
        answer = self._research_grade_chain.invoke(
            {"context": context, "question": question}
        )

        sources = []
        key_limitations = []
        seen = set()
        
        for doc in retrieved:
            meta = doc.metadata
            key = (meta.get("pmid"), meta.get("category"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "paper_title": meta.get("paper_title"),
                    "year": meta.get("year"),
                    "pmid": meta.get("pmid"),
                    "category": meta.get("category"),
                    "severity": meta.get("severity"),
                })
                
                if meta.get("severity") == "high":
                    key_limitations.append({
                        "text": doc.page_content[:200],
                        "category": meta.get("category"),
                        "severity": meta.get("severity"),
                    })

        # Estimate confidence based on number of sources
        confidence = "HIGH" if len(sources) > 20 else "MEDIUM" if len(sources) > 5 else "LOW"

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "key_limitations": key_limitations[:5],
        }

    def generate_insights(self, n_papers: int = 0) -> str:
        """
        Automatically generate top insights without user query.
        
        Uses vulnerability data to create structured insights.
        """
        logger.info("Generating automatic insights")
        
        # Retrieve a broad sample of limitations
        query = "limitations weaknesses research gaps methodology"
        retrieved = self._store.similarity_search(query, k=self._top_k * 3)
        
        if not retrieved:
            return "No data available for insights. Run ingestion first."

        context = _format_docs(retrieved)
        n = n_papers or len({d.metadata.get("pmid") for d in retrieved})
        
        response = self._insight_chain.invoke(
            {"context": context, "n_papers": n}
        )
        
        logger.info("Insights generated ({} chars)", len(response))
        return response

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _build_chain(self) -> None:
        """Pre-build both LangChain LCEL chains."""
        # Q&A chain: context + question → answer
        self._qa_chain = (
            RAG_ANSWER_PROMPT
            | self._llm
            | self._str_parser
        )
        # Synthesis chain: context + topic + n_papers → full report
        self._synthesis_chain = (
            SYNTHESIS_PROMPT
            | self._llm
            | self._str_parser
        )
        # Research-grade chain: enhanced answer with confidence + evidence
        self._research_grade_chain = (
            RESEARCH_GRADE_PROMPT
            | self._llm
            | self._str_parser
        )
        # Insight generator chain: automatic insights
        self._insight_chain = (
            INSIGHT_GENERATOR_PROMPT
            | self._llm
            | self._str_parser
        )
