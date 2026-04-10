"""
FastAPI backend for PubMed Research Limitation Analyzer.
"""
from __future__ import annotations

import asyncio
import json
import uuid
import re
from pathlib import Path
from typing import AsyncGenerator, Optional, List
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── project imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.orchestrator import ResearchLimitationAnalyzer
from src.logger import logger
from app.services.pdf_service import generate_pdf_from_markdown

# ── app setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="PubMed Limitation Analyzer", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── session store ──────────────────────────────────────────────────────────────
class Session:
    def __init__(self, sid: str):
        self.sid = sid
        self.analyzer = ResearchLimitationAnalyzer(
            model_name=settings.openai_model,
            embedding_model=settings.openai_embedding_model,
            top_k=settings.retrieval_top_k,
        )
        self.topic: Optional[str] = None
        self.ingested: bool = False
        self.n_papers: int = 0
        self.n_chunks: int = 0
        self.paper_type: Optional[str] = None
        self.date_from: int = 2020
        self.date_to: int = 2025
        self.results: list = []


_sessions: dict[str, Session] = {}


def _get_or_create(sid: Optional[str]) -> Session:
    if sid and sid in _sessions:
        return _sessions[sid]
    new_sid = sid or str(uuid.uuid4())
    _sessions[new_sid] = Session(sid=new_sid)
    return _sessions[new_sid]


# ── request / response models ──────────────────────────────────────────────────
class IngestRequest(BaseModel):
    sid: Optional[str] = None
    topic: str
    date_from: int = 2020
    date_to: int = 2025
    paper_type: Optional[str] = None
    max_papers: int = 10
    method: Optional[str] = None
    exclude_terms: Optional[List[str]] = None
    reset_knowledge_base: bool = True
    language: Optional[str] = None
    journal: Optional[str] = None
    free_full_text: Optional[bool] = None
    has_abstract: Optional[bool] = None


class AskRequest(BaseModel):
    sid: str
    question: str


class SynthesizeRequest(BaseModel):
    sid: str


class AskResearchGradeRequest(BaseModel):
    sid: str
    question: str
    filter_category: Optional[str] = None


class GenerateInsightsRequest(BaseModel):
    sid: str


# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>Static files not found</h1>", 404)


@app.get("/api/session/{sid}")
async def session_status(sid: str):
    """Get session status with vulnerabilities data."""
    if sid not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = _sessions[sid]
    
    # Get vulnerabilities data
    vulnerabilities = []
    by_category = {}
    total_mentions = 0
    
    if s.analyzer._vulnerabilities:
        for v in s.analyzer._vulnerabilities:
            vulnerabilities.append({
                "limitation": v.limitation,
                "frequency": v.frequency,
                "severity": v.severity,
                "category": v.category,
                "trend": v.trend,
                "papers": list(set(v.papers))[:10],
            })
            by_category[v.category] = by_category.get(v.category, 0) + v.frequency
            total_mentions += v.frequency
    
    return {
        "sid": s.sid,
        "topic": s.topic,
        "ingested": s.ingested,
        "n_papers": s.n_papers,
        "n_chunks": s.n_chunks,
        "paper_type": s.paper_type,
        "date_from": s.date_from,
        "date_to": s.date_to,
        "vulnerabilities": vulnerabilities,
        "by_category": by_category,
        "total_mentions": total_mentions,
    }


@app.delete("/api/session/{sid}")
async def clear_session(sid: str):
    """Clear a session."""
    _sessions.pop(sid, None)
    return {"ok": True}


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """Ingest papers and return results."""
    session = _get_or_create(req.sid)
    session.topic = req.topic
    session.date_from = req.date_from
    session.date_to = req.date_to
    session.paper_type = req.paper_type
    session.ingested = False
    
    # Reset analyzer for fresh ingestion
    session.analyzer = ResearchLimitationAnalyzer(
        model_name=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
        top_k=settings.retrieval_top_k,
    )

    def run_ingest():
        """Runs in thread pool."""
        from src.retriever.pubmed_client import PubMedClient, PubMedQueryParams
        from src.extractor.section_extractor import LimitationExtractor
        from src.processor.document_builder import DocumentBuilder
        from src.vectorstore.qdrant_store import LimitationVectorStore
        from langchain_openai import OpenAIEmbeddings
        import re

        def slug(t): return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")

        try:
            logger.info(f"Starting ingestion for topic: {req.topic}")
            if req.method:
                logger.info(f"Method: {req.method}")
            if req.exclude_terms:
                logger.info(f"Exclude terms: {req.exclude_terms}")

            raw_dir = settings.raw_data_dir / slug(req.topic)
            processed_dir = settings.processed_data_dir / slug(req.topic)
            raw_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Reset knowledge base if requested
            if req.reset_knowledge_base and session.analyzer._vector_store:
                try:
                    session.analyzer._vector_store.clear()
                    logger.info("Knowledge base cleared")
                except Exception as e:
                    logger.warning(f"Could not clear vector store: {e}")

            client = PubMedClient(
                email=settings.ncbi_email,
                api_key=settings.ncbi_api_key,
                cache_dir=raw_dir,
            )
            params = PubMedQueryParams(
                topic=req.topic,
                date_from=req.date_from,
                date_to=req.date_to,
                paper_type=req.paper_type,
                max_results=req.max_papers,
                method=req.method,
                exclude_terms=req.exclude_terms,
                use_synonym_expansion=True,
                language=req.language,
                journal=req.journal,
                free_full_text=req.free_full_text,
                has_abstract=req.has_abstract,
            )
            papers = client.search(params)

            if not papers:
                logger.warning("No papers found")
                return {"error": "No papers found"}

            logger.info(f"Found {len(papers)} papers")

            extractor = LimitationExtractor(model_name=settings.openai_model)
            
            def progress_callback(msg_type, data):
                if msg_type == "progress":
                    logger.info(f"[{data.get('percent', 0)}%] {data.get('msg', '')}")
                elif msg_type == "status":
                    logger.info(data.get('msg', ''))
                elif msg_type == "complete":
                    logger.info(f"Batch complete: {data}")
            
            extracted_list = extractor.extract_batch(papers, progress_callback=progress_callback)

            logger.info("Building documents")

            # Step 2.5: Clustering limitazioni (semantic deduplication)
            from src.processor.limitation_clusterer import (
                LimitationClusterer,
                extract_limitations_for_clustering,
            )
            limitations_for_clustering = extract_limitations_for_clustering(extracted_list)

            logger.info(f"Found {len(limitations_for_clustering)} limitations for clustering")
            if limitations_for_clustering:
                logger.info(f"Sample: {limitations_for_clustering[0]}")

            if limitations_for_clustering:
                try:
                    clusterer = LimitationClusterer(similarity_threshold=0.80)
                    clusters = clusterer.cluster(limitations_for_clustering)
                    logger.info(f"Created {len(clusters)} limitation clusters")

                    # Step 2.6: Vulnerability ranking
                    from src.processor.vulnerability_ranker import VulnerabilityRanker
                    ranker = VulnerabilityRanker()
                    vulnerabilities = ranker.rank(clusters)
                    logger.info(f"Ranked {len(vulnerabilities)} vulnerabilities")

                    # Salva nell'analyzer
                    session.analyzer._vulnerabilities = vulnerabilities
                    session.analyzer._clusters = clusters
                    
                    # KPI tracking
                    n_with_limitations = sum(
                        1 for e in extracted_list 
                        if e.limitations or e.classified_limitations or e.research_gaps
                    )
                    pct = n_with_limitations / len(extracted_list) * 100 if extracted_list else 0
                    logger.info(f"Extraction KPI: {n_with_limitations}/{len(extracted_list)} papers "
                                f"have limitations extracted ({pct:.1f}%)")
                except Exception as e:
                    logger.warning(f"Clustering/ranking failed: {e}")

            builder = DocumentBuilder(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            documents = builder.build(extracted_list)

            logger.info(f"Indexing {len(documents)} chunks into Qdrant")
            embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)
            if settings.qdrant_in_memory:
                vs = LimitationVectorStore.create_in_memory(
                    embeddings=embeddings,
                    collection_name=settings.qdrant_collection,
                )
            else:
                vs = LimitationVectorStore.create_persistent(
                    embeddings=embeddings,
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key,
                    collection_name=settings.qdrant_collection,
                )
            vs.add_documents(documents)

            from src.rag.pipeline import LimitationRAGPipeline
            session.analyzer._vector_store = vs
            session.analyzer.topic = req.topic
            session.analyzer._n_papers = len(papers)
            session.analyzer.rag = LimitationRAGPipeline(
                vector_store=vs,
                model_name=settings.openai_model,
                top_k=settings.retrieval_top_k,
            )
            session.ingested = True
            session.n_papers = len(papers)
            session.n_chunks = vs.count()
            session.results = extracted_list

            logger.info(f"Ingestion complete: {len(papers)} papers, {session.n_chunks} chunks")
            return {
                "sid": session.sid,
                "n_papers": len(papers),
                "n_chunks": session.n_chunks,
                "msg": f"Ready. Indexed {len(papers)} papers ({session.n_chunks} chunks).",
            }

        except Exception as exc:
            logger.exception(f"Ingest error: {exc}")
            return {"error": str(exc)}

    # Run ingestion in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_ingest)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/api/ask")
async def ask(req: AskRequest):
    """Answer a question using the RAG pipeline."""
    if req.sid not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[req.sid]
    if not session.ingested:
        raise HTTPException(status_code=400, detail="Run ingestion first")

    def run_ask():
        try:
            result = session.analyzer.ask_with_sources(req.question)
            return result
        except Exception as exc:
            logger.exception(f"Ask error: {exc}")
            return {"error": str(exc)}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_ask)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Generate a full synthesis report."""
    if req.sid not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[req.sid]
    if not session.ingested:
        raise HTTPException(status_code=400, detail="Run ingestion first")

    def run_synth():
        try:
            report = session.analyzer.synthesize()
            return {"report": report, "topic": session.topic}
        except Exception as exc:
            logger.exception(f"Synthesize error: {exc}")
            return {"error": str(exc)}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_synth)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/api/synthesize/pdf")
async def synthesize_pdf(req: SynthesizeRequest):
    """Generate a full synthesis report as PDF."""
    if req.sid not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[req.sid]
    if not session.ingested:
        raise HTTPException(status_code=400, detail="Run ingestion first")

    def run_pdf():
        try:
            report = session.analyzer.synthesize()
            title = f"Research Limitations Report - {session.topic}"
            pdf_buffer = generate_pdf_from_markdown(report, title)
            return pdf_buffer
        except Exception as exc:
            logger.exception(f"PDF generation error: {exc}")
            return None

    loop = asyncio.get_event_loop()
    pdf_buffer = await loop.run_in_executor(None, run_pdf)

    if pdf_buffer is None:
        raise HTTPException(status_code=500, detail="Failed to generate PDF")

    filename = f"limitations_{session.topic.replace(' ', '_')}.pdf"
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/api/ask-research-grade")
async def ask_research_grade(req: AskResearchGradeRequest):
    """Answer with research-grade output including confidence and evidence."""
    if req.sid not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[req.sid]
    if not session.ingested:
        raise HTTPException(status_code=400, detail="Run ingestion first")

    def run_ask():
        try:
            return session.analyzer.rag.ask_research_grade(
                req.question,
                filter_category=req.filter_category,
            )
        except Exception as exc:
            logger.exception(f"Research-grade ask error: {exc}")
            return {"error": str(exc)}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_ask)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/api/insights")
async def generate_insights(req: GenerateInsightsRequest):
    """Automatically generate top insights without user query."""
    if req.sid not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[req.sid]
    if not session.ingested:
        raise HTTPException(status_code=400, detail="Run ingestion first")

    def run_insights():
        try:
            return session.analyzer.rag.generate_insights(session.n_papers)
        except Exception as exc:
            logger.exception(f"Insights generation error: {exc}")
            return {"error": str(exc)}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_insights)

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return {"insights": result}
