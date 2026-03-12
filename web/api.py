"""
FastAPI backend for the PubMed Research Limitation Analyzer.

Endpoints
---------
GET  /                      → serve index.html
POST /api/ingest            → start ingestion pipeline (SSE stream)
POST /api/ask               → ask a question (SSE stream)
POST /api/synthesize        → generate full report (SSE stream)
GET  /api/session/{sid}     → get session status
DELETE /api/session/{sid}   → clear session

All heavy work runs in a thread pool so the event loop stays unblocked.
SSE (Server-Sent Events) streams progress messages and the final result
back to the browser in real time.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings
from src.orchestrator import ResearchLimitationAnalyzer
from src.logger import logger

# ── app setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="PubMed Limitation Analyzer", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_executor = ThreadPoolExecutor(max_workers=4)


# ── session store ──────────────────────────────────────────────────────────────
@dataclass
class Session:
    sid: str
    analyzer: ResearchLimitationAnalyzer = field(
        default_factory=ResearchLimitationAnalyzer
    )
    topic: Optional[str] = None
    ingested: bool = False
    n_papers: int = 0
    n_chunks: int = 0
    paper_type: Optional[str] = None
    date_from: int = 2020
    date_to: int = 2025


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


class AskRequest(BaseModel):
    sid: str
    question: str


class SynthesizeRequest(BaseModel):
    sid: str


# ── SSE helpers ────────────────────────────────────────────────────────────────
def _event(kind: str, data: dict | str) -> dict:
    payload = data if isinstance(data, str) else json.dumps(data)
    return {"event": kind, "data": payload}


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(str(html_path))


@app.get("/api/session/{sid}")
async def session_status(sid: str):
    if sid not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = _sessions[sid]
    return {
        "sid": s.sid,
        "topic": s.topic,
        "ingested": s.ingested,
        "n_papers": s.n_papers,
        "n_chunks": s.n_chunks,
        "paper_type": s.paper_type,
        "date_from": s.date_from,
        "date_to": s.date_to,
    }


@app.delete("/api/session/{sid}")
async def clear_session(sid: str):
    _sessions.pop(sid, None)
    return {"ok": True}


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """
    Stream ingestion progress via SSE.
    Sends events: progress, paper, done, error
    """
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

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_ingest():
        """Runs in thread pool, pushes events into the async queue."""
        import json as _json
        from src.retriever.pubmed_client import PubMedClient, PubMedQueryParams
        from src.extractor.section_extractor import LimitationExtractor
        from src.processor.document_builder import DocumentBuilder
        from src.vectorstore.qdrant_store import LimitationVectorStore
        from langchain_openai import OpenAIEmbeddings
        import re, time

        def slug(t): return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")
        def push(kind, data): loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

        try:
            push("progress", {"step": 1, "msg": f"Searching PubMed for «{req.topic}»…"})

            raw_dir = settings.raw_data_dir / slug(req.topic)
            processed_dir = settings.processed_data_dir / slug(req.topic)
            raw_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)

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
            )
            papers = client.search(params)

            if not papers:
                push("error", {"msg": "No papers found. Try a broader query or date range."})
                return

            push("progress", {
                "step": 1,
                "msg": f"Found {len(papers)} papers. Starting extraction…",
                "total": len(papers),
            })

            extractor = LimitationExtractor(model_name=settings.openai_model)
            extracted_list = []
            for i, paper in enumerate(papers):
                push("paper", {
                    "index": i + 1,
                    "total": len(papers),
                    "pmid": paper.pmid,
                    "title": paper.title,
                    "year": paper.year,
                    "journal": paper.journal,
                })
                extracted = extractor.extract(paper)
                extracted_list.append(extracted)
                out_path = processed_dir / f"{paper.pmid}.json"
                import json as jj
                with open(out_path, "w") as f:
                    jj.dump(extracted.model_dump(), f, indent=2)
                push("progress", {
                    "step": 2,
                    "msg": f"Extracted limitations from {i+1}/{len(papers)}: {paper.title[:60]}",
                    "done": i + 1,
                    "total": len(papers),
                })

            push("progress", {"step": 3, "msg": "Chunking and embedding documents…"})
            builder = DocumentBuilder(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            documents = builder.build(extracted_list)

            push("progress", {"step": 4, "msg": f"Indexing {len(documents)} chunks into Qdrant…"})
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

            push("done", {
                "sid": session.sid,
                "n_papers": len(papers),
                "n_chunks": session.n_chunks,
                "msg": f"Ready. Indexed {len(papers)} papers ({session.n_chunks} chunks).",
            })

        except Exception as exc:
            logger.exception("Ingest error: {}", exc)
            push("error", {"msg": str(exc)})

    async def stream() -> AsyncGenerator:
        loop.run_in_executor(_executor, run_ingest)
        while True:
            kind, data = await queue.get()
            yield _event(kind, data)
            if kind in ("done", "error"):
                break

    return EventSourceResponse(stream())


@app.post("/api/ask")
async def ask(req: AskRequest):
    """Stream the answer to a single question via SSE."""
    if req.sid not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[req.sid]
    if not session.ingested:
        raise HTTPException(status_code=400, detail="Run ingestion first")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_ask():
        def push(kind, data): loop.call_soon_threadsafe(queue.put_nowait, (kind, data))
        try:
            push("progress", {"msg": "Searching knowledge base…"})
            result = session.analyzer.ask_with_sources(req.question)
            push("done", {
                "answer": result["answer"],
                "sources": result["sources"],
            })
        except Exception as exc:
            logger.exception("Ask error: {}", exc)
            push("error", {"msg": str(exc)})

    async def stream() -> AsyncGenerator:
        loop.run_in_executor(_executor, run_ask)
        while True:
            kind, data = await queue.get()
            yield _event(kind, data)
            if kind in ("done", "error"):
                break

    return EventSourceResponse(stream())


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Stream the full synthesis report via SSE."""
    if req.sid not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[req.sid]
    if not session.ingested:
        raise HTTPException(status_code=400, detail="Run ingestion first")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_synth():
        def push(kind, data): loop.call_soon_threadsafe(queue.put_nowait, (kind, data))
        try:
            push("progress", {"msg": "Retrieving all limitation chunks…"})
            report = session.analyzer.synthesize()
            push("done", {"report": report, "topic": session.topic})
        except Exception as exc:
            logger.exception("Synthesize error: {}", exc)
            push("error", {"msg": str(exc)})

    async def stream() -> AsyncGenerator:
        loop.run_in_executor(_executor, run_synth)
        while True:
            kind, data = await queue.get()
            yield _event(kind, data)
            if kind in ("done", "error"):
                break

    return EventSourceResponse(stream())
