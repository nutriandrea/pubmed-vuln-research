[![Python](https://img.shields.io/badge/Python-3.10+-3776AB)](https://www.python.org/)
[![RAG](https://img.shields.io/badge/RAG-Qdrant-FF6F00)](https://qdrant.tech/)
[![PubMed](https://img.shields.io/badge/PubMed-E--utilities-4263F5)](https://www.ncbi.nlm.nih.gov/home/develop/api/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web_UI-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

# PubMed RAG Limitation Analyzer

**Automated vulnerability-discovery pipeline** that mines PubMed abstracts for research limitations — then cross-references them against CVE databases, exploit feeds, and vendor advisories to surface high-priority leads for security researchers.

No more manually reading hundreds of papers to find which protocol, library, or device has a known weakness you can exploit. Let the RAG pipeline do the first pass.

## How It Works

```
PubMed search → 10,000+ abstracts ingested → Chunk + embed → Vector DB (Qdrant)
                                                                │
User query: "TLS 1.3 handshake vulnerability" ←─────────────────┘
                                                                │
                                                        ┌───────┴────────┐
                                                        │  LLM synthesizes│
                                                        │  + CVE lookup   │
                                                        └───────┬────────┘
                                                                │
                                                        Report: PMIDs, CVSS scores,
                                                        affected packages, PoC references
```

## What Makes It Different

| Feature | PubMed RAG Limitation Analyzer | Manual Search |
|---|---|---|
| Papers scanned per query | 10,000+ | ~50 |
| Cross-references CVE + ExploitDB | Automated | Manual only |
| Limitations extracted by LLM | Structured | Skim only |
| Time per research question | ~30 seconds | 2–4 hours |
| Reproducible pipeline | `python main.py report ...` | Ad-hoc |

## Features

### CLI Pipeline (`main.py`)

Three subcommands for the complete workflow:

```bash
# Ingest papers into vector store
python main.py ingest --topic "WiFi CSI side-channel attack" \
  --date-from 2020 --date-to 2025 --max-papers 10 --preview

# Ask a specific question (auto-ingests if needed)
python main.py ask --topic "deep learning MRI" \
  --question "What are the main dataset limitations?" \
  --max-papers 15

# Full report: ingest + synthesize with sources
python main.py report --topic "TLS 1.3 handshake" \
  --date-from 2018 --date-to 2025 --max-papers 20 \
  --output report.md
```

### Web UI (FastAPI)

```bash
python serve.py
# → http://localhost:8000
```

The web UI provides:
- Search interface with topic, date range, paper type filters
- Interactive limitation explorer with source citations
- PDF report generation
- Session management for multiple concurrent analyses

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/search` | POST | Search and ingest papers |
| `/api/ask` | POST | Ask a specific question |
| `/api/report` | POST | Generate full limitation report |
| `/api/report/{id}/pdf` | GET | Download report as PDF |
| `/` | GET | Web UI (HTML) |

## Stack

| Layer | Technology |
|---|---|
| **Retrieval** | OpenAI `text-embedding-3-small` → Qdrant vector DB |
| **LLM** | GPT-4o-mini for synthesis + CVE matching |
| **Sources** | PubMed (via Biopython E-utilities), CVE API, ExploitDB |
| **Web UI** | FastAPI + HTMX + static assets |
| **Pipeline** | LangChain (document chains, QA chains, text splitters) |
| **Container** | Docker + Gunicorn + Uvicorn |

## Use Cases

- **CVE discovery prep** — find papers that describe the *exact conditions* under which a system fails, then test if those conditions apply to unpatched software
- **IoT vulnerability research** — PubMed indexes medical devices, implantable sensors, smart home protocols — all rich targets
- **Protocol weakness mining** — TLS, BLE, Zigbee, 802.11 — papers often publish the attack before vendors patch
- **Academic-security bridge** — turn literature review into actionable bug-hunt tickets

## Why PubMed?

Medical and IoT security research is uniquely suited to literature-mining because:
1. Implantable devices (pacemakers, insulin pumps) have published radio/crypto analyses
2. Hospital network protocols (HL7, DICOM) have documented weaknesses
3. Sensor fusion papers often reveal side-channels nobody has exploited yet
4. Authors are incentivized to publish *limitations* — which are exactly the attack surface

## Quick Start

```bash
# Clone
git clone https://github.com/nutriandrea/pubmed-vuln-research.git
cd pubmed-vuln-research

# Environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY

# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run CLI
python main.py report --topic "WiFi CSI" --max-papers 20

# Or web UI
python serve.py
```

### Docker

```bash
docker build -t pubmed-rag .
docker run -p 8000:8000 --env-file .env pubmed-rag
```

## Project Structure

```
├── main.py              # CLI entry point (ingest/ask/report)
├── serve.py             # FastAPI web server
├── config/
│   └── settings.py      # Central configuration (pydantic-settings)
├── app/
│   ├── api.py           # FastAPI routes (350 lines)
│   ├── services/        # PDF generation, helpers
│   └── core/            # Business logic
├── src/
│   ├── orchestrator.py  # ResearchLimitationAnalyzer (pipeline coordinator)
│   ├── extractor/       # PubMed paper extraction
│   ├── processor/       # Text chunking and processing
│   ├── rag/             # RAG query construction
│   ├── retriever/       # Vector search retrieval
│   ├── vectorstore/     # Qdrant integration
│   └── llm/             # LLM interaction layer
├── static/              # Web UI assets
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
└── Procfile             # Heroku/deployment config
```

## Architecture

The `ResearchLimitationAnalyzer` orchestrator manages the full pipeline:

1. **Ingest** — Fetch papers from PubMed via Biopython E-utilities, filter by date/type
2. **Chunk** — Split abstracts into overlapping chunks (LangChain text splitters)
3. **Embed** — Generate embeddings via OpenAI `text-embedding-3-small`
4. **Store** — Index in Qdrant vector database
5. **Retrieve** — Semantic search over indexed chunks
6. **Synthesize** — LLM generates structured limitation report with source citations
7. **Report** — Output as Markdown, JSON, or PDF

## License

MIT
