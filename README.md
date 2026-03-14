# PubMed RAG Limitation Analyzer

A CLI and web-based tool that retrieves biomedical literature from PubMed, extracts research limitations using LLMs, indexes them in a vector store (Qdrant), and enables question-answering or report generation.

## Features

- **PubMed Retrieval** - Search by topic, date range, publication type, and max results
- **Synonym Expansion** - Automatic query expansion with biomedical synonyms
- **Combined Search** - Support for multiple topics/methods with logical operators (AND, OR, NOT)
- **Title-Only Search** - Queries search only in article titles for precision
- **Limitation Extraction** - Two-stage heuristic + LLM pipeline that pulls out:
  - Explicit study limitations
  - Identified research gaps
  - Suggested future work
  - Methodological weaknesses
- **Vector Storage** - Chunks are embedded and stored in Qdrant (in-memory or persistent)
- **RAG Pipeline** - Retrieve relevant chunks and synthesize structured limitations report
- **PDF Generation** - Export reports as professional PDF documents

## Quick Start

### 1. Setup

```bash
# Clone or navigate to project
cd pubmed-rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file (or copy from `.env.example`):

```env
# OpenAI API Key (required)
OPENAI_API_KEY=sk-...

# NCBI PubMed (optional but recommended)
NCBI_API_KEY=your_api_key
NCBI_EMAIL=your_email@example.com
```

Get NCBI API key: https://www.ncbi.nlm.nih.gov/account/

### 3. Usage

#### Web Interface (Recommended)

```bash
source .venv/bin/activate
python serve.py
```

Open browser to: **http://localhost:8000**

#### CLI Commands

```bash
# Ingest papers only
python main.py ingest --topic "breast cancer" --max-papers 10

# Ask a specific question
python main.py ask --topic "breast cancer" --question "What are the dataset limitations?"

# Generate full report
python main.py report --topic "breast cancer" --max-papers 15 --output report.md
```

## Advanced Features

### Combined Search

Combine multiple topics and methods:

| Field | Example | Description |
|-------|---------|-------------|
| Topic | `breast cancer detection` | Main research topic |
| Method/Technology | `deep learning` | Method to search for |
| Exclude Terms | `animal study, in vitro` | Terms to exclude |

**Example**: Search for AI in cancer detection
- **Topic**: `cancer detection`
- **Method**: `deep learning`
- **Result**: Papers about cancer detection using deep learning

### Query Expansion

The system automatically expands queries with synonyms:

| Input | Expanded Query |
|-------|----------------|
| `breast cancer` | `("breast cancer"[Title] OR "breast tumor"[Title] OR "mammary carcinoma"[Title])` |

### Knowledge Base Reset

Each new search creates a fresh knowledge base by default:
- Previous vectors are cleared
- Only current topic papers are indexed
- Toggle with "Reset knowledge base" checkbox in web interface

## API Reference

### Ingest Endpoint
```bash
POST /api/ingest
Body:
{
  "topic": "cancer detection",
  "method": "deep learning",
  "exclude_terms": ["animal study"],
  "max_papers": 10,
  "reset_knowledge_base": true
}
```

### Ask Endpoint
```bash
POST /api/ask
Body:
{
  "sid": "session-id",
  "question": "What are the main limitations?"
}
```

### Synthesize Endpoint
```bash
POST /api/synthesize
Body: {"sid": "session-id"}

POST /api/synthesize/pdf  # Returns PDF file
Body: {"sid": "session-id"}
```

## Project Structure

```
pubmed-rag/
├── main.py                      # CLI entry point
├── serve.py                     # Web server entry point
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── .env.example                 # Template for .env
├── config/
│   └── settings.py              # Pydantic settings loaded from .env
├── src/
│   ├── __init__.py
│   ├── logger.py                # Loguru logger configuration
│   ├── orchestrator.py          # Coordinates ingestion pipeline
│   ├── retriever/
│   │   ├── pubmed_client.py     # PubMed API client with synonym expansion
│   │   ├── synonym_expander.py  # Query expansion with synonyms
│   │   ├── models.py            # PaperMetadata, etc.
│   │   └── __init__.py
│   ├── extractor/
│   │   ├── section_extractor.py # Heuristic + LLM limitation extraction
│   │   ├── models.py            # Pydantic models for extracted data
│   │   └── __init__.py
│   ├── processor/
│   │   └── document_builder.py  # Document chunking logic
│   ├── rag/
│   │   ├── pipeline.py          # RAG pipeline for Q&A
│   │   └── prompts.py           # LLM prompts
│   └── vectorstore/
│       ├── qdrant_store.py      # Qdrant wrapper
│       └── models.py
├── web/
│   ├── api.py                   # FastAPI backend
│   ├── static/
│   │   └── index.html           # Web interface
│   └── __init__.py
└── tests/                       # Unit tests
```

## Data Flow

1. **Search** → PubMed API with title-only query + synonym expansion
2. **Extract** → LLM identifies limitations/gaps/future work from papers
3. **Chunk** → Documents are split into searchable chunks
4. **Index** → Chunks embedded and stored in Qdrant vector store
5. **Query** → RAG retrieves relevant chunks and generates answers
6. **Export** → Reports generated as Markdown or PDF

## Requirements

- Python 3.10+
- OpenAI API key
- NCBI API key (optional, for higher rate limits)
- Qdrant (in-memory by default, or persistent server)

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
lsof -i :8000
kill -9 <PID>
```

**API key not set:**
Check `.env` file has valid `OPENAI_API_KEY`

**Rate limits:**
Add NCBI API key to `.env` for 10 req/s vs 3 req/s

**Memory issues:**
Use persistent Qdrant server instead of in-memory mode

## Development

### Run Tests
```bash
pytest tests/
```

### Add Synonyms
Edit `src/retriever/synonym_expander.py` to add custom synonyms

### Modify Prompts
Edit `src/rag/prompts.py` to customize LLM behavior

## License

MIT License
