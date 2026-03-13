# PubMed RAG Limitation Analyzer

A CLI tool that retrieves biomedical literature from PubMed, extracts research limitations using LLMs, indexes them in a vector store (Qdrant), and enables question‑answering or report generation.

## Features

- **PubMed retrieval** – search by topic, date range, publication type, and max results (uses your NCBI API key if provided).
- **Limitation extraction** – two‑stage heuristic + LLM pipeline that pulls out:
  - Explicit study limitations
  - Identified research gaps
  - Suggested future work
  - Methodological weaknesses
- **Vector storage** – chunks are embedded (OpenAI embeddings) and stored in Qdrant (in‑memory by default, configurable to a remote instance).
- **RAG pipeline** – retrieve relevant chunks and synthesize a structured limitations report or answer specific questions with citations.
- **CLI commands**:
  - `ingest` – fetch papers, extract limitations, and index them.
  - `ask` – ingest (if needed) then answer a question with source citations.
  - `report` – ingest + generate a full limitations report (Markdown output).

## Project Structure

```
pubmed-rag/
├── .env                 # environment variables (API keys, settings)
├ .env.example          # template for .env
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
├── config/
│   └── settings.py      # Pydantic settings loaded from .env
├── src/
│   ├── __init__.py
│   ├── logger.py        # Loguru logger configuration
│   ├── orchestrator.py  # coordinates ingestion pipeline
│   ├── extractor/
│   │   ├── __init__.py
│   │   ├── models.py    # Pydantic models for extracted data
│   │   └── section_extractor.py  # heuristic + LLM limitation extraction
│   ├── llm/
│   │   └── ...          # (if needed) LLM wrappers
│   ├── processor/
│   │   └── document_builder.py   # chunking logic
│   ├── rag/
│   │   └── pipeline.py      # Retrieval‑Augmented Generation for synthesis/QA
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── pubmed_client.py   # NCBI E‑utils wrapper with caching & full‑text fetch
│   │   └── models.py          # PaperMetadata, etc.
│   └── vectorstore/
│       ├── __init__.py
│       ├── qdrant_store.py    # Qdrant wrapper (in‑memory or remote)
│       └── models.py
├── data/
│   ├── raw/                 # downloaded XML/JSON from PubMed (optional)
│   └── processed/           # intermediate artifacts (optional)
├── logs/                    # log files (Loguru)
├── tests/                   # pytest tests
└── web/                     # (placeholder) for a future web interface
```

## Setup

1. **Clone the repository** (or copy the files).

2. **Create a Python virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Copy `.env.example` to `.env` and fill in the required keys:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env`:
     ```dotenv
     # OpenAI (required for LLM and embeddings)
     OPENAI_API_KEY=sk-...          # your OpenAI API key
     OPENAI_MODEL=gpt-4o-mini       # optional, default
     OPENAI_EMBEDDING_MODEL=text-embedding-3-small

     # NCBI / PubMed (optional but recommended to avoid rate limits)
     NCBI_API_KEY=your_ncbi_key     # get at https://www.ncbi.nlm.nih.gov/account/
     NCBI_EMAIL=your_email@example.com

     # Qdrant (leave as-is for local in‑memory mode)
     QDRANT_HOST=localhost
     QDRANT_PORT=6333
     # QDRANT_API_KEY=your_qdrant_cloud_key  # only if using Qdrant Cloud

     # Logging
     LOG_LEVEL=INFO                 # DEBUG for verbose extraction logs

     # Data paths (defaults are fine)
     RAW_DATA_DIR=data/raw
     PROCESSED_DATA_DIR=data/processed
     ```

5. **Initialize directories** (done automatically on first run via `settings.ensure_dirs()`).

## Usage

All commands are executed via `main.py`.  
Make sure the virtual environment is activated (`source .venv/bin/activate`).

### Common options

| Option            | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `--topic`         | Research topic to search for (required).                                    |
| `--date-from`     | Start year (inclusive), default `2020`.                                     |
| `--date-to`       | End year (inclusive), default current year.                                 |
| `--paper-type`    | Filter by PubMed publication type (`review`, `clinical_trial`, etc.).       |
| `--max-papers`    | Maximum number of papers to retrieve and process (default `10`).            |
| `--preview`       | (only for `ingest`) show a quick synthesis after indexing.                |
| `--output FILE`   | (only for `report`) save the generated Markdown report to `FILE`.         |

### Commands

#### 1. Ingest only
Fetch papers, extract limitations, and store them in the vector store.
```bash
python main.py ingest \
    --topic "breast cancer detection" \
    --date-from 2020 \
    --date-to 2025 \
    --max-papers 15
```

Add `--preview` to see a short limitations report after ingestion.

#### 2. Ask a question
Perform ingestion (if the topic isn’t already indexed) then answer a specific question with citations.
```bash
python main.py ask \
    --topic "breast cancer detection" \
    --question "What are the main dataset limitations reported in the literature?" \
    --max-papers 12
```

The answer will be returned in Markdown, followed by a list of source papers (PMID, year, title, PubMed URL).

#### 3. Full report
Ingest and generate a structured limitations report (Markdown) that groups findings by category (Dataset Limitations, Methodological Weaknesses, etc.).
```bash
python main.py report \
    --topic "deep learning MRI" \
    --date-from 2021 \
    --date-to 2025 \
    --max-papers 20 \
    --output report.md
```

The report will be printed to stdout and, if `--output` is given, also written to the specified file.

## How It Works (High‑Level)

1. **Retrieval** – `PubMedClient` uses NCBI’s E‑utils (`esearch` + `efetch`) to get article metadata (PMID, title, abstract, journal, year, etc.). If an NCBI API key and email are provided, full‑text XML from PMC is requested when available.

2. **Section extraction** – `LimitationExtractor` first tries to pull named sections (Limitations, Discussion, Future Work, Conclusions) via regex heuristics on the full text. If insufficient text is found, it falls back to the abstract (or the first 6000 chars of full text). The selected text is then sent to an LLM (GPT‑4o‑mini by default) with a prompt that instructs it to return a JSON object containing four lists: `limitations`, `research_gaps`, `future_work`, `methodological_weaknesses`. The LLM output is validated and attached to a `PaperMetadata` record.

3. **Chunking & embedding** – Each paper’s extracted fields (and optionally the raw text) are turned into `Document` objects (via `langchain_core.documents.Document`). A `RecursiveCharacterTextSplitter` splits them into chunks (~1000 chars with 200 overlap). Chunks are embedded using the OpenAI embedding model.

4. **Vector storage** – Chunks are added to a Qdrant collection named `research_limitations`. By default Qdrant runs in‑memory (no server needed); point `QDRANT_HOST`/`QDRANT_PORT` to a remote instance for persistence.

5. **RAG synthesis / QA** – 
   - For a **report**, the pipeline retrieves the top‑k chunks (default 8) for the given topic, concatenates them, and asks the LLM to produce a structured markdown report grouped by limitation categories, citing sources.
   - For **question answering**, the same retrieval step is performed, then the LLM answers the question using only the retrieved context, again returning inline citations (PMID, year).

## Logging

- Log level is controlled by `LOG_LEVEL` in `.env` (`INFO` by default).
- Detailed debug logs (including the raw text sent to the LLM and the LLM’s raw JSON response) appear when `LOG_LEVEL=DEBUG`.
- Logs are written to `logs/pubmed_rag.log` (rotating every 10 MB, kept 7 days) and also echoed to stderr with colours (via Loguru).

## Extending / Customizing

- **Change LLM** – edit `src/extractor/section_extractor.py` (model name, temperature) or `src/rag/pipeline.py`.
- **Adjust chunking** – modify `chunk_size` / `chunk_overlap` in `config/settings.py` or directly in `src/processor/document_builder.py`.
- **Different vector store** – replace `src/vectorstore/qdrant_store.py` with another implementation (FAISS, Chroma, Pinecone, etc.) while keeping the same interface (`add_documents`, `similarity_search`).
- **Additional extraction fields** – extend `_LLMExtractionSchema` and the prompt in `section_extractor.py`, then propagate changes through the orchestrator and RAG pipeline.

## Testing

Run the test suite with:
```bash
pytest
```
Tests live in the `tests/` directory and cover:
- PubMed client parsing
- Extraction heuristic and LLM fallback (mocked)
- Chunking logic
- Qdrant wrapper (in‑memory)
- End‑to‑end CLI commands (using small mocks)

## License

This project is provided as‑is for educational and research purposes. See the LICENSE file (if present) for details.

---

**Enjoy analyzing research limitations!** If you encounter any issues, check the logs (`logs/pubmed_rag.log`) or run with `LOG_LEVEL=DEBUG` for more insight.
