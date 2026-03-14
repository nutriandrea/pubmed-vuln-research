# PubMed RAG Limitation Analyzer

Extract and analyze research limitations from PubMed papers using RAG.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Run

### Web Interface
```bash
python serve.py
# Open http://localhost:8000
```

### CLI
```bash
python main.py ingest --topic "cancer" --max-papers 10
python main.py ask --topic "cancer" --question "What are limitations?"
python main.py report --topic "cancer" --max-papers 10 --output report.md
```

## Deploy (Docker)
```bash
docker build -t pubmed-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key pubmed-rag
```

Or use Render.com with Docker environment.

## Environment
Required: `OPENAI_API_KEY`
Optional: `NCBI_API_KEY`, `NCBI_EMAIL`
