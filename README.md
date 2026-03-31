# PubMed RAG Limitation Analyzer

## 🎯 Obiettivo

Sistema di **Retrieval-Augmented Generation (RAG)** per estrarre, analizzare e interrogare automaticamente le **limitazioni della ricerca** da migliaia di paper scientifici su PubMed.

Il sistema risponde a domande come:
- "Quali sono le principali limitazioni negli studi sulla distrofia endoteliale di Fuchs?"
- "Quali gap di ricerca esistono nel campo dei trapianti corneali?"
- "Quali sono le debolezze metodologiche comuni negli studi clinici?"

---

## 🏗️ Architettura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                │
│                         (HTML + JavaScript)                            │
│                   http://localhost:8000                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                                 │
│                           (app/api.py)                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │
│  │ /api/ingest │  │  /api/ask   │  │/api/synth  │                   │
│  └─────────────┘  └─────────────┘  └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     BUSINESS LOGIC (src/)                               │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │   RETRIEVER  │───▶│  EXTRACTOR   │───▶│  PROCESSOR   │            │
│  │ (PubMed)     │    │  (LLM/Hybrid)│    │ (Chunking)   │            │
│  └──────────────┘    └──────────────┘    └──────────────┘            │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │                  VECTOR STORE (Qdrant)                   │          │
│  │              Embeddings per ricerca similarity           │          │
│  └──────────────────────────────────────────────────────────┘          │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │                    RAG PIPELINE                            │          │
│  │          Query → Retrieve → Generate Response             │          │
│  └──────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Flusso di Esecuzione Dettagliato

### Fase 1: Ricerca PubMed

```
1.1 Costruzione Query
    - Cerca in Title/Abstract (ottimizzato per recall)
    - Espansione sinonimi (dizionario biomedico)
    - Filtri: date, tipo pubblicazione

1.2 Recupero PMIDs
    - API Entrez (NCBI)
    - Paginazione con WebEnv/QueryKey
    - Batch di 10.000 ID per chiamata
    - Rate limit: 10 req/sec (con API key)

1.3 Download Metadata
    - Per ogni PMID: titolo, autori, abstract, anno
    - Full text da PMC se disponibile
    - Cache locale (JSON) per evitare re-download
```

### Fase 2: Estrazione Limitazioni (APPROCCIO IBRIDO)

```
2.1 Rule-Based Extraction (Regex)
    - Pattern per sezioni: Limitations, Discussion, Future Work
    - 11 pattern ottimizzati per priorità
    - ~50ms per paper

2.2 LLM Extraction (GPT-4o-mini)
    - Analizza solo le sezioni estratte
    - Identifica: limitazioni, gap, debolezze metodologiche
    - Testo troncato a 2000 caratteri per risparmiare token

2.3 Ottimizzazione Token
    - Prima: ~3500 parole per paper
    - Dopo: ~500 parole per paper
    - Risparmio: ~85% token
```

### Fase 3: Processamento e Indicizzazione

```
3.1 Chunking
    - Suddivisione in chunk (~1000 caratteri)
    - Overlap 200 caratteri per continuità

3.2 Embeddings
    - OpenAI text-embedding-3-small
    - ~10 chunk per paper

3.3 Indicizzazione Qdrant
    - Vector store per similarity search
    - In-memory o persistente
```

### Fase 4: Query RAG

```
4.1 Recupero
    - Embedding della domanda
    - Top-k chunk più simili

4.2 Generazione
    - LLM risponde basandosi sui chunk recuperati
    - Include citazioni ai paper originali
```

---

## ⚡ Ottimizzazioni Attuali

### 1. Parallelismo PubMed

```python
# Download parallelo con 3-5 thread
with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_pmid = {executor.submit(_fetch_paper, pmid): pmid for pmid in pmids}
```

**Beneficio**: 3-5x più veloce del sequenziale

### 2. Approccio Ibrido (Regex + LLM)

```
Sezioni estratte con regex (~50ms)
    ↓
Testo troncato a 2000 caratteri
    ↓
LLM analizza solo sezioni rilevanti
```

**Beneficio**: ~85% meno token, 5x più veloce

### 3. Server-Sent Events (SSE)

```javascript
// Aggiornamenti in tempo reale
event: progress
data: {"percent": 45, "msg": "Extracting 45/100 papers"}
```

**Beneficio**: UI responsiva con barra progresso reale

### 4. Cache Locale

```
data/raw/<topic>/<pmid>.json  # Metadata cache
data/processed/<topic>/        # Extractions cache
```

**Beneficio**: Evita re-download paper già processati

---

## 🚀 Approcci per Processare 10,000 Paper

### Stima Tempi Attuali (Sequenziale)

| Fase | Operazione | Tempo | Totale |
|------|-----------|-------|--------|
| Ricerca PubMed | 10,000 ID | ~10 sec | 10 sec |
| Download Metadata | Per PMID | ~0.35s | ~1 min |
| **Estrazione LLM** | **1 call/paper** | **~5s** | **~14 ore** ⚠️ |
| Embeddings | ~10 chunk/paper | ~0.1s | ~3 ore |
| Indicizzazione Qdrant | 100k vectors | 0.01s | ~17 min |

**Totale: ~18 ore**

### Con Ottimizzazioni Proposte

| Strategia | Tempo | Costo API | Accuratezza |
|----------|-------|-----------|-------------|
| **Batch LLM (50/paper)** | ~25 min | ~$2 | ★★★★☆ |
| **Ollama GPU** | ~30 min | $0 | ★★★★☆ |
| **Rule-based only** | ~2 min | $0 | ★★☆☆☆ |
| **Cache (run successivi)** | ~5 min | ~$0.50 | ★★★★★ |

---

## 📋 Implementazione Ottimizzazioni Future

### 1. Batch LLM (Priorità Alta)

```python
# Processare 50 paper in una chiamata
def extract_batch(papers: list[PaperMetadata]) -> list[ExtractedLimitations]:
    """
    Estrae limitazioni per più paper in una singola chiamata LLM.
    
    Input: 50 paper
    Output: 50 oggetti ExtractedLimitations
    """
    # Costruisce prompt con tutti i paper
    prompt = build_batch_prompt(papers)
    
    # Una chiamata LLM invece di 50
    response = llm.invoke(prompt)
    
    # Parse e ritorna lista
    return parse_batch_response(response, papers)
```

**Beneficio**: 50x meno chiamate, ~98% risparmio costi

### 2. LLM Locale con Ollama (Priorità Media)

```bash
# Installare Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Scaricare modello
ollama pull llama3.2:3b

# Usare nel codice
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2:3b")
```

**Beneficio**: Gratuito, nessun rate limit, privacy

### 3. Distributed Workers (Priorità Bassa)

```python
# Pipeline parallela con Redis
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def extract_limitations(pmid_batch):
    # Processa batch in worker separato
    return extractor.extract_batch(pmid_batch)
```

### 4. Cache Intelligente (Da Implementare)

```python
class ExtractionCache:
    """Cache per limitazioni già estratte."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.index_file = cache_dir / "index.json"
    
    def get(self, pmid: str) -> Optional[ExtractedLimitations]:
        """Recupera da cache se esiste."""
        cache_file = self.cache_dir / f"{pmid}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return ExtractedLimitations(**json.load(f))
        return None
    
    def set(self, pmid: str, extraction: ExtractedLimitations):
        """Salva in cache."""
        cache_file = self.cache_dir / f"{pmid}.json"
        with open(cache_file, 'w') as f:
            json.dump(extraction.model_dump(), f, indent=2)
        
        # Aggiorna index
        index = self._load_index()
        index[pmid] = str(cache_file)
        self._save_index(index)
    
    def get_all(self, pmids: list[str]) -> tuple[list, list]:
        """Separa cache hit da miss."""
        cached = []
        missing = []
        for pmid in pmids:
            result = self.get(pmid)
            if result:
                cached.append(result)
            else:
                missing.append(pmid)
        return cached, missing
```

### 5. Streaming Updates (Implementato)

```python
@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """Invia aggiornamenti SSE per progresso in tempo reale."""
    
    async def event_generator():
        # Ricerca PubMed
        yield sse_event("status", "Searching...")
        pmids = await search_pubmed(req.topic)
        yield sse_event("pmid_count", {"total": len(pmids)})
        
        # Processing con progressi
        for i, batch in enumerate(batches(pmids, 10)):
            yield sse_event("progress", {
                "percent": int(i / len(batches) * 100),
                "msg": f"Processing {i}/{len(batches)}"
            })
            
            # Check cache per ogni batch
            cached, missing = cache.get_all(batch)
            new_extractions = extractor.extract_batch(missing)
            
            # Salva in cache
            for ext in new_extractions:
                cache.set(ext.pmid, ext)
        
        yield sse_event("complete", {"n_papers": len(pmids)})
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 📊 Stime Complete per 10,000 Paper

### Scenario 1: Batch LLM + Cache

| Fase | Tempo | Note |
|------|-------|------|
| Ricerca PubMed | 10 sec | 1 chiamata |
| Download Metadata (batch) | 1 min | 100 chiamate parallele |
| Cache Check | 1 sec | Skip paper già processati |
| Estrazione LLM (batch 50) | 15 min | 200 chiamate |
| Embeddings | 5 min | Parallelo |
| Indicizzazione | 2 min | Qdrant |
| **Totale (primo run)** | **~25 min** | |
| **Totale (con cache)** | **~5 min** | Solo nuovi paper |

### Scenario 2: Ollama GPU + Batch

| Fase | Tempo | Note |
|------|-------|------|
| Ricerca + Download | 2 min | |
| Estrazione LLM (locale) | 5 min | GPU RTX 3080 |
| Embeddings + Indicizzazione | 7 min | |
| **Totale** | **~15 min** | Nessun costo API |

### Scenario 3: Rule-Based Only

| Fase | Tempo | Note |
|------|-------|------|
| Ricerca + Download | 2 min | |
| Estrazione Regex | 30 sec | ~0.01s/paper |
| Embeddings + Indicizzazione | 7 min | |
| **Totale** | **~10 min** | Gratuito, veloce |

---

## 🔮 Potenziali Miglioramenti Futuri

### Alta Priorità
- [ ] **Batch LLM**: Processare 50 paper per chiamata
- [ ] **Cache intelligente**: Skip paper già estratti
- [ ] **Incrementale updates**: Solo paper nuovi/aggiornati

### Media Priorità
- [ ] **Ollama integration**: LLM locale gratuito
- [ ] **Multi-lingua**: Supporto italiano, spagnolo
- [ ] **Filtri avanzati**: Impact factor, citazioni, lingua

### Bassa Priorità
- [ ] **Distributed workers**: Pipeline parallela cloud
- [ ] **Web scraping**: PDF pubblici senza PMC
- [ ] **Fine-tuning**: Modello specializzato su limitazioni

---

## 📝 Limitazioni Attuali

### 1. Rate Limiting NCBI
- **Problema**: 10 req/sec con API key
- **Workaround**: Batch requests, cache locale

### 2. Costo API OpenAI
- **Problema**: ~$0.05 per 100 paper
- **Workaround**: Batch, Ollama, cache

### 3. Lingua
- **Problema**: Ottimizzato per inglese
- **Workaround**: Multi-lingua in sviluppo

### 4. Full Text Availability
- **Problema**: Non tutti i paper hanno PMC
- **Workaround**: Abstract come fallback

---

## 🛠️ Installazione e Utilizzo

### Requisiti
- Python 3.10+
- OpenAI API key
- NCBI API key (opzionale)
- Qdrant (locale o cloud)

### Setup
```bash
# Clona repository
git clone https://github.com/andreacacioppo/pubmed-rag.git
cd pubmed-rag

# Crea venv
python -m venv .venv
source .venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt

# Configura ambiente
cp .env.example .env
# Edita .env con le tue API keys

# Avvia server
python serve.py
```

### Utilizzo
1. Apri http://localhost:8000
2. Inserisci topic (es. "corneal endothelial failure")
3. Imposta date range e max papers
4. Clicca "Search & Ingest"
5. Attendi completamento (vedi progresso in tempo reale)
6. Fai domande sul topic
7. Genera report PDF

---

## 📚 Struttura File

```
pubmed-rag/
├── app/                    # Backend FastAPI
│   ├── api.py             # Endpoints
│   ├── services/
│   │   └── pdf_service.py # Generazione PDF
│   └── core/
├── src/                   # Logica di business
│   ├── retriever/        # PubMed client
│   ├── extractor/        # Estrazione limitazioni
│   ├── processor/        # Chunking
│   ├── vectorstore/      # Qdrant integration
│   └── rag/             # Pipeline RAG
├── static/               # Frontend
├── config/               # Settings
├── data/                 # Cache (non committare!)
├── main.py              # CLI
└── serve.py            # Web server
```

---

## 📜 Licenza

MIT License

---

## 🙏 Riferimenti

- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/home/develop/api/)
- [Biopython Entrez](https://biopython.org/docs/latest/Tutorial/chapter_entrez.html)
- [LangChain](https://python.langchain.com/)
- [Qdrant](https://qdrant.tech/)
- [FPDF2](https://py-pdf.github.io/fpdf2/)
