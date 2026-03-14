# Web Interface Guide

## Starting the Server

```bash
source .venv/bin/activate
python serve.py
```

Access at: **http://localhost:8000**

## Using the Interface

### Step 1: Enter Search Criteria

1. **Research Topic** (required)
   - Enter your main research topic
   - Examples: "breast cancer", "diabetes detection", "heart disease"
   - The system automatically searches in titles only and expands with synonyms

2. **Method/Technology** (optional)
   - Enter a method or technology to combine with the topic
   - Examples: "deep learning", "machine learning", "AI"
   - Combined query: Topic AND Method

3. **Exclude Terms** (optional)
   - Comma-separated terms to exclude
   - Examples: "animal study, in vitro, mouse"
   - These are excluded with NOT operator

4. **Year Range**
   - Default: 2020-2025
   - Adjust as needed

5. **Max Papers**
   - Number of papers to retrieve
   - Default: 5
   - Range: 1-50

6. **Reset Knowledge Base**
   - Checked by default
   - Clears previous results for fresh search

### Step 2: Search & Ingest

Click **"🔍 Search & Ingest"** to:
1. Search PubMed with your criteria
2. Extract limitations from papers
3. Index in vector store
4. Enable Q&A and report generation

### Step 3: Ask Questions

Once ingestion completes:
1. Switch to **"Ask Questions"** tab
2. Enter your question about limitations
3. Click **"Send"** or press Enter
4. View answer with source citations

**Example questions:**
- "What are the main dataset limitations?"
- "What methodologies have weaknesses?"
- "What future research is suggested?"

### Step 4: Generate Report

1. Switch to **"Report"** tab
2. Click **"Generate Report"**
3. View structured limitations report
4. Download as Markdown or PDF

## Tabs Overview

### Progress Tab
- Shows real-time ingestion logs
- Displays paper extraction status
- Lists indexed papers

### Ask Questions Tab
- Chat interface for Q&A
- Answers with source citations
- Shows paper titles and years

### Report Tab
- Structured limitations report
- Categorized by:
  - Dataset Limitations
  - Methodological Weaknesses
  - Reproducibility Issues
  - Research Gaps
  - Future Directions

## Example Workflows

### Workflow 1: Basic Search
1. Topic: `cancer detection`
2. Max Papers: 5
3. Click Search & Ingest
4. Ask: "What are the main limitations?"
5. Download PDF report

### Workflow 2: Combined Search
1. Topic: `breast cancer`
2. Method: `deep learning`
3. Exclude: `animal study`
4. Max Papers: 10
5. Click Search & Ingest
6. Download PDF report

### Workflow 3: Specific Question
1. Topic: `diabetes`
2. Ingest papers
3. Ask: "What are the data quality issues?"
4. View answer with sources

## Tips

- **Clear Session**: Click "Clear Session" to start fresh
- **Reset Knowledge**: Check "Reset knowledge base" for new searches
- **Download**: Use "PDF" for formatted report, "Markdown" for raw text
- **Sources**: Each answer includes citation links to PubMed

## Troubleshooting

**No papers found:**
- Broaden your topic
- Extend date range
- Remove exclusion terms

**Slow ingestion:**
- Reduce max papers
- Use simpler topics

**PDF not downloading:**
- Check browser pop-up blocker
- Try Markdown download instead

## API Access

The web interface uses these endpoints:
- `POST /api/ingest` - Start ingestion
- `POST /api/ask` - Ask questions
- `POST /api/synthesize` - Generate report
- `POST /api/synthesize/pdf` - Generate PDF
- `GET /api/session/{sid}` - Check session status
- `DELETE /api/session/{sid}` - Clear session

Access API docs at: **http://localhost:8000/docs**
