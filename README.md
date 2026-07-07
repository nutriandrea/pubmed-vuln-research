[![Python](https://img.shields.io/badge/Python-3.10+-3776AB)]
[![RAG](https://img.shields.io/badge/RAG-LLM-FF6F00)]
[![PubMed](https://img.shields.io/badge/PubMed-API-4263F5)]
[![License](https://img.shields.io/badge/License-MIT-yellow)]

# PubMed RAG Limitation Analyzer

**Automated vulnerability-discovery pipeline** that mines PubMed abstracts for research limitations — then cross-references them against CVE databases, exploit feeds, and vendor advisories to surface high-priority leads for security researchers.

No more manually reading hundreds of papers to find which protocol, library, or device has a known weakness you can exploit. Let the RAG pipeline do the first pass.

## How It Works

```
PubMed search → 10,000+ abstracts ingested → Chunk + embed → Vector DB
                                                        │
User query: "TLS 1.3 handshake vulnerability" ←────────┘
                                                        │
                                                ┌───────┴────────┐
                                                │  LLM synthesizes│
                                                │  + CVE lookup   │
                                                └───────┬────────┘
                                                        │
                                                Report: PMIDs, CVSS scores,
                                                affected packages, PoC references
```

## What makes it different

| Feature | PubMed RAG Limitation Analyzer | Manual search |
|---------|-------------------------------|--------------|
| Papers scanned per query | 10,000+ | ~50 |
| Cross-references CVE + ExploitDB | ✅ Automated | ❌ Manual |
| Limitations extracted by LLM | ✅ Structured | ❌ Skim only |
| Time per research question | ~30 seconds | 2–4 hours |
| Reproducible pipeline | ✅ `make run` | ❌ Ad-hoc |

## Quick Start

```bash
pip install -r requirements.txt
python analyzer.py --query "WiFi CSI side-channel attack" --top-k 50
```

Output: a structured report with relevant PMIDs, extracted limitations, CVSS scores, and exploit availability.

## Use Cases

- **CVE discovery prep** — find papers that describe the *exact conditions* under which a system fails, then test if those conditions apply to unpatched software
- **IoT vulnerability research** — PubMed indexes medical devices, implantable sensors, smart home protocols — all rich targets
- **Protocol weakness mining** — TLS, BLE, Zigbee, 802.11 — papers often publish the attack before vendors patch
- **Academic-security bridge** — turn literature review into actionable bug-hunt tickets

## Stack

| Layer | Tech |
|-------|------|
| Retrieval | Sentence-BERT + FAISS vector DB |
| LLM | GPT-4 / Gemini for synthesis + CVE matching |
| Sources | PubMed (via Biopython E-utilities), CVE API, ExploitDB |
| Pipeline | Python 3.10+, LangChain, Chroma |

## Why PubMed?

Medical and IoT security research is uniquely suited to literature-mining because:
1. Implantable devices (pacemakers, insulin pumps) have published radio/crypto analyses
2. Hospital network protocols (HL7, DICOM) have documented weaknesses
3. Sensor fusion papers often reveal side-channels nobody has exploited yet
4. Authors are incentivized to publish *limitations* — which are exactly the attack surface
