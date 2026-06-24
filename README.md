# Clinisight

**AI-powered clinical decision support** — symptom extraction, differential diagnosis, and evidence-based PubMed literature synthesis.

> ⚠️ **Medical Disclaimer:** Clinisight is for informational and educational purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## Features

- **Intelligent symptom extraction** — LLM-based NLP extracts and normalizes symptoms from free-text patient descriptions (not limited to a keyword list)
- **Structured differential diagnosis** — GPT-4o generates top 3–5 differential diagnoses with clinical reasoning, ICD-10 hints, urgency assessment, and recommended workup
- **PubMed integration** — Searches NCBI PubMed E-utilities for the latest relevant literature; returns title, abstract, authors, journal, date, MeSH terms, and URL
- **Literature synthesis** — Synthesizes multiple articles into key findings, evidence quality, and clinical implications
- **Dual interface** — REST API (FastAPI) and MCP tool server for Claude integration

---

## Architecture

```
Patient description (free text)
        │
        ▼
 [1] symptom_extractor.py      ← GPT-4o-mini: NLP extraction + normalization
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
 [2] diagnosis_symptoms.py                   PubMed E-utilities
     GPT-4o differential diagnosis           (NCBI REST API)
        │                                              │
        │                                    [3] pubmed_articles.py
        │                                    Fetch + parse XML metadata
        │                                              │
        │                                    [4] summarize_pubmed.py
        │                                    GPT-4o-mini synthesis
        │                                              │
        └──────────────────────────────────────────────┘
                         │
                         ▼
              Structured JSON response
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install uv
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

`.env`:
```
OPENAI_API_KEY=sk-...
```

### 3. Run the API

```bash
uv run python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

API docs available at: http://localhost:8080/docs

### 4. Run as MCP tool (for Claude)

```bash
mcp install mcp_tool.py
# or for development:
uv run mcp dev mcp_tool.py
```

---

## API Reference

### `POST /diagnosis`

Full clinical analysis pipeline.

**Request:**
```json
{
  "description": "I've had a severe headache for 2 days, along with fever and neck stiffness.",
  "max_articles": 3
}
```

**Response:**
```json
{
  "input": { "description": "..." },
  "symptom_analysis": {
    "symptoms": ["headache", "fever", "nuchal rigidity"],
    "duration_hints": "2 days",
    "severity_hint": "severe",
    "body_systems": ["neurological"]
  },
  "differential_diagnosis": {
    "differential_diagnoses": [
      {
        "condition": "Bacterial Meningitis",
        "likelihood": "High",
        "reasoning": "Classic triad of headache, fever, and neck stiffness...",
        "icd10_hint": "G00.9",
        "red_flags": ["altered consciousness", "petechiae"]
      }
    ],
    "recommended_workup": ["LP", "CBC", "Blood cultures", "CT head"],
    "urgency": "Emergency",
    "urgency_rationale": "Suspected meningitis requires immediate evaluation",
    "general_advice": "...",
    "disclaimer": "..."
  },
  "literature": {
    "articles": [...],
    "summary": {
      "synthesis": "...",
      "key_findings": [...],
      "evidence_quality": "Strong",
      "clinical_implications": "...",
      "knowledge_gaps": "...",
      "article_count": 3
    }
  },
  "meta": {
    "articles_found": 3,
    "disclaimer": "..."
  }
}
```

### `POST /symptoms/extract`

Symptom extraction only — faster, cheaper.

### `GET /health`

Health check.

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `analyze_symptoms` | Full pipeline: symptom extraction → diagnosis → literature |
| `extract_symptoms_only` | Extract and normalize symptoms from free text |
| `search_pubmed` | Search PubMed by clinical query |

---

## Project Structure

```
clinisight/
├── app.py                    # FastAPI REST server
├── mcp_tool.py               # MCP tool server (Claude integration)
├── pyproject.toml
├── .env                      # API keys (never commit)
├── .env.example
└── functions/
    ├── __init__.py
    ├── symptom_extractor.py  # LLM-based symptom NLP
    ├── diagnosis_symptoms.py # GPT-4o differential diagnosis
    ├── pubmed_articles.py    # NCBI E-utilities integration
    └── summarize_pubmed.py   # Literature synthesis
```

---

## License

MIT
