"""
app.py — Clinisight API
A clinical decision-support backend exposing symptom analysis, differential diagnosis,
and evidence-based literature synthesis via a RESTful API.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from functions.symptom_extractor import extract_symptoms
from functions.diagnosis_symptoms import get_diagnosis
from functions.pubmed_articles import fetch_pubmed_articles_with_metadata
from functions.summarize_pubmed import summarize_articles

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("clinisight")


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Clinisight API starting up")
    yield
    logger.info("Clinisight API shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinisight API",
    description=(
        "Clinical decision-support API providing symptom extraction, "
        "differential diagnosis, and PubMed literature synthesis. "
        "**For informational and educational use only. Not a substitute for professional medical advice.**"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed}ms)")
    return response


# ── Schemas ───────────────────────────────────────────────────────────────────
class SymptomInput(BaseModel):
    description: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        example="I've had a severe headache for 2 days, along with fever and neck stiffness.",
        description="Free-text patient symptom description",
    )
    max_articles: int = Field(default=3, ge=1, le=10, description="Max PubMed articles to retrieve")


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["System"])
def root():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/diagnosis", tags=["Clinical"])
async def diagnosis(data: SymptomInput):
    """
    Full clinical analysis pipeline:
    1. Extract structured symptoms from free-text description
    2. Generate differential diagnoses with urgency assessment
    3. Search PubMed for relevant literature
    4. Synthesize evidence into a clinical summary

    Returns a comprehensive structured response. **Educational use only.**
    """
    try:
        # Step 1: Extract symptoms
        logger.info("Extracting symptoms from patient description")
        symptom_data = extract_symptoms(data.description)

        if not symptom_data.get("symptoms"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No recognizable symptoms found in the description. Please provide more detail.",
            )

        # Step 2: Differential diagnosis
        logger.info(f"Generating diagnosis for symptoms: {symptom_data['symptoms']}")
        diagnosis_result = get_diagnosis(symptom_data)

        # Step 3: PubMed search using symptom terms
        query = " ".join(symptom_data["symptoms"][:5])
        logger.info(f"Fetching PubMed articles for: {query!r}")
        articles = fetch_pubmed_articles_with_metadata(query, max_results=data.max_articles)

        # Step 4: Synthesize literature
        logger.info(f"Synthesizing {len(articles)} articles")
        literature_summary = summarize_articles(articles)

        return {
            "input": {
                "description": data.description,
            },
            "symptom_analysis": symptom_data,
            "differential_diagnosis": diagnosis_result,
            "literature": {
                "articles": articles,
                "summary": literature_summary,
            },
            "meta": {
                "articles_found": len(articles),
                "disclaimer": (
                    "This output is generated by AI for informational and educational purposes only. "
                    "It does not constitute medical advice, diagnosis, or treatment. "
                    "Always consult a qualified healthcare professional."
                ),
            },
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in /diagnosis: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")


@app.post("/symptoms/extract", tags=["Clinical"])
async def extract_only(data: SymptomInput):
    """Extract structured symptoms from a free-text description without running the full pipeline."""
    try:
        return extract_symptoms(data.description)
    except Exception as e:
        logger.error(f"Symptom extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True, log_level="info")
