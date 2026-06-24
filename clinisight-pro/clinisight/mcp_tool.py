"""
mcp_tool.py — Clinisight MCP Server
Exposes Clinisight's clinical decision-support pipeline as an MCP tool
for use with Claude and other MCP-compatible clients.

Usage:
    mcp install mcp_tool.py
    uv run mcp dev mcp_tool.py
"""

import logging
from mcp.server.fastmcp import FastMCP

from functions.symptom_extractor import extract_symptoms
from functions.diagnosis_symptoms import get_diagnosis
from functions.pubmed_articles import fetch_pubmed_articles_with_metadata
from functions.summarize_pubmed import summarize_articles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinisight-mcp")

mcp = FastMCP(
    "Clinisight AI",
    instructions=(
        "Clinisight is a clinical decision-support tool. "
        "It extracts symptoms from patient descriptions, generates differential diagnoses, "
        "and synthesizes relevant PubMed literature. "
        "All output is for informational and educational purposes only, "
        "not as a substitute for professional medical advice."
    ),
)


@mcp.tool()
async def analyze_symptoms(
    symptom_text: str,
    max_articles: int = 3,
) -> dict:
    """
    Run the full Clinisight clinical analysis pipeline on a patient symptom description.

    Args:
        symptom_text: Free-text description of patient symptoms (e.g. "severe headache, fever, and neck stiffness for 2 days")
        max_articles: Number of PubMed articles to retrieve (1–10, default 3)

    Returns:
        Structured dict with symptom_analysis, differential_diagnosis, and literature summary.
    """
    logger.info(f"MCP: analyze_symptoms called with: {symptom_text[:80]}...")

    symptom_data = extract_symptoms(symptom_text)
    if not symptom_data.get("symptoms"):
        return {"error": "No recognizable symptoms found. Please provide a more detailed description."}

    diagnosis_result = get_diagnosis(symptom_data)

    query = " ".join(symptom_data["symptoms"][:5])
    articles = fetch_pubmed_articles_with_metadata(query, max_results=max_articles)
    literature_summary = summarize_articles(articles)

    return {
        "symptom_analysis": symptom_data,
        "differential_diagnosis": diagnosis_result,
        "literature": {
            "articles": articles,
            "summary": literature_summary,
        },
        "disclaimer": (
            "This output is for informational and educational purposes only. "
            "It does not constitute medical advice. "
            "Always consult a qualified healthcare professional."
        ),
    }


@mcp.tool()
async def extract_symptoms_only(symptom_text: str) -> dict:
    """
    Extract and normalize symptoms from a free-text patient description.

    Args:
        symptom_text: Patient's description of their symptoms

    Returns:
        Dict with normalized symptoms list, severity hint, duration hints, and affected body systems.
    """
    return extract_symptoms(symptom_text)


@mcp.tool()
async def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """
    Search PubMed for articles relevant to a clinical query.

    Args:
        query: Clinical search terms (e.g. "tension headache management")
        max_results: Number of articles to return (1–10)

    Returns:
        List of article dicts with title, abstract, authors, journal, date, and URL.
    """
    return fetch_pubmed_articles_with_metadata(query, max_results=max_results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
