"""
summarize_pubmed.py
Summarizes a list of PubMed articles into a clinically-oriented synthesis
using GPT-4o. Fixes the original bug of slicing a list instead of text.
"""

import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a clinical research synthesizer. Given a set of PubMed article titles, abstracts, and metadata, produce a structured clinical summary.

Return a JSON object with this schema:
{
  "synthesis": "2-4 sentence narrative synthesis of what the literature collectively shows",
  "key_findings": ["Finding 1", "Finding 2", ...],
  "evidence_quality": "Strong | Moderate | Weak | Conflicting",
  "clinical_implications": "What this means for patient management",
  "knowledge_gaps": "What remains unclear or under-studied",
  "article_count": 3
}

Return valid JSON only, no markdown fences.
"""


def summarize_articles(articles: list[dict]) -> dict:
    """
    Synthesize a list of PubMed article dicts into a clinical summary.

    Args:
        articles: List of article dicts from fetch_pubmed_articles_with_metadata()

    Returns:
        Dict with synthesis, key_findings, evidence_quality, etc.
    """
    if not articles:
        return {
            "synthesis": "No relevant literature found for the given symptoms.",
            "key_findings": [],
            "evidence_quality": "N/A",
            "clinical_implications": "Literature search returned no results.",
            "knowledge_gaps": "N/A",
            "article_count": 0,
        }

    # Build a compact but rich text representation of the articles
    article_texts = []
    for i, art in enumerate(articles, 1):
        chunk = (
            f"[{i}] {art.get('title', 'No title')}\n"
            f"    Journal: {art.get('journal', 'Unknown')} ({art.get('publication_date', '')})\n"
            f"    Authors: {', '.join(art.get('authors', [])[:3])}\n"
            f"    Abstract: {art.get('abstract', '')[:600]}\n"
        )
        if art.get("mesh_terms"):
            chunk += f"    MeSH terms: {', '.join(art['mesh_terms'][:5])}\n"
        article_texts.append(chunk)

    combined = "\n\n".join(article_texts)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Synthesize these {len(articles)} PubMed articles:\n\n{combined}"}
            ],
            temperature=0.2,
            max_tokens=700,
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        result["article_count"] = len(articles)
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error in summarizer: {e}")
        return {
            "synthesis": raw if 'raw' in dir() else "Summary unavailable.",
            "key_findings": [],
            "evidence_quality": "Unknown",
            "clinical_implications": "Could not parse structured summary.",
            "knowledge_gaps": "",
            "article_count": len(articles),
        }

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise RuntimeError(f"Summarization error: {e}") from e


# Legacy compatibility alias
def summarize_text(text: str) -> str:
    """Legacy wrapper — accepts raw text, returns plain string summary."""
    result = summarize_articles([{"title": "Input text", "abstract": text, "authors": [], "journal": "", "publication_date": "", "mesh_terms": []}])
    return result.get("synthesis", "")
