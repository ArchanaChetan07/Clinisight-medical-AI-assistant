"""
symptom_extractor.py
Uses an LLM to extract structured symptoms from free-text patient descriptions,
replacing the brittle 5-keyword regex with a robust NLP approach.
"""

import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a clinical NLP specialist. Extract all medical symptoms and signs from the patient description provided.

Return a JSON object with this exact schema:
{
  "symptoms": ["symptom1", "symptom2", ...],
  "duration_hints": "e.g. 3 days, chronic, sudden onset — or null",
  "severity_hint": "mild | moderate | severe | null",
  "body_systems": ["respiratory", "neurological", ...] 
}

Rules:
- Normalize symptoms to standard clinical terminology (e.g. "throwing up" → "vomiting")
- Include both explicit and implied symptoms
- symptoms list must contain at least 1 item; return [] only if truly no symptoms found
- Return valid JSON only, no markdown fences
"""


def extract_symptoms(text: str) -> dict:
    """
    Extract symptoms from free-text patient description.
    Returns a dict with symptoms list and metadata.
    """
    if not text or not text.strip():
        return {"symptoms": [], "duration_hints": None, "severity_hint": None, "body_systems": []}

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Patient description: {text.strip()}"}
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error from symptom extractor: {e}. Raw: {raw}")
        # Fallback: return raw text wrapped
        return {"symptoms": [text.strip()], "duration_hints": None, "severity_hint": None, "body_systems": []}

    except Exception as e:
        logger.error(f"Symptom extraction failed: {e}")
        raise RuntimeError(f"Symptom extraction error: {e}") from e
