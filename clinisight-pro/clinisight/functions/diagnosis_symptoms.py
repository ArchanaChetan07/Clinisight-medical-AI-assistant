"""
diagnosis_symptoms.py
Generates a structured differential diagnosis with safety disclaimers,
urgency flags, and actionable next steps using GPT-4o.
"""

import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a clinical decision-support AI assisting licensed healthcare professionals.
You provide differential diagnoses for educational and informational purposes only — NOT as a substitute for professional medical advice.

Given a list of symptoms and clinical context, return a JSON object with this schema:
{
  "differential_diagnoses": [
    {
      "condition": "Condition Name",
      "likelihood": "High | Moderate | Low",
      "reasoning": "Brief clinical reasoning (2-3 sentences)",
      "icd10_hint": "e.g. J06.9",
      "red_flags": ["symptom that would escalate urgency", ...] 
    }
  ],
  "recommended_workup": ["CBC", "CXR", ...],
  "urgency": "Routine | Urgent | Emergency",
  "urgency_rationale": "Why this urgency level was assigned",
  "general_advice": "Concise, evidence-based management guidance",
  "disclaimer": "This AI output is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional."
}

Return valid JSON only, no markdown fences. Limit differential to top 3-5 conditions.
"""


def get_diagnosis(symptom_data: dict) -> dict:
    """
    Generate structured differential diagnosis from extracted symptom data.
    
    Args:
        symptom_data: dict from extract_symptoms() with keys: symptoms, duration_hints, severity_hint, body_systems
    
    Returns:
        dict with differential diagnoses, workup, urgency, and advice
    """
    symptoms = symptom_data.get("symptoms", [])
    if not symptoms:
        return {
            "differential_diagnoses": [],
            "recommended_workup": [],
            "urgency": "Routine",
            "urgency_rationale": "No symptoms provided",
            "general_advice": "Please provide a symptom description.",
            "disclaimer": "This AI output is for informational purposes only."
        }

    user_prompt = f"""
Symptoms: {', '.join(symptoms)}
Duration: {symptom_data.get('duration_hints') or 'Not specified'}
Severity: {symptom_data.get('severity_hint') or 'Not specified'}
Body systems involved: {', '.join(symptom_data.get('body_systems', [])) or 'Not specified'}

Generate a differential diagnosis.
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error in diagnosis: {e}")
        return {"error": "Could not parse diagnosis output", "raw": raw}

    except Exception as e:
        logger.error(f"Diagnosis generation failed: {e}")
        raise RuntimeError(f"Diagnosis error: {e}") from e
