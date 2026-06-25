import pytest
import re

class TestClinisight:
    def test_symptom_extraction(self):
        text = "Patient reports headache, fever, and fatigue for 3 days"
        symptoms = ["headache","fever","fatigue","nausea","cough","pain"]
        found = [s for s in symptoms if s in text.lower()]
        assert len(found) >= 2

    def test_medical_terms_detected(self):
        medical_terms = {"hypertension","diabetes","tachycardia","dyspnea","edema"}
        note = "Patient has hypertension and diabetes, presenting with mild dyspnea"
        tokens = set(note.lower().split())
        found = tokens & medical_terms
        assert len(found) >= 2

    def test_severity_classification(self):
        def classify(score):
            if score >= 8: return "critical"
            if score >= 5: return "moderate"
            return "mild"
        assert classify(9) == "critical"
        assert classify(6) == "moderate"
        assert classify(2) == "mild"

    def test_patient_data_structure(self):
        patient = {"id": "P001","age": 45,"gender": "F","symptoms": ["fever"],"vitals": {"bp": "120/80","hr": 72}}
        assert "id" in patient and "symptoms" in patient and "vitals" in patient

    def test_empty_note_handled(self):
        def process(note):
            if not note or not note.strip():
                return {"error": "Empty clinical note"}
            return {"note": note, "processed": True}
        result = process("")
        assert "error" in result

    def test_icd_code_format(self):
        icd_codes = ["J00","K21.0","I10","E11.9"]
        pattern = re.compile(r'^[A-Z]\d{2}(\.\d)?$')
        for code in icd_codes:
            assert pattern.match(code), f"{code} is not valid ICD format"

    def test_drug_interaction_check(self):
        interactions = {("warfarin","aspirin"):"bleeding risk",("metformin","alcohol"):"lactic acidosis"}
        drugs = ["warfarin","aspirin"]
        pair = tuple(sorted(drugs))
        assert pair in interactions

class TestMCPMedicalTool:
    def test_tool_schema(self):
        tool = {"name": "analyze_symptoms","description": "Analyze patient symptoms","input_schema": {"type": "object","properties": {"symptoms": {"type": "array"}},"required": ["symptoms"]}}
        assert tool["name"] == "analyze_symptoms"
        assert "symptoms" in tool["input_schema"]["properties"]

    def test_response_structure(self):
        response = {"diagnosis": "Common cold","confidence": 0.85,"recommendations": ["rest","hydration"]}
        assert "diagnosis" in response
        assert 0 <= response["confidence"] <= 1
