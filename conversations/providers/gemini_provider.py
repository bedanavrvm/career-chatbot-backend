import json
from typing import Any, Dict

# Optional Gemini provider adapter. Only used when settings.NLP_PROVIDER == 'gemini'.
# Expects settings.GEMINI_API_KEY and optional settings.GEMINI_MODEL (default: 'gemini-1.5-flash').

def analyze_text(text: str, *, api_key: str, model_name: str = 'gemini-1.5-flash') -> Dict[str, Any]:
    """Call Gemini to analyze free text and return a structured payload compatible with nlp.analyze.

    Returns a dict with keys: grades (dict), traits (dict), intents (list), confidence (float).
    On any error, raises an exception to allow caller to fallback to local NLP.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel(model_name)

    system = (
        "You extract structured information for a high school career guidance assistant.\n"
        "Given a user's message, return ONLY valid JSON with keys: 'grades' (map of subject codes like MAT/ENG/KIS to grades A-,A,B+,... ),\n"
        "'traits' (map of RIASEC-like traits to scores 0..1), 'intents' (array of strings), 'confidence' (0..1).\n"
        "Subject codes: MAT, ENG, KIS, CHE, PHY, BIO, HIS, GEO, BUS, COM, CRE, IRE, AGR.\n"
        "Examples: 'Math A-' => grades.MAT='A-'; 'English B+' => grades.ENG='B+'.\n"
        "If information is absent, use empty values. Confidence reflects how certain you are.\n"
        "Output JSON only, no markdown, no extra text."
    )
    prompt = f"User: {text.strip()}\nReturn JSON now."

    resp = model.generate_content([system, prompt])
    raw = resp.text or "{}"
    # Some models wrap JSON in code fences; strip them
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip('`')
        # Remove language tag if present
        if raw.startswith('json'):
            raw = raw[4:]
    raw = raw.strip()
    data = json.loads(raw)

    grades = data.get('grades') or {}
    traits = data.get('traits') or {}
    intents = data.get('intents') or []
    try:
        conf = float(data.get('confidence') or 0.0)
    except Exception:
        conf = 0.0

    # Basic normalization
    grades = {str(k).upper(): str(v).upper().replace(' ', '') for k, v in grades.items()}
    traits = {str(k): float(v) for k, v in traits.items() if _is_number(v)}
    intents = [str(x) for x in intents]
    conf = max(0.0, min(1.0, conf))

    return {
        'grades': grades,
        'traits': traits,
        'intents': intents,
        'confidence': conf,
    }


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False
