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
        "Return ONLY valid JSON with keys: \n"
        "- 'grades': map of subject codes (MAT/ENG/KIS/...) to grades (A-,A,B+,...).\n"
        "- 'traits': map of RIASEC-like traits to scores 0..1.\n"
        "- 'intents': array of strings.\n"
        "- 'confidence': number 0..1.\n"
        "- 'lookup' (optional): when the user asks which universities/colleges offer a program, include { 'program_query': string, 'level': string (bachelor/diploma/certificate/masters) }.\n"
        "- 'career_paths' (optional): array of 3-8 concise suggested career paths aligned to the user's interests/goals, e.g., ['Aerospace Engineer','Data Scientist'].\n"
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
    lookup = data.get('lookup') or {}
    career_paths = data.get('career_paths') or []
    try:
        conf = float(data.get('confidence') or 0.0)
    except Exception:
        conf = 0.0

    # Basic normalization
    grades = {str(k).upper(): str(v).upper().replace(' ', '') for k, v in grades.items()}
    traits = {str(k): float(v) for k, v in traits.items() if _is_number(v)}
    intents = [str(x) for x in intents]
    # Normalize lookup
    if isinstance(lookup, dict):
        pq = str(lookup.get('program_query') or '').strip()
        lvl = str(lookup.get('level') or '').strip()
        lookup = {'program_query': pq, 'level': lvl} if pq else {}
    else:
        lookup = {}
    # Normalize career paths
    if isinstance(career_paths, list):
        career_paths = [str(x).strip() for x in career_paths if str(x).strip()]
    else:
        career_paths = []
    conf = max(0.0, min(1.0, conf))

    return {
        'grades': grades,
        'traits': traits,
        'intents': intents,
        'confidence': conf,
        'lookup': lookup,
        'career_paths': career_paths,
    }


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def compose_answer(user_text: str, context: Dict[str, Any], *, api_key: str, model_name: str = 'gemini-1.5-flash') -> str:
    """Ask Gemini to compose a grounded answer using provided structured context.

    Context may include: grades, traits, career_paths, program_recommendations (list),
    program_titles (list), institutions (list), and provider metadata. The model is
    instructed to strictly rely on the provided context and avoid fabrications.
    Returns plain text. Raises on API error so caller can fallback locally.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel(model_name)

    system = (
        "You are a career guidance assistant. You MUST ground responses ONLY on the provided context.\n"
        "Rules:\n"
        "- Do NOT invent universities, programs, or data not present in context.\n"
        "- If asked 'which programs', list specific program titles from 'program_titles' or from 'program_recommendations'.\n"
        "- If asked 'which universities', list items from 'institutions' in context.\n"
        "- Keep answers concise and helpful.\n"
        "- If insufficient data, ask a pointed follow-up (1 short line).\n"
        "Output plain text only."
    )
    ctx_json = json.dumps(context, ensure_ascii=False)
    prompt = (
        f"User: {user_text.strip()}\n"
        f"Context JSON (authoritative): {ctx_json}\n"
        "Compose the best possible grounded answer now."
    )

    resp = model.generate_content([system, prompt])
    text = (resp.text or '').strip()
    return text or ""
