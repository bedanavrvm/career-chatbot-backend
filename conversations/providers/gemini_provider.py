import json
import re
from typing import Any, Dict, List

# Optional Gemini provider adapter. Only used when settings.NLP_PROVIDER == 'gemini'.
# Expects settings.GEMINI_API_KEY and optional settings.GEMINI_MODEL (default: 'gemini-1.5-flash').

def analyze_text(text: str, *, api_key: str, model_name: str = 'gemini-1.5-flash') -> Dict[str, Any]:
    """Call Gemini to analyze free text and return a structured payload compatible with nlp.analyze.

    Returns a dict with keys: grades (dict), traits (dict), intents (list), confidence (float).
    On any error, raises an exception to allow caller to fallback to local NLP.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())

    system = (
        "You extract structured information for a high school career guidance assistant.\n"
        "Return ONLY valid JSON with keys: \n"
        "- 'grades': map of subject codes (MAT/ENG/KIS/...) to grades (A-,A,B+,...).\n"
        "- 'traits': map of RIASEC-like traits to scores 0..1.\n"
        "- 'intents': array of strings. Allowed intents include: "
        "'greeting', 'help', 'ask_grades', 'provide_grades', 'interests', "
        "'recommend' (career guidance / program recommendations), "
        "'career_paths' (user asks for possible career paths), "
        "'qualify' (user asks if they qualify/are eligible), "
        "'explain' (user asks why a recommendation was made), "
        "'catalog_lookup' (universities offering a program / requirements / cutoff / fees), "
        "'institutions_by_region' (universities/colleges in a region/county), "
        "'programs_near_me' (programs/courses offered near the user's home location).\n"
        "- 'confidence': number 0..1.\n"
        "- 'lookup' (optional): when the user asks which universities/colleges offer a program, include { 'program_query': string, 'level': string (bachelor/diploma/certificate/masters) }.\n"
        "- 'career_paths' (optional): array of 3-8 concise suggested career paths aligned to the user's interests/goals, e.g., ['Aerospace Engineer','Data Scientist'].\n"
        "Subject codes: MAT, ENG, KIS, CHE, PHY, BIO, HIS, GEO, BUS, COM, CRE, IRE, AGR.\n"
        "Examples: 'Math A-' => grades.MAT='A-'; 'English B+' => grades.ENG='B+'.\n"
        "If information is absent, use empty values. Confidence reflects how certain you are.\n"
        "Output JSON only, no markdown, no extra text."
    )
    prompt = f"User: {text.strip()}\nReturn JSON now."

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
        ),
    )
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


def _sanitize_plain_text(text: str) -> str:
    t = (text or '').replace('\r\n', '\n').strip()
    if not t:
        return ''

    if t.startswith('```'):
        t = re.sub(r"^```[a-zA-Z0-9_\-]*\s*", "", t)
        t = t.replace('```', '')

    t = t.replace('**', '').replace('__', '').replace('`', '')

    lines = t.split('\n')
    out_lines: List[str] = []
    n = 0
    for line in lines:
        m = re.match(r"^\s*([*\-•])\s+(.*)$", line)
        if m:
            n += 1
            out_lines.append(f"{n}. {m.group(2).strip()}")
            continue
        n = 0
        out_lines.append(line.rstrip())

    t2 = "\n".join(out_lines)
    t2 = re.sub(r"\n{3,}", "\n\n", t2)
    return t2.strip()


def compose_answer(user_text: str, context: Dict[str, Any], *, api_key: str, model_name: str = 'gemini-1.5-flash') -> str:
    """Ask Gemini to compose a grounded answer using provided structured context.

    Context may include: grades, traits, career_paths, program_recommendations (list),
    program_titles (list), institutions (list), and provider metadata. The model is
    instructed to strictly rely on the provided context and avoid fabrications.
    Returns plain text. Raises on API error so caller can fallback locally.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())

    system = (
        "You are a career guidance assistant. You MUST ground responses ONLY on the provided context.\n"
        "Rules:\n"
        "- Do NOT invent universities, programs, or data not present in context.\n"
        "- If asked 'which programs', list specific program titles from 'program_titles' or from 'program_recommendations'.\n"
        "- If asked 'which universities', list items from 'institutions' in context.\n"
        "- Keep answers concise and helpful.\n"
        "- If insufficient data, ask a pointed follow-up (1 short line).\n"
        "Formatting:\n"
        "- Output plain text only (no markdown, no asterisks, no bullet characters).\n"
        "- If listing items, use simple numbered lines like '1. ...', '2. ...'.\n"
        "- Avoid long lists: default to 5 items max unless the user asks for more."
    )
    ctx_json = json.dumps(context, ensure_ascii=False)
    prompt = (
        f"User: {user_text.strip()}\n"
        f"Context JSON (authoritative): {ctx_json}\n"
        "Compose the best possible grounded answer now."
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
        ),
    )
    text = (resp.text or '').strip()
    return _sanitize_plain_text(text) or ""


def compose_rag_answer(user_text: str, sources: List[Dict[str, Any]], *, api_key: str, model_name: str = 'gemini-1.5-flash') -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())

    system = (
        "You are a career guidance assistant. You MUST answer ONLY using the provided SOURCES.\n"
        "Rules:\n"
        "- Do NOT invent programs, universities, costs, cutoffs, or requirements not present in SOURCES.\n"
        "- Every factual claim MUST have at least one citation like [P1] or [P2].\n"
        "- If SOURCES do not contain the answer, say you don't have enough information and ask 1 short follow-up question.\n"
        "Formatting:\n"
        "- Output plain text only (no markdown, no asterisks, no bullet characters).\n"
        "- If listing items, use simple numbered lines like '1. ...', '2. ...'.\n"
        "- Keep citations inline at the end of the relevant sentence, e.g., '... [P1]'.\n"
        "- Avoid long lists: default to 5 items max unless the user asks for more."
    )

    src_json = json.dumps(sources or [], ensure_ascii=False)
    prompt = (
        f"User: {user_text.strip()}\n"
        f"SOURCES JSON (authoritative): {src_json}\n"
        "Answer the user, grounded strictly in SOURCES, with citations."
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
        ),
    )
    text = (resp.text or '').strip()
    return _sanitize_plain_text(text) or ""
