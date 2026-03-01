import json
import re
from typing import Any, Dict, List, Optional

# Optional Gemini provider adapter. Only used when settings.NLP_PROVIDER == 'gemini'.
# Expects settings.GEMINI_API_KEY and optional settings.GEMINI_MODEL (default: 'gemini-1.5-flash').


def gemini_turn(
    user_text: str,
    history_text: str,
    profile_context: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    *,
    api_key: str,
    model_name: str = 'gemini-1.5-flash',
) -> str:
    """Full conversational turn handled entirely by Gemini.

    Unlike compose_answer() which only fills in a template after the FSM has
    already chosen a handler, this function is the *primary* reply path when
    Gemini is selected.  It receives the full context it needs to handle ANY
    natural-language input — greetings, eligibility questions, comparisons,
    follow-ups, pivots — without going through the FSM dispatcher at all.

    Args:
        user_text:        The latest user message.
        history_text:     Prior turns formatted as "User: ...\\nAssistant: ..."
        profile_context:  Dict with keys: grades, traits, career_goals, region,
                          education_level.  May be partially filled.
        recommendations:  Pre-fetched top-k programs from the catalog DB.
        api_key:          Gemini API key.
        model_name:       Model name (default: gemini-1.5-flash).

    Returns plain text reply.  Raises on API error so caller can fall back to
    the local FSM path.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())

    system = (
        "You are a friendly, expert career guidance counsellor for Kenyan high school students "
        "applying through KUCCPS (Kenya Universities and Colleges Central Placement Service).\n\n"
        "Your role:\n"
        "- Help students discover suitable university/college programs based on their KCSE grades, "
        "interests (RIASEC traits), career goals, and location preferences.\n"
        "- Answer any question naturally: greetings, eligibility checks, comparisons, 'why?' follow-ups, "
        "clarifications, pivots to new fields — handle it all conversationally.\n"
        "- Use the provided PROFILE and CATALOG RECOMMENDATIONS as your source of truth. "
        "Never invent program names, institution names, cutoff points, or grades.\n"
        "- If the catalog is empty or the user asks about a field not in RECOMMENDATIONS, "
        "acknowledge it helpfully and suggest they ask again with a more specific program name "
        "(e.g. 'Bachelor of Medicine (MBChB)', 'Bachelor of Nursing', 'Bachelor of Laws (LLB)').\n"
        "- Always remember what was said in the conversation history — no loops, no repetition.\n\n"
        "CRITICAL RULE — when to list program recommendations:\n"
        "- ONLY list specific programs or institutions when the user EXPLICITLY asks for them "
        "(e.g. 'suggest programs', 'what should I study', 'recommend courses', 'I want to do medicine', "
        "'show me options', 'what can I study with these grades').\n"
        "- For GREETINGS (hello, hi, hey) and GENERAL questions (how are you, what can you do), "
        "respond warmly and briefly — ask TWO SHORT questions to learn what they want: "
        "(1) what field/career interests them, and (2) whether they have KCSE grades to share.\n"
        "- Do NOT mention specific program names or universities in a greeting response.\n\n"
        "PROGRAM CODE RULE:\n"
        "- Whenever you mention a specific program that appears in the CATALOG RECOMMENDATIONS, "
        "always append its program code immediately after the program name using this exact format: "
        "[CODE: 1263131]. Example: 'Bachelor of Medicine & Bachelor of Surgery (MBChB) [CODE: 1263131]'.\n"
        "- Use the program code exactly as it appears in the catalog data provided — do not guess.\n"
        "- If a program has no code in the catalog, omit the [CODE: ...] tag entirely.\n\n"
        "Formatting rules:\n"
        "- Plain text only. No markdown, no asterisks, no bullet dash characters.\n"
        "- Use numbered lists (1. 2. 3.) for multiple items.\n"
        "- Keep replies concise: 2-4 sentences for greetings, 3-6 sentences for explanations, "
        "up to 8 items for lists.\n"
        "- End with a short, specific follow-up question or action suggestion when appropriate.\n"
    )

    # ── Build grounded context block ──────────────────────────────────────────
    grades = profile_context.get('grades') or {}
    traits = profile_context.get('traits') or {}
    career_goals = profile_context.get('career_goals') or []
    region = (profile_context.get('region') or '').strip()
    education_level = (profile_context.get('education_level') or 'high_school').strip()

    ctx_parts: List[str] = ['--- USER PROFILE ---']
    if grades:
        grade_str = ', '.join(f"{k}: {v}" for k, v in list(grades.items())[:15])
        ctx_parts.append(f"KCSE Grades: {grade_str}")
    else:
        ctx_parts.append("KCSE Grades: not provided yet")
    if career_goals:
        ctx_parts.append(f"Career goals: {', '.join(str(g) for g in career_goals[:5])}")
    if traits:
        top_traits = sorted(traits.items(), key=lambda kv: -float(kv[1] or 0))[:4]
        ctx_parts.append("Interest profile (RIASEC): " + ', '.join(f"{k} ({v:.2f})" for k, v in top_traits))
    if region:
        ctx_parts.append(f"Preferred region: {region}")
    ctx_parts.append(f"Education level: {education_level}")

    if recommendations:
        ctx_parts.append('\n--- CATALOG RECOMMENDATIONS ---')
        ctx_parts.append(f"(Top {len(recommendations)} programs matched from the KUCCPS catalog)")
        for i, r in enumerate(recommendations[:10], 1):
            nm = (r.get('program_name') or '').strip()
            score = r.get('score')
            pcode = (r.get('program_code') or '').strip()
            if nm and pcode:
                line = f"{i}. {nm} [CODE: {pcode}]"
            else:
                line = f"{i}. {nm}"
            
            institutions = r.get('institutions', [])
            if institutions:
                inst_list = []
                for inst_dict in institutions[:3]:
                    iname = (inst_dict.get('institution_name') or '').strip()
                    code = (inst_dict.get('program_code') or '').strip()
                    if iname and code:
                        inst_list.append(f"{iname} [{code}]")
                    elif iname:
                        inst_list.append(iname)
                if inst_list:
                    line += f" — Offered at: {', '.join(inst_list)}"
                    if len(institutions) > 3:
                        line += f" (and {len(institutions) - 3} more)"
            
            if score is not None:
                line += f" (match score: {float(score):.2f})"
            ctx_parts.append(line)
    else:
        ctx_parts.append('\n--- CATALOG RECOMMENDATIONS ---')
        ctx_parts.append('(No catalog matches yet — the user has not specified a field or grades)')

    context_block = '\n'.join(ctx_parts)

    # ── Build prompt ──────────────────────────────────────────────────────────
    history_section = ''
    if history_text and history_text.strip():
        history_section = f"Conversation history (oldest first):\n{history_text.strip()}\n\n"

    prompt = (
        f"{history_section}"
        f"{context_block}\n\n"
        f"--- LATEST USER MESSAGE ---\n{user_text.strip()}\n\n"
        "Respond naturally and helpfully to the user's message using the profile and catalog above."
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.3,
        ),
    )
    text = (resp.text or '').strip()
    return _sanitize_plain_text(text) or ''


def gemini_turn_stream(
    user_text: str,
    history_text: str,
    profile_context: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    *,
    api_key: str,
    model_name: str = 'gemini-1.5-flash',
):
    """Streaming version of gemini_turn. Yields text chunks as they arrive."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())

    system = (
        "You are a friendly, expert career guidance counsellor for Kenyan high school students "
        "applying through KUCCPS (Kenya Universities and Colleges Central Placement Service).\n\n"
        "Your role:\n"
        "- Help students discover suitable university/college programs based on their KCSE grades, "
        "interests (RIASEC traits), career goals, and location preferences.\n"
        "- Answer any question naturally: greetings, eligibility checks, comparisons, 'why?' follow-ups, "
        "clarifications, pivots to new fields — handle it all conversationally.\n"
        "- Use the provided PROFILE and CATALOG RECOMMENDATIONS as your source of truth. "
        "Never invent program names, institution names, cutoff points, or grades.\n"
        "- If the catalog is empty or the user asks about a field not in RECOMMENDATIONS, "
        "acknowledge it helpfully and suggest a more specific program name.\n"
        "- Always remember what was said in the conversation history — no loops, no repetition.\n\n"
        "Formatting rules:\n"
        "- Plain text only. No markdown, no asterisks, no bullet dash characters.\n"
        "- Use numbered lists (1. 2. 3.) for multiple items.\n"
        "- When suggesting a specific program, you MUST append its program code exactly in the format [CODE: <program_code>]. Example: 'Bachelor of Medicine & Bachelor of Surgery (MBChB) [CODE: 1263131]'.\n"
        "- Keep replies concise: 3-6 sentences for explanations, up to 8 items for lists.\n"
    )

    grades = profile_context.get('grades') or {}
    traits = profile_context.get('traits') or {}
    career_goals = profile_context.get('career_goals') or []
    region = (profile_context.get('region') or '').strip()
    education_level = (profile_context.get('education_level') or 'high_school').strip()

    ctx_parts: List[str] = ['--- USER PROFILE ---']
    if grades:
        grade_str = ', '.join(f"{k}: {v}" for k, v in list(grades.items())[:15])
        ctx_parts.append(f"KCSE Grades: {grade_str}")
    else:
        ctx_parts.append("KCSE Grades: not provided yet")
    if career_goals:
        ctx_parts.append(f"Career goals: {', '.join(str(g) for g in career_goals[:5])}")
    if traits:
        top_traits = sorted(traits.items(), key=lambda kv: -float(kv[1] or 0))[:4]
        ctx_parts.append("Interest profile (RIASEC): " + ', '.join(f"{k} ({v:.2f})" for k, v in top_traits))
    if region:
        ctx_parts.append(f"Preferred region: {region}")
    ctx_parts.append(f"Education level: {education_level}")

    if recommendations:
        ctx_parts.append('\n--- CATALOG RECOMMENDATIONS ---')
        for i, r in enumerate(recommendations[:10], 1):
            nm = (r.get('program_name') or '').strip()
            pcode = (r.get('program_code') or '').strip()
            if nm and pcode:
                line = f"{i}. {nm} [CODE: {pcode}]"
            else:
                line = f"{i}. {nm}"
            
            institutions = r.get('institutions', [])
            if institutions:
                inst_list = []
                for inst_dict in institutions[:3]:
                    iname = (inst_dict.get('institution_name') or '').strip()
                    code = (inst_dict.get('program_code') or '').strip()
                    if iname and code:
                        inst_list.append(f"{iname} [{code}]")
                    elif iname:
                        inst_list.append(iname)
                if inst_list:
                    line += f" — Offered at: {', '.join(inst_list)}"
                    if len(institutions) > 3:
                        line += f" (and {len(institutions) - 3} more)"
            
            ctx_parts.append(line)
    else:
        ctx_parts.append('\n--- CATALOG RECOMMENDATIONS ---')
        ctx_parts.append('(No catalog matches yet)')

    context_block = '\n'.join(ctx_parts)
    history_section = ''
    if history_text and history_text.strip():
        history_section = f"Conversation history (oldest first):\n{history_text.strip()}\n\n"

    prompt = (
        f"{history_section}"
        f"{context_block}\n\n"
        f"--- LATEST USER MESSAGE ---\n{user_text.strip()}\n\n"
        "Respond naturally and helpfully."
    )

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.3,
        ),
    ):
        text = getattr(chunk, 'text', None) or ''
        if text:
            yield text


def analyze_text(
    text: str,
    *,
    api_key: str,
    model_name: str = 'gemini-1.5-flash',
    history_text: str = '',
) -> Dict[str, Any]:
    """Call Gemini to analyze free text and return a structured payload compatible with nlp.analyze.

    Args:
        text: The latest user message.
        api_key: Gemini API key.
        model_name: Model name to use.
        history_text: Prior conversation turns (oldest first) to provide
            multi-turn context. Format: "User: ...\nAssistant: ..."

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
    # Build prompt: prepend history for multi-turn context
    history_section = ''
    if history_text and history_text.strip():
        history_section = f"Conversation history (for context only):\n{history_text.strip()}\n\n"
    prompt = f"{history_section}Latest user message: {text.strip()}\nReturn JSON now."

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
        "- If the context includes 'tool_results' (e.g., eligibility criteria, explanations, or program details), you MUST use this information directly to answer the user's specific question.\n"
        "- If asked 'which programs', list specific program titles from 'program_titles' or from 'program_recommendations'.\n"
        "- If asked 'which universities', list items from 'institutions' in context.\n"
        "- Keep answers concise and helpful.\n"
        "- If insufficient data, ask a pointed follow-up (1 short line).\n"
        "Formatting:\n"
        "- Output plain text only (no markdown, no asterisks, no bullet characters).\n"
        "- If listing items, use simple numbered lines like '1. ...', '2. ...'.\n"
        "- When suggesting a specific program, you MUST append its program_code exactly in the format [CODE: <program_code>]. Example: 'Bachelor of Medicine & Bachelor of Surgery [CODE: 1263131]'.\n"
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


# ---------------------------------------------------------------------------
# Streaming variants — for SSE endpoints
# ---------------------------------------------------------------------------

def compose_answer_stream(
    user_text: str,
    context: Dict[str, Any],
    *,
    api_key: str,
    model_name: str = 'gemini-1.5-flash',
    history_text: str = '',
):
    """Streaming version of compose_answer. Yields raw text chunks as they arrive.

    Usage::

        for chunk in compose_answer_stream(text, ctx, api_key=key):
            yield chunk  # send as SSE 'delta' event
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())

    system = (
        "You are a career guidance assistant. You MUST ground responses ONLY on the provided context.\n"
        "Rules:\n"
        "- Do NOT invent universities, programs, or data not present in context.\n"
        "- If the context includes 'tool_results' (e.g., eligibility criteria, explanations, or program details), you MUST use this information directly to answer the user's specific question.\n"
        "- If asked 'which programs', list specific program titles from 'program_titles' or from 'program_recommendations'.\n"
        "- If asked 'which universities', list items from 'institutions' in context.\n"
        "- Keep answers concise and helpful.\n"
        "- If insufficient data, ask a pointed follow-up (1 short line).\n"
        "Formatting:\n"
        "- Output plain text only (no markdown, no asterisks, no bullet characters).\n"
        "- If listing items, use simple numbered lines like '1. ...', '2. ...'.\n"
        "- When suggesting a specific program, you MUST append its program_code exactly in the format [CODE: <program_code>]. Example: 'Bachelor of Medicine & Bachelor of Surgery [CODE: 1263131]'.\n"
        "- Avoid long lists: default to 5 items max unless the user asks for more."
    )

    history_section = ''
    if history_text and history_text.strip():
        history_section = f"Conversation history (for context):\n{history_text.strip()}\n\n"

    ctx_json = json.dumps(context, ensure_ascii=False)
    prompt = (
        f"{history_section}"
        f"User: {user_text.strip()}\n"
        f"Context JSON (authoritative): {ctx_json}\n"
        "Compose the best possible grounded answer now."
    )

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
        ),
    ):
        text = getattr(chunk, 'text', None) or ''
        if text:
            yield text


def compose_rag_answer_stream(
    user_text: str,
    sources: List[Dict[str, Any]],
    *,
    api_key: str,
    model_name: str = 'gemini-1.5-flash',
    history_text: str = '',
):
    """Streaming version of compose_rag_answer. Yields raw text chunks as they arrive."""
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

    history_section = ''
    if history_text and history_text.strip():
        history_section = f"Conversation history (for context):\n{history_text.strip()}\n\n"

    src_json = json.dumps(sources or [], ensure_ascii=False)
    prompt = (
        f"{history_section}"
        f"User: {user_text.strip()}\n"
        f"SOURCES JSON (authoritative): {src_json}\n"
        "Answer the user, grounded strictly in SOURCES, with citations."
    )

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
        ),
    ):
        text = getattr(chunk, 'text', None) or ''
        if text:
            yield text

