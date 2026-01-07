from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from django.conf import settings

from . import nlp


_ALLOWED_INTENTS = {
    'recommend_programs',
    'search_programs',
    'program_details',
    'check_eligibility',
    'filter_results',
    'explain_recommendation',
    'reset_context',
}

_ALLOWED_ACTIONS = {
    'get_user_context',
    'recommend_programs',
    'search_programs',
    'program_details',
    'check_eligibility',
    'explain_recommendation',
}

_ALLOWED_ENTITY_KEYS = {
    'goal',
    'program_query',
    'program_id',
    'level',
    'region',
    'institution',
    'budget',
    'mode',
    'sort_by',
}


PLANNER_JSON_SCHEMA: Dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['intent', 'entities', 'constraints', 'clarifying_question', 'proposed_actions'],
    'properties': {
        'intent': {'type': 'string', 'enum': sorted(list(_ALLOWED_INTENTS))},
        'entities': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {k: {} for k in sorted(list(_ALLOWED_ENTITY_KEYS))},
        },
        'constraints': {
            'type': 'object',
            'additionalProperties': True,
            'properties': {
                'k': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                'must_check_eligibility': {'type': 'boolean'},
                'tool_budget': {'type': 'integer', 'minimum': 0, 'maximum': 3},
                'sort_by': {'type': 'string'},
                'diversify': {'type': 'boolean'},
            },
        },
        'clarifying_question': {'type': 'string'},
        'proposed_actions': {'type': 'array', 'items': {'type': 'string', 'enum': sorted(list(_ALLOWED_ACTIONS))}},
    },
}


@dataclass(frozen=True)
class Plan:
    intent: str
    entities: Dict[str, Any]
    constraints: Dict[str, Any]
    clarifying_question: str
    proposed_actions: List[str]


def validate_plan_dict(raw: Any) -> Plan:
    if not isinstance(raw, dict):
        raw = {}

    intent = str(raw.get('intent') or '').strip() or 'recommend_programs'
    if intent not in _ALLOWED_INTENTS:
        intent = 'recommend_programs'

    entities_in = raw.get('entities')
    entities: Dict[str, Any] = {}
    if isinstance(entities_in, dict):
        for k, v in entities_in.items():
            kk = str(k or '').strip()
            if kk in _ALLOWED_ENTITY_KEYS:
                entities[kk] = v

    constraints_in = raw.get('constraints')
    constraints: Dict[str, Any] = {}
    if isinstance(constraints_in, dict):
        constraints = dict(constraints_in)

    # Normalize constraints
    k_raw = constraints.get('k', None)
    try:
        k_val = int(k_raw) if k_raw is not None else 10
    except Exception:
        k_val = 10
    constraints['k'] = max(1, min(20, k_val))

    mce = constraints.get('must_check_eligibility', None)
    constraints['must_check_eligibility'] = bool(mce) if mce is not None else False

    tool_budget_raw = constraints.get('tool_budget', None)
    try:
        tool_budget = int(tool_budget_raw) if tool_budget_raw is not None else 2
    except Exception:
        tool_budget = 2
    constraints['tool_budget'] = max(0, min(3, tool_budget))

    clarifying_question = str(raw.get('clarifying_question') or '').strip()

    proposed_actions_in = raw.get('proposed_actions')
    proposed_actions: List[str] = []
    if isinstance(proposed_actions_in, list):
        for a in proposed_actions_in:
            aa = str(a or '').strip()
            if aa in _ALLOWED_ACTIONS and aa not in proposed_actions:
                proposed_actions.append(aa)

    return Plan(
        intent=intent,
        entities=entities,
        constraints=constraints,
        clarifying_question=clarifying_question,
        proposed_actions=proposed_actions,
    )


def _strip_json_fences(raw: str) -> str:
    s = (raw or '').strip()
    if s.startswith('```'):
        s = s.strip('`')
        if s.startswith('json'):
            s = s[4:]
    return s.strip()


def build_planner_prompt(*, user_text: str, context: Dict[str, Any]) -> str:
    ctx_json = json.dumps(context or {}, ensure_ascii=False)
    tool_desc = (
        'Tools: get_user_context, recommend_programs, search_programs, program_details, check_eligibility, explain_recommendation.'
    )
    return (
        f"User: {str(user_text or '').strip()}\n"
        f"Context JSON: {ctx_json}\n"
        f"{tool_desc}\n"
        'Return ONLY valid JSON that matches the schema.'
    )


def plan_message(
    user_text: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    provider_override: str = '',
) -> Plan:
    """Return a structured Plan for the user message.

    Phase 1: planner output is validated but not yet wired into the main chat handler.
    """
    ctx = context or {}
    provider = (provider_override or '').strip().lower()

    if provider == 'gemini':
        api_key = (getattr(settings, 'GEMINI_API_KEY', '') or '').strip()
        if api_key:
            try:
                return _plan_with_gemini(user_text, ctx)
            except Exception:
                pass

    return _plan_local(user_text, ctx, provider_override=provider_override)


def _plan_local(user_text: str, ctx: Dict[str, Any], *, provider_override: str = '') -> Plan:
    analysis = nlp.analyze(user_text, provider_override=provider_override)
    intents = analysis.get('intents') or []
    lookup = analysis.get('lookup') or {}

    low = str(user_text or '').strip().lower()

    entities: Dict[str, Any] = {}
    constraints: Dict[str, Any] = {'k': 10, 'tool_budget': 2}

    reg = analysis.get('institutions_region')
    if ('filter by' in low) or low.startswith('filter ') or ('only in' in low):
        intent = 'filter_results'
        if reg:
            entities['region'] = str(reg)
        else:
            try:
                import re

                m = re.search(r"\bfilter\s+by\s+([a-z][a-z\s\-]{1,40})\b", low)
                if not m:
                    m = re.search(r"\bonly\s+in\s+([a-z][a-z\s\-]{1,40})\b", low)
                if not m:
                    m = re.search(r"\bfilter\s+([a-z][a-z\s\-]{1,40})\b", low)
                region_txt = (m.group(1) if m else '').strip() if m else ''
                if region_txt:
                    entities['region'] = region_txt.title()
            except Exception:
                pass

    elif 'qualify' in intents:
        intent = 'check_eligibility'
        constraints['must_check_eligibility'] = True
    elif 'catalog_lookup' in intents:
        intent = 'program_details'
        pq = str(lookup.get('program_query') or '').strip()
        entities['program_query'] = pq or str(user_text or '').strip()
    elif 'institutions_by_region' in intents:
        intent = 'filter_results'
        reg = analysis.get('institutions_region')
        if reg:
            entities['region'] = str(reg)
    elif 'explain' in intents:
        intent = 'explain_recommendation'
    else:
        intent = 'recommend_programs'
        pq = str(lookup.get('program_query') or '').strip()
        txt = pq or str(user_text or '').strip()
        if 0 < len(txt) <= 120:
            entities['goal'] = txt

    return validate_plan_dict(
        {
            'intent': intent,
            'entities': entities,
            'constraints': constraints,
            'clarifying_question': '',
            'proposed_actions': [],
        }
    )


def _plan_with_gemini(user_text: str, ctx: Dict[str, Any]) -> Plan:
    from google import genai
    from google.genai import types

    api_key = (getattr(settings, 'GEMINI_API_KEY', '') or '').strip()
    model_name = (getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash') or 'gemini-1.5-flash').strip()

    system = (
        "You are a routing planner for a career guidance assistant. "
        "Return ONLY valid JSON with keys: "
        "intent, entities, constraints, clarifying_question, proposed_actions. "
        "Allowed intents: recommend_programs, search_programs, program_details, check_eligibility, "
        "filter_results, explain_recommendation, reset_context. "
        "Entities may include: goal, program_query, program_id, level, region, institution, budget, mode, sort_by. "
        "Constraints may include: k (1..20), must_check_eligibility (bool), tool_budget (0..3). "
        "proposed_actions is optional and must be a list of allowed actions: get_user_context, recommend_programs, "
        "search_programs, program_details, check_eligibility, explain_recommendation. "
        "If a key is unknown, omit it. Output JSON only, no markdown, no extra text."
    )

    prompt = build_planner_prompt(user_text=user_text, context=ctx)

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
        ),
    )
    raw = _strip_json_fences(resp.text or '{}')
    data = json.loads(raw)
    return validate_plan_dict(data)
