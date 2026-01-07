from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .context import (
    ConversationContext,
    load_context,
    merge_context,
    save_context,
    set_last_intent,
    set_last_results,
    update_known_user_facts,
)
from .planner import Plan, plan_message
from .tools import (
    check_eligibility,
    explain_recommendation,
    get_user_context,
    program_details,
    recommend_programs,
    search_programs,
)


_ALLOWED_PROPOSED_ACTIONS = {
    'get_user_context',
    'recommend_programs',
    'search_programs',
    'program_details',
    'check_eligibility',
    'explain_recommendation',
}


def _approve_proposed_actions(proposed_actions: Any) -> Tuple[List[str], List[str]]:
    if not isinstance(proposed_actions, list):
        return [], []

    approved: List[str] = []
    dropped: List[str] = []

    for a in proposed_actions:
        aa = str(a or '').strip()
        if not aa:
            continue
        if aa in _ALLOWED_PROPOSED_ACTIONS:
            if aa not in approved:
                approved.append(aa)
        else:
            dropped.append(aa)

    return approved, dropped


def route_turn(
    session,
    user_text: str,
    *,
    uid: str,
    provider_override: str = '',
    tool_budget: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    ctx = load_context(session)

    slots = getattr(session, 'slots', None) or {}
    if not isinstance(slots, dict):
        slots = {}

    ctx_snapshot = {
        'session_id': str(getattr(session, 'id', '') or ''),
        'fsm_state': str(getattr(session, 'fsm_state', '') or ''),
        'known_user_facts': ctx.known_user_facts,
        'active_filters': ctx.active_filters,
        'last_results': [{'program_id': x.get('program_id')} for x in (ctx.last_results or []) if isinstance(x, dict)],
    }

    plan = plan_message(user_text, context=ctx_snapshot, provider_override=provider_override)
    approved_actions, dropped_actions = _approve_proposed_actions(plan.proposed_actions)
    ctx = merge_context(ctx, plan.entities)
    ctx = set_last_intent(ctx, plan.intent)

    if tool_budget is None:
        try:
            tool_budget = int(plan.constraints.get('tool_budget') or 0)
        except Exception:
            tool_budget = 0

    if tool_budget <= 0:
        save_context(session, ctx)
        return None, {'planner': _plan_payload(plan, approved_actions=approved_actions, dropped_actions=dropped_actions)}

    out, tool_meta = _execute_plan(plan, ctx, session, uid=uid, user_text=user_text, tool_budget=int(tool_budget))
    save_context(session, ctx)

    nlp_payload: Dict[str, Any] = {
        'planner': _plan_payload(plan, approved_actions=approved_actions, dropped_actions=dropped_actions)
    }
    if tool_meta:
        nlp_payload['tools'] = tool_meta

    return out, nlp_payload


def _plan_payload(
    plan: Plan,
    *,
    approved_actions: Optional[List[str]] = None,
    dropped_actions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload = {
        'intent': plan.intent,
        'entities': plan.entities,
        'constraints': plan.constraints,
        'clarifying_question': plan.clarifying_question,
        'proposed_actions': plan.proposed_actions,
    }

    if approved_actions is not None:
        payload['approved_actions'] = approved_actions
    if dropped_actions is not None:
        payload['dropped_actions'] = dropped_actions

    return payload


def _resolve_program_id(user_text: str, ctx: ConversationContext) -> Optional[int]:
    s = str(user_text or '').strip().lower()

    m = re.search(r"\b(?:id|program)\s*[:#]?\s*(\d{1,10})\b", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    m = re.search(r"\b(\d{1,2})\b", s)
    if m:
        try:
            idx = int(m.group(1))
        except Exception:
            idx = 0
        if 1 <= idx <= 20 and isinstance(ctx.last_results, list) and 1 <= idx <= len(ctx.last_results):
            try:
                return int(ctx.last_results[idx - 1].get('program_id'))
            except Exception:
                return None

    if 'first' in s or '1st' in s:
        if isinstance(ctx.last_results, list) and ctx.last_results:
            try:
                return int(ctx.last_results[0].get('program_id'))
            except Exception:
                return None

    return None


def _execute_plan(
    plan: Plan,
    ctx: ConversationContext,
    session,
    *,
    uid: str,
    user_text: str,
    tool_budget: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    tool_budget = max(0, min(3, int(tool_budget or 0)))
    tool_meta: Dict[str, Any] = {}

    if plan.clarifying_question:
        return {'type': 'clarify', 'question': plan.clarifying_question}, tool_meta

    if plan.intent == 'reset_context':
        ctx.active_filters = {}
        ctx.last_results = []
        return {'type': 'reset'}, tool_meta

    if plan.intent == 'recommend_programs':
        if tool_budget < 2:
            return None, tool_meta
        user_ctx = get_user_context(uid=str(uid), session_id=str(getattr(session, 'id', '') or ''))
        goal_text = str(plan.entities.get('goal') or '').strip() or str(user_text or '').strip()
        k = int(plan.constraints.get('k') or 10)
        level = str(plan.entities.get('level') or ctx.active_filters.get('level') or 'bachelor').strip() or 'bachelor'

        res = recommend_programs(ctx=user_ctx, goal_text=goal_text, k=k, level=level)
        recs = res.get('recommendations') or []
        tool_meta['recommend_programs'] = {'count': len(recs)}

        ctx2 = set_last_results(ctx, [{'program_id': r.get('program_id'), 'score': r.get('score'), 'eligibility': r.get('eligibility')} for r in recs if isinstance(r, dict)])
        ctx3 = update_known_user_facts(
            ctx2,
            grades_present=bool(user_ctx.grades),
            traits_present=bool(user_ctx.traits),
            goals_present=bool(user_ctx.career_goals or goal_text),
        )
        ctx.active_filters = ctx3.active_filters
        ctx.last_results = ctx3.last_results
        ctx.last_intent = ctx3.last_intent
        ctx.known_user_facts = ctx3.known_user_facts

        return {'type': 'recommendations', 'items': recs}, tool_meta

    if plan.intent == 'search_programs':
        q = str(plan.entities.get('program_query') or plan.entities.get('goal') or '').strip() or str(user_text or '').strip()
        level = str(plan.entities.get('level') or ctx.active_filters.get('level') or '').strip()
        limit = int(plan.constraints.get('k') or 10)
        res = search_programs(query=q, level=level, limit=limit)
        results = res.get('results') or []
        tool_meta['search_programs'] = {'count': len(results)}
        return {'type': 'search_results', 'items': results}, tool_meta

    if plan.intent == 'program_details':
        pid = plan.entities.get('program_id')
        if pid is None:
            pid = _resolve_program_id(user_text, ctx)
        if pid is None:
            q = str(plan.entities.get('program_query') or '').strip() or str(user_text or '').strip()
            sres = search_programs(query=q, level=str(plan.entities.get('level') or '').strip(), limit=3)
            results = sres.get('results') or []
            tool_meta['search_programs'] = {'count': len(results)}
            if results:
                pid = results[0].get('program_id')

        try:
            pid_int = int(pid)
        except Exception:
            return {'type': 'clarify', 'question': 'Which program should I show details for? Please share the program id.'}, tool_meta

        det = program_details(program_id=pid_int)
        tool_meta['program_details'] = {'program_id': pid_int}
        return {'type': 'program_details', 'details': det}, tool_meta

    if plan.intent == 'check_eligibility':
        if tool_budget < 2:
            return None, tool_meta
        user_ctx = get_user_context(uid=str(uid), session_id=str(getattr(session, 'id', '') or ''))
        pid = plan.entities.get('program_id')
        ids: List[int] = []
        if pid is not None:
            try:
                ids = [int(pid)]
            except Exception:
                ids = []
        if not ids:
            pid2 = _resolve_program_id(user_text, ctx)
            if pid2 is not None:
                ids = [int(pid2)]

        if not ids and isinstance(ctx.last_results, list) and ctx.last_results:
            for it in ctx.last_results[:3]:
                try:
                    ids.append(int(it.get('program_id')))
                except Exception:
                    continue

        if not ids:
            return {'type': 'clarify', 'question': 'Which program should I check eligibility for? Please share the program id.'}, tool_meta

        if not user_ctx.grades:
            return {'type': 'clarify', 'question': 'I do not have your grades yet. Please complete onboarding (KCSE grades) or paste your subject grades here.'}, tool_meta

        res = check_eligibility(ctx=user_ctx, program_ids=ids)
        results = res.get('results') or []
        tool_meta['check_eligibility'] = {'count': len(results)}
        return {'type': 'eligibility', 'results': results}, tool_meta

    if plan.intent == 'filter_results':
        if not isinstance(ctx.last_results, list) or not ctx.last_results:
            return {'type': 'clarify', 'question': 'There are no active results to filter yet. Ask for recommendations first.'}, tool_meta

        region = str(ctx.active_filters.get('region') or plan.entities.get('region') or '').strip()
        level = str(ctx.active_filters.get('level') or plan.entities.get('level') or '').strip()
        institution = str(ctx.active_filters.get('institution') or plan.entities.get('institution') or '').strip()
        mode = str(ctx.active_filters.get('mode') or plan.entities.get('mode') or '').strip()

        ids: List[int] = []
        for it in ctx.last_results:
            if not isinstance(it, dict):
                continue
            try:
                ids.append(int(it.get('program_id')))
            except Exception:
                continue
        if not ids:
            return {'type': 'clarify', 'question': 'There are no active results to filter yet.'}, tool_meta

        try:
            from catalog.models import Program  # type: ignore
        except Exception:
            Program = None  # type: ignore
        if Program is None:
            return {'type': 'error', 'message': 'catalog_unavailable'}, tool_meta

        qs = Program.objects.select_related('institution', 'field').filter(id__in=ids)
        if region:
            qs = qs.filter(region__icontains=region)
        if level:
            qs = qs.filter(level=str(level))
        if mode:
            qs = qs.filter(mode__icontains=mode)
        if institution:
            qs = qs.filter(institution__name__icontains=institution)

        out_items: List[Dict[str, Any]] = []
        for p in qs[:20]:
            out_items.append(
                {
                    'program_id': int(p.id),
                    'program_code': (getattr(p, 'code', '') or '').strip(),
                    'program_name': (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip(),
                    'institution_name': ((p.institution.name if getattr(p, 'institution_id', None) else '') or '').strip(),
                    'field_name': ((p.field.name if getattr(p, 'field_id', None) else '') or '').strip(),
                    'level': (getattr(p, 'level', '') or '').strip(),
                    'region': (getattr(p, 'region', '') or '').strip(),
                    'campus': (getattr(p, 'campus', '') or '').strip(),
                }
            )

        ctx.last_results = [{'program_id': it.get('program_id')} for it in out_items]
        return {'type': 'filtered_results', 'items': out_items}, tool_meta

    if plan.intent == 'explain_recommendation':
        if tool_budget < 2:
            return None, tool_meta
        user_ctx = get_user_context(uid=str(uid), session_id=str(getattr(session, 'id', '') or ''))

        pid = plan.entities.get('program_id')
        if pid is None:
            pid = _resolve_program_id(user_text, ctx)
        try:
            pid_int = int(pid)
        except Exception:
            return {'type': 'clarify', 'question': 'Tell me which recommendation (by number or id) you want me to explain.'}, tool_meta

        goal_text = str(plan.entities.get('goal') or '').strip()
        exp = explain_recommendation(ctx=user_ctx, program_id=pid_int, goal_text=goal_text)
        tool_meta['explain_recommendation'] = {'program_id': pid_int}
        return {'type': 'explanation', 'explanation': exp}, tool_meta

    return None, tool_meta
