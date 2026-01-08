from __future__ import annotations

from typing import Optional

from .compose import compose_response
from .context import load_context
from .fsm import TurnResult
from .router import route_turn


def planner_turn(session, user_text: str, *, uid: str, provider_override: str = '') -> Optional[TurnResult]:
    ctx = load_context(session)
    tool_budget = None
    if isinstance(ctx.known_user_facts, dict):
        tool_budget = None

    out, nlp_payload = route_turn(
        session,
        user_text,
        uid=str(uid),
        provider_override=provider_override,
        tool_budget=tool_budget,
    )

    if out is None:
        return None

    if isinstance(out, dict) and str(out.get('type') or '').strip() == 'recommendations':
        items = out.get('items') or []
        stretch = out.get('stretch_items') or []
        if isinstance(items, list) and isinstance(stretch, list) and isinstance(nlp_payload, dict):
            nlp_payload['turn_recommendations'] = {
                'count': len(items),
                'recommendations': items,
                'stretch_count': len(stretch),
                'stretch_recommendations': stretch,
                'goal_text': str(out.get('goal_text') or '').strip(),
                'k': out.get('k'),
                'level': str(out.get('level') or '').strip(),
            }

    reply = compose_response(out)
    if not reply:
        return None

    slots = dict(getattr(session, 'slots', None) or {})
    return TurnResult(
        reply=reply,
        next_state=str(getattr(session, 'fsm_state', '') or ''),
        confidence=1.0,
        slots=slots,
        nlp_payload=nlp_payload,
    )
