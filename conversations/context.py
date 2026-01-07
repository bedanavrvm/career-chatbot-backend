from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ConversationContext:
    active_filters: Dict[str, Any]
    last_results: List[Dict[str, Any]]
    last_intent: str
    known_user_facts: Dict[str, Any]


_CONTEXT_SLOT_KEY = 'chat_context_v1'


def load_context(session) -> ConversationContext:
    slots = getattr(session, 'slots', None) or {}
    raw = slots.get(_CONTEXT_SLOT_KEY) if isinstance(slots, dict) else None
    if not isinstance(raw, dict):
        raw = {}

    active_filters = raw.get('active_filters')
    if not isinstance(active_filters, dict):
        active_filters = {}

    last_results = raw.get('last_results')
    if not isinstance(last_results, list):
        last_results = []

    last_intent = str(raw.get('last_intent') or '').strip()

    known_user_facts = raw.get('known_user_facts')
    if not isinstance(known_user_facts, dict):
        known_user_facts = {}

    return ConversationContext(
        active_filters=dict(active_filters),
        last_results=list(last_results),
        last_intent=last_intent,
        known_user_facts=dict(known_user_facts),
    )


def save_context(session, ctx: ConversationContext) -> None:
    slots = getattr(session, 'slots', None) or {}
    if not isinstance(slots, dict):
        slots = {}

    slots[_CONTEXT_SLOT_KEY] = {
        'active_filters': ctx.active_filters or {},
        'last_results': ctx.last_results or [],
        'last_intent': ctx.last_intent or '',
        'known_user_facts': ctx.known_user_facts or {},
    }

    session.slots = slots


_ALLOWED_FILTER_KEYS = {'region', 'budget', 'level', 'mode', 'institution', 'sort_by'}


def merge_context(ctx: ConversationContext, entities: Dict[str, Any]) -> ConversationContext:
    if not isinstance(entities, dict):
        return ctx

    active = dict(ctx.active_filters or {})
    for k in _ALLOWED_FILTER_KEYS:
        if k in entities and entities.get(k) not in (None, '', [], {}):
            active[k] = entities.get(k)

    return ConversationContext(
        active_filters=active,
        last_results=list(ctx.last_results or []),
        last_intent=str(ctx.last_intent or ''),
        known_user_facts=dict(ctx.known_user_facts or {}),
    )


def set_last_results(ctx: ConversationContext, items: List[Dict[str, Any]]) -> ConversationContext:
    return ConversationContext(
        active_filters=dict(ctx.active_filters or {}),
        last_results=list(items or []),
        last_intent=str(ctx.last_intent or ''),
        known_user_facts=dict(ctx.known_user_facts or {}),
    )


def set_last_intent(ctx: ConversationContext, intent: str) -> ConversationContext:
    return ConversationContext(
        active_filters=dict(ctx.active_filters or {}),
        last_results=list(ctx.last_results or []),
        last_intent=str(intent or ''),
        known_user_facts=dict(ctx.known_user_facts or {}),
    )


def update_known_user_facts(
    ctx: ConversationContext,
    *,
    grades_present: Optional[bool] = None,
    traits_present: Optional[bool] = None,
    goals_present: Optional[bool] = None,
) -> ConversationContext:
    facts = dict(ctx.known_user_facts or {})
    if grades_present is not None:
        facts['grades_present'] = bool(grades_present)
    if traits_present is not None:
        facts['traits_present'] = bool(traits_present)
    if goals_present is not None:
        facts['goals_present'] = bool(goals_present)

    return ConversationContext(
        active_filters=dict(ctx.active_filters or {}),
        last_results=list(ctx.last_results or []),
        last_intent=str(ctx.last_intent or ''),
        known_user_facts=facts,
    )
