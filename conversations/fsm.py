from dataclasses import dataclass
from typing import Dict, Any
from django.conf import settings
from .models import Session, Profile
from . import nlp
from .recommend import recommend_top_k


@dataclass
class TurnResult:
    reply: str
    next_state: str
    confidence: float
    slots: Dict[str, Any]
    nlp_payload: Dict[str, Any]


def _merge_grades(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    merged = dict(old or {})
    for k, v in (new or {}).items():
        merged[k] = v
    return merged


def _ensure_profile(session: Session) -> Profile:
    prof, _ = Profile.objects.get_or_create(session=session)
    return prof


def next_turn(session: Session, user_text: str) -> TurnResult:
    """Compute the next assistant reply/state from user_text using lightweight NLP.
    - Updates session slots and profile (grades/traits) deterministically.
    - Applies a confidence threshold to trigger clarifying prompts.
    States: greeting -> collect_grades -> collect_interests -> summarize
    """
    analysis = nlp.analyze(user_text)
    conf = float(analysis.get('confidence') or 0.0)
    grades = analysis.get('grades') or {}
    traits = analysis.get('traits') or {}
    intents = analysis.get('intents') or []

    prof = _ensure_profile(session)
    # Merge grades/traits into profile
    if grades:
        prof.grades = _merge_grades(prof.grades or {}, grades)
    if traits:
        # Take max per trait as a simple aggregator
        tcur = prof.traits or {}
        for k, v in traits.items():
            tcur[k] = max(float(tcur.get(k, 0.0)), float(v))
        prof.traits = tcur
    prof.save()

    # Copy into session slots snapshot
    slots = dict(session.slots or {})
    if prof.grades:
        slots['grades'] = prof.grades
    if prof.traits:
        slots['traits'] = prof.traits

    try:
        threshold = float(getattr(settings, 'NLP_MIN_CONFIDENCE', 0.4) or 0.4)
    except Exception:
        threshold = 0.4
    state = (session.fsm_state or 'greeting').strip().lower()

    def ask_for_grades() -> TurnResult:
        return TurnResult(
            reply=("I can help with recommendations. Could you share some KCSE grades? "
                   "For example: 'Math A-, English B+, Chemistry B'."),
            next_state='collect_grades',
            confidence=conf,
            slots=slots,
            nlp_payload=analysis,
        )

    def ask_for_interests() -> TurnResult:
        return TurnResult(
            reply=("Great, thanks. What subjects or activities do you enjoy? "
                   "For example: 'I enjoy coding and designing websites'."),
            next_state='collect_interests',
            confidence=conf,
            slots=slots,
            nlp_payload=analysis,
        )

    def summarize() -> TurnResult:
        gtxt = ", ".join(f"{k}:{v}" for k, v in (prof.grades or {}).items()) or "(none)"
        itxt = ", ".join(sorted((prof.traits or {}).keys())) or "(none)"
        reply = ("Thanks! Here's what I have so far.\n"
                 f"- Grades: {gtxt}\n"
                 f"- Interests: {itxt}\n"
                 "I'll use this to tailor program recommendations next.")
        return TurnResult(reply=reply, next_state='summarize', confidence=conf, slots=slots, nlp_payload=analysis)

    def recommend() -> TurnResult:
        gtxt = ", ".join(f"{k}:{v}" for k, v in (prof.grades or {}).items()) or "(none)"
        itxt = ", ".join(sorted((prof.traits or {}).keys())) or "(none)"
        recs = recommend_top_k(prof.grades or {}, prof.traits or {}, k=5)
        if recs:
            lines = ["Top recommendations:"]
            for i, r in enumerate(recs, 1):
                lines.append(f"{i}. {r['program_name']} — {r['institution_name']} [{r['program_code']}]")
            lines.append("")
            lines.append("We can refine by region, cost, or mode. Try: 'filter by Nairobi' or 'rank by cost'.")
            body = "\n".join(lines)
        else:
            body = "No strong matches yet. Share more interests or subjects to personalize results."
        reply = ("Thanks! Here's what I have so far.\n"
                 f"- Grades: {gtxt}\n"
                 f"- Interests: {itxt}\n\n"
                 f"{body}")
        return TurnResult(reply=reply, next_state='recommend', confidence=conf, slots=slots, nlp_payload=analysis)

    # Transitions
    if state == 'greeting':
        if conf < threshold and not grades:
            return ask_for_grades()
        if 'recommend' in intents or 'next' in intents or 'help' in intents:
            return recommend()
        if grades:
            return ask_for_interests()
        return ask_for_grades()

    if state == 'collect_grades':
        if grades:
            if 'recommend' in intents or 'next' in intents or 'help' in intents:
                return recommend()
            return ask_for_interests()
        if conf < threshold:
            return ask_for_grades()
        return ask_for_interests()

    if state == 'collect_interests':
        if 'recommend' in intents or 'next' in intents or 'help' in intents:
            return recommend()
        if traits or 'interests' in intents or conf >= threshold:
            return recommend()
        if conf < threshold and not traits:
            return ask_for_interests()
        return recommend()

    return recommend()
