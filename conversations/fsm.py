from dataclasses import dataclass
from typing import Dict, Any
from django.conf import settings
from .models import Session, Profile
from . import nlp
from .recommend import recommend_top_k, infer_career_paths, lookup_institutions_for_program, suggest_program_titles
try:
    from .providers.gemini_provider import compose_answer as _gem_compose  # type: ignore
except Exception:
    _gem_compose = None  # type: ignore


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

    # Global fast-path: answer catalog lookups immediately in any state
    # (e.g., "best universities that offer Bachelor of Arts").
    # This ensures it's not blocked by grade/interest collection.
    # We still define catalog_lookup() below.
    # Note: defer calling to after function is defined; so we only check here.

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
        # Prepare potential program titles from career paths/traits
        gem_paths_all = analysis.get('career_paths') or []
        if isinstance(gem_paths_all, list):
            paths_for_titles = [str(p).strip() for p in gem_paths_all if str(p).strip()]
        else:
            paths_for_titles = []
        if not paths_for_titles:
            paths_for_titles = infer_career_paths(prof.traits or {})
        program_titles = suggest_program_titles(paths_for_titles, prof.traits or {}, limit=8)
        if recs:
            lines = ["Top recommendations:"]
            for i, r in enumerate(recs, 1):
                lines.append(f"{i}. {r['program_name']} — {r['institution_name']} [{r['program_code']}]")
            lines.append("")
            if program_titles:
                lines.append("Programs that suit your interests:")
                for i, nm in enumerate(program_titles[:6], 1):
                    lines.append(f"- {nm}")
                lines.append("")
            lines.append("We can refine by region, cost, or mode. Try: 'filter by Nairobi' or 'rank by cost'.")
            body = "\n".join(lines)
        else:
            # Prefer Gemini-provided career paths if available; else fallback to local inference
            gem_paths = analysis.get('career_paths') or []
            if isinstance(gem_paths, list) and gem_paths:
                paths = [str(p).strip() for p in gem_paths if str(p).strip()]
            else:
                paths = infer_career_paths(prof.traits or {})
            if program_titles:
                plines = ["Programs that suit your career paths:"]
                for i, nm in enumerate(program_titles, 1):
                    plines.append(f"{i}. {nm}")
                plines.append("")
                plines.append("Ask me for universities that offer any of these (e.g., 'who offers Bachelor of Science in Physics?').")
                body = "\n".join(plines)
            elif paths:
                plines = ["Likely career paths (based on your interests):"]
                for i, p in enumerate(paths, 1):
                    plines.append(f"{i}. {p}")
                plines.append("")
                plines.append("Share more preferences (region/cost/mode) or more subjects to refine specific programs.")
                body = "\n".join(plines)
            else:
                body = "No strong matches yet. Share more interests or subjects to personalize results."
        # If Gemini is the provider, let it compose a grounded answer using context
        provider = str(analysis.get('provider') or '').strip().lower()
        if provider == 'gemini' and _gem_compose:
            try:
                api_key = (getattr(settings, 'GEMINI_API_KEY', '') or '').strip()
                model_name = (getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash') or 'gemini-1.5-flash').strip()
                context = {
                    'grades': prof.grades or {},
                    'traits': prof.traits or {},
                    'career_paths': analysis.get('career_paths') or [],
                    'program_recommendations': recs or [],
                    'program_titles': program_titles or [],
                    'provider': analysis.get('provider') or 'gemini',
                }
                composed = _gem_compose(user_text, context, api_key=api_key, model_name=model_name) if api_key else ''
                if composed:
                    reply = composed
                else:
                    reply = ("Thanks! Here's what I have so far.\n"
                             f"- Grades: {gtxt}\n"
                             f"- Interests: {itxt}\n\n"
                             f"{body}")
            except Exception:
                reply = ("Thanks! Here's what I have so far.\n"
                         f"- Grades: {gtxt}\n"
                         f"- Interests: {itxt}\n\n"
                         f"{body}")
        else:
            reply = ("Thanks! Here's what I have so far.\n"
                     f"- Grades: {gtxt}\n"
                     f"- Interests: {itxt}\n\n"
                     f"{body}")
        return TurnResult(reply=reply, next_state='recommend', confidence=conf, slots=slots, nlp_payload=analysis)

    def catalog_lookup() -> TurnResult:
        """Answer queries like 'best universities that offer Bachelor of Arts' from KUCCPS CSV."""
        lookup = (analysis.get('lookup') or {})
        program_query = (lookup.get('program_query') or '').strip()
        level = (lookup.get('level') or '').strip()
        if not program_query:
            return TurnResult(
                reply=("I can look up universities offering a program. Please specify the program name, e.g., "
                       "'Bachelor of Arts' or 'Diploma in Nursing'."),
                next_state=state,
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )
        rows = lookup_institutions_for_program(program_query, level=level, limit=30)
        if rows:
            title_prog = program_query.title()
            lines = [f"Universities offering {title_prog} (showing up to {len(rows)}):"]
            for i, r in enumerate(rows, 1):
                code = f" [{r['program_code']}]" if r.get('program_code') else ""
                region = f" — {r['region']}" if r.get('region') else ""
                lines.append(f"{i}. {r['institution_name']}{region}{code}")
            lines.append("")
            lines.append("You can refine by region, mode or ask for recommended picks.")
            return TurnResult(
                reply="\n".join(lines),
                next_state='recommend',
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )
        else:
            return TurnResult(
                reply=(f"I couldn't find institutions for '{program_query}'. Try a different phrasing or level (e.g., 'bachelor', 'diploma')."),
                next_state=state,
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

    # Transitions
    if 'catalog_lookup' in intents:
        return catalog_lookup()
    if state == 'greeting':
        if 'catalog_lookup' in intents:
            return catalog_lookup()
        if conf < threshold and not grades:
            return ask_for_grades()
        if 'recommend' in intents or 'next' in intents or 'help' in intents:
            return recommend()
        if grades:
            return ask_for_interests()
        return ask_for_grades()

    if state == 'collect_grades':
        if 'catalog_lookup' in intents:
            return catalog_lookup()
        if grades:
            if 'recommend' in intents or 'next' in intents or 'help' in intents:
                return recommend()
            return ask_for_interests()
        if conf < threshold:
            return ask_for_grades()
        return ask_for_interests()

    if state == 'collect_interests':
        if 'catalog_lookup' in intents:
            return catalog_lookup()
        if 'recommend' in intents or 'next' in intents or 'help' in intents:
            return recommend()
        if traits or 'interests' in intents or conf >= threshold:
            return recommend()
        if conf < threshold and not traits:
            return ask_for_interests()
        return recommend()

    return recommend()
