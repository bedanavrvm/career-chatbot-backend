from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from django.conf import settings
try:
    from django.db.models import Q
except Exception:
    Q = None
from .models import Session, Profile
from . import nlp
from .recommend import recommend_top_k, infer_career_paths, lookup_institutions_for_program, lookup_institutions_by_region, suggest_program_titles
try:
    from accounts.models import UserProfile as _UserProfile, OnboardingProfile as _OnboardingProfile
except Exception:
    _UserProfile = None
    _OnboardingProfile = None
try:
    from .providers.gemini_provider import compose_answer as _gem_compose  # type: ignore
except Exception:
    _gem_compose = None  # type: ignore

try:
    from .providers.gemini_provider import compose_rag_answer as _gem_rag_compose  # type: ignore
except Exception:
    _gem_rag_compose = None  # type: ignore

try:
    from .rag import retrieve_catalog_documents  # type: ignore
except Exception:
    retrieve_catalog_documents = None  # type: ignore

try:
    from scripts.etl.kuccps.eligibility import SUBJECT_CODE_ALIASES as _SUBJECT_CODE_ALIASES  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_CANON_TO_NUM as _SUBJECT_CANON_TO_NUM  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_TOKEN_ALIASES as _SUBJECT_TOKEN_ALIASES  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_TOKEN_CANON_TO_ALIASES as _SUBJECT_TOKEN_CANON_TO_ALIASES  # type: ignore
except Exception:
    _SUBJECT_CODE_ALIASES = {}
    _SUBJECT_CANON_TO_NUM = {}
    _SUBJECT_TOKEN_ALIASES = {}
    _SUBJECT_TOKEN_CANON_TO_ALIASES = {}


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


def _top_traits(traits: Dict[str, float], limit: int = 3, min_weight: float = 0.15) -> Dict[str, float]:
    items = []
    for k, v in (traits or {}).items():
        try:
            items.append((str(k), float(v or 0.0)))
        except Exception:
            items.append((str(k), 0.0))
    items.sort(key=lambda kv: -kv[1])
    out: Dict[str, float] = {}
    for k, v in items:
        if v <= 0:
            continue
        if v < float(min_weight or 0.0):
            continue
        out[k] = v
        if limit and len(out) >= int(limit):
            break
    return out


_GRADE_POINTS = {
    'A': 12,
    'A-': 11,
    'B+': 10,
    'B': 9,
    'B-': 8,
    'C+': 7,
    'C': 6,
    'C-': 5,
    'D+': 4,
    'D': 3,
    'D-': 2,
    'E': 1,
}


def _norm_grade(g: str) -> str:
    return str(g or '').strip().upper().replace(' ', '')


def _meets_min_grade(candidate: str, minimum: str) -> bool:
    c = _GRADE_POINTS.get(_norm_grade(candidate))
    m = _GRADE_POINTS.get(_norm_grade(minimum))
    if c is None or m is None:
        return False
    return int(c) >= int(m)


def _seed_profile_from_onboarding(session: Session, prof: Profile) -> None:
    if not session or not prof:
        return
    uid = (getattr(session, 'owner_uid', '') or '').strip()
    if not uid:
        return
    if _UserProfile is None or _OnboardingProfile is None:
        return
    try:
        user = _UserProfile.objects.filter(uid=uid).first()
        if not user:
            return
        ob = _OnboardingProfile.objects.filter(user=user).first()
        if not ob:
            return
    except Exception:
        return

    traits: Dict[str, float] = {}
    try:
        scores = ob.riasec_scores or {}
        if isinstance(scores, dict):
            raw: List[Tuple[str, float]] = []
            max_score = 0.0
            min_score = 0.0
            min_set = False
            for k, v in scores.items():
                try:
                    fv = float(v or 0.0)
                except Exception:
                    continue
                raw.append((str(k), fv))
                if not min_set:
                    min_score = fv
                    min_set = True
                if fv > max_score:
                    max_score = fv
                if fv < min_score:
                    min_score = fv
            rng = float(max_score - min_score)
            if rng > 0:
                tmp: List[Tuple[str, float]] = []
                for k, fv in raw:
                    tmp.append((k, max(0.0, min(1.0, (float(fv) - float(min_score)) / float(rng)))))
                tmp.sort(key=lambda kv: -kv[1])
                traits = {k: v for k, v in tmp[:4] if v and v > 0}
    except Exception:
        traits = {}

    try:
        hs = ob.high_school or {}
        fav = hs.get('favorite_subjects') if isinstance(hs, dict) else None
        if isinstance(fav, list) and fav:
            more = nlp.extract_traits(' '.join([str(x) for x in fav if str(x).strip()]))
            for k, v in (more or {}).items():
                traits[k] = max(float(traits.get(k, 0.0)), float(v))
    except Exception:
        pass

    prefs: Dict[str, Any] = {}
    try:
        uni = ob.universal or {}
        if isinstance(uni, dict):
            region = (uni.get('region') or '').strip()
            if region:
                prefs['region'] = region
            raw_goals = uni.get('careerGoals') if isinstance(uni, dict) else None
            if raw_goals is None:
                raw_goals = uni.get('career_goals') if isinstance(uni, dict) else None
            goals: List[str] = []
            if isinstance(raw_goals, list):
                goals = [str(x).strip() for x in raw_goals if str(x).strip()]
            elif isinstance(raw_goals, str):
                s = str(raw_goals).strip()
                if s:
                    parts = [p.strip() for p in s.replace('\n', ',').split(',')]
                    goals = [p for p in parts if p]
            if goals:
                prefs['career_goals'] = goals
    except Exception:
        pass
    try:
        p2 = ob.preferences or {}
        if isinstance(p2, dict):
            for k, v in p2.items():
                if k not in prefs:
                    prefs[k] = v
    except Exception:
        pass

    changed = False
    if traits and not (prof.traits or {}):
        prof.traits = traits
        changed = True

    if not (prof.grades or {}):
        try:
            hs = ob.high_school or {}
            grades = hs.get('subject_grades') if isinstance(hs, dict) else None
            if isinstance(grades, dict) and grades:
                gmap = {str(k).strip().upper(): str(v).strip().upper().replace(' ', '') for k, v in grades.items() if str(k).strip()}
                if gmap:
                    prof.grades = gmap
                    changed = True
        except Exception:
            pass
    if prefs:
        curp = prof.preferences or {}
        if not isinstance(curp, dict):
            curp = {}
        merged = dict(curp)
        for k, v in prefs.items():
            if k not in merged or not merged.get(k):
                merged[k] = v
        if merged != curp:
            prof.preferences = merged
            changed = True
    if changed:
        prof.save(update_fields=['traits', 'grades', 'preferences', 'updated_at'])


def next_turn(session: Session, user_text: str, provider_override: str = '') -> TurnResult:
    """Compute the next assistant reply/state from user_text using lightweight NLP.
    - Updates session slots and profile (grades/traits) deterministically.
    - Applies a confidence threshold to trigger clarifying prompts.
    States: greeting -> collect_interests -> summarize
    """
    analysis = nlp.analyze(user_text, provider_override=provider_override)
    conf = float(analysis.get('confidence') or 0.0)
    grades = analysis.get('grades') or {}
    traits = analysis.get('traits') or {}
    intents = analysis.get('intents') or []

    prof = _ensure_profile(session)
    _seed_profile_from_onboarding(session, prof)
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

    low = (user_text or '').strip().lower()
    if isinstance(intents, list) and 'programs_near_me' not in intents:
        try:
            import re
            m = re.search(r"\bfilter\s+by\s+([a-z][a-z\s\-]{1,40})\b", low)
            if not m:
                m = re.search(r"\bnear\s+([a-z][a-z\s\-]{1,40})\b", low)
            region_txt = (m.group(1) if m else '').strip() if m else ''
            if region_txt:
                prefs = prof.preferences or {}
                if not isinstance(prefs, dict):
                    prefs = {}
                if region_txt not in {'home', 'my home', 'home location', 'my home location'}:
                    prefs['region'] = region_txt
                    prof.preferences = prefs
                    prof.save(update_fields=['preferences', 'updated_at'])
                intents = list(intents) + ['programs_near_me']
        except Exception:
            pass

    def _get_career_goals() -> List[str]:
        try:
            prefs = prof.preferences or {}
        except Exception:
            prefs = {}
        raw = prefs.get('career_goals') if isinstance(prefs, dict) else None
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        if isinstance(raw, str) and str(raw).strip():
            return [str(raw).strip()]
        return []

    def _why_reasons_for_rec(r: Dict[str, Any], tuse: Dict[str, float], goals: List[str]) -> List[str]:
        nm = str(r.get('program_name') or '').strip()
        field_name = str(r.get('field_name') or '').strip()
        txt = f"{nm} {field_name}".lower()
        reasons: List[str] = []

        matches: List[str] = []
        for trait, w in (tuse or {}).items():
            try:
                hints = nlp.TRAIT_FIELD_HINTS.get(str(trait), [])  # type: ignore[attr-defined]
            except Exception:
                from .recommend import TRAIT_FIELD_HINTS
                hints = TRAIT_FIELD_HINTS.get(str(trait), [])
            hits = [h for h in (hints or []) if str(h).lower() in txt]
            if hits:
                matches.append(f"{trait} ({', '.join(sorted(set([str(h).lower() for h in hits]))[:3])})")
        if matches:
            reasons.append("Matches your interests: " + "; ".join(matches[:2]))

        if goals:
            joined = ' '.join(goals).lower()
            toks = [t for t in ''.join((ch if ch.isalnum() else ' ') for ch in joined).split() if len(t) >= 4]
            stop = {'become', 'becoming', 'want', 'wants', 'would', 'like', 'study', 'studying', 'career', 'goal', 'goals', 'work'}
            toks2 = [t for t in toks if t not in stop]
            hits2 = [t for t in sorted(set(toks2[:12])) if t in txt]
            if hits2:
                reasons.append("Matches your career goals: " + ", ".join(hits2[:4]))

        if not reasons:
            reasons.append("Good overall match based on your saved grades/traits.")
        return reasons

    def _eligibility_for_program(program: Any, grades_map: Dict[str, str]) -> Dict[str, Any]:
        try:
            groups = list(program.requirement_groups.all().order_by('order').prefetch_related('options', 'options__subject'))
        except Exception:
            groups = []
        if not groups:
            return {'eligible': None, 'missing': [], 'unmet_groups': 0}

        gmap: Dict[str, str] = {}
        for k, v in (grades_map or {}).items():
            kk = str(k).strip().upper().replace(' ', '')
            vv = _norm_grade(v)
            if not kk or not vv:
                continue
            canon = str(_SUBJECT_TOKEN_ALIASES.get(kk) or kk).strip().upper().replace(' ', '')
            gmap[kk] = vv
            if canon:
                gmap[canon] = vv
                for a in (_SUBJECT_TOKEN_CANON_TO_ALIASES.get(canon) or []):
                    aa = str(a or '').strip().upper().replace(' ', '')
                    if aa:
                        gmap[aa] = vv
            if kk in _SUBJECT_CODE_ALIASES:
                gmap[str(_SUBJECT_CODE_ALIASES.get(kk) or '').strip().upper()] = vv
            if kk in _SUBJECT_CANON_TO_NUM:
                gmap[str(_SUBJECT_CANON_TO_NUM.get(kk) or '').strip().upper()] = vv
            if canon in _SUBJECT_CANON_TO_NUM:
                gmap[str(_SUBJECT_CANON_TO_NUM.get(canon) or '').strip().upper()] = vv

        missing: List[str] = []
        unmet_groups = 0
        for grp in groups:
            try:
                pick = int(getattr(grp, 'pick', 1) or 1)
            except Exception:
                pick = 1
            try:
                opts = list(grp.options.all().order_by('order'))
            except Exception:
                opts = []
            if not opts:
                continue

            satisfied = 0
            grp_missing: List[str] = []
            for opt in opts:
                subj_code = ''
                try:
                    if getattr(opt, 'subject_id', None):
                        subj_code = (opt.subject.code or '').strip().upper()
                except Exception:
                    subj_code = ''
                if not subj_code:
                    subj_code = (getattr(opt, 'subject_code', '') or '').strip().upper()
                if not subj_code:
                    continue
                min_grade = _norm_grade(getattr(opt, 'min_grade', '') or '')
                cand = gmap.get(subj_code, '')
                if cand and (not min_grade or _meets_min_grade(cand, min_grade)):
                    satisfied += 1
                else:
                    grp_missing.append(f"{subj_code} >= {min_grade}" if min_grade else subj_code)

            if satisfied < pick:
                unmet_groups += 1
                missing.extend(grp_missing[: max(1, pick)])

        return {'eligible': unmet_groups == 0, 'missing': missing, 'unmet_groups': unmet_groups}

    try:
        threshold = float(getattr(settings, 'NLP_MIN_CONFIDENCE', 0.4) or 0.4)
    except Exception:
        threshold = 0.4
    state = (session.fsm_state or 'greeting').strip().lower()

    # Backward-compat: older sessions may still have this state.
    if state == 'collect_grades':
        state = 'collect_interests'

    # Global fast-path: answer catalog lookups immediately in any state
    # (e.g., "best universities that offer Bachelor of Arts").
    # This ensures it's not blocked by grade/interest collection.
    # We still define catalog_lookup() below.
    # Note: defer calling to after function is defined; so we only check here.

    def ask_for_grades() -> TurnResult:
        return TurnResult(
            reply=("I can help with recommendations. Could you share some KCSE grades? "
                   "For example: 'Math A-, English B+, Chemistry B'."),
            next_state='collect_interests',
            confidence=conf,
            slots=slots,
            nlp_payload=analysis,
        )

    def _goal_text_for_recommendation() -> str:
        goals = _get_career_goals()
        parts = [g for g in goals if str(g).strip()]

        # Strongly weight the user's current query when they are explicitly asking
        # for programs/careers (e.g., "medicine") so results aren't driven only by traits.
        if 'recommend' in intents:
            lookup = analysis.get('lookup') or {}
            program_query = str((lookup.get('program_query') or '')).strip()
            if program_query:
                parts.append(program_query)
            else:
                txt = str(user_text or '').strip()
                if 0 < len(txt) <= 80:
                    parts.append(txt)

        return " ".join(parts).strip()

    def greet() -> TurnResult:
        return TurnResult(
            reply=("Hi! I can help you with KUCCPS program recommendations, cutoffs, and requirements. "
                   "Tell me what you want to study or your career goal (e.g., 'Bachelor of Arts' or 'I want to do graphic design'). "
                   "If you want eligibility-checked recommendations, I'll use your saved onboarding grades (if available) or ask for them."),
            next_state='greeting',
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
        tshow = _top_traits(prof.traits or {}, limit=3, min_weight=0.15)
        itxt = ", ".join(list(tshow.keys())) or "(none)"
        reply = ("Thanks! Here's what I have so far.\n"
                 f"- Grades: {gtxt}\n"
                 f"- Interests: {itxt}\n"
                 "I'll use this to tailor program recommendations next.")
        return TurnResult(reply=reply, next_state='summarize', confidence=conf, slots=slots, nlp_payload=analysis)

    def recommend() -> TurnResult:
        gtxt = ", ".join(f"{k}:{v}" for k, v in (prof.grades or {}).items()) or "(none)"
        tuse = _top_traits(prof.traits or {}, limit=3, min_weight=0.15)
        itxt = ", ".join(list(tuse.keys())) or "(none)"
        goals = _get_career_goals()
        goal_text = _goal_text_for_recommendation()
        recs = recommend_top_k(prof.grades or {}, prof.traits or {}, k=5, goal_text=goal_text)
        # Persist the last recommendation set for follow-ups like "Why?" and "Which do I qualify for?"
        try:
            slots['last_recommendations'] = [
                {
                    'program_id': r.get('program_id'),
                    'program_code': r.get('program_code'),
                    'program_name': r.get('program_name'),
                    'institution_name': r.get('institution_name'),
                    'field_name': r.get('field_name'),
                    'level': r.get('level'),
                    'score': r.get('score'),
                }
                for r in (recs or [])
            ]
        except Exception:
            pass
        # Prepare potential program titles from career paths/traits and saved career goals
        gem_paths_all = analysis.get('career_paths') or []
        if isinstance(gem_paths_all, list):
            paths_for_titles = [str(p).strip() for p in gem_paths_all if str(p).strip()]
        else:
            paths_for_titles = []
        try:
            prefs = prof.preferences or {}
        except Exception:
            prefs = {}
        goals_raw = prefs.get('career_goals') if isinstance(prefs, dict) else None
        goals: List[str] = []
        if isinstance(goals_raw, list):
            goals = [str(x).strip() for x in goals_raw if str(x).strip()]
        elif isinstance(goals_raw, str) and str(goals_raw).strip():
            goals = [str(goals_raw).strip()]
        if goals:
            paths_for_titles = goals + paths_for_titles
        if not paths_for_titles:
            paths_for_titles = infer_career_paths(tuse or {})
        program_titles = suggest_program_titles(paths_for_titles, tuse or {}, limit=8)
        if recs:
            lines = ["Top recommendations:"]
            for i, r in enumerate(recs, 1):
                lines.append(f"{i}. {r['program_name']} — {r['institution_name']} [{r['program_code']}]")
            lines.append("")
            if program_titles:
                lines.append("Programs that suit your interests:")
                for i, nm in enumerate(program_titles[:6], 1):
                    lines.append(f"{i}. {nm}")
                lines.append("")
            lines.append("We can refine by region, cost, or mode. Try: 'filter by Nairobi' or 'rank by cost'.")
            lines.append("Ask: 'Why these recommendations?' or 'Which of these do I qualify for?'")
            body = "\n".join(lines)
        else:
            # Prefer Gemini-provided career paths if available; else fallback to local inference
            gem_paths = analysis.get('career_paths') or []
            if isinstance(gem_paths, list) and gem_paths:
                paths = [str(p).strip() for p in gem_paths if str(p).strip()]
            else:
                paths = infer_career_paths(tuse or {})
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
        can_ground = bool(recs) or bool(program_titles)
        if provider == 'gemini' and _gem_compose and can_ground:
            try:
                api_key = (getattr(settings, 'GEMINI_API_KEY', '') or '').strip()
                model_name = (getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash') or 'gemini-1.5-flash').strip()
                context = {
                    'grades': prof.grades or {},
                    'traits': tuse or {},
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

    def career_paths() -> TurnResult:
        low = (user_text or '').lower()
        if any(k in low for k in ['music', 'musician', 'singer', 'singing', 'composer', 'songwriter']):
            paths = ['musician', 'composer', 'songwriter', 'music producer', 'sound engineer', 'music teacher']
        else:
            tuse = _top_traits(prof.traits or {}, limit=3, min_weight=0.15)
            paths = infer_career_paths(tuse or {}, limit=8)
        if not paths:
            return TurnResult(
                reply="Tell me what you enjoy (e.g., music, design, science) and I will suggest possible career paths.",
                next_state=state,
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )
        lines = ["Possible career paths:"]
        for i, p in enumerate(paths[:8], 1):
            lines.append(f"{i}. {p}")
        lines.append("")
        lines.append("If you want, tell me your home region and I can suggest programs near you for any of these paths.")
        return TurnResult(reply="\n".join(lines), next_state='recommend', confidence=conf, slots=slots, nlp_payload=analysis)

    def explain() -> TurnResult:
        low = (user_text or '').lower()
        if any(k in low for k in ['asterisk', 'asterisks', 'clutter', 'cluttered', 'markdown', 'bullets', 'bullet']):
            return TurnResult(
                reply=(
                    "That formatting came from markdown-style bullets in an earlier response. "
                    "I will now output plain text lists (1., 2., 3.) so it stays clean in the chat."
                ),
                next_state='recommend',
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        kind = str(slots.get('last_reply_kind') or '').strip()
        if kind in ('catalog_lookup', 'catalog_lookup_empty'):
            q = str(slots.get('last_catalog_query') or '').strip()
            cnt = slots.get('last_catalog_count')
            try:
                cnti = int(cnt) if cnt is not None else None
            except Exception:
                cnti = None
            if kind == 'catalog_lookup_empty':
                msg = f"I couldn't find catalog matches for '{q}'."
                if not q:
                    msg = "I couldn't find catalog matches for your last query."
                msg += " This usually happens when the query is too broad or the wording doesn't match the catalog names."
                msg += " Try a more specific phrase (e.g., 'MBChB', 'Bachelor of Pharmacy', 'Dental Surgery')."
                return TurnResult(reply=msg, next_state='recommend', confidence=conf, slots=slots, nlp_payload=analysis)
            msg = "The previous answer was a catalog lookup."
            if q:
                msg += f" Query: '{q}'."
            if cnti is not None:
                msg += f" Matches found: {cnti}."
            msg += " I listed the closest matches from the KUCCPS catalog; ask for 'requirements', 'cutoff', or 'fees' for any item." 
            return TurnResult(reply=msg, next_state='recommend', confidence=conf, slots=slots, nlp_payload=analysis)

        last = slots.get('last_recommendations')
        if not isinstance(last, list) or not last:
            return TurnResult(
                reply="I can explain recommendations after I suggest programs. Ask me for recommendations first (or say what you want to study).",
                next_state='recommend',
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        tuse = _top_traits(prof.traits or {}, limit=3, min_weight=0.15)
        goals = _get_career_goals()
        lines = ["Why these were recommended:"]
        for i, r in enumerate(last[:5], 1):
            reasons = _why_reasons_for_rec(r or {}, tuse or {}, goals)
            title = f"{r.get('program_name') or ''} — {r.get('institution_name') or ''}".strip(' -')
            lines.append(f"{i}. {title}")
            lines.append(f"   Reason: {reasons[0]}")
        return TurnResult(reply="\n".join(lines), next_state='recommend', confidence=conf, slots=slots, nlp_payload=analysis)

    def qualify() -> TurnResult:
        grades_map = prof.grades or {}
        if not isinstance(grades_map, dict) or not grades_map:
            return TurnResult(
                reply="To check eligibility, please share your KCSE grades (e.g., 'Math A-, English B+, Biology B').",
                next_state='collect_interests',
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        last = slots.get('last_recommendations')
        if not isinstance(last, list) or not last:
            return TurnResult(
                reply="I can check eligibility for the programs I recommended last. Ask me for recommendations first, then say 'Which of these do I qualify for?'.",
                next_state='recommend',
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        try:
            from catalog.models import Program  # type: ignore
        except Exception:
            Program = None  # type: ignore
        if Program is None:
            return TurnResult(
                reply="I can't access the catalog database right now to check eligibility.",
                next_state='recommend',
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        ids: List[int] = []
        for r in last:
            try:
                pid = r.get('program_id') if isinstance(r, dict) else None
                if pid is not None:
                    ids.append(int(pid))
            except Exception:
                continue
        by_id: Dict[int, Any] = {}
        if ids:
            try:
                for p in Program.objects.filter(id__in=ids).prefetch_related('requirement_groups', 'requirement_groups__options', 'requirement_groups__options__subject'):
                    by_id[int(p.id)] = p
            except Exception:
                by_id = {}

        eligible_rows: List[Tuple[str, Dict[str, Any]]] = []
        unknown_titles: List[str] = []
        not_eligible_missing: List[str] = []

        for r in last[:5]:
            if not isinstance(r, dict):
                continue
            pid = r.get('program_id')
            nm = (r.get('program_name') or '').strip()
            inst = (r.get('institution_name') or '').strip()
            title = f"{nm} — {inst}".strip(' -')

            prog = None
            try:
                prog = by_id.get(int(pid)) if pid is not None else None
            except Exception:
                prog = None

            if prog is None:
                unknown_titles.append(title)
                continue

            elig = _eligibility_for_program(prog, grades_map)
            ok = elig.get('eligible')
            missing = elig.get('missing') or []
            if ok is True:
                eligible_rows.append((title, r))
            elif ok is False:
                if missing:
                    not_eligible_missing.extend([str(x) for x in missing[:3]])
            else:
                unknown_titles.append(title)

        lines: List[str] = []
        if eligible_rows:
            lines.append("You qualify for:")
            for i, (title, _r) in enumerate(eligible_rows[:5], 1):
                lines.append(f"{i}. {title}")
            if unknown_titles:
                lines.append("")
                lines.append("Some programs couldn't be eligibility-checked because requirements are missing in the catalog.")
        else:
            lines.append("You don't currently qualify for any of the last recommended programs based on your saved grades.")
            if not_eligible_missing:
                uniq = []
                seen = set()
                for x in not_eligible_missing:
                    k = str(x).strip().upper()
                    if not k or k in seen:
                        continue
                    seen.add(k)
                    uniq.append(str(x))
                    if len(uniq) >= 5:
                        break
                if uniq:
                    lines.append(f"Common missing requirements: {', '.join(uniq)}")

            goals = _get_career_goals()
            goal_text = " ".join([g for g in goals if str(g).strip()]).strip()
            candidates = recommend_top_k(prof.grades or {}, prof.traits or {}, k=25, goal_text=goal_text)
            cand_ids: List[int] = []
            for c in (candidates or []):
                try:
                    pid = c.get('program_id') if isinstance(c, dict) else None
                    if pid is not None:
                        cand_ids.append(int(pid))
                except Exception:
                    continue

            by_id2: Dict[int, Any] = {}
            if cand_ids:
                try:
                    for p in Program.objects.filter(id__in=cand_ids).prefetch_related('requirement_groups', 'requirement_groups__options', 'requirement_groups__options__subject'):
                        by_id2[int(p.id)] = p
                except Exception:
                    by_id2 = {}

            suggested: List[str] = []
            seen_titles = set([t.upper() for t in unknown_titles] + [t.upper() for t, _ in eligible_rows])
            for c in (candidates or []):
                if not isinstance(c, dict):
                    continue
                pid = c.get('program_id')
                nm = (c.get('program_name') or '').strip()
                inst = (c.get('institution_name') or '').strip()
                title = f"{nm} — {inst}".strip(' -')
                if not title or title.upper() in seen_titles:
                    continue
                prog = None
                try:
                    prog = by_id2.get(int(pid)) if pid is not None else None
                except Exception:
                    prog = None
                if prog is None:
                    continue
                elig = _eligibility_for_program(prog, grades_map)
                if elig.get('eligible') is True:
                    suggested.append(title)
                    seen_titles.add(title.upper())
                if len(suggested) >= 5:
                    break

            if suggested:
                lines.append("")
                lines.append("Similar options you qualify for:")
                for i, t in enumerate(suggested, 1):
                    lines.append(f"{i}. {t}")
            elif unknown_titles:
                lines.append("")
                lines.append("Some programs couldn't be checked because requirements are missing; open Program Details for full requirements.")

        return TurnResult(reply="\n".join(lines), next_state='recommend', confidence=conf, slots=slots, nlp_payload=analysis)

    def catalog_lookup() -> TurnResult:
        """Answer queries like 'best universities that offer Bachelor of Arts' from KUCCPS CSV."""
        lookup = (analysis.get('lookup') or {})
        program_query = (lookup.get('program_query') or '').strip()
        level = (lookup.get('level') or '').strip()
        qtext = program_query or user_text
        if not qtext:
            return TurnResult(
                reply=("I can look up programs and universities in the catalog. Please specify a program name, e.g., "
                       "'Bachelor of Arts' or 'Diploma in Nursing'."),
                next_state=state,
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        slots['last_reply_kind'] = 'catalog_lookup'
        slots['last_catalog_query'] = str(qtext)
        docs = []
        if retrieve_catalog_documents:
            try:
                docs = retrieve_catalog_documents(qtext, level=level, limit=8)
            except Exception:
                docs = []

        if not docs:
            lowq = (qtext or '').strip().lower()
            if any(t in lowq for t in ('medicine', 'medical', 'doctor', 'doctors', 'physician', 'surgeon', 'surgery')) and retrieve_catalog_documents:
                try:
                    q2 = 'medicine mbchb surgery dental pharmacy nursing'
                    docs = retrieve_catalog_documents(q2, level=level, limit=8)
                    if docs:
                        qtext = q2
                        slots['last_catalog_query'] = str(qtext)
                except Exception:
                    docs = []

        analysis['rag'] = {
            'query': qtext,
            'level': level,
            'count': len(docs),
            'sources': [d.get('meta') or {} for d in (docs or [])],
        }

        slots['last_catalog_count'] = len(docs)

        if not docs:
            slots['last_reply_kind'] = 'catalog_lookup_empty'
            return TurnResult(
                reply=(f"I couldn't find catalog matches for '{qtext}'. Try a different phrasing or include the level (bachelor/diploma/certificate)."),
                next_state=state,
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        provider = str(analysis.get('provider') or '').strip().lower()
        if provider == 'gemini' and _gem_rag_compose:
            try:
                api_key = (getattr(settings, 'GEMINI_API_KEY', '') or '').strip()
                model_name = (getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash') or 'gemini-1.5-flash').strip()
                composed = _gem_rag_compose(user_text, docs, api_key=api_key, model_name=model_name) if api_key else ''
                if composed:
                    c_low = composed.strip().lower()
                    if ('cannot recommend' not in c_low) and ("can't recommend" not in c_low):
                        return TurnResult(
                            reply=composed,
                            next_state='recommend',
                            confidence=conf,
                            slots=slots,
                            nlp_payload=analysis,
                        )
                    # Fall back to deterministic local listing below.
            except Exception:
                pass

        low = (user_text or '').lower()
        want_reqs = any(k in low for k in ['requirement', 'requirements', 'cluster', 'subjects'])
        want_cutoff = any(k in low for k in ['cutoff', 'cut-offs', 'cut off', 'points'])
        want_cost = any(k in low for k in ['cost', 'fee', 'fees', 'tuition', 'price'])

        lines = ["Here are relevant catalog matches:"]
        for d in docs:
            meta = d.get('meta') or {}
            cite = meta.get('citation') or d.get('citation') or ''
            prog = meta.get('program_name') or ''
            inst = meta.get('institution_name') or ''
            code = meta.get('program_code') or ''
            region = meta.get('region') or ''
            campus = meta.get('campus') or ''
            parts = [f"{prog} — {inst}"]
            if region:
                parts.append(region)
            if campus:
                parts.append(campus)
            if code:
                parts.append(f"[{code}]")
            if cite:
                parts.append(f"[{cite}]")
            lines.append(f"- " + " · ".join([p for p in parts if p]))
            if want_reqs and meta.get('requirements_preview'):
                lines.append(f"  Reqs: {meta.get('requirements_preview')} [{cite}]" if cite else f"  Reqs: {meta.get('requirements_preview')}")
            if want_cutoff and (meta.get('latest_cutoff') or {}).get('cutoff') is not None:
                yc = meta.get('latest_cutoff') or {}
                lines.append(f"  Cutoff {yc.get('year')}: {yc.get('cutoff')} [{cite}]" if cite else f"  Cutoff {yc.get('year')}: {yc.get('cutoff')}")
            if want_cost and (meta.get('cost') or {}).get('amount') is not None:
                pc = meta.get('cost') or {}
                lines.append(f"  Cost: {pc.get('amount')} {pc.get('currency')} [{cite}]" if cite else f"  Cost: {pc.get('amount')} {pc.get('currency')}")

        lines.append("")
        lines.append("Ask a follow-up like: 'requirements for P1' or 'cutoff for P2'.")
        return TurnResult(
            reply="\n".join(lines),
            next_state='recommend',
            confidence=conf,
            slots=slots,
            nlp_payload=analysis,
        )

    def programs_near_me() -> TurnResult:
        prefs = prof.preferences or {}
        home = ''
        if isinstance(prefs, dict):
            raw = prefs.get('region')
            home = str(raw or '').strip()
        if not home:
            return TurnResult(
                reply="I can filter programs by your home location. What county/region are you in (e.g., Nairobi, Kiambu, Central)?",
                next_state=state,
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        try:
            from catalog.models import Program  # type: ignore
        except Exception:
            Program = None  # type: ignore
        if Program is None:
            return TurnResult(
                reply="I can't access the catalog database right now, so I can't filter programs by your home location.",
                next_state='recommend',
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        tuse = _top_traits(prof.traits or {}, limit=3, min_weight=0.15)
        goals: List[str] = []
        try:
            prefs2 = prof.preferences or {}
        except Exception:
            prefs2 = {}
        goals_raw = prefs2.get('career_goals') if isinstance(prefs2, dict) else None
        if isinstance(goals_raw, list):
            goals = [str(x).strip() for x in goals_raw if str(x).strip()]
        elif isinstance(goals_raw, str) and str(goals_raw).strip():
            goals = [str(goals_raw).strip()]

        try:
            qs = Program.objects.select_related('institution', 'field').filter(level='bachelor')
            if Q is not None:
                qs = qs.filter(
                    Q(region__icontains=home)
                    | Q(institution__region__icontains=home)
                    | Q(institution__county__icontains=home)
                    | Q(campus__icontains=home)
                )
            else:
                qs = qs.filter(region__icontains=home)
        except Exception:
            qs = []

        picked = []
        try:
            for p in list(qs[:500]):
                nm = (p.normalized_name or p.name or '').strip()
                inst = (p.institution.name or '').strip() if getattr(p, 'institution_id', None) else ''
                region = (p.region or '').strip() or (p.institution.region or '').strip() if getattr(p, 'institution_id', None) else ''
                field_name = (p.field.name if getattr(p, 'field_id', None) else '') or ''
                txt = f"{nm} {field_name}".lower()
                sc = 0.0
                for trait, w in (tuse or {}).items():
                    try:
                        hints = nlp.TRAIT_FIELD_HINTS.get(str(trait), [])  # type: ignore[attr-defined]
                    except Exception:
                        from .recommend import TRAIT_FIELD_HINTS
                        hints = TRAIT_FIELD_HINTS.get(str(trait), [])
                    hits = sum(1 for h in (hints or []) if str(h).lower() in txt)
                    try:
                        ww = float(w or 0.0)
                    except Exception:
                        ww = 0.0
                    if hits > 0:
                        sc += ww
                if goals:
                    joined = ' '.join(goals).lower()
                    toks = [t for t in ''.join((ch if ch.isalnum() else ' ') for ch in joined).split() if len(t) >= 4]
                    stop = {'become', 'becoming', 'want', 'wants', 'would', 'like', 'study', 'studying', 'career', 'goal', 'goals', 'work'}
                    toks2 = [t for t in toks if t not in stop]
                    hits2 = sum(1 for t in set(toks2[:12]) if t in txt)
                    if hits2 > 0:
                        sc += min(1.0, 0.25 * float(hits2))
                picked.append((float(sc), nm, inst, region))
        except Exception:
            picked = []

        if not picked:
            return TurnResult(
                reply=f"I couldn't find catalog programs matching your home location ({home}). If you tell me a county/region name (e.g., Kiambu, Nairobi, Central), I can try again.",
                next_state='recommend',
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        picked.sort(key=lambda x: (-x[0], x[1]))
        top = picked[:5]
        lines = [f"Here are programs offered close to your home location ({home}):"]
        for i, (_sc, nm, inst, region) in enumerate(top, 1):
            parts = [nm]
            if inst:
                parts.append(inst)
            if region:
                parts.append(region)
            lines.append(f"{i}. " + " — ".join([p for p in parts if p]))
        lines.append("")
        lines.append("If you share a field (e.g., 'nursing' or 'engineering'), I can narrow it further.")
        return TurnResult(
            reply="\n".join(lines),
            next_state='recommend',
            confidence=conf,
            slots=slots,
            nlp_payload=analysis,
        )

    def institutions_by_region() -> TurnResult:
        region = (analysis.get('institutions_region') or '').strip()
        if not region:
            return TurnResult(
                reply="Which region are you interested in? For example: Central, Eastern, Rift Valley, Nairobi, Coast.",
                next_state=state,
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        insts = []
        try:
            insts = lookup_institutions_by_region(region, limit=20)
        except Exception:
            insts = []

        if not insts:
            return TurnResult(
                reply=(
                    f"I couldn't find any universities/colleges for '{region}' in the catalog database. "
                    "This usually means the database hasn't been populated with institution region/county data yet. "
                    "If you're running this locally, run the ETL load step to populate the Institution tables."
                ),
                next_state=state,
                confidence=conf,
                slots=slots,
                nlp_payload=analysis,
            )

        top_n = 3
        low = (user_text or '').lower()
        m = None
        try:
            import re
            m = re.search(r"\b(\d+)\b", low)
        except Exception:
            m = None
        if m:
            try:
                top_n = max(1, min(10, int(m.group(1))))
            except Exception:
                top_n = 3

        lines = [f"Here are {min(top_n, len(insts))} institutions in {region}:"]
        for i, r in enumerate(insts[:top_n], 1):
            nm = (r.get('institution_name') or '').strip()
            county = (r.get('county') or '').strip()
            parts = [nm]
            if county:
                parts.append(county)
            lines.append(f"{i}. " + " — ".join([p for p in parts if p]))

        lines.append("")
        lines.append("Ask a follow-up like: 'What programs do they offer for P1?' or 'Show universities in Nairobi'.")
        return TurnResult(
            reply="\n".join(lines),
            next_state='recommend',
            confidence=conf,
            slots=slots,
            nlp_payload=analysis,
        )

    # Transitions
    if 'explain' in intents:
        return explain()
    if 'qualify' in intents:
        return qualify()
    if 'career_paths' in intents:
        return career_paths()
    if 'catalog_lookup' in intents:
        return catalog_lookup()
    if 'institutions_by_region' in intents:
        return institutions_by_region()
    if 'programs_near_me' in intents:
        return programs_near_me()
    if state == 'greeting':
        if 'catalog_lookup' in intents:
            return catalog_lookup()
        if 'institutions_by_region' in intents:
            return institutions_by_region()
        if 'recommend' in intents or 'next' in intents:
            if prof.grades or prof.traits or (isinstance(prof.preferences, dict) and (prof.preferences.get('career_goals') or prof.preferences.get('region'))):
                return recommend()
            return ask_for_interests()
        if 'ask_grades' in intents:
            return ask_for_grades()
        if grades:
            return ask_for_interests()
        if traits or 'interests' in intents:
            return ask_for_interests()
        if 'help' in intents or 'greeting' in intents:
            return greet()
        return greet()

    if state == 'collect_interests':
        if 'catalog_lookup' in intents:
            return catalog_lookup()
        if 'institutions_by_region' in intents:
            return institutions_by_region()
        if 'programs_near_me' in intents:
            return programs_near_me()
        if 'recommend' in intents or 'next' in intents or 'help' in intents:
            return recommend()
        if traits or 'interests' in intents or conf >= threshold:
            return recommend()
        if conf < threshold and not traits:
            return ask_for_interests()
        return recommend()

    return recommend()
