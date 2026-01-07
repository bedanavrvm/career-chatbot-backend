from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from django.db.models import Q

from .models import Profile, Session
from .recommendations_service import build_recommendations, eligibility_for_program, score_program_breakdown

try:
    from accounts.models import UserProfile as _UserProfile, OnboardingProfile as _OnboardingProfile  # type: ignore
except Exception:
    _UserProfile = None
    _OnboardingProfile = None


@dataclass(frozen=True)
class UserContext:
    uid: str
    grades: Dict[str, str]
    traits: Dict[str, float]
    career_goals: List[str]
    preferences: Dict[str, Any]


def _normalize_traits_from_onboarding(ob) -> Dict[str, float]:
    tops = getattr(ob, 'riasec_top', None) or []
    if isinstance(tops, list) and tops:
        out: Dict[str, float] = {}
        t0 = str(tops[0] or '').strip()
        t1 = str(tops[1] or '').strip() if len(tops) > 1 else ''
        t2 = str(tops[2] or '').strip() if len(tops) > 2 else ''
        if t0:
            out[t0] = 1.0
        if t1:
            out[t1] = 0.7
        if t2:
            out[t2] = 0.5
        return out

    scores = getattr(ob, 'riasec_scores', None) or {}
    if not isinstance(scores, dict) or not scores:
        return {}

    raw = []
    for k, v in scores.items():
        try:
            raw.append((str(k), float(v or 0.0)))
        except Exception:
            continue
    if not raw:
        return {}

    vals = [v for _k, v in raw]
    mx = max(vals)
    mn = min(vals)
    rng = float(mx - mn)
    if rng <= 0:
        raw.sort(key=lambda kv: kv[0])
        return {k: 0.25 for k, _v in raw[:3]}

    tmp = [(k, max(0.0, min(1.0, (v - mn) / rng))) for k, v in raw]
    tmp.sort(key=lambda kv: -kv[1])
    return {k: float(v) for k, v in tmp[:4] if v > 0}


def _seed_profile_from_onboarding(uid: str, prof: Profile) -> None:
    if not uid or not prof:
        return
    if _UserProfile is None or _OnboardingProfile is None:
        return

    try:
        user = _UserProfile.objects.filter(uid=str(uid)).first()
        if not user:
            return
        ob = _OnboardingProfile.objects.filter(user=user).first()
        if not ob:
            return
    except Exception:
        return

    changed = False

    # Traits
    traits_from_ob = _normalize_traits_from_onboarding(ob)
    if traits_from_ob:
        cur = prof.traits or {}
        if not isinstance(cur, dict):
            cur = {}
        merged = dict(cur)
        for k, v in traits_from_ob.items():
            try:
                merged[str(k)] = max(float(merged.get(str(k), 0.0) or 0.0), float(v or 0.0))
            except Exception:
                merged[str(k)] = float(v or 0.0)
        if merged != cur:
            prof.traits = merged
            changed = True

    # Grades
    hs = getattr(ob, 'high_school', None) or {}
    grades = hs.get('subject_grades') if isinstance(hs, dict) else None
    if isinstance(grades, dict) and grades:
        gmap = {str(k).strip().upper(): str(v).strip().upper().replace(' ', '') for k, v in grades.items() if str(k).strip()}
        if gmap:
            curg = prof.grades or {}
            if not isinstance(curg, dict):
                curg = {}
            mergedg = dict(curg)
            for k, v in gmap.items():
                if str(mergedg.get(k) or '').strip():
                    continue
                mergedg[k] = v
            if mergedg != curg:
                prof.grades = mergedg
                changed = True

    # Goals
    prefs = prof.preferences or {}
    if not isinstance(prefs, dict):
        prefs = {}

    cur_goal = prefs.get('career_goals')
    has_goal = False
    if isinstance(cur_goal, list):
        has_goal = any(str(x).strip() for x in cur_goal)
    elif isinstance(cur_goal, str):
        has_goal = bool(str(cur_goal).strip())

    if not has_goal:
        uni = getattr(ob, 'universal', None) or {}
        raw = uni.get('careerGoals') if isinstance(uni, dict) else None
        if raw is None and isinstance(uni, dict):
            raw = uni.get('career_goals')
        goals: List[str] = []
        if isinstance(raw, list):
            goals = [str(x).strip() for x in raw if str(x).strip()]
        elif isinstance(raw, str) and str(raw).strip():
            s = str(raw).strip()
            parts = [p.strip() for p in s.replace('\n', ',').split(',')]
            goals = [p for p in parts if p]
        if goals:
            prof.preferences = {**prefs, 'career_goals': goals}
            changed = True

    if changed:
        try:
            prof.save(update_fields=['traits', 'grades', 'preferences', 'updated_at'])
        except Exception:
            prof.save()


def get_user_context(*, uid: str, session_id: Optional[str] = None) -> UserContext:
    sess = None
    prof = None
    if session_id:
        try:
            sess = Session.objects.get(id=session_id)
        except Exception:
            sess = None
        if sess is not None:
            prof, _ = Profile.objects.get_or_create(session=sess)

    if prof is not None:
        _seed_profile_from_onboarding(str(uid), prof)

        grades = prof.grades or {}
        traits = prof.traits or {}
        prefs = prof.preferences or {}
        if not isinstance(grades, dict):
            grades = {}
        if not isinstance(traits, dict):
            traits = {}
        if not isinstance(prefs, dict):
            prefs = {}
    else:
        grades = {}
        traits = {}
        prefs = {}
        if _UserProfile is not None and _OnboardingProfile is not None:
            try:
                user = _UserProfile.objects.filter(uid=str(uid)).first()
                ob = _OnboardingProfile.objects.filter(user=user).first() if user else None
            except Exception:
                ob = None

            if ob is not None:
                traits = _normalize_traits_from_onboarding(ob)
                hs = getattr(ob, 'high_school', None) or {}
                g = hs.get('subject_grades') if isinstance(hs, dict) else None
                if isinstance(g, dict) and g:
                    grades = {str(k).strip().upper(): str(v).strip().upper().replace(' ', '') for k, v in g.items() if str(k).strip()}
                prefs = {}
                uni = getattr(ob, 'universal', None) or {}
                if isinstance(uni, dict):
                    raw = uni.get('careerGoals')
                    if raw is None:
                        raw = uni.get('career_goals')
                    goals: List[str] = []
                    if isinstance(raw, list):
                        goals = [str(x).strip() for x in raw if str(x).strip()]
                    elif isinstance(raw, str) and str(raw).strip():
                        s = str(raw).strip()
                        parts = [p.strip() for p in s.replace('\n', ',').split(',')]
                        goals = [p for p in parts if p]
                    if goals:
                        prefs['career_goals'] = goals

    raw_goals = prefs.get('career_goals')
    goals: List[str] = []
    if isinstance(raw_goals, list):
        goals = [str(x).strip() for x in raw_goals if str(x).strip()]
    elif isinstance(raw_goals, str) and str(raw_goals).strip():
        goals = [str(raw_goals).strip()]

    traits_f: Dict[str, float] = {}
    for k, v in traits.items():
        try:
            traits_f[str(k)] = float(v or 0.0)
        except Exception:
            continue

    return UserContext(
        uid=str(uid),
        grades={str(k).strip().upper(): str(v).strip().upper().replace(' ', '') for k, v in grades.items() if str(k).strip()},
        traits=traits_f,
        career_goals=goals,
        preferences=prefs,
    )


def recommend_programs(
    *,
    ctx: UserContext,
    goal_text: str = '',
    k: int = 10,
    level: str = 'bachelor',
) -> Dict[str, Any]:
    goal_text_all = ' '.join([str(goal_text or '').strip()] + [g for g in (ctx.career_goals or []) if str(g).strip()]).strip()
    k = max(1, min(20, int(k or 10)))

    recs, stretch = build_recommendations(
        ctx.grades or {},
        ctx.traits or {},
        goal_text=goal_text_all,
        k=k,
        level=str(level or 'bachelor') or 'bachelor',
        max_scan=2000,
        raise_on_missing_catalog=False,
    )

    return {
        'count': len(recs or []),
        'recommendations': recs or [],
        'stretch_count': len(stretch or []),
        'stretch_recommendations': stretch or [],
    }


def search_programs(*, query: str, level: str = '', limit: int = 10) -> Dict[str, Any]:
    try:
        from catalog.models import Program  # type: ignore
    except Exception:
        Program = None  # type: ignore

    if Program is None:
        return {'count': 0, 'results': []}

    q = (query or '').strip()
    if not q:
        return {'count': 0, 'results': []}

    limit = max(1, min(20, int(limit or 10)))

    qs = Program.objects.all()
    if level:
        qs = qs.filter(level=str(level).strip())
    qs = qs.filter(Q(normalized_name__icontains=q) | Q(name__icontains=q))

    out: List[Dict[str, Any]] = []
    for p in qs[:limit]:
        out.append(
            {
                'program_id': int(p.id),
                'program_code': (getattr(p, 'code', '') or '').strip(),
                'program_name': (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip(),
                'institution_name': ((p.institution.name if getattr(p, 'institution_id', None) else '') or '').strip(),
                'field_name': ((p.field.name if getattr(p, 'field_id', None) else '') or '').strip(),
                'level': (getattr(p, 'level', '') or '').strip(),
            }
        )

    return {'count': len(out), 'results': out}


def explain_recommendation(*, ctx: UserContext, program_id: int, goal_text: str = '') -> Dict[str, Any]:
    try:
        from catalog.models import Program, YearlyCutoff  # type: ignore
    except Exception:
        Program = None  # type: ignore
        YearlyCutoff = None  # type: ignore

    if Program is None:
        return {'detail': 'catalog_unavailable'}

    try:
        p = Program.objects.select_related('institution', 'field').filter(id=int(program_id)).first()
    except Exception:
        p = None

    if not p:
        return {'detail': 'not_found'}

    nm = (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip()
    field_name = (p.field.name if getattr(p, 'field_id', None) else '') or ''

    elig = eligibility_for_program(p, ctx.grades or {})
    cp = elig.get('cluster_points', None) if isinstance(elig, dict) else None
    cutoff = None
    if YearlyCutoff is not None:
        try:
            yc = YearlyCutoff.objects.filter(program=p).order_by('-year').first()
            if yc:
                cutoff = float(yc.cutoff)
        except Exception:
            cutoff = None
    br = score_program_breakdown(
        nm,
        field_name,
        ctx.traits or {},
        goal_text=str(goal_text or '').strip(),
        cluster_points=cp,
        cutoff=cutoff,
    )

    return {
        'program_id': int(p.id),
        'program_name': nm,
        'institution_name': (p.institution.name or '').strip() if getattr(p, 'institution_id', None) else '',
        'field_name': field_name,
        'eligibility': elig,
        'score_breakdown': br,
    }


def get_program_details(*, program_id: int) -> Dict[str, Any]:
    try:
        from catalog.models import Program  # type: ignore
    except Exception:
        Program = None  # type: ignore

    if Program is None:
        return {'detail': 'catalog_unavailable'}

    try:
        p = Program.objects.filter(id=int(program_id)).first()
    except Exception:
        p = None

    if not p:
        return {'detail': 'not_found'}

    subject_requirements = getattr(p, 'subject_requirements', None)
    if not isinstance(subject_requirements, dict):
        subject_requirements = {}

    return {
        'program_id': int(p.id),
        'program_code': (getattr(p, 'code', '') or '').strip(),
        'program_name': (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip(),
        'institution_name': ((p.institution.name if getattr(p, 'institution_id', None) else '') or '').strip(),
        'field_name': ((p.field.name if getattr(p, 'field_id', None) else '') or '').strip(),
        'level': (getattr(p, 'level', '') or '').strip(),
        'subject_requirements': subject_requirements,
    }


def program_details(*, program_id: int) -> Dict[str, Any]:
    return get_program_details(program_id=int(program_id))


def check_eligibility(*, ctx: UserContext, program_ids: List[int]) -> Dict[str, Any]:
    try:
        from catalog.models import Program  # type: ignore
    except Exception:
        Program = None  # type: ignore

    if Program is None:
        return {'detail': 'catalog_unavailable'}

    ids: List[int] = []
    for pid in program_ids or []:
        try:
            ids.append(int(pid))
        except Exception:
            continue
    if not ids:
        return {'count': 0, 'results': []}

    by_id: Dict[int, Any] = {}
    try:
        for p in Program.objects.filter(id__in=ids).prefetch_related('requirement_groups', 'requirement_groups__options', 'requirement_groups__options__subject'):
            by_id[int(p.id)] = p
    except Exception:
        by_id = {}

    out: List[Dict[str, Any]] = []
    for pid in ids:
        prog = by_id.get(int(pid))
        if not prog:
            continue
        elig = eligibility_for_program(prog, ctx.grades or {})
        out.append(
            {
                'program_id': int(pid),
                'eligible': elig.get('eligible'),
                'missing': elig.get('missing') or [],
                'cluster_points': elig.get('cluster_points'),
                'cutoff_ok': elig.get('cutoff_ok'),
            }
        )

    return {'count': len(out), 'results': out}
