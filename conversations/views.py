import json
from typing import Any, Dict, Optional

from django.utils import timezone
from django.conf import settings

from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response

from utils.drf_auth import FirebaseAuthentication, IsFirebaseAuthenticated
from utils.errors import error_response
from .models import Session, Message, Profile
from .fsm import next_turn
from .orchestrator import planner_turn
from .recommendations_service import build_recommendations

try:
    from accounts.models import UserProfile as _UserProfile, OnboardingProfile as _OnboardingProfile  # type: ignore
except Exception:
    _UserProfile = None
    _OnboardingProfile = None

try:
    from scripts.etl.kuccps.eligibility import evaluate_eligibility as _kuccps_evaluate_eligibility  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_CODE_ALIASES as _SUBJECT_CODE_ALIASES  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_CANON_TO_NUM as _SUBJECT_CANON_TO_NUM  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_TOKEN_ALIASES as _SUBJECT_TOKEN_ALIASES  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_TOKEN_CANON_TO_ALIASES as _SUBJECT_TOKEN_CANON_TO_ALIASES  # type: ignore
except Exception:
    _kuccps_evaluate_eligibility = None
    _SUBJECT_CODE_ALIASES = {}
    _SUBJECT_CANON_TO_NUM = {}
    _SUBJECT_TOKEN_ALIASES = {}
    _SUBJECT_TOKEN_CANON_TO_ALIASES = {}


def _now():
    return timezone.now()


def _ensure_session_ttl(sess: Session) -> None:
    sess.ensure_ttl()
    sess.save(update_fields=['expires_at'])


def _require_uid(request):
    uid = getattr(getattr(request, 'user', None), 'uid', None)
    if not uid:
        return None, error_response('Missing bearer token', status_code=status.HTTP_401_UNAUTHORIZED, code='missing_token')
    return str(uid), None


def _serialize_message(m: Message) -> Dict[str, Any]:
    return {
        'id': m.id,
        'role': m.role,
        'content': m.content,
        'fsm_state': m.fsm_state,
        'confidence': m.confidence,
        'created_at': m.created_at.isoformat(),
        'nlp': m.nlp or {},
    }


def _serialize_session(s: Session, limit_messages: int = 20) -> Dict[str, Any]:
    msgs = list(s.messages.order_by('-created_at')[:limit_messages])
    msgs = list(reversed(msgs))
    return {
        'id': str(s.id),
        'status': s.status,
        'fsm_state': s.fsm_state,
        'slots': s.slots,
        'expires_at': s.expires_at.isoformat() if s.expires_at else None,
        'messages': [_serialize_message(m) for m in msgs],
    }


def _serialize_session_list_item(s: Session) -> Dict[str, Any]:
    return {
        'id': str(s.id),
        'status': s.status,
        'fsm_state': s.fsm_state,
        'expires_at': s.expires_at.isoformat() if s.expires_at else None,
        'created_at': s.created_at.isoformat() if s.created_at else None,
        'last_activity_at': s.last_activity_at.isoformat() if s.last_activity_at else None,
    }


@api_view(['GET'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def get_session(request, session_id):
    """GET /api/conversations/sessions/{id}
    Returns session summary and recent messages. Creates a new session on first access.
    """
    uid, err = _require_uid(request)
    if err:
        return err
    try:
        try:
            s = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            s = Session(id=session_id)
            s.owner_uid = uid
            s.set_external_user_id(uid)
            s.ensure_ttl()
            s.save()
        if s.owner_uid and s.owner_uid != uid:
            return error_response('Forbidden', status_code=status.HTTP_403_FORBIDDEN, code='forbidden')
        if not s.owner_uid:
            s.owner_uid = uid
            s.set_external_user_id(uid)
            s.save(update_fields=['owner_uid', 'external_user_id_encrypted'])
        _ensure_session_ttl(s)
        return Response(_serialize_session(s))
    except Exception as e:
        detail = 'Server error'
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')


@api_view(['GET', 'POST'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def sessions_collection(request):
    """GET/POST /api/conversations/sessions

    - GET returns a list of the current user's sessions (most recent first).
    - POST creates a new session owned by the current user.
    """
    uid, err = _require_uid(request)
    if err:
        return err
    if request.method == 'GET':
        try:
            limit = int(request.GET.get('limit', '20') or '20')
        except Exception:
            limit = 20
        limit = max(1, min(50, limit))
        qs = Session.objects.filter(owner_uid=uid).order_by('-last_activity_at')
        items = [_serialize_session_list_item(s) for s in qs[:limit]]
        return Response({'count': qs.count(), 'results': items})

    if request.method == 'POST':
        try:
            s = Session(owner_uid=uid)
            s.set_external_user_id(uid)
            s.ensure_ttl()
            s.save()
            _ensure_session_ttl(s)
            return Response({'session': _serialize_session(s)}, status=status.HTTP_201_CREATED)
        except Exception as e:
            detail = 'Server error'
            if settings.DEBUG:
                detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
            return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')
    return error_response('Method not allowed', status_code=status.HTTP_405_METHOD_NOT_ALLOWED, code='method_not_allowed')


def _grade_points(g: str) -> int:
    s = (g or '').strip().upper().replace(' ', '')
    mapping = {
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
    return int(mapping.get(s, 0))


def _meets_min_grade(user_grade: str, min_grade: str) -> bool:
    if not (min_grade or '').strip():
        return True
    return _grade_points(user_grade) >= _grade_points(min_grade)


def _has_meaningful_grades(grades: Dict[str, str]) -> bool:
    if not isinstance(grades, dict) or not grades:
        return False
    for _k, v in grades.items():
        if str(v or '').strip():
            return True
    return False


def _has_meaningful_traits(traits: Dict[str, Any]) -> bool:
    if not isinstance(traits, dict) or not traits:
        return False
    for _k, v in traits.items():
        try:
            if float(v or 0.0) > 0.001:
                return True
        except Exception:
            continue
    return False


def _normalize_riasec_scores_to_traits(scores: Dict[str, Any]) -> Dict[str, float]:
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


def _norm_subject_code(s: str) -> str:
    return (s or '').strip().upper().replace(' ', '')


def _expand_grades_with_subject_aliases(grades: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (grades or {}).items():
        kk = _norm_subject_code(str(k))
        vv = str(v or '').strip().upper().replace(' ', '')
        if not kk or not vv:
            continue
        out[kk] = vv
        canon = _norm_subject_code(_SUBJECT_TOKEN_ALIASES.get(kk) or kk)
        if canon:
            out[canon] = vv
            for a in (_SUBJECT_TOKEN_CANON_TO_ALIASES.get(canon) or []):
                aa = _norm_subject_code(a)
                if aa:
                    out[aa] = vv
        if kk in _SUBJECT_CODE_ALIASES:
            out[_norm_subject_code(_SUBJECT_CODE_ALIASES.get(kk) or '')] = vv
        if kk in _SUBJECT_CANON_TO_NUM:
            out[_norm_subject_code(_SUBJECT_CANON_TO_NUM.get(kk) or '')] = vv
        if canon in _SUBJECT_CANON_TO_NUM:
            out[_norm_subject_code(_SUBJECT_CANON_TO_NUM.get(canon) or '')] = vv
    return out


def _format_req_label(subj_token: str, numeric_code: str, min_g: str) -> str:
    subj = _norm_subject_code(subj_token)
    code = _norm_subject_code(numeric_code)
    canon = ''
    if subj and subj in _SUBJECT_CANON_TO_NUM:
        canon = subj
    elif subj and subj in _SUBJECT_TOKEN_ALIASES:
        canon = _norm_subject_code(_SUBJECT_TOKEN_ALIASES.get(subj) or '')
    elif code and code in _SUBJECT_CODE_ALIASES:
        canon = _norm_subject_code(_SUBJECT_CODE_ALIASES.get(code) or '')
    label = canon or subj or code
    if min_g:
        return f"{label} >= {min_g}"
    return label


def _seed_profile_from_onboarding_uid(uid: str, prof: Profile) -> None:
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

    tops = ob.riasec_top or []
    traits_from_ob = {}
    if isinstance(tops, list) and tops:
        t0 = str(tops[0] or '').strip()
        t1 = str(tops[1] or '').strip() if len(tops) > 1 else ''
        t2 = str(tops[2] or '').strip() if len(tops) > 2 else ''
        if t0:
            traits_from_ob[t0] = 1.0
        if t1:
            traits_from_ob[t1] = 0.7
        if t2:
            traits_from_ob[t2] = 0.5
    if not traits_from_ob:
        traits_from_ob = _normalize_riasec_scores_to_traits(ob.riasec_scores or {})

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

    hs = ob.high_school or {}
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

    prefs = prof.preferences or {}
    if isinstance(prefs, dict):
        cur_goal = prefs.get('career_goals')
        has_goal = False
        if isinstance(cur_goal, list):
            has_goal = any(str(x).strip() for x in cur_goal)
        elif isinstance(cur_goal, str):
            has_goal = bool(str(cur_goal).strip())
        if not has_goal:
            uni = ob.universal or {}
            if isinstance(uni, dict):
                raw = uni.get('careerGoals')
                if raw is None:
                    raw = uni.get('career_goals')
                goals = []
                if isinstance(raw, list):
                    goals = [str(x).strip() for x in raw if str(x).strip()]
                elif isinstance(raw, str):
                    s = str(raw).strip()
                    if s:
                        parts = [p.strip() for p in s.replace('\n', ',').split(',')]
                        goals = [p for p in parts if p]
                if goals:
                    prof.preferences = {**prefs, 'career_goals': goals}
                changed = True

    if changed:
        try:
            prof.save(update_fields=['traits', 'grades', 'preferences', 'updated_at'])
        except Exception:
            try:
                prof.save()
            except Exception:
                return


def _missing_from_subject_requirements(req: Dict[str, Any], grades: Dict[str, str]) -> list[str]:
    out: list[str] = []
    gmap = _expand_grades_with_subject_aliases(grades or {})
    if not isinstance(req, dict) or not req:
        return out

    required = req.get('required', []) or []
    groups = req.get('groups', []) or []

    for it in required:
        subj_raw = (it.get('subject') or it.get('subject_code') or '').strip()
        code_raw = (it.get('code') or '').strip()
        subj = _norm_subject_code(subj_raw or code_raw)
        if not subj and not code_raw:
            continue
        min_g = (it.get('min_grade') or '').strip().upper().replace(' ', '')
        user_g = gmap.get(subj, '')
        if (not user_g) or (min_g and not _meets_min_grade(user_g, min_g)):
            out.append(_format_req_label(subj_raw or subj, code_raw or subj, min_g))

    for grp in groups:
        try:
            pick = int(grp.get('pick') or 0)
        except Exception:
            pick = 0
        if pick <= 0:
            continue
        opts = grp.get('options', []) or []
        satisfied = 0
        missing: list[str] = []
        for opt in opts:
            subj_raw = (opt.get('subject') or opt.get('subject_code') or '').strip()
            code_raw = (opt.get('code') or '').strip()
            subj = _norm_subject_code(subj_raw or code_raw)
            if not subj and not code_raw:
                continue
            min_g = (opt.get('min_grade') or '').strip().upper().replace(' ', '')
            user_g = gmap.get(subj, '')
            if user_g and (not min_g or _meets_min_grade(user_g, min_g)):
                satisfied += 1
            else:
                if min_g:
                    missing.append(_format_req_label(subj_raw or subj, code_raw or subj, min_g))
                else:
                    missing.append(subj)
        if satisfied < pick:
            out.extend(missing[: max(1, pick)])
    return out


def _eligibility_from_subject_requirements(program, grades: Dict[str, str], cutoff: Optional[float] = None) -> Dict[str, Any]:
    req = getattr(program, 'subject_requirements', None) or {}
    if not isinstance(req, dict) or not req:
        return _eligibility_from_requirements(program, grades)
    if _kuccps_evaluate_eligibility is None:
        return _eligibility_from_requirements(program, grades)

    try:
        row = {
            'subject_requirements_json': json.dumps(req),
            'normalized_name': (getattr(program, 'normalized_name', '') or ''),
            'name': (getattr(program, 'name', '') or ''),
        }
        res = _kuccps_evaluate_eligibility(row, grades or {}) or {}
        cluster_points = res.get('cluster_points', None)
        missing = _missing_from_subject_requirements(req, grades)

        eligible = True
        if isinstance(res.get('eligible', None), bool):
            eligible = bool(res.get('eligible'))

        cutoff_ok = None
        if cutoff is not None and cluster_points is not None:
            try:
                cutoff_ok = float(cluster_points) >= float(cutoff)
            except Exception:
                cutoff_ok = None
            if cutoff_ok is False:
                eligible = False
                missing.append(f"cluster_points >= {float(cutoff):g}")

        return {
            'eligible': eligible,
            'missing': missing,
            'unmet_groups': None,
            'cluster_points': cluster_points,
            'cutoff_ok': cutoff_ok,
        }
    except Exception:
        return _eligibility_from_requirements(program, grades)


def _eligibility_from_requirements(program, grades: Dict[str, str]) -> Dict[str, Any]:
    """Best-effort eligibility evaluation based on ProgramRequirementGroup/Option.

    Returns:
      - eligible: bool|None
      - missing: list[str]
      - unmet_groups: int
    """
    try:
        groups = list(program.requirement_groups.all().order_by('order').prefetch_related('options', 'options__subject'))
    except Exception:
        groups = []
    if not groups:
        return {'eligible': None, 'missing': [], 'unmet_groups': 0}

    gmap = _expand_grades_with_subject_aliases(grades or {})
    missing: list[str] = []
    unmet_groups = 0
    for grp in groups:
        pick = int(getattr(grp, 'pick', 1) or 1)
        opts = list(grp.options.all().order_by('order'))
        if not opts:
            continue
        satisfied = 0
        # Track missing items to provide a hint (not exhaustive)
        grp_missing: list[str] = []
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
            user_grade = gmap.get(subj_code, '')
            min_grade = (getattr(opt, 'min_grade', '') or '').strip().upper().replace(' ', '')
            if user_grade and _meets_min_grade(user_grade, min_grade):
                satisfied += 1
            else:
                if min_grade:
                    grp_missing.append(f"{subj_code} >= {min_grade}")
                else:
                    grp_missing.append(subj_code)
        if satisfied < pick:
            unmet_groups += 1
            missing.extend(grp_missing[: max(1, pick)])

    return {
        'eligible': unmet_groups == 0,
        'missing': missing,
        'unmet_groups': unmet_groups,
    }


def _score_program(name: str, field_name: str, traits: Dict[str, float], goal_text: str = '') -> float:
    from .recommend import TRAIT_FIELD_HINTS

    text = f"{name} {field_name}".lower()
    score = 0.0
    for trait, weight in (traits or {}).items():
        try:
            w = float(weight or 0.0)
        except Exception:
            w = 0.0
        if w <= 0:
            continue
        hints = TRAIT_FIELD_HINTS.get(str(trait), [])
        hits = sum(1 for h in hints if str(h).lower() in text)
        if hits > 0:
            score += w

    gt = (goal_text or '').strip().lower()
    if gt:
        toks = [t for t in ''.join((ch if ch.isalnum() else ' ') for ch in gt).split() if len(t) >= 4]
        if toks:
            stop = {'become', 'becoming', 'want', 'wants', 'would', 'like', 'study', 'studying', 'career', 'goal', 'goals', 'work'}
            toks2 = [t for t in toks if t not in stop]
            hits2 = sum(1 for t in set(toks2[:12]) if t in text)
            if hits2 > 0:
                score += min(1.0, 0.25 * float(hits2))
    return float(score)


@api_view(['GET'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def session_recommendations(request, session_id):
    """GET /api/conversations/sessions/{id}/recommendations

    Returns DB-backed structured recommendations for the session's collected Profile.
    """
    uid, err = _require_uid(request)
    if err:
        return err
    try:
        try:
            s = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            return error_response('Session not found', status_code=status.HTTP_404_NOT_FOUND, code='not_found')
        if s.owner_uid and s.owner_uid != uid:
            return error_response('Forbidden', status_code=status.HTTP_403_FORBIDDEN, code='forbidden')

        prof, _ = Profile.objects.get_or_create(session=s)

        _seed_profile_from_onboarding_uid(uid, prof)

        grades = prof.grades or {}
        traits = prof.traits or {}
        goal_text = ''
        try:
            prefs = prof.preferences or {}
            if isinstance(prefs, dict):
                raw = prefs.get('career_goals')
                if isinstance(raw, list):
                    goals = [str(x).strip() for x in raw if str(x).strip()]
                    goal_text = ' '.join(goals).strip()
                else:
                    goal_text = str(raw or '').strip()
        except Exception:
            goal_text = ''

        try:
            k = int(request.GET.get('k', '10') or '10')
        except Exception:
            k = 10
        k = max(1, min(20, k))

        recs, stretch_items = build_recommendations(
            grades,
            traits,
            goal_text=goal_text,
            k=k,
            level='bachelor',
            raise_on_missing_catalog=True,
        )

        return Response({
            'session_id': str(s.id),
            'profile': {
                'grades': grades,
                'traits': traits,
            },
            'count': len(recs),
            'recommendations': recs,
            'stretch_count': len(stretch_items),
            'stretch_recommendations': stretch_items,
        })
    except Exception as e:
        detail = 'Server error'
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')


@api_view(['DELETE', 'POST'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def delete_session(request, session_id):
    """DELETE/POST /api/conversations/sessions/{id}/delete
    Deletes a conversation session and its messages. Returns 204 on success.
    """
    uid, err = _require_uid(request)
    if err:
        return err
    try:
        try:
            s = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            return Response(status=status.HTTP_204_NO_CONTENT)
        if s.owner_uid and s.owner_uid != uid:
            return error_response('Forbidden', status_code=status.HTTP_403_FORBIDDEN, code='forbidden')
        s.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        detail = 'Server error'
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')


@api_view(['POST'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def post_message(request, session_id):
    """POST /api/conversations/sessions/{id}/messages
    Body: { text: str, idempotency_key?: str, user_id?: str }
    Creates the session if it does not exist yet (with TTL), enforces idempotency, and appends the user message.
    Returns a stub assistant reply and current FSM state.
    """
    uid, err = _require_uid(request)
    if err:
        return err
    try:
        data = json.loads(request.body.decode('utf-8') or '{}')
        text = (data.get('text') or '').strip()
        idem = (data.get('idempotency_key') or '').strip()
        nlp_provider = (data.get('nlp_provider') or '').strip().lower()
        use_planner = bool(data.get('use_planner')) if 'use_planner' in data else bool(getattr(settings, 'CHAT_PLANNER_ENABLED', False))
        shadow_mode = bool(getattr(settings, 'CHAT_PLANNER_SHADOW_MODE', False))
        if not text:
            return error_response('text is required', status_code=status.HTTP_400_BAD_REQUEST, code='validation_error', fields={'text': ['This field is required.']})
        try:
            s = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            s = Session(id=session_id)
        if s.owner_uid and s.owner_uid != uid:
            return error_response('Forbidden', status_code=status.HTTP_403_FORBIDDEN, code='forbidden')
        if not s.owner_uid:
            s.owner_uid = uid
            s.set_external_user_id(uid)
        s.ensure_ttl()
        s.last_activity_at = _now()
        s.save()

        # Idempotency: return prior message result if same key exists (user role only)
        if idem:
            try:
                prior = Message.objects.get(session=s, idempotency_key=idem)
                # Return the current session snapshot
                return Response({
                    'session': _serialize_session(s),
                    'duplicate': True,
                })
            except Message.DoesNotExist:
                pass

        # Store user message (encrypted)
        um = Message(session=s, role='user', idempotency_key=idem)
        um.content = text
        um.fsm_state = s.fsm_state
        # Compute next turn using FSM+NLP
        tr = None
        planner_tr = None
        if use_planner:
            planner_tr = planner_turn(s, text, uid=uid, provider_override=nlp_provider)
            if not shadow_mode:
                tr = planner_tr

        if tr is None:
            tr = next_turn(s, text, provider_override=nlp_provider)

        if planner_tr is not None and shadow_mode:
            nlp_shadow = planner_tr.nlp_payload or {}
            nlp_main = tr.nlp_payload or {}
            tr.nlp_payload = {**nlp_main, 'shadow_planner': nlp_shadow}
        # Persist NLP payload on the user message for observability
        um.nlp = tr.nlp_payload or {}
        um.save()

        # Update session state and slots
        s.fsm_state = tr.next_state
        s.slots = tr.slots
        s.last_activity_at = _now()
        s.save(update_fields=['fsm_state', 'slots', 'last_activity_at'])

        # Assistant reply
        am = Message(session=s, role='assistant', fsm_state=tr.next_state)
        am.content = tr.reply
        am.confidence = tr.confidence
        am.nlp = tr.nlp_payload or {}
        am.save()

        s.last_activity_at = _now()
        s.save(update_fields=['last_activity_at'])

        turn_recs = None
        try:
            nlp_payload = tr.nlp_payload or {}
            if isinstance(nlp_payload, dict) and isinstance(nlp_payload.get('turn_recommendations'), dict):
                turn_recs = nlp_payload.get('turn_recommendations')
        except Exception:
            turn_recs = None

        return Response({'session': _serialize_session(s), 'turn_recommendations': turn_recs})
    except Exception as e:
        detail = 'Server error'
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')


@api_view(['POST'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def post_profile(request):
    """POST /api/conversations/profile
    Body: { session_id: UUID, traits?: {}, grades?: {}, preferences?: {}, version?: str }
    Upserts the profile for the session.
    """
    uid, err = _require_uid(request)
    if err:
        return err
    try:
        data = json.loads(request.body.decode('utf-8') or '{}')
        session_id = data.get('session_id')
        if not session_id:
            return error_response('session_id is required', status_code=status.HTTP_400_BAD_REQUEST, code='validation_error', fields={'session_id': ['This field is required.']})
        try:
            s = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            s = Session(id=session_id)
            s.owner_uid = uid
            s.set_external_user_id(uid)
            s.ensure_ttl()
            s.save()

        if s.owner_uid and s.owner_uid != uid:
            return error_response('Forbidden', status_code=status.HTTP_403_FORBIDDEN, code='forbidden')
        if not s.owner_uid:
            s.owner_uid = uid
            s.set_external_user_id(uid)
            s.save(update_fields=['owner_uid', 'external_user_id_encrypted'])

        prof, _ = Profile.objects.get_or_create(session=s)
        if 'traits' in data and isinstance(data['traits'], dict):
            prof.traits = data['traits']
        if 'grades' in data and isinstance(data['grades'], dict):
            prof.grades = data['grades']
        if 'preferences' in data and isinstance(data['preferences'], dict):
            prof.preferences = data['preferences']
        if 'version' in data and isinstance(data['version'], str):
            prof.version = data['version']
        prof.save()
        return Response({
            'session_id': str(s.id),
            'profile': {
                'traits': prof.traits,
                'grades': prof.grades,
                'preferences': prof.preferences,
                'version': prof.version,
            }
        })
    except Exception as e:
        detail = 'Server error'
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')
