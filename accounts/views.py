from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework import status
from django.db import transaction
from django.db.utils import OperationalError, ProgrammingError
from django.conf import settings
import math
import logging
import json
from .models import UserProfile, OnboardingProfile
from .serializers import UserProfileSerializer, OnboardingProfileSerializer
from .auth import get_bearer_token, firebase_init_error, verify_firebase_id_token

from utils.drf_auth import FirebaseAuthentication, IsFirebaseAuthenticated
from utils.errors import error_response

try:
    from scripts.etl.kuccps.grades import normalize_grade as _kcse_normalize_grade  # type: ignore
    from scripts.etl.kuccps.grades import grade_points as _kcse_grade_points  # type: ignore
except Exception:
    _kcse_normalize_grade = None
    _kcse_grade_points = None


_logger = logging.getLogger(__name__)


def _firebase_unavailable_response() -> Response:
    detail = 'Firebase admin not initialized'
    ferr = firebase_init_error()
    if ferr:
        detail = f"{detail}: {ferr}"
    return error_response(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='firebase_unavailable')


def _require_claims(request):
    claims = getattr(request, 'auth', None)
    if isinstance(claims, dict) and claims.get('uid'):
        return claims, None

    user = getattr(request, 'user', None)
    user_claims = getattr(user, 'claims', None) if user else None
    if isinstance(user_claims, dict) and user_claims.get('uid'):
        return user_claims, None

    token = get_bearer_token(request)
    claims, err, http_status = verify_firebase_id_token(token)
    if err:
        if int(http_status or 0) == 503:
            return None, _firebase_unavailable_response()
        return None, error_response(str(err), status_code=status.HTTP_401_UNAUTHORIZED, code='unauthorized')
    return claims, None


def _upsert_profile_from_claims(claims):
    uid = claims.get('uid')
    email = claims.get('email')
    name = claims.get('name') or claims.get('displayName')
    picture = claims.get('picture')
    if not uid:
        return None
    with transaction.atomic():
        profile, _ = UserProfile.objects.select_for_update().get_or_create(uid=uid)
        # Update fields if changed
        changed = False
        for field, value in (('email', email), ('display_name', name), ('photo_url', picture)):
            if value and getattr(profile, field) != value:
                setattr(profile, field, value)
                changed = True
        if changed:
            profile.save()
    return profile


@api_view(['POST'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def register(request):
    """Create a user profile if it doesn't exist; otherwise return existing."""
    claims, err = _require_claims(request)
    if err:
        return err

    try:
        profile = _upsert_profile_from_claims(claims)
    except (ProgrammingError, OperationalError) as e:
        _logger.error("Database not initialized (missing migrations?): %s: %s", e.__class__.__name__, str(e))
        detail = "Database not initialized"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='db_unavailable')
    except Exception as e:
        _logger.exception("Unexpected error during register profile upsert: %s: %s", e.__class__.__name__, str(e))
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')

    if not profile:
        return error_response("Invalid token", status_code=status.HTTP_401_UNAUTHORIZED, code='invalid_token')
    return Response(UserProfileSerializer(profile).data)

# ------------------------------
# Onboarding APIs
# ------------------------------

def _require_user(request):
    claims, err = _require_claims(request)
    if err:
        return None, err

    try:
        profile = _upsert_profile_from_claims(claims)
    except (ProgrammingError, OperationalError) as e:
        _logger.error("Database not initialized (missing migrations?): %s: %s", e.__class__.__name__, str(e))
        detail = "Database not initialized"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return None, error_response(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='db_unavailable')
    except Exception as e:
        _logger.exception("Unexpected error during request user upsert: %s: %s", e.__class__.__name__, str(e))
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return None, error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')

    if not profile:
        return None, error_response("Invalid token", status_code=status.HTTP_401_UNAUTHORIZED, code='invalid_token')
    return profile, None


def _compute_kcse_cluster_summary(ob: OnboardingProfile) -> dict:
    hs = ob.high_school or {}
    grades = hs.get('subject_grades') or {}
    if not isinstance(grades, dict) or not grades:
        return {
            'has_grades': False,
            'cluster_score': None,
            'subjects_provided': 0,
            'top4_points': 0,
            'top7_points': 0,
            'subjects': [],
            'top4_subjects': [],
            'top7_subjects': [],
            'formula': None,
        }

    cand_pts = {}
    norm_grades = {}
    for code, raw in grades.items():
        k = str(code or '').strip().upper().replace(' ', '')
        v = str(raw or '').strip().upper().replace(' ', '')
        if not k or not v:
            continue
        if _kcse_normalize_grade:
            v = _kcse_normalize_grade(v) or ''
        if not v:
            continue
        norm_grades[k] = v
        pts = 0
        if _kcse_grade_points:
            try:
                pts = int(_kcse_grade_points(v) or 0)
            except Exception:
                pts = 0
        else:
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
            pts = int(mapping.get(v, 0) or 0)
        if pts > 0:
            cand_pts[k] = pts

    if not cand_pts:
        return {
            'has_grades': False,
            'cluster_score': None,
            'subjects_provided': 0,
            'top4_points': 0,
            'top7_points': 0,
            'subjects': [],
            'top4_subjects': [],
            'top7_subjects': [],
            'formula': None,
        }

    sorted_pairs = sorted(cand_pts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    top4_pairs = sorted_pairs[:4]
    top7_pairs = sorted_pairs[:7]
    top4 = [int(v) for _k, v in top4_pairs]
    top7 = [int(v) for _k, v in top7_pairs]

    r_sum = sum(top4)
    t_sum = sum(top7)
    R = 48
    top_n = max(1, len(top7))
    T = 12 * top_n
    try:
        cluster = math.sqrt((r_sum / R) * (t_sum / T)) * 48
    except Exception:
        cluster = 0.0

    subjects = []
    for k, pts in sorted_pairs:
        subjects.append({
            'subject_code': k,
            'grade': (norm_grades.get(k) or ''),
            'points': int(pts),
        })

    return {
        'has_grades': True,
        'cluster_score': round(float(cluster), 3),
        'subjects_provided': len(cand_pts),
        'top4_points': int(r_sum),
        'top7_points': int(t_sum),
        'subjects': subjects,
        'top4_subjects': [k for k, _v in top4_pairs],
        'top7_subjects': [k for k, _v in top7_pairs],
        'formula': {
            'r_sum': int(r_sum),
            't_sum': int(t_sum),
            'R': int(R),
            'T': int(T),
            'top_n': int(top_n),
        },
    }


@api_view(['GET'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def onboarding_me(request):
    """Return current user's onboarding profile (creates one if missing)."""
    user, err = _require_user(request)
    if err:
        return err
    ob, _ = OnboardingProfile.objects.get_or_create(user=user)
    return Response(OnboardingProfileSerializer(ob).data)


@api_view(['POST'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def onboarding_save(request):
    """Upsert onboarding data for the current user.

    Body accepts a partial payload matching OnboardingProfileSerializer fields.
    If 'riasec_answers' present, compute 'riasec_scores' and 'riasec_top'.
    Sets status to 'complete' when universal + (high_school or college) + riasec_scores exist.
    """
    user, err = _require_user(request)
    if err:
        return err
    try:
        data = request.data if hasattr(request, 'data') else json.loads(request.body.decode('utf-8') or '{}')
    except Exception:
        data = {}
    with transaction.atomic():
        ob, _ = OnboardingProfile.objects.select_for_update().get_or_create(user=user)
        # Assign fields if provided
        for fld in [
            'education_level','universal','high_school','college','riasec_answers',
            'strengths','skills','work_style','lifestyle','preferences','version','status']:
            if fld in data:
                setattr(ob, fld, data.get(fld))
        # Compute RIASEC if answers provided
        if 'riasec_answers' in data:
            ob.compute_riasec()
        # Auto-complete status
        if (ob.universal or {}) and (ob.high_school or ob.college) and (ob.riasec_scores or {}):
            ob.status = ob.status or 'complete'
        ob.save()
    return Response(OnboardingProfileSerializer(ob).data)


@api_view(['GET'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def onboarding_dashboard(request):
    """Compute and return a dashboard summary for the current user.

    Includes RIASEC scores/top, basic profile context, and simple narratives.
    """
    user, err = _require_user(request)
    if err:
        return err
    ob, _ = OnboardingProfile.objects.get_or_create(user=user)
    # Derive simple narrative
    top = list(ob.riasec_top or [])
    narrative = ""
    if top:
        mapping = {
            'Realistic': 'Doer (hands-on, practical)',
            'Investigative': 'Thinker (analytical, research)',
            'Artistic': 'Creator (creative, design)',
            'Social': 'Helper (people-oriented)',
            'Enterprising': 'Persuader (leadership, business)',
            'Conventional': 'Organizer (structured, detail)',
        }
        labels = [mapping.get(t, t) for t in top]
        narrative = ", ".join(labels)
    kcse = _compute_kcse_cluster_summary(ob)
    return Response({
        'profile': OnboardingProfileSerializer(ob).data,
        'riasec': {
            'scores': ob.riasec_scores or {},
            'top': top,
            'narrative': narrative,
        },
        'kcse': kcse,
    })


@api_view(['POST'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def login(request):
    """Verify token and upsert profile (Firebase handles auth)."""
    claims, err = _require_claims(request)
    if err:
        return err

    try:
        profile = _upsert_profile_from_claims(claims)
    except (ProgrammingError, OperationalError) as e:
        _logger.error("Database not initialized (missing migrations?): %s: %s", e.__class__.__name__, str(e))
        detail = "Database not initialized"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='db_unavailable')
    except Exception as e:
        _logger.exception("Unexpected error during login profile upsert: %s: %s", e.__class__.__name__, str(e))
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')

    if not profile:
        return error_response("Invalid token", status_code=status.HTTP_401_UNAUTHORIZED, code='invalid_token')
    return Response(UserProfileSerializer(profile).data)


@api_view(['GET'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def me(request):
    claims, err = _require_claims(request)
    if err:
        return err

    try:
        profile = _upsert_profile_from_claims(claims)
    except (ProgrammingError, OperationalError) as e:
        _logger.error("Database not initialized (missing migrations?): %s: %s", e.__class__.__name__, str(e))
        detail = "Database not initialized"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='db_unavailable')
    except Exception as e:
        _logger.exception("Unexpected error during me profile upsert: %s: %s", e.__class__.__name__, str(e))
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')

    if not profile:
        return error_response("Invalid token", status_code=status.HTTP_401_UNAUTHORIZED, code='invalid_token')
    return Response(UserProfileSerializer(profile).data)
