from __future__ import annotations

from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response

from utils.errors import error_response

from .mapping_models import OnetFieldOccupationMapping
from .models import OnetOccupation


def _level_from_job_zone(job_zone: int | None) -> str | None:
    if job_zone is None:
        return None
    # O*NET Job Zones range 1-5. We map them to a simple progression.
    if job_zone <= 2:
        return 'entry'
    if job_zone == 3:
        return 'mid'
    return 'senior'


def _job_zone_by_code(codes: list[str]) -> dict[str, int]:
    try:
        from .models import OnetJobZone  # type: ignore
    except Exception:
        OnetJobZone = None  # type: ignore

    if not codes or OnetJobZone is None:
        return {}

    try:
        # If there are multiple rows per occupation (domain_source/date), take max(job_zone)
        # which corresponds to higher preparation.
        rows = (
            OnetJobZone.objects
            .filter(onetsoc_code__in=codes)
            .values_list('onetsoc_code', 'job_zone')
        )
        out: dict[str, int] = {}
        for code, jz in rows:
            try:
                code_s = str(code)
                jz_i = int(getattr(jz, 'job_zone', jz))
            except Exception:
                continue
            prev = out.get(code_s)
            if prev is None or jz_i > prev:
                out[code_s] = jz_i
        return out
    except Exception:
        return {}


def _as_bool(v: object, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if not s:
        return bool(default)
    return s in {'1', 'true', 'yes', 'on'}


def _as_int(v: object, default: int) -> int:
    try:
        s = str(v).strip()
        if not s:
            return int(default)
        return int(s)
    except Exception:
        return int(default)


def _apply_level_quotas(path: dict[str, list[dict]], per_level: int) -> tuple[dict[str, list[dict]], list[dict]]:
    n = max(0, int(per_level))
    if n <= 0:
        flat = []
        for k in ['entry', 'mid', 'senior', 'unknown']:
            flat.extend(list(path.get(k) or []))
        return path, flat

    trimmed = {
        'entry': list((path.get('entry') or [])[:n]),
        'mid': list((path.get('mid') or [])[:n]),
        'senior': list((path.get('senior') or [])[:n]),
        'unknown': list((path.get('unknown') or [])[:n]),
    }
    flat = []
    for k in ['entry', 'mid', 'senior', 'unknown']:
        flat.extend(trimmed[k])
    return trimmed, flat


def _occupation_payload(occ: OnetOccupation) -> dict:
    return {
        'onetsoc_code': occ.onetsoc_code,
        'title': occ.title,
        'description': occ.description,
    }


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_field_careers(request, field_slug: str):
    return _api_field_careers_impl(request, field_slug, force_career_path=None, force_path_only=None)


def _api_field_careers_impl(request, field_slug: str, force_career_path: bool | None, force_path_only: bool | None):
    slug = (field_slug or '').strip()
    if not slug:
        return error_response('field_slug is required', status_code=status.HTTP_400_BAD_REQUEST, code='validation_error')

    mappings = (
        OnetFieldOccupationMapping.objects.select_related('field')
        .filter(field__slug__iexact=slug)
        .order_by('-weight', 'occupation_code')
    )

    limit_raw = (request.query_params.get('limit') or '').strip()
    if limit_raw:
        try:
            limit = max(1, int(limit_raw))
            mappings = mappings[:limit]
        except Exception:
            pass

    codes = [m.occupation_code for m in mappings]
    occs = {o.onetsoc_code: o for o in OnetOccupation.objects.filter(onetsoc_code__in=codes)}
    job_zone_by_code = _job_zone_by_code(codes)

    # Automated behavior: career_path grouping enabled by default.
    # Opt-out with career_path=0.
    career_path = force_career_path if force_career_path is not None else _as_bool(request.query_params.get('career_path'), default=True)
    path_only = force_path_only if force_path_only is not None else _as_bool(request.query_params.get('path_only'), default=False)
    per_level = _as_int(request.query_params.get('per_level'), 3)

    results: list[dict] = []
    for m in mappings:
        occ = occs.get(m.occupation_code)
        if not occ:
            continue
        jz = job_zone_by_code.get(occ.onetsoc_code)
        results.append({
            **_occupation_payload(occ),
            'weight': float(m.weight) if m.weight is not None else None,
            'notes': m.notes,
            'job_zone': jz,
            'level': _level_from_job_zone(jz),
        })

    if career_path:
        path = {'entry': [], 'mid': [], 'senior': [], 'unknown': []}
        for r in results:
            lvl = (r.get('level') or 'unknown').strip().lower()
            if lvl not in path:
                lvl = 'unknown'
            path[lvl].append(r)
        trimmed_path, selected = _apply_level_quotas(path, per_level=per_level)
        if path_only:
            return Response({'field_slug': slug, 'count': len(selected), 'career_path': trimmed_path})
        return Response({'field_slug': slug, 'count': len(selected), 'results': selected, 'career_path': trimmed_path})

    return Response({'field_slug': slug, 'count': len(results), 'results': results})


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_field_career_path(request, field_slug: str):
    return _api_field_careers_impl(request, field_slug, force_career_path=True, force_path_only=True)
