from __future__ import annotations

from typing import Any

from django.db.models import Q
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response

from utils.errors import error_response

from .models import (
    OnetContentElement,
    OnetInterest,
    OnetOccupation,
    OnetRelatedOccupation,
    OnetSkill,
    OnetTaskStatement,
)


_RIASEC_ELEMENT_IDS = {
    'R': '1.B.1.a',
    'I': '1.B.1.b',
    'A': '1.B.1.c',
    'S': '1.B.1.d',
    'E': '1.B.1.e',
    'C': '1.B.1.f',
}


def _float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_onet_occupations(request):
    q = (request.GET.get('q') or '').strip()
    page = max(1, int(request.GET.get('page', 1) or 1))
    page_size = max(1, min(50, int(request.GET.get('page_size', 20) or 20)))

    qs = OnetOccupation.objects.all()
    if q:
        qs = qs.filter(Q(title__icontains=q) | Q(onetsoc_code__icontains=q))

    total = int(qs.count())
    if total <= 0:
        return Response({'count': 0, 'page': page, 'page_size': page_size, 'results': []})

    start = (page - 1) * page_size
    end = start + page_size
    rows = list(qs.order_by('title')[start:end].values('onetsoc_code', 'title', 'description'))
    return Response({'count': total, 'page': page, 'page_size': page_size, 'results': rows})


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_onet_occupation_detail(request, soc_code: str):
    soc = (soc_code or '').strip()
    if not soc:
        return error_response('soc_code is required', status_code=status.HTTP_400_BAD_REQUEST, code='validation_error')

    occ = OnetOccupation.objects.filter(onetsoc_code__iexact=soc).first()
    if not occ:
        return error_response('Occupation not found', status_code=status.HTTP_404_NOT_FOUND, code='not_found')

    # Tasks
    tasks = list(
        OnetTaskStatement.objects.filter(onetsoc_code=occ)
        .order_by('task_type', 'task_id')
        .values('task_id', 'task', 'task_type')[:50]
    )

    # Skills (top 20 by importance IM)
    skill_rows = (
        OnetSkill.objects.filter(onetsoc_code=occ, scale_id_id='IM')
        .select_related('element_id')
        .order_by('-data_value')
        .values('element_id_id', 'element_id__element_name', 'data_value')[:20]
    )
    skills = [
        {
            'element_id': r['element_id_id'],
            'name': r['element_id__element_name'],
            'importance': _float(r['data_value']),
        }
        for r in list(skill_rows)
    ]

    # Interests (RIASEC)
    interests_qs = (
        OnetInterest.objects.filter(onetsoc_code=occ, scale_id_id='OI', element_id_id__in=set(_RIASEC_ELEMENT_IDS.values()))
        .values('element_id_id', 'data_value')
    )
    scores = {k: 0.0 for k in _RIASEC_ELEMENT_IDS.keys()}
    for r in list(interests_qs):
        eid = r.get('element_id_id')
        for key, expected in _RIASEC_ELEMENT_IDS.items():
            if eid == expected:
                scores[key] = _float(r.get('data_value'))

    # Related occupations
    rel_qs = (
        OnetRelatedOccupation.objects.filter(onetsoc_code=occ)
        .select_related('related_onetsoc_code')
        .order_by('related_index')
        .values(
            'related_onetsoc_code__onetsoc_code',
            'related_onetsoc_code__title',
            'relatedness_tier',
            'related_index',
        )[:30]
    )
    related = [
        {
            'soc_code': r['related_onetsoc_code__onetsoc_code'],
            'title': r['related_onetsoc_code__title'],
            'tier': r['relatedness_tier'],
            'index': int(r['related_index'] or 0),
        }
        for r in list(rel_qs)
    ]

    return Response(
        {
            'onetsoc_code': occ.onetsoc_code,
            'title': occ.title,
            'description': occ.description,
            'riasec': scores,
            'tasks': tasks,
            'top_skills': skills,
            'related': related,
        }
    )


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_onet_recommendations(request):
    # Accept either R,I,A,S,E,C or realistic/investigative/... query params
    def g(*names: str) -> float:
        for n in names:
            if n in request.GET:
                return _float(request.GET.get(n))
        return 0.0

    scores_in = {
        'R': g('R', 'r', 'realistic'),
        'I': g('I', 'i', 'investigative'),
        'A': g('A', 'a', 'artistic'),
        'S': g('S', 's', 'social'),
        'E': g('E', 'e', 'enterprising'),
        'C': g('C', 'c', 'conventional'),
    }

    top_n = max(1, min(50, int(request.GET.get('top_n', 20) or 20)))

    # If no signal provided, return validation error
    if all(v == 0.0 for v in scores_in.values()):
        return error_response(
            'Provide at least one RIASEC score',
            status_code=status.HTTP_400_BAD_REQUEST,
            code='validation_error',
        )

    # Fetch OI interest values for RIASEC elements for all occupations.
    # We compute a simple dot product score.
    interests = (
        OnetInterest.objects.filter(scale_id_id='OI', element_id_id__in=set(_RIASEC_ELEMENT_IDS.values()))
        .values('onetsoc_code_id', 'element_id_id', 'data_value')
    )

    occ_scores: dict[str, float] = {}
    for row in interests.iterator(chunk_size=5000):
        soc = row.get('onetsoc_code_id')
        eid = row.get('element_id_id')
        if not soc or not eid:
            continue
        if soc not in occ_scores:
            occ_scores[soc] = 0.0

        for k, expected in _RIASEC_ELEMENT_IDS.items():
            if eid == expected:
                occ_scores[soc] += scores_in.get(k, 0.0) * _float(row.get('data_value'))
                break

    # Pick top N
    ranked = sorted(occ_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    socs = [s for s, _ in ranked]
    occ_map = {o.onetsoc_code: o for o in OnetOccupation.objects.filter(onetsoc_code__in=socs)}

    results = []
    for soc, score in ranked:
        occ = occ_map.get(soc)
        if not occ:
            continue
        results.append({'onetsoc_code': occ.onetsoc_code, 'title': occ.title, 'description': occ.description, 'score': round(float(score), 4)})

    return Response({'inputs': scores_in, 'count': len(results), 'results': results})
