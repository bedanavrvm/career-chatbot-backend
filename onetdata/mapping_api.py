from __future__ import annotations

from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response

from utils.errors import error_response

from .mapping_models import OnetFieldOccupationMapping
from .models import OnetOccupation


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
    slug = (field_slug or '').strip()
    if not slug:
        return error_response('field_slug is required', status_code=status.HTTP_400_BAD_REQUEST, code='validation_error')

    mappings = (
        OnetFieldOccupationMapping.objects.select_related('field')
        .filter(field__slug__iexact=slug)
        .order_by('-weight', 'occupation_code')
    )

    codes = [m.occupation_code for m in mappings]
    occs = {o.onetsoc_code: o for o in OnetOccupation.objects.filter(onetsoc_code__in=codes)}

    results = []
    for m in mappings:
        occ = occs.get(m.occupation_code)
        if not occ:
            continue
        results.append({
            **_occupation_payload(occ),
            'weight': float(m.weight) if m.weight is not None else None,
            'notes': m.notes,
        })

    return Response({'field_slug': slug, 'count': len(results), 'results': results})
