from __future__ import annotations

from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response

from utils.errors import error_response


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_program_careers(request, program_id: str):
    try:
        from catalog.models import Program  # type: ignore
    except Exception:
        Program = None  # type: ignore

    if Program is None:
        return error_response('Catalog DB not available', status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='catalog_unavailable')

    pid = (program_id or '').strip()
    if not pid:
        return error_response('program_id is required', status_code=status.HTTP_400_BAD_REQUEST, code='validation_error')

    # Only support numeric program ID for now to keep it unambiguous.
    try:
        program_pk = int(pid)
    except Exception:
        return error_response('program_id must be a numeric id', status_code=status.HTTP_400_BAD_REQUEST, code='validation_error')

    program = Program.objects.select_related('field').filter(id=program_pk).first()
    if not program:
        return error_response('Program not found', status_code=status.HTTP_404_NOT_FOUND, code='not_found')

    field = getattr(program, 'field', None)
    if not field:
        return Response({'program_id': program_pk, 'field_slug': None, 'count': 0, 'results': []})

    # Delegate to mapping model
    from onetdata.mapping_models import OnetFieldOccupationMapping  # type: ignore
    from onetdata.models import OnetOccupation  # type: ignore

    mappings = (
        OnetFieldOccupationMapping.objects.filter(field=field)
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
            'onetsoc_code': occ.onetsoc_code,
            'title': occ.title,
            'description': occ.description,
            'weight': float(m.weight) if m.weight is not None else None,
            'notes': m.notes,
        })

    return Response({'program_id': program_pk, 'field_slug': field.slug, 'count': len(results), 'results': results})
