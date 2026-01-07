import json

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError

from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response

from utils.errors import error_response


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_programs(request):
    try:
        q = (request.GET.get("q") or "").strip().lower()
        field = (request.GET.get("field") or "").strip().lower()
        level = (request.GET.get("level") or "").strip().lower()
        region = (request.GET.get("region") or "").strip().lower()
        page = max(1, int(request.GET.get("page", 1)))
        page_size = max(1, min(50, int(request.GET.get("page_size", 20))))

        try:
            from catalog.models import Program  # type: ignore
        except Exception:
            Program = None  # type: ignore

        if Program is None:
            return error_response(
                "Catalog DB not available",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code='catalog_unavailable',
            )
        try:
            from django.db.models import Q
        except Exception:
            Q = None

        qs = Program.objects.select_related("institution", "field").all()
        if q:
            if Q is not None:
                qs = qs.filter(
                    Q(name__icontains=q)
                    | Q(normalized_name__icontains=q)
                    | Q(institution__name__icontains=q)
                    | Q(institution__alias__icontains=q)
                )
            else:
                qs = qs.filter(normalized_name__icontains=q)
        if level:
            qs = qs.filter(level__iexact=level)
        if region:
            if Q is not None:
                qs = qs.filter(
                    Q(region__icontains=region)
                    | Q(institution__region__icontains=region)
                    | Q(institution__county__icontains=region)
                )
            else:
                qs = qs.filter(region__icontains=region)
        if field:
            if Q is not None:
                qs = qs.filter(Q(field__name__iexact=field) | Q(field__slug__iexact=field))
            else:
                qs = qs.filter(field__name__iexact=field)

        total = int(qs.count())
        if total <= 0:
            return Response(
                {
                    "count": 0,
                    "page": page,
                    "page_size": page_size,
                    "results": [],
                    "detail": "No programmes found in the catalog database. This usually means the database has not been populated yet. Run the ETL load step to populate Program data.",
                }
            )

        start = (page - 1) * page_size
        end = start + page_size
        items = []
        for p in list(qs.order_by("normalized_name")[start:end]):
            inst = getattr(p, "institution", None)
            fld = getattr(p, "field", None)
            items.append(
                {
                    "id": int(getattr(p, "id", 0) or 0),
                    "program_code": (getattr(p, "code", "") or ""),
                    "name": (getattr(p, "name", "") or ""),
                    "normalized_name": (getattr(p, "normalized_name", "") or ""),
                    "institution_name": (getattr(inst, "name", "") or "") if inst else "",
                    "field_name": (getattr(fld, "name", "") or "") if fld else "",
                    "level": (getattr(p, "level", "") or ""),
                    "campus": (getattr(p, "campus", "") or ""),
                    "region": (getattr(p, "region", "") or ""),
                    "award": (getattr(p, "award", "") or ""),
                }
            )
        return Response({"count": total, "page": page, "page_size": page_size, "results": items})
    except (ProgrammingError, OperationalError) as e:
        detail = "Database not initialized"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='db_unavailable')
    except Exception as e:
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_institutions(request):
    try:
        q = (request.GET.get("q") or "").strip().lower()
        region = (request.GET.get("region") or "").strip()
        county = (request.GET.get("county") or "").strip()

        try:
            from catalog.models import Institution  # type: ignore
        except Exception:
            Institution = None  # type: ignore

        if Institution is None:
            return error_response(
                "Institution catalog database is not available.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code='catalog_unavailable',
            )

        try:
            from django.db.models import Q
        except Exception:
            Q = None

        qs = Institution.objects.all()
        if q:
            if Q is not None:
                qs = qs.filter(Q(name__icontains=q) | Q(alias__icontains=q))
            else:
                qs = qs.filter(name__icontains=q)
        if region:
            qs = qs.filter(region__icontains=region)
        if county:
            qs = qs.filter(county__icontains=county)

        total = int(qs.count())
        if total <= 0:
            return Response(
                {
                    "count": 0,
                    "results": [],
                    "detail": "No institutions found in the catalog database. This usually means the database has not been populated yet. Run the ETL load step to populate Institution data.",
                }
            )

        rows = list(qs.order_by("name")[:500].values("code", "name", "alias", "region", "county", "website"))
        return Response({"count": total, "results": rows})
    except Exception as e:
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_fields(request):
    try:
        q = (request.GET.get("q") or "").strip().lower()
        parent = (request.GET.get("parent") or "").strip().lower()
        try:
            from catalog.models import Field  # type: ignore
        except Exception:
            Field = None  # type: ignore

        if Field is None:
            return error_response(
                "Catalog DB not available",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code='catalog_unavailable',
            )

        try:
            from django.db.models import Q
        except Exception:
            Q = None

        qs = Field.objects.select_related("parent").all()
        if q:
            qs = qs.filter(name__icontains=q)
        if parent:
            if Q is not None:
                qs = qs.filter(Q(parent__slug__iexact=parent) | Q(parent__name__iexact=parent))
            else:
                qs = qs.filter(parent__slug__iexact=parent)

        total = int(qs.count())
        if total <= 0:
            return Response(
                {
                    "count": 0,
                    "results": [],
                    "detail": "No fields found in the catalog database. This usually means the database has not been populated yet. Run the ETL load step to populate Field data.",
                }
            )

        rows = []
        for f in list(qs.order_by("name")[:500]):
            rows.append(
                {
                    "name": (getattr(f, "name", "") or ""),
                    "slug": (getattr(f, "slug", "") or ""),
                    "parent": (getattr(getattr(f, "parent", None), "name", "") or ""),
                }
            )
        return Response({"count": total, "results": rows})
    except (ProgrammingError, OperationalError) as e:
        detail = "Database not initialized"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='db_unavailable')
    except Exception as e:
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_search(request):
    try:
        q = (request.GET.get("q") or "").strip().lower()
        if not q:
            return error_response(
                "q is required",
                status_code=status.HTTP_400_BAD_REQUEST,
                code='validation_error',
                fields={'q': ['This field is required.']},
            )
        try:
            from catalog.models import Field, Institution, Program  # type: ignore
        except Exception:
            Field = None  # type: ignore
            Institution = None  # type: ignore
            Program = None  # type: ignore

        if Field is None or Institution is None or Program is None:
            return error_response(
                "Catalog DB not available",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code='catalog_unavailable',
            )

        try:
            from django.db.models import Q
        except Exception:
            Q = None

        programs = []
        pqs = Program.objects.select_related("institution", "field").all()
        if Q is not None:
            pqs = pqs.filter(
                Q(name__icontains=q)
                | Q(normalized_name__icontains=q)
                | Q(institution__name__icontains=q)
                | Q(institution__alias__icontains=q)
            )
        else:
            pqs = pqs.filter(normalized_name__icontains=q)
        for p in list(pqs.order_by("normalized_name")[:10]):
            inst = getattr(p, "institution", None)
            fld = getattr(p, "field", None)
            programs.append(
                {
                    "program_code": (getattr(p, "code", "") or ""),
                    "name": (getattr(p, "name", "") or ""),
                    "normalized_name": (getattr(p, "normalized_name", "") or ""),
                    "institution_name": (getattr(inst, "name", "") or "") if inst else "",
                    "field_name": (getattr(fld, "name", "") or "") if fld else "",
                    "level": (getattr(p, "level", "") or ""),
                }
            )

        institutions = []
        iqs = Institution.objects.all()
        if Q is not None:
            iqs = iqs.filter(Q(name__icontains=q) | Q(alias__icontains=q))
        else:
            iqs = iqs.filter(name__icontains=q)
        for inst in list(iqs.order_by("name")[:10]):
            institutions.append(
                {
                    "code": (getattr(inst, "code", "") or ""),
                    "name": (getattr(inst, "name", "") or ""),
                    "alias": (getattr(inst, "alias", "") or ""),
                    "region": (getattr(inst, "region", "") or ""),
                    "county": (getattr(inst, "county", "") or ""),
                    "website": (getattr(inst, "website", "") or ""),
                }
            )

        fields = []
        fqs = Field.objects.select_related("parent").filter(name__icontains=q).order_by("name")[:10]
        for f in list(fqs):
            fields.append(
                {
                    "name": (getattr(f, "name", "") or ""),
                    "slug": (getattr(f, "slug", "") or ""),
                    "parent": (getattr(getattr(f, "parent", None), "name", "") or ""),
                }
            )

        return Response({"programs": programs, "institutions": institutions, "fields": fields})
    except (ProgrammingError, OperationalError) as e:
        detail = "Database not initialized"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='db_unavailable')
    except Exception as e:
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')


@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def api_eligibility(request):
    try:
        data = request.data or {}
        if not isinstance(data, dict):
            return error_response(
                "Invalid JSON body",
                status_code=status.HTTP_400_BAD_REQUEST,
                code='validation_error',
            )
        program_code = str(data.get("program_code") or "").strip()
        grades = data.get("grades") or {}
        if not program_code or not isinstance(grades, dict):
            fields = {}
            if not program_code:
                fields['program_code'] = ['This field is required.']
            if not isinstance(grades, dict) or not grades:
                fields['grades'] = ['This field is required.']
            return error_response(
                "program_code and grades are required",
                status_code=status.HTTP_400_BAD_REQUEST,
                code='validation_error',
                fields=fields or None,
            )
        try:
            from catalog.models import Program  # type: ignore
        except Exception:
            Program = None  # type: ignore

        if Program is None:
            return error_response(
                "Catalog DB not available",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code='catalog_unavailable',
            )

        try:
            from scripts.etl.kuccps.eligibility import evaluate_eligibility  # type: ignore
        except Exception as e:
            detail = "Import error"
            if settings.DEBUG:
                detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
            return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='import_error')

        p = Program.objects.select_related("institution").filter(code__iexact=program_code).first()
        if not p:
            return error_response(
                f"Program {program_code} not found",
                status_code=status.HTTP_404_NOT_FOUND,
                code='not_found',
            )

        inst = getattr(p, "institution", None)
        prog_row = {
            "institution_code": (getattr(inst, "code", "") or "").strip() if inst else "",
            "institution_name": (getattr(inst, "name", "") or "").strip() if inst else "",
            "program_code": (getattr(p, "code", "") or "").strip(),
            "name": (getattr(p, "name", "") or "").strip(),
            "normalized_name": (getattr(p, "normalized_name", "") or "").strip(),
            "subject_requirements_json": json.dumps(getattr(p, "subject_requirements", None) or {}),
        }

        result = evaluate_eligibility(prog_row, grades)
        return Response(
            {
                "institution_code": prog_row["institution_code"],
                "institution_name": prog_row["institution_name"],
                "program_code": prog_row["program_code"],
                "program_name": prog_row["name"] or prog_row["normalized_name"],
                "normalized_name": prog_row["normalized_name"],
                "result": result,
            }
        )
    except (ProgrammingError, OperationalError) as e:
        detail = "Database not initialized"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code='db_unavailable')
    except Exception as e:
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, code='server_error')
