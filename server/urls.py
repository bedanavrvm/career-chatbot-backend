"""
URL configuration for server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
import os
import base64
import json
import csv
from pathlib import Path
import sys
import firebase_admin
from firebase_admin import auth as fb_auth, credentials

if not firebase_admin._apps:
    b64 = os.getenv('FIREBASE_CREDENTIALS_JSON_B64')
    if b64:
        try:
            data = json.loads(base64.b64decode(b64).decode('utf-8'))
            cred = credentials.Certificate(data)
            firebase_admin.initialize_app(cred)
        except Exception:
            # Fallback: app remains uninitialized; secure endpoints will error until configured
            pass


def health(_request):
    return JsonResponse({"status": "ok"})


def secure_ping(request):
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    token = ''
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ', 1)[1]
    elif 'token' in request.GET:
        token = request.GET.get('token', '')

    if not token:
        return JsonResponse({"detail": "Missing bearer token"}, status=401)

    try:
        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get('uid')
        return JsonResponse({"status": "ok", "uid": uid})
    except Exception:
        return JsonResponse({"detail": "Invalid token"}, status=401)


# ---- Minimal CSV-backed API (no DB) ----
# Utilities to locate processed CSVs and robustly read TSV/CSV
def _processed_dir() -> Path:
    # Default to KUCCPS processed dir relative to backend/
    backend_dir = Path(__file__).resolve().parent.parent
    default_dir = backend_dir / "scripts" / "etl" / "kuccps" / "processed"
    env_dir = os.getenv("KUCCPS_PROCESSED_DIR")
    return Path(env_dir).resolve() if env_dir else default_dir


def _detect_reader(f) -> csv.DictReader:
    head = f.readline()
    rest = f.read(8192)
    f.seek(0)
    try:
        if "\t" in head:
            return csv.DictReader(f, delimiter="\t")
        sample = head + rest
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
        return csv.DictReader(f, dialect=dialect)
    except Exception:
        f.seek(0)
        return csv.DictReader(f, delimiter="\t")


def _programs_path() -> Path:
    base = _processed_dir()
    cand = base / "programs_deduped.csv"
    if cand.exists():
        return cand
    return base / "programs.csv"


def _institutions_path() -> Path:
    return _processed_dir() / "institutions.csv"


def _fields_path() -> Path:
    return _processed_dir() / "fields.csv"


def api_programs(request):
    """GET /api/etl/programs
    Query params: q, field, level, region, page, page_size
    Reads programs_deduped.csv (or programs.csv) and returns paginated results.
    """
    try:
        fp = _programs_path()
        if not fp.exists():
            return JsonResponse({"detail": f"Programs file not found at {fp}"}, status=500)
        q = (request.GET.get("q") or "").strip().lower()
        field = (request.GET.get("field") or "").strip().lower()
        level = (request.GET.get("level") or "").strip().lower()
        region = (request.GET.get("region") or "").strip().lower()
        page = max(1, int(request.GET.get("page", 1)))
        page_size = max(1, min(100, int(request.GET.get("page_size", 20))))

        rows = []
        with open(fp, encoding="utf-8") as f:
            reader = _detect_reader(f)
            for row in reader:
                name = (row.get("name") or "").lower()
                norm = (row.get("normalized_name") or "").lower()
                inst = (row.get("institution_name") or "").lower()
                field_name = (row.get("field_name") or "").lower()
                level_val = (row.get("level") or "").lower()
                region_val = (row.get("region") or "").lower()
                if q and not (q in name or q in norm or q in inst):
                    continue
                if field and field != field_name:
                    continue
                if level and level != level_val:
                    continue
                if region and region != region_val:
                    continue
                rows.append(row)

        total = len(rows)
        start = (page - 1) * page_size
        end = start + page_size
        items = rows[start:end]
        return JsonResponse({
            "count": total,
            "page": page,
            "page_size": page_size,
            "results": items,
        })
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


def api_eligibility(request):
    """POST /api/etl/eligibility
    Body: { program_code: str, grades: {SUBJ: GRADE, ...} }
    Returns evaluate_eligibility output for the program code.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
        program_code = str(data.get("program_code") or "").strip()
        grades = data.get("grades") or {}
        if not program_code or not isinstance(grades, dict):
            return JsonResponse({"detail": "program_code and grades are required"}, status=400)

        # Import eligibility helper from scripts folder
        kuccps_dir = Path(__file__).resolve().parent.parent / "scripts" / "etl" / "kuccps"
        sys.path.append(str(kuccps_dir))
        try:
            from eligibility import evaluate_eligibility  # type: ignore
        except Exception as e:
            return JsonResponse({"detail": f"Import error: {e}"}, status=500)

        # Load program row
        prog_row = None
        with open(_programs_path(), encoding="utf-8") as f:
            reader = _detect_reader(f)
            for row in reader:
                code = (row.get("program_code") or row.get("code") or "").strip()
                if code == program_code:
                    prog_row = row
                    break
        if not prog_row:
            return JsonResponse({"detail": f"Program {program_code} not found"}, status=404)

        result = evaluate_eligibility(prog_row, grades)
        out = {
            "institution_code": (prog_row.get("institution_code") or "").strip(),
            "institution_name": (prog_row.get("institution_name") or "").strip(),
            "program_code": program_code,
            "program_name": (prog_row.get("name") or prog_row.get("normalized_name") or "").strip(),
            "normalized_name": (prog_row.get("normalized_name") or "").strip(),
            "result": result,
        }
        return JsonResponse(out)
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


def api_institutions(request):
    """GET /api/etl/institutions
    Query params: q, region, county
    Reads processed/institutions.csv and returns filtered results.
    """
    try:
        fp = _institutions_path()
        if not fp.exists():
            return JsonResponse({"detail": f"Institutions file not found at {fp}"}, status=500)
        q = (request.GET.get("q") or "").strip().lower()
        region = (request.GET.get("region") or "").strip().lower()
        county = (request.GET.get("county") or "").strip().lower()
        rows = []
        with open(fp, encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                name = (row.get("name") or "").lower()
                alias = (row.get("alias") or "").lower()
                reg = (row.get("region") or "").lower()
                cty = (row.get("county") or "").lower()
                if q and not (q in name or q in alias):
                    continue
                if region and region != reg:
                    continue
                if county and county != cty:
                    continue
                rows.append(row)
        return JsonResponse({"count": len(rows), "results": rows})
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


def api_fields(request):
    """GET /api/etl/fields
    Query params: q, parent
    Reads processed/fields.csv and returns filtered results.
    """
    try:
        fp = _fields_path()
        if not fp.exists():
            return JsonResponse({"detail": f"Fields file not found at {fp}"}, status=500)
        q = (request.GET.get("q") or "").strip().lower()
        parent = (request.GET.get("parent") or "").strip().lower()
        rows = []
        with open(fp, encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                name = (row.get("name") or "").lower()
                parent_val = (row.get("parent") or "").lower()
                if q and q not in name:
                    continue
                if parent and parent != parent_val:
                    continue
                rows.append(row)
        return JsonResponse({"count": len(rows), "results": rows})
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


def api_search(request):
    """GET /api/etl/search?q=
    Simple search across programs, institutions, and fields (top 10 each).
    """
    try:
        q = (request.GET.get("q") or "").strip().lower()
        if not q:
            return JsonResponse({"detail": "q is required"}, status=400)
        # Programs
        programs = []
        pfp = _programs_path()
        if pfp.exists():
            with open(pfp, encoding="utf-8") as f:
                rdr = _detect_reader(f)
                for row in rdr:
                    name = (row.get("name") or "").lower()
                    norm = (row.get("normalized_name") or "").lower()
                    inst = (row.get("institution_name") or "").lower()
                    if q in name or q in norm or q in inst:
                        programs.append({
                            "program_code": row.get("program_code") or row.get("code"),
                            "name": row.get("name"),
                            "normalized_name": row.get("normalized_name"),
                            "institution_name": row.get("institution_name"),
                            "field_name": row.get("field_name"),
                            "level": row.get("level"),
                        })
                        if len(programs) >= 10:
                            break
        # Institutions
        institutions = []
        if _institutions_path().exists():
            with open(_institutions_path(), encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    name = (row.get("name") or "").lower()
                    alias = (row.get("alias") or "").lower()
                    if q in name or q in alias:
                        institutions.append(row)
                        if len(institutions) >= 10:
                            break
        # Fields
        fields = []
        if _fields_path().exists():
            with open(_fields_path(), encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    name = (row.get("name") or "").lower()
                    if q in name:
                        fields.append(row)
                        if len(fields) >= 10:
                            break
        return JsonResponse({"programs": programs, "institutions": institutions, "fields": fields})
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


@staff_member_required
def admin_etl_upload(request):
    """POST /admin/etl/upload (multipart/form-data)
    Saves uploaded file into KUCCPS raw/uploads and returns stored path.
    """
    base = _processed_dir().parent / "raw" / "uploads"
    if request.method == "GET":
        # Render an admin page with an upload form
        return render(request, "admin/etl_upload.html", {"upload_dir": str(base)})
    try:
        ctx = {"upload_dir": str(base)}
        if not request.FILES:
            ctx["error"] = "No file provided"
            return render(request, "admin/etl_upload.html", ctx)
        fobj = next(iter(request.FILES.values()))
        base.mkdir(parents=True, exist_ok=True)
        safe_name = os.path.basename(fobj.name).replace("..", "_")
        dest = base / safe_name
        with open(dest, "wb") as out:
            for chunk in fobj.chunks():
                out.write(chunk)
        ctx["stored"] = str(dest)
        return render(request, "admin/etl_upload.html", ctx)
    except Exception as e:
        return render(request, "admin/etl_upload.html", {"upload_dir": str(base), "error": str(e)})


@staff_member_required
def admin_etl_process(request):
    """POST /admin/etl/process
    Optional body: {"action": "transform-normalize"}
    Invokes transform-normalize to regenerate processed CSVs using local config.
    """
    if request.method == "GET":
        return render(request, "admin/etl_process.html", {})
    try:
        # Allow both JSON body and form submission
        action = "transform-normalize"
        if request.content_type and "application/json" in request.content_type:
            body = json.loads(request.body.decode("utf-8") or "{}")
            action = (body.get("action") or action).strip()
        else:
            action = (request.POST.get("action") or action).strip()
        etl_dir = Path(__file__).resolve().parent.parent / "scripts" / "etl" / "kuccps"
        sys.path.append(str(etl_dir))
        from etl import Config, transform_normalize  # type: ignore
        cfg = Config(dataset_year=2024, dataset_root=etl_dir, inputs={}, raw_dir=etl_dir / "raw", processed_dir=etl_dir / "processed")
        ran = False
        if action == "transform-normalize":
            transform_normalize(cfg)
            ran = True
        return render(request, "admin/etl_process.html", {"action": action, "ran": ran, "processed_dir": str(cfg.processed_dir)})
    except Exception as e:
        return render(request, "admin/etl_process.html", {"error": str(e)})


urlpatterns = [
    path('_health/', health, name='health'),
    path('_health', health, name='health_no_slash'),
    path('api/auth/', include('accounts.urls')),
    path('api/secure-ping/', secure_ping, name='secure_ping'),
    path('api/etl/programs', api_programs, name='api_programs'),
    path('api/etl/eligibility', api_eligibility, name='api_eligibility'),
    path('api/etl/institutions', api_institutions, name='api_institutions'),
    path('api/etl/fields', api_fields, name='api_fields'),
    path('api/etl/search', api_search, name='api_search'),
    path('admin/etl/upload', admin_etl_upload, name='admin_etl_upload'),
    path('admin/etl/process', admin_etl_process, name='admin_etl_process'),
    path('admin/', admin.site.urls),
]
