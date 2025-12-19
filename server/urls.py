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
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_protect, csrf_exempt
from django.conf import settings
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


# ---- Simple in-process CSV cache and ETag helpers (dev-friendly) ----
# Cache keyed by absolute path; values: {"sig": (mtime, size), "rows": [...]}.
_CSV_CACHE: dict[str, dict] = {}


def _file_sig(fp: Path) -> tuple:
    try:
        st = fp.stat()
        return (int(st.st_mtime), int(st.st_size))
    except Exception:
        return (0, 0)


def _etag_for(sig: tuple) -> str:
    # Weak ETag using mtime and size; stable across reads until file changes
    return f'W/"{sig[0]}-{sig[1]}"'


def _read_programs_cached() -> tuple[list[dict], str]:
    fp = _programs_path()
    sig = _file_sig(fp)
    key = str(fp.resolve())
    cached = _CSV_CACHE.get(key)
    if not cached or cached.get("sig") != sig:
        rows: list[dict] = []
        with open(fp, encoding="utf-8") as f:
            rdr = _detect_reader(f)
            rows = [r for r in rdr]
        _CSV_CACHE[key] = {"sig": sig, "rows": rows}
    etag = _etag_for(sig)
    return _CSV_CACHE[key]["rows"], etag


def _read_csv_cached(fp: Path) -> tuple[list[dict], str]:
    sig = _file_sig(fp)
    key = str(fp.resolve())
    cached = _CSV_CACHE.get(key)
    if not cached or cached.get("sig") != sig:
        rows: list[dict] = []
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f)]
        _CSV_CACHE[key] = {"sig": sig, "rows": rows}
    etag = _etag_for(sig)
    return _CSV_CACHE[key]["rows"], etag


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
        page_size = max(1, min(50, int(request.GET.get("page_size", 20))))

        all_rows, etag = _read_programs_cached()
        # Conditional GET via ETag
        inm = request.META.get("HTTP_IF_NONE_MATCH")
        if inm and inm == etag:
            return HttpResponse(status=304)

        rows = []
        for row in all_rows:
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
        resp = JsonResponse({
            "count": total,
            "page": page,
            "page_size": page_size,
            "results": items,
        })
        resp["ETag"] = etag
        return resp
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
        rows_all, etag = _read_csv_cached(fp)
        inm = request.META.get("HTTP_IF_NONE_MATCH")
        if inm and inm == etag:
            return HttpResponse(status=304)
        rows = []
        for row in rows_all:
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
        resp = JsonResponse({"count": len(rows), "results": rows})
        resp["ETag"] = etag
        return resp
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
        rows_all, etag = _read_csv_cached(fp)
        inm = request.META.get("HTTP_IF_NONE_MATCH")
        if inm and inm == etag:
            return HttpResponse(status=304)
        rows = []
        for row in rows_all:
            name = (row.get("name") or "").lower()
            parent_val = (row.get("parent") or "").lower()
            if q and q not in name:
                continue
            if parent and parent != parent_val:
                continue
            rows.append(row)
        resp = JsonResponse({"count": len(rows), "results": rows})
        resp["ETag"] = etag
        return resp
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
        prog_rows, etag_p = _read_programs_cached()
        for row in prog_rows:
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
        inst_rows, etag_i = _read_csv_cached(_institutions_path())
        for row in inst_rows:
            name = (row.get("name") or "").lower()
            alias = (row.get("alias") or "").lower()
            if q in name or q in alias:
                institutions.append(row)
                if len(institutions) >= 10:
                    break
        # Fields
        fields = []
        field_rows, etag_f = _read_csv_cached(_fields_path())
        for row in field_rows:
            name = (row.get("name") or "").lower()
            if q in name:
                fields.append(row)
                if len(fields) >= 10:
                    break
        etag = f'{etag_p},{etag_i},{etag_f}'
        inm = request.META.get("HTTP_IF_NONE_MATCH")
        if inm and inm == etag:
            return HttpResponse(status=304)
        resp = JsonResponse({"programs": programs, "institutions": institutions, "fields": fields})
        resp["ETag"] = etag
        return resp
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


@staff_member_required
@ensure_csrf_cookie
@csrf_protect
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
@ensure_csrf_cookie
@csrf_protect
def admin_etl_process(request):
    """POST /admin/etl/process
    Optional body: {"action": "transform-normalize"}
    Invokes transform-normalize to regenerate processed CSVs using local config.
    """
    if request.method == "GET":
        actions = [
            "extract",
            "extract-programs",
            "transform",
            "transform-programs",
            "transform-normalize",
            "dedup-programs",
            "dq-report",
            "load",
            "all",
        ]
        return render(request, "admin/etl_process.html", {"actions": actions})
    try:
        # Allow both JSON body and form submission
        action = (request.POST.get("action") or "transform-normalize").strip()
        config_path = (request.POST.get("config") or "").strip()
        inplace = bool(request.POST.get("inplace"))
        dry_run = bool(request.POST.get("dry_run"))

        etl_dir = Path(__file__).resolve().parent.parent / "scripts" / "etl" / "kuccps"
        sys.path.append(str(etl_dir))
        from etl import (
            Config,
            copy_inputs,
            bootstrap_csvs,
            extract_programs,
            transform_programs,
            transform_normalize,
            dedup_programs,
            dq_report,
            load_csvs,
        )  # type: ignore

        # Resolve config
        cfg = None
        msg = []
        etl_stats = {}
        if config_path:
            # Support absolute paths or paths relative to backend/ root
            backend_dir = Path(__file__).resolve().parent.parent
            p = Path(config_path)
            if not p.is_absolute():
                p = (backend_dir / p).resolve()
            try:
                cfg = Config.from_yaml(p)
                msg.append(f"Using config: {p}")
            except Exception as e:
                return render(request, "admin/etl_process.html", {"error": f"Invalid config: {e}", "actions": [
                    "extract","extract-programs","transform","transform-programs","transform-normalize","dedup-programs","dq-report","load","all"
                ]})
        else:
            # Default to local KUCCPS directories
            cfg = Config(dataset_year=2024, dataset_root=etl_dir, inputs={}, raw_dir=etl_dir / "raw", processed_dir=etl_dir / "processed")

        ran = False
        load_summary = None
        if action == "extract":
            copy_inputs(cfg)
            ran = True
        elif action == "extract-programs":
            # If no programs_pdf configured, try to discover latest uploaded PDF under raw/uploads
            if not cfg.inputs.get("programs_pdf"):
                uploads = (etl_dir / "raw" / "uploads")
                if uploads.exists():
                    pdfs = sorted(uploads.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if pdfs:
                        latest = pdfs[0]
                        try:
                            rel = latest.resolve().relative_to(cfg.dataset_root)
                        except Exception:
                            # fallback to path relative to etl_dir
                            rel = latest.resolve().relative_to(etl_dir)
                        cfg.inputs["programs_pdf"] = str(rel)
                        # Infer a source_id from the PDF filename for namespacing/provenance
                        cfg.inputs["source_id"] = latest.stem
                        msg.append(f"Detected uploaded PDF: {latest.name}")
                if not cfg.inputs.get("programs_pdf"):
                    return render(request, "admin/etl_process.html", {
                        "error": "No programs PDF configured or found under raw/uploads. Upload a PDF first on the Upload page.",
                        "actions": [
                            "extract","extract-programs","transform","transform-programs","transform-normalize","dedup-programs","dq-report","load","all"
                        ],
                        "messages": msg,
                    })
            # Hard dependency check for pdfplumber
            try:
                import pdfplumber  # noqa: F401
            except Exception as dep_err:
                return render(request, "admin/etl_process.html", {
                    "error": f"Missing dependency: pdfplumber not installed ({dep_err}). Please install it in venv.",
                    "actions": [
                        "extract","extract-programs","transform","transform-programs","transform-normalize","dedup-programs","dq-report","load","all"
                    ],
                    "messages": msg,
                })
            extract_programs(cfg)
            ran = True
            # Stats: raw artifacts
            text_dir = cfg.raw_dir / "programs_text"
            tables_dir = cfg.raw_dir / "programs_tables"
            uploads = cfg.raw_dir / "uploads"
            pdf_count = len(list(cfg.raw_dir.glob("*.pdf"))) + (len(list(uploads.glob("*.pdf"))) if uploads.exists() else 0)
            etl_stats.update({
                "raw_dir": str(cfg.raw_dir),
                "programs_pdf_count": pdf_count,
                "programs_text_pages": len(list(text_dir.glob("*.txt"))) if text_dir.exists() else 0,
                "programs_tables": len(list(tables_dir.glob("*.csv"))) if tables_dir.exists() else 0,
            })
        elif action == "transform":
            bootstrap_csvs(cfg)
            ran = True
            # Stats: existence of templates
            for fname in ("programs.csv","yearly_cutoffs.csv","institutions.csv","fields.csv"):
                fp = cfg.processed_dir / fname
                etl_stats[f"exists:{fname}"] = fp.exists()
        elif action == "transform-programs":
            transform_programs(cfg)
            ran = True
            # Stats: counts from processed outputs
            def _count(fp: Path) -> int:
                if not fp.exists():
                    return 0
                with open(fp, encoding="utf-8") as f:
                    try:
                        rdr = _detect_reader(f)
                        return sum(1 for _ in rdr)
                    except Exception:
                        f.seek(0)
                        return sum(1 for _ in csv.DictReader(f))
            etl_stats.update({
                "programs.csv_rows": _count(cfg.processed_dir / "programs.csv"),
                "yearly_cutoffs.csv_rows": _count(cfg.processed_dir / "yearly_cutoffs.csv"),
                "_code_corrections.csv_rows": _count(cfg.processed_dir / "_code_corrections.csv"),
                "created:program_costs.csv": (cfg.processed_dir / "program_costs.csv").exists(),
                "program_costs.csv_rows": _count(cfg.processed_dir / "program_costs.csv"),
            })
        elif action == "transform-normalize":
            transform_normalize(cfg)
            ran = True
            # Stats: normalized outputs
            def _count(fp: Path) -> int:
                if not fp.exists():
                    return 0
                with open(fp, encoding="utf-8") as f:
                    try:
                        rdr = _detect_reader(f)
                        return sum(1 for _ in rdr)
                    except Exception:
                        f.seek(0)
                        return sum(1 for _ in csv.DictReader(f))
            def _columns(fp: Path) -> str:
                if not fp.exists():
                    return ""
                with open(fp, encoding="utf-8") as f:
                    try:
                        rdr = _detect_reader(f)
                        return ",".join(rdr.fieldnames or [])
                    except Exception:
                        f.seek(0)
                        return ",".join(csv.DictReader(f).fieldnames or [])
            etl_stats.update({
                "programs.csv_rows": _count(cfg.processed_dir / "programs.csv"),
                "institutions.csv_rows": _count(cfg.processed_dir / "institutions.csv"),
                "_unclassified_programs.csv_rows": _count(cfg.processed_dir / "_unclassified_programs.csv"),
                # New derivative outputs
                "created:program_offerings.csv": (cfg.processed_dir / "program_offerings.csv").exists(),
                "program_offerings.csv_rows": _count(cfg.processed_dir / "program_offerings.csv"),
                "created:program_offerings_broad.csv": (cfg.processed_dir / "program_offerings_broad.csv").exists(),
                "program_offerings_broad.csv_rows": _count(cfg.processed_dir / "program_offerings_broad.csv"),
                "created:dedup_candidates.csv": (cfg.processed_dir / "dedup_candidates.csv").exists(),
                "dedup_candidates.csv_rows": _count(cfg.processed_dir / "dedup_candidates.csv"),
                "created:dedup_summary.csv": (cfg.processed_dir / "dedup_summary.csv").exists(),
                "dedup_summary.csv_rows": _count(cfg.processed_dir / "dedup_summary.csv"),
                # Headers for quick visibility
                "columns:programs.csv": _columns(cfg.processed_dir / "programs.csv"),
                "columns:institutions.csv": _columns(cfg.processed_dir / "institutions.csv"),
            })
        elif action == "dedup-programs":
            dedup_programs(cfg, inplace=inplace)
            ran = True
            # Stats: dedup outputs
            def _count(fp: Path) -> int:
                if not fp.exists():
                    return 0
                with open(fp, encoding="utf-8") as f:
                    try:
                        rdr = _detect_reader(f)
                        return sum(1 for _ in rdr)
                    except Exception:
                        f.seek(0)
                        return sum(1 for _ in csv.DictReader(f))
            etl_stats.update({
                "programs_deduped.csv_rows": _count(cfg.processed_dir / "programs_deduped.csv"),
                "dedup_suppressed.csv_rows": _count(cfg.processed_dir / "dedup_suppressed.csv"),
            })
        elif action == "dq-report":
            dq_report(cfg)
            ran = True
            # Stats: read dq_report.csv metrics
            dq_fp = cfg.processed_dir / "dq_report.csv"
            metrics = {}
            if dq_fp.exists():
                with open(dq_fp, encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        metrics[r.get("metric") or "metric"] = r.get("value")
            etl_stats.update(metrics)
        elif action == "load":
            changes = load_csvs(cfg, dry_run=dry_run)
            ran = True
            if not dry_run:
                try:
                    # Post-load verification: expanded counts
                    from catalog.models import (
                        Institution, Field, Program, YearlyCutoff,
                        InstitutionCampus, ProgramOfferingAggregate, ProgramOfferingBroadAggregate,
                        DedupCandidateGroup, DedupSummary, CodeCorrectionAudit,
                        ETLRun, DQReportEntry, ClusterSubjectRule, ProgramRequirementNormalized,
                        ProgramRequirementGroup, ProgramRequirementOption,
                    )  # type: ignore
                    totals = (changes or {}).get("totals") or {}
                    by_source = (changes or {}).get("by_source") or {}
                    load_summary = {
                        "institutions": Institution.objects.count(),
                        "fields": Field.objects.count(),
                        "programs": Program.objects.count(),
                        "yearly_cutoffs": YearlyCutoff.objects.count(),
                        "institution_campuses": InstitutionCampus.objects.count(),
                        "offerings": ProgramOfferingAggregate.objects.count(),
                        "offerings_broad": ProgramOfferingBroadAggregate.objects.count(),
                        "dedup_candidate_groups": DedupCandidateGroup.objects.count(),
                        "dedup_summaries": DedupSummary.objects.count(),
                        "code_corrections": CodeCorrectionAudit.objects.count(),
                        "etl_runs": ETLRun.objects.count(),
                        "dq_entries": DQReportEntry.objects.count(),
                        "cluster_subject_rules": ClusterSubjectRule.objects.count(),
                        "normalized_requirements": ProgramRequirementNormalized.objects.count(),
                        "requirement_groups": ProgramRequirementGroup.objects.count(),
                        "requirement_options": ProgramRequirementOption.objects.count(),
                        "changes_totals": totals,
                        "changes_by_source": by_source,
                    }
                    # Flatten for quick visibility
                    for k, v in totals.items():
                        etl_stats[f"changes:totals:{k}"] = v
                    for sid, sub in by_source.items():
                        for k, v in sub.items():
                            etl_stats[f"changes:by_source:{sid}:{k}"] = v
                except Exception:
                    pass
        elif action == "all":
            copy_inputs(cfg)
            bootstrap_csvs(cfg)
            # Auto-extract all uploaded PDFs under raw/uploads (multi-source)
            uploads = (etl_dir / "raw" / "uploads")
            if uploads.exists():
                pdfs = sorted(uploads.glob("*.pdf"))
                for latest in pdfs:
                    try:
                        rel = latest.resolve().relative_to(cfg.dataset_root)
                    except Exception:
                        rel = latest.resolve().relative_to(etl_dir)
                    cfg.inputs["programs_pdf"] = str(rel)
                    cfg.inputs["source_id"] = latest.stem
                    extract_programs(cfg)
            transform_programs(cfg)
            transform_normalize(cfg)
            dedup_programs(cfg, inplace=False)
            dq_report(cfg)
            changes = load_csvs(cfg, dry_run=dry_run)
            ran = True
            if not dry_run:
                try:
                    from catalog.models import (
                        Institution, Field, Program, YearlyCutoff,
                        InstitutionCampus, ProgramOfferingAggregate, ProgramOfferingBroadAggregate,
                        DedupCandidateGroup, DedupSummary, CodeCorrectionAudit,
                        ETLRun, DQReportEntry, ClusterSubjectRule, ProgramRequirementNormalized,
                        ProgramRequirementGroup, ProgramRequirementOption,
                    )  # type: ignore
                    totals = (changes or {}).get("totals") or {}
                    by_source = (changes or {}).get("by_source") or {}
                    load_summary = {
                        "institutions": Institution.objects.count(),
                        "fields": Field.objects.count(),
                        "programs": Program.objects.count(),
                        "yearly_cutoffs": YearlyCutoff.objects.count(),
                        "institution_campuses": InstitutionCampus.objects.count(),
                        "offerings": ProgramOfferingAggregate.objects.count(),
                        "offerings_broad": ProgramOfferingBroadAggregate.objects.count(),
                        "dedup_candidate_groups": DedupCandidateGroup.objects.count(),
                        "dedup_summaries": DedupSummary.objects.count(),
                        "code_corrections": CodeCorrectionAudit.objects.count(),
                        "etl_runs": ETLRun.objects.count(),
                        "dq_entries": DQReportEntry.objects.count(),
                        "cluster_subject_rules": ClusterSubjectRule.objects.count(),
                        "normalized_requirements": ProgramRequirementNormalized.objects.count(),
                        "requirement_groups": ProgramRequirementGroup.objects.count(),
                        "requirement_options": ProgramRequirementOption.objects.count(),
                        "changes_totals": totals,
                        "changes_by_source": by_source,
                    }
                    for k, v in totals.items():
                        etl_stats[f"changes:totals:{k}"] = v
                    for sid, sub in by_source.items():
                        for k, v in sub.items():
                            etl_stats[f"changes:by_source:{sid}:{k}"] = v
                except Exception:
                    pass

        actions = [
            "extract","extract-programs","transform","transform-programs","transform-normalize","dedup-programs","dq-report","load","all"
        ]
        # Prepare stats list for template rendering
        etl_stats_kv = sorted([(k, etl_stats[k]) for k in etl_stats.keys()]) if etl_stats else []
        return render(request, "admin/etl_process.html", {
            "action": action,
            "ran": ran,
            "processed_dir": str(cfg.processed_dir),
            "raw_dir": str(cfg.raw_dir),
            "inplace": inplace,
            "dry_run": dry_run,
            "messages": msg,
            "actions": actions,
            "config_value": config_path,
            "load_summary": load_summary,
            "etl_stats_kv": etl_stats_kv,
        })
    except Exception as e:
        actions = [
            "extract","extract-programs","transform","transform-programs","transform-normalize","dedup-programs","dq-report","load","all"
        ]
        return render(request, "admin/etl_process.html", {"error": str(e), "actions": actions})

# Dev-only: bypass CSRF for admin ETL endpoints when DEBUG or env toggle is set
# Never enable this in production.
if settings.DEBUG or os.getenv("DISABLE_CSRF_DEV", "").lower() in ("1", "true", "yes"):
    admin_etl_upload = csrf_exempt(admin_etl_upload)
    admin_etl_process = csrf_exempt(admin_etl_process)


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
