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
import subprocess
import base64
import json
import csv
import zipfile
from pathlib import Path
import sys
import math
from datetime import datetime
import firebase_admin
from firebase_admin import auth as fb_auth, credentials

_FIREBASE_INIT_ERROR: str = ''

def _ensure_firebase_initialized() -> bool:
    global _FIREBASE_INIT_ERROR
    if firebase_admin._apps:
        return True

    path = (os.getenv('FIREBASE_CREDENTIALS_JSON_PATH') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or '').strip()
    if path:
        try:
            cred = credentials.Certificate(path)
            firebase_admin.initialize_app(cred)
            _FIREBASE_INIT_ERROR = ''
            return True
        except Exception as e:
            _FIREBASE_INIT_ERROR = f"{e.__class__.__name__}: {str(e)}".strip()
            return False

    b64 = os.getenv('FIREBASE_CREDENTIALS_JSON_B64')
    if not b64:
        _FIREBASE_INIT_ERROR = 'Missing FIREBASE_CREDENTIALS_JSON_B64'
        return False
    try:
        data = json.loads(base64.b64decode(b64).decode('utf-8'))
        cred = credentials.Certificate(data)
        firebase_admin.initialize_app(cred)
        _FIREBASE_INIT_ERROR = ''
        return True
    except Exception as e:
        _FIREBASE_INIT_ERROR = f"{e.__class__.__name__}: {str(e)}".strip()
        return False

_ensure_firebase_initialized()

def health(_request):
    return JsonResponse({"status": "ok"})

def secure_ping(request):
    if not _ensure_firebase_initialized():
        detail = "Firebase admin not initialized"
        if _FIREBASE_INIT_ERROR:
            detail = f"{detail}: {_FIREBASE_INIT_ERROR}"
        return JsonResponse({"detail": detail}, status=503)
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

        if Program is not None:
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
                    qs = qs.filter(Q(region__icontains=region) | Q(institution__region__icontains=region) | Q(institution__county__icontains=region))
                else:
                    qs = qs.filter(region__icontains=region)
            if field:
                if Q is not None:
                    qs = qs.filter(Q(field__name__iexact=field) | Q(field__slug__iexact=field))
                else:
                    qs = qs.filter(field__name__iexact=field)

            total = int(qs.count())
            if total <= 0:
                return JsonResponse({
                    "count": 0,
                    "page": page,
                    "page_size": page_size,
                    "results": [],
                    "detail": "No programmes found in the catalog database. This usually means the database has not been populated yet. Run the ETL load step to populate Program data.",
                })

            start = (page - 1) * page_size
            end = start + page_size
            items = []
            for p in list(qs.order_by("normalized_name")[start:end]):
                inst = getattr(p, "institution", None)
                fld = getattr(p, "field", None)
                items.append({
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
                })
            return JsonResponse({
                "count": total,
                "page": page,
                "page_size": page_size,
                "results": items,
            })

        fp = _programs_path()
        if not fp.exists():
            return JsonResponse({
                "count": 0,
                "page": page,
                "page_size": page_size,
                "results": [],
                "detail": f"Programs file not found at {fp}",
            }, status=503)
        all_rows, etag = _read_programs_cached()
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
            if region and region not in region_val:
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


def api_catalog_institution_detail(request, institution_code: str):
    """GET /api/catalog/institutions/<code>
    Returns a DB-backed Institution detail payload including a list of programs offered.
    """
    if request.method != "GET":
        return HttpResponseBadRequest("GET required")
    try:
        try:
            from catalog.models import Institution, Program  # type: ignore
        except Exception:
            Institution = None  # type: ignore
            Program = None  # type: ignore

        if Institution is None or Program is None:
            return JsonResponse({"detail": "Catalog DB not available"}, status=503)

        code = (institution_code or "").strip()
        if not code:
            return JsonResponse({"detail": "Invalid institution code"}, status=400)

        try:
            inst = Institution.objects.get(code__iexact=code)
        except Institution.DoesNotExist:
            return JsonResponse({"detail": "Institution not found"}, status=404)

        prog_qs = Program.objects.select_related("field").filter(institution_id=inst.id).order_by("normalized_name")
        programs_total = int(prog_qs.count())
        programs_rows = list(prog_qs[:200].values(
            "id",
            "code",
            "name",
            "normalized_name",
            "level",
            "campus",
            "region",
            "field__name",
        ))
        programs = []
        for r in programs_rows:
            programs.append({
                "id": r.get("id"),
                "program_code": (r.get("code") or ""),
                "name": (r.get("name") or ""),
                "normalized_name": (r.get("normalized_name") or ""),
                "level": (r.get("level") or ""),
                "campus": (r.get("campus") or ""),
                "region": (r.get("region") or ""),
                "field_name": (r.get("field__name") or ""),
            })

        payload = {
            "code": inst.code,
            "name": inst.name,
            "alias": inst.alias,
            "region": inst.region,
            "county": inst.county,
            "website": inst.website,
            "metadata": inst.metadata or {},
            "programs": programs,
            "programs_count": programs_total,
        }
        return JsonResponse(payload)
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


def api_catalog_program_detail(request, program_id: int):
    """GET /api/catalog/programs/<id>
    Returns a DB-backed Program detail payload including institution info, costs, and all yearly cutoffs.
    """
    if request.method != "GET":
        return HttpResponseBadRequest("GET required")
    try:
        try:
            from catalog.models import Program, YearlyCutoff, ProgramCost, ProgramRequirementGroup  # type: ignore
        except Exception:
            Program = None  # type: ignore
            YearlyCutoff = None  # type: ignore
            ProgramCost = None  # type: ignore
            ProgramRequirementGroup = None  # type: ignore

        if Program is None:
            return JsonResponse({"detail": "Catalog DB not available"}, status=503)

        try:
            pid = int(program_id)
        except Exception:
            return JsonResponse({"detail": "Invalid program id"}, status=400)

        qs = Program.objects.select_related("institution", "field").prefetch_related(
            "cutoffs",
            "costs",
            "requirement_groups",
            "requirement_groups__options",
            "requirement_groups__options__subject",
        )
        try:
            p = qs.get(id=pid)
        except Program.DoesNotExist:
            return JsonResponse({"detail": "Program not found"}, status=404)

        inst = getattr(p, "institution", None)
        field = getattr(p, "field", None)

        cutoffs = []
        try:
            for c in list(p.cutoffs.all().order_by("-year")):
                cutoffs.append({
                    "year": int(getattr(c, "year", 0) or 0),
                    "cutoff": float(getattr(c, "cutoff", 0) or 0),
                    "capacity": getattr(c, "capacity", None),
                    "notes": (getattr(c, "notes", "") or ""),
                })
        except Exception:
            cutoffs = []

        costs = []
        try:
            # Include linked costs, plus any unlinked costs matched by program_code
            linked = list(p.costs.all().order_by("-updated_at"))
        except Exception:
            linked = []
        try:
            extra = []
            if ProgramCost is not None:
                code = (getattr(p, "code", "") or "").strip()
                if code:
                    extra = list(ProgramCost.objects.filter(program_code=code, program__isnull=True).order_by("-updated_at")[:10])
        except Exception:
            extra = []

        for pc in (linked + extra)[:20]:
            try:
                costs.append({
                    "amount": float(pc.amount) if getattr(pc, "amount", None) is not None else None,
                    "currency": (pc.currency or "") if hasattr(pc, "currency") else "",
                    "raw_cost": (pc.raw_cost or "") if hasattr(pc, "raw_cost") else "",
                    "source_id": (pc.source_id or "") if hasattr(pc, "source_id") else "",
                    "updated_at": pc.updated_at.isoformat() if getattr(pc, "updated_at", None) else "",
                })
            except Exception:
                continue

        requirement_groups = []
        try:
            for g in list(p.requirement_groups.all().order_by("order").prefetch_related("options", "options__subject")):
                opts = []
                try:
                    for opt in list(g.options.all().order_by("order")):
                        subj_code = ""
                        subj_name = ""
                        try:
                            if getattr(opt, "subject_id", None):
                                subj_code = (opt.subject.code or "").strip().upper()
                                subj_name = (opt.subject.name or "").strip()
                        except Exception:
                            subj_code = ""
                            subj_name = ""
                        if not subj_code:
                            subj_code = (getattr(opt, "subject_code", "") or "").strip().upper()
                        opts.append({
                            "subject_code": subj_code,
                            "subject_name": subj_name,
                            "min_grade": (getattr(opt, "min_grade", "") or "").strip(),
                            "order": int(getattr(opt, "order", 0) or 0),
                        })
                except Exception:
                    opts = []
                requirement_groups.append({
                    "name": (getattr(g, "name", "") or "").strip(),
                    "pick": int(getattr(g, "pick", 1) or 1),
                    "order": int(getattr(g, "order", 0) or 0),
                    "options": opts,
                })
        except Exception:
            requirement_groups = []

        req_preview = ""
        try:
            req_preview = p.requirements_preview() or ""
        except Exception:
            req_preview = ""

        estimated_cluster_points = None
        cluster_points_breakdown = None
        try:
            auth_header = request.META.get('HTTP_AUTHORIZATION', '')
            token = ''
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ', 1)[1]
            if token:
                decoded = fb_auth.verify_id_token(token)
                uid = decoded.get('uid')
                if uid:
                    try:
                        from accounts.models import UserProfile, OnboardingProfile  # type: ignore
                    except Exception:
                        UserProfile = None  # type: ignore
                        OnboardingProfile = None  # type: ignore
                    if UserProfile is not None and OnboardingProfile is not None:
                        up = UserProfile.objects.filter(uid=str(uid)).first()
                        ob = OnboardingProfile.objects.filter(user=up).first() if up else None
                        hs = (ob.high_school or {}) if ob else {}
                        grades = hs.get('subject_grades') or {}
                        if isinstance(grades, dict) and grades:
                            try:
                                from scripts.etl.kuccps.grades import normalize_grade, grade_points, meets_min_grade  # type: ignore
                            except Exception:
                                normalize_grade = None  # type: ignore
                                grade_points = None  # type: ignore
                                meets_min_grade = None  # type: ignore

                            def _norm_subj_key(s: str) -> str:
                                return str(s or '').strip().upper().replace(' ', '')

                            _SUBJ_ALIASES = {
                                "ENGLISH": "ENG",
                                "KISWAHILI": "KIS",
                                "MATHEMATICS": "MAT",
                                "MATH": "MAT",
                                "BIOLOGY": "BIO",
                                "CHEMISTRY": "CHE",
                                "PHYSICS": "PHY",
                                "HISTORY": "HIS",
                                "HISTORYANDGOVERNMENT": "HIS",
                                "GEOGRAPHY": "GEO",
                                "CRE": "CRE",
                                "CHRISTIANRELIGIOUSEDUCATION": "CRE",
                                "IRE": "IRE",
                                "ISLAMICRELIGIOUSEDUCATION": "IRE",
                                "BUSINESSSTUDIES": "BST",
                                "AGRICULTURE": "AGR",
                                "COMPUTERSTUDIES": "CSC",
                            }

                            def _to_code(s: str) -> str:
                                k = _norm_subj_key(s)
                                return _SUBJ_ALIASES.get(k, k)

                            pts_by_subj = {}
                            grade_by_subj = {}
                            for subj, raw_grade in grades.items():
                                k = _to_code(subj)
                                if not k:
                                    continue
                                g = str(raw_grade or '').strip().upper().replace(' ', '')
                                if normalize_grade:
                                    g = normalize_grade(g) or ''
                                if not g:
                                    continue
                                pval = None
                                if grade_points:
                                    try:
                                        pval = int(grade_points(g) or 0)
                                    except Exception:
                                        pval = 0
                                if pval and pval > 0:
                                    pts_by_subj[k] = int(pval)
                                    grade_by_subj[k] = g

                            if pts_by_subj:
                                sorted_pairs = sorted(pts_by_subj.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
                                top7_pairs = sorted_pairs[:7]
                                if len(top7_pairs) < 7:
                                    cluster_points_breakdown = {
                                        'reason': 'need_at_least_7_subjects',
                                        'subjects_provided': int(len(sorted_pairs)),
                                    }
                                else:
                                    t_sum = sum(int(v) for _k, v in top7_pairs)
                                    T = 84

                                    req = getattr(p, 'subject_requirements', None) or {}
                                    cols = []
                                    for item in (req.get('required', []) if isinstance(req, dict) else []):
                                        cols.append({'pick': 1, 'options': [item], 'label': 'Required'})
                                    for g in (req.get('groups', []) if isinstance(req, dict) else []):
                                        cols.append({'pick': int(g.get('pick') or 1), 'options': (g.get('options') or []), 'label': 'Group'})
                                    required_count = sum(max(1, int(c.get('pick') or 1)) for c in cols)
                                    if not cols:
                                        cluster_points_breakdown = {
                                            'reason': 'insufficient_program_subject_data',
                                            'required_count': 0,
                                            'expected_required_count': 4,
                                        }
                                    elif required_count > 4:
                                        cluster_points_breakdown = {
                                            'reason': 'insufficient_program_subject_data',
                                            'required_count': int(required_count),
                                            'expected_required_count': 4,
                                        }
                                    else:
                                        requirements_incomplete = required_count < 4

                                        chosen = []
                                        missing = []
                                        col_idx = 1
                                        for col in cols:
                                            pick = max(1, int(col.get('pick') or 1))
                                            options = col.get('options') if isinstance(col.get('options'), list) else []
                                            scored = []
                                            for opt in options:
                                                subj = _to_code((opt.get('subject') or opt.get('subject_code') or ''))
                                                if not subj:
                                                    continue
                                                if subj not in pts_by_subj:
                                                    continue
                                                min_g = str(opt.get('min_grade') or '').strip().upper().replace(' ', '')
                                                if min_g and meets_min_grade:
                                                    ok = meets_min_grade(grade_by_subj.get(subj, ''), min_g)
                                                    if ok is not True:
                                                        continue
                                                scored.append((subj, int(pts_by_subj[subj]), min_g))

                                            scored.sort(key=lambda x: (-x[1], x[0]))
                                            picked = []
                                            used = set(x['subject_code'] for x in chosen)
                                            for cand in scored:
                                                if cand[0] in used:
                                                    continue
                                                picked.append(cand)
                                                used.add(cand[0])
                                                if len(picked) >= pick:
                                                    break
                                            if len(picked) < pick:
                                                missing.append({'column': col_idx, 'needed': pick, 'found': len(picked)})
                                            for subj, pval, min_g in picked:
                                                chosen.append({
                                                    'column': col_idx,
                                                    'subject_code': subj,
                                                    'grade': grade_by_subj.get(subj, ''),
                                                    'points': int(pval),
                                                    'min_grade': min_g,
                                                })
                                            col_idx += 1

                                        if chosen and not missing:
                                            used_codes = set(x['subject_code'] for x in chosen)
                                            filled_subjects = []
                                            if len(chosen) < 4:
                                                for k, v in sorted_pairs:
                                                    if k in used_codes:
                                                        continue
                                                    chosen.append({
                                                        'column': 0,
                                                        'subject_code': k,
                                                        'grade': grade_by_subj.get(k, ''),
                                                        'points': int(v),
                                                        'min_grade': '',
                                                    })
                                                    filled_subjects.append(k)
                                                    used_codes.add(k)
                                                    if len(chosen) >= 4:
                                                        break

                                            if len(chosen) == 4:
                                                r_sum = sum(int(x['points']) for x in chosen)
                                                R = 48
                                                try:
                                                    est = math.sqrt((r_sum / R) * (t_sum / T)) * 48
                                                except Exception:
                                                    est = 0.0
                                                estimated_cluster_points = round(float(est), 3)
                                                cluster_points_breakdown = {
                                                    'requirements_incomplete': bool(requirements_incomplete),
                                                    'filled_subjects': filled_subjects,
                                                    't_sum': int(t_sum),
                                                    'T': int(T),
                                                    'r_sum': int(r_sum),
                                                    'R': int(R),
                                                    'selected_subjects': chosen,
                                                    'top7_subjects': [{'subject_code': k, 'points': int(v), 'grade': grade_by_subj.get(k, '')} for k, v in top7_pairs],
                                                }
                                            else:
                                                cluster_points_breakdown = {
                                                    'reason': 'need_at_least_4_cluster_subjects',
                                                    'selected_subjects': chosen,
                                                }
                                        elif chosen and missing:
                                            cluster_points_breakdown = {
                                                'reason': 'missing_required_subjects',
                                                'missing': missing,
                                                'selected_subjects': chosen,
                                            }
        except Exception:
            estimated_cluster_points = None
            cluster_points_breakdown = None

        out = {
            "id": int(getattr(p, "id", 0) or 0),
            "program_code": (getattr(p, "code", "") or "").strip(),
            "program_name": (getattr(p, "name", "") or "").strip(),
            "normalized_name": (getattr(p, "normalized_name", "") or "").strip(),
            "level": (getattr(p, "level", "") or "").strip(),
            "campus": (getattr(p, "campus", "") or "").strip(),
            "region": (getattr(p, "region", "") or "").strip(),
            "duration_years": float(p.duration_years) if getattr(p, "duration_years", None) is not None else None,
            "award": (getattr(p, "award", "") or "").strip(),
            "mode": (getattr(p, "mode", "") or "").strip(),
            "field_name": (field.name or "").strip() if getattr(p, "field_id", None) and field else "",
            "estimated_cluster_points": estimated_cluster_points,
            "cluster_points_breakdown": cluster_points_breakdown,
            "institution": {
                "code": (inst.code or "").strip() if inst else "",
                "name": (inst.name or "").strip() if inst else "",
                "region": (inst.region or "").strip() if inst else "",
                "county": (inst.county or "").strip() if inst else "",
                "website": (inst.website or "").strip() if inst else "",
            },
            "requirements_preview": req_preview,
            "requirement_groups": requirement_groups,
            "cutoffs": cutoffs,
            "costs": costs,
        }
        return JsonResponse(out)
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


def api_suffix_mapping(request):
    """GET /api/catalog/suffix-mapping
    Query params: q (search on course_suffix or normalized_name), active (0/1), page, page_size
    Returns DB-backed CourseSuffixMapping rows.
    """
    try:
        from catalog.models import CourseSuffixMapping  # type: ignore
        q = (request.GET.get("q") or "").strip().lower()
        active = request.GET.get("active")
        page = max(1, int(request.GET.get("page", 1)))
        page_size = max(1, min(50, int(request.GET.get("page_size", 20))))
        qs = CourseSuffixMapping.objects.all().order_by("course_suffix")
        if active in ("0", "1"):
            qs = qs.filter(is_active=(active == "1"))
        items = []
        for obj in qs:
            if q and not (
                q in (obj.course_suffix or "").lower() or q in (obj.normalized_name or "").lower()
            ):
                continue
            items.append({
                "course_suffix": obj.course_suffix,
                "normalized_name": obj.normalized_name,
                "field_name": obj.field_name,
                "is_active": obj.is_active,
                "updated_at": obj.updated_at.isoformat() if obj.updated_at else "",
            })
        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        return JsonResponse({
            "count": total,
            "page": page,
            "page_size": page_size,
            "results": items[start:end],
        })
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


def api_program_costs(request):
    """GET /api/catalog/program-costs
    Query params: program_code, q (search name/institution), page, page_size
    Returns DB-backed ProgramCost rows with linked Program context when available.
    """
    try:
        from catalog.models import ProgramCost, Program  # type: ignore
        q = (request.GET.get("q") or "").strip().lower()
        program_code = (request.GET.get("program_code") or "").strip()
        page = max(1, int(request.GET.get("page", 1)))
        page_size = max(1, min(50, int(request.GET.get("page_size", 20))))
        qs = ProgramCost.objects.all().order_by("-updated_at")
        if program_code:
            qs = qs.filter(program_code=program_code)
        rows = []
        for pc in qs:
            prog_norm = ""
            inst_code = ""
            if pc.program_id:
                try:
                    prog = Program.objects.only("normalized_name", "institution_id", "code").get(id=pc.program_id)
                    prog_norm = prog.normalized_name
                    inst_code = prog.institution.code if prog.institution_id else ""
                except Exception:
                    pass
            item = {
                "program_code": pc.program_code,
                "program_name": pc.program_name,
                "institution_name": pc.institution_name,
                "amount": float(pc.amount) if pc.amount is not None else None,
                "currency": pc.currency or "",
                "source_id": pc.source_id or "",
                "raw_cost": pc.raw_cost or "",
                "program_normalized_name": prog_norm,
                "institution_code": inst_code,
                "updated_at": pc.updated_at.isoformat() if pc.updated_at else "",
            }
            if q:
                hay = " ".join([
                    item.get("program_name") or "",
                    item.get("institution_name") or "",
                    item.get("program_normalized_name") or "",
                ]).lower()
                if q not in hay:
                    continue
            rows.append(item)
        total = len(rows)
        start = (page - 1) * page_size
        end = start + page_size
        return JsonResponse({
            "count": total,
            "page": page,
            "page_size": page_size,
            "results": rows[start:end],
        })
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


def api_institutions(request):
    """GET /api/etl/institutions
    Query params: q, region, county
    Reads processed/institutions.csv and returns filtered results.
    """
    try:
        q = (request.GET.get("q") or "").strip().lower()
        region = (request.GET.get("region") or "").strip()
        county = (request.GET.get("county") or "").strip()

        try:
            from catalog.models import Institution  # type: ignore
        except Exception:
            Institution = None  # type: ignore

        if Institution is None:
            return JsonResponse({
                "count": 0,
                "results": [],
                "detail": "Institution catalog database is not available.",
            }, status=503)

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
            return JsonResponse({
                "count": 0,
                "results": [],
                "detail": "No institutions found in the catalog database. This usually means the database has not been populated yet. Run the ETL load step to populate Institution data.",
            })

        rows = list(qs.order_by("name")[:500].values(
            "code", "name", "alias", "region", "county", "website",
        ))
        return JsonResponse({"count": total, "results": rows})
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
    etl_root = _processed_dir().parent
    base = etl_root / "raw" / "uploads"
    processed_dir = etl_root / "processed"
    if request.method == "GET":
        # Render an admin page with an upload form
        return render(request, "admin/etl_upload.html", {"upload_dir": str(base), "processed_dir": str(processed_dir)})
    try:
        ctx = {"upload_dir": str(base), "processed_dir": str(processed_dir)}
        if not request.FILES:
            ctx["error"] = "No file provided"
            return render(request, "admin/etl_upload.html", ctx)
        fobj = next(iter(request.FILES.values()))
        safe_name = os.path.basename(fobj.name).replace("..", "_")
        ext = os.path.splitext(safe_name)[1].lower()

        if ext == ".zip":
            processed_dir.mkdir(parents=True, exist_ok=True)
            dest = processed_dir / safe_name
            with open(dest, "wb") as out:
                for chunk in fobj.chunks():
                    out.write(chunk)
            extracted = 0
            with zipfile.ZipFile(dest, "r") as zf:
                for member in zf.infolist():
                    name = member.filename
                    if not name or name.endswith("/"):
                        continue
                    p = Path(name)
                    parts = list(p.parts)
                    if parts and parts[0].lower() == "processed":
                        parts = parts[1:]
                    if not parts:
                        continue
                    target = (processed_dir / Path(*parts)).resolve()
                    if processed_dir.resolve() not in target.parents and target != processed_dir.resolve():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted += 1
            ctx["stored"] = str(dest)
            ctx["extracted"] = extracted
            return render(request, "admin/etl_upload.html", ctx)

        base.mkdir(parents=True, exist_ok=True)
        dest = base / safe_name
        with open(dest, "wb") as out:
            for chunk in fobj.chunks():
                out.write(chunk)
        ctx["stored"] = str(dest)
        return render(request, "admin/etl_upload.html", ctx)
    except Exception as e:
        return render(request, "admin/etl_upload.html", {"upload_dir": str(base), "processed_dir": str(processed_dir), "error": str(e)})


@staff_member_required
@ensure_csrf_cookie
@csrf_protect
def admin_etl_process(request):
    """POST /admin/etl/process
    Optional body: {"action": "transform-normalize"}
    Invokes transform-normalize to regenerate processed CSVs using local config.
    """
    etl_dir = Path(__file__).resolve().parent.parent / "scripts" / "etl" / "kuccps"
    job_dir = etl_dir / "logs"
    job_dir.mkdir(parents=True, exist_ok=True)
    job_state_path = job_dir / "admin_etl_job.json"

    def _load_job_state():
        try:
            if job_state_path.exists():
                return json.loads(job_state_path.read_text(encoding="utf-8") or "{}") or {}
        except Exception:
            return {}
        return {}

    def _pid_running(pid: int) -> bool:
        try:
            if pid <= 0:
                return False
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def _tail_text(fp: Path, max_lines: int = 200) -> str:
        try:
            if not fp.exists():
                return ""
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            lines = txt.splitlines()
            return "\n".join(lines[-max_lines:])
        except Exception:
            return ""

    job_state = _load_job_state()
    job_pid = int(job_state.get("pid") or 0) if job_state else 0
    job_log_path = Path(job_state.get("log_path") or "") if job_state.get("log_path") else None
    job_running = _pid_running(job_pid) if job_pid else False
    job_log_tail = _tail_text(job_log_path) if job_log_path else ""

    full_actions = [
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

    prod_mode = os.getenv("ETL_PRODUCTION_MODE", "").strip().lower() in ("1", "true", "yes")
    allowed_env = (os.getenv("ETL_ALLOWED_ACTIONS", "") or "").strip()
    if allowed_env:
        allowed_actions = [a.strip() for a in allowed_env.split(",") if a.strip()]
    elif prod_mode:
        allowed_actions = ["load", "dq-report"]
    else:
        allowed_actions = full_actions
    actions = [a for a in full_actions if a in set(allowed_actions)]
    local_only_actions = [a for a in full_actions if a not in set(allowed_actions)]

    if request.method == "GET":
        return render(request, "admin/etl_process.html", {
            "actions": actions,
            "job_running": job_running,
            "job_pid": job_pid,
            "job_log_path": str(job_log_path) if job_log_path else "",
            "job_log_tail": job_log_tail,
            "prod_mode": prod_mode,
            "local_only_actions": local_only_actions,
        })
    try:
        # Allow both JSON body and form submission
        action = (request.POST.get("action") or "transform-normalize").strip()
        config_path = (request.POST.get("config") or "").strip()
        inplace = bool(request.POST.get("inplace"))
        dry_run = bool(request.POST.get("dry_run"))

        if action not in actions:
            msg = [f"Action '{action}' is not enabled on this server"]
            return render(request, "admin/etl_process.html", {
                "actions": actions,
                "action": action,
                "ran": False,
                "messages": msg,
                "config_value": config_path,
                "inplace": inplace,
                "dry_run": dry_run,
                "job_running": job_running,
                "job_pid": job_pid,
                "job_log_path": str(job_log_path) if job_log_path else "",
                "job_log_tail": job_log_tail,
                "prod_mode": prod_mode,
                "local_only_actions": local_only_actions,
            })

        async_actions_env = (os.getenv("ETL_ASYNC_ACTIONS", "") or "").strip()
        if async_actions_env:
            async_actions = [a.strip() for a in async_actions_env.split(",") if a.strip()]
        else:
            async_actions = ["all", "load", "dq-report"]

        if action in async_actions and os.getenv("ETL_RUN_ALL_ASYNC", "1").strip().lower() not in ("0", "false", "no"):
            msg = []
            if job_running:
                msg.append("ETL job already running")
                return render(request, "admin/etl_process.html", {
                    "actions": actions,
                    "action": action,
                    "ran": False,
                    "messages": msg,
                    "config_value": config_path,
                    "inplace": inplace,
                    "dry_run": dry_run,
                    "job_running": job_running,
                    "job_pid": job_pid,
                    "job_log_path": str(job_log_path) if job_log_path else "",
                    "job_log_tail": job_log_tail,
                    "prod_mode": prod_mode,
                    "local_only_actions": local_only_actions,
                })

            log_fp = job_dir / f"admin_etl_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
            cmd = [
                sys.executable,
                "-u",
                "manage.py",
                "kuccps_etl",
                "--action",
                action,
            ]
            if config_path:
                cmd.extend(["--config", config_path])
            if inplace:
                cmd.append("--inplace")
            if dry_run:
                cmd.append("--dry-run")

            with open(log_fp, "a", encoding="utf-8") as logf:
                p = subprocess.Popen(
                    cmd,
                    cwd=str(Path(__file__).resolve().parent.parent),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdout=logf,
                    stderr=logf,
                    start_new_session=True,
                )

            job_state = {
                "pid": p.pid,
                "log_path": str(log_fp),
                "started_at": datetime.utcnow().isoformat() + "Z",
                "action": action,
                "config": config_path,
                "dry_run": bool(dry_run),
                "inplace": bool(inplace),
            }
            try:
                job_state_path.write_text(json.dumps(job_state), encoding="utf-8")
            except Exception:
                pass

            msg.append(f"Started ETL job pid={p.pid}")
            return render(request, "admin/etl_process.html", {
                "actions": actions,
                "action": action,
                "ran": False,
                "messages": msg,
                "config_value": config_path,
                "inplace": inplace,
                "dry_run": dry_run,
                "job_running": True,
                "job_pid": p.pid,
                "job_log_path": str(log_fp),
                "job_log_tail": _tail_text(log_fp),
                "prod_mode": prod_mode,
                "local_only_actions": local_only_actions,
            })

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
            if os.getenv("ETL_SKIP_PDF_EXTRACT", "").strip().lower() not in ("1", "true", "yes"):
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
            "job_running": job_running,
            "job_pid": job_pid,
            "job_log_path": str(job_log_path) if job_log_path else "",
            "job_log_tail": job_log_tail,
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
    path('api/catalog/suffix-mapping', api_suffix_mapping, name='api_suffix_mapping'),
    path('api/catalog/program-costs', api_program_costs, name='api_program_costs'),
    path('api/catalog/programs/<int:program_id>', api_catalog_program_detail, name='api_catalog_program_detail'),
    path('api/catalog/institutions/<str:institution_code>', api_catalog_institution_detail, name='api_catalog_institution_detail'),
    path('admin/etl/upload', admin_etl_upload, name='admin_etl_upload'),
    path('admin/etl/process', admin_etl_process, name='admin_etl_process'),
    # Conversations API
    path('api/', include('conversations.urls')),
    path('admin/', admin.site.urls),
]
