import csv
import math
from pathlib import Path

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError

from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response

from utils.drf_auth import optional_firebase_uid
from utils.errors import error_response

_CSV_CACHE: dict[str, dict] = {}

def _university_campuses_draft_path() -> Path:
    backend_dir = Path(__file__).resolve().parent.parent
    return backend_dir / "scripts" / "etl" / "kuccps" / "mappings" / "university_campuses_draft.csv"

def _file_sig(fp: Path) -> tuple:
    try:
        st = fp.stat()
        return (int(st.st_mtime), int(st.st_size))
    except Exception:
        return (0, 0)

def _etag_for(sig: tuple) -> str:
    return f'W/"{sig[0]}-{sig[1]}"'

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


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_suffix_mapping(request):
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
        return Response({
            "count": total,
            "page": page,
            "page_size": page_size,
            "results": items[start:end],
        })
    except Exception as e:
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=500, code='server_error')


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_program_costs(request):
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
        return Response({
            "count": total,
            "page": page,
            "page_size": page_size,
            "results": rows[start:end],
        })
    except Exception as e:
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=500, code='server_error')


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_catalog_status(_request):
    out: dict = {
        'status': 'ok',
        'database': {},
        'catalog': {
            'available': False,
            'counts': {},
        },
        'rag': {
            'enabled': bool(getattr(settings, 'RAG_USE_PGVECTOR', False)),
            'vectorstore_installed': 'vectorstore' in (getattr(settings, 'INSTALLED_APPS', []) or []),
            'gemini_api_key_present': bool((getattr(settings, 'GEMINI_API_KEY', '') or '').strip()),
            'embedding_model': (getattr(settings, 'GEMINI_EMBEDDING_MODEL', '') or '').strip(),
            'embeddings': {
                'available': False,
                'total': 0,
                'populated': 0,
            },
            'mode': 'lexical',
        },
        'etl': {
            'last_run': None,
        },
        'warnings': [],
    }

    try:
        db_conf = (getattr(settings, 'DATABASES', {}) or {}).get('default') or {}
        out['database'] = {
            'engine': str(db_conf.get('ENGINE') or ''),
            'name': str(db_conf.get('NAME') or ''),
        }
    except Exception:
        out['database'] = {}

    try:
        from catalog.models import Program, Institution, Field, Subject, YearlyCutoff, ProgramCost, ETLRun  # type: ignore
    except Exception:
        Program = None  # type: ignore
        Institution = None  # type: ignore
        Field = None  # type: ignore
        Subject = None  # type: ignore
        YearlyCutoff = None  # type: ignore
        ProgramCost = None  # type: ignore
        ETLRun = None  # type: ignore

    if Program is not None and Institution is not None:
        try:
            out['catalog']['available'] = True
            out['catalog']['counts'] = {
                'programs': int(Program.objects.count()),
                'institutions': int(Institution.objects.count()),
                'fields': int(Field.objects.count()) if Field is not None else 0,
                'subjects': int(Subject.objects.count()) if Subject is not None else 0,
                'yearly_cutoffs': int(YearlyCutoff.objects.count()) if YearlyCutoff is not None else 0,
                'program_costs': int(ProgramCost.objects.count()) if ProgramCost is not None else 0,
            }
        except (ProgrammingError, OperationalError) as e:
            detail = 'Database not initialized'
            if settings.DEBUG:
                detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
            out['catalog'] = {
                'available': False,
                'detail': detail,
                'counts': {},
            }
        except Exception as e:
            detail = 'Server error'
            if settings.DEBUG:
                detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
            out['catalog'] = {
                'available': False,
                'detail': detail,
                'counts': {},
            }

    if ETLRun is not None:
        try:
            r = ETLRun.objects.order_by('-created_at').first()
            if r:
                out['etl']['last_run'] = {
                    'action': getattr(r, 'action', ''),
                    'started_at': r.started_at.isoformat() if getattr(r, 'started_at', None) else None,
                    'finished_at': r.finished_at.isoformat() if getattr(r, 'finished_at', None) else None,
                    'created_at': r.created_at.isoformat() if getattr(r, 'created_at', None) else None,
                    'stats': getattr(r, 'stats', None) or {},
                }
        except Exception:
            out['etl']['last_run'] = None

    if out['rag'].get('vectorstore_installed'):
        try:
            from vectorstore.models import ProgramEmbedding  # type: ignore
            total = int(ProgramEmbedding.objects.count())
            populated = int(ProgramEmbedding.objects.filter(embedding__isnull=False).count())
            out['rag']['embeddings'] = {
                'available': True,
                'total': total,
                'populated': populated,
            }
        except (ProgrammingError, OperationalError):
            out['rag']['embeddings'] = {
                'available': False,
                'total': 0,
                'populated': 0,
            }
        except Exception:
            out['rag']['embeddings'] = {
                'available': False,
                'total': 0,
                'populated': 0,
            }

    try:
        emb_pop = int((out.get('rag') or {}).get('embeddings', {}).get('populated', 0) or 0)
        if out['rag'].get('enabled') and emb_pop > 0:
            out['rag']['mode'] = 'vector'
        else:
            out['rag']['mode'] = 'lexical'
    except Exception:
        out['rag']['mode'] = 'lexical'

    warnings: list[str] = []
    try:
        counts = (out.get('catalog') or {}).get('counts') or {}
        if out.get('catalog', {}).get('available') and (
            int(counts.get('programs', 0) or 0) == 0 or int(counts.get('institutions', 0) or 0) == 0
        ):
            warnings.append('catalog_db_empty')
    except Exception:
        pass

    try:
        if out.get('rag', {}).get('enabled') and out.get('rag', {}).get('mode') != 'vector':
            warnings.append('rag_lexical_only')
    except Exception:
        pass

    out['warnings'] = warnings
    return Response(out)


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_catalog_program_detail(request, program_id: int):
    try:
        try:
            from catalog.models import Program, YearlyCutoff, ProgramCost, ProgramRequirementGroup  # type: ignore
        except Exception:
            Program = None  # type: ignore
            YearlyCutoff = None  # type: ignore
            ProgramCost = None  # type: ignore
            ProgramRequirementGroup = None  # type: ignore

        if Program is None:
            return error_response("Catalog DB not available", status_code=503, code='catalog_unavailable')

        try:
            pid = int(program_id)
        except Exception:
            return error_response(
                "Invalid program id",
                status_code=400,
                code='validation_error',
                fields={'program_id': ['Invalid integer.']},
            )

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
            return error_response("Program not found", status_code=404, code='not_found')

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
            linked = list(p.costs.all().order_by("-updated_at"))
        except Exception:
            linked = []
        try:
            extra = []
            if ProgramCost is not None:
                code = (getattr(p, "code", "") or "").strip()
                if code:
                    extra = list(
                        ProgramCost.objects.filter(program_code=code, program__isnull=True)
                        .order_by("-updated_at")[:10]
                    )
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
            uid = optional_firebase_uid(request)
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
                                for grp in (req.get('groups', []) if isinstance(req, dict) else []):
                                    cols.append({'pick': int(grp.get('pick') or 1), 'options': (grp.get('options') or []), 'label': 'Group'})
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

                                    req = getattr(p, 'subject_requirements', None) or {}
                                    cols = []
                                    for item in (req.get('required', []) if isinstance(req, dict) else []):
                                        cols.append({'pick': 1, 'options': [item], 'label': 'Required'})
                                    for grp in (req.get('groups', []) if isinstance(req, dict) else []):
                                        cols.append({'pick': int(grp.get('pick') or 1), 'options': (grp.get('options') or []), 'label': 'Group'})
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
            pass

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
        return Response(out)
    except Exception as e:
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=500, code='server_error')


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def api_catalog_institution_detail(request, institution_code: str):
    try:
        try:
            from catalog.models import Institution, Program, InstitutionCampus  # type: ignore
        except Exception:
            Institution = None  # type: ignore
            Program = None  # type: ignore
            InstitutionCampus = None  # type: ignore

        if Institution is None or Program is None:
            return error_response("Catalog DB not available", status_code=503, code='catalog_unavailable')

        code = (institution_code or "").strip()
        if not code:
            return error_response(
                "Invalid institution code",
                status_code=400,
                code='validation_error',
                fields={'institution_code': ['This field is required.']},
            )

        try:
            inst = Institution.objects.get(code__iexact=code)
        except Institution.DoesNotExist:
            return error_response("Institution not found", status_code=404, code='not_found')

        campuses = []
        try:
            draft_fp = _university_campuses_draft_path()
            if draft_fp.exists():
                rows, _etag = _read_csv_cached(draft_fp)
                code_key = (inst.code or "").strip()
                draft_rows = [r for r in rows if (str(r.get("institution_code") or "").strip() == code_key)]
                for r in draft_rows:
                    campus_name = (r.get("campus") or "").strip()
                    town = (r.get("town") or "").strip()
                    county = (r.get("county") or "").strip()
                    region = (r.get("region") or "").strip()
                    source_url = (r.get("source_url") or "").strip()
                    is_main = (str(r.get("is_main") or "").strip().lower() in ("yes", "y", "1", "true"))

                    label_parts = [p for p in [campus_name, town, county, region, inst.name] if p]
                    seen = set()
                    deduped = []
                    for p in label_parts:
                        k = (p or "").strip().lower()
                        if not k or k in seen:
                            continue
                        seen.add(k)
                        deduped.append(p)
                    label = ", ".join(deduped)

                    campuses.append({
                        "campus": campus_name,
                        "town": town,
                        "county": county,
                        "region": region,
                        "is_main": bool(is_main),
                        "map_query": label,
                        "source_url": source_url,
                    })

                if campuses:
                    main_seen = False
                    for c in campuses:
                        if c.get("is_main") and not main_seen:
                            main_seen = True
                        elif c.get("is_main") and main_seen:
                            c["is_main"] = False
                    if not main_seen:
                        campuses[0]["is_main"] = True

            if not campuses and InstitutionCampus is not None:
                rows = list(InstitutionCampus.objects.filter(institution_id=inst.id).order_by("campus").values(
                    "campus", "town", "county", "region",
                ))
                for r in rows:
                    campus_name = (r.get("campus") or "").strip()
                    town = (r.get("town") or "").strip()
                    county = (r.get("county") or "").strip()
                    region = (r.get("region") or "").strip()

                    label_parts = [p for p in [campus_name, town, county, region, inst.name] if p]
                    seen = set()
                    deduped = []
                    for p in label_parts:
                        k = (p or "").strip().lower()
                        if not k or k in seen:
                            continue
                        seen.add(k)
                        deduped.append(p)
                    label = ", ".join(deduped)

                    score = 0
                    cn = campus_name.lower()
                    if cn:
                        if any(k in cn for k in ("main", "head", "central", "hq", "headquarters")):
                            score += 100
                    inst_county = (getattr(inst, "county", "") or "").strip().lower()
                    inst_region = (getattr(inst, "region", "") or "").strip().lower()
                    if inst_county and county and inst_county == county.lower():
                        score += 40
                    if inst_region and region and inst_region == region.lower():
                        score += 25
                    iname = (getattr(inst, "name", "") or "").strip().lower()
                    if iname and cn and iname in cn:
                        score += 10

                    campuses.append({
                        "campus": campus_name,
                        "town": town,
                        "county": county,
                        "region": region,
                        "is_main": False,
                        "map_query": label,
                        "_score": int(score),
                    })

                if campuses:
                    best_idx = 0
                    best_score = int(campuses[0].get("_score") or 0)
                    for i, c in enumerate(campuses):
                        sc = int(c.get("_score") or 0)
                        if sc > best_score:
                            best_score = sc
                            best_idx = i
                    campuses[best_idx]["is_main"] = True
                for c in campuses:
                    c.pop("_score", None)
        except Exception:
            campuses = []

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
            "campuses": campuses,
            "programs": programs,
            "programs_count": programs_total,
        }
        return Response(payload)
    except Exception as e:
        detail = "Server error"
        if settings.DEBUG:
            detail = f"{detail}: {e.__class__.__name__}: {str(e)}".strip()
        return error_response(detail, status_code=500, code='server_error')
