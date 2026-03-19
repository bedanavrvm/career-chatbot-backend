import json
import os
import csv
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.db import transaction
from django.db.models import Count
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_protect

from utils.http import is_local_request

from .models import (
    OnetContentElement,
    OnetInterest,
    OnetOccupation,
    OnetOccupationSnapshot,
    OnetRelatedOccupation,
    OnetScale,
    OnetSkill,
    OnetTaskStatement,
)


def _parse_keywords(raw: str) -> list[str]:
    parts = re.split(r"[\s,|]+", (raw or '').strip())
    out = []
    for p in parts:
        p = (p or '').strip()
        if not p:
            continue
        if p not in out:
            out.append(p)
    return out


def _safe_count(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _db_table_exists(connection, table_name: str) -> bool:
    try:
        return table_name in set(connection.introspection.table_names())
    except Exception:
        return False


def _detect_onet_table_status() -> dict:
    from django.db import connection

    snapshot_count = _safe_count(lambda: int(OnetOccupationSnapshot.objects.count()), default=None)
    occupation_count = _safe_count(lambda: int(OnetOccupation.objects.count()), default=None)

    def _job_zone_count() -> int:
        with connection.cursor() as c:
            c.execute('SELECT COUNT(*) FROM job_zones')
            row = c.fetchone()
            return int(row[0]) if row else 0

    job_zone_count = _safe_count(_job_zone_count, default=None)

    core_tables = [
        'occupation_data',
        'content_model_reference',
        'scales_reference',
        'skills',
        'interests',
        'task_statements',
        'related_occupations',
    ]
    core_present = all(_db_table_exists(connection, t) for t in core_tables)
    full_tables = [
        'job_zones',
        'education_training_experience',
        'ete_categories',
    ]
    full_present = core_present and all(_db_table_exists(connection, t) for t in full_tables)

    return {
        'snapshot_count': snapshot_count,
        'occupation_count': occupation_count,
        'job_zone_count': job_zone_count,
        'core_tables_present': core_present,
        'full_tables_present': full_present,
    }


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]{1,}")

_DEFAULT_EXCLUDED_SOC_PREFIXES = {
    '55-',
    '53-',
    '51-',
}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _token_variants(token: str) -> list[str]:
    t = (token or '').strip().lower()
    if not t:
        return []
    out = [t]
    if len(t) >= 4 and t.endswith('s') and not t.endswith('ss'):
        out.append(t[:-1])
    return out


def _keywords_from_program_names(names_with_counts: list[tuple[str, int]], max_keywords: int) -> list[str]:
    stop = {
        'bachelor',
        'of',
        'in',
        'and',
        'with',
        'the',
        'for',
        'arts',
        'science',
        'technology',
        'studies',
        'management',
        'education',
        'business',
        'engineering',
        'medicine',
        'health',
        'programme',
        'program',
        'degree',
        'diploma',
        'certificate',
        'bsc',
        'ba',
        'beng',
        'btech',
        'bed',
        'llb',
        'general',
        'applied',
        'systems',
        'system',
        'development',
        'public',
        'community',
        'services',
        'service',
        'information',
        'communication',
        'communications',
        'media',
        'network',
        'networks',
        'planning',
        'resource',
        'resources',
        'policy',
        'leadership',
        'analysis',
        'analytics',
    }

    from collections import Counter

    c: Counter[str] = Counter()
    for name, freq in names_with_counts:
        toks = _tokens(name)
        for t in toks:
            if t in stop:
                continue
            if len(t) <= 2:
                continue
            c[t] += int(freq)

    return [k for k, _ in c.most_common(max_keywords)]


def _expand_field_keywords(field_slug: str, base_keywords: list[str]) -> list[str]:
    slug = (field_slug or '').strip().lower()
    slug_tokens = set(_tokens(slug.replace('-', ' ')))
    expansions = {
        'economics': [
            'economist',
            'economics',
            'econometric',
            'econometrics',
            'policy',
            'market',
            'markets',
            'trade',
            'statistics',
            'statistician',
            'analyst',
            'finance',
            'financial',
            'bank',
            'banker',
            'banking',
            'credit',
            'loan',
            'lending',
            'mortgage',
            'investment',
            'investments',
            'investor',
            'equity',
            'securities',
            'broker',
            'trader',
            'treasury',
            'budget',
            'budgeting',
            'accountant',
            'accounting',
            'audit',
            'auditor',
            'tax',
            'taxation',
            'insurance',
            'underwriter',
            'risk',
        ],
        'finance-accounting': ['accountant', 'accounting', 'auditor', 'tax', 'finance', 'financial', 'controller', 'bookkeeping'],
        'mathematics-statistics': ['mathematics', 'mathematical', 'statistician', 'statistics', 'data', 'analytics', 'quantitative', 'actuary'],
        'public-administration': ['government', 'policy', 'public', 'administration', 'governance', 'civil', 'regulation', 'planning'],
        'natural-resources': ['environment', 'environmental', 'conservation', 'forestry', 'wildlife', 'agriculture', 'resource', 'sustainability'],
        'geospatial-surveying': ['gis', 'geospatial', 'survey', 'surveying', 'cartography', 'mapping', 'remote', 'sensing', 'geomatics'],
        'pharmacy': ['pharmacist', 'pharmaceutical', 'drug', 'medication', 'clinical'],
        'medicine': ['medical', 'physician', 'doctor', 'clinical', 'health', 'hospital'],
        'dentistry': ['dentist', 'dental', 'oral'],
        'law': ['legal', 'lawyer', 'attorney', 'paralegal', 'judge', 'compliance'],
    }

    extra: list[str] = []
    if slug in expansions:
        extra.extend(expansions[slug])

    rules: list[tuple[set[str], list[str]]] = [
        ({'engineering', 'engineer'}, ['engineer', 'engineering', 'technician', 'technicians', 'technologist', 'technologists', 'design', 'maintenance']),
        ({'computer', 'computing', 'software', 'it', 'information', 'technology', 'cybersecurity', 'security'}, ['software', 'developer', 'developers', 'programmer', 'programmers', 'data', 'systems', 'network', 'networks', 'security', 'analyst', 'analytics', 'database', 'cloud']),
        ({'business', 'commerce', 'management', 'marketing', 'sales', 'entrepreneurship'}, ['business', 'manager', 'management', 'marketing', 'sales', 'operations', 'strategy', 'analyst', 'consultant', 'consulting', 'customer', 'customers']),
        ({'accounting', 'audit', 'auditing'}, ['accountant', 'accounting', 'audit', 'auditor', 'tax', 'taxation', 'bookkeeping', 'payroll', 'controller']),
        ({'finance', 'banking', 'investment'}, ['finance', 'financial', 'bank', 'banking', 'credit', 'loan', 'lending', 'investment', 'investments', 'securities', 'equity', 'portfolio', 'risk', 'treasury']),
        ({'education', 'teaching', 'teacher', 'teachers'}, ['education', 'teacher', 'teachers', 'instructor', 'instructors', 'tutor', 'tutors', 'curriculum', 'training', 'trainer', 'trainers']),
        ({'health', 'healthcare', 'medicine', 'medical', 'nursing', 'nurse', 'nurses'}, ['health', 'healthcare', 'medical', 'clinical', 'nurse', 'nursing', 'therapist', 'therapists', 'technologist', 'technologists', 'assistant', 'assistants']),
        ({'law', 'legal', 'justice', 'criminology'}, ['legal', 'law', 'lawyer', 'attorney', 'paralegal', 'compliance', 'investigator', 'investigators', 'court', 'courts', 'judge', 'judges']),
        ({'agriculture', 'agricultural', 'forestry', 'wildlife', 'natural', 'resources'}, ['agriculture', 'agricultural', 'farm', 'farming', 'forestry', 'wildlife', 'conservation', 'environment', 'environmental', 'resource', 'resources', 'sustainability']),
        ({'environment', 'environmental', 'climate', 'sustainability'}, ['environment', 'environmental', 'climate', 'sustainability', 'conservation', 'ecology', 'waste', 'water', 'pollution', 'energy']),
        ({'construction', 'building', 'architecture', 'surveying', 'civil'}, ['construction', 'building', 'architect', 'architectural', 'drafting', 'surveyor', 'surveying', 'civil', 'planning', 'inspector', 'inspectors']),
        ({'math', 'mathematics', 'statistics', 'statistical', 'data', 'analytics'}, ['mathematics', 'statistician', 'statistics', 'data', 'analytics', 'analyst', 'quantitative', 'model', 'modeling', 'research']),
    ]

    for triggers, words in rules:
        if slug_tokens & triggers:
            extra.extend(words)

    out = []
    for t in (extra + list(base_keywords or [])):
        tt = (t or '').strip().lower()
        if not tt:
            continue
        if tt not in out:
            out.append(tt)
    return out


def _score_occupations(occupations: list[dict], keywords: list[str]) -> dict[str, float]:
    kw: list[str] = []
    for k in keywords:
        ks = (str(k) if k is not None else '').strip().lower()
        if not ks:
            continue
        for vv in _token_variants(ks):
            if vv and vv not in kw:
                kw.append(vv)
    if not kw:
        return {}

    from collections import defaultdict

    scores: dict[str, float] = defaultdict(float)
    for occ in occupations:
        code = (occ.get('onetsoc_code') or '').strip()
        if not code:
            continue

        if any(code.startswith(pfx) for pfx in _DEFAULT_EXCLUDED_SOC_PREFIXES):
            continue

        title = (occ.get('title') or '').strip()
        desc = (occ.get('description') or '').strip()
        if not (title or desc):
            continue

        title_tokens_raw = _tokens(title)
        desc_tokens_raw = _tokens(desc)
        title_tokens = set(v for t in title_tokens_raw for v in _token_variants(t))
        desc_tokens = set(v for t in desc_tokens_raw for v in _token_variants(t))
        if not (title_tokens or desc_tokens):
            continue

        title_hits = sum(1 for k in kw if k in title_tokens)
        desc_hits = sum(1 for k in kw if k in desc_tokens)

        if title_hits <= 0 and desc_hits < 3:
            continue
        if title_hits < 2 and desc_hits < 3:
            continue
        s = (3.0 * float(title_hits)) + (1.0 * float(desc_hits))
        if s:
            scores[code] += s
    return dict(scores)


def _cleanup_old_logs(job_dir: Path) -> None:
    try:
        keep_days = int(os.getenv('ONET_LOG_RETENTION_DAYS', '14') or '14')
    except Exception:
        keep_days = 14
    try:
        keep_max = int(os.getenv('ONET_LOG_RETENTION_MAX_FILES', '200') or '200')
    except Exception:
        keep_max = 200

    try:
        if not job_dir.exists():
            return
        now = datetime.utcnow()
        cutoff = now - timedelta(days=max(0, keep_days))
        logs = sorted(job_dir.glob('admin_onet_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)

        for fp in logs[keep_max:]:
            try:
                fp.unlink(missing_ok=True)
            except Exception:
                pass

        for fp in logs[:keep_max]:
            try:
                ts = datetime.utcfromtimestamp(fp.stat().st_mtime)
                if ts < cutoff:
                    fp.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        return


def _pid_status(pid: int) -> str:
    try:
        if pid <= 0:
            return 'not_running'
        os.kill(pid, 0)
        return 'running'
    except Exception:
        return 'not_running'


def _tail_text(fp: Path, max_lines: int = 200) -> str:
    if not fp:
        return ''
    for _ in range(3):
        try:
            if not fp.exists():
                return ''
            data = fp.read_bytes()
            try:
                txt = data.decode('utf-8')
            except Exception:
                txt = data.decode('utf-8', errors='ignore')
            lines = txt.splitlines()
            return '\n'.join(lines[-max_lines:])
        except Exception:
            time.sleep(0.1)
            continue
    return ''


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_import_impl(request):
    backend_dir = Path(__file__).resolve().parent.parent
    onet_dir = backend_dir / 'onet'
    job_dir = onet_dir / 'logs'
    job_dir.mkdir(parents=True, exist_ok=True)
    job_state_path = job_dir / 'admin_onet_job.json'

    _cleanup_old_logs(job_dir)

    def _load_job_state():
        try:
            if job_state_path.exists():
                return json.loads(job_state_path.read_text(encoding='utf-8') or '{}') or {}
        except Exception:
            return {}
        return {}

    job_state = _load_job_state()
    job_pid = int(job_state.get('pid') or 0) if job_state else 0
    job_log_path = Path(job_state.get('log_path') or '') if job_state.get('log_path') else None
    job_status = _pid_status(job_pid) if job_pid else 'not_running'
    job_running = job_status == 'running'
    job_log_tail = _tail_text(job_log_path) if job_log_path else ''

    # Lightweight status counts (safe even if tables are not imported yet)
    status_counts = {}
    status_error = ''
    try:
        status_counts = {
            'occupation_data': int(OnetOccupation.objects.count()),
            'content_model_reference': int(OnetContentElement.objects.count()),
            'scales_reference': int(OnetScale.objects.count()),
            'interests': int(OnetInterest.objects.count()),
            'skills': int(OnetSkill.objects.count()),
            'task_statements': int(OnetTaskStatement.objects.count()),
            'related_occupations': int(OnetRelatedOccupation.objects.count()),
        }
    except Exception as e:
        status_error = str(e)

    # Allow import selection
    allowed_modes = ['core', 'all']

    if request.method == 'GET':
        return render(
            request,
            'admin/onet_import.html',
            {
                'modes': allowed_modes,
                'job_running': job_running,
                'job_pid': job_pid,
                'job_log_path': str(job_log_path) if job_log_path else '',
                'job_log_tail': job_log_tail,
                'sql_dir_default': str((backend_dir / 'onet' / 'db_30_2_mysql' / 'db_30_2_mysql').resolve()),
                'status_counts': status_counts,
                'status_error': status_error,
            },
        )

    try:
        mode = (request.POST.get('mode') or 'core').strip().lower()
        sql_dir = (request.POST.get('sql_dir') or '').strip()
        skip_create = bool(request.POST.get('skip_create'))

        msg = []
        if mode not in allowed_modes:
            msg.append(f"Mode '{mode}' is not supported")
            return render(
                request,
                'admin/onet_import.html',
                {
                    'modes': allowed_modes,
                    'mode': mode,
                    'ran': False,
                    'messages': msg,
                    'job_running': job_running,
                    'job_pid': job_pid,
                    'job_log_path': str(job_log_path) if job_log_path else '',
                    'job_log_tail': job_log_tail,
                    'sql_dir_default': sql_dir,
                    'skip_create': skip_create,
                    'status_counts': status_counts,
                    'status_error': status_error,
                },
            )

        if job_running:
            msg.append('O*NET import already running')
            return render(
                request,
                'admin/onet_import.html',
                {
                    'modes': allowed_modes,
                    'mode': mode,
                    'ran': False,
                    'messages': msg,
                    'job_running': True,
                    'job_pid': job_pid,
                    'job_log_path': str(job_log_path) if job_log_path else '',
                    'job_log_tail': job_log_tail,
                    'sql_dir_default': sql_dir,
                    'skip_create': skip_create,
                    'status_counts': status_counts,
                    'status_error': status_error,
                },
            )

        log_fp = job_dir / f"admin_onet_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
        cmd = [
            sys.executable,
            '-u',
            'manage.py',
            'import_onet',
            '--tables',
            mode,
        ]
        if sql_dir:
            cmd.extend(['--sql-dir', sql_dir])
        if skip_create:
            cmd.append('--skip-create')

        backend_cwd = str(backend_dir)
        proc_env = {**os.environ, 'PYTHONUNBUFFERED': '1'}

        with open(log_fp, 'a', encoding='utf-8') as logf:
            logf.write(f"[{datetime.utcnow().isoformat()}Z] admin_onet: starting\n")
            logf.write(f"[{datetime.utcnow().isoformat()}Z] admin_onet: mode={mode} skip_create={bool(skip_create)}\n")
            logf.write(f"[{datetime.utcnow().isoformat()}Z] admin_onet: cwd={backend_cwd}\n")
            logf.write(f"[{datetime.utcnow().isoformat()}Z] admin_onet: cmd={' '.join(cmd)}\n")
            logf.flush()

            creationflags = 0
            popen_kwargs = {}
            # On Windows, make the job independent from the runserver console.
            # This prevents Ctrl+C / autoreloader restarts from sending KeyboardInterrupt
            # into the import process.
            if os.name == 'nt':
                try:
                    creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
                except Exception:
                    pass
                try:
                    creationflags |= subprocess.DETACHED_PROCESS
                except Exception:
                    pass
                popen_kwargs['stdin'] = subprocess.DEVNULL

            p = subprocess.Popen(
                cmd,
                cwd=backend_cwd,
                env=proc_env,
                stdout=logf,
                stderr=logf,
                start_new_session=True,
                creationflags=creationflags,
                **popen_kwargs,
            )
            logf.write(f"[{datetime.utcnow().isoformat()}Z] admin_onet: spawned pid={p.pid}\n")
            logf.flush()

        job_state = {
            'pid': p.pid,
            'log_path': str(log_fp),
            'started_at': datetime.utcnow().isoformat() + 'Z',
            'mode': mode,
            'sql_dir': sql_dir,
            'skip_create': bool(skip_create),
        }
        try:
            job_state_path.write_text(json.dumps(job_state), encoding='utf-8')
        except Exception:
            pass

        msg.append(f'Started O*NET import pid={p.pid}')
        return render(
            request,
            'admin/onet_import.html',
            {
                'modes': allowed_modes,
                'mode': mode,
                'ran': False,
                'messages': msg,
                'job_running': True,
                'job_pid': p.pid,
                'job_log_path': str(log_fp),
                'job_log_tail': _tail_text(log_fp),
                'sql_dir_default': sql_dir,
                'skip_create': skip_create,
                'status_counts': status_counts,
                'status_error': status_error,
            },
        )
    except Exception as e:
        return render(
            request,
            'admin/onet_import.html',
            {
                'modes': allowed_modes,
                'job_running': job_running,
                'job_pid': job_pid,
                'job_log_path': str(job_log_path) if job_log_path else '',
                'job_log_tail': job_log_tail,
                'error': str(e),
                'status_counts': status_counts,
                'status_error': status_error,
            },
        )


@staff_member_required
@ensure_csrf_cookie
def admin_onet_import(request):
    # Same safety stance as ETL admin views: in production, still allow staff.
    # In DEBUG + local requests, we can bypass csrf_protect convenience wrapper.
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_import_impl(request)
    return csrf_protect(_admin_onet_import_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_mapping_suggest_impl(request):
    try:
        from catalog.models import Field, Program  # type: ignore
    except Exception:
        Field = None  # type: ignore
        Program = None  # type: ignore

    if Field is None or Program is None:
        return render(request, 'admin/onet_mapping_suggest.html', {'error': 'Catalog app not available'})

    defaults = {
        'top_k_programs': 30,
        'max_keywords': 25,
        'top_n_occupations': 10,
        'min_score': 6.0,
        'lenient': False,
    }

    if request.method == 'GET':
        return render(request, 'admin/onet_mapping_suggest.html', {**defaults})

    try:
        def _as_int(v, default):
            try:
                return int(str(v).strip())
            except Exception:
                return int(default)

        def _as_float(v, default):
            try:
                return float(str(v).strip())
            except Exception:
                return float(default)

        top_k_programs = _as_int(request.POST.get('top_k_programs'), defaults['top_k_programs'])
        max_keywords = _as_int(request.POST.get('max_keywords'), defaults['max_keywords'])
        top_n_occupations = _as_int(request.POST.get('top_n_occupations'), defaults['top_n_occupations'])
        min_score = _as_float(request.POST.get('min_score'), defaults['min_score'])
        lenient = str(request.POST.get('lenient') or '').strip().lower() in {'1', 'true', 'on', 'yes'}
        field_slugs_raw = (request.POST.get('field_slugs') or '').strip()
        field_slugs = [s.strip().lower() for s in field_slugs_raw.split(',') if s.strip()]

        fields_qs = Field.objects.all().order_by('name')
        if field_slugs:
            fields_qs = fields_qs.filter(slug__in=field_slugs)

        occs = []
        occ_err = None
        try:
            occs = list(OnetOccupation.objects.all().values('onetsoc_code', 'title', 'description'))
        except Exception as e:
            occ_err = e
            try:
                occs = list(OnetOccupationSnapshot.objects.all().values('onetsoc_code', 'title', 'description'))
            except Exception:
                occs = []

        if not occs:
            msg = (
                'No O*NET occupation data is available in the current database. '
                'Either import O*NET (Admin → O*NET → Import) or load a Snapshot (Admin → O*NET → Snapshot → Import). '
                f'Details: {occ_err}'
            )
            return render(request, 'admin/onet_mapping_suggest.html', {**defaults, 'error': msg})

        title_by_code = {o['onetsoc_code']: (o.get('title') or '') for o in occs if o.get('onetsoc_code')}

        from collections import Counter

        resp = HttpResponse(content_type='text/csv')
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        resp['Content-Disposition'] = f'attachment; filename=onet_field_mapping_suggestions_{ts}.csv'

        w = csv.writer(resp)
        w.writerow([
            'field_slug',
            'field_name',
            'occupation_code',
            'occupation_title',
            'score',
            'keywords',
            'sample_programs',
        ])

        for fld in fields_qs:
            prog_names = list(
                Program.objects.filter(field=fld)
                .exclude(normalized_name='')
                .values_list('normalized_name', flat=True)
            )
            if not prog_names:
                continue

            counts = Counter(prog_names)
            top_names = counts.most_common(max(1, top_k_programs))
            sample_programs = ' | '.join([n for n, _ in top_names[:10]])

            kws = _keywords_from_program_names(top_names, max_keywords=max(1, max_keywords))
            for t in _tokens(getattr(fld, 'name', '') or ''):
                if t and t not in kws:
                    kws.insert(0, t)

            scores = _score_occupations(occs, kws)
            if not scores:
                continue

            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            ranked_filtered = [(code, sc) for code, sc in ranked if sc >= min_score][:max(1, top_n_occupations)]
            if not ranked_filtered and lenient:
                ranked_filtered = ranked[:max(1, top_n_occupations)]
            ranked = ranked_filtered
            if not ranked:
                continue

            for code, sc in ranked:
                title = (title_by_code.get(code, '') or '').strip()
                if fld.slug != 'education':
                    tt = set(_tokens(title))
                    if 'teacher' in tt or 'teachers' in tt:
                        continue
                w.writerow([
                    fld.slug,
                    fld.name,
                    code,
                    title,
                    f'{sc:.3f}',
                    ' '.join(kws[:max(1, max_keywords)]),
                    sample_programs,
                ])

        return resp
    except Exception as e:
        return render(request, 'admin/onet_mapping_suggest.html', {**defaults, 'error': str(e)})


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_mapping_manual_impl(request):
    try:
        from catalog.models import Field  # type: ignore
    except Exception:
        Field = None  # type: ignore

    try:
        from .mapping_models import OnetFieldOccupationMapping  # type: ignore
    except Exception:
        OnetFieldOccupationMapping = None  # type: ignore

    if Field is None or OnetFieldOccupationMapping is None:
        return render(request, 'admin/onet_mapping_manual.html', {'error': 'Required apps/models not available'})

    field_slug = (request.GET.get('field_slug') or request.POST.get('field_slug') or '').strip().lower()
    q = (request.GET.get('q') or request.POST.get('q') or '').strip()
    keywords_raw = (request.GET.get('keywords') or request.POST.get('keywords') or '').strip()
    keywords = _parse_keywords(keywords_raw)
    source = (request.GET.get('source') or request.POST.get('source') or 'auto').strip().lower()
    smart = str(request.GET.get('smart') or request.POST.get('smart') or '').strip().lower() in {'1', 'true', 'on', 'yes'}
    include_desc = str(request.GET.get('include_desc') or request.POST.get('include_desc') or '').strip().lower() in {'1', 'true', 'on', 'yes'}
    min_score_raw = (request.GET.get('min_score') or request.POST.get('min_score') or '').strip()
    try:
        min_score = float(min_score_raw) if min_score_raw else 6.0
    except Exception:
        min_score = 6.0
    job_zone_raw = (request.GET.get('job_zone') or request.POST.get('job_zone') or '').strip()
    offset_raw = (request.GET.get('offset') or request.POST.get('offset') or '').strip()
    limit_raw = (request.GET.get('limit') or request.POST.get('limit') or '').strip()
    try:
        offset = max(0, int(offset_raw)) if offset_raw else 0
    except Exception:
        offset = 0
    try:
        limit = max(1, min(200, int(limit_raw))) if limit_raw else 50
    except Exception:
        limit = 50

    try:
        job_zone = int(job_zone_raw) if job_zone_raw else None
        if job_zone is not None and job_zone not in {1, 2, 3, 4, 5}:
            job_zone = None
    except Exception:
        job_zone = None

    fields = list(Field.objects.all().order_by('name').values('slug', 'name'))
    fld = Field.objects.filter(slug__iexact=field_slug).first() if field_slug else None

    existing_codes = set()
    if fld:
        existing_codes = set(OnetFieldOccupationMapping.objects.filter(field=fld).values_list('occupation_code', flat=True))

    occs = []
    occ_source = ''
    occ_scores = {}
    smart_keywords_used = []
    smart_note = ''

    prev_offset = max(0, int(offset) - int(limit))
    next_offset = int(offset) + int(limit)

    def _apply_occ_filters(qs, *, supports_job_zone: bool):
        if q:
            qs = qs.filter(title__icontains=q)
        for kw in keywords:
            qs = qs.filter(title__icontains=kw)
        if supports_job_zone and job_zone is not None:
            qs = qs.filter(job_zone=job_zone)
        return qs

    def _apply_occ_filters_onet(qs):
        if q:
            qs = qs.filter(title__icontains=q)
        for kw in keywords:
            qs = qs.filter(title__icontains=kw)
        if job_zone is not None:
            try:
                from .models import OnetJobZone  # type: ignore
                codes = list(
                    OnetJobZone.objects.filter(job_zone_id=job_zone)
                    .values_list('onetsoc_code_id', flat=True)[:5000]
                )
                if codes:
                    qs = qs.filter(onetsoc_code__in=codes)
            except Exception:
                pass
        return qs

    preferred_sources = []
    if source == 'snapshot':
        preferred_sources = ['snapshot']
    elif source == 'onet':
        preferred_sources = ['onet']
    else:
        preferred_sources = ['snapshot', 'onet']

    for src in preferred_sources:
        if src == 'snapshot':
            try:
                if smart and fld:
                    qs_all = OnetOccupationSnapshot.objects.all()
                    if job_zone is not None:
                        qs_all = qs_all.filter(job_zone=job_zone)
                    occs_all = list(qs_all.values('onetsoc_code', 'title', 'description'))

                    stop = {
                        'and', 'or', 'of', 'in', 'for', 'with', 'to', 'the',
                        'studies', 'science', 'arts', 'technology', 'management',
                    }
                    derived = [t for t in _tokens(getattr(fld, 'name', '') or '') if t not in stop and len(t) > 2]
                    smart_keywords = []

                    try:
                        from catalog.models import Program  # type: ignore
                    except Exception:
                        Program = None  # type: ignore

                    prog_kws: list[str] = []
                    if Program is not None:
                        try:
                            from collections import Counter
                            prog_names = list(
                                Program.objects.filter(field=fld)
                                .exclude(normalized_name='')
                                .values_list('normalized_name', flat=True)[:5000]
                            )
                            counts = Counter(prog_names)
                            top_names = counts.most_common(40)
                            prog_kws = _keywords_from_program_names(top_names, max_keywords=25)
                        except Exception:
                            prog_kws = []

                    merged = _expand_field_keywords(getattr(fld, 'slug', '') or '', derived + prog_kws + keywords)
                    for t in merged:
                        if t and t not in smart_keywords:
                            smart_keywords.append(t)

                    smart_keywords_used = list(smart_keywords)

                    if q:
                        for t in _tokens(q):
                            if t and t not in smart_keywords:
                                smart_keywords.insert(0, t)

                    smart_keywords_used = list(smart_keywords)

                    if not include_desc:
                        occs_for_score = [{'onetsoc_code': o.get('onetsoc_code'), 'title': o.get('title'), 'description': ''} for o in occs_all]
                    else:
                        occs_for_score = occs_all

                    scores = _score_occupations(occs_for_score, smart_keywords)
                    occ_scores = scores
                    ranked = sorted(
                        occs_all,
                        key=lambda o: (-(scores.get((o.get('onetsoc_code') or '').strip(), 0.0)), (o.get('title') or '')),
                    )

                    ranked_nonzero = [o for o in ranked if scores.get((o.get('onetsoc_code') or '').strip(), 0.0) >= float(min_score)]
                    if ranked_nonzero:
                        occs = ranked_nonzero[offset:offset + limit]
                    else:
                        smart_note = f'No results met min_score={min_score:g}; showing top ranked results instead.'
                        occs = ranked[offset:offset + limit]
                    occ_source = 'snapshot'
                else:
                    qs = OnetOccupationSnapshot.objects.all()
                    qs = _apply_occ_filters(qs, supports_job_zone=True)
                    occs = list(
                        qs.order_by('title')
                        .values('onetsoc_code', 'title', 'description')
                        [offset:offset + limit]
                    )
                    occ_source = 'snapshot'
            except Exception:
                occs = []
            if occs:
                break
        if src == 'onet':
            try:
                if smart and fld:
                    qs_all = OnetOccupation.objects.all()
                    qs_all = _apply_occ_filters_onet(qs_all) if (q or keywords or job_zone is not None) else qs_all
                    occs_all = list(qs_all.values('onetsoc_code', 'title', 'description'))

                    stop = {
                        'and', 'or', 'of', 'in', 'for', 'with', 'to', 'the',
                        'studies', 'science', 'arts', 'technology', 'management',
                    }
                    derived = [t for t in _tokens(getattr(fld, 'name', '') or '') if t not in stop and len(t) > 2]
                    smart_keywords = []

                    try:
                        from catalog.models import Program  # type: ignore
                    except Exception:
                        Program = None  # type: ignore

                    prog_kws: list[str] = []
                    if Program is not None:
                        try:
                            from collections import Counter
                            prog_names = list(
                                Program.objects.filter(field=fld)
                                .exclude(normalized_name='')
                                .values_list('normalized_name', flat=True)[:5000]
                            )
                            counts = Counter(prog_names)
                            top_names = counts.most_common(40)
                            prog_kws = _keywords_from_program_names(top_names, max_keywords=25)
                        except Exception:
                            prog_kws = []

                    merged = _expand_field_keywords(getattr(fld, 'slug', '') or '', derived + prog_kws + keywords)
                    for t in merged:
                        if t and t not in smart_keywords:
                            smart_keywords.append(t)

                    smart_keywords_used = list(smart_keywords)

                    if q:
                        for t in _tokens(q):
                            if t and t not in smart_keywords:
                                smart_keywords.insert(0, t)

                    smart_keywords_used = list(smart_keywords)

                    if not include_desc:
                        occs_for_score = [{'onetsoc_code': o.get('onetsoc_code'), 'title': o.get('title'), 'description': ''} for o in occs_all]
                    else:
                        occs_for_score = occs_all

                    scores = _score_occupations(occs_for_score, smart_keywords)
                    occ_scores = scores
                    ranked = sorted(
                        occs_all,
                        key=lambda o: (-(scores.get((o.get('onetsoc_code') or '').strip(), 0.0)), (o.get('title') or '')),
                    )

                    ranked_nonzero = [o for o in ranked if scores.get((o.get('onetsoc_code') or '').strip(), 0.0) >= float(min_score)]
                    if ranked_nonzero:
                        occs = ranked_nonzero[offset:offset + limit]
                    else:
                        smart_note = f'No results met min_score={min_score:g}; showing top ranked results instead.'
                        occs = ranked[offset:offset + limit]
                    occ_source = 'onet'
                else:
                    qs = OnetOccupation.objects.all()
                    qs = _apply_occ_filters_onet(qs)
                    occs = list(
                        qs.order_by('title')
                        .values('onetsoc_code', 'title', 'description')
                        [offset:offset + limit]
                    )
                    occ_source = 'onet'
            except Exception:
                occs = []
            if occs:
                break

    if smart and occ_scores and occs:
        try:
            for o in occs:
                code = (o.get('onetsoc_code') or '').strip()
                if not code:
                    continue
                o['score'] = float(occ_scores.get(code, 0.0) or 0.0)
        except Exception:
            pass

    if request.method == 'GET':
        return render(
            request,
            'admin/onet_mapping_manual.html',
            {
                'fields': fields,
                'field_slug': field_slug,
                'field_name': getattr(fld, 'name', '') if fld else '',
                'q': q,
                'keywords': keywords_raw,
                'source': source,
                'smart': smart,
                'include_desc': include_desc,
                'min_score': min_score,
                'job_zone': job_zone,
                'offset': offset,
                'prev_offset': prev_offset,
                'next_offset': next_offset,
                'limit': limit,
                'occupations': occs,
                'occ_scores': occ_scores,
                'smart_keywords_used': ' '.join(smart_keywords_used),
                'smart_note': smart_note,
                'existing_codes': existing_codes,
                'occ_source': occ_source,
            },
        )

    try:
        if not fld:
            return render(
                request,
                'admin/onet_mapping_manual.html',
                {
                    'fields': fields,
                    'field_slug': field_slug,
                    'q': q,
                    'keywords': keywords_raw,
                    'source': source,
                    'smart': smart,
                    'include_desc': include_desc,
                    'min_score': min_score,
                    'job_zone': job_zone,
                    'offset': offset,
                    'prev_offset': prev_offset,
                    'next_offset': next_offset,
                    'limit': limit,
                    'occupations': occs,
                    'occ_scores': occ_scores,
                    'smart_keywords_used': ' '.join(smart_keywords_used),
                    'smart_note': smart_note,
                    'existing_codes': existing_codes,
                    'occ_source': occ_source,
                    'error': 'Choose a Field first',
                },
            )

        codes = [c.strip() for c in request.POST.getlist('occupation_code') if str(c).strip()]
        if not codes:
            return render(
                request,
                'admin/onet_mapping_manual.html',
                {
                    'fields': fields,
                    'field_slug': field_slug,
                    'field_name': getattr(fld, 'name', ''),
                    'q': q,
                    'keywords': keywords_raw,
                    'source': source,
                    'smart': smart,
                    'include_desc': include_desc,
                    'min_score': min_score,
                    'job_zone': job_zone,
                    'offset': offset,
                    'prev_offset': prev_offset,
                    'next_offset': next_offset,
                    'limit': limit,
                    'occupations': occs,
                    'occ_scores': occ_scores,
                    'smart_keywords_used': ' '.join(smart_keywords_used),
                    'smart_note': smart_note,
                    'existing_codes': existing_codes,
                    'occ_source': occ_source,
                    'error': 'Select at least one occupation',
                },
            )

        replace_existing = str(request.POST.get('replace_existing') or '').strip().lower() in {'1', 'true', 'on', 'yes'}
        weight_raw = (request.POST.get('weight') or '').strip()
        notes = (request.POST.get('notes') or '').strip()
        weight = None
        if weight_raw:
            try:
                weight = float(weight_raw)
            except Exception:
                weight = None

        with transaction.atomic():
            if replace_existing:
                OnetFieldOccupationMapping.objects.filter(field=fld).delete()

            wrote = 0
            for code in sorted(set(codes)):
                obj, _ = OnetFieldOccupationMapping.objects.get_or_create(field=fld, occupation_code=code)
                if weight is not None:
                    try:
                        obj.weight = weight
                    except Exception:
                        pass
                if notes:
                    obj.notes = notes
                obj.save()
                wrote += 1

        existing_codes2 = set(OnetFieldOccupationMapping.objects.filter(field=fld).values_list('occupation_code', flat=True))

        return render(
            request,
            'admin/onet_mapping_manual.html',
            {
                'fields': fields,
                'field_slug': field_slug,
                'field_name': getattr(fld, 'name', ''),
                'q': q,
                'keywords': keywords_raw,
                'source': source,
                'smart': smart,
                'include_desc': include_desc,
                'min_score': min_score,
                'job_zone': job_zone,
                'offset': offset,
                'prev_offset': prev_offset,
                'next_offset': next_offset,
                'limit': limit,
                'occupations': occs,
                'occ_scores': occ_scores,
                'smart_keywords_used': ' '.join(smart_keywords_used),
                'smart_note': smart_note,
                'existing_codes': existing_codes2,
                'occ_source': occ_source,
                'ok': f'Wrote {wrote} mappings for field {fld.slug}',
                'replace_existing': replace_existing,
                'weight': weight_raw,
                'notes': notes,
            },
        )
    except Exception as e:
        return render(
            request,
            'admin/onet_mapping_manual.html',
            {
                'fields': fields,
                'field_slug': field_slug,
                'field_name': getattr(fld, 'name', '') if fld else '',
                'q': q,
                'keywords': keywords_raw,
                'source': source,
                'smart': smart,
                'include_desc': include_desc,
                'min_score': min_score,
                'job_zone': job_zone,
                'offset': offset,
                'prev_offset': prev_offset,
                'next_offset': next_offset,
                'limit': limit,
                'occupations': occs,
                'occ_scores': occ_scores,
                'smart_keywords_used': ' '.join(smart_keywords_used),
                'smart_note': smart_note,
                'existing_codes': existing_codes,
                'occ_source': occ_source,
                'error': str(e),
            },
        )


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_dashboard_impl(request):
    backend_dir = Path(__file__).resolve().parent.parent
    job_dir = backend_dir / 'onet' / 'logs'
    job_state_path = job_dir / 'admin_onet_job.json'

    job_state = {}
    try:
        if job_state_path.exists():
            job_state = json.loads(job_state_path.read_text(encoding='utf-8') or '{}') or {}
    except Exception:
        job_state = {}

    job_pid = int(job_state.get('pid') or 0) if job_state else 0
    job_log_path = Path(job_state.get('log_path') or '') if job_state.get('log_path') else None
    job_status = _pid_status(job_pid) if job_pid else 'not_running'

    status = _detect_onet_table_status()
    snapshot_count = status.get('snapshot_count')
    occupation_count = status.get('occupation_count')

    return render(
        request,
        'admin/onet_dashboard.html',
        {
            'job_status': job_status,
            'job_pid': job_pid,
            'job_log_path': str(job_log_path) if job_log_path else '',
            'job_log_tail': _tail_text(job_log_path) if job_log_path else '',
            'snapshot_count': snapshot_count,
            'occupation_count': occupation_count,
            'core_tables_present': bool(status.get('core_tables_present')),
            'full_tables_present': bool(status.get('full_tables_present')),
        },
    )


@staff_member_required
@ensure_csrf_cookie
def admin_onet_dashboard(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_dashboard_impl(request)
    return csrf_protect(_admin_onet_dashboard_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def admin_onet_mapping_manual(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_mapping_manual_impl(request)
    return csrf_protect(_admin_onet_mapping_manual_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def admin_onet_mapping_suggest(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_mapping_suggest_impl(request)
    return csrf_protect(_admin_onet_mapping_suggest_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_mapping_import_impl(request):
    defaults = {
        'replace_existing': False,
        'create_missing_fields': False,
    }

    if request.method == 'GET':
        return render(request, 'admin/onet_mapping_import.html', {**defaults})

    try:
        from catalog.models import Field  # type: ignore
    except Exception:
        Field = None  # type: ignore

    if Field is None:
        return render(request, 'admin/onet_mapping_import.html', {**defaults, 'error': 'Catalog app not available'})

    try:
        from .mapping_models import OnetFieldOccupationMapping  # type: ignore
    except Exception:
        OnetFieldOccupationMapping = None  # type: ignore

    if OnetFieldOccupationMapping is None:
        return render(request, 'admin/onet_mapping_import.html', {**defaults, 'error': 'Mapping model not available'})

    try:
        replace_existing = str(request.POST.get('replace_existing') or '').strip().lower() in {'1', 'true', 'on', 'yes'}
        create_missing_fields = str(request.POST.get('create_missing_fields') or '').strip().lower() in {'1', 'true', 'on', 'yes'}
        if not request.FILES:
            return render(
                request,
                'admin/onet_mapping_import.html',
                {**defaults, 'error': 'No file provided', 'replace_existing': replace_existing, 'create_missing_fields': create_missing_fields},
            )

        fobj = next(iter(request.FILES.values()))
        data = fobj.read()
        try:
            txt = data.decode('utf-8')
        except Exception:
            txt = data.decode('utf-8', errors='ignore')

        import io
        r = csv.DictReader(io.StringIO(txt), skipinitialspace=True)
        if not r.fieldnames:
            return render(
                request,
                'admin/onet_mapping_import.html',
                {**defaults, 'error': 'CSV has no header', 'replace_existing': replace_existing, 'create_missing_fields': create_missing_fields},
            )

        header = [h.strip() for h in (r.fieldnames or [])]
        header_l = [h.lower() for h in header]
        try:
            fld_slug_key = header[header_l.index('field_slug')]
            fld_name_key = header[header_l.index('field_name')] if 'field_name' in header_l else None
            occ_code_key = header[header_l.index('occupation_code')]
        except Exception:
            return render(
                request,
                'admin/onet_mapping_import.html',
                {
                    **defaults,
                    'error': 'CSV must include field_slug and occupation_code columns',
                    'replace_existing': replace_existing,
                    'create_missing_fields': create_missing_fields,
                },
            )

        weight_key = header[header_l.index('weight')] if 'weight' in header_l else None
        notes_key = header[header_l.index('notes')] if 'notes' in header_l else None

        rows = []
        seen = set()
        for row in r:
            slug = (row.get(fld_slug_key) or '').strip().lower()
            fname = (row.get(fld_name_key) or '').strip() if fld_name_key else ''
            code = (row.get(occ_code_key) or '').strip()
            if not slug or not code:
                continue
            k = (slug, fname.lower(), code)
            if k in seen:
                continue
            seen.add(k)
            weight_raw = (row.get(weight_key) or '').strip() if weight_key else ''
            notes = (row.get(notes_key) or '').strip() if notes_key else ''
            rows.append((slug, fname, code, weight_raw, notes))

        if not rows:
            return render(
                request,
                'admin/onet_mapping_import.html',
                {**defaults, 'error': 'No valid rows found', 'replace_existing': replace_existing, 'create_missing_fields': create_missing_fields},
            )

        from django.db.models import Count
        from django.utils.text import slugify

        slugs = sorted({s for s, _, _, _, _ in rows})
        names = sorted({n for _, n, _, _, _ in rows if n})

        def _slug_variants(s: str) -> set[str]:
            raw = (s or '').strip().lower()
            if not raw:
                return set()
            out = {raw}
            out.add(raw.replace('&', 'and'))
            out.add(raw.replace(' and ', ' '))
            out.add(raw.replace(' and ', '-'))
            out.add(raw.replace('--', '-'))
            return {v.strip('-') for v in out if v}

        def _name_slug_variants(n: str) -> set[str]:
            raw = (n or '').strip()
            if not raw:
                return set()
            out = {slugify(raw)}
            out.add(slugify(raw.replace('&', 'and')))
            out.add(slugify(raw.replace(' and ', ' ')))
            return {v for v in out if v}

        candidate_slugs = set()
        for s in slugs:
            candidate_slugs |= _slug_variants(s)
        for n in names:
            candidate_slugs |= _name_slug_variants(n)

        fields_by_slug = {f.slug.lower(): f for f in Field.objects.filter(slug__in=sorted(candidate_slugs))}
        fields_by_name = {f.name.strip().lower(): f for f in Field.objects.filter(name__in=names)}

        # Prefer resolving mappings onto Fields that are actually used by Programs.
        candidate_field_ids = sorted({f.id for f in fields_by_slug.values()} | {f.id for f in fields_by_name.values()})
        program_counts_by_field_id = {
            int(r['field_id']): int(r['c'])
            for r in (
                Field.objects.filter(id__in=candidate_field_ids)
                .annotate(c=Count('programs', distinct=True))
                .values('id', 'c')
            )
        }

        def _resolve_field(slug: str, name: str):
            s = (slug or '').strip().lower()
            n = (name or '').strip()

            candidates = []

            if s:
                f = fields_by_slug.get(s)
                if f:
                    candidates.append(f)

            n_l = n.strip().lower()
            if n_l:
                f = fields_by_name.get(n_l)
                if f:
                    candidates.append(f)

            for cand in sorted(_slug_variants(s)):
                f = fields_by_slug.get(cand)
                if f:
                    candidates.append(f)

            for cand in sorted(_name_slug_variants(n)):
                f = fields_by_slug.get(cand)
                if f:
                    candidates.append(f)

            if not candidates:
                return None

            # De-dup while preserving order.
            uniq = []
            seen_ids = set()
            for f in candidates:
                if f.id in seen_ids:
                    continue
                seen_ids.add(f.id)
                uniq.append(f)

            # If any candidate has programs, prefer the one with the most.
            best = None
            best_prog_count = -1
            for f in uniq:
                pc = int(program_counts_by_field_id.get(int(f.id), 0))
                if pc > best_prog_count:
                    best_prog_count = pc
                    best = f

            return best

        missing = [s for s in slugs if _resolve_field(s, '') is None]
        if missing:
            if create_missing_fields:
                created = 0
                for s in missing:
                    try:
                        # Prefer a readable default name while preserving the slug.
                        default_name = (s or '').replace('-', ' ').strip().title() or s
                        Field.objects.get_or_create(slug=s, defaults={'name': default_name})
                        created += 1
                    except Exception:
                        continue
                fields_by_slug = {f.slug.lower(): f for f in Field.objects.filter(slug__in=sorted(candidate_slugs | set(slugs)))}
                missing = [s for s in slugs if _resolve_field(s, '') is None]

            if missing:
                msg = (
                    f"Unknown field slugs: {', '.join(missing[:50])}. "
                    "These Fields must exist in the DB before mappings can be imported. "
                    "Either run the catalog ETL that loads Fields, or re-import with 'Create missing Fields' enabled."
                )
                return render(
                    request,
                    'admin/onet_mapping_import.html',
                    {
                        **defaults,
                        'error': msg,
                        'replace_existing': replace_existing,
                        'create_missing_fields': create_missing_fields,
                    },
                )

        if replace_existing:
            resolved_field_ids = sorted({f.id for s, n, _, _, _ in rows for f in [(_resolve_field(s, n))] if f is not None})
            if resolved_field_ids:
                OnetFieldOccupationMapping.objects.filter(field_id__in=resolved_field_ids).delete()

        wrote = 0
        for slug, fname, code, weight_raw, notes in rows:
            fld = _resolve_field(slug, fname)
            if not fld:
                continue
            weight = None
            if weight_raw:
                try:
                    weight = float(weight_raw)
                except Exception:
                    weight = None
            obj, created = OnetFieldOccupationMapping.objects.get_or_create(field=fld, occupation_code=code)
            if weight is not None:
                try:
                    obj.weight = weight
                except Exception:
                    pass
            if notes:
                obj.notes = notes
            obj.save()
            wrote += 1

        return render(
            request,
            'admin/onet_mapping_import.html',
            {
                **defaults,
                'ok': f'Imported {wrote} mappings',
                'replace_existing': replace_existing,
                'create_missing_fields': create_missing_fields,
            },
        )
    except Exception as e:
        return render(request, 'admin/onet_mapping_import.html', {**defaults, 'error': str(e)})


@staff_member_required
@ensure_csrf_cookie
def admin_onet_mapping_import(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_mapping_import_impl(request)
    return csrf_protect(_admin_onet_mapping_import_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_snapshot_import_impl(request):
    defaults = {
        'truncate': False,
    }

    if request.method == 'GET':
        try:
            existing = int(OnetOccupationSnapshot.objects.count())
        except Exception:
            existing = None
        return render(request, 'admin/onet_snapshot_import.html', {**defaults, 'existing_count': existing})

    try:
        truncate = str(request.POST.get('truncate') or '').strip().lower() in {'1', 'true', 'on', 'yes'}
        if not request.FILES:
            return render(request, 'admin/onet_snapshot_import.html', {**defaults, 'error': 'No file provided', 'truncate': truncate})

        fobj = next(iter(request.FILES.values()))
        data = fobj.read()
        try:
            txt = data.decode('utf-8')
        except Exception:
            txt = data.decode('utf-8', errors='ignore')

        import io

        r = csv.DictReader(io.StringIO(txt), skipinitialspace=True)
        if not r.fieldnames:
            return render(request, 'admin/onet_snapshot_import.html', {**defaults, 'error': 'CSV has no header', 'truncate': truncate})

        header = [h.strip() for h in (r.fieldnames or [])]
        header_l = [h.lower() for h in header]
        required = {'onetsoc_code', 'title', 'description', 'job_zone'}
        if not required.issubset(set(header_l)):
            return render(
                request,
                'admin/onet_snapshot_import.html',
                {**defaults, 'error': 'CSV must include onetsoc_code, title, description, job_zone columns', 'truncate': truncate},
            )

        def _key(col: str) -> str:
            return header[header_l.index(col)]

        k_code = _key('onetsoc_code')
        k_title = _key('title')
        k_desc = _key('description')
        k_jz = _key('job_zone')

        rows: list[tuple[str, str, str, int | None]] = []
        seen = set()
        for row in r:
            code = str(row.get(k_code) or '').strip()
            if not code:
                continue
            if code in seen:
                continue
            seen.add(code)
            title = str(row.get(k_title) or '').strip()
            desc = str(row.get(k_desc) or '').strip()
            jz_raw = str(row.get(k_jz) or '').strip()
            job_zone = None
            if jz_raw:
                try:
                    job_zone = int(float(jz_raw))
                except Exception:
                    job_zone = None
            rows.append((code, title, desc, job_zone))

        if not rows:
            return render(request, 'admin/onet_snapshot_import.html', {**defaults, 'error': 'No valid rows found', 'truncate': truncate})

        with transaction.atomic():
            if truncate:
                OnetOccupationSnapshot.objects.all().delete()

            wrote = 0
            for code, title, desc, job_zone in rows:
                OnetOccupationSnapshot.objects.update_or_create(
                    onetsoc_code=code,
                    defaults={'title': title, 'description': desc, 'job_zone': job_zone},
                )
                wrote += 1

        existing = int(OnetOccupationSnapshot.objects.count())
        return render(
            request,
            'admin/onet_snapshot_import.html',
            {**defaults, 'ok': f'Imported {wrote} snapshot rows', 'truncate': truncate, 'existing_count': existing},
        )
    except Exception as e:
        return render(request, 'admin/onet_snapshot_import.html', {**defaults, 'error': str(e)})


@staff_member_required
@ensure_csrf_cookie
def admin_onet_snapshot_import(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_snapshot_import_impl(request)
    return csrf_protect(_admin_onet_snapshot_import_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_snapshot_export_impl(request):
    try:
        rows = list(OnetOccupationSnapshot.objects.all().values('onetsoc_code', 'title', 'description', 'job_zone'))
    except Exception:
        rows = []

    resp = HttpResponse(content_type='text/csv')
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    resp['Content-Disposition'] = f'attachment; filename=onet_occupation_snapshot_{ts}.csv'

    w = csv.writer(resp)
    w.writerow(['onetsoc_code', 'title', 'description', 'job_zone'])
    for r in rows:
        w.writerow([
            (r.get('onetsoc_code') or '').strip(),
            (r.get('title') or '').strip(),
            (r.get('description') or '').strip(),
            '' if r.get('job_zone') is None else str(r.get('job_zone')),
        ])
    return resp


@staff_member_required
@ensure_csrf_cookie
def admin_onet_snapshot_export(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_snapshot_export_impl(request)
    return csrf_protect(_admin_onet_snapshot_export_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_snapshot_generate_impl(request):
    defaults = {
        'truncate': True,
        'max_description_len': 600,
    }

    def _existing_count():
        try:
            return int(OnetOccupationSnapshot.objects.count())
        except Exception:
            return None

    if request.method == 'GET':
        return render(
            request,
            'admin/onet_snapshot_generate.html',
            {
                **defaults,
                'existing_count': _existing_count(),
            },
        )

    try:
        truncate = str(request.POST.get('truncate') or '').strip().lower() in {'1', 'true', 'on', 'yes'}
        max_description_len_raw = (request.POST.get('max_description_len') or '').strip()
        try:
            max_description_len = int(max_description_len_raw) if max_description_len_raw else int(defaults['max_description_len'])
        except Exception:
            max_description_len = int(defaults['max_description_len'])

        if max_description_len < 0:
            max_description_len = int(defaults['max_description_len'])

        try:
            from .models import OnetJobZone  # type: ignore
        except Exception:
            OnetJobZone = None  # type: ignore

        # Load job zones if available
        job_zone_by_code = {}
        if OnetJobZone is not None:
            try:
                for code, jz in OnetJobZone.objects.all().values_list('onetsoc_code', 'job_zone'):
                    try:
                        code_s = str(code)
                        jz_i = int(getattr(jz, 'job_zone', jz))
                    except Exception:
                        continue
                    prev = job_zone_by_code.get(code_s)
                    if prev is None or jz_i > prev:
                        job_zone_by_code[code_s] = jz_i
            except Exception:
                job_zone_by_code = {}

        wrote = 0
        started = time.time()
        with transaction.atomic():
            if truncate:
                OnetOccupationSnapshot.objects.all().delete()

            qs = OnetOccupation.objects.all().order_by('onetsoc_code').values_list('onetsoc_code', 'title', 'description')
            for code, title, desc in qs.iterator(chunk_size=5000):
                code_s = str(code)
                desc_s = (desc or '')
                if max_description_len and max_description_len > 0:
                    desc_s = desc_s[:max_description_len]
                OnetOccupationSnapshot.objects.update_or_create(
                    onetsoc_code=code_s,
                    defaults={
                        'title': title or '',
                        'description': desc_s,
                        'job_zone': job_zone_by_code.get(code_s),
                    },
                )
                wrote += 1

        elapsed = time.time() - started
        return render(
            request,
            'admin/onet_snapshot_generate.html',
            {
                **defaults,
                'ok': f'Generated {wrote} snapshot rows in {elapsed:.1f}s',
                'truncate': truncate,
                'max_description_len': max_description_len,
                'existing_count': _existing_count(),
            },
        )
    except Exception as e:
        return render(
            request,
            'admin/onet_snapshot_generate.html',
            {
                **defaults,
                'error': str(e),
                'existing_count': _existing_count(),
            },
        )


@staff_member_required
@ensure_csrf_cookie
def admin_onet_snapshot_generate(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_snapshot_generate_impl(request)
    return csrf_protect(_admin_onet_snapshot_generate_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_program_field_occupation_export_impl(request):
    try:
        from catalog.models import Program, Field  # type: ignore
    except Exception:
        Program = None  # type: ignore
        Field = None  # type: ignore

    try:
        from .mapping_models import OnetFieldOccupationMapping  # type: ignore
    except Exception:
        OnetFieldOccupationMapping = None  # type: ignore

    if Program is None or Field is None or OnetFieldOccupationMapping is None:
        return render(request, 'admin/onet_dashboard.html', {'error': 'Required apps/models not available'})

    field_slug = (request.GET.get('field_slug') or '').strip().lower()
    include_unassigned = str(request.GET.get('include_unassigned') or '').strip().lower() in {'1', 'true', 'on', 'yes'}

    title_by_code: dict[str, str] = {}
    try:
        title_by_code = {
            str(code).strip(): (title or '')
            for code, title in OnetOccupationSnapshot.objects.all().values_list('onetsoc_code', 'title')
            if code
        }
    except Exception:
        title_by_code = {}

    qs_prog = Program.objects.select_related('field').all().order_by('id')
    if not include_unassigned:
        qs_prog = qs_prog.filter(field__isnull=False)
    if field_slug:
        qs_prog = qs_prog.filter(field__slug__iexact=field_slug)

    qs_map = OnetFieldOccupationMapping.objects.select_related('field').all()
    if field_slug:
        qs_map = qs_map.filter(field__slug__iexact=field_slug)

    mappings_by_field_id: dict[int, list[tuple[str, str, str, str]]] = {}
    for m in qs_map.values('field_id', 'occupation_code', 'weight', 'notes'):
        fid = int(m.get('field_id') or 0)
        if not fid:
            continue
        code = (m.get('occupation_code') or '').strip()
        if not code:
            continue
        weight = '' if m.get('weight') is None else str(m.get('weight'))
        notes = (m.get('notes') or '').strip()
        title = (title_by_code.get(code, '') or '').strip()
        mappings_by_field_id.setdefault(fid, []).append((code, title, weight, notes))

    resp = HttpResponse(content_type='text/csv')
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    resp['Content-Disposition'] = f'attachment; filename=program_field_occupation_mappings_{ts}.csv'

    w = csv.writer(resp)
    w.writerow([
        'program_id',
        'program_name',
        'field_slug',
        'field_name',
        'occupation_code',
        'occupation_title',
        'weight',
        'notes',
    ])

    for p in qs_prog.iterator(chunk_size=2000):
        fld = getattr(p, 'field', None)
        if fld is None:
            if include_unassigned:
                w.writerow([
                    str(getattr(p, 'id', '') or ''),
                    (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip(),
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                ])
            continue

        occs = mappings_by_field_id.get(int(getattr(fld, 'id', 0) or 0), [])
        if not occs:
            continue

        prog_id = str(getattr(p, 'id', '') or '')
        prog_name = (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip()
        fld_slug = (getattr(fld, 'slug', '') or '').strip()
        fld_name = (getattr(fld, 'name', '') or '').strip()
        for code, title, weight, notes in occs:
            w.writerow([
                prog_id,
                prog_name,
                fld_slug,
                fld_name,
                code,
                title,
                weight,
                notes,
            ])

    return resp


@staff_member_required
@ensure_csrf_cookie
def admin_onet_program_field_occupation_export(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_program_field_occupation_export_impl(request)
    return csrf_protect(_admin_onet_program_field_occupation_export_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_mapping_coverage_impl(request):
    try:
        from catalog.models import Field, Program  # type: ignore
    except Exception:
        Field = None  # type: ignore
        Program = None  # type: ignore

    try:
        from .mapping_models import OnetFieldOccupationMapping  # type: ignore
    except Exception:
        OnetFieldOccupationMapping = None  # type: ignore

    if Field is None or Program is None or OnetFieldOccupationMapping is None:
        return render(request, 'admin/onet_mapping_coverage.html', {'error': 'Required apps/models not available'})

    repair_summary = None
    repair_error = None

    if request.method == 'POST' and str(request.POST.get('action') or '').strip().lower() == 'repair_duplicates':
        try:
            from django.db import transaction
            from django.utils.text import slugify

            def _canon(s: str) -> str:
                raw = (s or '').strip().lower()
                if not raw:
                    return ''
                raw = raw.replace('&', 'and')
                raw = raw.replace(' and ', ' ')
                raw = raw.replace('_', ' ')
                return slugify(raw)

            fields = list(
                Field.objects
                .annotate(program_count=Count('programs', distinct=True))
                .values('id', 'slug', 'name', 'program_count')
            )

            by_key: dict[str, list[dict]] = {}
            for f in fields:
                key = _canon(str(f.get('slug') or '')) or _canon(str(f.get('name') or ''))
                if not key:
                    continue
                by_key.setdefault(key, []).append(f)

            groups = [g for g in by_key.values() if len(g) > 1]

            moved_rows = 0
            deduped_rows = 0
            affected_groups = 0
            affected_target_fields = set()
            affected_source_fields = set()

            with transaction.atomic():
                for g in groups:
                    g_sorted = sorted(
                        g,
                        key=lambda x: (int(x.get('program_count') or 0), -int(x.get('id') or 0)),
                        reverse=True,
                    )
                    target = g_sorted[0]
                    sources = g_sorted[1:]
                    target_id = int(target['id'])

                    changed_this_group = False
                    for src in sources:
                        src_id = int(src['id'])
                        if src_id == target_id:
                            continue

                        src_mappings = list(
                            OnetFieldOccupationMapping.objects
                            .filter(field_id=src_id)
                            .values('occupation_code', 'weight', 'notes')
                        )
                        if not src_mappings:
                            continue

                        for m in src_mappings:
                            code = str(m.get('occupation_code') or '').strip()
                            if not code:
                                continue
                            obj, created = OnetFieldOccupationMapping.objects.get_or_create(
                                field_id=target_id,
                                occupation_code=code,
                                defaults={'weight': m.get('weight'), 'notes': m.get('notes') or ''},
                            )
                            if created:
                                moved_rows += 1
                            else:
                                deduped_rows += 1

                        OnetFieldOccupationMapping.objects.filter(field_id=src_id).delete()
                        changed_this_group = True
                        affected_source_fields.add(src_id)

                    if changed_this_group:
                        affected_groups += 1
                        affected_target_fields.add(target_id)

            repair_summary = {
                'groups_found': len(groups),
                'groups_changed': affected_groups,
                'target_fields': len(affected_target_fields),
                'source_fields': len(affected_source_fields),
                'moved_rows': moved_rows,
                'deduped_rows': deduped_rows,
            }
        except Exception as e:
            repair_error = str(e)

    total_fields = int(Field.objects.count())
    mapped_fields = int(OnetFieldOccupationMapping.objects.values('field_id').distinct().count())
    unmapped_fields = max(0, total_fields - mapped_fields)

    total_programs = int(Program.objects.count())
    programs_no_field = int(Program.objects.filter(field__isnull=True).count())
    programs_field_unmapped = int(
        Program.objects.filter(field__isnull=False)
        .annotate(field_mapping_count=Count('field__onet_mappings', distinct=True))
        .filter(field_mapping_count=0)
        .count()
    )

    qs = (
        Field.objects
        .annotate(mapping_count=Count('onet_mappings', distinct=True))
        .annotate(program_count=Count('programs', distinct=True))
        .filter(mapping_count=0)
        .order_by('-program_count', 'name')
    )

    limit_raw = (request.GET.get('limit') or '').strip()
    try:
        limit = max(1, min(500, int(limit_raw))) if limit_raw else 200
    except Exception:
        limit = 200

    rows = list(qs.values('id', 'name', 'slug', 'program_count')[:limit])

    prog_limit_raw = (request.GET.get('program_limit') or '').strip()
    try:
        program_limit = max(1, min(500, int(prog_limit_raw))) if prog_limit_raw else 200
    except Exception:
        program_limit = 200

    programs_no_field_rows = list(
        Program.objects.filter(field__isnull=True)
        .order_by('normalized_name')
        .values('id', 'normalized_name', 'level', 'institution_id')[:program_limit]
    )

    programs_field_unmapped_rows = list(
        Program.objects.filter(field__isnull=False)
        .annotate(field_mapping_count=Count('field__onet_mappings', distinct=True))
        .filter(field_mapping_count=0)
        .order_by('field__name', 'normalized_name')
        .values('id', 'normalized_name', 'level', 'field__name', 'field__slug', 'institution_id')[:program_limit]
    )

    return render(
        request,
        'admin/onet_mapping_coverage.html',
        {
            'total_fields': total_fields,
            'mapped_fields': mapped_fields,
            'unmapped_fields': unmapped_fields,
            'unmapped_rows': rows,
            'limit': limit,
            'total_programs': total_programs,
            'programs_no_field': programs_no_field,
            'programs_field_unmapped': programs_field_unmapped,
            'program_limit': program_limit,
            'programs_no_field_rows': programs_no_field_rows,
            'programs_field_unmapped_rows': programs_field_unmapped_rows,
            'repair_summary': repair_summary,
            'repair_error': repair_error,
        },
    )


@staff_member_required
@ensure_csrf_cookie
def admin_onet_mapping_coverage(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_mapping_coverage_impl(request)
    return csrf_protect(_admin_onet_mapping_coverage_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_dq_coverage_impl(request):
    try:
        from catalog.models import Field, Program  # type: ignore
    except Exception:
        Field = None  # type: ignore
        Program = None  # type: ignore

    try:
        from .mapping_models import OnetFieldOccupationMapping  # type: ignore
    except Exception:
        OnetFieldOccupationMapping = None  # type: ignore

    if Field is None or Program is None or OnetFieldOccupationMapping is None:
        return render(request, 'admin/onet_dq_coverage.html', {'error': 'Required apps/models not available'})

    def _as_int(v: object, default: int) -> int:
        try:
            return int(str(v).strip())
        except Exception:
            return int(default)

    defaults = {
        'max_programs_no_field': 0,
        'max_programs_in_unmapped_fields': 0,
        'max_unmapped_fields_with_programs': 0,
    }

    if request.method == 'POST':
        thresholds = {
            'max_programs_no_field': _as_int(request.POST.get('max_programs_no_field'), defaults['max_programs_no_field']),
            'max_programs_in_unmapped_fields': _as_int(request.POST.get('max_programs_in_unmapped_fields'), defaults['max_programs_in_unmapped_fields']),
            'max_unmapped_fields_with_programs': _as_int(request.POST.get('max_unmapped_fields_with_programs'), defaults['max_unmapped_fields_with_programs']),
        }
    else:
        thresholds = defaults

    total_fields = int(Field.objects.count())
    mapped_fields = int(OnetFieldOccupationMapping.objects.values('field_id').distinct().count())
    unmapped_fields = max(0, total_fields - mapped_fields)

    total_programs = int(Program.objects.count())
    programs_no_field = int(Program.objects.filter(field__isnull=True).count())
    programs_in_unmapped_fields = int(
        Program.objects.filter(field__isnull=False)
        .annotate(field_mapping_count=Count('field__onet_mappings', distinct=True))
        .filter(field_mapping_count=0)
        .count()
    )
    unmapped_fields_qs = (
        Field.objects
        .annotate(mapping_count=Count('onet_mappings', distinct=True))
        .annotate(program_count=Count('programs', distinct=True))
        .filter(mapping_count=0)
        .order_by('-program_count', 'name')
    )
    unmapped_fields_with_programs = int(unmapped_fields_qs.filter(program_count__gt=0).count())

    detail_limit_raw = (request.POST.get('detail_limit') or request.GET.get('detail_limit') or '').strip()
    try:
        detail_limit = max(1, min(500, int(detail_limit_raw))) if detail_limit_raw else 50
    except Exception:
        detail_limit = 50

    unmapped_field_rows = list(unmapped_fields_qs.values('id', 'name', 'slug', 'program_count')[:detail_limit])

    # Sample programs affected by "programs in unmapped fields"
    program_rows = list(
        Program.objects.filter(field__isnull=False)
        .annotate(field_mapping_count=Count('field__onet_mappings', distinct=True))
        .filter(field_mapping_count=0)
        .order_by('-id')
        .values('id', 'normalized_name', 'level', 'field__name', 'field__slug')[:detail_limit]
    )

    failed = []
    if programs_no_field > thresholds['max_programs_no_field']:
        failed.append('programs_no_field')
    if programs_in_unmapped_fields > thresholds['max_programs_in_unmapped_fields']:
        failed.append('programs_in_unmapped_fields')
    if unmapped_fields_with_programs > thresholds['max_unmapped_fields_with_programs']:
        failed.append('unmapped_fields_with_programs')

    passed = len(failed) == 0

    return render(
        request,
        'admin/onet_dq_coverage.html',
        {
            'thresholds': thresholds,
            'passed': passed,
            'failed': failed,
            'total_fields': total_fields,
            'mapped_fields': mapped_fields,
            'unmapped_fields': unmapped_fields,
            'total_programs': total_programs,
            'programs_no_field': programs_no_field,
            'programs_in_unmapped_fields': programs_in_unmapped_fields,
            'unmapped_fields_with_programs': unmapped_fields_with_programs,
            'detail_limit': detail_limit,
            'unmapped_field_rows': unmapped_field_rows,
            'program_rows': program_rows,
        },
    )


@staff_member_required
@ensure_csrf_cookie
def admin_onet_dq_coverage(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_dq_coverage_impl(request)
    return csrf_protect(_admin_onet_dq_coverage_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_program_field_review_impl(request):
    try:
        from catalog.models import Field, Program  # type: ignore
    except Exception:
        Field = None  # type: ignore
        Program = None  # type: ignore

    if Field is None or Program is None:
        return render(request, 'admin/onet_program_field_review.html', {'error': 'Catalog app not available'})

    limit_raw = (request.GET.get('limit') or '').strip()
    try:
        limit = max(1, min(500, int(limit_raw))) if limit_raw else 200
    except Exception:
        limit = 200

    q = (request.GET.get('q') or '').strip()
    qs = Program.objects.filter(field__isnull=True)
    if q:
        qs = qs.filter(normalized_name__icontains=q)
    programs = list(qs.order_by('normalized_name').values('id', 'normalized_name', 'level')[:limit])
    fields = list(Field.objects.all().order_by('name').values('id', 'name', 'slug'))

    if request.method == 'GET':
        return render(
            request,
            'admin/onet_program_field_review.html',
            {
                'programs': programs,
                'fields': fields,
                'limit': limit,
                'q': q,
            },
        )

    try:
        field_id = (request.POST.get('field_id') or '').strip()
        prog_ids = request.POST.getlist('program_id')
        try:
            field_id_i = int(field_id)
        except Exception:
            field_id_i = 0

        if not field_id_i:
            return render(
                request,
                'admin/onet_program_field_review.html',
                {
                    'programs': programs,
                    'fields': fields,
                    'limit': limit,
                    'q': q,
                    'error': 'You must choose a Field',
                },
            )

        if not prog_ids:
            return render(
                request,
                'admin/onet_program_field_review.html',
                {
                    'programs': programs,
                    'fields': fields,
                    'limit': limit,
                    'q': q,
                    'error': 'Select at least one Program',
                },
            )

        fld = Field.objects.filter(id=field_id_i).first()
        if not fld:
            return render(
                request,
                'admin/onet_program_field_review.html',
                {
                    'programs': programs,
                    'fields': fields,
                    'limit': limit,
                    'q': q,
                    'error': 'Field not found',
                },
            )

        with transaction.atomic():
            wrote = int(Program.objects.filter(id__in=prog_ids, field__isnull=True).update(field=fld))

        # Refresh list after update
        qs2 = Program.objects.filter(field__isnull=True)
        if q:
            qs2 = qs2.filter(normalized_name__icontains=q)
        programs2 = list(qs2.order_by('normalized_name').values('id', 'normalized_name', 'level')[:limit])

        return render(
            request,
            'admin/onet_program_field_review.html',
            {
                'programs': programs2,
                'fields': fields,
                'limit': limit,
                'q': q,
                'ok': f'Updated {wrote} programs',
            },
        )
    except Exception as e:
        return render(
            request,
            'admin/onet_program_field_review.html',
            {
                'programs': programs,
                'fields': fields,
                'limit': limit,
                'q': q,
                'error': str(e),
            },
        )


@staff_member_required
@ensure_csrf_cookie
def admin_program_field_review(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_program_field_review_impl(request)
    return csrf_protect(_admin_program_field_review_impl)(request)
