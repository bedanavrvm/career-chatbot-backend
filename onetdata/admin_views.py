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
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_protect

from utils.http import is_local_request

from .models import (
    OnetContentElement,
    OnetInterest,
    OnetOccupation,
    OnetRelatedOccupation,
    OnetScale,
    OnetSkill,
    OnetTaskStatement,
)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]{1,}")

_DEFAULT_EXCLUDED_SOC_PREFIXES = {
    '55-',
    '53-',
    '51-',
}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


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


def _score_occupations(occupations: list[dict], keywords: list[str]) -> dict[str, float]:
    kw = [k.lower() for k in keywords if k and str(k).strip()]
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

        title_tokens = set(_tokens(title))
        desc_tokens = set(_tokens(desc))
        if not (title_tokens or desc_tokens):
            continue

        title_hits = sum(1 for k in kw if k in title_tokens)
        if title_hits <= 0:
            continue

        desc_hits = sum(1 for k in kw if k in desc_tokens)
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
        field_slugs_raw = (request.POST.get('field_slugs') or '').strip()
        field_slugs = [s.strip().lower() for s in field_slugs_raw.split(',') if s.strip()]

        fields_qs = Field.objects.all().order_by('name')
        if field_slugs:
            fields_qs = fields_qs.filter(slug__in=field_slugs)

        try:
            occs = list(OnetOccupation.objects.all().values('onetsoc_code', 'title', 'description'))
        except Exception as e:
            msg = (
                'O*NET tables are not available in the current database. '
                'Import O*NET first (Admin → O*NET → Import) or switch your local DB to Postgres and run import_onet. '
                f'Details: {e}'
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
            ranked = [(code, sc) for code, sc in ranked if sc >= min_score][:max(1, top_n_occupations)]
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
def admin_onet_mapping_suggest(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_mapping_suggest_impl(request)
    return csrf_protect(_admin_onet_mapping_suggest_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_onet_mapping_import_impl(request):
    defaults = {
        'replace_existing': False,
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
        if not request.FILES:
            return render(request, 'admin/onet_mapping_import.html', {**defaults, 'error': 'No file provided', 'replace_existing': replace_existing})

        fobj = next(iter(request.FILES.values()))
        data = fobj.read()
        try:
            txt = data.decode('utf-8')
        except Exception:
            txt = data.decode('utf-8', errors='ignore')

        import io
        r = csv.DictReader(io.StringIO(txt), skipinitialspace=True)
        if not r.fieldnames:
            return render(request, 'admin/onet_mapping_import.html', {**defaults, 'error': 'CSV has no header', 'replace_existing': replace_existing})

        header = [h.strip() for h in (r.fieldnames or [])]
        header_l = [h.lower() for h in header]
        try:
            fld_slug_key = header[header_l.index('field_slug')]
            occ_code_key = header[header_l.index('occupation_code')]
        except Exception:
            return render(
                request,
                'admin/onet_mapping_import.html',
                {**defaults, 'error': 'CSV must include field_slug and occupation_code columns', 'replace_existing': replace_existing},
            )

        weight_key = header[header_l.index('weight')] if 'weight' in header_l else None
        notes_key = header[header_l.index('notes')] if 'notes' in header_l else None

        rows = []
        seen = set()
        for row in r:
            slug = (row.get(fld_slug_key) or '').strip().lower()
            code = (row.get(occ_code_key) or '').strip()
            if not slug or not code:
                continue
            k = (slug, code)
            if k in seen:
                continue
            seen.add(k)
            weight_raw = (row.get(weight_key) or '').strip() if weight_key else ''
            notes = (row.get(notes_key) or '').strip() if notes_key else ''
            rows.append((slug, code, weight_raw, notes))

        if not rows:
            return render(request, 'admin/onet_mapping_import.html', {**defaults, 'error': 'No valid rows found', 'replace_existing': replace_existing})

        slugs = sorted({s for s, _, _, _ in rows})
        fields = {f.slug.lower(): f for f in Field.objects.filter(slug__in=slugs)}

        missing = [s for s in slugs if s not in fields]
        if missing:
            return render(
                request,
                'admin/onet_mapping_import.html',
                {**defaults, 'error': f"Unknown field slugs: {', '.join(missing[:50])}", 'replace_existing': replace_existing},
            )

        if replace_existing:
            OnetFieldOccupationMapping.objects.filter(field__slug__in=slugs).delete()

        wrote = 0
        for slug, code, weight_raw, notes in rows:
            fld = fields.get(slug)
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

        return render(request, 'admin/onet_mapping_import.html', {**defaults, 'ok': f'Imported {wrote} mappings', 'replace_existing': replace_existing})
    except Exception as e:
        return render(request, 'admin/onet_mapping_import.html', {**defaults, 'error': str(e)})


@staff_member_required
@ensure_csrf_cookie
def admin_onet_mapping_import(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_onet_mapping_import_impl(request)
    return csrf_protect(_admin_onet_mapping_import_impl)(request)
