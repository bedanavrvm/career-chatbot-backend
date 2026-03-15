import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
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
    try:
        if not fp.exists():
            return ''
        txt = fp.read_text(encoding='utf-8', errors='ignore')
        lines = txt.splitlines()
        return '\n'.join(lines[-max_lines:])
    except Exception:
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

            p = subprocess.Popen(
                cmd,
                cwd=backend_cwd,
                env=proc_env,
                stdout=logf,
                stderr=logf,
                start_new_session=True,
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
