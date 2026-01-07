import json
import os
import subprocess
import sys
import csv
import zipfile
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_protect

from utils.http import is_local_request


def _processed_dir() -> Path:
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


@staff_member_required
@ensure_csrf_cookie
def _admin_etl_upload_impl(request):
    etl_root = _processed_dir().parent
    base = etl_root / "raw" / "uploads"
    processed_dir = etl_root / "processed"
    if request.method == "GET":
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
        return render(
            request,
            "admin/etl_upload.html",
            {"upload_dir": str(base), "processed_dir": str(processed_dir), "error": str(e)},
        )


@staff_member_required
@ensure_csrf_cookie
def admin_etl_upload(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_etl_upload_impl(request)
    return csrf_protect(_admin_etl_upload_impl)(request)


@staff_member_required
@ensure_csrf_cookie
def _admin_etl_process_impl(request):
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
        return render(
            request,
            "admin/etl_process.html",
            {
                "actions": actions,
                "job_running": job_running,
                "job_pid": job_pid,
                "job_log_path": str(job_log_path) if job_log_path else "",
                "job_log_tail": job_log_tail,
                "prod_mode": prod_mode,
                "local_only_actions": local_only_actions,
            },
        )

    try:
        action = (request.POST.get("action") or "transform-normalize").strip()
        config_path = (request.POST.get("config") or "").strip()
        inplace = bool(request.POST.get("inplace"))
        dry_run = bool(request.POST.get("dry_run"))

        if action not in actions:
            msg = [f"Action '{action}' is not enabled on this server"]
            return render(
                request,
                "admin/etl_process.html",
                {
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
                },
            )

        async_actions_env = (os.getenv("ETL_ASYNC_ACTIONS", "") or "").strip()
        if async_actions_env:
            async_actions = [a.strip() for a in async_actions_env.split(",") if a.strip()]
        else:
            async_actions = ["all", "load", "dq-report"]

        if action in async_actions and os.getenv("ETL_RUN_ALL_ASYNC", "1").strip().lower() not in ("0", "false", "no"):
            msg = []
            if job_running:
                msg.append("ETL job already running")
                return render(
                    request,
                    "admin/etl_process.html",
                    {
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
                    },
                )

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
            return render(
                request,
                "admin/etl_process.html",
                {
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
                },
            )

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

        cfg = None
        msg = []
        etl_stats = {}
        if config_path:
            backend_dir = Path(__file__).resolve().parent.parent
            p = Path(config_path)
            if not p.is_absolute():
                p = (backend_dir / p).resolve()
            try:
                cfg = Config.from_yaml(p)
                msg.append(f"Using config: {p}")
            except Exception as e:
                return render(
                    request,
                    "admin/etl_process.html",
                    {
                        "error": f"Invalid config: {e}",
                        "actions": full_actions,
                    },
                )
        else:
            cfg = Config(dataset_year=2024, dataset_root=etl_dir, inputs={}, raw_dir=etl_dir / "raw", processed_dir=etl_dir / "processed")

        ran = False
        load_summary = None
        if action == "extract":
            copy_inputs(cfg)
            ran = True
        elif action == "extract-programs":
            if not cfg.inputs.get("programs_pdf"):
                uploads = (etl_dir / "raw" / "uploads")
                if uploads.exists():
                    pdfs = sorted(uploads.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if pdfs:
                        latest = pdfs[0]
                        try:
                            rel = latest.resolve().relative_to(cfg.dataset_root)
                        except Exception:
                            rel = latest.resolve().relative_to(etl_dir)
                        cfg.inputs["programs_pdf"] = str(rel)
                        cfg.inputs["source_id"] = latest.stem
                        msg.append(f"Detected uploaded PDF: {latest.name}")
                if not cfg.inputs.get("programs_pdf"):
                    return render(
                        request,
                        "admin/etl_process.html",
                        {
                            "error": "No programs PDF configured or found under raw/uploads. Upload a PDF first on the Upload page.",
                            "actions": full_actions,
                            "messages": msg,
                        },
                    )
            try:
                import pdfplumber  # noqa: F401
            except Exception as dep_err:
                return render(
                    request,
                    "admin/etl_process.html",
                    {
                        "error": f"Missing dependency: pdfplumber not installed ({dep_err}). Please install it in venv.",
                        "actions": full_actions,
                        "messages": msg,
                    },
                )
            extract_programs(cfg)
            ran = True
        elif action == "transform":
            bootstrap_csvs(cfg)
            ran = True
        elif action == "transform-programs":
            transform_programs(cfg)
            ran = True
        elif action == "transform-normalize":
            transform_normalize(cfg)
            ran = True
        elif action == "dedup-programs":
            dedup_programs(cfg, inplace=inplace)
            ran = True
        elif action == "dq-report":
            dq_report(cfg)
            ran = True
        elif action == "load":
            load_csvs(cfg, dry_run=dry_run)
            ran = True
        elif action == "all":
            copy_inputs(cfg)
            bootstrap_csvs(cfg)
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
            load_csvs(cfg, dry_run=dry_run)
            ran = True

        etl_stats_kv = sorted([(k, etl_stats[k]) for k in etl_stats.keys()]) if etl_stats else []
        return render(
            request,
            "admin/etl_process.html",
            {
                "action": action,
                "ran": ran,
                "processed_dir": str(cfg.processed_dir),
                "raw_dir": str(cfg.raw_dir),
                "inplace": inplace,
                "dry_run": dry_run,
                "messages": msg,
                "actions": full_actions,
                "config_value": config_path,
                "load_summary": load_summary,
                "etl_stats_kv": etl_stats_kv,
                "job_running": job_running,
                "job_pid": job_pid,
                "job_log_path": str(job_log_path) if job_log_path else "",
                "job_log_tail": job_log_tail,
            },
        )
    except Exception as e:
        return render(request, "admin/etl_process.html", {"error": str(e), "actions": full_actions})


@staff_member_required
@ensure_csrf_cookie
def admin_etl_process(request):
    if settings.DEBUG and is_local_request(request):
        return _admin_etl_process_impl(request)
    return csrf_protect(_admin_etl_process_impl)(request)
