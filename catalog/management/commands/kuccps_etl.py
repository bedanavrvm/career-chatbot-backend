from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run KUCCPS ETL pipeline"

    def add_arguments(self, parser):
        parser.add_argument("--action", default="all")
        parser.add_argument("--config", default="")
        parser.add_argument("--inplace", action="store_true")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        action = (options.get("action") or "all").strip()
        config_path = (options.get("config") or "").strip()
        inplace = bool(options.get("inplace"))
        dry_run = bool(options.get("dry_run"))

        cmd_file = Path(__file__).resolve()
        backend_dir = cmd_file.parents[3]
        # Robustly locate backend root (Render path layouts can differ)
        for p in cmd_file.parents:
            if (p / "manage.py").exists() and (p / "scripts" / "etl" / "kuccps" / "etl.py").exists():
                backend_dir = p
                break
        etl_dir = backend_dir / "scripts" / "etl" / "kuccps"

        if not (etl_dir / "etl.py").exists():
            raise RuntimeError(f"KUCCPS ETL module not found at {etl_dir} (expected etl.py). backend_dir={backend_dir}")

        # IMPORTANT: don't import `etl` directly, because this repo also has a Django app package named `etl`.
        # Import the KUCCPS ETL module explicitly.
        try:
            from scripts.etl.kuccps.etl import (  # type: ignore
                Config,
                copy_inputs,
                bootstrap_csvs,
                extract_programs,
                transform_programs,
                transform_normalize,
                dedup_programs,
                dq_report,
                load_csvs,
            )
        except Exception:
            import importlib.util

            spec = importlib.util.spec_from_file_location("kuccps_etl_module", str((etl_dir / "etl.py").resolve()))
            if not spec or not spec.loader:
                raise
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)

            Config = getattr(m, "Config")
            copy_inputs = getattr(m, "copy_inputs")
            bootstrap_csvs = getattr(m, "bootstrap_csvs")
            extract_programs = getattr(m, "extract_programs")
            transform_programs = getattr(m, "transform_programs")
            transform_normalize = getattr(m, "transform_normalize")
            dedup_programs = getattr(m, "dedup_programs")
            dq_report = getattr(m, "dq_report")
            load_csvs = getattr(m, "load_csvs")

        if config_path:
            p = Path(config_path)
            if not p.is_absolute():
                p = (backend_dir / p).resolve()
            cfg = Config.from_yaml(p)
        else:
            cfg = Config(dataset_year=2024, dataset_root=etl_dir, inputs={}, raw_dir=etl_dir / "raw", processed_dir=etl_dir / "processed")

        if action == "extract":
            copy_inputs(cfg)
            return
        if action == "extract-programs":
            extract_programs(cfg)
            return
        if action == "transform":
            bootstrap_csvs(cfg)
            return
        if action == "transform-programs":
            transform_programs(cfg)
            return
        if action == "transform-normalize":
            transform_normalize(cfg)
            return
        if action == "dedup-programs":
            dedup_programs(cfg, inplace=inplace)
            return
        if action == "dq-report":
            dq_report(cfg)
            return
        if action == "load":
            load_csvs(cfg, dry_run=dry_run)
            return
        if action == "all":
            copy_inputs(cfg)
            bootstrap_csvs(cfg)

            if (cfg.raw_dir / "uploads").exists() or (etl_dir / "raw" / "uploads").exists():
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
            return

        raise ValueError(f"Unknown action: {action}")
