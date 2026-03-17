from __future__ import annotations

import csv
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction


class Command(BaseCommand):
    help = 'Import a lightweight O*NET snapshot CSV into onet_occupation_snapshot (Render-friendly).'

    def add_arguments(self, parser):
        parser.add_argument(
            '--path',
            default='onet_occupation_snapshot.csv',
            help='CSV path (default: onet_occupation_snapshot.csv)',
        )
        parser.add_argument(
            '--truncate',
            action='store_true',
            help='Delete existing snapshot rows before import.',
        )

    def handle(self, *args, **options):
        from onetdata.models import OnetOccupationSnapshot

        csv_path = Path(str(options.get('path') or 'onet_occupation_snapshot.csv')).resolve()
        if not csv_path.exists():
            raise SystemExit(f'CSV not found: {csv_path}')

        truncate = bool(options.get('truncate'))

        with open(csv_path, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                raise SystemExit('CSV has no header')

            required = {'onetsoc_code', 'title', 'description', 'job_zone'}
            missing = required - {h.strip() for h in (r.fieldnames or [])}
            if missing:
                raise SystemExit(f'CSV missing columns: {sorted(missing)}')

            rows = []
            for row in r:
                code = str(row.get('onetsoc_code') or '').strip()
                if not code:
                    continue
                title = str(row.get('title') or '').strip()
                desc = str(row.get('description') or '').strip()
                jz_raw = str(row.get('job_zone') or '').strip()
                job_zone = None
                if jz_raw:
                    try:
                        job_zone = int(float(jz_raw))
                    except Exception:
                        job_zone = None

                rows.append((code, title, desc, job_zone))

        if not rows:
            self.stdout.write('No rows to import')
            return

        with transaction.atomic():
            if truncate:
                OnetOccupationSnapshot.objects.all().delete()

            wrote = 0
            for code, title, desc, job_zone in rows:
                obj, _ = OnetOccupationSnapshot.objects.update_or_create(
                    onetsoc_code=code,
                    defaults={
                        'title': title,
                        'description': desc,
                        'job_zone': job_zone,
                    },
                )
                wrote += 1

        self.stdout.write(self.style.SUCCESS(f'Imported {wrote} snapshot rows from {csv_path}'))
