from __future__ import annotations

import csv
from pathlib import Path

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Export a lightweight O*NET snapshot CSV (onetsoc_code,title,description,job_zone) from imported O*NET tables.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--out',
            default='onet_occupation_snapshot.csv',
            help='Output CSV path (default: onet_occupation_snapshot.csv)',
        )
        parser.add_argument(
            '--max-description-len',
            default='1000',
            help='Truncate description to this many characters (default: 1000). Use 0 for no truncation.',
        )

    def handle(self, *args, **options):
        from onetdata.models import OnetJobZone, OnetOccupation

        out_path = Path(str(options.get('out') or 'onet_occupation_snapshot.csv')).resolve()
        max_desc_len = int(str(options.get('max_description_len') or '1000') or '1000')

        job_zone_by_code: dict[str, int] = {}
        for code, jz in OnetJobZone.objects.all().values_list('onetsoc_code', 'job_zone'):
            try:
                code_s = str(code)
                jz_i = int(getattr(jz, 'job_zone', jz))
            except Exception:
                continue
            prev = job_zone_by_code.get(code_s)
            if prev is None or jz_i > prev:
                job_zone_by_code[code_s] = jz_i

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['onetsoc_code', 'title', 'description', 'job_zone'])
            qs = OnetOccupation.objects.all().order_by('onetsoc_code').values_list('onetsoc_code', 'title', 'description')
            n = 0
            for code, title, desc in qs.iterator(chunk_size=5000):
                desc_s = (desc or '')
                if max_desc_len > 0:
                    desc_s = desc_s[:max_desc_len]
                w.writerow([code, title or '', desc_s, job_zone_by_code.get(str(code), '')])
                n += 1

        self.stdout.write(self.style.SUCCESS(f'Wrote {n} rows to {out_path}'))
