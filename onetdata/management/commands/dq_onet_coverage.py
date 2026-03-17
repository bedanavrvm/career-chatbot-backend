from __future__ import annotations

import json

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Data-quality gate for O*NET coverage. Exits non-zero when coverage fails.'

    def add_arguments(self, parser):
        parser.add_argument('--max-programs-no-field', default='0')
        parser.add_argument('--max-programs-in-unmapped-fields', default='0')
        parser.add_argument('--max-unmapped-fields-with-programs', default='0')
        parser.add_argument('--json', action='store_true', help='Output JSON summary')

    def handle(self, *args, **options):
        from catalog.models import Field, Program
        from onetdata.mapping_models import OnetFieldOccupationMapping

        def _as_int(v: object, default: int) -> int:
            try:
                return int(str(v).strip())
            except Exception:
                return int(default)

        max_programs_no_field = _as_int(options.get('max_programs_no_field'), 0)
        max_programs_in_unmapped = _as_int(options.get('max_programs_in_unmapped_fields'), 0)
        max_unmapped_fields_with_programs = _as_int(options.get('max_unmapped_fields_with_programs'), 0)
        as_json = bool(options.get('json'))

        total_fields = int(Field.objects.count())
        mapped_fields = int(OnetFieldOccupationMapping.objects.values('field_id').distinct().count())
        unmapped_fields = max(0, total_fields - mapped_fields)

        programs_no_field = int(Program.objects.filter(field__isnull=True).count())

        # Programs whose field has zero mappings
        from django.db.models import Count

        programs_in_unmapped_fields = int(
            Program.objects.filter(field__isnull=False)
            .annotate(field_mapping_count=Count('field__onet_mappings', distinct=True))
            .filter(field_mapping_count=0)
            .count()
        )

        unmapped_fields_with_programs = int(
            Field.objects.annotate(mapping_count=Count('onet_mappings', distinct=True))
            .annotate(program_count=Count('programs', distinct=True))
            .filter(mapping_count=0, program_count__gt=0)
            .count()
        )

        summary = {
            'total_fields': total_fields,
            'mapped_fields': mapped_fields,
            'unmapped_fields': unmapped_fields,
            'programs_no_field': programs_no_field,
            'programs_in_unmapped_fields': programs_in_unmapped_fields,
            'unmapped_fields_with_programs': unmapped_fields_with_programs,
            'thresholds': {
                'max_programs_no_field': max_programs_no_field,
                'max_programs_in_unmapped_fields': max_programs_in_unmapped,
                'max_unmapped_fields_with_programs': max_unmapped_fields_with_programs,
            },
        }

        failed = []
        if programs_no_field > max_programs_no_field:
            failed.append('programs_no_field')
        if programs_in_unmapped_fields > max_programs_in_unmapped:
            failed.append('programs_in_unmapped_fields')
        if unmapped_fields_with_programs > max_unmapped_fields_with_programs:
            failed.append('unmapped_fields_with_programs')

        summary['passed'] = len(failed) == 0
        summary['failed_checks'] = failed

        if as_json:
            self.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
        else:
            self.stdout.write('O*NET coverage DQ summary:')
            for k, v in summary.items():
                if isinstance(v, (dict, list)):
                    continue
                self.stdout.write(f'  {k}: {v}')
            self.stdout.write('Thresholds:')
            for k, v in summary['thresholds'].items():
                self.stdout.write(f'  {k}: {v}')
            if failed:
                self.stdout.write(self.style.ERROR(f'FAILED checks: {", ".join(failed)}'))
            else:
                self.stdout.write(self.style.SUCCESS('PASSED'))

        if failed:
            raise SystemExit(2)
