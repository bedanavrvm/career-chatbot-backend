from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import connection, transaction


_DEFAULT_SQL_DIR = Path(__file__).resolve().parents[4] / 'onet' / 'db_30_2_mysql' / 'db_30_2_mysql'


class Command(BaseCommand):
    help = 'Import O*NET MySQL .sql dump files into the configured database.'

    def _ensure_surrogate_ids_postgres(self, cursor):
        if getattr(connection, 'vendor', '') != 'postgresql':
            return

        cursor.execute(
            """
DO $$
BEGIN
    -- Ensure Django-admin-friendly surrogate PKs for O*NET tables that use composite keys.
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'interests'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'interests' AND column_name = 'id'
        ) THEN
            ALTER TABLE interests ADD COLUMN id BIGSERIAL;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE table_name = 'interests' AND constraint_type = 'PRIMARY KEY'
        ) THEN
            ALTER TABLE interests ADD CONSTRAINT interests_pkey PRIMARY KEY (id);
        END IF;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'skills'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'skills' AND column_name = 'id'
        ) THEN
            ALTER TABLE skills ADD COLUMN id BIGSERIAL;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE table_name = 'skills' AND constraint_type = 'PRIMARY KEY'
        ) THEN
            ALTER TABLE skills ADD CONSTRAINT skills_pkey PRIMARY KEY (id);
        END IF;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'related_occupations'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'related_occupations' AND column_name = 'id'
        ) THEN
            ALTER TABLE related_occupations ADD COLUMN id BIGSERIAL;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE table_name = 'related_occupations' AND constraint_type = 'PRIMARY KEY'
        ) THEN
            ALTER TABLE related_occupations ADD CONSTRAINT related_occupations_pkey PRIMARY KEY (id);
        END IF;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'job_zones'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'job_zones' AND column_name = 'id'
        ) THEN
            ALTER TABLE job_zones ADD COLUMN id BIGSERIAL;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE table_name = 'job_zones' AND constraint_type = 'PRIMARY KEY'
        ) THEN
            ALTER TABLE job_zones ADD CONSTRAINT job_zones_pkey PRIMARY KEY (id);
        END IF;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'education_training_experience'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'education_training_experience' AND column_name = 'id'
        ) THEN
            ALTER TABLE education_training_experience ADD COLUMN id BIGSERIAL;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE table_name = 'education_training_experience' AND constraint_type = 'PRIMARY KEY'
        ) THEN
            ALTER TABLE education_training_experience ADD CONSTRAINT education_training_experience_pkey PRIMARY KEY (id);
        END IF;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'ete_categories'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'ete_categories' AND column_name = 'id'
        ) THEN
            ALTER TABLE ete_categories ADD COLUMN id BIGSERIAL;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE table_name = 'ete_categories' AND constraint_type = 'PRIMARY KEY'
        ) THEN
            ALTER TABLE ete_categories ADD CONSTRAINT ete_categories_pkey_id PRIMARY KEY (id);
        END IF;
    END IF;
END $$;
"""
        )

    def add_arguments(self, parser):
        parser.add_argument(
            '--sql-dir',
            default=str(_DEFAULT_SQL_DIR),
            help='Directory containing the O*NET MySQL .sql files (default: onet/db_30_2_mysql/db_30_2_mysql)',
        )
        parser.add_argument(
            '--tables',
            default='core',
            help=(
                'Which tables to import. Options: core, all, or a comma-separated list of sql filenames without extension '
                '(e.g. 01_content_model_reference,03_occupation_data). Default: core'
            ),
        )
        parser.add_argument(
            '--skip-create',
            action='store_true',
            help='Skip CREATE TABLE statements and only run INSERTs.',
        )

    def _iter_statements(self, sql_path: Path):
        """Yield SQL statements from a dump file, streaming line-by-line.

        The O*NET MySQL dumps are mostly one statement per line (INSERTs), with some CREATE TABLE blocks.
        We also skip MySQL-specific transaction comments like /*! START TRANSACTION */.
        """
        buf = ''
        with open(sql_path, encoding='utf-8', errors='replace') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('/*!') and line.endswith('*/;'):
                    continue
                if line.startswith('--'):
                    continue

                buf += raw
                if ';' not in raw:
                    continue

                # flush the buffer as a statement; these dumps don’t use semicolons inside strings.
                stmt = buf.strip()
                buf = ''
                if not stmt:
                    continue
                yield stmt

        tail = buf.strip()
        if tail:
            yield tail

    def handle(self, *args, **options):
        sql_dir = Path(str(options.get('sql_dir') or '')).resolve()
        if not sql_dir.exists() or not sql_dir.is_dir():
            raise SystemExit(f'Invalid --sql-dir: {sql_dir}')

        tables_opt = (options.get('tables') or 'core').strip().lower()
        skip_create = bool(options.get('skip_create'))

        # Core set powers: occupation browsing + RIASEC matching + tasks/skills + related careers
        core_files = [
            '01_content_model_reference.sql',
            '03_occupation_data.sql',
            '04_scales_reference.sql',
            '13_interests.sql',
            '16_skills.sql',
            '17_task_statements.sql',
            '27_related_occupations.sql',
        ]

        if tables_opt == 'core':
            sql_files = core_files
        elif tables_opt == 'all':
            sql_files = [p.name for p in sorted(sql_dir.glob('*.sql'))]
        else:
            parts = [p.strip() for p in tables_opt.split(',') if p.strip()]
            sql_files = [f'{p}.sql' if not p.endswith('.sql') else p for p in parts]

        missing = [f for f in sql_files if not (sql_dir / f).exists()]
        if missing:
            raise SystemExit(f'Missing SQL files in {sql_dir}: {missing}')

        self.stdout.write(f'Importing O*NET SQL from: {sql_dir}')
        self.stdout.write(f'Files: {sql_files}')

        with transaction.atomic():
            with connection.cursor() as cursor:
                for filename in sql_files:
                    sql_path = sql_dir / filename
                    self.stdout.write(f'Loading {filename}...')

                    n = 0
                    for stmt in self._iter_statements(sql_path):
                        s = stmt.strip()
                        if not s:
                            continue

                        upper = s.lstrip().upper()
                        if skip_create and upper.startswith('CREATE TABLE'):
                            continue

                        if upper.startswith('CREATE TABLE'):
                            # If re-running, drop first to avoid "already exists".
                            # Table name is the token after CREATE TABLE.
                            try:
                                table_name = s.split('CREATE TABLE', 1)[1].strip().split(None, 1)[0]
                                table_name = table_name.strip('`"')
                            except Exception:
                                table_name = ''

                            if table_name:
                                cursor.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE;')

                        cursor.execute(s)
                        n += 1

                        if n % 5000 == 0:
                            self.stdout.write(f'  executed {n} statements...')

                    self.stdout.write(f'  executed {n} statements from {filename}')

                # If the import included these tables, ensure surrogate PKs exist (admin-friendly).
                self._ensure_surrogate_ids_postgres(cursor)

        self.stdout.write(self.style.SUCCESS('O*NET import complete.'))
