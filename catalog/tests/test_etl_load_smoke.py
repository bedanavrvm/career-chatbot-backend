import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.test import TestCase


class EtlLoadSmokeTests(TestCase):
    def test_kuccps_load_creates_catalog_rows(self):
        with TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir(parents=True, exist_ok=True)

            with (processed / "institutions.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["code", "name", "alias", "region", "county", "country", "website"])
                w.writerow(["1170", "MACHAKOS UNIVERSITY", "MKSU", "Eastern", "Machakos", "Kenya", "https://mksu.ac.ke"])

            with (processed / "fields.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["name", "parent", "description"])
                w.writerow(["Engineering", "", ""])

            # Minimal programs file. The loader is flexible about extra columns but requires:
            # institution_code, field_name, normalized_name (or name), level, campus.
            with (processed / "programs.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "institution_code",
                    "institution_name",
                    "field_name",
                    "program_code",
                    "name",
                    "normalized_name",
                    "level",
                    "campus",
                    "region",
                    "duration_years",
                    "award",
                    "mode",
                    "subject_requirements_json",
                    "source_id",
                ])
                w.writerow([
                    "1170",
                    "MACHAKOS UNIVERSITY",
                    "Engineering",
                    "1170114",
                    "BSc Civil Engineering",
                    "BSC CIVIL ENGINEERING",
                    "bachelor",
                    "Main",
                    "Eastern",
                    "4",
                    "BSc",
                    "Full-time",
                    "{}",
                    "test",
                ])

            from scripts.etl.kuccps.etl import Config, load_csvs

            cfg = Config(
                dataset_year=2024,
                dataset_root=Path(tmpdir),
                inputs={},
                raw_dir=Path(tmpdir) / "raw",
                processed_dir=processed,
            )

            # We are already running inside Django's test runner; prevent the ETL loader
            # from calling django.setup() again.
            with patch("scripts.etl.kuccps.etl.setup_django", autospec=True) as _setup:
                _setup.return_value = None
                load_csvs(cfg, dry_run=False)

            from catalog.models import Institution, Program

            self.assertGreaterEqual(Institution.objects.count(), 1)
            self.assertGreaterEqual(Program.objects.count(), 1)
            self.assertTrue(Program.objects.filter(code="1170114").exists())
