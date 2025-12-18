import unittest
import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Make kuccps package importable when running from backend/
THIS_DIR = Path(__file__).resolve().parent
KUCCPS_DIR = THIS_DIR.parent
if str(KUCCPS_DIR) not in sys.path:
    sys.path.append(str(KUCCPS_DIR))

from etl import Config, dedup_programs, dq_report, EXPECTED  # type: ignore


class TestETLReports(unittest.TestCase):
    def test_dedup_and_dq_report(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "processed"
            processed.mkdir(parents=True, exist_ok=True)

            # Write a tiny programs.csv with a duplicate group (same inst/name/level/campus)
            programs_path = processed / "programs.csv"
            with programs_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(EXPECTED["programs.csv"])  # header
                # Duplicate pair
                w.writerow([
                    1,  # source_index
                    "1170",  # institution_code
                    "MACHAKOS UNIVERSITY",  # institution_name
                    "Environmental Sciences",  # field_name
                    "1170213",  # program_code
                    "213",  # course_suffix
                    "BACHELOR OF ENVIRONMENTAL SCIENCE",  # name
                    "BACHELOR OF ENVIRONMENTAL SCIENCE",  # normalized_name
                    "bachelor",  # level
                    "",  # campus
                    "Eastern",  # region
                    "",  # duration_years
                    "",  # award
                    "",  # mode
                    "{}",  # subject_requirements_json
                ])
                w.writerow([
                    2,
                    "1170",
                    "MACHAKOS UNIVERSITY",
                    "Environmental Sciences",
                    "1170213",
                    "213",
                    "BACHELOR OF ENVIRONMENTAL SCIENCE",
                    "BACHELOR OF ENVIRONMENTAL SCIENCE",
                    "bachelor",
                    "",
                    "Eastern",
                    "",
                    "",
                    "",
                    "{}",
                ])
                # A unique row
                w.writerow([
                    3,
                    "1112",
                    "TECHNICAL UNIVERSITY OF KENYA",
                    "Architecture",
                    "1112102",
                    "102",
                    "BACHELOR OF ARCHITECTURAL STUDIES/BACHELOR OF ARCHITECTURE",
                    "BACHELOR OF ARCHITECTURAL STUDIES/BACHELOR OF ARCHITECTURE",
                    "bachelor",
                    "",
                    "Nairobi",
                    "",
                    "",
                    "",
                    "{}",
                ])

            cfg = Config(dataset_year=2024, dataset_root=tmp_path, inputs={}, raw_dir=tmp_path / "raw", processed_dir=processed)

            # Run dedup (no inplace) and verify outputs
            dedup_programs(cfg, inplace=False)

            deduped_fp = processed / "programs_deduped.csv"
            self.assertTrue(deduped_fp.exists())
            rows = list(csv.DictReader(deduped_fp.open(encoding="utf-8"), delimiter="\t"))
            # We expect 2 masters: one from the duplicate group and the unique one
            self.assertEqual(len(rows), 2)

            suppressed_fp = processed / "dedup_suppressed.csv"
            self.assertTrue(suppressed_fp.exists())
            suppressed_rows = list(csv.DictReader(suppressed_fp.open(encoding="utf-8")))
            self.assertGreaterEqual(len(suppressed_rows), 1)

            # Run DQ report and verify it exists and has basic metrics
            dq_report(cfg)
            dq_fp = processed / "dq_report.csv"
            self.assertTrue(dq_fp.exists())
            metrics = {r["metric"]: r for r in csv.DictReader(dq_fp.open(encoding="utf-8"))}
            self.assertIn("programs_total", metrics)
            self.assertEqual(int(metrics["programs_total"]["value"]), 2)  # reading from deduped


if __name__ == "__main__":
    unittest.main()
