import unittest
import math
import sys
from pathlib import Path

# Make kuccps module importable
THIS_DIR = Path(__file__).resolve().parent
KUCCPS_DIR = THIS_DIR.parent
if str(KUCCPS_DIR) not in sys.path:
    sys.path.append(str(KUCCPS_DIR))

from eligibility import evaluate_eligibility, compute_cluster_points  # noqa: E402


def kcse_points(grade: str) -> int:
    mapping = {
        "A": 12, "A-": 11, "B+": 10, "B": 9, "B-": 8,
        "C+": 7, "C": 6, "C-": 5, "D+": 4, "D": 3, "D-": 2, "E": 1,
    }
    return mapping[grade]


class TestEligibilityCluster(unittest.TestCase):
    def test_json_primary_selection_and_cluster(self):
        # Program row with JSON requirements (required + groups)
        prog = {
            "name": "BACHELOR OF SCIENCE (COMPUTER TECHNOLOGY)",
            "normalized_name": "BACHELOR OF SCIENCE (COMPUTER TECHNOLOGY)",
            "subject_requirements_json": (
                '{"required": '
                '[{"subject_code":"ENG","min_grade":"C"},'
                ' {"subject_code":"MAT","min_grade":"C+"}],'
                ' "groups": ['
                '   {"pick":1, "options":['
                '       {"subject_code":"PHY","min_grade":"C"},'
                '       {"subject_code":"CHE","min_grade":"C"}]},'
                '   {"pick":1, "options":['
                '       {"subject_code":"BIO","min_grade":"D+"},'
                '       {"subject_code":"GEO","min_grade":"C"}]}'
                ' ]}'
            )
        }
        # Candidate grades cover at least 7 subjects
        grades = {
            "ENG": "B",   # 9
            "MAT": "A-",  # 11
            "PHY": "B",   # 9
            "CHE": "C",   # 6
            "BIO": "C-",  # 5
            "GEO": "B",   # 9
            "KIS": "B+",  # 10
        }
        res = evaluate_eligibility(prog, grades)
        self.assertTrue(res["eligible"])  # meets required and groups
        # Compute expected cluster points for 4 subjects chosen by JSON logic: ENG, MAT, PHY, GEO
        r = kcse_points("B") + kcse_points("A-") + kcse_points("B") + kcse_points("B")  # 9+11+9+9 = 38
        top7 = sorted([kcse_points(g) for g in grades.values()], reverse=True)
        t = sum(top7[:7])  # 59
        expected = math.sqrt((r / 48) * (t / 84)) * 48
        self.assertAlmostEqual(res["cluster_points"], round(expected, 3), places=2)
        # Also verify compute_cluster_points returns consistent result
        cp = compute_cluster_points(prog, grades)
        self.assertAlmostEqual(cp["cluster_points"], round(expected, 3), places=2)
        self.assertEqual(len(cp["subjects"]), 4)

    def test_mapping_fallback_when_json_empty(self):
        # Program row with empty JSON; use mapping fallback for Medicine & Surgery
        prog = {
            "name": "BACHELOR OF MEDICINE AND SURGERY",
            "normalized_name": "BACHELOR OF MEDICINE AND SURGERY",
            "subject_requirements_json": "{}",
        }
        grades = {
            "BIO": "A",    # 12
            "CHE": "A-",   # 11
            "MAT": "A",    # 12
            "PHY": "B+",   # 10
            "ENG": "B",    # 9
            "KIS": "A-",   # 11
            "HIS": "B",    # 9
        }
        cp = compute_cluster_points(prog, grades)
        # Expected mapping chooses BIO, CHE, best of {MAT, PHY} -> MAT, best of {ENG, KIS} -> KIS
        r = kcse_points("A") + kcse_points("A-") + kcse_points("A") + kcse_points("A-")  # 12+11+12+11 = 46
        top7 = sorted([kcse_points(g) for g in grades.values()], reverse=True)
        t = sum(top7[:7])  # 74
        expected = math.sqrt((r / 48) * (t / 84)) * 48
        self.assertAlmostEqual(cp["cluster_points"], round(expected, 3), places=2)
        self.assertEqual(len(cp["subjects"]), 4)


if __name__ == "__main__":
    unittest.main()
