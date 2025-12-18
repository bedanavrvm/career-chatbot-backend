import unittest
import sys
from pathlib import Path

# Make kuccps module importable
THIS_DIR = Path(__file__).resolve().parent
KUCCPS_DIR = THIS_DIR.parent
if str(KUCCPS_DIR) not in sys.path:
    sys.path.append(str(KUCCPS_DIR))

from grades import normalize_grade, grade_points, compare_grades, meets_min_grade  # noqa: E402


class TestGrades(unittest.TestCase):
    def test_normalize_grade_valid(self):
        self.assertEqual(normalize_grade(" b+ "), "B+")
        self.assertEqual(normalize_grade("A-"), "A-")
        self.assertEqual(normalize_grade("c"), "C")
        self.assertEqual(normalize_grade("D -".replace(" ", "")), "D-")

    def test_normalize_grade_invalid(self):
        self.assertIsNone(normalize_grade("A+"))  # not in KCSE scale
        self.assertIsNone(normalize_grade("Z"))
        self.assertIsNone(normalize_grade(""))
        self.assertIsNone(normalize_grade(None))

    def test_grade_points_mapping(self):
        self.assertEqual(grade_points("A"), 12)
        self.assertEqual(grade_points("A-"), 11)
        self.assertEqual(grade_points("B+"), 10)
        self.assertEqual(grade_points("B"), 9)
        self.assertEqual(grade_points("B-"), 8)
        self.assertEqual(grade_points("C+"), 7)
        self.assertEqual(grade_points("C"), 6)
        self.assertEqual(grade_points("C-"), 5)
        self.assertEqual(grade_points("D+"), 4)
        self.assertEqual(grade_points("D"), 3)
        self.assertEqual(grade_points("D-"), 2)
        self.assertEqual(grade_points("E"), 1)
        self.assertIsNone(grade_points("A+"))

    def test_compare_grades(self):
        self.assertEqual(compare_grades("B+", "B"), 1)
        self.assertEqual(compare_grades("C", "C"), 0)
        self.assertEqual(compare_grades("D", "C-"), -1)
        self.assertIsNone(compare_grades("A+", "A"))
        self.assertIsNone(compare_grades("A", "Z"))

    def test_meets_min_grade(self):
        self.assertTrue(meets_min_grade("B", "C+"))
        self.assertTrue(meets_min_grade("C+", "C+"))
        self.assertFalse(meets_min_grade("C", "C+"))
        self.assertFalse(meets_min_grade("E", "D-"))
        self.assertTrue(meets_min_grade("A-", "B+"))


if __name__ == "__main__":
    unittest.main()
