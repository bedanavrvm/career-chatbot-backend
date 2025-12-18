"""
KCSE grade conversion utilities.

Functions here help parse and compare KCSE grades and convert them to numeric points.
Intended use: subject requirement checks and later cluster points calculations.

Scale (12-point):
A=12, A-=11, B+=10, B=9, B-=8, C+=7, C=6, C-=5, D+=4, D=3, D-=2, E=1
"""
from __future__ import annotations

from typing import Optional

# Canonical grade ordering (best -> worst)
GRADES_ORDER = (
    "A",
    "A-",
    "B+",
    "B",
    "B-",
    "C+",
    "C",
    "C-",
    "D+",
    "D",
    "D-",
    "E",
)

# Points mapping aligned to GRADES_ORDER
POINTS_BY_GRADE = {
    "A": 12,
    "A-": 11,
    "B+": 10,
    "B": 9,
    "B-": 8,
    "C+": 7,
    "C": 6,
    "C-": 5,
    "D+": 4,
    "D": 3,
    "D-": 2,
    "E": 1,
}


def normalize_grade(raw: str) -> Optional[str]:
    """Return a canonical grade token like 'B+' or 'C-' or None if invalid.
    Trims whitespace and uppercases letters.
    """
    if not raw:
        return None
    s = raw.strip().upper().replace(" ", "")
    # Common artifacts: 'B +', 'C -', etc. Already collapsed above
    if s in POINTS_BY_GRADE:
        return s
    return None


def grade_points(grade: str) -> Optional[int]:
    """Map a canonical grade to 12-point scale (A=12 ... E=1)."""
    g = normalize_grade(grade)
    return POINTS_BY_GRADE.get(g) if g else None


def compare_grades(g1: str, g2: str) -> Optional[int]:
    """Compare two grades.
    Returns:
    - 1 if g1 better than g2
    - -1 if g1 worse than g2
    - 0 if equal
    - None if either invalid
    """
    p1, p2 = grade_points(g1), grade_points(g2)
    if p1 is None or p2 is None:
        return None
    if p1 > p2:
        return 1
    if p1 < p2:
        return -1
    return 0


def meets_min_grade(candidate_grade: str, min_required: str) -> Optional[bool]:
    """True if candidate_grade is equal or better than min_required.
    Returns None if any grade is invalid.
    """
    cmp = compare_grades(candidate_grade, min_required)
    return (cmp is not None) and (cmp >= 0)
