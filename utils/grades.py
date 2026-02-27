"""
utils/grades.py
Single source of truth for KCSE grade normalization and comparison.

All modules (conversations, accounts, catalog, etl) should import from here
instead of maintaining their own copies.
"""
from typing import Dict, Optional

GRADE_POINTS: Dict[str, int] = {
    'A': 12,
    'A-': 11,
    'B+': 10,
    'B': 9,
    'B-': 8,
    'C+': 7,
    'C': 6,
    'C-': 5,
    'D+': 4,
    'D': 3,
    'D-': 2,
    'E': 1,
}

# Ordered list (best → worst) for range iteration.
GRADE_ORDER = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'E']


def normalize_grade(grade: str) -> str:
    """Normalize a raw grade string to uppercase, stripped, no spaces.

    Examples:
        'a-'  → 'A-'
        ' B + ' → 'B+'
        'c plus' → 'C+'   (not handled, returned as-is after strip/upper)
    """
    return str(grade or '').strip().upper().replace(' ', '')


def grade_points(grade: str) -> Optional[int]:
    """Return the numeric points for a given grade string, or None if unknown."""
    return GRADE_POINTS.get(normalize_grade(grade))


def meets_min_grade(candidate: str, minimum: str) -> bool:
    """Return True if candidate grade is at least as good as minimum grade.

    Both values are normalized before comparison. Returns False if either
    is unrecognized.
    """
    c = grade_points(candidate)
    m = grade_points(minimum)
    if c is None or m is None:
        return False
    return c >= m


def normalize_grade_map(grades: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of the grade dict with all keys and values normalized."""
    return {
        str(k).strip().upper().replace(' ', ''): normalize_grade(v)
        for k, v in (grades or {}).items()
        if str(k).strip() and str(v).strip()
    }


def cluster_points(grade_map: Dict[str, str], subject_keys: list) -> float:
    """Compute total KCSE cluster points for a given list of subject keys.

    Returns 0.0 if no matching grades are found.
    """
    ngm = normalize_grade_map(grade_map)
    total = 0.0
    for key in subject_keys:
        g = ngm.get(str(key).strip().upper().replace(' ', ''), '')
        p = grade_points(g)
        if p is not None:
            total += p
    return total
