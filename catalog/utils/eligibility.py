"""
catalog/utils/eligibility.py

Re-exports the core eligibility constants and functions from the ETL scripts
so that the conversations FSM and catalog APIs can import from a stable
catalog-owned path rather than reaching into the scripts/ directory.

The canonical implementation stays in ``scripts.etl.kuccps.eligibility``;
this module is a thin façade.

Usage::

    from catalog.utils.eligibility import (
        evaluate_eligibility,
        compute_cluster_points,
        SUBJECT_CODE_ALIASES,
        SUBJECT_CANON_TO_NUM,
        SUBJECT_TOKEN_ALIASES,
        SUBJECT_TOKEN_CANON_TO_ALIASES,
    )
"""
from __future__ import annotations

try:
    from scripts.etl.kuccps.eligibility import (  # type: ignore
        SUBJECT_CODE_ALIASES,
        SUBJECT_CANON_TO_NUM,
        SUBJECT_TOKEN_ALIASES,
        SUBJECT_TOKEN_CANON_TO_ALIASES,
        evaluate_eligibility,
        compute_cluster_points,
        _expand_candidate_grades,
        _best_group_picks,
    )
except ImportError:
    # Graceful fallback for environments where the ETL scripts aren't on
    # sys.path (e.g., stripped production deployments).
    SUBJECT_CODE_ALIASES: dict = {}
    SUBJECT_CANON_TO_NUM: dict = {}
    SUBJECT_TOKEN_ALIASES: dict = {}
    SUBJECT_TOKEN_CANON_TO_ALIASES: dict = {}

    def evaluate_eligibility(program_row, candidate_grades_raw):  # type: ignore
        return {'eligible': None, 'reasons': ['eligibility module unavailable'], 'used_points': [], 'cluster_points': 0}

    def compute_cluster_points(program_row, candidate_grades):  # type: ignore
        return {'cluster_points': 0.0, 'subjects': [], 'r': 0, 't': 0}

    def _expand_candidate_grades(candidate_grades_raw):  # type: ignore
        return {}

    def _best_group_picks(options, pick, candidate_grades):  # type: ignore
        return (0, [], [])

__all__ = [
    'SUBJECT_CODE_ALIASES',
    'SUBJECT_CANON_TO_NUM',
    'SUBJECT_TOKEN_ALIASES',
    'SUBJECT_TOKEN_CANON_TO_ALIASES',
    'evaluate_eligibility',
    'compute_cluster_points',
    '_expand_candidate_grades',
    '_best_group_picks',
]
