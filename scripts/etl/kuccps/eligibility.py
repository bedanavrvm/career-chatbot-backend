"""
Eligibility and (scaffold) cluster points evaluation for KUCCPS programs.

Given a program row (with subject_requirements_json) and a candidate's KCSE grades,
this module checks eligibility and computes a simple aggregate score placeholder.

Usage example:
    from eligibility import evaluate_eligibility
    prog = {...}  # a Dict row from programs_deduped.csv
    grades = {"ENG": "B+", "MAT": "A-", "PHY": "B", "CHE": "B-"}
    result = evaluate_eligibility(prog, grades)
    print(result["eligible"], result["cluster_points"], result["reasons"])

Notes:
- The current scoring is a scaffolding: it sums points used to satisfy
  required subjects plus the best 'pick' options in each group.
- A more accurate KUCCPS cluster points formula can replace `compute_cluster_points`
  in a later sprint without changing the public API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import json
import math
import csv
from pathlib import Path

try:
    from .grades import normalize_grade, grade_points, meets_min_grade  # type: ignore
except Exception:
    try:
        from grades import normalize_grade, grade_points, meets_min_grade  # type: ignore
    except Exception:
        import sys
        _HERE = Path(__file__).resolve().parent
        if str(_HERE) not in sys.path:
            sys.path.append(str(_HERE))
        from grades import normalize_grade, grade_points, meets_min_grade  # type: ignore


SUBJECT_CODE_ALIASES: Dict[str, str] = {
    '101': 'ENG',
    '102': 'KIS',
    '121': 'MAT',
    '231': 'BIO',
    '232': 'PHY',
    '233': 'CHE',
    '442': 'ART',
    '451': 'CSC',
    '501': 'FRE',
    '502': 'GER',
    '511': 'MUS',
}


SUBJECT_CANON_TO_NUM: Dict[str, str] = {v: k for k, v in SUBJECT_CODE_ALIASES.items()}


SUBJECT_TOKEN_ALIASES: Dict[str, str] = {
    'MATB': 'MAT',
    'MUC': 'MUS',
    'MUS': 'MUS',
    'ARD': 'ART',
    'ART': 'ART',
    'CMP': 'CSC',
    'CSC': 'CSC',
    'COM': 'CSC',
}


SUBJECT_TOKEN_CANON_TO_ALIASES: Dict[str, List[str]] = {}
for _alias, _canon in SUBJECT_TOKEN_ALIASES.items():
    SUBJECT_TOKEN_CANON_TO_ALIASES.setdefault(_canon, []).append(_alias)


def _norm_subject_token(s: str) -> str:
    """Normalize subject code tokens (collapse spaces, uppercase).
    Examples: "M AT" -> "MAT"; "MU C" -> "MUC"; "MAT B" -> "MATB"
    """
    return (s or "").replace(" ", "").strip().upper()


def _expand_candidate_grades(candidate_grades_raw: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (candidate_grades_raw or {}).items():
        key = _norm_subject_token(k)
        g = normalize_grade(v)
        if not key or not g:
            continue
        canon = SUBJECT_TOKEN_ALIASES.get(key, key)
        out[key] = g
        if canon:
            out[canon] = g
            for a in (SUBJECT_TOKEN_CANON_TO_ALIASES.get(canon) or []):
                out[a] = g
        if key in SUBJECT_CODE_ALIASES:
            out[SUBJECT_CODE_ALIASES[key]] = g
        if key in SUBJECT_CANON_TO_NUM:
            out[SUBJECT_CANON_TO_NUM[key]] = g
        if canon in SUBJECT_CANON_TO_NUM:
            out[SUBJECT_CANON_TO_NUM[canon]] = g
    return out


def _best_group_picks(options: List[Dict[str, Any]], pick: int, candidate_grades: Dict[str, str]) -> Tuple[int, List[Tuple[str, str, int]], List[str]]:
    """Return the number of qualifying picks achieved, the chosen (subject,grade,points) list, and reasons.
    Chooses the best options by candidate points subject to meeting minimum grades.
    """
    scored: List[Tuple[str, str, int]] = []
    reasons: List[str] = []
    for opt in options:
        subj = _norm_subject_token(opt.get("subject") or opt.get("subject_code") or "")
        min_g = normalize_grade(opt.get("min_grade", "")) or opt.get("min_grade", "")
        cand_g_raw = candidate_grades.get(subj)
        if not cand_g_raw:
            reasons.append(f"missing grade for {subj} in group")
            continue
        ok = meets_min_grade(cand_g_raw, min_g)
        if not ok:
            reasons.append(f"grade {cand_g_raw} < min {min_g} for {subj} in group")
            continue
        pts = grade_points(cand_g_raw) or 0
        scored.append((subj, cand_g_raw, pts))
    # Choose best by points
    scored.sort(key=lambda x: (-x[2], x[0]))
    chosen = scored[: max(0, pick)]
    return (len(chosen), chosen, reasons)


def _load_cluster_subjects_map() -> List[Tuple[str, List[List[str]]]]:
    """Load program → 4-subject cluster groups mapping from mappings/cluster_subjects.csv.
    subjects grammar: groups separated by ';', alternatives within a group separated by '|'.
    Example: BIO;CHE;MAT|PHY;ENG|KIS → [[BIO],[CHE],[MAT,PHY],[ENG,KIS]]
    """
    here = Path(__file__).resolve().parent
    csv_path = here / "mappings" / "cluster_subjects.csv"
    out: List[Tuple[str, List[List[str]]]] = []
    if not csv_path.exists():
        return out
    with open(csv_path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            patt = (row.get("program_pattern") or "").strip().upper()
            spec = (row.get("subjects") or "").strip().upper()
            if not patt or not spec:
                continue
            groups: List[List[str]] = []
            for grp in [g.strip() for g in spec.split(";") if g.strip()]:
                alts = [a.strip() for a in grp.split("|") if a.strip()]
                groups.append(alts)
            if groups:
                out.append((patt, groups))
    return out


def _select_clusters_from_mapping(program_name: str, candidate_pts: Dict[str, int]) -> List[Tuple[str, int]]:
    """If a program_pattern matches program_name, select 4 cluster subjects using the mapped groups.
    For each group, pick the alternative with highest candidate points (if present). If <4 after groups,
    fill from remaining highest points to reach 4.
    """
    prog = (program_name or "").upper()
    mapping = _load_cluster_subjects_map()
    chosen: List[Tuple[str, int]] = []
    used = set()
    for patt, groups in mapping:
        if patt and patt in prog:
            for alts in groups:
                best = None
                for code in alts:
                    c = code.replace(" ", "").upper()
                    pts = candidate_pts.get(c)
                    if pts is None:
                        continue
                    if (best is None) or (pts > best[1]) or (pts == best[1] and c < best[0]):
                        best = (c, pts)
                if best:
                    chosen.append(best)
                    used.add(best[0])
            break  # stop at first matching pattern
    # Top-up to 4 if needed
    if len(chosen) < 4:
        remaining = sorted([(c, p) for c, p in candidate_pts.items() if c not in used], key=lambda x: (-x[1], x[0]))
        while len(chosen) < 4 and remaining:
            code, pts = remaining.pop(0)
            chosen.append((code, pts))
            used.add(code)
    # Trim to 4
    chosen.sort(key=lambda x: (-x[1], x[0]))
    return chosen[:4]


def compute_cluster_points(program_row: Dict[str, Any], candidate_grades: Dict[str, str]) -> Dict[str, Any]:
    """Compute weighted cluster points per KUCCPS-style formula.
    C = sqrt((r/R) * (t/T)) * 48
    - r: sum of points in the 4 selected cluster subjects for the program
    - R: 48 (max points for 4 subjects)
    - t: sum of points in the best 7 subjects overall for the candidate (or fewer if <7 provided)
    - T: 84 (max for 7 subjects) or 12 * min(7, subjects_provided) for resilience
    Note: Official practice relies on prescribed 4 cluster subjects per program; here we approximate
    using the program's required and best group picks to reach 4, then fill remaining slots by highest
    remaining subject points if needed.
    Returns details: { 'cluster_points': float, 'subjects': [(code, points)], 'r': int, 't': int }
    """
    # Parse spec
    try:
        spec = json.loads(program_row.get("subject_requirements_json") or "{}")
    except Exception:
        spec = {}
    required = spec.get("required", []) or []
    groups = spec.get("groups", []) or []

    # Build candidate points map {code: points}
    cand_pts: Dict[str, int] = {}
    for code, grade in (candidate_grades or {}).items():
        g = normalize_grade(grade)
        if g:
            cand_pts[code] = int(grade_points(g) or 0)

    # Prefer subject_requirements_json-derived selection; fallback to mapping if empty
    prog_name = (program_row.get("normalized_name") or program_row.get("name") or "").strip()
    tmp: List[Tuple[str, int]] = []
    # Required subjects (use if present)
    for it in required:
        sc = _norm_subject_token(it.get("subject") or it.get("subject_code") or "")
        if sc and sc in cand_pts:
            tmp.append((sc, cand_pts[sc]))
    # Group picks
    for grp in groups:
        pick = int(grp.get("pick") or 0)
        opts = grp.get("options", []) or []
        have, chosen, _ = _best_group_picks(opts, pick, {k: normalize_grade(v) or v for k, v in candidate_grades.items()})
        for sc, g, pts in chosen:
            tmp.append((sc, pts))
    # Ensure 4
    used_map = {c for c, _ in tmp}
    remaining = sorted([(c, p) for c, p in cand_pts.items() if c not in used_map], key=lambda x: (-x[1], x[0]))
    while len(tmp) < 4 and remaining:
        tmp.append(remaining.pop(0))
    tmp.sort(key=lambda x: (-x[1], x[0]))
    used: List[Tuple[str, int]] = tmp[:4]
    # Fallback to mapping only if requirements yielded nothing
    if not used:
        used = _select_clusters_from_mapping(prog_name, cand_pts)

    r_sum = sum(p for _, p in used)
    R = 48

    # Best 7 overall
    all_pts_sorted = sorted(cand_pts.values(), reverse=True)
    top_n = min(7, len(all_pts_sorted))
    t_sum = sum(all_pts_sorted[:top_n])
    T = 12 * top_n if top_n else 84

    try:
        cluster = math.sqrt((r_sum / R) * (t_sum / T)) * 48
    except Exception:
        cluster = 0.0
    return {
        "cluster_points": round(cluster, 3),
        "subjects": used,
        "r": r_sum,
        "t": t_sum,
    }


def evaluate_eligibility(program_row: Dict[str, Any], candidate_grades_raw: Dict[str, str]) -> Dict[str, Any]:
    """Evaluate eligibility against a single program row.
    Inputs:
      - program_row: a dict (as read from programs_deduped.csv) with subject_requirements_json
      - candidate_grades_raw: mapping like {"ENG": "B+", "MAT": "A-", ...}
    Returns dict with keys: eligible(bool), reasons(list), used_points(list of (subject,grade,points)), cluster_points(int)
    """
    candidate_grades = _expand_candidate_grades(candidate_grades_raw or {})

    req_json = program_row.get("subject_requirements_json") or "{}"
    try:
        spec = json.loads(req_json)
    except Exception:
        spec = {}
    required = spec.get("required", []) or []
    groups = spec.get("groups", []) or []

    reasons: List[str] = []
    used_points: List[Tuple[str, str, int]] = []

    # Check required subjects
    for it in required:
        subj = _norm_subject_token(it.get("subject") or it.get("subject_code") or "")
        min_g = normalize_grade(it.get("min_grade", "")) or it.get("min_grade", "")
        cand_g_raw = candidate_grades.get(subj)
        if not cand_g_raw:
            reasons.append(f"missing grade for required {subj}")
            continue
        ok = meets_min_grade(cand_g_raw, min_g)
        if not ok:
            reasons.append(f"grade {cand_g_raw} < min {min_g} for required {subj}")
            continue
        used_points.append((subj, cand_g_raw, grade_points(cand_g_raw) or 0))

    # Check groups
    for grp in groups:
        pick = int(grp.get("pick") or 0)
        options = grp.get("options", []) or []
        have, chosen, grp_reasons = _best_group_picks(options, pick, candidate_grades)
        reasons.extend(grp_reasons)
        if pick > 0 and have < pick:
            reasons.append(f"group requires {pick}, only {have} options meet min")
        used_points.extend(chosen)

    # Eligibility result
    eligible = True
    # Required subjects must all have been satisfied
    if required:
        # For each required subj we expect a corresponding used_points entry (by subject code)
        req_codes = {_norm_subject_token(it.get("subject") or it.get("subject_code") or "") for it in required}
        satisfied = {code for code, _, _ in used_points if code in req_codes}
        if req_codes - satisfied:
            eligible = False
    # Each group must satisfy pick
    for grp in groups:
        pick = int(grp.get("pick") or 0)
        if pick > 0:
            # We conservatively rely on reasons from above; mark ineligible if any group unmet
            text = f"group requires {pick}, only "
            if any(text in r for r in reasons):
                eligible = False
                break

    # Compute KUCCPS-style weighted cluster points
    cluster = compute_cluster_points(program_row, candidate_grades)

    return {
        "eligible": eligible,
        "reasons": reasons,
        "used_points": used_points,
        "cluster_points": cluster["cluster_points"],
        "cluster_details": cluster,
    }
