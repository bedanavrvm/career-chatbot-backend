import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Inline documentation: Simple recommendation utility that reads processed KUCCPS
# CSVs (programs_deduped.csv or programs.csv), optionally checks eligibility using
# the existing ETL eligibility helper, and ranks programs based on user traits.

TRAIT_FIELD_HINTS: Dict[str, List[str]] = {
    # Investigative -> science/tech/engineering
    'Investigative': ['computer', 'technology', 'engineering', 'science', 'statistics', 'mathematics', 'informatics', 'software', 'data', 'physics', 'chemistry'],
    # Artistic -> design/art/media
    'Artistic': ['design', 'architecture', 'art', 'media', 'film', 'music', 'graphics'],
    # Social -> education/health/community
    'Social': ['education', 'nursing', 'medicine', 'community', 'social', 'counsel', 'psychology'],
    # Enterprising -> business/management/law
    'Enterprising': ['business', 'commerce', 'management', 'entrepreneur', 'marketing', 'law'],
    # Conventional -> accounting/finance/admin/logistics
    'Conventional': ['account', 'finance', 'procurement', 'logistics', 'administration', 'records'],
    # Realistic -> agriculture/mechanical/construction
    'Realistic': ['agriculture', 'mechanical', 'civil', 'construction', 'automotive', 'mechatronics'],
}


def _processed_dir() -> Path:
    # Prefer env override to allow custom dataset location
    env_dir = os.getenv('KUCCPS_PROCESSED_DIR')
    if env_dir:
        return Path(env_dir).resolve()
    backend_dir = Path(__file__).resolve().parents[2]
    return backend_dir / 'scripts' / 'etl' / 'kuccps' / 'processed'


def _programs_path() -> Path:
    base = _processed_dir()
    p = base / 'programs_deduped.csv'
    return p if p.exists() else (base / 'programs.csv')


def _load_program_rows(limit: int = 0) -> List[Dict[str, str]]:
    fp = _programs_path()
    rows: List[Dict[str, str]] = []
    if not fp.exists():
        return rows
    with open(fp, encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for i, r in enumerate(rdr):
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def _eligibility_checker():
    # Inline: Import eligibility.evaluate_eligibility from ETL scripts (best-effort)
    try:
        etl_dir = Path(__file__).resolve().parents[2] / 'scripts' / 'etl' / 'kuccps'
        sys.path.append(str(etl_dir))
        from eligibility import evaluate_eligibility  # type: ignore
        return evaluate_eligibility
    except Exception:
        return None


def _score_by_traits(name: str, field_name: str, traits: Dict[str, float]) -> float:
    # Inline: score by matching trait hints to program name/field
    text = f"{name} {field_name}".lower()
    score = 0.0
    for trait, weight in traits.items():
        hints = TRAIT_FIELD_HINTS.get(trait, [])
        hits = sum(1 for h in hints if h in text)
        score += float(weight) * (0.5 + 0.5 * min(1, hits))
    return score


def recommend_top_k(grades: Dict[str, str], traits: Dict[str, float], k: int = 5) -> List[Dict[str, Any]]:
    # Inline: Recommend top-K programs filtered by eligibility (when available) and scored by traits
    rows = _load_program_rows()
    if not rows:
        return []
    eval_elig = _eligibility_checker()
    scored: List[Tuple[float, Dict[str, str]]] = []
    for r in rows:
        name = (r.get('normalized_name') or r.get('name') or '').strip()
        field_name = (r.get('field_name') or '').strip()
        level = (r.get('level') or '').strip().lower()
        code = (r.get('program_code') or r.get('code') or '').strip()
        inst = (r.get('institution_name') or '').strip()
        if not name or not code:
            continue
        if level and level not in ('bachelor', 'bachelors', 'degree', ''):
            continue
        # Eligibility filter (best-effort)
        if eval_elig and grades:
            try:
                ok = eval_elig(r, grades)
                if not (ok or (isinstance(ok, dict) and ok.get('eligible'))):
                    continue
            except Exception:
                # If evaluator fails, skip strict gating
                pass
        score = _score_by_traits(name, field_name, traits)
        # Prefer stronger grade signal if no traits
        if not traits and grades:
            score += 0.2
        scored.append((score, r))
    # Rank by score then lexicographically for stability
    top = sorted(scored, key=lambda t: (-t[0], t[1].get('normalized_name') or t[1].get('name') or ''))[:k]
    out: List[Dict[str, Any]] = []
    for sc, r in top:
        out.append({
            'program_code': (r.get('program_code') or r.get('code') or '').strip(),
            'program_name': (r.get('normalized_name') or r.get('name') or '').strip(),
            'institution_name': (r.get('institution_name') or '').strip(),
            'field_name': (r.get('field_name') or '').strip(),
            'level': (r.get('level') or '').strip(),
            'score': round(float(sc), 3),
        })
    return out
