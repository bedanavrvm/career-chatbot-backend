import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
try:
    # Prefer DB-backed catalog when available
    from catalog.models import Program, ProgramLevel  # type: ignore
except Exception:  # pragma: no cover - during migrations or no DB
    Program = None  # type: ignore
    ProgramLevel = None  # type: ignore

# Inline documentation: Simple recommendation utility that reads processed KUCCPS
# CSVs (programs_deduped.csv or programs.csv), optionally checks eligibility using
# the existing ETL eligibility helper, and ranks programs based on user traits.

TRAIT_FIELD_HINTS: Dict[str, List[str]] = {
    # Investigative -> science/tech/engineering
    'Investigative': [
        'computer', 'technology', 'engineering', 'science', 'statistics', 'mathematics', 'informatics', 'software',
        'data', 'physics', 'chemistry', 'astronomy', 'astrophysics', 'space', 'aerospace', 'satellite', 'rocket',
    ],
    # Artistic -> design/art/media
    'Artistic': ['design', 'architecture', 'art', 'media', 'film', 'music', 'graphics'],
    # Social -> education/health/community
    'Social': ['education', 'nursing', 'medicine', 'community', 'social', 'counsel', 'psychology'],
    # Enterprising -> business/management/law
    'Enterprising': ['business', 'commerce', 'management', 'entrepreneur', 'marketing', 'law'],
    # Conventional -> accounting/finance/admin/logistics
    'Conventional': ['account', 'finance', 'procurement', 'logistics', 'administration', 'records'],
    # Realistic -> agriculture/mechanical/construction
    'Realistic': ['agriculture', 'mechanical', 'civil', 'construction', 'automotive', 'mechatronics', 'aviation', 'aerospace', 'aircraft'],
}


# Heuristic mapping from career paths to program keyword seeds
CAREER_PATH_KEYWORDS: Dict[str, List[str]] = {
    'astronomer': ['astronomy', 'astrophys', 'physics', 'space'],
    'astrophysicist': ['astrophys', 'astronomy', 'physics', 'space'],
    'space scientist': ['space science', 'astronomy', 'astrophys', 'physics'],
    'aerospace engineer': ['aerospace', 'aeronautical', 'aviation', 'aircraft', 'mechanical'],
    'optical engineer': ['optical', 'photonics', 'electronics', 'electrical', 'physics'],
    'data scientist': ['data science', 'statistics', 'computer science', 'informatics'],
    'software engineer': ['software', 'computer science', 'informatics'],
    'physicist': ['physics', 'applied physics', 'space'],
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


def _lookup_institutions_for_program_db(program_query: str, level: str = '', limit: int = 30) -> List[Dict[str, Any]]:
    """Use ORM to fetch institutions offering a program. Returns empty if DB unavailable."""
    if Program is None:
        return []
    q = (program_query or '').strip()
    if not q:
        return []
    lvl = (level or '').strip().lower()
    qs = Program.objects.all()
    if lvl:
        qs = qs.filter(level=lvl)
    # Match normalized_name contains query (case-insensitive)
    qs = qs.filter(normalized_name__icontains=q)
    qs = qs.select_related('institution', 'field').order_by('institution__name')

    # Group by institution and select the best matching program per institution
    def score_name(nm: str) -> tuple:
        n = (nm or '').strip().lower()
        ql = q.strip().lower()
        return (
            1 if n == ql else 0,           # exact match
            1 if n.startswith(ql) else 0,  # starts-with
            1 if ql in n else 0,           # contains
            -len(n),                        # prefer shorter names
        )

    best_by_inst: Dict[str, Dict[str, Any]] = {}
    for p in qs.iterator():
        inst = (p.institution.name or '').strip()
        key = str(p.institution_id)
        prog_name = (p.normalized_name or p.name or '').strip()
        candidate = {
            'institution_name': inst,
            'program_name': prog_name,
            'program_code': (p.code or '').strip(),
            'field_name': (p.field.name if p.field_id else ''),
            'level': (p.level or '').strip(),
            'region': (p.region or '').strip(),
            'campus': (p.campus or '').strip(),
            '_score': score_name(prog_name),
        }
        prev = best_by_inst.get(key)
        if (prev is None) or (candidate['_score'] > prev['_score']):
            best_by_inst[key] = candidate

    out = list(best_by_inst.values())
    for r in out:
        r.pop('_score', None)
    out.sort(key=lambda x: (x['institution_name'] or '').upper())
    if limit:
        out = out[:limit]
    return out


def lookup_institutions_for_program(program_query: str, level: str = '', limit: int = 30) -> List[Dict[str, Any]]:
    """Inline documentation: Catalog lookup to answer queries like
    "best universities that offer bachelor of arts" directly from KUCCPS CSV.

    Args:
        program_query: Free-text query for the program (e.g., 'bachelor of arts', 'bachelor of science in nursing').
        level: Optional level filter (e.g., 'bachelor', 'diploma'). Case-insensitive.
        limit: Max number of results to return.

    Returns a list of dictionaries each with institution/program info. Results are
    deduplicated by institution+program name and sorted by institution name.
    DB is preferred; falls back to CSV when DB data is unavailable.
    """
    # 1) Try DB-backed lookup first
    out_db = _lookup_institutions_for_program_db(program_query, level=level, limit=limit)
    if out_db:
        return out_db

    # 2) Fallback to CSV files
    q = (program_query or '').strip().upper()
    lvl = (level or '').strip().lower()
    if not q:
        return []
    rows = _load_program_rows()
    if not rows:
        return []

    def _canon_name(r: Dict[str, str]) -> str:
        return ((r.get('normalized_name') or r.get('name') or '').strip().upper())

    def _match(r: Dict[str, str]) -> bool:
        nm = _canon_name(r)
        if not nm:
            return False
        if lvl:
            rlevel = (r.get('level') or '').strip().lower()
            if rlevel and rlevel != lvl:
                return False
        # prefer contains match; allow exact equality when query is a full title
        return (q == nm) or (q in nm)

    # Group by institution; choose best matching program per institution
    def score_name2(nm: str) -> tuple:
        n = (nm or '').strip().lower()
        ql = (program_query or '').strip().lower()
        return (
            1 if n == ql else 0,
            1 if n.startswith(ql) else 0,
            1 if ql in n else 0,
            -len(n),
        )

    best_by_inst: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if not _match(r):
            continue
        inst = (r.get('institution_name') or '').strip()
        inst_code = (r.get('institution_code') or '').strip()
        prog = (r.get('normalized_name') or r.get('name') or '').strip()
        code = (r.get('program_code') or r.get('code') or '').strip()
        key = (inst_code or inst.upper())
        candidate = {
            'institution_name': inst,
            'institution_code': inst_code,
            'program_name': prog,
            'program_code': code,
            'field_name': (r.get('field_name') or '').strip(),
            'level': (r.get('level') or '').strip(),
            'region': (r.get('region') or '').strip(),
            'campus': (r.get('campus') or '').strip(),
            '_score': score_name2(prog),
        }
        prev = best_by_inst.get(key)
        if (prev is None) or (candidate['_score'] > prev['_score']):
            best_by_inst[key] = candidate

    out = list(best_by_inst.values())
    for r in out:
        r.pop('_score', None)
    out.sort(key=lambda x: (x['institution_name'] or '').upper())
    if limit:
        out = out[:limit]
    return out


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


def infer_career_paths(traits: Dict[str, float]) -> List[str]:
    # Inline: Heuristically map RIASEC traits to career families
    t = {k: float(v) for k, v in (traits or {}).items()}
    if not t:
        return []
    paths: List[str] = []
    # Investigative
    if t.get('Investigative', 0) >= 0.3:
        paths.append('Science & Engineering (e.g., Physics, Computer Science, Data Science)')
    # Artistic
    if t.get('Artistic', 0) >= 0.3:
        paths.append('Design & Creative (e.g., Architecture, Graphic Design, Media)')
    # Social
    if t.get('Social', 0) >= 0.3:
        paths.append('Health & Education (e.g., Nursing, Teaching, Community Work)')
    # Enterprising
    if t.get('Enterprising', 0) >= 0.3:
        paths.append('Business & Leadership (e.g., Commerce, Management, Marketing)')
    # Conventional
    if t.get('Conventional', 0) >= 0.3:
        paths.append('Finance & Operations (e.g., Accounting, Logistics, Administration)')
    # Realistic
    if t.get('Realistic', 0) >= 0.3:
        paths.append('Technical & Mechanical (e.g., Mechatronics, Civil, Aviation)')
    # If none pass threshold, return the top 2 traits by score with generic names
    if not paths:
        top = sorted(t.items(), key=lambda kv: -kv[1])[:2]
        for k, _ in top:
            paths.append(f"{k} pathways")
    return paths[:4]


def suggest_program_titles(career_paths: List[str], traits: Dict[str, float], limit: int = 12) -> List[str]:
    """Suggest Bachelor program titles aligned to career paths/traits.
    Prefers DB Program records and dedupes by normalized_name; falls back to CSV.
    """
    kws: List[str] = []
    for p in (career_paths or []):
        key = (p or '').strip().lower()
        for ck, seeds in CAREER_PATH_KEYWORDS.items():
            if ck in key:
                kws.extend(seeds)
    # If no seeds from paths, derive from top trait hints
    if not kws:
        for tname, weight in sorted((traits or {}).items(), key=lambda kv: -float(kv[1])):
            if weight and tname in TRAIT_FIELD_HINTS:
                # take first few hints
                kws.extend(TRAIT_FIELD_HINTS[tname][:4])
                break
    # Normalize keywords
    kwset = sorted({k.strip().lower() for k in kws if k and len(k.strip()) >= 3})

    def score_nm(nm: str) -> int:
        n = (nm or '').strip().lower()
        return sum(1 for k in kwset if k in n)

    out_names: List[str] = []

    # DB path
    if Program is not None and kwset:
        try:
            qs = Program.objects.filter(level='bachelor')
            names = {}
            for kw in kwset:
                for p in qs.filter(normalized_name__icontains=kw).values_list('normalized_name', flat=True)[:limit*2]:
                    nm = (p or '').strip()
                    if not nm:
                        continue
                    names[nm] = max(names.get(nm, 0), score_nm(nm))
            out_names = [nm for nm, _ in sorted(names.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]
        except Exception:
            out_names = []

    # CSV fallback
    if not out_names and kwset:
        rows = _load_program_rows()
        names2: Dict[str, int] = {}
        for r in rows:
            lvl = (r.get('level') or '').strip().lower()
            if lvl and lvl not in ('bachelor', 'bachelors', 'degree', ''):
                continue
            nm = (r.get('normalized_name') or r.get('name') or '').strip()
            if not nm:
                continue
            s = score_nm(nm)
            if s > 0:
                names2[nm] = max(names2.get(nm, 0), s)
        out_names = [nm for nm, _ in sorted(names2.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]

    return out_names

def recommend_top_k(grades: Dict[str, str], traits: Dict[str, float], k: int = 5) -> List[Dict[str, Any]]:
    # Inline: Recommend top-K programs filtered by eligibility (when available) and scored by traits
    rows = _load_program_rows()
    if not rows:
        return []
    eval_elig = _eligibility_checker()

    def _rank(use_eligibility: bool) -> List[Tuple[float, Dict[str, str]]]:
        scored_inner: List[Tuple[float, Dict[str, str]]] = []
        for r in rows:
            name = (r.get('normalized_name') or r.get('name') or '').strip()
            field_name = (r.get('field_name') or '').strip()
            level = (r.get('level') or '').strip().lower()
            code = (r.get('program_code') or r.get('code') or '').strip()
            if not name or not code:
                continue
            # Prefer undergrad programs, but do not exclude if empty/unknown
            if level and level not in ('bachelor', 'bachelors', 'degree', ''):
                continue
            # Eligibility (best-effort)
            if use_eligibility and eval_elig and grades:
                try:
                    ok = eval_elig(r, grades)
                    if not (ok or (isinstance(ok, dict) and (ok.get('eligible') or ok.get('eligible') is True))):
                        continue
                except Exception:
                    # Ignore evaluator failures
                    pass
            score = _score_by_traits(name, field_name, traits)
            if not traits and grades:
                score += 0.2
            # Keyword boost for space/rockets/astronomy/aerospace
            txt = f"{name} {field_name}".lower()
            if any(kw in txt for kw in ['astronomy', 'astrophysics', 'space', 'aerospace', 'rocket', 'satellite', 'aviation']):
                score += 0.5
            scored_inner.append((score, r))
        return scored_inner

    # Pass 1: with eligibility (if available)
    scored = _rank(use_eligibility=True)
    # If empty, Pass 2: without eligibility gating (always show something)
    if not scored:
        scored = _rank(use_eligibility=False)
    # If still empty (very unlikely), coarse keyword-based picks
    if not scored:
        coarse: List[Tuple[float, Dict[str, str]]] = []
        for r in rows:
            name = (r.get('normalized_name') or r.get('name') or '').strip()
            field_name = (r.get('field_name') or '').strip()
            code = (r.get('program_code') or r.get('code') or '').strip()
            if not name or not code:
                continue
            txt = f"{name} {field_name}".lower()
            base = 0.0
            if any(kw in txt for kw in ['physics', 'astronomy', 'astrophysics', 'aerospace', 'engineering']):
                base = 0.6
            elif any(kw in txt for kw in ['computer science', 'software', 'data']):
                base = 0.5
            if base > 0:
                coarse.append((base, r))
        scored = coarse or [(0.1, r) for r in rows[:k]]

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
