import csv
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
try:
    from django.db.models import Q
except Exception:
    Q = None
try:
    # Prefer DB-backed catalog when available
    from catalog.models import Institution, Program, ProgramLevel, InstitutionCampus  # type: ignore
except Exception:  # pragma: no cover - during migrations or no DB
    Institution = None  # type: ignore
    Program = None  # type: ignore
    ProgramLevel = None  # type: ignore
    InstitutionCampus = None  # type: ignore

# Inline documentation: Simple recommendation utility that reads processed KUCCPS
# CSVs (programs_deduped.csv or programs.csv), optionally checks eligibility using
# the existing ETL eligibility helper, and ranks programs based on user traits.

TRAIT_FIELD_HINTS: Dict[str, List[str]] = {
    'Investigative': [
        'computer', 'technology', 'engineering', 'science', 'statistics', 'mathematics', 'informatics', 'software',
        'data', 'physics', 'chemistry', 'astronomy', 'astrophysics', 'space', 'aerospace', 'satellite', 'rocket',
    ],
    'Artistic': ['design', 'architecture', 'art', 'media', 'film', 'music', 'graphics'],
    'Social': ['education', 'nursing', 'medicine', 'community', 'social', 'counsel', 'psychology'],
    'Enterprising': ['business', 'commerce', 'management', 'entrepreneur', 'marketing', 'law'],
    'Conventional': ['account', 'finance', 'procurement', 'logistics', 'administration', 'records'],
    'Realistic': ['agriculture', 'mechanical', 'civil', 'construction', 'automotive', 'mechatronics', 'aviation', 'aerospace', 'aircraft'],
}

CAREER_PATH_KEYWORDS: Dict[str, List[str]] = {
    'astronomer': ['astronomy', 'astrophys', 'physics', 'space'],
    'astrophysicist': ['astrophys', 'astronomy', 'physics', 'space'],
    'space scientist': ['space science', 'astronomy', 'astrophys', 'physics'],
    'aerospace engineer': ['aerospace', 'aeronautical', 'aviation', 'aircraft', 'mechanical'],
    'optical engineer': ['optical', 'photonics', 'electronics', 'electrical', 'physics'],
    'data scientist': ['data science', 'statistics', 'computer science', 'informatics'],
    'software engineer': ['software', 'computer science', 'informatics'],
    'physicist': ['physics', 'applied physics', 'space'],
    'musician': ['music', 'musician', 'performance', 'instrument', 'singing'],
    'composer': ['music', 'composition', 'composer', 'orchestration'],
    'songwriter': ['music', 'songwriting', 'songwriter', 'lyrics'],
    'music producer': ['music', 'production', 'producer', 'recording', 'studio'],
    'sound engineer': ['audio', 'sound', 'recording', 'studio', 'music'],
    'music teacher': ['music', 'education', 'teaching', 'teacher'],
}

def infer_career_paths(traits: Dict[str, float], limit: int = 8) -> List[str]:
    traits = traits or {}
    try:
        traits_items = [(str(k), float(v or 0.0)) for k, v in traits.items()]
    except Exception:
        traits_items = [(str(k), 0.0) for k in (traits or {}).keys()]
    traits_items.sort(key=lambda kv: -kv[1])
    traits = {k: v for k, v in traits_items[:4] if v and v > 0}
    weighted_terms: List[Tuple[float, str]] = []
    for tname, score in traits.items():
        try:
            sc = float(score)
        except Exception:
            sc = 0.0
        if sc <= 0:
            continue
        for h in (TRAIT_FIELD_HINTS.get(str(tname), []) or []):
            weighted_terms.append((sc, str(h).lower()))

    terms = [t for _w, t in sorted(weighted_terms, key=lambda x: -x[0])]
    term_set = set(terms)

    scored: List[Tuple[float, str]] = []
    for career, kws in (CAREER_PATH_KEYWORDS or {}).items():
        score = 0.0
        for kw in kws:
            k = str(kw).lower()
            if k in term_set:
                score += 1.0
            elif any(k in t or t in k for t in terms[:50]):
                score += 0.5
        if score > 0:
            scored.append((score, career))

    if not scored:
        top_trait = ''
        try:
            top_trait = max(traits.items(), key=lambda kv: float(kv[1] or 0.0))[0]
        except Exception:
            top_trait = ''
        tt = str(top_trait).lower()
        if tt in ('investigative', 'realistic'):
            scored = [(1.0, 'software engineer'), (0.9, 'data scientist'), (0.8, 'physicist')]
        elif tt in ('artistic',):
            scored = [(1.0, 'graphic designer'), (0.9, 'architect')]
        elif tt in ('social',):
            scored = [(1.0, 'teacher'), (0.9, 'counselor')]
        elif tt in ('enterprising',):
            scored = [(1.0, 'entrepreneur'), (0.9, 'business manager')]
        elif tt in ('conventional',):
            scored = [(1.0, 'accountant'), (0.9, 'procurement officer')]

    scored.sort(key=lambda x: (-x[0], x[1]))
    out = [name for _s, name in scored]
    if limit:
        out = out[: max(1, int(limit))]
    return out

def _processed_dir() -> Path:
    env_dir = os.getenv('KUCCPS_PROCESSED_DIR')
    if env_dir:
        return Path(env_dir).resolve()
    backend_dir = Path(__file__).resolve().parents[2]
    return backend_dir / 'scripts' / 'etl' / 'kuccps' / 'processed'

def _mappings_dir() -> Path:
    backend_dir = Path(__file__).resolve().parents[2]
    return backend_dir / 'scripts' / 'etl' / 'kuccps' / 'mappings'

def _programs_path() -> Path:
    base = _processed_dir()
    p = base / 'programs_deduped.csv'
    return p if p.exists() else (base / 'programs.csv')

def _institutions_path() -> Path:
    base = _processed_dir()
    return base / 'institutions.csv'

def _institutions_geo_path() -> Path:
    return _mappings_dir() / 'institutions_geo.csv'

def _load_institution_rows(limit: int = 0) -> List[Dict[str, str]]:
    fp = _institutions_path()
    rows: List[Dict[str, str]] = []
    if not fp.exists():
        return rows
    with open(fp, encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for _i, r in enumerate(rdr):
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows

def _load_institutions_geo_rows(limit: int = 0) -> List[Dict[str, str]]:
    fp = _institutions_geo_path()
    rows: List[Dict[str, str]] = []
    if not fp.exists():
        return rows
    with open(fp, encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for _i, r in enumerate(rdr):
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows

def _normalize_region_query(region: str) -> str:
    s = (region or '').strip()
    if not s:
        return ''
    low = s.lower().strip()
    low = re.sub(r"\b(region|county)\b", " ", low).strip()
    low = re.sub(r"\s+", " ", low)
    synonyms = {
        'mt kenya': 'Central',
        'mount kenya': 'Central',
        'central kenya': 'Central',
        'rift': 'Rift Valley',
        'riftvalley': 'Rift Valley',
        'north eastern': 'North Eastern',
        'northeastern': 'North Eastern',
    }
    if low in synonyms:
        return synonyms[low]
    return low.title()

def _score_by_traits(name: str, field_name: str, traits: Dict[str, float]) -> float:
    text = f"{name} {field_name}".lower()
    score = 0.0
    for trait, weight in (traits or {}).items():
        try:
            w = float(weight or 0.0)
        except Exception:
            w = 0.0
        if w <= 0:
            continue
        hints = TRAIT_FIELD_HINTS.get(str(trait), [])
        hits = sum(1 for h in hints if str(h).lower() in text)
        if hits > 0:
            score += w
    return float(score)

def _eligibility_checker():
    """Return a best-effort eligibility evaluator for KUCCPS rows.

    If unavailable, returns None.
    """
    # Try normal import first (works if scripts is on PYTHONPATH and has __init__.py)
    try:
        from scripts.etl.kuccps.eligibility import evaluate_eligibility  # type: ignore
        return lambda row, grades: evaluate_eligibility(row, grades)
    except Exception:
        pass

    # Fallback: load by file path (works even if scripts isn't a package)
    try:
        backend_dir = Path(__file__).resolve().parents[1]
        fp = backend_dir / 'scripts' / 'etl' / 'kuccps' / 'eligibility.py'
        if not fp.exists():
            return None
        spec = importlib.util.spec_from_file_location('kuccps_eligibility', str(fp))
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        fn = getattr(mod, 'evaluate_eligibility', None)
        if not fn:
            return None
        return lambda row, grades: fn(row, grades)
    except Exception:
        return None

def lookup_institutions_for_program(program_query: str, level: str = '', limit: int = 30) -> List[Dict[str, Any]]:
    """Find institutions offering a program (DB-only)."""
    qtext = (program_query or '').strip()
    if not qtext:
        return []
    lvl = (level or '').strip().lower()

    out: List[Dict[str, Any]] = []
    seen = set()

    # DB-backed lookup
    if Program is not None:
        try:
            qs = Program.objects.select_related('institution')
            if lvl:
                qs = qs.filter(level__icontains=lvl)
            if Q is not None:
                qs = qs.filter(Q(normalized_name__icontains=qtext) | Q(name__icontains=qtext))
            else:
                qs = qs.filter(normalized_name__icontains=qtext)
            qs = qs.order_by('institution__name', 'normalized_name')[:500]
            for p in qs:
                prog_name = (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip()
                prog_code = (getattr(p, 'code', '') or '').strip()
                inst_name = ''
                inst_code = ''
                try:
                    inst = getattr(p, 'institution', None)
                    if inst is not None:
                        inst_name = (getattr(inst, 'name', '') or '').strip()
                        inst_code = (getattr(inst, 'code', '') or '').strip()
                except Exception:
                    inst_name = inst_name or ''
                key = (inst_code or inst_name.upper()) + '|' + (prog_code or prog_name.upper())
                if not inst_name or not prog_name or key in seen:
                    continue
                seen.add(key)
                out.append({
                    'program_code': prog_code,
                    'program_name': prog_name,
                    'institution_name': inst_name,
                    'institution_code': inst_code,
                    'region': (getattr(p, 'region', '') or '').strip(),
                    'campus': (getattr(p, 'campus', '') or '').strip(),
                    'level': (getattr(p, 'level', '') or '').strip(),
                })
                if limit and len(out) >= int(limit):
                    break
            if out:
                return out
        except Exception:
            out = []
    return out

def suggest_program_titles(career_paths: List[str], traits: Dict[str, float], limit: int = 8) -> List[str]:
    """Suggest program titles matching inferred career paths and traits."""
    terms: List[str] = []
    for cp in (career_paths or []):
        k = str(cp).strip().lower()
        if not k:
            continue
        if k in CAREER_PATH_KEYWORDS:
            terms.extend([str(x).lower() for x in (CAREER_PATH_KEYWORDS.get(k) or [])])
        terms.extend([t for t in re.split(r"[^a-z0-9]+", k) if len(t) >= 4])

    traits_sorted: List[Tuple[str, float]] = []
    for t, v in (traits or {}).items():
        try:
            traits_sorted.append((str(t), float(v or 0.0)))
        except Exception:
            traits_sorted.append((str(t), 0.0))
    traits_sorted.sort(key=lambda kv: -kv[1])
    for t, _w in traits_sorted[:3]:
        terms.extend([str(h).lower() for h in (TRAIT_FIELD_HINTS.get(t, []) or [])[:6]])

    terms = [t for t in terms if t and len(t) >= 3]
    if not terms:
        return []

    seen = set()
    out: List[str] = []

    # DB-backed title suggestions
    if Program is not None and Q is not None:
        try:
            q = Q()
            for t in sorted(set(terms))[:12]:
                q |= Q(normalized_name__icontains=t) | Q(name__icontains=t)
            qs = Program.objects.filter(q).order_by('normalized_name')[:2000]
            for p in qs:
                nm = (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip()
                key = nm.upper()
                if not nm or key in seen:
                    continue
                seen.add(key)
                out.append(nm)
                if limit and len(out) >= int(limit):
                    break
            if out:
                return out
        except Exception:
            out = []
    return out

def lookup_institutions_by_region(region: str, limit: int = 30) -> List[Dict[str, Any]]:
    """List institutions in a region/county/area.

    Used for questions like: "Name 3 universities in the central region".

    Prefers DB-backed Institution.
    """
    r = _normalize_region_query(region)
    if not r:
        return []

    if Institution is not None:
        out_union: List[Dict[str, Any]] = []
        seen = set()
        try:
            if Q is not None:
                qs = Institution.objects.filter(Q(region__icontains=r) | Q(county__icontains=r)).order_by('name')
                for inst in qs[: max(1, int(limit or 30))]:
                    code = (inst.code or '').strip()
                    key = code or (inst.name or '').strip().upper()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out_union.append({
                        'institution_name': (inst.name or '').strip(),
                        'institution_code': code,
                        'region': (inst.region or '').strip(),
                        'county': (inst.county or '').strip(),
                        'website': (inst.website or '').strip(),
                    })
            else:
                raise Exception('Q unavailable')
        except Exception:
            try:
                qs1 = Institution.objects.filter(region__icontains=r).order_by('name')
            except Exception:
                qs1 = Institution.objects.none()
            try:
                qs2 = Institution.objects.filter(county__icontains=r).order_by('name')
            except Exception:
                qs2 = Institution.objects.none()
            for inst in list(qs1[: max(1, int(limit or 30))]) + list(qs2[: max(1, int(limit or 30))]):
                code = (inst.code or '').strip()
                key = code or (inst.name or '').strip().upper()
                if not key or key in seen:
                    continue
                seen.add(key)
                out_union.append({
                    'institution_name': (inst.name or '').strip(),
                    'institution_code': code,
                    'region': (inst.region or '').strip(),
                    'county': (inst.county or '').strip(),
                    'website': (inst.website or '').strip(),
                })

        if out_union:
            out_union.sort(key=lambda x: (x.get('institution_name') or '').upper())
            return out_union

        if InstitutionCampus is not None:
            seen2 = set()
            outc: List[Dict[str, Any]] = []
            try:
                if Q is not None:
                    cqs = InstitutionCampus.objects.select_related('institution').filter(Q(region__icontains=r) | Q(county__icontains=r))
                else:
                    cqs = InstitutionCampus.objects.select_related('institution').filter(region__icontains=r)
            except Exception:
                cqs = []
            for c in list(cqs)[: max(1, int(limit or 30))]:
                inst = getattr(c, 'institution', None)
                if not inst:
                    continue
                code = (inst.code or '').strip()
                key = code or (inst.name or '').strip().upper()
                if not key or key in seen2:
                    continue
                seen2.add(key)
                outc.append({
                    'institution_name': (inst.name or '').strip(),
                    'institution_code': code,
                    'region': (c.region or '').strip() or (inst.region or '').strip(),
                    'county': (c.county or '').strip() or (inst.county or '').strip(),
                    'website': (inst.website or '').strip(),
                })
                if limit and len(outc) >= int(limit):
                    break
            if outc:
                outc.sort(key=lambda x: (x.get('institution_name') or '').upper())
                return outc

    return []

def _recommend_top_k_db(grades: Dict[str, str], traits: Dict[str, float], k: int = 5, goal_text: str = '') -> List[Dict[str, Any]]:
    if Program is None:
        return []
    try:
        qs = Program.objects.select_related('institution', 'field').filter(level='bachelor')
    except Exception:
        return []

    scored: List[Tuple[float, Any]] = []

    traits_sorted: List[Tuple[str, float]] = []
    for t, v in (traits or {}).items():
        try:
            traits_sorted.append((str(t), float(v)))
        except Exception:
            traits_sorted.append((str(t), 0.0))
    traits_sorted.sort(key=lambda kv: -kv[1])

    toks2: List[str] = []
    gt = (goal_text or '').strip().lower()
    if gt:
        toks = [t for t in ''.join((ch if ch.isalnum() else ' ') for ch in gt).split() if len(t) >= 4]
        if toks:
            stop = {'become', 'becoming', 'want', 'wants', 'would', 'like', 'study', 'studying', 'career', 'goal', 'goals', 'work'}
            toks2 = [t for t in toks if t not in stop]

    if Q is not None and (traits_sorted or toks2):
        hints: List[str] = []
        for t, _w in traits_sorted[:3]:
            hints.extend((TRAIT_FIELD_HINTS.get(t, []) or [])[:4])
        if toks2:
            hints.extend(toks2[:10])
        hints = [h for h in hints if h and len(h) >= 3]
        if hints:
            q = Q()
            for h in sorted(set(hints))[:10]:
                q |= Q(normalized_name__icontains=h) | Q(name__icontains=h) | Q(field__name__icontains=h)
            try:
                narrowed = qs.filter(q)
                if narrowed.exists():
                    qs = narrowed
            except Exception:
                pass

    for p in qs[:2000]:
        nm = (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip()
        if not nm:
            continue
        field_name = ''
        try:
            if getattr(p, 'field_id', None):
                field_name = (p.field.name or '').strip()
        except Exception:
            field_name = ''
        sc = _score_by_traits(nm, field_name, traits or {})
        if not traits and grades:
            sc += 0.2
        if toks2:
            txt = f"{nm} {field_name}".lower()
            hits2 = sum(1 for t in set(toks2[:12]) if t in txt)
            if hits2 > 0:
                sc += min(1.0, 0.25 * float(hits2))
        scored.append((float(sc), p))

    if not scored:
        return []
    scored.sort(key=lambda t: (-t[0], (getattr(t[1], 'normalized_name', '') or getattr(t[1], 'name', '') or '')))
    top = scored[: max(1, int(k or 5))]

    out: List[Dict[str, Any]] = []
    for sc, p in top:
        inst_name = ''
        try:
            inst_name = (p.institution.name or '').strip() if getattr(p, 'institution_id', None) else ''
        except Exception:
            inst_name = ''
        out.append({
            'program_id': getattr(p, 'id', None),
            'program_code': (getattr(p, 'code', '') or '').strip(),
            'program_name': (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip(),
            'institution_name': inst_name,
            'field_name': (p.field.name if getattr(p, 'field_id', None) else '') or '',
            'level': (getattr(p, 'level', '') or '').strip(),
            'score': round(float(sc), 3),
        })
    return out


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


def recommend_top_k(grades: Dict[str, str], traits: Dict[str, float], k: int = 5, goal_text: str = '') -> List[Dict[str, Any]]:
    # Inline: Recommend top-K programs filtered by eligibility (when available) and scored by traits
    try:
        if Program is not None:
            out_db = _recommend_top_k_db(grades or {}, traits or {}, k=k, goal_text=goal_text)
            if out_db:
                return out_db
    except Exception:
        pass
    return []
