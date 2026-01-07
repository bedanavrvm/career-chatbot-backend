import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from django.conf import settings

# Lightweight NLP utilities (no heavy external deps). Optional LLM provider can be added.

# Canonical subject codes and common synonyms
_SUBJECT_SYNONYMS = {
    'math': 'MAT', 'mathematics': 'MAT', 'mat': 'MAT', 'mah': 'MAT',
    'english': 'ENG', 'eng': 'ENG', 'lang': 'ENG',
    'kiswahili': 'KIS', 'swahili': 'KIS', 'kisw': 'KIS', 'kis': 'KIS',
    'chemistry': 'CHE', 'chem': 'CHE',
    'physics': 'PHY', 'phy': 'PHY',
    'biology': 'BIO', 'bio': 'BIO',
    'history': 'HIS', 'hist': 'HIS',
    'geography': 'GEO', 'geo': 'GEO',
    'business': 'BUS', 'commerce': 'BUS', 'bus': 'BUS',
    'computer': 'COM', 'computing': 'COM', 'ict': 'COM', 'cs': 'COM',
    'cre': 'CRE', 'ire': 'IRE', 'agric': 'AGR', 'agriculture': 'AGR', 'agr': 'AGR',
    'music': 'MUS', 'mus': 'MUS', 'muc': 'MUS',
    'german': 'GER', 'ger': 'GER',
    'french': 'FRE', 'fre': 'FRE',
    'woodwork': 'WWK', 'wood': 'WWK', 'wwk': 'WWK',
    'art': 'ART', 'artanddesign': 'ART', 'design': 'ART', 'ard': 'ART',
    'computers': 'CSC', 'computerstudies': 'CSC', 'comp': 'CSC', 'csc': 'CSC',
}

# Grades pattern (KCSE-like): A, A-, B+, B, B-, C+, C, C-, D+, D, E
# Match grades like A-, B+, including when followed by punctuation. Avoid trailing word chars.
_GRADE_RE = re.compile(r"(?<![A-Za-z])(?:A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E)(?![A-Za-z])", re.IGNORECASE)

# Split tokens and punctuation
_TOKEN_RE = re.compile(r"[A-Za-z+\-]+|\d+|[.,;:/]")


def _norm_subject_token(tok: str) -> str:
    t = tok.strip().lower()
    return _SUBJECT_SYNONYMS.get(t, t.upper())


def extract_catalog_lookup(text: str) -> Dict[str, Any]:
    """Extract a catalog lookup query from free text.
    Returns a dict like {'program_query': 'bachelor of arts', 'level': 'bachelor'} when detected; else {}.
    This supports phrasings such as:
      - "best universities that offer the course bachelor of arts"
      - "which universities offer bachelor of science in computer science"
      - "where can I study diploma in nursing"
      - "are there programs on music"
    """
    s = (text or '').strip()
    if not s:
        return {}
    low = s.lower()
    # Quick gate: allow (a) provider+offer context, (b) explicit degree phrases,
    # or (c) generic "programs/programmes/courses in|on|for <field>" questions.
    if not ((('universit' in low or 'college' in low or 'institute' in low) and ('offer' in low or 'study' in low or 'offers' in low))):
        if not re.search(r"\b(bachelor|masters?|diploma|certificate)\b", low):
            if not re.search(r"\b(programs?|programmes?|courses?)\s+(in|on|for|about|related\s+to)\s+", low):
                if not re.search(r"\b(programs?|programmes?|courses?)\s+in\s+the\s+field\s+of\s+", low):
                    return {}

    # Capture explicit degree phrase first
    m = re.search(r"\b(bachelor|masters?|diploma|certificate)\s+of\s+([A-Za-z &/\-]+)", s, flags=re.IGNORECASE)
    if m:
        level = m.group(1).lower()
        field = re.sub(r"\s+", " ", m.group(2)).strip()
        prog = f"{level} of {field}".lower()
        return {'program_query': prog, 'level': level}

    # Fallback: after 'course' or 'program' tokens
    m2 = re.search(r"\b(course|program|degree)\s+(in|on|of)?\s*([A-Za-z &/\-]+)$", s, flags=re.IGNORECASE)
    if m2:
        phrase = re.sub(r"\s+", " ", m2.group(3)).strip().lower()
        # If phrase already contains level-like words, keep; else leave level empty
        lvl = 'bachelor' if 'bachelor' in phrase else ('diploma' if 'diploma' in phrase else '')
        return {'program_query': phrase, 'level': lvl}

    # Fallback: after 'offer ...' capture remainder
    m3 = re.search(r"offer[s]?\s+(the\s+)?(course|program|degree)?\s*([A-Za-z ].+)$", s, flags=re.IGNORECASE)
    if m3:
        phrase = re.sub(r"\s+", " ", m3.group(3)).strip().lower()
        lvl = 'bachelor' if phrase.startswith('bachelor') else ('diploma' if phrase.startswith('diploma') else '')
        return {'program_query': phrase, 'level': lvl}

    # Generic: "programs/courses in|on|for <field>"
    m4 = re.search(r"\b(programs?|programmes?|courses?)\s+(in|on|for|about)\s+([A-Za-z &/\-]+)", s, flags=re.IGNORECASE)
    if m4:
        phrase = re.sub(r"\s+", " ", m4.group(3)).strip().lower()
        if phrase in {'doctor', 'doctors', 'medicine doctor', 'medical doctor'}:
            phrase = 'medicine'
        return {'program_query': phrase, 'level': ''}

    # Generic: "programs related to <field>" / "courses related to <field>"
    m5 = re.search(r"\b(programs?|programmes?|courses?)\s+related\s+to\s+([A-Za-z &/\-]+)", s, flags=re.IGNORECASE)
    if m5:
        phrase = re.sub(r"\s+", " ", m5.group(2)).strip().lower()
        if phrase in {'doctor', 'doctors', 'medicine doctor', 'medical doctor'}:
            phrase = 'medicine'
        return {'program_query': phrase, 'level': ''}

    # Generic: "programs in the field of <field>"
    m6 = re.search(r"\b(programs?|programmes?|courses?)\s+in\s+the\s+field\s+of\s+([A-Za-z &/\-]+)", s, flags=re.IGNORECASE)
    if m6:
        phrase = re.sub(r"\s+", " ", m6.group(2)).strip().lower()
        if phrase in {'doctor', 'doctors', 'medicine doctor', 'medical doctor'}:
            phrase = 'medicine'
        return {'program_query': phrase, 'level': ''}

    return {}


def extract_subject_grade_pairs(text: str) -> Dict[str, str]:
    """Extract subject-grade pairs from free text.
    Supports patterns like:
      - "Math A-", "ENG: B+", "Kisw B, Chem C+"
      - "Physics B minus" (minus/plus words are normalized)
    Returns mapping SUBJECT_CODE -> GRADE (uppercase, e.g., 'A-', 'B+')
    """
    s = (text or '').strip()
    if not s:
        return {}

    # Normalize spelled plus/minus
    s = re.sub(r"\bplus\b", "+", s, flags=re.IGNORECASE)
    s = re.sub(r"\bminus\b", "-", s, flags=re.IGNORECASE)

    # Tokenize and scan: subject token followed by grade token
    toks = _TOKEN_RE.findall(s)
    out: Dict[str, str] = {}
    for i, tok in enumerate(toks):
        subj = _norm_subject_token(tok)
        # Must look like a subject code after normalization
        if subj in _SUBJECT_SYNONYMS.values():
            # Look ahead for grade token
            window = " ".join(toks[i+1:i+4])
            m = _GRADE_RE.search(window)
            if m:
                grade = m.group(0).upper()
                # Normalize ordering such as 'B +' -> 'B+'
                grade = grade.replace(" ", "")
                out[subj] = grade
                continue
        # Fallback for abbreviations like 'ENG:B+'
        if tok.endswith(":") and i + 1 < len(toks):
            subj2 = _norm_subject_token(tok[:-1])
            nxt = toks[i+1]
            m2 = _GRADE_RE.fullmatch(nxt)
            if m2:
                out[subj2] = m2.group(0).upper()
    return out


_RIASEC_KEYWORDS = {
    'Realistic': ['build', 'fix', 'tools', 'machines', 'outdoors', 'hands-on'],
    'Investigative': ['research', 'analyze', 'code', 'coding', 'compute', 'experiment', 'science'],
    'Artistic': ['design', 'art', 'music', 'musical', 'instrument', 'instruments', 'creative', 'write', 'draw'],
    'Social': ['help', 'teach', 'care', 'community', 'volunteer'],
    'Enterprising': ['lead', 'business', 'sell', 'entrepreneur', 'startup', 'pitch'],
    'Conventional': ['organize', 'accounting', 'plan', 'data entry', 'schedule'],
}


def extract_traits(text: str) -> Dict[str, float]:
    """Simple keyword-based RIASEC trait scoring. Returns scores per category [0..1]."""
    s = (text or '').lower()
    if not s:
        return {}
    scores: Dict[str, float] = {}
    for trait, kws in _RIASEC_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in s)
        # Normalize by 3 to cap quickly; clamp to [0,1]
        scores[trait] = min(1.0, hits / 3.0)
    return {k: v for k, v in scores.items() if v > 0}


_KNOWN_REGIONS = [
    'central',
    'eastern',
    'western',
    'nairobi',
    'coast',
    'rift valley',
    'nyanza',
    'north eastern',
]

_COUNTIES_CACHE: List[str] = []


def _backend_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _institutions_geo_path() -> Path:
    return _backend_dir() / 'scripts' / 'etl' / 'kuccps' / 'mappings' / 'institutions_geo.csv'


def _load_counties() -> List[str]:
    global _COUNTIES_CACHE
    if _COUNTIES_CACHE:
        return _COUNTIES_CACHE
    fp = _institutions_geo_path()
    if not fp.exists():
        _COUNTIES_CACHE = []
        return _COUNTIES_CACHE
    out: List[str] = []
    seen = set()
    try:
        with open(fp, encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                c = (row.get('county') or '').strip()
                if not c:
                    continue
                key = c.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
    except Exception:
        out = []
    out.sort(key=lambda x: x.lower())
    _COUNTIES_CACHE = out
    return _COUNTIES_CACHE


def extract_region_query(text: str) -> str:
    """Extract a region name from user text (best-effort)."""
    s = (text or '').strip().lower()
    if not s:
        return ''
    for r in _KNOWN_REGIONS:
        if r in s:
            return r.title()
    # County fallback (so queries like "universities in Kiambu" work)
    s_clean = re.sub(r"[^a-z]+", "", s)
    for c in _load_counties():
        cl = (c or '').strip().lower()
        if not cl:
            continue
        if cl in s:
            return c
        cl_clean = re.sub(r"[^a-z]+", "", cl)
        if cl_clean and cl_clean in s_clean:
            return c
    return ''


def detect_intents(text: str, grades: Dict[str, str]) -> List[str]:
    s = (text or '').lower()
    intents: List[str] = []
    if re.search(r"\b(hi|hello|hey)\b", s):
        intents.append('greeting')
    if grades:
        intents.append('provide_grades')
    if any(w in s for w in ["interests", "like", "enjoy", "love", "prefer"]):
        intents.append('interests')
    if any(w in s for w in ["help", "assist", "support"]):
        intents.append('help')
    if re.search(r"\bwhy\b", s) or s.strip() in ('why?', 'why'):
        intents.append('explain')
    if (
        any(k in s for k in ['qualify', 'qualified', 'qualification', 'eligible', 'eligibility'])
        or re.search(r"\bdo i qualify\b|\bam i eligible\b", s)
        or re.search(r"\bam i\s+(?:eligible|qualified)\s+for\b", s)
        or re.search(r"\bcan i\s+(?:do|study|pursue|take|apply\s+for)\s+\b", s)
    ):
        intents.append('qualify')
    if re.search(r"\bcareer\s+paths?\b", s) or any(k in s for k in ['possible career', 'career options', 'career direction']):
        intents.append('career_paths')
    if any(w in s for w in ["recommend", "suggest", "results", "options", "program", "programs", "career", "careers", "career path", "paths", "path", "job", "jobs", "profession", "occupations"]):
        intents.append('recommend')
    if (
        any(k in s for k in [
            'near me', 'near my', 'nearby', 'close to my', 'close to', 'closest', 'around me', 'my home', 'home location'
        ])
        and any(k in s for k in ['program', 'programs', 'course', 'courses'])
    ):
        intents.append('programs_near_me')
    if any(w in s.split() for w in ["next", "go", "continue"]):
        intents.append('next')
    if any(w in s for w in ["grade", "grades", "kcse"]):
        intents.append('ask_grades')
    # Catalog/program lookup intent: look for provider+offer phrasing or degree phrases
    if ((('universit' in s or 'college' in s or 'institute' in s) and ('offer' in s or 'offers' in s or 'study' in s))
         or re.search(r"\b(bachelor|masters?|diploma|certificate)\s+of\b", s)
         or re.search(r"\b(programs?|programmes?|courses?)\s+(in|on|for|about|related\s+to)\s+", s)
         or re.search(r"\b(programs?|programmes?|courses?)\s+in\s+the\s+field\s+of\s+", s)
         or any(k in s for k in [
             'cutoff', 'cut off', 'cut-offs', 'points',
             'requirements', 'requirement', 'cluster', 'subjects',
             'fees', 'fee', 'tuition', 'cost', 'price'
         ]
        )
    ):
        intents.append('catalog_lookup')

    # Institutions-by-region intent: "universities in the central region"
    if ('universit' in s or 'college' in s or 'institute' in s) and (('region' in s) or extract_region_query(s)):
        intents.append('institutions_by_region')
    return intents


def _normalize_intents(intents: Any) -> List[str]:
    out: List[str] = []
    if not intents:
        return out
    try:
        items = list(intents) if isinstance(intents, (list, tuple, set)) else [intents]
    except Exception:
        items = [intents]

    mapping = {
        'recommendations': 'recommend',
        'recommendation': 'recommend',
        'career_path': 'recommend',
        'career_paths': 'career_paths',
        'careerpath': 'recommend',
        'careerpaths': 'recommend',
        'career_guidance': 'recommend',
        'career_guidance_recommendations': 'recommend',
        'program_lookup': 'catalog_lookup',
        'programs_lookup': 'catalog_lookup',
        'catalog': 'catalog_lookup',
        'universities_by_region': 'institutions_by_region',
        'institutions_region': 'institutions_by_region',
        'near_me': 'programs_near_me',
        'nearby_programs': 'programs_near_me',
        'programs_near_home': 'programs_near_me',
        'ask_for_grades': 'ask_grades',
        'eligibility': 'qualify',
        'eligible': 'qualify',
        'qualification': 'qualify',
        'explanation': 'explain',
        'why': 'explain',
        'careerpaths': 'career_paths',
        'career_pathing': 'career_paths',
    }

    for it in items:
        s = str(it or '').strip().lower()
        if not s:
            continue
        s = s.replace(' ', '_')
        s = mapping.get(s, s)
        if s not in out:
            out.append(s)
    return out


def compute_confidence(grades: Dict[str, str], traits: Dict[str, float]) -> float:
    # Heuristic: more signals -> higher confidence
    g = len(grades)
    t = len([k for k, v in traits.items() if v > 0])
    # Cap at 1.0
    return min(1.0, 0.2 * g + 0.1 * t)


def analyze(text: str, provider_override: str = '') -> Dict[str, Any]:
    """End-to-end NLP analysis with optional Gemini provider.
    Returns dict with keys: grades, traits, intents, confidence.
    """
    override = (provider_override or '').strip().lower()
    if override not in ('local', 'gemini'):
        override = ''

    # Default provider: gemini if API key present and provider not explicitly set; else local
    prov_env = getattr(settings, 'NLP_PROVIDER', '') or ''
    api_key_env = (getattr(settings, 'GEMINI_API_KEY', '') or '').strip()
    provider = (override or prov_env or ('gemini' if api_key_env else 'local')).strip().lower()
    if provider == 'gemini':
        api_key = (getattr(settings, 'GEMINI_API_KEY', '') or '').strip()
        model_name = (getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash') or 'gemini-1.5-flash').strip()
        if api_key:
            try:
                from .providers.gemini_provider import analyze_text  # type: ignore
                out = analyze_text(text, api_key=api_key, model_name=model_name)
                out['provider'] = 'gemini'
                try:
                    local_intents = detect_intents(text, out.get('grades') or {})
                    intents = _normalize_intents(out.get('intents') or [])
                    local_norm = _normalize_intents(local_intents)
                    merged: List[str] = []
                    for x in list(intents) + list(local_norm):
                        if x and x not in merged:
                            merged.append(x)
                    out['intents'] = merged
                except Exception:
                    pass
                # Always enrich with local catalog lookup extraction for robustness
                lookup = extract_catalog_lookup(text)
                if lookup:
                    out['lookup'] = lookup
                    intents = _normalize_intents(out.get('intents') or [])
                    if 'catalog_lookup' not in intents:
                        intents.append('catalog_lookup')
                    out['intents'] = intents

                # Enrich with region-based institutions lookup
                reg = extract_region_query(text)
                if reg:
                    out['institutions_region'] = reg
                    intents = _normalize_intents(out.get('intents') or [])
                    lowtxt = (text or '').lower()
                    if 'institutions_by_region' not in intents and ('universit' in lowtxt or 'college' in lowtxt or 'institute' in lowtxt):
                        intents.append('institutions_by_region')
                        out['intents'] = intents
                return out
            except Exception as e:
                fallback = {'provider': 'local', 'provider_error': str(e)}
                grades = extract_subject_grade_pairs(text)
                traits = extract_traits(text)
                intents = detect_intents(text, grades)
                lookup = extract_catalog_lookup(text)
                if lookup and 'catalog_lookup' not in intents:
                    intents.append('catalog_lookup')
                reg = extract_region_query(text)
                lowtxt = (text or '').lower()
                if reg and 'institutions_by_region' not in intents and ('universit' in lowtxt or 'college' in lowtxt or 'institute' in lowtxt):
                    intents.append('institutions_by_region')
                conf = compute_confidence(grades, traits)
                fallback.update({'grades': grades, 'traits': traits, 'intents': intents, 'confidence': conf, 'lookup': lookup, 'institutions_region': reg})
                return fallback
    # Local lightweight pipeline
    grades = extract_subject_grade_pairs(text)
    traits = extract_traits(text)
    intents = detect_intents(text, grades)
    lookup = extract_catalog_lookup(text)
    if lookup and 'catalog_lookup' not in intents:
        intents.append('catalog_lookup')
    reg = extract_region_query(text)
    lowtxt = (text or '').lower()
    if reg and 'institutions_by_region' not in intents and ('universit' in lowtxt or 'college' in lowtxt or 'institute' in lowtxt):
        intents.append('institutions_by_region')
    conf = compute_confidence(grades, traits)
    return {
        'grades': grades,
        'traits': traits,
        'intents': intents,
        'confidence': conf,
        'provider': 'local',
        'lookup': lookup,
        'institutions_region': reg,
    }
