import re
from typing import Dict, List, Tuple, Any

# Lightweight NLP utilities (no heavy external deps). Optional LLM provider can be added.

# Canonical subject codes and common synonyms
_SUBJECT_SYNONYMS = {
    'math': 'MAT', 'mathematics': 'MAT', 'mat': 'MAT',
    'english': 'ENG', 'eng': 'ENG', 'lang': 'ENG',
    'kiswahili': 'KIS', 'swahili': 'KIS', 'kisw': 'KIS', 'kis': 'KIS',
    'chemistry': 'CHE', 'chem': 'CHE',
    'physics': 'PHY', 'phy': 'PHY',
    'biology': 'BIO', 'bio': 'BIO',
    'history': 'HIS', 'hist': 'HIS',
    'geography': 'GEO', 'geo': 'GEO',
    'business': 'BUS', 'commerce': 'BUS', 'bus': 'BUS',
    'computer': 'COM', 'computing': 'COM', 'ict': 'COM', 'cs': 'COM',
    'cre': 'CRE', 'ire': 'IRE', 'agric': 'AGR', 'agriculture': 'AGR',
}

# Grades pattern (KCSE-like): A, A-, B+, B, B-, C+, C, C-, D+, D, E
# Match grades like A-, B+, including when followed by punctuation. Avoid trailing word chars.
_GRADE_RE = re.compile(r"(?<![A-Za-z])(?:A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E)(?![A-Za-z])", re.IGNORECASE)

# Split tokens and punctuation
_TOKEN_RE = re.compile(r"[A-Za-z+\-]+|\d+|[.,;:/]")


def _norm_subject_token(tok: str) -> str:
    t = tok.strip().lower()
    return _SUBJECT_SYNONYMS.get(t, t.upper())


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
    'Artistic': ['design', 'art', 'music', 'creative', 'write', 'draw'],
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


def detect_intents(text: str, grades: Dict[str, str]) -> List[str]:
    s = (text or '').lower()
    intents: List[str] = []
    if any(w in s for w in ["hi", "hello", "hey"]):
        intents.append('greeting')
    if grades:
        intents.append('provide_grades')
    if any(w in s for w in ["interests", "like", "enjoy", "love", "prefer"]):
        intents.append('interests')
    return intents


def compute_confidence(grades: Dict[str, str], traits: Dict[str, float]) -> float:
    # Heuristic: more signals -> higher confidence
    g = len(grades)
    t = len([k for k, v in traits.items() if v > 0])
    # Cap at 1.0
    return min(1.0, 0.2 * g + 0.1 * t)


def analyze(text: str) -> Dict[str, Any]:
    """End-to-end lightweight NLP analysis returning a dict with
    - grades: subject->grade
    - traits: RIASEC-like scores
    - intents: list of intents
    - confidence: float [0..1]
    """
    grades = extract_subject_grade_pairs(text)
    traits = extract_traits(text)
    intents = detect_intents(text, grades)
    conf = compute_confidence(grades, traits)
    return {
        'grades': grades,
        'traits': traits,
        'intents': intents,
        'confidence': conf,
    }
