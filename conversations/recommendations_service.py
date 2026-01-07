import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .recommend import TRAIT_FIELD_HINTS

try:
    from django.db.models import Q
except Exception:
    Q = None

try:
    from scripts.etl.kuccps.eligibility import evaluate_eligibility as _kuccps_evaluate_eligibility  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_CODE_ALIASES as _SUBJECT_CODE_ALIASES  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_CANON_TO_NUM as _SUBJECT_CANON_TO_NUM  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_TOKEN_ALIASES as _SUBJECT_TOKEN_ALIASES  # type: ignore
    from scripts.etl.kuccps.eligibility import SUBJECT_TOKEN_CANON_TO_ALIASES as _SUBJECT_TOKEN_CANON_TO_ALIASES  # type: ignore
except Exception:
    _kuccps_evaluate_eligibility = None
    _SUBJECT_CODE_ALIASES = {}
    _SUBJECT_CANON_TO_NUM = {}
    _SUBJECT_TOKEN_ALIASES = {}
    _SUBJECT_TOKEN_CANON_TO_ALIASES = {}

def _grade_points(g: str) -> int:
    s = (g or '').strip().upper().replace(' ', '')
    mapping = {
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
    return int(mapping.get(s, 0))

def _meets_min_grade(user_grade: str, min_grade: str) -> bool:
    if not (min_grade or '').strip():
        return True
    return _grade_points(user_grade) >= _grade_points(min_grade)

def _norm_subject_code(s: str) -> str:
    return (s or '').strip().upper().replace(' ', '')

def _expand_grades_with_subject_aliases(grades: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (grades or {}).items():
        kk = _norm_subject_code(str(k))
        vv = str(v or '').strip().upper().replace(' ', '')
        if not kk or not vv:
            continue
        out[kk] = vv
        canon = _norm_subject_code(_SUBJECT_TOKEN_ALIASES.get(kk) or kk)
        if canon:
            out[canon] = vv
            for a in (_SUBJECT_TOKEN_CANON_TO_ALIASES.get(canon) or []):
                aa = _norm_subject_code(a)
                if aa:
                    out[aa] = vv
        if kk in _SUBJECT_CODE_ALIASES:
            out[_norm_subject_code(_SUBJECT_CODE_ALIASES.get(kk) or '')] = vv
        if kk in _SUBJECT_CANON_TO_NUM:
            out[_norm_subject_code(_SUBJECT_CANON_TO_NUM.get(kk) or '')] = vv
        if canon in _SUBJECT_CANON_TO_NUM:
            out[_norm_subject_code(_SUBJECT_CANON_TO_NUM.get(canon) or '')] = vv
    return out

def _format_req_label(subj_token: str, numeric_code: str, min_g: str) -> str:
    subj = _norm_subject_code(subj_token)
    code = _norm_subject_code(numeric_code)
    canon = ''
    if subj and subj in _SUBJECT_CANON_TO_NUM:
        canon = subj
    elif subj and subj in _SUBJECT_TOKEN_ALIASES:
        canon = _norm_subject_code(_SUBJECT_TOKEN_ALIASES.get(subj) or '')
    elif code and code in _SUBJECT_CODE_ALIASES:
        canon = _norm_subject_code(_SUBJECT_CODE_ALIASES.get(code) or '')
    label = canon or subj or code
    if min_g:
        return f"{label} >= {min_g}"
    return label

def _missing_from_subject_requirements(req: Dict[str, Any], grades: Dict[str, str]) -> list[str]:
    out: list[str] = []
    gmap = _expand_grades_with_subject_aliases(grades or {})
    if not isinstance(req, dict) or not req:
        return out

    required = req.get('required', []) or []
    groups = req.get('groups', []) or []

    for it in required:
        subj_raw = (it.get('subject') or it.get('subject_code') or '').strip()
        code_raw = (it.get('code') or '').strip()
        subj = _norm_subject_code(subj_raw or code_raw)
        if not subj and not code_raw:
            continue
        min_g = (it.get('min_grade') or '').strip().upper().replace(' ', '')
        user_g = gmap.get(subj, '')
        if (not user_g) or (min_g and not _meets_min_grade(user_g, min_g)):
            out.append(_format_req_label(subj_raw or subj, code_raw or subj, min_g))

    for grp in groups:
        try:
            pick = int(grp.get('pick') or 0)
        except Exception:
            pick = 0
        if pick <= 0:
            continue
        opts = grp.get('options', []) or []
        satisfied = 0
        missing: list[str] = []
        for opt in opts:
            subj_raw = (opt.get('subject') or opt.get('subject_code') or '').strip()
            code_raw = (opt.get('code') or '').strip()
            subj = _norm_subject_code(subj_raw or code_raw)
            if not subj and not code_raw:
                continue
            min_g = (opt.get('min_grade') or '').strip().upper().replace(' ', '')
            user_g = gmap.get(subj, '')
            if user_g and (not min_g or _meets_min_grade(user_g, min_g)):
                satisfied += 1
            else:
                missing.append(_format_req_label(subj_raw or subj, code_raw or subj, min_g))
        if satisfied < pick:
            out.extend(missing[: max(1, pick)])
    return out

def _eligibility_from_requirements(program, grades: Dict[str, str]) -> Dict[str, Any]:
    try:
        groups = list(program.requirement_groups.all().order_by('order').prefetch_related('options', 'options__subject'))
    except Exception:
        groups = []
    if not groups:
        return {'eligible': None, 'missing': [], 'unmet_groups': 0}

    if not (grades or {}):
        return {'eligible': None, 'missing': [], 'unmet_groups': 0}

    gmap = _expand_grades_with_subject_aliases(grades or {})
    missing: list[str] = []
    unmet_groups = 0
    for grp in groups:
        pick = int(getattr(grp, 'pick', 1) or 1)
        opts = list(grp.options.all().order_by('order'))
        if not opts:
            continue
        satisfied = 0
        grp_missing: list[str] = []
        for opt in opts:
            subj_code = ''
            try:
                if getattr(opt, 'subject_id', None):
                    subj_code = (opt.subject.code or '').strip().upper()
            except Exception:
                subj_code = ''
            if not subj_code:
                subj_code = (getattr(opt, 'subject_code', '') or '').strip().upper()
            if not subj_code:
                continue
            user_grade = gmap.get(subj_code, '')
            min_grade = (getattr(opt, 'min_grade', '') or '').strip().upper().replace(' ', '')
            if user_grade and _meets_min_grade(user_grade, min_grade):
                satisfied += 1
            else:
                if min_grade:
                    grp_missing.append(f"{subj_code} >= {min_grade}")
                else:
                    grp_missing.append(subj_code)
        if satisfied < pick:
            unmet_groups += 1
            missing.extend(grp_missing[: max(1, pick)])

    return {
        'eligible': unmet_groups == 0,
        'missing': missing,
        'unmet_groups': unmet_groups,
    }

def eligibility_for_program(program, grades: Dict[str, str], cutoff: Optional[float] = None) -> Dict[str, Any]:
    req = getattr(program, 'subject_requirements', None) or {}

    if not (grades or {}):
        return {
            'eligible': None,
            'missing': [],
            'unmet_groups': 0,
            'cluster_points': None,
            'cutoff_ok': None,
        }

    if not isinstance(req, dict) or not req:
        return _eligibility_from_requirements(program, grades)
    if _kuccps_evaluate_eligibility is None:
        return _eligibility_from_requirements(program, grades)

    try:
        row = {
            'subject_requirements_json': json.dumps(req),
            'normalized_name': (getattr(program, 'normalized_name', '') or ''),
            'name': (getattr(program, 'name', '') or ''),
        }
        res = _kuccps_evaluate_eligibility(row, grades or {}) or {}
        cluster_points = res.get('cluster_points', None)
        missing = _missing_from_subject_requirements(req, grades)

        eligible = True
        if isinstance(res.get('eligible', None), bool):
            eligible = bool(res.get('eligible'))

        cutoff_ok = None
        if cutoff is not None and cluster_points is not None:
            try:
                cutoff_ok = float(cluster_points) >= float(cutoff)
            except Exception:
                cutoff_ok = None
            if cutoff_ok is False:
                eligible = False
                missing.append(f"cluster_points >= {float(cutoff):g}")

        return {
            'eligible': eligible,
            'missing': missing,
            'unmet_groups': None,
            'cluster_points': cluster_points,
            'cutoff_ok': cutoff_ok,
        }
    except Exception:
        return _eligibility_from_requirements(program, grades)


def score_program(name: str, field_name: str, traits: Dict[str, float], goal_text: str = '') -> float:
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

    gt = (goal_text or '').strip().lower()
    if gt:
        toks = [t for t in ''.join((ch if ch.isalnum() else ' ') for ch in gt).split() if len(t) >= 4]
        if toks:
            stop = {'become', 'becoming', 'want', 'wants', 'would', 'like', 'study', 'studying', 'career', 'goal', 'goals', 'work'}
            toks2 = [t for t in toks if t not in stop]
            hits2 = sum(1 for t in set(toks2[:12]) if t in text)
            if hits2 > 0:
                score += min(2.0, 0.6 * float(hits2))
    return float(score)


def score_program_breakdown(
    name: str,
    field_name: str,
    traits: Dict[str, float],
    *,
    goal_text: str = '',
    cluster_points: Optional[float] = None,
    cutoff: Optional[float] = None,
) -> Dict[str, Any]:
    def _norm_match_text(s: str) -> str:
        ss = (s or '').lower()
        ss = re.sub(r"[^a-z0-9]+", " ", ss)
        ss = re.sub(r"\s+", " ", ss).strip()
        return f" {ss} " if ss else " "

    def _term_in(text_norm: str, term: str) -> bool:
        tt = _norm_match_text(term)
        if not tt.strip():
            return False
        # word-boundary-ish match: match whole normalized token sequence
        return tt in text_norm

    name_norm = _norm_match_text(name)
    field_norm = _norm_match_text(field_name)
    text = f"{name_norm}{field_norm}".strip()

    riasec_score = 0.0
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
            riasec_score += w

    goal_score = 0.0
    gt = (goal_text or '').strip().lower()
    explicit_goal = bool(gt)
    if gt:
        base_terms = _goal_terms(gt)
        expanded_terms = _expanded_goal_terms(base_terms, goal_text=gt)
        terms = expanded_terms or base_terms
        uniq = [str(t).lower() for t in (terms or []) if str(t).strip()]
        uniq = list(dict.fromkeys(uniq))

        generic_goal_terms = {
            'science',
            'studies',
            'management',
            'manager',
            'business',
            'commerce',
            'technology',
            'information',
            'systems',
            'administration',
            'policy',
            'leadership',
            'planning',
            'resource',
            'resources',
            'officer',
        }
        filtered = [t for t in uniq if t not in generic_goal_terms]
        if filtered:
            uniq = filtered

        hits_name = sum(1 for t in uniq[:12] if t and _term_in(name_norm, t))
        hits_field = sum(1 for t in uniq[:12] if t and _term_in(field_norm, t))

        # Field matches can be noisy (ETL mapping can be wrong). Treat field-only matches
        # as a weaker signal so misclassified fields don't dominate.
        hits_field_weight = 2
        if hits_name == 0 and hits_field > 0:
            hits_field_weight = 1

        hits2 = hits_name + (hits_field_weight * hits_field)
        if hits2 > 0:
            goal_score = min(2.0, 0.6 * float(hits2))

    grade_margin_score = 0.0
    if cluster_points is not None and cutoff is not None:
        try:
            grade_margin_score = float(cluster_points) - float(cutoff)
            grade_margin_score = max(-12.0, min(12.0, grade_margin_score)) / 6.0
        except Exception:
            grade_margin_score = 0.0

    if explicit_goal:
        w_goal, w_riasec, w_grade = 0.85, 0.12, 0.03
    else:
        w_goal, w_riasec, w_grade = 0.20, 0.55, 0.25

    total = (w_goal * goal_score) + (w_riasec * riasec_score) + (w_grade * grade_margin_score)
    return {
        'total_score': float(total),
        'goal_score': float(goal_score),
        'riasec_score': float(riasec_score),
        'grade_margin_score': float(grade_margin_score),
        'weights': {'goal': w_goal, 'riasec': w_riasec, 'grade_margin': w_grade},
        'explicit_goal': bool(explicit_goal),
    }


def _goal_terms(goal_text: str) -> List[str]:
    gt = (goal_text or '').strip().lower()
    if not gt:
        return []
    goal_terms: List[str] = []
    try:
        toks = [t for t in ''.join((ch if ch.isalnum() else ' ') for ch in gt).split() if len(t) >= 4]
        if toks:
            stop = {
                'become',
                'becoming',
                'career',
                'dream',
                'dreaming',
                'enjoy',
                'enjoying',
                'goal',
                'goals',
                'hope',
                'hoping',
                'interested',
                'like',
                'love',
                'passion',
                'passionate',
                'prefer',
                'study',
                'studying',
                'want',
                'wants',
                'work',
                'would',
            }
            goal_terms = [t for t in toks if t not in stop]
    except Exception:
        goal_terms = []
    return goal_terms


def _expanded_goal_terms(goal_terms: List[str], *, goal_text: str = '') -> List[str]:
    expanded_terms = list(goal_terms or [])
    try:
        s = set(expanded_terms)
        raw = (goal_text or '').strip().lower()
        raw_norm = re.sub(r"[^a-z0-9]+", " ", raw)
        raw_norm = re.sub(r"\s+", " ", raw_norm).strip()

        def _has_phrase(phrase: str) -> bool:
            p = (phrase or '').strip().lower()
            if not p:
                return False
            p = re.sub(r"[^a-z0-9]+", " ", p)
            p = re.sub(r"\s+", " ", p).strip()
            if not p:
                return False
            return f" {p} " in f" {raw_norm} "

        rules = [
            (("doctor", "doctors", "physician", "surgeon", "medical"), ("medicine", "mbbs", "mbchb", "clinical", "surgery")),
            (("medicine", "medical", "surgery", "surgeon"), ("mbbs", "mbchb", "clinical")),
            (("dentist", "dental"), ("dental", "surgery", "bds")),
            (("pharmacy", "pharmacist"), ("pharmacy", "pharm")),
            (("pharmaceutical",), ("pharmaceutical", "pharmaceutical technology", "pharmacy", "pharm")),
            (("nurse", "nursing", "midwife", "midwifery"), ("nursing", "midwifery", "clinical")),
            (("lawyer", "law", "legal", "advocate"), ("law", "laws", "llb", "ll b", "legal")),
            (("magistrate", "judge", "judiciary", "judicial"), ("law", "legal", "judiciary", "llb")),
            (("paralegal",), ("paralegal", "legal", "law")),
            (("criminologist", "criminology"), ("criminology", "forensic", "criminal justice", "security")),

            (("journalist", "journalism"), ("journalism", "media", "broadcast", "news", "mass communication")),
            (("architect", "architecture", "architectural"), ("architecture", "architectural", "design", "urban planning", "planning")),
            (("physiotherapist", "physiotherapy", "therapist"), ("physiotherapy", "physical therapy", "rehabilitation", "therapy")),
            (("nutritionist", "nutrition", "dietitian", "dietetics"), ("nutrition", "dietetics", "dietitian")),

            (("software",), ("software", "computer", "computing", "information technology", "ict")),
            (("engineer", "engineering"), ("engineering", "technology")),
            (("network",), ("network", "networking", "telecommunication", "telecommunications", "computer", "computing", "ict")),
            (("cybersecurity",), ("cybersecurity", "information security", "security", "network", "computer", "computing", "forensic")),
            (("pilot", "aviation", "aeronaut", "aeronautics"), ("aviation", "aeronautics", "flight", "aircraft")),

            (("database",), ("database", "information systems", "computer science", "ict")),

            (("computer", "computers", "computing"), ("computer science", "software", "information technology", "ict")),
            (("coding", "programming", "developer"), ("computer science", "software", "programming", "information technology")),

            (("accountant", "accounting"), ("accounting", "finance", "auditing", "tax")),
            (("auditor", "auditing"), ("auditing", "accounting", "finance")),
            (("banker", "banking"), ("banking", "finance", "investment", "economics")),
            (("economist", "economics"), ("economics", "econometrics")),
            (("statistician", "statistics"), ("statistics", "applied statistics", "actuarial", "biostatistics", "data science", "mathematics")),
            (("actuary", "actuarial"), ("actuarial", "actuarial science", "statistics", "mathematics", "risk", "insurance", "finance")),
            (("investment",), ("investment", "finance", "economics", "banking")),
            (("credit",), ("credit", "banking", "finance", "microfinance")),
            (("insurance",), ("insurance", "risk", "actuarial", "finance")),
            (("data",), ("data science", "statistics", "applied statistics", "computer science", "machine learning")),
            (("marketing",), ("marketing", "sales", "commerce", "business")),
            (("sales",), ("sales", "marketing", "commerce", "business")),
            (("entrepreneur", "entrepreneurship"), ("entrepreneurship", "business", "startup", "commerce", "management")),

            (("procurement", "purchasing"), ("procurement", "supply chain", "purchasing", "logistics", "stores")),
            (("logistics",), ("logistics", "supply chain", "transport", "procurement")),
            (("surveyor", "surveying"), ("surveying", "quantity surveying", "construction", "built environment")),
            (("quantity",), ("quantity surveying", "building economics", "construction", "real estate")),

            (("teacher", "teaching"), ("education", "teaching", "pedagogy")),
            (("lecturer",), ("education", "teaching", "pedagogy")),
            (("social",), ("social work", "sociology", "community")),
            (("counsellor", "counselor", "counseling", "counselling"), ("counselling", "counseling", "psychology", "guidance")),
            (("police",), ("criminology", "forensic", "criminal justice", "security")),
            (("military", "army", "defence", "defense"), ("defence", "security", "strategic", "military")),
            (("researcher", "research"), ("research", "laboratory", "science")),

            (("radiographer", "radiography"), ("radiography", "imaging", "medical imaging", "radiology")),
            (("photographer", "photography"), ("photography", "film", "media", "multimedia")),
            (("animator", "animation"), ("animation", "film", "multimedia", "graphic design")),
            (("chef", "culinary"), ("culinary", "hospitality", "food", "food production")),
            (("content",), ("media", "multimedia", "film", "digital media")),
            (("forester", "forestry"), ("forestry", "forest", "environment", "conservation")),
            (("geologist", "geology"), ("geology", "earth science", "mining", "petroleum")),
            (("agronomist", "agronomy"), ("agronomy", "crop", "agriculture", "food science")),
            (("biochemist", "biochemistry"), ("biochemistry", "biology", "chemistry", "biotechnology", "biomedical")),
            (("microbiologist", "microbiology"), ("microbiology", "biology", "biochemistry", "biomedical")),
            (("laboratory",), ("laboratory", "medical laboratory", "laboratory science", "laboratory technology")),
            (("environmental",), ("environment", "climate", "sustainability", "conservation")),
            (("agricultural",), ("agriculture", "agribusiness", "agro", "food science")),
            (("veterinarian", "veterinary"), ("veterinary", "animal", "animal health")),
            (("tourism",), ("tourism", "hospitality", "hotel", "travel")),
            (("hotel",), ("hotel", "hospitality", "tourism")),
        ]

        for triggers, additions in rules:
            if any(t in s for t in triggers):
                expanded_terms.extend(list(additions))

        if _has_phrase('public relations') or (('public' in s) and ('relations' in s)) or ('pr' in s):
            expanded_terms.extend(['public relations', 'media', 'journalism', 'mass communication', 'corporate communication', 'pr'])

        if _has_phrase('air traffic') or _has_phrase('air traffic controller'):
            expanded_terms.extend(['air traffic', 'aviation', 'flight', 'aeronautics'])

        if _has_phrase('clinical officer') or _has_phrase('clinical medicine'):
            expanded_terms.extend(['clinical medicine', 'clinical', 'medicine', 'medical', 'surgery', 'health'])

        if _has_phrase('pharmaceutical technologist'):
            expanded_terms.extend(['pharmacy', 'pharmaceutical', 'pharmaceutical technology', 'pharm'])

        if _has_phrase('insurance officer'):
            expanded_terms.extend(['insurance', 'risk', 'actuarial', 'finance'])

        if _has_phrase('credit officer'):
            expanded_terms.extend(['credit', 'banking', 'finance', 'microfinance'])

        if _has_phrase('work in hospitals') or _has_phrase('work in hospital'):
            expanded_terms.extend(['health', 'medicine', 'medical', 'nursing', 'clinical'])

        if _has_phrase('enjoy helping people'):
            expanded_terms.extend(['social work', 'counselling', 'nursing', 'community health', 'public health'])

        if _has_phrase('ui ux') or _has_phrase('user experience') or _has_phrase('user interface'):
            expanded_terms.extend(['user experience', 'user interface', 'interaction design', 'graphic design', 'design'])

        if _has_phrase('systems analyst'):
            expanded_terms.extend(['information systems', 'computer science', 'software', 'ict'])

        if _has_phrase('database administrator'):
            expanded_terms.extend(['database', 'information systems', 'computer science', 'ict'])

        if _has_phrase('quantity surveyor') or _has_phrase('quantity surveying'):
            expanded_terms.extend(['quantity surveying', 'surveying', 'construction', 'built environment', 'building economics'])

        if _has_phrase('procurement officer') or _has_phrase('supply chain'):
            expanded_terms.extend(['procurement', 'supply chain', 'logistics', 'purchasing'])

        if _has_phrase('i like computers') or _has_phrase('i love computers'):
            expanded_terms.extend(['computer', 'computing', 'computer science', 'information technology', 'ict'])

        if _has_phrase('i love drawing') or _has_phrase('i like drawing'):
            expanded_terms.extend(['art', 'design', 'graphic design', 'fine art', 'animation'])

        if _has_phrase('it support') or _has_phrase('it support specialist'):
            expanded_terms.extend(['information technology', 'ict', 'computer science', 'software', 'computing'])

        if _has_phrase('systems analyst'):
            expanded_terms.extend(['information systems', 'computer science', 'software', 'ict'])

        if _has_phrase('database administrator'):
            expanded_terms.extend(['database', 'information systems', 'computer science', 'ict'])

        if _has_phrase('investment analyst'):
            expanded_terms.extend(['investment', 'finance', 'economics', 'banking'])

        if _has_phrase('financial analyst'):
            expanded_terms.extend(['finance', 'economics', 'accounting', 'investment'])

        if _has_phrase('data analyst'):
            expanded_terms.extend(['data', 'statistics', 'data science', 'analytics'])

        if _has_phrase('data scientist'):
            expanded_terms.extend(['data science', 'statistics', 'applied statistics', 'computer science', 'machine learning'])

        if _has_phrase('graphic designer'):
            expanded_terms.extend(['graphic', 'design', 'animation', 'film', 'multimedia', 'visual communication'])
        expanded_terms = [t for t in expanded_terms if t and len(t) >= 3]
    except Exception:
        expanded_terms = [t for t in expanded_terms if t and len(t) >= 3]
    return expanded_terms


def _catalog_models():
    try:
        from catalog.models import Program, ProgramCost, YearlyCutoff  # type: ignore
        return Program, ProgramCost, YearlyCutoff
    except Exception:
        return None, None, None


def _latest_cutoff_for_program(YearlyCutoff, program) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    cutoff = None
    cutoff_val = None
    try:
        yc = YearlyCutoff.objects.filter(program=program).order_by('-year').first()
        if yc:
            cutoff_val = float(yc.cutoff)
            cutoff = {
                'year': yc.year,
                'cutoff': float(yc.cutoff),
                'capacity': yc.capacity,
            }
    except Exception:
        cutoff = None
        cutoff_val = None
    return cutoff, cutoff_val


def _latest_cost_for_program(ProgramCost, program) -> Optional[Dict[str, Any]]:
    cost = None
    try:
        pc = ProgramCost.objects.filter(program=program).order_by('-updated_at').first()
        if not pc and (getattr(program, 'code', '') or '').strip():
            pc = ProgramCost.objects.filter(program_code=(getattr(program, 'code', '') or '').strip()).order_by('-updated_at').first()
        if pc:
            cost = {
                'amount': float(pc.amount) if pc.amount is not None else None,
                'currency': pc.currency or 'KES',
                'source_id': pc.source_id or '',
                'raw_cost': pc.raw_cost or '',
            }
    except Exception:
        cost = None
    return cost


def _requirements_preview_for_program(program) -> str:
    try:
        return program.requirements_preview()
    except Exception:
        return ''


def build_recommendations(
    grades: Dict[str, str],
    traits: Dict[str, float],
    *,
    goal_text: str = '',
    k: int = 10,
    level: str = 'bachelor',
    max_scan: int = 2000,
    raise_on_missing_catalog: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    Program, _ProgramCost, _YearlyCutoff = _catalog_models()
    if Program is None:
        if raise_on_missing_catalog:
            raise Exception('Catalog DB not available')
        return [], []

    ProgramCost = _ProgramCost
    YearlyCutoff = _YearlyCutoff

    try:
        k = max(1, min(20, int(k or 10)))
    except Exception:
        k = 10

    try:
        qs = Program.objects.select_related('institution', 'field').filter(level=str(level or 'bachelor'))
    except Exception:
        if raise_on_missing_catalog:
            raise Exception('Catalog DB not available')
        return [], []

    traits_sorted: List[Tuple[str, float]] = []
    for t, v in (traits or {}).items():
        try:
            traits_sorted.append((str(t), float(v)))
        except Exception:
            traits_sorted.append((str(t), 0.0))
    traits_sorted.sort(key=lambda kv: -kv[1])

    goal_terms = _goal_terms(goal_text)
    expanded_terms = _expanded_goal_terms(goal_terms, goal_text=goal_text)

    if (traits_sorted or goal_text) and Q is not None:
        hints: List[str] = []
        if expanded_terms:
            hints.extend(expanded_terms[:10])
        elif goal_text:
            hints.extend(goal_terms[:10])

        for t, _w in traits_sorted[:3]:
            hints.extend((TRAIT_FIELD_HINTS.get(t, []) or [])[:4])

        hints = [h for h in hints if h and len(h) >= 3]
        if hints:
            seen = set()
            ordered: List[str] = []
            for h in hints:
                hh = str(h).strip().lower()
                if not hh or hh in seen:
                    continue
                seen.add(hh)
                ordered.append(hh)

            q = Q()
            for h in ordered[:10]:
                q |= Q(normalized_name__icontains=h) | Q(name__icontains=h) | Q(field__name__icontains=h)
            try:
                narrowed = qs.filter(q)
                if narrowed.exists():
                    qs = narrowed
            except Exception:
                pass

    scored: List[Tuple[float, Any]] = []
    scan_n = max(1, min(int(max_scan or 2000), 5000))
    for p in qs[:scan_n]:
        nm = (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip()
        if not nm:
            continue
        field_name = ''
        try:
            if getattr(p, 'field_id', None):
                field_name = (p.field.name or '').strip()
        except Exception:
            field_name = ''
        br = score_program_breakdown(nm, field_name, traits or {}, goal_text=goal_text)
        sc = float(br.get('total_score') or 0.0)
        if not (traits or {}) and (grades or {}):
            sc += 0.2
        scored.append((float(sc), p))

    scored.sort(key=lambda t: (-t[0], (getattr(t[1], 'normalized_name', '') or getattr(t[1], 'name', '') or '')))

    eligible_recs: List[Tuple[Optional[float], float, Dict[str, Any]]] = []
    unknown_recs: List[Tuple[float, Optional[float], Dict[str, Any]]] = []
    stretch_recs: List[Tuple[Optional[float], float, Dict[str, Any]]] = []

    scan_limit = min(len(scored), max(k * 8, 80))
    stretch_target = max(3, min(10, int(k or 10)))
    max_iter = min(len(scored), max(scan_limit, stretch_target * 20))

    for sc, p in scored[:max_iter]:
        nm = (getattr(p, 'normalized_name', '') or getattr(p, 'name', '') or '').strip()
        field_name = (p.field.name if getattr(p, 'field_id', None) else '') or ''

        cutoff = None
        cutoff_val = None
        if YearlyCutoff is not None:
            cutoff, cutoff_val = _latest_cutoff_for_program(YearlyCutoff, p)

        elig = eligibility_for_program(p, grades or {}, cutoff=cutoff_val)
        cp = None
        try:
            if isinstance(elig, dict):
                cp = elig.get('cluster_points', None)
        except Exception:
            cp = None

        br2 = score_program_breakdown(nm, field_name, traits or {}, goal_text=goal_text, cluster_points=cp, cutoff=cutoff_val)

        if elig and elig.get('eligible') is False:
            if expanded_terms and len(stretch_recs) < stretch_target and _is_goal_match(nm, field_name, (getattr(p, 'code', '') or '').strip(), expanded_terms):
                req_preview = _requirements_preview_for_program(p)
                reason: Dict[str, Any] = {}
                try:
                    cp = elig.get('cluster_points', None) if isinstance(elig, dict) else None
                    if cp is not None and cutoff_val is not None:
                        try:
                            reason['cutoff_gap'] = round(float(cutoff_val) - float(cp), 3)
                        except Exception:
                            pass
                except Exception:
                    pass
                item = {
                    'program_id': getattr(p, 'id', None),
                    'program_code': (getattr(p, 'code', '') or '').strip(),
                    'program_name': nm,
                    'institution_name': (p.institution.name or '').strip() if getattr(p, 'institution_id', None) else '',
                    'institution_code': (p.institution.code or '').strip() if getattr(p, 'institution_id', None) else '',
                    'field_name': field_name,
                    'level': (getattr(p, 'level', '') or '').strip(),
                    'region': (getattr(p, 'region', '') or '').strip(),
                    'campus': (getattr(p, 'campus', '') or '').strip(),
                    'score': round(float(sc), 3),
                    'eligibility': elig,
                    'requirements_preview': req_preview,
                    'cost': None,
                    'latest_cutoff': cutoff,
                    'stretch_reason': reason,
                }
                stretch_recs.append((cutoff_val, float(sc), item))
            continue

        cost = None
        if ProgramCost is not None:
            cost = _latest_cost_for_program(ProgramCost, p)
        req_preview = _requirements_preview_for_program(p)

        item = {
            'program_id': getattr(p, 'id', None),
            'program_code': (getattr(p, 'code', '') or '').strip(),
            'program_name': nm,
            'institution_name': (p.institution.name or '').strip() if getattr(p, 'institution_id', None) else '',
            'institution_code': (p.institution.code or '').strip() if getattr(p, 'institution_id', None) else '',
            'field_name': field_name,
            'level': (getattr(p, 'level', '') or '').strip(),
            'region': (getattr(p, 'region', '') or '').strip(),
            'campus': (getattr(p, 'campus', '') or '').strip(),
            'score': round(float(sc), 3),
            'score_breakdown': br2,
            'eligibility': elig,
            'requirements_preview': req_preview,
            'cost': cost,
            'latest_cutoff': cutoff,
        }

        if elig and elig.get('eligible') is True:
            eligible_recs.append((cutoff_val, float(sc), item))
        else:
            unknown_recs.append((float(sc), cutoff_val, item))

        max_collect = k * 3
        if not goal_text:
            max_collect = k * 12
        if len(eligible_recs) + len(unknown_recs) >= max_collect:
            break

    eligible_recs.sort(key=lambda t: (-(t[0] if t[0] is not None else -1.0), -t[1]))
    unknown_recs.sort(key=lambda t: (-t[0], -(t[1] if t[1] is not None else -1.0)))
    stretch_recs.sort(key=lambda t: (-(t[0] if t[0] is not None else -1.0), -t[1]))

    recs_all = [x[2] for x in eligible_recs] + [x[2] for x in unknown_recs]
    recs = recs_all[:k]
    stretch_items = [x[2] for x in stretch_recs][:stretch_target]

    if not goal_text:
        # Diversify: for users without an explicit goal, return a varied set of programs
        # across fields instead of over-concentrating in one field.
        by_field: Dict[str, List[Dict[str, Any]]] = {}
        field_order: List[str] = []
        seen: set[str] = set()
        for r in recs_all:
            try:
                nm = (r.get('program_name') or '') if isinstance(r, dict) else ''
                key = str(nm).strip().lower()
            except Exception:
                key = ''
            if key and key in seen:
                continue
            if key:
                seen.add(key)

            try:
                fn = (r.get('field_name') or '').strip() if isinstance(r, dict) else ''
            except Exception:
                fn = ''
            fn = fn or 'Other'
            if fn not in by_field:
                by_field[fn] = []
                field_order.append(fn)
            by_field[fn].append(r)

        diversified: List[Dict[str, Any]] = []
        idx = 0
        while len(diversified) < k:
            progressed = False
            for fn in field_order:
                items = by_field.get(fn) or []
                if idx < len(items):
                    diversified.append(items[idx])
                    progressed = True
                    if len(diversified) >= k:
                        break
            if not progressed:
                break
            idx += 1

        if diversified:
            recs = diversified[:k]

    if goal_text:
        aligned: List[Dict[str, Any]] = []
        for r in recs_all:
            try:
                br = r.get('score_breakdown') if isinstance(r, dict) else None
                gs = float(br.get('goal_score') or 0.0) if isinstance(br, dict) else 0.0
            except Exception:
                gs = 0.0
            if gs > 0:
                aligned.append(r)

        min_aligned = 3
        try:
            min_aligned = max(1, min(5, int(k or 10)))
        except Exception:
            min_aligned = 3
        min_aligned = min(min_aligned, 3)

        if aligned:
            recs = aligned[:k]
    return recs, stretch_items


def _is_goal_match(nm: str, field_name: str, program_code: str, expanded_terms: List[str]) -> bool:
    try:
        txt = f"{nm} {field_name} {program_code}".lower()
    except Exception:
        txt = ''
    if not expanded_terms:
        return False
    return any(t in txt for t in set(expanded_terms) if t)
