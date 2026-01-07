from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _tokenize(text: str) -> List[str]:
    s = (text or '').strip().lower()
    if not s:
        return []
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    toks = [t for t in s.split() if len(t) >= 3]
    stop = {
        'and',
        'the',
        'for',
        'with',
        'to',
        'in',
        'of',
        'a',
        'an',
        'be',
        'become',
        'becoming',
        'want',
        'wants',
        'would',
        'like',
        'study',
        'studying',
        'career',
        'goal',
        'goals',
        'work',
    }
    return [t for t in toks if t not in stop]


def _load_goals_from_file(path: str) -> List[str]:
    p = (path or '').strip()
    if not p:
        return []
    with open(p, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.read().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith('#')]


def _default_goals_50() -> List[str]:
    # Common career goals (curated). Override via --goals-file for your own list.
    return [
        'Doctor',
        'Nurse',
        'Dentist',
        'Pharmacist',
        'Veterinarian',
        'Physiotherapist',
        'Nutritionist',
        'Clinical Psychologist',
        'Teacher',
        'Lecturer',
        'Lawyer',
        'Magistrate',
        'Accountant',
        'Auditor',
        'Banker',
        'Entrepreneur',
        'Business Manager',
        'Project Manager',
        'Human Resource Manager',
        'Marketing Manager',
        'Sales Manager',
        'Journalist',
        'Public Relations Officer',
        'Graphic Designer',
        'Architect',
        'Civil Engineer',
        'Mechanical Engineer',
        'Electrical Engineer',
        'Software Engineer',
        'Data Scientist',
        'Cybersecurity Analyst',
        'Network Engineer',
        'Doctor (MBChB)',
        'Pilot',
        'Air Traffic Controller',
        'PhD Researcher',
        'Biochemist',
        'Microbiologist',
        'Laboratory Technologist',
        'Environmental Scientist',
        'Agricultural Officer',
        'Veterinary Officer',
        'Economist',
        'Statistician',
        'Social Worker',
        'Counsellor',
        'Police Officer',
        'Military Officer',
        'Tourism Manager',
        'Hotel Manager',
    ]


def _kenya_goals() -> List[str]:
    return [
        'Clinical Officer',
        'Community Health Assistant',
        'Public Health Officer',
        'Pharmaceutical Technologist',
        'Medical Laboratory Scientist',
        'Laboratory Technologist',
        'Radiographer',
        'Nutrition and Dietetics',
        'Occupational Therapist',
        'Biomedical Engineer',
        'Dentist',
        'Dental Technologist',
        'Civil Engineer',
        'Electrical Engineer',
        'Mechanical Engineer',
        'Quantity Surveyor',
        'Land Surveyor',
        'Architect',
        'Urban Planner',
        'Real Estate Valuer',
        'Procurement Officer',
        'Supply Chain Manager',
        'Logistics Officer',
        'Human Resource Officer',
        'Credit Officer',
        'Banker',
        'Actuary',
        'Insurance Officer',
        'Co-operative Officer',
        'Agricultural Officer',
        'Extension Officer',
        'Veterinary Officer',
        'Fisheries Officer',
        'Forester',
        'Teacher',
        'Secondary School Teacher',
        'Special Needs Teacher',
        'Police Officer',
        'Criminologist',
        'Forensic Scientist',
        'Lawyer',
        'Paralegal',
        'Magistrate',
    ]


def _phrasing_variant_goals() -> List[str]:
    return [
        'I like computers',
        'I love computers',
        'I like coding',
        'I want to learn programming',
        'I like drawing',
        'I love drawing',
        'I enjoy helping people',
        'I want to work in banking',
        'I want to work in procurement',
        'I want to work in aviation',
        'I want to work in hospitals',
        'I want to be in construction',
    ]


def _extended_goals() -> List[str]:
    return [
        'Web Developer',
        'Mobile App Developer',
        'Database Administrator',
        'Systems Analyst',
        'IT Support Specialist',
        'Information Security Officer',
        'Data Analyst',
        'Machine Learning Engineer',
        'UI UX Designer',
        'Animator',
        'Film Producer',
        'Photographer',
        'Journalist',
        'Content Creator',
        'Teacher',
        'Early Childhood Teacher',
        'Primary School Teacher',
        'Secondary School Teacher',
        'Pharmacist',
        'Nurse',
        'Midwife',
        'Clinical Psychologist',
        'Counsellor',
        'Social Worker',
        'Nutritionist',
        'Physiotherapist',
        'Radiographer',
        'Medical Laboratory Scientist',
        'Biomedical Scientist',
        'Medical Engineer',
        'Environmental Scientist',
        'Geologist',
        'Economist',
        'Statistician',
        'Accountant',
        'Auditor',
        'Financial Analyst',
        'Investment Analyst',
        'Marketing Manager',
        'Sales Representative',
        'Entrepreneur',
        'Project Manager',
        'Procurement Officer',
        'Supply Chain Manager',
        'Logistics Manager',
        'Human Resource Manager',
        'Public Relations Officer',
        'Lawyer',
        'Paralegal',
        'Actuary',
        'Quantity Surveyor',
        'Civil Engineer',
        'Mechanical Engineer',
        'Electrical Engineer',
        'Architect',
        'Quantity Surveying',
        'Construction Manager',
        'Pilot',
        'Air Traffic Controller',
        'Tourism Manager',
        'Hotel Manager',
        'Chef',
        'Agronomist',
        'Food Scientist',
        'Veterinarian',
        'Marine Biologist',
        'Biochemist',
        'Microbiologist',
    ]


def _goals_for_suites(*, suites: List[str], goals_file: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    suite_map = {
        'default50': _default_goals_50,
        'kenya': _kenya_goals,
        'phrasing': _phrasing_variant_goals,
        'extended': _extended_goals,
    }

    for s in suites:
        ss = str(s or '').strip().lower()
        if not ss:
            continue
        fn = suite_map.get(ss)
        if fn is None:
            raise ValueError(f"Unknown suite: {ss}")
        for g in fn() or []:
            gg = str(g or '').strip()
            if gg:
                out.append((ss, gg))

    if goals_file:
        for g in _load_goals_from_file(goals_file) or []:
            gg = str(g or '').strip()
            if gg:
                out.append(('file', gg))

    return out


def _build_eval_user_context(*, uid: str) -> Any:
    # We keep this deterministic and simple: a generic RIASEC-ish profile with no grades.
    from conversations.tools import UserContext

    return UserContext(
        uid=str(uid),
        grades={},
        traits={
            'Investigative': 0.9,
            'Social': 0.6,
            'Realistic': 0.5,
            'Enterprising': 0.4,
            'Artistic': 0.3,
            'Conventional': 0.3,
        },
        career_goals=[],
        preferences={},
    )


def _recommend_for_goal(*, ctx: Any, goal: str, k: int, level: str) -> Dict[str, Any]:
    from conversations.tools import recommend_programs

    return recommend_programs(ctx=ctx, goal_text=str(goal or '').strip(), k=int(k), level=str(level or '').strip())


def _appropriateness_signals(*, goal: str, rec_item: Dict[str, Any]) -> Dict[str, Any]:
    goal_toks = set(_tokenize(goal))

    nm = str(rec_item.get('program_name') or '')
    field = str(rec_item.get('field_name') or '')
    hay = f"{nm} {field}"
    hay_toks = set(_tokenize(hay))

    overlap = sorted(list(goal_toks.intersection(hay_toks)))

    br = rec_item.get('score_breakdown')
    goal_score = None
    explicit_goal = None
    if isinstance(br, dict):
        goal_score = br.get('goal_score')
        explicit_goal = br.get('explicit_goal')

    elig = rec_item.get('eligibility')
    elig_status = None
    if isinstance(elig, dict):
        elig_flag = elig.get('eligible')
        if elig_flag is True:
            elig_status = 'eligible'
        elif elig_flag is False:
            elig_status = 'not_eligible'
        else:
            elig_status = 'unknown'

    label = 'weak'
    try:
        if overlap:
            label = 'strong'
        elif goal_score is not None and float(goal_score or 0.0) > 0:
            label = 'strong'
    except Exception:
        pass

    return {
        'overlap_terms': overlap,
        'overlap_count': len(overlap),
        'goal_score': goal_score,
        'explicit_goal': explicit_goal,
        'eligibility_status': elig_status,
        'appropriateness_label': label,
    }


def _flatten_rows(
    *,
    goal: str,
    recs: List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, it in enumerate((recs or [])[: max(1, int(top_n or 10))], start=1):
        if not isinstance(it, dict):
            continue
        sig = _appropriateness_signals(goal=goal, rec_item=it)
        rows.append(
            {
                'goal': goal,
                'rank': rank,
                'program_id': it.get('program_id'),
                'program_name': it.get('program_name'),
                'institution_name': it.get('institution_name'),
                'field_name': it.get('field_name'),
                'level': it.get('level'),
                'region': it.get('region'),
                'score': it.get('score'),
                'eligibility_status': sig.get('eligibility_status'),
                'appropriateness_label': sig.get('appropriateness_label'),
                'overlap_count': sig.get('overlap_count'),
                'overlap_terms': ' '.join(sig.get('overlap_terms') or []),
                'goal_score': sig.get('goal_score'),
                'explicit_goal': sig.get('explicit_goal'),
            }
        )
    return rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    p = (path or '').strip()
    if not p:
        return
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_json(path: str, payload: Any) -> None:
    p = (path or '').strip()
    if not p:
        return
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate chatbot program recommendations across many career goals.')
    parser.add_argument('--goals-file', default='', help='Optional path to a newline-delimited list of goals to evaluate.')
    parser.add_argument(
        '--suites',
        default='default50,kenya,phrasing,extended',
        help='Comma-separated suites to run: default50,kenya,phrasing,extended. goals-file is appended as suite=file.',
    )
    parser.add_argument('--dedupe-goals', default=1, type=int, help='If 1 (default), de-duplicate goals across suites.')
    parser.add_argument('--max-goals', default=0, type=int, help='If >0, cap total number of goals evaluated.')
    parser.add_argument('--k', default=10, type=int, help='How many recommendations to request (1..20).')
    parser.add_argument('--level', default='bachelor', help='Program level (e.g., bachelor, diploma).')
    parser.add_argument('--top-n', default=10, type=int, help='How many top recommendations to include per goal in the report.')
    parser.add_argument('--uid', default='eval_user', help='Synthetic UID used for the evaluation context.')
    parser.add_argument('--out-csv', default='scripts/goal_eval_report.csv', help='Output CSV file path.')
    parser.add_argument('--out-json', default='scripts/goal_eval_report.json', help='Output JSON file path.')

    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
    import django

    django.setup()

    suites = [s.strip() for s in str(args.suites or '').split(',') if s.strip()]
    pairs = _goals_for_suites(suites=suites, goals_file=str(args.goals_file or '').strip())

    dedupe = bool(int(args.dedupe_goals or 0))
    if dedupe:
        seen_goals = set()
        deduped: List[Tuple[str, str]] = []
        for suite, goal in pairs:
            key = str(goal).strip().lower()
            if not key or key in seen_goals:
                continue
            seen_goals.add(key)
            deduped.append((suite, goal))
        pairs = deduped

    max_goals = int(args.max_goals or 0)
    if max_goals > 0:
        pairs = pairs[:max_goals]

    k = max(1, min(20, int(args.k or 10)))
    top_n = max(1, min(20, int(args.top_n or 10)))
    level = str(args.level or 'bachelor').strip() or 'bachelor'

    ctx = _build_eval_user_context(uid=str(args.uid))

    report_items: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for suite, g in pairs:
        goal = str(g or '').strip()
        if not goal:
            continue

        res = _recommend_for_goal(ctx=ctx, goal=goal, k=k, level=level)
        recs = res.get('recommendations') or []
        if not isinstance(recs, list):
            recs = []

        item = {
            'suite': suite,
            'goal': goal,
            'requested_k': k,
            'level': level,
            'recommendations': recs[:top_n],
        }
        report_items.append(item)
        suite_rows = _flatten_rows(goal=goal, recs=recs, top_n=top_n)
        for r in suite_rows:
            r['suite'] = suite
        csv_rows.extend(suite_rows)

    meta = {
        'generated_at_utc': datetime.utcnow().isoformat() + 'Z',
        'goals_count': len(pairs),
        'suites': suites,
        'k': k,
        'top_n': top_n,
        'level': level,
        'user_context': asdict(ctx),
    }

    _write_csv(str(args.out_csv), csv_rows)
    _write_json(str(args.out_json), {'meta': meta, 'items': report_items})

    print(f"Wrote CSV: {args.out_csv} ({len(csv_rows)} rows)")
    print(f"Wrote JSON: {args.out_json} ({len(report_items)} goals)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
