from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from django.conf import settings
from django.core.management.base import BaseCommand


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]{1,}")


_DEFAULT_EXCLUDED_SOC_PREFIXES = {
    "55-",
    "53-",
    "51-",
}


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _keywords_from_program_names(names_with_counts: Iterable[Tuple[str, int]], max_keywords: int) -> List[str]:
    stop = {
        "bachelor",
        "of",
        "in",
        "and",
        "with",
        "the",
        "for",
        "arts",
        "science",
        "technology",
        "studies",
        "management",
        "education",
        "business",
        "engineering",
        "medicine",
        "health",
        "programme",
        "program",
        "degree",
        "diploma",
        "certificate",
        "bsc",
        "ba",
        "beng",
        "btech",
        "bed",
        "llb",
        "general",
        "applied",
        "systems",
        "system",
        "development",
        "public",
        "community",
        "services",
        "service",
        "information",
        "communication",
        "communications",
        "media",
        "network",
        "networks",
        "planning",
        "resource",
        "resources",
        "policy",
        "leadership",
        "analysis",
        "analytics",
    }

    c: Counter[str] = Counter()
    for name, freq in names_with_counts:
        toks = _tokens(name)
        for t in toks:
            if t in stop:
                continue
            if len(t) <= 2:
                continue
            c[t] += int(freq)

    return [k for k, _ in c.most_common(max_keywords)]


def _score_occupations(
    occupations: List[Dict[str, str]],
    keywords: List[str],
) -> Dict[str, float]:
    kw = [k.lower() for k in keywords if k and k.strip()]
    if not kw:
        return {}

    scores: Dict[str, float] = defaultdict(float)
    for occ in occupations:
        code = (occ.get("onetsoc_code") or "").strip()
        if not code:
            continue

        if any(code.startswith(pfx) for pfx in _DEFAULT_EXCLUDED_SOC_PREFIXES):
            continue

        title = (occ.get("title") or "").strip()
        desc = (occ.get("description") or "").strip()
        if not (title or desc):
            continue

        title_tokens = set(_tokens(title))
        desc_tokens = set(_tokens(desc))
        if not (title_tokens or desc_tokens):
            continue

        title_hits = sum(1 for k in kw if k in title_tokens)
        if title_hits <= 0:
            continue

        desc_hits = sum(1 for k in kw if k in desc_tokens)
        if title_hits < 2 and desc_hits < 3:
            continue

        s = (3.0 * float(title_hits)) + (1.0 * float(desc_hits))
        if s:
            scores[code] += s
    return scores


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("--output", default="", help="Output CSV path. Defaults to <BASE_DIR>/onet/suggested_field_mappings.csv")
        parser.add_argument("--top-k-programs", type=int, default=30)
        parser.add_argument("--max-keywords", type=int, default=25)
        parser.add_argument("--top-n-occupations", type=int, default=10)
        parser.add_argument("--min-score", type=float, default=6.0)
        parser.add_argument("--field", action="append", default=[], help="Repeatable. Filter by Field slug (case-insensitive).")

    def handle(self, *args, **options):
        from catalog.models import Field, Program
        from onetdata.models import OnetOccupation

        out_arg = (options.get("output") or "").strip()
        if out_arg:
            out_path = Path(out_arg).expanduser().resolve()
        else:
            base = Path(getattr(settings, "BASE_DIR", ".")).resolve()
            out_path = (base / "onet" / "suggested_field_mappings.csv").resolve()

        top_k_programs = int(options.get("top_k_programs") or 30)
        max_keywords = int(options.get("max_keywords") or 25)
        top_n = int(options.get("top_n_occupations") or 25)
        min_score = float(options.get("min_score") or 0.0)
        slugs = [str(s).strip().lower() for s in (options.get("field") or []) if str(s).strip()]

        fields_qs = Field.objects.all().order_by("name")
        if slugs:
            fields_qs = fields_qs.filter(slug__in=slugs)

        occs = list(OnetOccupation.objects.all().values("onetsoc_code", "title", "description"))
        title_by_code = {o["onetsoc_code"]: (o.get("title") or "") for o in occs if o.get("onetsoc_code")}

        out_path.parent.mkdir(parents=True, exist_ok=True)

        wrote = 0
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "field_slug",
                    "field_name",
                    "occupation_code",
                    "occupation_title",
                    "score",
                    "keywords",
                    "sample_programs",
                ]
            )

            for fld in fields_qs:
                prog_qs = (
                    Program.objects.filter(field=fld)
                    .exclude(normalized_name="")
                    .values_list("normalized_name", flat=True)
                )
                names = list(prog_qs)
                if not names:
                    continue

                counts = Counter(names)
                top_names = counts.most_common(top_k_programs)
                sample_programs = " | ".join([n for n, _ in top_names[:10]])

                kws = _keywords_from_program_names(top_names, max_keywords=max_keywords)
                if fld.name:
                    for t in _tokens(fld.name):
                        if t and t not in kws:
                            kws.insert(0, t)

                scores = _score_occupations(occs, kws)
                if not scores:
                    continue

                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                ranked = [(code, sc) for code, sc in ranked if sc >= min_score][:top_n]
                if not ranked:
                    continue

                for code, sc in ranked:
                    title = (title_by_code.get(code, "") or "").strip()
                    if fld.slug != "education":
                        tt = set(_tokens(title))
                        if "teacher" in tt or "teachers" in tt:
                            continue
                    w.writerow(
                        [
                            fld.slug,
                            fld.name,
                            code,
                            title,
                            f"{sc:.3f}",
                            " ".join(kws[:max_keywords]),
                            sample_programs,
                        ]
                    )
                    wrote += 1

        self.stdout.write(str(out_path))
        self.stdout.write(f"rows={wrote}")
