import re
from typing import Any, Dict, List

from django.conf import settings
from django.db.models import Q


def _tokenize(q: str) -> List[str]:
    toks = [t.strip().lower() for t in re.split(r"[^a-zA-Z0-9]+", (q or "").strip())]
    return [t for t in toks if len(t) >= 3]


def retrieve_catalog_documents(query: str, *, level: str = "", limit: int = 8) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    try:
        from catalog.models import Program  # type: ignore
    except Exception:
        return []

    lvl = (level or "").strip().lower()
    tokens = _tokenize(q)

    vector_rank: Dict[int, int] = {}
    if getattr(settings, 'RAG_USE_PGVECTOR', False):
        try:
            from vectorstore.embeddings import get_embedding  # type: ignore
            from vectorstore.models import ProgramEmbedding  # type: ignore
            from pgvector.django import CosineDistance  # type: ignore

            q_emb = get_embedding(q, task_type='retrieval_query')
            if q_emb:
                vec_limit = max(10, min(100, limit * 12))
                pes = (
                    ProgramEmbedding.objects
                    .filter(embedding__isnull=False)
                    .annotate(distance=CosineDistance('embedding', q_emb))
                    .order_by('distance')[:vec_limit]
                )
                for i, pe in enumerate(pes, 1):
                    vector_rank[int(pe.program_id)] = i
        except Exception:
            vector_rank = {}

    qs = (
        Program.objects.select_related("institution", "field")
        .prefetch_related(
            "requirement_groups",
            "requirement_groups__options",
            "requirement_groups__options__subject",
            "cutoffs",
            "costs",
        )
        .all()
    )

    if lvl:
        qs = qs.filter(level=lvl)

    phrase = q.lower()
    qobj = Q(normalized_name__icontains=phrase) | Q(name__icontains=phrase) | Q(institution__name__icontains=phrase)
    if tokens:
        for t in tokens[:8]:
            qobj |= (
                Q(normalized_name__icontains=t)
                | Q(name__icontains=t)
                | Q(institution__name__icontains=t)
                | Q(field__name__icontains=t)
            )

    candidates = list(qs.filter(qobj)[:200])
    if vector_rank:
        try:
            vp = list(qs.filter(id__in=list(vector_rank.keys()))[:200])
            candidates.extend(vp)
        except Exception:
            pass
    if not candidates:
        return []

    # De-duplicate candidates
    by_id = {}
    for p in candidates:
        try:
            by_id[int(p.id)] = p
        except Exception:
            pass
    candidates = list(by_id.values())

    def score(p) -> float:
        text = f"{p.normalized_name} {p.name} {p.institution.name if p.institution_id else ''} {p.field.name if p.field_id else ''}".lower()
        s: float = 0.0
        if phrase and phrase in text:
            s += 5.0
        for t in tokens[:10]:
            if t in text:
                s += 1.0

        # Hybrid boost: prefer vector-nearest programs (lower distance => smaller rank)
        if vector_rank:
            try:
                r = vector_rank.get(int(p.id))
                if r is not None:
                    denom = float(max(1, len(vector_rank)))
                    s += 3.0 * (1.0 - (float(r - 1) / denom))
            except Exception:
                pass
        return s

    ranked = sorted(candidates, key=lambda p: (-score(p), (p.normalized_name or p.name or "")))
    ranked = ranked[: max(1, min(20, limit * 3))]

    docs: List[Dict[str, Any]] = []
    for idx, p in enumerate(ranked[:limit], 1):
        try:
            reqs = p.requirements_preview()
        except Exception:
            reqs = ""

        cutoff = None
        try:
            cutoffs = list(getattr(p, "cutoffs", []).all())
            if cutoffs:
                yc = sorted(cutoffs, key=lambda c: int(getattr(c, "year", 0) or 0), reverse=True)[0]
                cutoff = {
                    "year": yc.year,
                    "cutoff": float(yc.cutoff),
                    "capacity": yc.capacity,
                    "notes": yc.notes or "",
                }
        except Exception:
            cutoff = None

        cost = None
        try:
            costs = list(getattr(p, "costs", []).all())
            if costs:
                def _cost_ts(c) -> float:
                    dt = getattr(c, "updated_at", None)
                    try:
                        return float(dt.timestamp()) if dt else 0.0
                    except Exception:
                        return 0.0

                pc = sorted(costs, key=_cost_ts, reverse=True)[0]
                cost = {
                    "amount": float(pc.amount) if pc.amount is not None else None,
                    "currency": pc.currency or "KES",
                    "raw_cost": pc.raw_cost or "",
                    "source_id": pc.source_id or "",
                }
        except Exception:
            cost = None

        doc_id = f"program:{p.id}"
        citation = f"P{idx}"
        meta = {
            "citation": citation,
            "doc_id": doc_id,
            "program_id": getattr(p, "id", None),
            "program_code": (p.code or "").strip(),
            "program_name": (p.normalized_name or p.name or "").strip(),
            "institution_name": (p.institution.name or "").strip() if getattr(p, "institution_id", None) else "",
            "institution_code": (p.institution.code or "").strip() if getattr(p, "institution_id", None) else "",
            "level": (p.level or "").strip(),
            "campus": (p.campus or "").strip(),
            "region": (p.region or "").strip(),
            "field_name": (p.field.name or "").strip() if getattr(p, "field_id", None) else "",
            "requirements_preview": reqs,
            "latest_cutoff": cutoff,
            "cost": cost,
        }
        text = (
            f"{meta['program_name']} ({meta['level']}) at {meta['institution_name']}"
            + (f" | Campus: {meta['campus']}" if meta["campus"] else "")
            + (f" | Region: {meta['region']}" if meta["region"] else "")
            + (f" | Field: {meta['field_name']}" if meta["field_name"] else "")
            + (f" | Requirements: {reqs}" if reqs else "")
            + (f" | Latest cutoff: {cutoff['year']}={cutoff['cutoff']}" if cutoff else "")
            + (
                f" | Cost: {cost['amount']} {cost['currency']}" if (cost and cost.get("amount") is not None) else ""
            )
            + (f" | Cost raw: {cost['raw_cost']}" if (cost and cost.get("raw_cost")) else "")
        )

        docs.append({"id": doc_id, "citation": citation, "text": text, "meta": meta})

    return docs
