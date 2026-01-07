from __future__ import annotations

from typing import Any, Dict, List


def compose_response(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ''

    typ = str(payload.get('type') or '').strip()

    if typ == 'recommendations':
        items = payload.get('items') or []
        if not isinstance(items, list) or not items:
            return 'I cannot access the programs catalog right now. Please try again later.'

        lines: List[str] = ['Here are some programs that match your profile:']
        for i, it in enumerate(items[:6], start=1):
            nm = str(it.get('program_name') or '').strip()
            inst = str(it.get('institution_name') or '').strip()
            pid = it.get('program_id')
            tail = f" — {inst}" if inst else ''
            if pid is not None:
                tail = f"{tail} (id {pid})"
            lines.append(f"{i}. {nm}{tail}")

        lines.append('')
        lines.append('Tell me:')
        lines.append('1) Which one should I check eligibility for? (share the id or the number)')
        lines.append('2) Or ask: "details for id 123"')
        return '\n'.join(lines).strip()

    if typ == 'search_results':
        items = payload.get('items') or []
        if not isinstance(items, list) or not items:
            return 'I could not find any programs for that search.'

        lines: List[str] = ['Here are matching programs:']
        for i, it in enumerate(items[:10], start=1):
            nm = str(it.get('program_name') or '').strip()
            inst = str(it.get('institution_name') or '').strip()
            pid = it.get('program_id')
            tail = f" — {inst}" if inst else ''
            if pid is not None:
                tail = f"{tail} (id {pid})"
            lines.append(f"{i}. {nm}{tail}")
        return '\n'.join(lines).strip()

    if typ == 'filtered_results':
        items = payload.get('items') or []
        if not isinstance(items, list) or not items:
            return 'No programs matched that filter.'

        lines: List[str] = ['Filtered results:']
        for i, it in enumerate(items[:10], start=1):
            nm = str(it.get('program_name') or '').strip()
            inst = str(it.get('institution_name') or '').strip()
            pid = it.get('program_id')
            tail = f" — {inst}" if inst else ''
            if pid is not None:
                tail = f"{tail} (id {pid})"
            lines.append(f"{i}. {nm}{tail}")
        return '\n'.join(lines).strip()

    if typ == 'program_details':
        det = payload.get('details') or {}
        if not isinstance(det, dict):
            return ''
        if det.get('detail') == 'not_found':
            return 'I could not find that program.'
        if det.get('detail') == 'catalog_unavailable':
            return 'I cannot access the programs catalog right now. Please try again later.'

        nm = str(det.get('program_name') or '').strip()
        inst = str(det.get('institution_name') or '').strip()
        field = str(det.get('field_name') or '').strip()
        lines: List[str] = [nm]
        if inst:
            lines.append(f"Institution: {inst}")
        if field:
            lines.append(f"Field: {field}")
        return '\n'.join(lines).strip()

    if typ == 'eligibility':
        results = payload.get('results') or []
        if not isinstance(results, list) or not results:
            return 'I could not evaluate eligibility for that right now.'

        lines: List[str] = ['Eligibility results:']
        for r in results[:3]:
            pid = r.get('program_id')
            ok = r.get('eligible')
            missing = r.get('missing') or []
            if ok is True:
                lines.append(f"- Program {pid}: eligible")
            elif ok is False:
                if isinstance(missing, list) and missing:
                    lines.append(f"- Program {pid}: not eligible (missing: {', '.join([str(x) for x in missing[:6]])})")
                else:
                    lines.append(f"- Program {pid}: not eligible")
            else:
                lines.append(f"- Program {pid}: unknown")
        return '\n'.join(lines).strip()

    if typ == 'explanation':
        exp = payload.get('explanation') or {}
        if not isinstance(exp, dict):
            return ''
        if exp.get('detail') == 'not_found':
            return 'I could not find that program.'
        if exp.get('detail') == 'catalog_unavailable':
            return 'I cannot access the programs catalog right now. Please try again later.'

        nm = str(exp.get('program_name') or '').strip()
        inst = str(exp.get('institution_name') or '').strip()
        head = nm + (f" — {inst}" if inst else '')

        br = exp.get('score_breakdown') or {}
        if not isinstance(br, dict):
            br = {}

        lines: List[str] = [f"Why this program: {head}"]
        lines.append('')
        lines.append('Score breakdown (grounded):')
        lines.append(f"- goal_score: {br.get('goal_score')}")
        lines.append(f"- riasec_score: {br.get('riasec_score')}")
        lines.append(f"- grade_margin_score: {br.get('grade_margin_score')}")

        elig = exp.get('eligibility')
        if isinstance(elig, dict):
            ok = elig.get('eligible')
            if ok is True:
                lines.append('')
                lines.append('Eligibility: eligible')
            elif ok is False:
                missing = elig.get('missing') or []
                if isinstance(missing, list) and missing:
                    lines.append('')
                    lines.append('Eligibility: not eligible')
                    lines.append('Missing: ' + ', '.join([str(x) for x in missing[:6]]))
                else:
                    lines.append('')
                    lines.append('Eligibility: not eligible')
            else:
                lines.append('')
                lines.append('Eligibility: unknown')

        return '\n'.join(lines).strip()

    if typ == 'reset':
        return 'Okay — I have cleared the previous results. What would you like to explore next?'

    if typ == 'clarify':
        q = str(payload.get('question') or '').strip()
        return q or 'Could you clarify what you mean?'

    if typ == 'error':
        return str(payload.get('message') or '').strip() or 'Something went wrong.'

    return ''
