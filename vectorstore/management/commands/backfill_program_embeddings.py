import hashlib
import os
from django.core.management.base import BaseCommand

from catalog.models import Program
from vectorstore.embeddings import get_embedding
from vectorstore.models import ProgramEmbedding


def _program_text(p: Program) -> str:
    inst = p.institution.name if getattr(p, 'institution_id', None) else ''
    field = p.field.name if getattr(p, 'field_id', None) else ''
    try:
        reqs = p.requirements_preview()
    except Exception:
        reqs = ''
    parts = [
        (p.normalized_name or p.name or '').strip(),
        inst.strip(),
        field.strip(),
        (p.level or '').strip(),
        (p.region or '').strip(),
        (p.campus or '').strip(),
        reqs.strip(),
    ]
    return " | ".join([x for x in parts if x])


class Command(BaseCommand):
    help = 'Backfill pgvector embeddings for catalog programs.'

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=0)

    def handle(self, *args, **options):
        model_name = (os.getenv('GEMINI_EMBEDDING_MODEL', 'text-embedding-004') or 'text-embedding-004').strip()
        limit = int(options.get('limit') or 0)
        qs = Program.objects.select_related('institution', 'field').all()
        if limit > 0:
            qs = qs[:limit]

        updated = 0
        skipped = 0
        missing = 0

        for p in qs:
            text = _program_text(p)
            if not text:
                skipped += 1
                continue

            h = hashlib.sha256(text.encode('utf-8')).hexdigest()
            pe, _ = ProgramEmbedding.objects.get_or_create(program=p)
            if pe.content_hash == h and pe.embedding is not None:
                skipped += 1
                continue

            emb = get_embedding(text, task_type='retrieval_document')
            if not emb:
                missing += 1
                continue

            pe.embedding = emb
            pe.model_name = model_name
            pe.content_hash = h
            pe.save(update_fields=['embedding', 'model_name', 'content_hash', 'updated_at'])
            updated += 1

        self.stdout.write(f'updated={updated} skipped={skipped} missing_embedding={missing}')
