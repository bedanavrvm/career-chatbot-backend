from django.conf import settings
from django.db import models

from pgvector.django import HnswIndex, VectorField


class ProgramEmbedding(models.Model):
    program = models.OneToOneField('catalog.Program', on_delete=models.CASCADE, related_name='embedding')
    embedding = VectorField(dimensions=int(getattr(settings, 'RAG_EMBED_DIM', 768) or 768), null=True)
    model_name = models.CharField(max_length=128, blank=True, default='')
    content_hash = models.CharField(max_length=64, blank=True, default='', db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            HnswIndex(
                name='program_embedding_hnsw',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            ),
        ]
