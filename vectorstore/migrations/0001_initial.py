import os
import django.db.models.deletion
from django.db import migrations, models
from pgvector.django import HnswIndex, VectorExtension, VectorField


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('catalog', '0005_programcost'),
    ]

    operations = [
        VectorExtension(),
        migrations.CreateModel(
            name='ProgramEmbedding',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding', VectorField(dimensions=int(os.getenv('RAG_EMBED_DIM', '768') or '768'), null=True)),
                ('model_name', models.CharField(blank=True, default='', max_length=128)),
                ('content_hash', models.CharField(blank=True, db_index=True, default='', max_length=64)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('program', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='embedding', to='catalog.program')),
            ],
            options={
                'indexes': [
                    HnswIndex(
                        name='program_embedding_hnsw',
                        fields=['embedding'],
                        m=16,
                        ef_construction=64,
                        opclasses=['vector_cosine_ops'],
                    ),
                ],
            },
        ),
    ]
