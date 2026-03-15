from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ('catalog', '0005_programcost'),
    ]

    operations = [
        migrations.CreateModel(
            name='OnetFieldOccupationMapping',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('weight', models.DecimalField(blank=True, decimal_places=3, max_digits=8, null=True)),
                ('notes', models.CharField(blank=True, max_length=255)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                (
                    'field',
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='onet_mappings', to='catalog.field'),
                ),
                (
                    'occupation_code',
                    models.CharField(max_length=10),
                ),
            ],
            options={
                'ordering': ['field_id', '-weight', 'occupation_code'],
                'unique_together': {('field', 'occupation_code')},
            },
        ),
        migrations.AddIndex(
            model_name='onetfieldoccupationmapping',
            index=models.Index(fields=['occupation_code'], name='onet_field_occ_code_idx'),
        ),
    ]
