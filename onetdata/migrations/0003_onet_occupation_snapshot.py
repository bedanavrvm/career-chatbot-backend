from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('onetdata', '0002_add_surrogate_ids'),
    ]

    operations = [
        migrations.CreateModel(
            name='OnetOccupationSnapshot',
            fields=[
                ('onetsoc_code', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=150)),
                ('description', models.TextField(blank=True)),
                ('job_zone', models.PositiveSmallIntegerField(blank=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'onet_occupation_snapshot',
            },
        ),
    ]
