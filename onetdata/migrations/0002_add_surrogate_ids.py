from django.db import migrations


def _add_surrogate_ids(apps, schema_editor):
    vendor = getattr(schema_editor.connection, 'vendor', '')
    if vendor != 'postgresql':
        # Local dev may use SQLite; this migration is only required for Postgres where
        # O*NET tables are imported without surrogate PKs.
        return

    schema_editor.execute(
        """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'interests' AND column_name = 'id'
    ) THEN
        ALTER TABLE interests ADD COLUMN id BIGSERIAL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = 'interests' AND constraint_type = 'PRIMARY KEY'
    ) THEN
        ALTER TABLE interests ADD CONSTRAINT interests_pkey PRIMARY KEY (id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'skills' AND column_name = 'id'
    ) THEN
        ALTER TABLE skills ADD COLUMN id BIGSERIAL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = 'skills' AND constraint_type = 'PRIMARY KEY'
    ) THEN
        ALTER TABLE skills ADD CONSTRAINT skills_pkey PRIMARY KEY (id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'related_occupations' AND column_name = 'id'
    ) THEN
        ALTER TABLE related_occupations ADD COLUMN id BIGSERIAL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = 'related_occupations' AND constraint_type = 'PRIMARY KEY'
    ) THEN
        ALTER TABLE related_occupations ADD CONSTRAINT related_occupations_pkey PRIMARY KEY (id);
    END IF;
END $$;
"""
    )


class Migration(migrations.Migration):
    dependencies = [
        ('onetdata', '0001_field_occupation_mapping'),
    ]

    operations = [
        migrations.RunPython(_add_surrogate_ids, reverse_code=migrations.RunPython.noop)
    ]
