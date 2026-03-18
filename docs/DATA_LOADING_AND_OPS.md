# Data Loading & Operations Guide

This guide covers how to load data into the system and operate it in:

- Local/dev environments (where full O*NET import may be feasible)
- Render/production-like environments (where snapshot mode is preferred)

## 1) Catalog ETL (KUCCPS)

The backend repository includes ETL management commands (see `README.md`).

Typical flow:

- `python manage.py kuccps_etl --action transform-programs`
- `python manage.py kuccps_etl --action transform-normalize`
- `python manage.py kuccps_etl --action dedup-programs`
- `python manage.py kuccps_etl --action dq-report`
- `python manage.py kuccps_etl --action load`

Key outputs:

- Fields and programs end up in managed catalog models (e.g. `catalog.Field`, `catalog.Program`).

## 2) O*NET data: two modes

### 2.1 Full O*NET mode (local/dev)

Use this when you can import the heavy O*NET SQL tables into Postgres.

Workflow:

1. Import O*NET tables
   - Admin URL: `/admin/onet/import`
2. Generate snapshot table from the heavy tables
   - Admin URL: `/admin/onet/snapshot/generate`
3. Export the snapshot to a CSV for use elsewhere
   - Admin URL: `/admin/onet/snapshot/export`

### 2.2 Snapshot mode (Render / production-friendly)

Use this when you cannot import the full O*NET tables.

Workflow:

1. Import the snapshot CSV
   - Admin URL: `/admin/onet/snapshot/import`
   - Required columns:
     - `onetsoc_code,title,description,job_zone`
2. Import curated Field→Occupation mappings
   - Admin URL: `/admin/onet/mappings/import`
   - Required columns:
     - `field_slug,occupation_code`
   - Optional columns:
     - `weight,notes`

Important:

- `OnetFieldOccupationMapping` is authoritative for which careers appear for a field/program.
- If mappings exist but the snapshot is missing an occupation code, some UIs may show codes without titles/descriptions.

## 3) Mapping suggestion and verification loop

The system supports a loop where you:

1. Generate mapping suggestions (CSV preview)
   - Admin URL: `/admin/onet/mappings/suggest`
   - This does not write to the DB.
2. Review/clean the CSV externally (spreadsheet/editor)
3. Import verified mappings into the DB
   - Admin URL: `/admin/onet/mappings/import`

## 4) Handling missing Fields during mapping import

Mapping import is based on `catalog.Field.slug`.

If your mapping CSV includes field slugs that aren’t present in the DB:

- Preferably run the catalog ETL load so the `Field` rows exist.
- Alternatively, the mapping import UI supports an option to create missing `Field` rows by slug during import.

Operational best practice:

- In production-like environments, load catalog data first.
- Then import snapshot.
- Then import mappings.

## 5) Coverage checks

Use the coverage dashboard to identify gaps:

- Admin URL: `/admin/onet/mappings/coverage`

This is the fastest way to spot:

- Fields with zero mappings
- Programs with no field assigned
- Programs whose field has no O*NET mappings

## 6) Exports for analysis / QA

### 6.1 Snapshot export

- URL: `/admin/onet/snapshot/export`
- CSV columns:
  - `onetsoc_code,title,description,job_zone`

### 6.2 Program → Field → Occupation export

This export joins:

- `catalog.Program`
- its `Program.field`
- curated `OnetFieldOccupationMapping`
- titles/descriptions from `OnetOccupationSnapshot`

- URL: `/admin/onet/exports/program-field-occupations`

Intended uses:

- Sanity-check which occupations each program/field will return
- Offline analytics / coverage QA

## 7) Operational ordering (recommended)

### 7.1 Local/dev (authoring data)

- Run catalog ETL (load Fields/Programs)
- Import full O*NET tables
- Generate snapshot
- Generate mapping suggestions
- Review mappings
- Import mappings
- Export snapshot + program-field-occupations for use on Render

### 7.2 Render/prod-like (consuming data)

- Run catalog ETL load (or otherwise ensure `Field`/`Program` rows exist)
- Import snapshot CSV
- Import mappings CSV
- Validate via coverage dashboard
