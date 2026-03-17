# O*NET Integration (Delphine)

This document describes how O*NET is integrated into this project, how career outcomes and career-path progression are produced, and how to operate the system using the Django Admin GUI.

The integration supports two deployment modes:

- **Full O*NET mode (local/dev)**: O*NET tables exist in the DB (imported) and unmanaged Django models query them.
- **Snapshot mode (Render/production-friendly)**: Only a small managed table (`onet_occupation_snapshot`) is required; APIs will use it and do not require importing the full O*NET dataset.

---

## 1) Architecture Overview

### 1.1 Data sources

- **Catalog domain data** (ETL-loaded):
  - `catalog.Field` (broad field of study)
  - `catalog.Program` (programs offered by institutions)

- **O*NET dataset** (optional, heavy):
  - Imported into Postgres (typically locally) from the O*NET SQL dump.
  - Represented by **unmanaged** Django models in `onetdata.models`.

- **Curated mapping layer** (required for career outcomes):
  - `onetdata.mapping_models.OnetFieldOccupationMapping`
  - Maps a `Field` to one or more O*NET occupations.

- **Snapshot table** (optional but recommended for production):
  - `onetdata.models.OnetOccupationSnapshot`
  - Managed Django table containing only:
    - `onetsoc_code`
    - `title`
    - `description`
    - `job_zone`


### 1.2 Key apps & responsibilities

- `catalog/`
  - Stores `Field`, `Program`, and other catalog structures.
  - Provides the Program careers API (`catalog/careers_api.py`).

- `onetdata/`
  - Unmanaged models for O*NET tables + managed snapshot model.
  - Admin GUI tooling for import, snapshot generation/import, mapping suggestion, mapping import, and coverage.
  - Field careers API (`onetdata/mapping_api.py`).

- `server/urls.py`
  - Routes admin pages and APIs.

---

## 2) Database Models

### 2.1 Catalog models

- `catalog.Field`
  - `name`, `slug`, optional `parent`

- `catalog.Program`
  - `field` (nullable)
  - `normalized_name` is used heavily for mapping and search.


### 2.2 Mapping model (Field → O*NET)

- `onetdata.mapping_models.OnetFieldOccupationMapping`
  - `field` → FK to `catalog.Field`
  - `occupation_code` → O*NET SOC code (`onetsoc_code`)
  - `weight` (optional)
  - `notes` (optional)

This table is the **authoritative source** used by the careers APIs to decide which occupations belong to each field.

If a Field has **no rows** in this table, careers endpoints for programs in that field will likely return **empty** results.


### 2.3 O*NET unmanaged models (heavy tables)

These models represent imported O*NET tables. They are marked `managed = False`.

Defined in `onetdata/models.py`:

- `OnetOccupation` → `occupation_data`
- `OnetJobZoneReference` → `job_zone_reference`
- `OnetJobZone` → `job_zones`
- `OnetEducationTrainingExperience` → `education_training_experience`
- `OnetEteCategory` → `ete_categories`
- `OnetRelatedOccupation` → `related_occupations`
- `OnetSkill` → `skills`
- `OnetInterest` → `interests`
- `OnetTaskStatement` → `task_statements`

**Note about surrogate primary keys:**
Some O*NET tables do not ship with a single-column primary key. The project includes logic/migrations to add surrogate `id` columns to improve admin usability.


### 2.4 Snapshot model (lightweight)

- `onetdata.models.OnetOccupationSnapshot`
  - `db_table = 'onet_occupation_snapshot'`
  - `managed = True`

Fields:

- `onetsoc_code` (PK)
- `title`
- `description` (text)
- `job_zone` (nullable)
- `updated_at` (auto)

This model is intended for **Render deployments** (or any constrained environment) where importing the full O*NET dataset is not feasible.

---

## 3) Admin GUI Operations (day-to-day)

All O*NET tooling is accessed from custom admin pages. These require a staff user.

### 3.1 Import O*NET tables (local/dev)

- **URL:** `/admin/onet/import`
- **Code:** `onetdata/admin_views.py` → `admin_onet_import`

Modes:

- `core` (recommended): imports minimal tables required by this project.
- `all`: imports everything.

The import runs in a background process (Windows-safe detached spawn) and streams logs into the admin page.


### 3.2 Generate Snapshot table (local/dev)

- **URL:** `/admin/onet/snapshot/generate`
- **Code:** `onetdata/admin_views.py` → `admin_onet_snapshot_generate`

This will populate `onet_occupation_snapshot` by reading:

- `OnetOccupation` (`occupation_data`)
- `OnetJobZone` (`job_zones`) if present

Options:

- **Truncate first** (recommended): rebuild from scratch.
- **Max description length**: trims long descriptions to reduce DB/storage and API payload size.


### 3.3 Import Snapshot CSV (Render/production)

- **URL:** `/admin/onet/snapshot/import`
- **Code:** `onetdata/admin_views.py` → `admin_onet_snapshot_import`

Upload a CSV containing:

- `onetsoc_code,title,description,job_zone`

This writes directly to `OnetOccupationSnapshot`.


### 3.4 Suggest mappings (Field → O*NET) as a CSV preview

- **URL:** `/admin/onet/mappings/suggest`
- **Code:** `onetdata/admin_views.py` → `admin_onet_mapping_suggest`

This generates a downloadable CSV of suggested mapping candidates.

It does **not** write to the database.

The suggestion logic includes aggressive filtering to improve precision (SOC exclusions, stopwords, title-weighted scoring).


### 3.5 Import verified mappings into DB

- **URL:** `/admin/onet/mappings/import`
- **Code:** `onetdata/admin_views.py` → `admin_onet_mapping_import`

Upload a reviewed CSV and write to `OnetFieldOccupationMapping`.

Required columns:

- `field_slug`
- `occupation_code`

Optional columns:

- `weight`
- `notes`

Option:

- **Replace existing mappings for those fields** (useful when refreshing a field’s curated set)


### 3.6 Coverage dashboard (Fields + Programs)

- **URL:** `/admin/onet/mappings/coverage`
- **Code:** `onetdata/admin_views.py` → `admin_onet_mapping_coverage`

Shows:

- Field coverage:
  - Total Fields
  - Fields with at least one mapping
  - Unmapped Fields
  - Unmapped fields table ordered by program count

- Program coverage:
  - Total Programs
  - Programs with no field assigned
  - Programs whose field has zero O*NET mappings
  - Tables listing sample programs for quick triage

Query parameters:

- `limit` (unmapped fields rows shown; default 200, max 500)
- `program_limit` (program rows shown; default 200, max 500)

This page is the fastest way to identify gaps like **Law** or **Medicine** and confirm whether the root cause is:

- The program has `field = null`
- The program’s field has no rows in `OnetFieldOccupationMapping`

---

## 4) Career Outcomes & Career Path APIs

The system exposes careers in two contexts:

- **Field careers**: careers for a given `Field`.
- **Program careers**: careers for a given `Program`, derived from its `Field`.

### 4.1 Field careers endpoints

Implemented in `onetdata/mapping_api.py`.

- `GET /api/onet/fields/<field_slug>/careers`
- `GET /api/onet/fields/<field_slug>/career-path`


### 4.2 Program careers endpoints

Implemented in `catalog/careers_api.py`.

- `GET /api/catalog/programs/<id>/careers`
- `GET /api/catalog/programs/<id>/career-path`


### 4.3 Career-path grouping logic (Entry/Mid/Senior)

Career-path grouping is based on **O*NET Job Zone**.

Job Zone mapping:

- `1–2` → `entry`
- `3` → `mid`
- `4–5` → `senior`

The API attaches:

- `job_zone`
- `level` (`entry|mid|senior|unknown`)

Parameters (depending on endpoint):

- `career_path=1|0` (default enabled)
- `path_only=1|0` (return only grouped path)
- `per_level=<int>` (quota per entry/mid/senior)


### 4.4 Snapshot fallback behavior (critical for Render)

Both Field and Program career endpoints prefer `OnetOccupationSnapshot` **when available**:

- If a snapshot row exists for a SOC code, title/description/job_zone come from snapshot.
- If a snapshot row is missing, the API falls back to unmanaged O*NET tables (when present).

This makes snapshot mode safe even if the snapshot CSV is partially populated.

---

## 5) O*NET Occupation browsing/detail endpoints

Implemented in `onetdata/api.py`.

- `GET /api/onet/occupations`
  - Uses snapshot results if present; else uses `OnetOccupation`.

- `GET /api/onet/occupations/<soc_code>`
  - If full O*NET tables exist, includes tasks/skills/interests/related occupations.
  - In snapshot-only mode, returns basic details (no tasks/skills/related) but still returns title/description/job_zone.

---

## 6) Frontend integration

### 6.1 Primary call

The frontend should call the dedicated career-path endpoint for stable grouping.

- `/api/catalog/programs/<id>/career-path`

The frontend API client (`career-chatbot-frontend/src/lib/api.js`) was updated to:

- Call the career-path endpoint
- Preserve backward compatibility by flattening grouped results when needed

If the UI shows no careers for a program, use the coverage dashboard to determine why.

---

## 7) Render deployment strategy (Snapshot mode)

### 7.1 Why snapshot mode

Render environments can struggle with importing the full O*NET dataset (time/memory/CPU constraints). Snapshot mode avoids this by requiring only a single lightweight table.

### 7.2 Required steps

1) Deploy backend
2) Run migrations (creates `onet_occupation_snapshot`)
3) Load snapshot:
   - Use **Admin → O*NET → Snapshot → Import** (`/admin/onet/snapshot/import`)
4) Ensure your verified Field→O*NET mappings exist in DB:
   - Use **Admin → O*NET → Mappings → Import** (`/admin/onet/mappings/import`)

No full O*NET import is required in production.

---

## 8) Troubleshooting

### 8.1 "Some programs in Law/Medicine show no career outcomes"

Likely causes:

- The program has `field = null`
- The program’s `field` exists, but the field has **0 mappings** in `OnetFieldOccupationMapping`

What to do:

- Go to `/admin/onet/mappings/coverage`
- Check:
  - "Programs with no Field"
  - "Programs in unmapped Fields"
  - "Unmapped Fields" list

Then:

- Fix the `Program.field` assignment (ETL / admin edits), or
- Import verified mappings for the field via `/admin/onet/mappings/import`


### 8.2 "Career-path levels are missing"

If `job_zone` is missing:

- In full O*NET mode: confirm `job_zones` table exists and was imported.
- In snapshot mode: confirm your snapshot CSV includes `job_zone`.


### 8.3 "Snapshot generate fails"

Snapshot generation requires `occupation_data` at minimum.

- Ensure you ran `/admin/onet/import` first (core mode is enough).

---

## 9) Key files reference

Backend:

- O*NET models: `onetdata/models.py`
- Mapping model: `onetdata/mapping_models.py`
- Admin GUI: `onetdata/admin_views.py`
- Field careers API: `onetdata/mapping_api.py`
- O*NET occupation APIs: `onetdata/api.py`
- Program careers API: `catalog/careers_api.py`
- URL routes: `server/urls.py`

Templates:

- `templates/admin/onet_import.html`
- `templates/admin/onet_mapping_suggest.html`
- `templates/admin/onet_mapping_import.html`
- `templates/admin/onet_snapshot_generate.html`
- `templates/admin/onet_snapshot_import.html`
- `templates/admin/onet_mapping_coverage.html`
