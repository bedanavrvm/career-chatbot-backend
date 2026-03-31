# SKILLSYNC — Documentation Index

This folder contains the authoritative, developer-facing documentation for the **career-chatbot-backend** (Django) project.

## Start here

- `ARCHITECTURE.md`
  - High-level overview of the system (backend + frontend) and data flows.
- `BACKEND_GUIDE.md`
  - Backend developer guide: apps/modules, auth, API layout, background jobs, settings.
- `DATA_LOADING_AND_OPS.md`
  - Operations guide: ETL, loading data locally vs Render, O*NET full vs snapshot, mapping import/export workflows.
- `DEPLOYMENT.md`
  - Deployment notes (Render-focused) and environment variables.
- `TROUBLESHOOTING.md`
  - Common problems + fixes.

## Domain-specific

- `ONET_INTEGRATION.md`
  - Deep dive on the O*NET integration, snapshot mode, mapping workflows, and careers APIs.

## Admin/data maintenance

- `ADMIN_OVERRIDES_GUIDE.md`
  - How to maintain overrides such as `CourseSuffixMapping` via the Django admin.

## Quick links (URLs)

### Backend APIs (public)

- Auth
  - `POST /api/auth/register/`
  - `POST /api/auth/login/`
  - `GET /api/auth/me/`
- Conversations
  - Base: `/api/conversations/…` (see `conversations/urls.py`)
- Catalog + ETL
  - Base: `/api/etl/…`, `/api/catalog/…`
- O*NET APIs
  - Base: `/api/onet/…`

### Django Admin (staff-only)

- Django admin home
  - `/admin/`
- O*NET custom dashboard and tools
  - `/admin/onet`
  - Snapshot
    - Generate (local/dev full O*NET): `/admin/onet/snapshot/generate`
    - Import (Render/prod snapshot): `/admin/onet/snapshot/import`
    - Export CSV: `/admin/onet/snapshot/export`
  - Field→Occupation mappings
    - Suggest CSV: `/admin/onet/mappings/suggest`
    - Import CSV: `/admin/onet/mappings/import`
    - Coverage dashboard: `/admin/onet/mappings/coverage`
  - Program→Field→Occupation export CSV
    - `/admin/onet/exports/program-field-occupations`
