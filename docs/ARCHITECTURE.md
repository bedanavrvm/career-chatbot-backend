# Architecture Overview

This document describes the overall system architecture for the Delphine Career Chatbot.

## 1) Repositories / components

- **Backend**: `career-chatbot-backend/`
  - Django + Django REST Framework APIs.
  - ETL (management commands) to load domain data.
  - O*NET integration tooling (full import locally; snapshot mode for constrained environments).
  - Background work via Celery (Redis broker).

- **Frontend**: `career-chatbot-frontend/`
  - Vue 3 + Vite.
  - Firebase Authentication (client-side).
  - Calls backend APIs using an `Authorization: Bearer <Firebase ID token>` header.

## 2) High-level request flow

### 2.1 Auth

- The frontend authenticates with Firebase.
- The frontend retrieves an ID token via `src/lib/useAuth.js`.
- The token is sent to the backend in requests:
  - `Authorization: Bearer <idToken>`

Backend endpoints under `/api/auth/…` typically:
- Verify the Firebase ID token.
- Create / update the backend user record.

### 2.2 Chat (core user experience)

- The frontend routes to `/chat` (requires auth + completed onboarding).
- Chat UI calls conversation endpoints under `/api/conversations/…`.
- The backend:
  - Stores sessions/messages.
  - Performs NLP/RAG work (provider-controlled; can be Gemini, etc.).
  - May dispatch background tasks (Celery) depending on endpoint.

### 2.3 Catalog + careers data

The chatbot experience is backed by domain data:

- **Catalog data** (KUCCPS/programs/institutions/fields)
  - Loaded by ETL into managed Django models (app: `catalog`).

- **Careers data** (O*NET)
  - Either:
    - Full O*NET imported into Postgres (local/dev), queried via unmanaged models, then distilled; or
    - Snapshot mode: only `onet_occupation_snapshot` exists (Render-friendly)

- **Curated mappings**
  - `Field -> Occupation` relationships are stored in `OnetFieldOccupationMapping`.
  - This mapping table is the *authoritative* source that drives careers returned by APIs.

## 3) Data model overview (conceptual)

- `catalog.Field`
  - Broad field of study (has `slug`).

- `catalog.Program`
  - Academic program; often linked to a `Field`.

- `onetdata.OnetOccupationSnapshot`
  - Minimal occupation table: `onetsoc_code`, `title`, `description`, `job_zone`.

- `onetdata.OnetFieldOccupationMapping`
  - Curated links from `Field.slug` to `Onet` SOC occupation codes.

## 4) Environment modes

### 4.1 Local / dev (full O*NET mode)

- You can import full O*NET tables into Postgres.
- Use `/admin/onet/import` to import tables.
- Use `/admin/onet/snapshot/generate` to populate `onet_occupation_snapshot` from the heavy tables.

### 4.2 Render / constrained production (snapshot mode)

- You typically avoid importing the full O*NET dataset.
- You instead:
  - Import a snapshot CSV into `OnetOccupationSnapshot` via `/admin/onet/snapshot/import`.
  - Import curated mapping CSVs into `OnetFieldOccupationMapping` via `/admin/onet/mappings/import`.

This is the recommended operational mode for production-like environments where full O*NET import is too heavy.

## 5) Frontend routing + guards

Frontend routes are defined in `src/router/index.js`.

Important guards:
- Routes like `/chat` and `/dashboard` require:
  - `meta.requiresAuth`
  - `meta.requiresOnboarding`
- The router waits for `authReady` (first Firebase auth state resolution) to avoid redirect flicker.
- Onboarding status is checked via backend endpoints (see `src/lib/api.js`).
