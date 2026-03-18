# Backend Developer Guide (Django)

This guide explains how the backend is structured, how requests are authenticated, where key APIs live, and how to develop locally.

## 1) Project layout

At a high level:

- `server/`
  - Django project settings + root URL config.
- `accounts/`
  - Auth endpoints: register/login/me, onboarding helpers.
- `conversations/`
  - Chat sessions/messages APIs.
- `catalog/`
  - Domain models (Fields/Programs/etc) + catalog APIs.
- `onetdata/`
  - O*NET integration (models, mapping APIs, custom admin tools for import/export).
- `etl/`
  - Management command(s) for loading/transformation.
- `vectorstore/`
  - Vector/RAG related logic (when enabled).

## 2) Local setup

See the repository `README.md` for OS-specific commands.

Key points:

- Run migrations:
  - `python manage.py migrate`
- Run the dev server:
  - `python manage.py runserver`

## 3) Configuration (settings)

Settings are defined in `server/settings.py`.

### 3.1 Core env vars

- `DJANGO_DEBUG`
- `DJANGO_SECRET_KEY`
- `DATABASE_URL`

### 3.2 Firebase Admin (backend)

Backend verifies Firebase ID tokens using Firebase Admin credentials.

Provide one of:

- `FIREBASE_CREDENTIALS_JSON_PATH`
- `FIREBASE_CREDENTIALS_JSON_B64`

### 3.3 PII encryption

In non-debug mode, `PII_ENCRYPTION_KEY` must be set.

### 3.4 NLP / RAG providers

- Gemini: `GEMINI_API_KEY`
- Vector/RAG: flags in settings (e.g. pgvector usage).

## 4) URL routing

The root URL router is in `server/urls.py`.

It mounts:

- `/api/auth/…` from `accounts/urls.py`
- `/api/conversations/…` from `conversations/urls.py`
- `/api/etl/…` and `/api/catalog/…` (catalog + ETL APIs)
- `/api/onet/…` (O*NET field/program careers endpoints)

It also mounts custom Django admin pages under `/admin/onet…`.

## 5) Authentication model

### 5.1 How frontend authenticates

- Frontend uses Firebase Auth.
- Frontend sends `Authorization: Bearer <Firebase ID token>` to backend.

### 5.2 Backend verification

Backend endpoints (especially under `/api/auth/…` and any authenticated APIs) validate the Firebase token server-side.

Important implications:

- If Firebase env vars/credentials are missing on the backend, authenticated endpoints will fail.
- If `VITE_API_BASE_URL` is pointing at the wrong backend URL, the frontend can appear “logged in” in Firebase but still fail all backend calls.

## 6) Conversations (chat) APIs

Conversation routes are defined in `conversations/urls.py`.

Typical capabilities include:

- Creating/retrieving a session
- Posting messages (sync, async, and streaming variants)
- Fetching recommendations and status

Implementation lives in the `conversations` app; use `conversations/urls.py` as the canonical index.

## 7) O*NET integration entry points

See:

- `docs/ONET_INTEGRATION.md` for the full operational model.
- `onetdata/admin_views.py` for admin tooling.

Key admin URLs:

- `/admin/onet` (dashboard)
- Snapshot:
  - `/admin/onet/snapshot/generate`
  - `/admin/onet/snapshot/import`
  - `/admin/onet/snapshot/export`
- Mappings:
  - `/admin/onet/mappings/suggest`
  - `/admin/onet/mappings/import`
  - `/admin/onet/mappings/coverage`
- Exports:
  - `/admin/onet/exports/program-field-occupations`

## 8) Background jobs (Celery)

The backend is configured to use Celery (see `server/settings.py`).

Operational considerations:

- Celery requires a broker (commonly Redis).
- Some heavy tasks (like large imports) should run out-of-band.

## 9) Testing

Run tests as in the repo `README.md`:

- `python manage.py test catalog conversations accounts`

## 10) Logging

Logging configuration is defined in `server/settings.py`.

When debugging in production-like environments:

- Ensure the process manager captures stdout/stderr.
- Prefer logging structured context (endpoint, user, trace IDs if present).
