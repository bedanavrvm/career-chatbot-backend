# Deployment Guide

This document focuses on production-like deployment patterns, especially Render.

## 1) Backend deployment (Render)

The backend repository includes `start.sh`, which performs:

- Migrations
- Optional superuser creation (based on env)
- `collectstatic`
- Starts Gunicorn with a computed worker count

Operational notes:

- Ensure `DATABASE_URL` points to a Postgres instance.
- Ensure the process has permission to write any configured logs.

## 2) Required backend environment variables

At minimum:

- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG` (should be false in production)
- `DATABASE_URL`

If using Firebase-authenticated endpoints:

- `FIREBASE_CREDENTIALS_JSON_PATH` or `FIREBASE_CREDENTIALS_JSON_B64`

If using Gemini features:

- `GEMINI_API_KEY`

If `DJANGO_DEBUG` is false:

- `PII_ENCRYPTION_KEY`

## 3) Snapshot-mode operational requirement

In Render-like environments, prefer snapshot mode.

After deploying:

- Ensure you can access `/admin/` with a staff user.
- Navigate to `/admin/onet`.
- Import:
  - Snapshot CSV at `/admin/onet/snapshot/import`
  - Mappings CSV at `/admin/onet/mappings/import`

## 4) Frontend deployment

Frontend is a Vue 3 + Vite app.

It needs:

- Firebase client env vars (`VITE_FIREBASE_*`)
- Optionally `VITE_API_BASE_URL`

If `VITE_API_BASE_URL` is not set, the frontend will default to same-origin calls.

## 5) Cross-origin considerations

If frontend and backend are hosted on different domains:

- Configure backend CORS appropriately.
- Ensure auth headers are accepted.
- Ensure redirects (login/onboarding) behave as expected.
