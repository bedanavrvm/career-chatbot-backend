# Troubleshooting

This document collects common issues encountered in development and deployment.

## 1) Frontend is logged in, but backend requests fail

Symptoms:

- Firebase login succeeds.
- API calls fail with 401/403 or generic network errors.

Checks:

- Backend has Firebase Admin credentials configured:
  - `FIREBASE_CREDENTIALS_JSON_PATH` or `FIREBASE_CREDENTIALS_JSON_B64`
- Frontend is pointing to the correct backend:
  - `VITE_API_BASE_URL`
- Requests include:
  - `Authorization: Bearer <idToken>`

## 2) O*NET mapping import fails with unknown field slugs

Symptoms:

- Importing `/admin/onet/mappings/import` yields an error like “Unknown field slugs …”.

Root cause:

- The mapping CSV references `field_slug` values that do not exist in `catalog.Field.slug` in this environment.

Fix:

- Load catalog data first (ETL) so the Field rows exist.
- Or, use the mapping import checkbox to create missing `Field` rows by slug during import.

## 3) Careers endpoints return empty results

Most common causes:

- The program has `field = null`.
- The program’s field has no rows in `OnetFieldOccupationMapping`.

Fix:

- Use `/admin/onet/mappings/coverage` to identify gaps.
- Ensure the mapping CSV has been imported.

## 4) Snapshot imported, but occupation titles/descriptions are missing

Root cause:

- Snapshot is missing the `onetsoc_code` referenced by mappings.

Fix:

- Re-import a newer snapshot CSV.
- Export snapshot from a full O*NET environment and import it into Render.

## 5) O*NET full import is too slow / killed

If local O*NET import consumes too much memory/CPU and is killed:

- Prefer importing only the core tables (admin import supports modes).
- Run the import on a machine with more resources.
- Use snapshot mode on constrained hosts.

## 6) Onboarding redirect loops

The frontend router guard:

- Waits for Firebase auth resolution (`authReady`).
- Checks onboarding status.

If onboarding status endpoint is down, the router is designed to avoid trapping users in loops.

If you see persistent loops:

- Verify the onboarding endpoints are reachable.
- Check `VITE_API_BASE_URL`.
- Inspect backend logs for auth failures.
