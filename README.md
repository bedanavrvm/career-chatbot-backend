# Backend (Django + ETL) README

## Setup

### Windows (PowerShell)
```
\.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
```

### macOS / Linux
```
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
```

## ETL Commands (KUCCPS)

Run the management command from `career-chatbot-backend/`:

```
python manage.py kuccps_etl --action transform-programs
python manage.py kuccps_etl --action transform-normalize
python manage.py kuccps_etl --action dedup-programs
python manage.py kuccps_etl --action dq-report
python manage.py kuccps_etl --action load
```

### Course Suffix Overrides
Precedence when writing `programs_deduped.csv` and `processed/course_suffix_map.csv`:
1. DB overrides: `CourseSuffixMapping` (active rows)
2. CSV overrides: `backend/scripts/etl/kuccps/mappings/course_suffix_map_overrides.csv`
3. Heuristic majority from valid rows

Manage overrides via Admin → Catalog → Course suffix mappings. See `docs/ADMIN_OVERRIDES_GUIDE.md`.

## APIs
- `GET /api/etl/programs` (DB-backed)
- `GET /api/etl/institutions` (DB-backed)
- `GET /api/etl/fields` (DB-backed)
- `GET /api/etl/search` (DB-backed)
- `POST /api/etl/eligibility` (DB-backed)
- `GET /api/catalog/suffix-mapping` (DB-backed; q, active, pagination)
- `GET /api/catalog/program-costs` (DB-backed; program_code, q, pagination)

## Environment Variables

### Core
- `DJANGO_DEBUG` (default true)
- `DJANGO_SECRET_KEY`
- `DATABASE_URL` (optional; Postgres in production)

### Firebase Admin (backend)
One of:
- `FIREBASE_CREDENTIALS_JSON_PATH` (path to service account JSON)
- `FIREBASE_CREDENTIALS_JSON_B64` (base64-encoded service account JSON)

### Gemini (optional)
- `GEMINI_API_KEY` (if set, Gemini provider can be used)

### PII encryption
- `PII_ENCRYPTION_KEY` (required when `DJANGO_DEBUG` is false)

## Tests

### Windows (PowerShell)
```
\.\venv\Scripts\Activate.ps1
python manage.py test catalog conversations accounts
```

### macOS / Linux
```
source venv/bin/activate
python manage.py test catalog conversations accounts
```

## CI
GitHub Actions workflow at `.github/workflows/ci.yml` runs migrations, tests, and ETL smoke.
