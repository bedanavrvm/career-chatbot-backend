# Backend (Django + ETL) README

## Setup
```
source venv/bin/activate
pip install -r backend/requirements.txt
python backend/manage.py migrate
```

## ETL Commands (KUCCPS)
```
source venv/bin/activate
python backend/scripts/etl/kuccps/etl.py transform-programs --config backend/scripts/etl/kuccps/config.yaml
python backend/scripts/etl/kuccps/etl.py transform-normalize --config backend/scripts/etl/kuccps/config.yaml
python backend/scripts/etl/kuccps/etl.py dedup-programs --config backend/scripts/etl/kuccps/config.yaml
python backend/scripts/etl/kuccps/etl.py dq-report --config backend/scripts/etl/kuccps/config.yaml
python backend/scripts/etl/kuccps/etl.py load --config backend/scripts/etl/kuccps/config.yaml
```

### Course Suffix Overrides
Precedence when writing `programs_deduped.csv` and `processed/course_suffix_map.csv`:
1. DB overrides: `CourseSuffixMapping` (active rows)
2. CSV overrides: `backend/scripts/etl/kuccps/mappings/course_suffix_map_overrides.csv`
3. Heuristic majority from valid rows

Manage overrides via Admin → Catalog → Course suffix mappings. See `docs/ADMIN_OVERRIDES_GUIDE.md`.

## APIs
- `GET /api/etl/programs` (CSV-backed)
- `GET /api/etl/institutions` (CSV-backed)
- `GET /api/etl/fields` (CSV-backed)
- `GET /api/etl/search` (CSV-backed)
- `POST /api/etl/eligibility` (CSV-backed)
- `GET /api/catalog/suffix-mapping` (DB-backed; q, active, pagination)
- `GET /api/catalog/program-costs` (DB-backed; program_code, q, pagination)

## Tests
```
source venv/bin/activate
python backend/manage.py test --verbosity 2
```

## CI
GitHub Actions workflow at `.github/workflows/ci.yml` runs migrations, tests, and ETL smoke.
