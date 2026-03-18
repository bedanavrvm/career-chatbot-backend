# Admin Overrides Guide

This document describes admin-maintained overrides that affect how catalog data behaves.

## 1) Course suffix overrides

The ETL process generates normalized/deduplicated program outputs, and may infer course “suffixes”.

Overrides are applied in the following precedence order when writing outputs such as `programs_deduped.csv` and `processed/course_suffix_map.csv`:

1. DB overrides: `CourseSuffixMapping` (active rows)
2. CSV overrides: `backend/scripts/etl/kuccps/mappings/course_suffix_map_overrides.csv`
3. Heuristic majority from valid rows

## 2) Managing overrides in Django admin

- Go to `/admin/`
- Navigate to:
  - Catalog → Course suffix mappings

Guidelines:

- Prefer DB overrides for production-like behavior.
- Keep overrides minimal and explicit.
- When changing overrides, re-run the relevant ETL steps so downstream outputs reflect changes.

## 3) Verification

After updating overrides:

- Re-run ETL steps that produce outputs depending on suffix mappings.
- Use catalog APIs to spot-check:
  - `GET /api/catalog/suffix-mapping`
