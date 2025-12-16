#!/usr/bin/env bash
set -euo pipefail

# Activate virtualenv if present (Render uses .venv)
if [ -f "./venv/bin/activate" ]; then
  source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
  source ./.venv/bin/activate
fi

export DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE:-server.settings}
export PORT=${PORT:-8000}

# Optional: collect static files if requested
if [ "${COLLECTSTATIC:-0}" = "1" ]; then
  echo "[start] Collecting static files"
  python manage.py collectstatic --noinput
fi

# Determine worker count (override with WEB_CONCURRENCY)
if [ -z "${WEB_CONCURRENCY:-}" ]; then
  # 2-5 workers heuristic
  WORKERS=$(python - <<'PY'
import multiprocessing as mp
try:
    c = mp.cpu_count() or 2
except Exception:
    c = 2
print(max(2, min(5, (c * 2) + 1)))
PY
)
else
  WORKERS="$WEB_CONCURRENCY"
fi

exec gunicorn server.wsgi:application \
  --bind 0.0.0.0:"$PORT" \
  --workers "$WORKERS" \
  --timeout "${GUNICORN_TIMEOUT:-60}" \
  --log-level "${GUNICORN_LOG_LEVEL:-info}" \
  --access-logfile - \
  --error-logfile -
