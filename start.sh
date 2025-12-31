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

if [ "${RUN_MIGRATIONS:-0}" = "1" ]; then
  echo "[start] Applying database migrations"
  python manage.py migrate --noinput
fi

if [ "${CREATE_SUPERUSER:-0}" = "1" ]; then
  if [ -n "${DJANGO_SUPERUSER_USERNAME:-}" ] && [ -n "${DJANGO_SUPERUSER_EMAIL:-}" ] && [ -n "${DJANGO_SUPERUSER_PASSWORD:-}" ]; then
    echo "[start] Ensuring Django superuser exists"
    python - <<'PY'
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', os.getenv('DJANGO_SETTINGS_MODULE', 'server.settings'))
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()
username = os.environ['DJANGO_SUPERUSER_USERNAME']
email = os.environ['DJANGO_SUPERUSER_EMAIL']
password = os.environ['DJANGO_SUPERUSER_PASSWORD']

u, created = User.objects.get_or_create(username=username, defaults={'email': email})
changed = False
if not u.is_staff:
    u.is_staff = True
    changed = True
if not u.is_superuser:
    u.is_superuser = True
    changed = True
if created:
    u.set_password(password)
    changed = True
else:
    # If a password is provided, keep existing password unless you explicitly rotate it elsewhere.
    pass

if changed:
    if not u.email:
        u.email = email
    u.save()
PY
  else
    echo "[start] CREATE_SUPERUSER=1 set but missing DJANGO_SUPERUSER_USERNAME/EMAIL/PASSWORD"
  fi
fi

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
