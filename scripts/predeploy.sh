#!/usr/bin/env bash
set -euo pipefail

# Activate virtualenv if present (Render uses .venv)
if [ -f "./venv/bin/activate" ]; then
  source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
  source ./.venv/bin/activate
fi

echo "[predeploy] Running Django migrations"
python manage.py migrate --noinput
