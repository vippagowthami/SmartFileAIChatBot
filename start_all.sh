#!/usr/bin/env bash
# Start backend + frontend via the repo's run_all.py
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  . "$ROOT_DIR/.venv/bin/activate"
fi
python3 "$ROOT_DIR/run_all.py"
