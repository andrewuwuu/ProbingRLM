#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
echo "[1/3] Running test suite..."
uv run python -m pytest -q

echo "[2/3] Building SolidJS frontend..."
cd "$ROOT_DIR/frontend"
if [[ ! -d node_modules ]]; then
  npm install
fi
npm run build

echo "[3/3] Starting backend (serving compiled frontend) on http://localhost:8000..."
cd "$ROOT_DIR"
PROBINGRLM_SKIP_FRONTEND_BUILD=1 uv run main.py web
