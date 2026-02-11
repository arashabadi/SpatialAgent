#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/apps/backend"
FRONTEND_DIR="$ROOT_DIR/apps/frontend"

if ! command -v python >/dev/null 2>&1; then
  echo "python is required but not found in PATH"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but not found in PATH"
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "spatial_agent" ]]; then
  echo "warning: active conda env is '${CONDA_DEFAULT_ENV:-none}'. expected 'spatial_agent'."
  echo "continuing anyway..."
fi

echo "Installing backend web dependencies into current Python environment..."
python -m pip install -r "$BACKEND_DIR/requirements.txt"

echo "Installing frontend dependencies..."
npm --prefix "$FRONTEND_DIR" install

echo "Web app setup complete."
