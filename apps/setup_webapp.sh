#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/apps/backend"
FRONTEND_DIR="$ROOT_DIR/apps/frontend"
CLI_SCRIPT="$ROOT_DIR/apps/spatial_agent"
RUNTIME_DIR="$ROOT_DIR/apps/.runtime"
NPM_CACHE_DIR="$RUNTIME_DIR/npm-cache"

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

mkdir -p "$NPM_CACHE_DIR"

echo "Installing frontend dependencies..."
npm --prefix "$FRONTEND_DIR" --cache "$NPM_CACHE_DIR" install --no-package-lock

if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -d "${CONDA_PREFIX}/bin" ]]; then
  ln -sf "$CLI_SCRIPT" "${CONDA_PREFIX}/bin/spatial_agent"
  echo "Installed CLI command: spatial_agent"
  echo "Try: spatial_agent start"
else
  echo "warning: CONDA_PREFIX not detected; CLI not installed into environment bin."
  echo "You can still run: ./apps/spatial_agent start"
fi

echo "Web app setup complete."
