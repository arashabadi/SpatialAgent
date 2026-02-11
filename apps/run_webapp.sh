#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/apps/backend"
FRONTEND_DIR="$ROOT_DIR/apps/frontend"
RUNTIME_DIR="$ROOT_DIR/apps/.runtime"
RUNS_DIR="$RUNTIME_DIR/runs"
NPM_CACHE_DIR="$RUNTIME_DIR/npm-cache"
VITE_CACHE_DIR="$RUNTIME_DIR/vite-cache"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

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
  echo "activate with: conda activate spatial_agent"
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "frontend dependencies missing. run: ./apps/setup_webapp.sh"
  exit 1
fi

mkdir -p "$RUNS_DIR" "$NPM_CACHE_DIR" "$VITE_CACHE_DIR"

check_port_free() {
  local port="$1"
  local service_name="$2"
  local env_name="$3"
  if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "error: ${service_name} port ${port} is already in use."
      echo "stop the existing process or run with ${env_name}=<new_port>."
      exit 1
    fi
  fi
}

check_port_free "$BACKEND_PORT" "backend" "BACKEND_PORT"
check_port_free "$FRONTEND_PORT" "frontend" "FRONTEND_PORT"

echo "Starting backend on http://localhost:${BACKEND_PORT}"
(
  cd "$BACKEND_DIR"
  SPATIALAGENT_WEB_RUNS_DIR="$RUNS_DIR" \
    python -m uvicorn app:app --reload --host 0.0.0.0 --port "$BACKEND_PORT"
) &
BACK_PID=$!

echo "Starting frontend on http://localhost:${FRONTEND_PORT}"
(
  cd "$FRONTEND_DIR"
  NPM_CONFIG_CACHE="$NPM_CACHE_DIR" \
    VITE_CACHE_DIR="$VITE_CACHE_DIR" \
    VITE_API_BASE="http://localhost:${BACKEND_PORT}/api" \
    npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
) &
FRONT_PID=$!

cleanup() {
  echo "Shutting down..."
  kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

while true; do
  if ! kill -0 "$BACK_PID" 2>/dev/null; then
    wait "$BACK_PID" 2>/dev/null || true
    break
  fi
  if ! kill -0 "$FRONT_PID" 2>/dev/null; then
    wait "$FRONT_PID" 2>/dev/null || true
    break
  fi
  sleep 1
done
