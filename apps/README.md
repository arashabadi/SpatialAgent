# SpatialAgent Web App (`./apps`)

This directory adds a clean web interface for non-coder users, without changing the core repository code.

## Structure

- `backend/`: FastAPI API wrapper around `SpatialAgent`
- `frontend/`: React + Vite user interface
- `runs/`: per-run outputs (`apps/runs/<run_id>/`)

## Design choices

- Keep all app code isolated under `./apps`
- Default to **safe mode** for browser users
- Keep each run output in a dedicated folder for traceability
- Polling-based UX (simple and robust on localhost)

## One-flow setup (recommended)

Use the original repository flow, then install/run web UI in one place:

```bash
# From repo root
./setup_env.sh
conda activate spatial_agent

# Set one API key matching your chosen model
export ANTHROPIC_API_KEY=your_key
# or export OPENAI_API_KEY=your_key
# or export GOOGLE_API_KEY=your_key

# Web app setup (backend + frontend deps)
./apps/setup_webapp.sh

# Start backend + frontend together
spatial_agent start
```

Default URLs:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000/api`

After `./apps/setup_webapp.sh`, the `spatial_agent` command is linked into your active conda environment.
If not linked, use: `./apps/spatial_agent start`

Generated runtime files are written under `apps/.runtime/` (gitignored), including:
- npm cache
- Vite cache
- web run outputs

## Backend setup (manual)

From repo root:

```bash
cd apps/backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The backend imports `spatialagent` from the repo root.

## Frontend setup (manual)

From repo root:

```bash
cd apps/frontend
npm install
npm run dev
```

Default UI URL: `http://localhost:5173`

## API key environment variables

Use the model provider you select in the UI:

- Claude: `ANTHROPIC_API_KEY`
- OpenAI: `OPENAI_API_KEY`
- Azure OpenAI: `AZURE_API_KEY` + `AZURE_API_ENDPOINT`
- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Build frontend and serve from backend (optional)

```bash
cd apps/frontend
npm run build
```

When `apps/frontend/dist` exists, backend will serve it at `/`.
