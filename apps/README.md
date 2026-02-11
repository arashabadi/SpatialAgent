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

## Backend setup

From repo root:

```bash
cd apps/backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The backend imports `spatialagent` from the repo root.

## Frontend setup

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
