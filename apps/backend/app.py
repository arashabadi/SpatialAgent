from __future__ import annotations

import asyncio
import contextlib
import io
import os
import re
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.tools import tool
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialagent.agent import (  # noqa: E402
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    SpatialAgent,
    make_llm,
)
from spatialagent.agent.utils import load_all_tools  # noqa: E402


RUNS_DIR = REPO_ROOT / "apps" / "runs"
DATA_DIR = REPO_ROOT / "data"
FRONTEND_DIST = REPO_ROOT / "apps" / "frontend" / "dist"

RUNS_DIR.mkdir(parents=True, exist_ok=True)

MAX_QUERY_CHARS = 8000
MAX_STDIO_CHARS = 300_000
MAX_MESSAGES = 400
MAX_ARTIFACTS = 300

MODEL_OPTIONS = [
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_GEMINI_MODEL,
    "gpt-4o",
    "o3",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-pro",
]

RUN_STATUS = Literal["queued", "running", "completed", "failed", "blocked"]

QUERY_BLOCKLIST = [
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bmalware\b", re.IGNORECASE),
    re.compile(r"\bransomware\b", re.IGNORECASE),
    re.compile(r"\bcredential\s+steal", re.IGNORECASE),
    re.compile(r"\bprivilege\s+escalat", re.IGNORECASE),
]


@tool("execute_bash")
def execute_bash_safe(
    command: Annotated[str, Field(description="Bash command to execute")],
) -> str:
    """Bash is disabled for safe web sessions."""
    _ = command
    return (
        "Bash execution is disabled in web safe mode. "
        "Use Python-based tools and library functions instead."
    )


@dataclass
class RunRecord:
    run_id: str
    query: str
    model: str
    safe_mode: bool
    tool_retrieval_method: str
    created_at: str
    run_dir: Path
    status: RUN_STATUS = "queued"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    conclusion: str = ""
    stdout: str = ""
    stderr: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    action_count: int = 0
    observation_count: int = 0


class RunRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=MAX_QUERY_CHARS)
    model: str = Field(default=DEFAULT_CLAUDE_MODEL)
    safe_mode: bool = Field(default=True)
    tool_retrieval_method: Literal["llm", "embedding", "all"] = Field(default="llm")
    recursion_limit: int = Field(default=35, ge=10, le=80)
    act_timeout: int = Field(default=240, ge=30, le=1800)
    auto_interpret_figures: bool = Field(default=False)


class RunCreated(BaseModel):
    run_id: str
    status: RUN_STATUS


class RunSummary(BaseModel):
    run_id: str
    status: RUN_STATUS
    model: str
    safe_mode: bool
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    query_preview: str
    error: Optional[str]


class RunDetail(BaseModel):
    run_id: str
    status: RUN_STATUS
    model: str
    safe_mode: bool
    tool_retrieval_method: str
    query: str
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    error: Optional[str]
    conclusion: str
    stdout: str
    stderr: str
    action_count: int
    observation_count: int
    messages: List[Dict[str, str]]


class ModelInfo(BaseModel):
    model: str
    provider: str
    required_env: List[str]
    ready: bool


class ArtifactInfo(BaseModel):
    path: str
    size_bytes: int
    modified_at: str


app = FastAPI(title="SpatialAgent Web API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


RUNS: Dict[str, RunRecord] = {}
RUNS_LOCK = asyncio.Lock()
RUN_EXECUTION_LOCK = asyncio.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int = MAX_STDIO_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n...[truncated]"


def _query_block_reason(query: str) -> Optional[str]:
    for pattern in QUERY_BLOCKLIST:
        if pattern.search(query):
            return f"Blocked by safe mode policy: matched pattern `{pattern.pattern}`"
    return None


def _provider_for_model(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "google"
    if model.startswith("us.anthropic"):
        return "aws-bedrock"
    if model.startswith(("gpt-", "o3", "o4")):
        return "openai-or-azure"
    return "custom"


def _model_env_status(model: str) -> ModelInfo:
    provider = _provider_for_model(model)

    if provider == "anthropic":
        envs = ["ANTHROPIC_API_KEY"]
        ready = bool(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "google":
        envs = ["GEMINI_API_KEY or GOOGLE_API_KEY"]
        ready = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    elif provider == "aws-bedrock":
        envs = ["AWS credentials/profile"]
        ready = bool(os.getenv("AWS_PROFILE") or os.getenv("AWS_ACCESS_KEY_ID"))
    elif provider == "openai-or-azure":
        envs = [
            "OPENAI_API_KEY",
            "or AZURE_API_KEY + AZURE_API_ENDPOINT",
            "or AZURE_OPENAI_API_KEY_SC / AZURE_OPENAI_API_KEY_EUS2",
        ]
        ready = bool(
            os.getenv("OPENAI_API_KEY")
            or (os.getenv("AZURE_API_KEY") and os.getenv("AZURE_API_ENDPOINT"))
            or os.getenv("AZURE_OPENAI_API_KEY_SC")
            or os.getenv("AZURE_OPENAI_API_KEY_EUS2")
        )
    else:
        envs = ["Model-specific credentials"]
        ready = False

    return ModelInfo(model=model, provider=provider, required_env=envs, ready=ready)


def _prepare_env_for_model(model: str) -> None:
    """Apply compatibility env fallbacks before constructing provider clients."""
    if model.startswith("gemini") and not os.getenv("GEMINI_API_KEY"):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            os.environ["GEMINI_API_KEY"] = google_api_key


def _serialize_messages(raw_messages: List[Any]) -> List[Dict[str, str]]:
    serialized: List[Dict[str, str]] = []
    for msg in raw_messages[-MAX_MESSAGES:]:
        cls_name = msg.__class__.__name__.lower()
        if "human" in cls_name:
            role = "user"
        elif "system" in cls_name:
            role = "system"
        else:
            role = "assistant"

        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content)
        serialized.append({"role": role, "content": text})
    return serialized


def _extract_conclusion(messages: List[Dict[str, str]]) -> str:
    for msg in reversed(messages):
        content = msg["content"]
        match = re.search(r"<conclude>(.*?)</conclude>", content, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _count_tag(messages: List[Dict[str, str]], tag: str) -> int:
    marker = f"<{tag}>"
    return sum(message["content"].count(marker) for message in messages)


def _list_artifacts(run_dir: Path) -> List[ArtifactInfo]:
    if not run_dir.exists():
        return []

    artifacts: List[ArtifactInfo] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        rel = path.relative_to(run_dir).as_posix()
        stat = path.stat()
        artifacts.append(
            ArtifactInfo(
                path=rel,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            )
        )
        if len(artifacts) >= MAX_ARTIFACTS:
            break
    return artifacts


def _build_tools_for_run(run_dir: Path, safe_mode: bool):
    tools = load_all_tools(save_path=str(run_dir), data_path=str(DATA_DIR))
    if not safe_mode:
        return tools

    patched = []
    for tool_obj in tools:
        if getattr(tool_obj, "name", "") == "execute_bash":
            patched.append(execute_bash_safe)
        else:
            patched.append(tool_obj)
    return patched


def _execute_run_sync(run: RunRecord, request: RunRequest) -> None:
    run.started_at = _utc_now_iso()
    run.status = "running"

    if request.safe_mode:
        block_reason = _query_block_reason(request.query)
        if block_reason:
            run.status = "blocked"
            run.error = block_reason
            run.finished_at = _utc_now_iso()
            return

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        _prepare_env_for_model(request.model)
        llm = make_llm(request.model, streaming=False, track_cost=True)
        tools = _build_tools_for_run(run.run_dir, request.safe_mode)

        agent = SpatialAgent(
            llm=llm,
            tools=tools,
            data_path=str(DATA_DIR),
            save_path=str(run.run_dir),
            tool_retrieval=True,
            tool_retrieval_method=request.tool_retrieval_method,
            skill_retrieval=True,
            auto_interpret_figures=request.auto_interpret_figures,
            act_timeout=request.act_timeout,
        )

        guarded_query = request.query.strip()
        if request.safe_mode:
            guarded_query = (
                "Safety policy: prioritize Python tools and do not execute bash commands.\n\n"
                f"User request:\n{guarded_query}"
            )

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            final_state = agent.run(
                guarded_query,
                config={
                    "thread_id": f"web-{run.run_id}",
                    "recursion_limit": request.recursion_limit,
                },
            )

        messages = _serialize_messages(final_state.get("messages", []) if final_state else [])
        run.messages = messages
        run.conclusion = _extract_conclusion(messages)
        run.action_count = _count_tag(messages, "act")
        run.observation_count = _count_tag(messages, "observation")
        run.status = "completed"

    except Exception as exc:  # noqa: BLE001
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc(limit=20)
        stderr_buffer.write("\n" + tb)

    run.stdout = _truncate(stdout_buffer.getvalue())
    run.stderr = _truncate(stderr_buffer.getvalue())
    run.finished_at = _utc_now_iso()


async def _run_in_background(run: RunRecord, request: RunRequest) -> None:
    async with RUN_EXECUTION_LOCK:
        await asyncio.to_thread(_execute_run_sync, run, request)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "time": _utc_now_iso()}


@app.get("/api/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    return [_model_env_status(model) for model in MODEL_OPTIONS]


@app.get("/api/runs", response_model=List[RunSummary])
async def list_runs() -> List[RunSummary]:
    async with RUNS_LOCK:
        runs = sorted(RUNS.values(), key=lambda r: r.created_at, reverse=True)
        return [
            RunSummary(
                run_id=run.run_id,
                status=run.status,
                model=run.model,
                safe_mode=run.safe_mode,
                created_at=run.created_at,
                started_at=run.started_at,
                finished_at=run.finished_at,
                query_preview=(run.query[:120] + "..." if len(run.query) > 120 else run.query),
                error=run.error,
            )
            for run in runs
        ]


@app.post("/api/runs", response_model=RunCreated, status_code=202)
async def create_run(request: RunRequest) -> RunCreated:
    query = request.query.strip()
    if len(query) < 5:
        raise HTTPException(status_code=400, detail="Query is too short.")

    if request.model not in MODEL_OPTIONS:
        raise HTTPException(status_code=400, detail="Unsupported model. Pick one from /api/models.")

    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run = RunRecord(
        run_id=run_id,
        query=query,
        model=request.model,
        safe_mode=request.safe_mode,
        tool_retrieval_method=request.tool_retrieval_method,
        created_at=_utc_now_iso(),
        run_dir=run_dir,
    )

    async with RUNS_LOCK:
        RUNS[run_id] = run

    asyncio.create_task(_run_in_background(run, request))
    return RunCreated(run_id=run_id, status=run.status)


@app.get("/api/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: str) -> RunDetail:
    async with RUNS_LOCK:
        run = RUNS.get(run_id)

    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    return RunDetail(
        run_id=run.run_id,
        status=run.status,
        model=run.model,
        safe_mode=run.safe_mode,
        tool_retrieval_method=run.tool_retrieval_method,
        query=run.query,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error=run.error,
        conclusion=run.conclusion,
        stdout=run.stdout,
        stderr=run.stderr,
        action_count=run.action_count,
        observation_count=run.observation_count,
        messages=run.messages,
    )


@app.get("/api/runs/{run_id}/artifacts", response_model=List[ArtifactInfo])
async def list_artifacts(run_id: str) -> List[ArtifactInfo]:
    async with RUNS_LOCK:
        run = RUNS.get(run_id)

    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    return _list_artifacts(run.run_dir)


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
async def download_artifact(run_id: str, artifact_path: str):
    async with RUNS_LOCK:
        run = RUNS.get(run_id)

    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    candidate = (run.run_dir / artifact_path).resolve()
    run_root = run.run_dir.resolve()
    if run_root not in candidate.parents and candidate != run_root:
        raise HTTPException(status_code=400, detail="Invalid artifact path.")

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")

    return FileResponse(candidate)


@app.delete("/api/runs/{run_id}", status_code=204)
async def delete_run(run_id: str):
    async with RUNS_LOCK:
        run = RUNS.pop(run_id, None)

    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    # Keep run files on disk for reproducibility.
    return None


if FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
