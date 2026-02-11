import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  FlaskConical,
  LoaderCircle,
  Play,
  ShieldCheck,
  ShieldOff,
  Sparkles,
  TerminalSquare
} from 'lucide-react';

type RunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'blocked';
type RetrievalMethod = 'llm' | 'embedding' | 'all';

type ModelInfo = {
  model: string;
  provider: string;
  required_env: string[];
  ready: boolean;
};

type RunSummary = {
  run_id: string;
  status: RunStatus;
  model: string;
  safe_mode: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  query_preview: string;
  error: string | null;
};

type RunMessage = {
  role: 'user' | 'assistant' | 'system';
  content: string;
};

type RunDetail = {
  run_id: string;
  status: RunStatus;
  model: string;
  safe_mode: boolean;
  tool_retrieval_method: RetrievalMethod;
  query: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  conclusion: string;
  stdout: string;
  stderr: string;
  action_count: number;
  observation_count: number;
  messages: RunMessage[];
};

type ArtifactInfo = {
  path: string;
  size_bytes: number;
  modified_at: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000/api';
const TERMINAL_STATUSES = new Set<RunStatus>(['completed', 'failed', 'blocked']);

const fmtTime = (value: string | null): string => {
  if (!value) return '-';
  return new Date(value).toLocaleString();
};

const artifactUrl = (runId: string, path: string): string => {
  const encoded = path.split('/').map(encodeURIComponent).join('/');
  return `${API_BASE}/runs/${encodeURIComponent(runId)}/artifacts/${encoded}`;
};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === 'string') {
        detail = payload.detail;
      }
    } catch {
      // Ignore parse failures.
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [activeRunId, setActiveRunId] = useState<string>('');
  const [activeRun, setActiveRun] = useState<RunDetail | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactInfo[]>([]);

  const [query, setQuery] = useState('Summarize key spatial biology skills in this repository and suggest a beginner-friendly first analysis workflow.');
  const [model, setModel] = useState('');
  const [safeMode, setSafeMode] = useState(true);
  const [retrievalMethod, setRetrievalMethod] = useState<RetrievalMethod>('llm');

  const [busy, setBusy] = useState(false);
  const [errorText, setErrorText] = useState('');

  const activeModelInfo = useMemo(
    () => models.find((item) => item.model === model) ?? null,
    [models, model]
  );

  const loadRuns = async (): Promise<void> => {
    const data = await apiFetch<RunSummary[]>('/runs');
    setRuns(data);
    if (!activeRunId && data.length > 0) {
      setActiveRunId(data[0].run_id);
    }
  };

  const loadRunDetail = async (runId: string): Promise<void> => {
    const detail = await apiFetch<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
    setActiveRun(detail);

    if (TERMINAL_STATUSES.has(detail.status)) {
      const files = await apiFetch<ArtifactInfo[]>(`/runs/${encodeURIComponent(runId)}/artifacts`);
      setArtifacts(files);
    } else {
      setArtifacts([]);
    }
  };

  useEffect(() => {
    const bootstrap = async (): Promise<void> => {
      try {
        setBusy(true);
        setErrorText('');
        const [modelData] = await Promise.all([
          apiFetch<ModelInfo[]>('/models'),
          loadRuns()
        ]);
        setModels(modelData);
        if (!model && modelData.length > 0) {
          const ready = modelData.find((item) => item.ready);
          setModel((ready ?? modelData[0]).model);
        }
      } catch (error) {
        setErrorText(error instanceof Error ? error.message : 'Unable to load app data.');
      } finally {
        setBusy(false);
      }
    };

    void bootstrap();
  }, []);

  useEffect(() => {
    if (!activeRunId) return;
    setArtifacts([]);
    void loadRunDetail(activeRunId).catch((error: unknown) => {
      setErrorText(error instanceof Error ? error.message : 'Unable to load run details.');
    });
  }, [activeRunId]);

  useEffect(() => {
    if (!activeRun || !activeRunId) return;
    if (TERMINAL_STATUSES.has(activeRun.status)) return;

    const timer = window.setInterval(() => {
      void loadRunDetail(activeRunId)
        .then(() => loadRuns())
        .catch((error: unknown) => {
          setErrorText(error instanceof Error ? error.message : 'Auto-refresh failed.');
        });
    }, 2500);

    return () => window.clearInterval(timer);
  }, [activeRun, activeRunId]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (!query.trim()) return;

    try {
      setBusy(true);
      setErrorText('');
      const created = await apiFetch<{ run_id: string; status: RunStatus }>('/runs', {
        method: 'POST',
        body: JSON.stringify({
          query,
          model,
          safe_mode: safeMode,
          tool_retrieval_method: retrievalMethod
        })
      });

      setActiveRunId(created.run_id);
      setArtifacts([]);
      await loadRuns();
      await loadRunDetail(created.run_id);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : 'Unable to start run.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="bg-orb bg-orb-cyan" />
      <div className="bg-orb bg-orb-blue" />

      <header className="panel hero">
        <div>
          <p className="eyebrow">SpatialAgent Web</p>
          <h1>Non-coder workflow for spatial biology analysis</h1>
          <p className="subtle">FastAPI backend + React UI with safe-mode defaults and isolated run outputs.</p>
        </div>
        <FlaskConical className="hero-icon" />
      </header>

      <main className="grid-main">
        <section className="panel">
          <h2>New Analysis</h2>
          <form onSubmit={onSubmit} className="stack">
            <label>
              Prompt
              <textarea
                rows={8}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Describe the biology task in plain language..."
              />
            </label>

            <div className="row-2">
              <label>
                Model
                <select value={model} onChange={(event) => setModel(event.target.value)}>
                  {models.map((item) => (
                    <option value={item.model} key={item.model}>
                      {item.model} {item.ready ? '' : '(env missing)'}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Retrieval
                <select
                  value={retrievalMethod}
                  onChange={(event) => setRetrievalMethod(event.target.value as RetrievalMethod)}
                >
                  <option value="llm">LLM (recommended)</option>
                  <option value="embedding">Embedding</option>
                  <option value="all">All tools</option>
                </select>
              </label>
            </div>

            <button type="button" className="toggle" onClick={() => setSafeMode((value) => !value)}>
              {safeMode ? <ShieldCheck size={18} /> : <ShieldOff size={18} />}
              Safe mode: {safeMode ? 'on' : 'off'}
            </button>

            <button type="submit" disabled={busy || !model || !query.trim()} className="primary-btn">
              {busy ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
              Run Analysis
            </button>
          </form>
        </section>

        <section className="panel">
          <h2>Run Status</h2>
          <div className="status-box">
            <p className={`status-pill status-${activeRun?.status ?? 'queued'}`}>
              <Sparkles size={14} /> {activeRun?.status ?? 'No run selected'}
            </p>
            <p><strong>Created:</strong> {fmtTime(activeRun?.created_at ?? null)}</p>
            <p><strong>Started:</strong> {fmtTime(activeRun?.started_at ?? null)}</p>
            <p><strong>Finished:</strong> {fmtTime(activeRun?.finished_at ?? null)}</p>
            <p><strong>Actions:</strong> {activeRun?.action_count ?? 0}</p>
            <p><strong>Observations:</strong> {activeRun?.observation_count ?? 0}</p>
          </div>

          {activeModelInfo && (
            <div className="hint-box">
              <h3>Model Environment</h3>
              <p>Provider: {activeModelInfo.provider}</p>
              <p>Requires: {activeModelInfo.required_env.join(', ')}</p>
              <p>Status: {activeModelInfo.ready ? 'ready' : 'missing env vars'}</p>
            </div>
          )}

          {activeRun?.error && (
            <div className="error-box">
              <AlertTriangle size={16} />
              <span>{activeRun.error}</span>
            </div>
          )}
        </section>
      </main>

      <section className="panel">
        <h2>Conclusion</h2>
        <pre className="output-box">{activeRun?.conclusion || 'No conclusion yet. Start a run or open one from history.'}</pre>

        <details>
          <summary><TerminalSquare size={16} /> Runtime logs</summary>
          <pre className="output-box">{activeRun?.stdout || '(no stdout)'}</pre>
          {activeRun?.stderr ? <pre className="output-box error-text">{activeRun.stderr}</pre> : null}
        </details>
      </section>

      <section className="panel split-panel">
        <div>
          <h2>Message Timeline</h2>
          <div className="timeline">
            {(activeRun?.messages ?? []).map((message, index) => (
              <article key={`${message.role}-${index}`} className="timeline-item">
                <p className={`role-tag role-${message.role}`}>{message.role}</p>
                <pre>{message.content}</pre>
              </article>
            ))}
          </div>
        </div>

        <div>
          <h2>Run History</h2>
          <div className="history-list">
            {runs.map((run) => (
              <button
                key={run.run_id}
                type="button"
                className={`history-item ${run.run_id === activeRunId ? 'active' : ''}`}
                onClick={() => setActiveRunId(run.run_id)}
              >
                <span>{run.query_preview}</span>
                <span className={`status-dot status-${run.status}`}>{run.status}</span>
              </button>
            ))}
          </div>

          <h2>Artifacts</h2>
          <ul className="artifact-list">
            {artifacts.length === 0 ? (
              <li className="subtle">No files yet.</li>
            ) : (
              artifacts.map((artifact) => (
                <li key={artifact.path}>
                  <a href={artifactUrl(activeRunId, artifact.path)} target="_blank" rel="noreferrer">
                    {artifact.path}
                  </a>
                  <small>{Math.max(1, Math.round(artifact.size_bytes / 1024))} KB</small>
                </li>
              ))
            )}
          </ul>
        </div>
      </section>

      {errorText ? <footer className="error-box"><AlertTriangle size={16} /> {errorText}</footer> : null}
    </div>
  );
}
