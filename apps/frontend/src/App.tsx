import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BookOpenCheck,
  BrainCircuit,
  ChevronRight,
  Copy,
  FileArchive,
  FlaskConical,
  History,
  Laptop,
  LoaderCircle,
  Play,
  Server,
  ShieldAlert,
  ShieldCheck,
  ShieldOff,
  Sparkles,
  TerminalSquare
} from 'lucide-react';

type RunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'blocked';
type RetrievalMethod = 'llm' | 'embedding' | 'all';
type BuilderMode = 'guided' | 'custom';
type ExecutionProfile = 'interactive' | 'hpc';

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

type WorkflowTemplate = {
  key: string;
  title: string;
  summary: string;
  when_to_use: string;
  default_question: string;
  suggested_outputs: string[];
};

type SafetyProfile = {
  safe_mode_default: boolean;
  blocked_capabilities: string[];
  query_block_patterns: string[];
  notes: string[];
};

type GuidedInputs = {
  scientificQuestion: string;
  organism: string;
  tissue: string;
  datasetPath: string;
  expectedOutputs: string;
  constraints: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000/api';
const TERMINAL_STATUSES = new Set<RunStatus>(['completed', 'failed', 'blocked']);

const STATUS_LABEL: Record<RunStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  blocked: 'Blocked'
};

const RETRIEVAL_HINT: Record<RetrievalMethod, string> = {
  llm: 'Highest relevance for biology tasks (recommended).',
  embedding: 'Fast local retrieval, useful for lightweight passes.',
  all: 'Loads all tools, slower and usually unnecessary for beginners.'
};

const fmtTime = (value: string | null): string => {
  if (!value) return '-';
  return new Date(value).toLocaleString();
};

const formatSizeKB = (bytes: number): string => `${Math.max(1, Math.round(bytes / 1024))} KB`;

const artifactUrl = (runId: string, path: string): string => {
  const encoded = path.split('/').map(encodeURIComponent).join('/');
  return `${API_BASE}/runs/${encodeURIComponent(runId)}/artifacts/${encoded}`;
};

const toPythonTripleQuote = (value: string): string =>
  value.replace(/\\/g, '\\\\').replace(/"""/g, '\\"""');

const buildGuidedPrompt = (workflow: WorkflowTemplate | null, inputs: GuidedInputs): string => {
  const objective = inputs.scientificQuestion.trim() || workflow?.default_question || 'Summarize the main biological findings from this spatial dataset.';
  const outputText = inputs.expectedOutputs.trim() || (workflow?.suggested_outputs.join(', ') ?? 'clear interpretation, key tables, and figures');
  const constraints = inputs.constraints.trim() || 'Use reproducible steps and call out uncertainty explicitly.';

  return [
    `Workflow focus: ${workflow?.title ?? 'Exploratory Spatial Biology Analysis'}`,
    `Goal: ${objective}`,
    `Organism: ${inputs.organism || 'not specified'}`,
    `Tissue / region: ${inputs.tissue.trim() || 'not specified'}`,
    `Dataset path or source: ${inputs.datasetPath.trim() || 'not provided'}`,
    `Expected outputs: ${outputText}`,
    `Constraints: ${constraints}`,
    '',
    'Please run this as a stepwise analysis for a non-coder biologist:',
    '1. Validate assumptions and input structure.',
    '2. Execute the relevant analysis workflow.',
    '3. Summarize biological findings and caveats in plain language.'
  ].join('\n');
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
  const [workflows, setWorkflows] = useState<WorkflowTemplate[]>([]);
  const [safety, setSafety] = useState<SafetyProfile | null>(null);

  const [activeRunId, setActiveRunId] = useState<string>('');
  const [activeRun, setActiveRun] = useState<RunDetail | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactInfo[]>([]);

  const [builderMode, setBuilderMode] = useState<BuilderMode>('guided');
  const [workflowKey, setWorkflowKey] = useState('');
  const [scientificQuestion, setScientificQuestion] = useState('');
  const [organism, setOrganism] = useState('human');
  const [tissue, setTissue] = useState('');
  const [datasetPath, setDatasetPath] = useState('');
  const [expectedOutputs, setExpectedOutputs] = useState('');
  const [constraints, setConstraints] = useState('');

  const [query, setQuery] = useState('');
  const [queryTouched, setQueryTouched] = useState(false);

  const [executionProfile, setExecutionProfile] = useState<ExecutionProfile>('interactive');
  const [model, setModel] = useState('');
  const [safeMode, setSafeMode] = useState(true);
  const [retrievalMethod, setRetrievalMethod] = useState<RetrievalMethod>('llm');
  const [recursionLimit, setRecursionLimit] = useState(35);
  const [actTimeout, setActTimeout] = useState(240);
  const [autoInterpretFigures, setAutoInterpretFigures] = useState(false);

  const [busy, setBusy] = useState(false);
  const [errorText, setErrorText] = useState('');
  const [infoText, setInfoText] = useState('');

  const activeModelInfo = useMemo(
    () => models.find((item) => item.model === model) ?? null,
    [models, model]
  );

  const selectedWorkflow = useMemo(
    () => workflows.find((item) => item.key === workflowKey) ?? null,
    [workflows, workflowKey]
  );

  const generatedPrompt = useMemo(
    () =>
      buildGuidedPrompt(selectedWorkflow, {
        scientificQuestion,
        organism,
        tissue,
        datasetPath,
        expectedOutputs,
        constraints
      }),
    [selectedWorkflow, scientificQuestion, organism, tissue, datasetPath, expectedOutputs, constraints]
  );

  const hpcScript = useMemo(() => {
    const rawQuery = query.trim() || generatedPrompt;
    const guarded = safeMode
      ? `Safety policy: prioritize Python tools and do not execute bash commands.\n\nUser request:\n${rawQuery}`
      : rawQuery;
    const escapedQuery = toPythonTripleQuote(guarded);

    return [
      'python - <<\'PY\'',
      'from spatialagent.agent import SpatialAgent, make_llm',
      '',
      `query = """${escapedQuery}"""`,
      `model_name = "${model || 'claude-sonnet-4-5-20250929'}"`,
      '',
      'agent = SpatialAgent(',
      '    llm=make_llm(model_name, streaming=False),',
      '    save_path="./experiments/hpc_run",',
      '    tool_retrieval=True,',
      `    tool_retrieval_method="${retrievalMethod}",`,
      `    auto_interpret_figures=${autoInterpretFigures ? 'True' : 'False'},`,
      `    act_timeout=${Math.min(1800, Math.max(30, actTimeout))},`,
      ')',
      '',
      'result = agent.run(',
      '    query,',
      '    config={',
      '        "thread_id": "hpc_run_01",',
      `        "recursion_limit": ${Math.min(80, Math.max(10, recursionLimit))}`,
      '    }',
      ')',
      'print("Run completed.")',
      'print(result.get("messages", [])[-1].content if result else "No result payload")',
      'PY'
    ].join('\n');
  }, [query, generatedPrompt, safeMode, model, retrievalMethod, autoInterpretFigures, actTimeout, recursionLimit]);

  const slurmScript = useMemo(
    () =>
      [
        '#!/bin/bash',
        '#SBATCH --job-name=spatialagent',
        '#SBATCH --time=08:00:00',
        '#SBATCH --cpus-per-task=8',
        '#SBATCH --mem=64G',
        '#SBATCH --output=spatialagent_%j.log',
        '',
        'source ~/.bashrc',
        'conda activate spatial_agent',
        '',
        hpcScript
      ].join('\n'),
    [hpcScript]
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

        const [modelData, workflowData, safetyData, runData] = await Promise.all([
          apiFetch<ModelInfo[]>('/models'),
          apiFetch<WorkflowTemplate[]>('/workflows'),
          apiFetch<SafetyProfile>('/safety'),
          apiFetch<RunSummary[]>('/runs')
        ]);

        setModels(modelData);
        setWorkflows(workflowData);
        setSafety(safetyData);
        setRuns(runData);

        if (!model && modelData.length > 0) {
          const ready = modelData.find((item) => item.ready);
          setModel((ready ?? modelData[0]).model);
        }

        if (!workflowKey && workflowData.length > 0) {
          setWorkflowKey(workflowData[0].key);
        }

        if (!activeRunId && runData.length > 0) {
          setActiveRunId(runData[0].run_id);
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
    if (queryTouched) return;
    if (!generatedPrompt.trim()) return;
    setQuery(generatedPrompt);
  }, [generatedPrompt, queryTouched]);

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

    const finalQuery = query.trim();
    if (!finalQuery) {
      setErrorText('Prompt is empty. Build or write a query before running.');
      return;
    }

    if (executionProfile === 'hpc') {
      setErrorText('HPC handoff mode is selected. Use the generated script instead of browser submission.');
      return;
    }

    const safeRecursion = Math.min(80, Math.max(10, recursionLimit));
    const safeTimeout = Math.min(1800, Math.max(30, actTimeout));

    try {
      setBusy(true);
      setErrorText('');
      const created = await apiFetch<{ run_id: string; status: RunStatus }>('/runs', {
        method: 'POST',
        body: JSON.stringify({
          query: finalQuery,
          model,
          safe_mode: safeMode,
          tool_retrieval_method: retrievalMethod,
          recursion_limit: safeRecursion,
          act_timeout: safeTimeout,
          auto_interpret_figures: autoInterpretFigures
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

  const copyText = async (text: string, successMessage: string): Promise<void> => {
    if (!navigator.clipboard) {
      setErrorText('Clipboard access is unavailable in this browser.');
      return;
    }

    try {
      await navigator.clipboard.writeText(text);
      setInfoText(successMessage);
      window.setTimeout(() => setInfoText(''), 1800);
    } catch {
      setErrorText('Failed to copy text to clipboard.');
    }
  };

  const status = activeRun?.status ?? 'queued';

  return (
    <div className="app-shell">
      <header className="panel hero reveal reveal-1">
        <div className="hero-copy">
          <p className="eyebrow">SpatialAgent Workbench</p>
          <h1>A guided interface for biologists who do not code</h1>
          <p>
            Build a structured analysis prompt, run safely when local execution is feasible, or hand off the same
            configuration to HPC batch execution.
          </p>
        </div>
        <div className="hero-mark">
          <FlaskConical size={30} />
          <span>Spatial Biology</span>
        </div>
      </header>

      {errorText ? (
        <div className="notice notice-error" role="alert">
          <AlertTriangle size={16} />
          <span>{errorText}</span>
        </div>
      ) : null}

      {infoText ? (
        <div className="notice notice-info" role="status">
          <Sparkles size={16} />
          <span>{infoText}</span>
        </div>
      ) : null}

      <main className="grid-two">
        <section className="panel reveal reveal-2">
          <div className="section-head">
            <BookOpenCheck size={18} />
            <h2>Guided Prompt Builder</h2>
          </div>

          <div className="mode-switch" role="tablist" aria-label="Builder mode">
            <button
              type="button"
              className={builderMode === 'guided' ? 'active' : ''}
              onClick={() => setBuilderMode('guided')}
            >
              Guided
            </button>
            <button
              type="button"
              className={builderMode === 'custom' ? 'active' : ''}
              onClick={() => setBuilderMode('custom')}
            >
              Custom
            </button>
          </div>

          {builderMode === 'guided' ? (
            <div className="stack">
              <label>
                Workflow goal
                <select value={workflowKey} onChange={(event) => setWorkflowKey(event.target.value)}>
                  {workflows.map((workflow) => (
                    <option value={workflow.key} key={workflow.key}>
                      {workflow.title}
                    </option>
                  ))}
                </select>
              </label>

              {selectedWorkflow ? (
                <article className="workflow-card">
                  <p className="workflow-title">{selectedWorkflow.title}</p>
                  <p>{selectedWorkflow.summary}</p>
                  <p className="muted">When to use: {selectedWorkflow.when_to_use}</p>
                </article>
              ) : null}

              <label>
                Biological question
                <textarea
                  rows={3}
                  value={scientificQuestion}
                  onChange={(event) => setScientificQuestion(event.target.value)}
                  placeholder="Example: Compare immune niches between tumor edge and core"
                />
              </label>

              <div className="row-2">
                <label>
                  Organism
                  <select value={organism} onChange={(event) => setOrganism(event.target.value)}>
                    <option value="human">Human</option>
                    <option value="mouse">Mouse</option>
                    <option value="other">Other</option>
                  </select>
                </label>

                <label>
                  Tissue / region
                  <input
                    value={tissue}
                    onChange={(event) => setTissue(event.target.value)}
                    placeholder="brain cortex, liver, lymph node..."
                  />
                </label>
              </div>

              <label>
                Dataset path or source
                <input
                  value={datasetPath}
                  onChange={(event) => setDatasetPath(event.target.value)}
                  placeholder="/data/study/sample.h5ad or dataset identifier"
                />
              </label>

              <label>
                Expected outputs
                <input
                  value={expectedOutputs}
                  onChange={(event) => setExpectedOutputs(event.target.value)}
                  placeholder="marker table, annotated UMAP, domain map, concise interpretation"
                />
              </label>

              <label>
                Constraints or preferences
                <input
                  value={constraints}
                  onChange={(event) => setConstraints(event.target.value)}
                  placeholder="Prioritize reproducibility and call out low-confidence conclusions"
                />
              </label>

              <div className="builder-preview">
                <p className="preview-label">Generated draft prompt</p>
                <pre>{generatedPrompt}</pre>
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() => {
                    setQuery(generatedPrompt);
                    setQueryTouched(true);
                  }}
                >
                  <ChevronRight size={16} /> Use this draft
                </button>
              </div>
            </div>
          ) : (
            <p className="muted">
              Custom mode leaves prompt writing fully manual. Keep your task concrete: biological question, data path,
              expected outputs, and acceptance criteria.
            </p>
          )}

          <label>
            Final prompt sent to SpatialAgent
            <textarea
              rows={9}
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setQueryTouched(true);
              }}
              placeholder="Describe the analysis in plain language..."
            />
          </label>
        </section>

        <section className="panel reveal reveal-3">
          <div className="section-head">
            <BrainCircuit size={18} />
            <h2>Execution Controls</h2>
          </div>

          <div className="mode-switch" role="tablist" aria-label="Execution profile">
            <button
              type="button"
              className={executionProfile === 'interactive' ? 'active' : ''}
              onClick={() => setExecutionProfile('interactive')}
            >
              <Laptop size={14} /> Interactive (Local)
            </button>
            <button
              type="button"
              className={executionProfile === 'hpc' ? 'active' : ''}
              onClick={() => setExecutionProfile('hpc')}
            >
              <Server size={14} /> HPC Batch Handoff
            </button>
          </div>

          <form onSubmit={onSubmit} className="stack">
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
              Tool retrieval strategy
              <select
                value={retrievalMethod}
                onChange={(event) => setRetrievalMethod(event.target.value as RetrievalMethod)}
              >
                <option value="llm">LLM (recommended)</option>
                <option value="embedding">Embedding</option>
                <option value="all">All tools</option>
              </select>
            </label>
            <p className="helper">{RETRIEVAL_HINT[retrievalMethod]}</p>

            <div className="row-2">
              <label>
                Recursion limit (10-80)
                <input
                  type="number"
                  min={10}
                  max={80}
                  value={recursionLimit}
                  onChange={(event) => setRecursionLimit(Number(event.target.value) || 35)}
                />
              </label>

              <label>
                Act timeout seconds (30-1800)
                <input
                  type="number"
                  min={30}
                  max={1800}
                  value={actTimeout}
                  onChange={(event) => setActTimeout(Number(event.target.value) || 240)}
                />
              </label>
            </div>

            <label className="check-row">
              <input
                type="checkbox"
                checked={autoInterpretFigures}
                onChange={(event) => setAutoInterpretFigures(event.target.checked)}
              />
              Auto-interpret generated figures (slower, but better for non-coder summaries)
            </label>

            <button type="button" className="toggle" onClick={() => setSafeMode((value) => !value)}>
              {safeMode ? <ShieldCheck size={16} /> : <ShieldOff size={16} />}
              Safe mode: {safeMode ? 'on' : 'off'}
            </button>

            {activeModelInfo ? (
              <article className="model-card">
                <p>
                  <strong>Provider:</strong> {activeModelInfo.provider}
                </p>
                <p>
                  <strong>Environment:</strong> {activeModelInfo.required_env.join(', ')}
                </p>
                <p>
                  <strong>Status:</strong> {activeModelInfo.ready ? 'ready' : 'missing credentials'}
                </p>
              </article>
            ) : null}

            {executionProfile === 'interactive' ? (
              <button type="submit" disabled={busy || !model || !query.trim()} className="primary-btn">
                {busy ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
                Run analysis
              </button>
            ) : (
              <article className="hpc-card">
                <p className="hpc-title">HPC Handoff Script</p>
                <p className="muted">
                  Use this on a compute node or batch job where browser access is unavailable.
                </p>
                <pre>{hpcScript}</pre>
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() => copyText(hpcScript, 'HPC script copied.')}
                >
                  <Copy size={16} /> Copy script
                </button>
                <p className="hpc-title">SLURM Template</p>
                <pre>{slurmScript}</pre>
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() => copyText(slurmScript, 'SLURM template copied.')}
                >
                  <Copy size={16} /> Copy SLURM template
                </button>
              </article>
            )}
          </form>

          <article className="safety-card">
            <div className="section-head tight">
              <ShieldAlert size={16} />
              <h3>Safe Mode Policy</h3>
            </div>
            <p className="muted">Default: {safety?.safe_mode_default ? 'enabled' : 'disabled'}</p>
            <ul>
              {(safety?.blocked_capabilities ?? []).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
            <p className="muted">Query filters: {(safety?.query_block_patterns ?? []).join(' | ') || '-'}</p>
            <ul>
              {(safety?.notes ?? []).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </article>
        </section>
      </main>

      <section className="panel reveal reveal-4">
        <div className="run-header">
          <div className="section-head">
            <Sparkles size={18} />
            <h2>Run Monitor</h2>
          </div>
          <p className={`status-pill status-${status}`}>{STATUS_LABEL[status]}</p>
        </div>

        <div className="stats-grid">
          <p>
            <strong>Created:</strong> {fmtTime(activeRun?.created_at ?? null)}
          </p>
          <p>
            <strong>Started:</strong> {fmtTime(activeRun?.started_at ?? null)}
          </p>
          <p>
            <strong>Finished:</strong> {fmtTime(activeRun?.finished_at ?? null)}
          </p>
          <p>
            <strong>Actions:</strong> {activeRun?.action_count ?? 0}
          </p>
          <p>
            <strong>Observations:</strong> {activeRun?.observation_count ?? 0}
          </p>
          <p>
            <strong>Safe mode:</strong> {activeRun?.safe_mode ? 'on' : 'off'}
          </p>
        </div>

        {activeRun?.error ? (
          <p className="run-error">
            <AlertTriangle size={14} /> {activeRun.error}
          </p>
        ) : null}

        <h3>Conclusion</h3>
        <pre className="output-box">{activeRun?.conclusion || 'No conclusion yet.'}</pre>

        <details>
          <summary>
            <TerminalSquare size={16} /> Runtime logs
          </summary>
          <pre className="output-box">{activeRun?.stdout || '(no stdout)'}</pre>
          {activeRun?.stderr ? <pre className="output-box error-text">{activeRun.stderr}</pre> : null}
        </details>
      </section>

      <section className="panel split-panel reveal reveal-5">
        <div>
          <div className="section-head">
            <TerminalSquare size={18} />
            <h2>Message Timeline</h2>
          </div>
          <div className="timeline">
            {(activeRun?.messages ?? []).map((message, index) => (
              <article key={`${message.role}-${index}`} className="timeline-item">
                <p className={`role-tag role-${message.role}`}>{message.role}</p>
                <pre>{message.content}</pre>
              </article>
            ))}
          </div>
        </div>

        <div className="stack">
          <div>
            <div className="section-head">
              <History size={18} />
              <h2>Run History</h2>
            </div>
            <div className="history-list">
              {runs.map((run) => (
                <button
                  key={run.run_id}
                  type="button"
                  className={`history-item ${run.run_id === activeRunId ? 'active' : ''}`}
                  onClick={() => setActiveRunId(run.run_id)}
                >
                  <span>{run.query_preview}</span>
                  <span className={`status-dot status-${run.status}`}>{STATUS_LABEL[run.status]}</span>
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="section-head">
              <FileArchive size={18} />
              <h2>Artifacts</h2>
            </div>
            <ul className="artifact-list">
              {artifacts.length === 0 ? (
                <li className="muted">No files yet.</li>
              ) : (
                artifacts.map((artifact) => (
                  <li key={artifact.path}>
                    <a href={artifactUrl(activeRunId, artifact.path)} target="_blank" rel="noreferrer">
                      {artifact.path}
                    </a>
                    <small>{formatSizeKB(artifact.size_bytes)}</small>
                  </li>
                ))
              )}
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
