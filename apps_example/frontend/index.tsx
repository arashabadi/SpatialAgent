import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import bundledIconPng from './icon/icon.png';
import {
  Layers,
  Github,
  UploadCloud,
  Settings,
  Play,
  RotateCcw,
  Loader2,
  Sparkles,
  AlertCircle,
  BrainCircuit,
  Cpu,
  Download,
  Diamond,
  Zap,
  ChevronRight,
  Terminal,
  TrendingDown,
  Target,
  Info,
  CheckCircle2,
  X,
  FileImage,
  FileText,
  FileSpreadsheet,
  Image,
  Package,
  BookOpen
} from 'lucide-react';
import WorkflowSchematic from './Workflow';

declare global {
  interface Window {
    Plotly: any;
  }
}

const API_URL = "/api";
const APP_VERSION = "0.4.4";
const APP_ICON_URL = "/static/icon.png";
const DEFAULT_ISSUE_REPORT_EMAIL = "abaghera@uab.edu";
const CLUSTER_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#795548'];

const colorForCluster = (cluster: number | string) => {
  const numeric = Number(cluster);
  const safe = Number.isFinite(numeric) ? Math.abs(Math.trunc(numeric)) : 0;
  return CLUSTER_COLORS[safe % CLUSTER_COLORS.length];
};

type ClusterChatMessage = {
  role: 'user' | 'assistant' | 'system';
  text: string;
  ts: number;
  literature?: Array<{
    pmid: string;
    title: string;
    journal?: string;
    pubdate?: string;
    url: string;
  }>;
};

type MountainPathway = {
  idx: number;
  pathway: string;
  cluster: number;
  module: string;
  nes: number;
  p_value: number | null;
  adj_p_value: number | null;
  genes: string[];
};

type MountainDataPayload = {
  status?: string;
  has_data: boolean;
  pathways: MountainPathway[];
  ranked_genes: Array<{ gene: string; logfc: number }>;
};

type MountainStudyContext = {
  disease: string;
  tissue: string;
  organism: string;
  technology: string;
  cohort: string;
  notes: string;
};

type GeneMetadataSource = {
  label: string;
  note: string;
  url?: string;
};

type MountainGeneMetadata = {
  fullName: string;
  functionText: string;
  sources: GeneMetadataSource[];
};

type TrailPaper = {
  pmid: string;
  title: string;
  journal?: string;
  year?: string;
  doi?: string;
  pmcid?: string;
  url?: string;
  doi_url?: string;
  abstract?: string;
  selection_reason?: string;
  selection_source?: string;
  source_tags?: string[];
};

type TrailNamingClue = {
  namespace?: string;
  prefix?: string;
  is_msigdb_go?: boolean;
  go_id?: string;
  go_term_name?: string;
  go_definition?: string;
  definition_references?: string[];
  msigdb_card_url?: string;
  quickgo_url?: string;
  amigo_term_url?: string;
  go_ref_ids?: string[];
  go_ref_urls?: string[];
  reference_pmids?: string[];
  reference_dois?: string[];
  reference_urls?: string[];
  brief_description?: string;
  amigo_reference_pmids?: string[];
  go_ref_reference_pmids?: string[];
};

type TrailSearchSpec = {
  local_search?: string;
  local_search_notes?: string[];
  llm_search?: string;
  llm_no_match?: boolean;
  llm_confidence?: string;
  llm_model_tier?: string;
};

type ProcessingRunMode = 'manual' | 'auto';
type ProcessingStepStatus = 'pending' | 'running' | 'done' | 'error' | 'skipped';

type ProcessingStep = {
  id: number;
  key: string;
  label: string;
  status: ProcessingStepStatus;
  note: string;
  updatedAt: number;
};

type TrailTalkStep = {
  id: number;
  key: 'prepare' | 'configure' | 'local' | 'llm' | 'finalize';
  label: string;
  detail: string;
  status: ProcessingStepStatus;
  updatedAt: number;
};

type QuickExportPromptType = 'html' | 'json';

type MountainExportFormat = 'png' | 'jpeg' | 'webp' | 'svg' | 'tiff';

type IssueReportInfo = {
  email: string;
  subject: string;
  issueSummary: string;
};

const defaultModuleChatMessage = (): ClusterChatMessage => ({
  role: 'system',
  text: '',
  ts: Date.now()
});

const extractApiErrorMessage = (data: any, fallback: string): string => {
  if (!data) return fallback;
  if (typeof data === 'string') return data;
  if (typeof data.message === 'string') return data.message;
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) {
    const parts = data.detail.map((d: any) => {
      if (!d) return null;
      if (typeof d === 'string') return d;
      if (typeof d.msg === 'string') return d.msg;
      return null;
    }).filter(Boolean);
    return parts.length > 0 ? parts.join(' | ') : fallback;
  }
  if (data.detail && typeof data.detail === 'object') {
    try {
      return JSON.stringify(data.detail);
    } catch {
      return fallback;
    }
  }
  return fallback;
};

const buildProcessingWorkflow = (mode: ProcessingRunMode, includeAgentStep: boolean): ProcessingStep[] => {
  const baseSteps = mode === 'auto'
    ? [
      { key: 'validate_session', label: 'Validate Session + Inputs', note: 'Checking session, columns, and API prerequisites.' },
      { key: 'auto_k_metrics', label: 'Auto K-Metric Sweep', note: 'Computing elbow + silhouette metrics across the configured k-range.' },
      { key: 'auto_k_decision', label: 'AI K Decision', note: 'Running AI-assisted k selection from computed metrics.' },
      { key: 'auto_module_mapping', label: 'Auto Module Mapping', note: 'Applying module mapping with the selected k and preparing cluster stats.' },
      { key: 'mountain_index', label: 'Build Mountain Explorer Data', note: 'Loading pathway ranks and module-linked evidence payload.' },
      { key: 'agent_annotation', label: 'Queue Agent Annotation', note: 'Submitting concise module annotation task.' },
      { key: 'finalize_dashboard', label: 'Finalize Dashboard', note: 'Switching UI from processing to results dashboard.' }
    ]
    : [
      { key: 'validate_session', label: 'Validate Session + Inputs', note: 'Checking uploaded inputs and selected analysis columns.' },
      { key: 'module_mapping', label: 'Run Module Mapping', note: 'Computing pathway similarity and clustering into modules.' },
      { key: 'mountain_index', label: 'Build Mountain Explorer Data', note: 'Preparing pathway-level ranking and evidence structures.' },
      { key: 'agent_annotation', label: 'Queue Agent Annotation', note: 'Submitting concise module annotation task.' },
      { key: 'finalize_dashboard', label: 'Finalize Dashboard', note: 'Switching UI from processing to results dashboard.' }
    ];
  const now = Date.now();
  return baseSteps.map((step, idx) => {
    const isAgentStep = step.key === 'agent_annotation';
    const status: ProcessingStepStatus = isAgentStep && !includeAgentStep
      ? 'skipped'
      : idx === 0
        ? 'running'
        : 'pending';
    const note = isAgentStep && !includeAgentStep
      ? 'Skipped: agent annotation disabled or API key missing.'
      : step.note;
    return {
      id: idx + 1,
      key: step.key,
      label: step.label,
      status,
      note,
      updatedAt: now
    };
  });
};

const buildTrailTalkWorkflow = (
  pathway: string,
  useNamingClue: boolean,
  goOnly: boolean,
  turboEnabled: boolean
): TrailTalkStep[] => {
  const now = Date.now();
  const normalizedPathway = String(pathway || '').replace(/\s+/g, ' ').trim();
  const pathwayLabel = normalizedPathway.length > 72
    ? `${normalizedPathway.slice(0, 71)}...`
    : (normalizedPathway || 'selected pathway');
  const configDetail = `${useNamingClue ? 'Naming Clues ON' : 'Naming Clues OFF'} | ${goOnly ? 'Mode: Origin' : 'Mode: Origin + Context'} | LLM Tier: ${turboEnabled ? 'Turbo' : 'Standard'}`;
  return [
    {
      id: 1,
      key: 'prepare',
      label: 'Prepare Pathway Context',
      detail: `Capturing pathway: ${pathwayLabel}.`,
      status: 'running',
      updatedAt: now
    },
    {
      id: 2,
      key: 'configure',
      label: 'Configure Retrieval Strategy',
      detail: configDetail,
      status: 'pending',
      updatedAt: now
    },
    {
      id: 3,
      key: 'local',
      label: 'Local Deterministic Retrieval',
      detail: 'Running exact-title and metadata-guided local candidate collection.',
      status: 'pending',
      updatedAt: now
    },
    {
      id: 4,
      key: 'llm',
      label: 'LLM Validation',
      detail: 'Validating origin relevance only within retrieved candidate papers.',
      status: 'pending',
      updatedAt: now
    },
    {
      id: 5,
      key: 'finalize',
      label: 'Finalize Evidence Panel',
      detail: 'Rendering concise message, clues, and evidence list.',
      status: 'pending',
      updatedAt: now
    }
  ];
};

const processingStatusLabel = (status: ProcessingStepStatus): string => {
  if (status === 'pending') return 'pending';
  if (status === 'running') return 'running';
  if (status === 'done') return 'done';
  if (status === 'error') return 'error';
  return 'skipped';
};

const processingStatusClasses = (status: ProcessingStepStatus): string => {
  if (status === 'done') return 'text-emerald-300 border-emerald-500/40 bg-emerald-950/40';
  if (status === 'running') return 'text-cyan-200 border-cyan-500/50 bg-cyan-950/40';
  if (status === 'error') return 'text-red-200 border-red-500/50 bg-red-950/40';
  if (status === 'skipped') return 'text-slate-300 border-slate-500/40 bg-slate-800/40';
  return 'text-slate-400 border-slate-700/60 bg-slate-900/50';
};

const escapeForTsLiteral = (value: string): string => String(value || '').replace(/\\/g, '\\\\').replace(/"/g, '\\"');

const compactWorkflowNote = (value: any, maxLen = 180): string => {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (!text) return '';
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 1)}…`;
};

const withIconFallback = (event: React.SyntheticEvent<HTMLImageElement>) => {
  const img = event.currentTarget;
  if (img.dataset.iconFallbackApplied === '1') return;
  img.dataset.iconFallbackApplied = '1';
  img.src = bundledIconPng;
};

const ProcessingStatusIcon = ({ status }: { status: ProcessingStepStatus }) => {
  if (status === 'done') return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
  if (status === 'running') return <Loader2 className="w-4 h-4 text-cyan-300 animate-spin" />;
  if (status === 'error') return <AlertCircle className="w-4 h-4 text-red-300" />;
  if (status === 'skipped') return <Terminal className="w-4 h-4 text-slate-400" />;
  return <Terminal className="w-4 h-4 text-slate-500" />;
};

const PROCESSING_MASCOT_GEM_X = [52, 84, 116, 148, 180, 212];

const MascotGem = ({
  size = 16,
  className = '',
  style,
  glow = true
}: {
  size?: number;
  className?: string;
  style?: React.CSSProperties;
  glow?: boolean;
}) => (
  <span
    className={`inline-block ${className}`}
    style={{ width: size, height: size, ...style }}
  >
    <svg
      viewBox="0 0 24 24"
      className={`h-full w-full ${glow ? 'drop-shadow-[0_0_8px_rgba(103,232,249,0.9)]' : ''}`}
      fill="none"
      aria-hidden
    >
      <polygon points="12,1.5 21.2,8 16,20.4 8,20.4 2.8,8" fill="#66dff6" />
      <polygon points="12,2.5 19.4,7.5 12,11.1 4.6,7.5" fill="#d5f8ff" />
      <polygon points="12,11.1 4.6,7.5 8,20.2" fill="#45bdd5" />
      <polygon points="12,11.1 19.4,7.5 16,20.2" fill="#299ab2" />
      <polygon points="12,11.1 16,20.2 8,20.2" fill="#7cecff" />
      <polygon points="10.2,5.8 12.8,4.2 13.6,6.8 11.1,8.1" fill="#ffffff" opacity="0.9" />
      <polygon points="12,1.5 21.2,8 16,20.4 8,20.4 2.8,8" stroke="#b5f7ff" strokeWidth="0.9" />
    </svg>
  </span>
);

const ProcessingMascot = () => {
  const [tick, setTick] = useState(0);
  const [collected, setCollected] = useState<boolean[]>(
    () => new Array(PROCESSING_MASCOT_GEM_X.length).fill(false)
  );
  const [heldGemIndex, setHeldGemIndex] = useState<number | null>(null);
  const [bagCount, setBagCount] = useState(0);
  const [bagPulse, setBagPulse] = useState(0);
  const [throws, setThrows] = useState<Array<{
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    rot: number;
    vr: number;
    age: number;
    ttl: number;
  }>>([]);

  const throwIdRef = useRef(0);
  const pickLatchRef = useRef('');
  const throwLatchRef = useRef('');
  const resetTimerRef = useRef<number | null>(null);

  useEffect(() => {
    const id = window.setInterval(() => setTick((prev) => prev + 1), 90);
    return () => window.clearInterval(id);
  }, []);

  const totalGems = PROCESSING_MASCOT_GEM_X.length;
  const segment = 30;
  const routeLength = totalGems * segment;
  const routeTick = tick % routeLength;
  const targetIndex = Math.min(totalGems - 1, Math.floor(routeTick / segment));
  const localT = (routeTick % segment) / (segment - 1);
  const easedT = 0.5 - 0.5 * Math.cos(Math.PI * localT);

  const startX = targetIndex === 0 ? 24 : PROCESSING_MASCOT_GEM_X[targetIndex - 1];
  const endX = PROCESSING_MASCOT_GEM_X[targetIndex];
  const runnerX = startX + (endX - startX) * easedT;
  const walkBob = Math.sin(localT * Math.PI * 4) * 1.15;
  const crouch = localT > 0.65 && localT < 0.8 ? 2.8 : 0;
  const runnerY = walkBob + crouch;
  const bagX = 248;
  const bagY = 134;
  const canPickCurrent = !collected[targetIndex] && heldGemIndex === null;
  const isPickFrame = localT >= 0.68 && localT <= 0.78 && canPickCurrent;
  const isThrowFrame = heldGemIndex === targetIndex && (localT >= 0.82 && localT <= 0.93);
  const armAngle = isPickFrame ? 56 : isThrowFrame ? -46 : heldGemIndex === targetIndex ? -16 : 18;
  const legOffset = tick % 2 === 0 ? 0 : 1.5;
  const allPicked = collected.every(Boolean);
  const animationComplete = allPicked && heldGemIndex === null && throws.length === 0;
  const bagScale = 1 + bagPulse * 0.07;
  const bagTilt = bagPulse * 3;

  useEffect(() => {
    let bagHits = 0;
    setThrows((prev) => prev.flatMap((item) => {
      const x = item.x + item.vx;
      const y = item.y + item.vy;
      const vy = item.vy + 0.24;
      const age = item.age + 1;
      const rot = item.rot + item.vr;
      const distToBag = Math.hypot(x - bagX, y - bagY);
      const reachedBag = distToBag < 13 && age > 4;
      if (reachedBag) {
        bagHits += 1;
        return [];
      }
      if (age > item.ttl || y > 188) {
        return [];
      }
      return [{ ...item, x, y, vy, age, rot }];
    }));
    if (bagHits > 0) {
      setBagCount((prev) => Math.min(totalGems, prev + bagHits));
      setBagPulse(1);
    } else {
      setBagPulse((prev) => Math.max(0, prev - 0.11));
    }
  }, [tick, bagX, bagY, totalGems]);

  useEffect(() => {
    if (!isPickFrame) {
      pickLatchRef.current = '';
      return;
    }
    const latchKey = `${targetIndex}:${Math.floor(routeTick / segment)}`;
    if (pickLatchRef.current === latchKey) return;
    pickLatchRef.current = latchKey;

    setCollected((prev) => {
      if (prev[targetIndex]) return prev;
      const next = [...prev];
      next[targetIndex] = true;
      return next;
    });
    setHeldGemIndex(targetIndex);
  }, [isPickFrame, routeTick, segment, targetIndex]);

  useEffect(() => {
    if (!isThrowFrame) {
      throwLatchRef.current = '';
      return;
    }
    const latchKey = `${targetIndex}:${Math.floor(routeTick / segment)}`;
    if (throwLatchRef.current === latchKey) return;
    throwLatchRef.current = latchKey;

    const handX = 4 + runnerX + 36;
    const handY = 120 + runnerY + 31;
    const dx = bagX - handX;
    const flightFrames = Math.max(16, Math.min(26, Math.round(Math.abs(dx) / 7) + 14));
    const vx = dx / flightFrames;
    const vy = -3.5 - Math.abs(dx) / 95;

    setThrows((prev) => [
      ...prev,
      {
        id: ++throwIdRef.current,
        x: handX,
        y: handY,
        vx,
        vy,
        rot: 0,
        vr: 16 + Math.random() * 8,
        age: 0,
        ttl: flightFrames + 8
      }
    ]);
    setHeldGemIndex(null);
  }, [isThrowFrame, routeTick, segment, targetIndex, runnerX, runnerY, bagX, bagY]);

  useEffect(() => {
    if (!animationComplete) return;
    if (resetTimerRef.current) window.clearTimeout(resetTimerRef.current);
    resetTimerRef.current = window.setTimeout(() => {
      setCollected(new Array(totalGems).fill(false));
      setThrows([]);
      setHeldGemIndex(null);
      setBagCount(0);
      setBagPulse(0);
    }, 1500);
  }, [animationComplete, totalGems]);

  useEffect(() => {
    return () => {
      if (resetTimerRef.current) window.clearTimeout(resetTimerRef.current);
    };
  }, []);

  return (
    <div
      className="relative h-48 w-72 overflow-hidden rounded-[34px] border border-cyan-500/35 bg-gradient-to-b from-slate-900/75 via-slate-950/85 to-slate-950 shadow-[0_0_45px_rgba(14,116,144,0.35)] select-none"
      aria-hidden
    >
      <div className="absolute -left-8 -top-10 h-24 w-24 rounded-full bg-cyan-500/20 blur-3xl animate-pulse"></div>
      <div className="absolute right-2 top-5 h-20 w-20 rounded-full bg-indigo-500/20 blur-3xl"></div>
      <div className="absolute bottom-4 left-0 right-0 h-[6px] bg-gradient-to-r from-cyan-500/0 via-cyan-400/85 to-cyan-500/0"></div>
      <div className="absolute bottom-5 left-0 right-0 h-11 rounded-t-[40px] bg-gradient-to-t from-cyan-950/70 to-cyan-900/10"></div>

      {PROCESSING_MASCOT_GEM_X.map((x, idx) => (
        !collected[idx] ? (
          <React.Fragment key={`ground-gem-${idx}`}>
            <span
              className="absolute h-1.5 w-4 rounded-full bg-slate-950/70"
              style={{ left: `${x + 1}px`, top: `${158 + (idx % 2)}px` }}
            ></span>
            <MascotGem
              size={16}
              className="absolute"
              style={{
                left: `${x}px`,
                top: `${146 + (idx % 2 === 0 ? -1 : 1)}px`,
                transform: `rotate(${idx % 2 === 0 ? 7 : -8}deg) scale(${0.94 + 0.08 * (0.5 + 0.5 * Math.sin((tick + idx * 5) / 7))})`
              }}
            />
          </React.Fragment>
        ) : null
      ))}

      {throws.map((item) => {
        return (
          <MascotGem
            key={`flying-gem-${item.id}`}
            size={14}
            className="absolute"
            style={{
              left: `${item.x}px`,
              top: `${item.y}px`,
              transform: `rotate(${item.rot}deg)`,
              opacity: 1 - Math.max(0, item.age - item.ttl * 0.82) / (item.ttl * 0.18)
            }}
          />
        );
      })}

      <div
        className="absolute right-3 bottom-8 transition-transform duration-100"
        style={{ transform: `scale(${bagScale}) rotate(${bagTilt}deg)` }}
      >
        <div className="relative h-14 w-14">
          <div className="absolute bottom-0 h-11 w-14 rounded-[12px] border border-amber-300/55 bg-gradient-to-b from-amber-500/65 to-amber-800/70"></div>
          <div className="absolute top-1 left-1 right-1 h-3 rounded-full border border-amber-100/55 bg-slate-950/60"></div>
          <div className="absolute bottom-1.5 left-2 right-2 h-8">
            {Array.from({ length: Math.min(bagCount, 6) }).map((_, idx) => (
              <MascotGem
                key={`bag-gem-${idx}`}
                size={10}
                className="absolute"
                style={{
                  left: `${(idx % 3) * 10 + (idx > 2 ? 2 : 0)}px`,
                  top: `${idx > 2 ? 8 : 2}px`,
                  transform: `rotate(${idx % 2 === 0 ? 10 : -12}deg)`,
                  opacity: 0.95
                }}
                glow={false}
              />
            ))}
          </div>
        </div>
      </div>

      <div
        className="absolute bottom-8 left-4 transition-transform duration-100 ease-linear"
        style={{ transform: `translate(${runnerX}px, ${runnerY}px)` }}
      >
        <div className="relative h-16 w-12">
          <div className="absolute left-2 top-0 h-4 w-8 rounded-t-full rounded-b-sm border border-rose-300/80 bg-rose-500"></div>
          <div className="absolute left-3 top-4 h-7 w-6 rounded-[10px] border border-amber-100/40 bg-amber-100"></div>
          <div className="absolute left-4 top-6 flex items-center gap-1">
            <span className="h-1 w-1 rounded-full bg-slate-800"></span>
            <span className="h-1 w-1 rounded-full bg-slate-800"></span>
          </div>
          <div className="absolute left-3 top-8 h-1.5 w-6 rounded-full bg-slate-800"></div>
          <div className="absolute left-1 top-10 h-6 w-10 rounded-md border border-blue-300/40 bg-blue-500"></div>
          <div className="absolute left-3 top-9 h-5 w-1 rounded bg-blue-300/90"></div>
          <div className="absolute left-8 top-9 h-5 w-1 rounded bg-blue-300/90"></div>
          <div className="absolute -right-1 top-8 h-6 w-4 rounded-md border border-amber-300/50 bg-amber-700/75"></div>
          <div className="absolute bottom-0 left-1 h-2.5 w-3 rounded bg-slate-800 transition-transform duration-75" style={{ transform: `translateY(${legOffset}px)` }}></div>
          <div className="absolute bottom-0 right-1 h-2.5 w-3 rounded bg-slate-800 transition-transform duration-75" style={{ transform: `translateY(${1.5 - legOffset}px)` }}></div>
          <div
            className="absolute -right-1 top-10 h-1 w-6 rounded bg-amber-300/90 transition-transform duration-75"
            style={{ transform: `rotate(${armAngle}deg)` }}
          ></div>
          <div
            className={`absolute -right-3 top-8 h-3 w-3 rounded-sm border border-cyan-100/70 bg-cyan-200 transition-transform duration-75 ${isPickFrame ? 'shadow-[0_0_10px_rgba(165,243,252,0.9)]' : ''}`}
            style={{ transform: `rotate(${armAngle}deg)` }}
          ></div>
          {heldGemIndex === targetIndex && (
            <MascotGem
              size={10}
              className="absolute -right-4 top-6"
              style={{ transform: `rotate(${armAngle * 0.7}deg)` }}
            />
          )}
        </div>
      </div>

      {animationComplete && (
        <Sparkles className="absolute right-8 top-8 h-5 w-5 text-cyan-200 animate-ping" />
      )}
    </div>
  );
};

// --- PLOTLY COMPONENT (Dark Mode + Auto-Rotation) ---
const PlotlyGraph = ({
  data,
  layout,
  onPointClick
}: {
  data: any,
  layout: any,
  onPointClick?: (point: any) => void
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const angleRef = useRef(0);
  const [isRotating, setIsRotating] = useState(true);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    if (window.Plotly && containerRef.current && data) {
      const normalizedData = Array.isArray(data)
        ? data.map((trace: any) => {
            if (String(trace?.type || '').toLowerCase() !== 'scatter3d') return trace;
            const hasCustomData = Array.isArray(trace?.customdata) && trace.customdata.length > 0;
            const moduleLine = hasCustomData ? 'Module: %{customdata[0]}<br>' : '';
            return {
              ...trace,
              hovertemplate:
                '<b>%{hovertext}</b><br>' +
                moduleLine +
                'Dim1: %{x:.3f}<br>' +
                'Dim2: %{y:.3f}<br>' +
                'Dim3: %{z:.3f}<extra></extra>'
            };
          })
        : data;
      const defaultLegend = {
        orientation: 'v',
        x: 0.01,
        y: 0.98,
        xanchor: 'left' as const,
        yanchor: 'top' as const,
        bgcolor: 'rgba(15,23,42,0.6)',
        bordercolor: '#334155',
        borderwidth: 1,
        font: { color: '#cbd5e1', size: 11 }
      };
      const darkLayout = {
        ...layout,
        autosize: true,
        height: 520,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        hovermode: 'closest',
        hoverlabel: {
          ...layout.hoverlabel,
          align: 'left',
          namelength: -1,
          bgcolor: 'rgba(15,23,42,0.98)',
          bordercolor: '#334155',
          font: {
            ...(layout.hoverlabel?.font || {}),
            color: '#e2e8f0',
            size: 12
          }
        },
        margin: { l: 8, r: 22, b: 8, t: 44 },
        legend: layout.legend ? { ...defaultLegend, ...layout.legend } : defaultLegend,
        scene: {
          ...layout.scene,
          xaxis: { ...layout.scene.xaxis, gridcolor: '#334155', zerolinecolor: '#475569', showbackground: false },
          yaxis: { ...layout.scene.yaxis, gridcolor: '#334155', zerolinecolor: '#475569', showbackground: false },
          zaxis: { ...layout.scene.zaxis, gridcolor: '#334155', zerolinecolor: '#475569', showbackground: false },
          aspectmode: 'cube',
          bgcolor: 'rgba(0,0,0,0)',
          camera: {
            eye: { x: 1.35, y: 1.35, z: 1.0 },
            center: { x: 0, y: 0, z: 0 },
            up: { x: 0, y: 0, z: 1 }
          }
        }
      };
      const config = { responsive: true, displaylogo: false, scrollZoom: false, doubleClick: 'reset' as const };
      window.Plotly.react(containerRef.current, normalizedData, darkLayout, config).then(() => {
        if (!containerRef.current || !window.Plotly) return;
        try {
          window.Plotly.relayout(containerRef.current, {
            'scene.aspectmode': 'cube',
            'scene.camera.center': { x: 0, y: 0, z: 0 },
            'scene.camera.up': { x: 0, y: 0, z: 1 },
            'scene.camera.projection.type': 'perspective'
          });
          window.Plotly.Plots.resize(containerRef.current);
        } catch {
          // Keep non-blocking if relayout fails on a specific renderer/browser.
        }
      });
    }
  }, [data, layout]);

  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current || !window.Plotly) return;
      try {
        window.Plotly.Plots.resize(containerRef.current);
      } catch {
        // No-op
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (!containerRef.current || !window.Plotly || !onPointClick) return;
    const plotNode: any = containerRef.current;
    const handleClick = (evt: any) => {
      const point = Array.isArray(evt?.points) && evt.points.length > 0 ? evt.points[0] : null;
      if (point) onPointClick(point);
    };
    if (typeof plotNode.on === 'function') {
      plotNode.on('plotly_click', handleClick);
    }
    return () => {
      if (typeof plotNode.removeAllListeners === 'function') {
        plotNode.removeAllListeners('plotly_click');
      }
    };
  }, [onPointClick, data]);

  // Keep page scroll responsive while pointer is over the Plotly canvas.
  useEffect(() => {
    if (!containerRef.current) return;
    const wheelHandler = (event: WheelEvent) => {
      window.scrollBy({ top: event.deltaY, left: 0, behavior: 'auto' });
    };
    const node = containerRef.current;
    node.addEventListener('wheel', wheelHandler, { passive: true });
    return () => {
      node.removeEventListener('wheel', wheelHandler);
    };
  }, []);

  // Auto-rotation effect
  useEffect(() => {
    const animate = () => {
      if (!isRotating || !containerRef.current || !window.Plotly) return;

      angleRef.current += 0.15;  // Slower rotation
      const radians = angleRef.current * Math.PI / 180;

      window.Plotly.relayout(containerRef.current, {
        'scene.camera': {
          eye: {
            x: 1.8 * Math.cos(radians),
            y: 1.8 * Math.sin(radians),
            z: 0.8
          }
        }
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    if (isRotating) {
      animationRef.current = requestAnimationFrame(animate);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRotating]);

  return (
    <div className="relative w-full h-[460px] bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl overflow-visible">
      <div ref={containerRef} className="w-full h-full" />
      {/* Rotation Controls */}
      <div className="absolute bottom-4 left-4 flex gap-2">
        <button
          onClick={() => setIsRotating(!isRotating)}
          className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${isRotating
            ? 'bg-cyan-600 border-cyan-500 text-white'
            : 'bg-slate-800/80 border-slate-700 text-slate-400 hover:border-cyan-500/50'
            }`}
        >
          {isRotating ? '⏸ Pause' : '▶ Rotate'}
        </button>
        <button
          onClick={() => {
            angleRef.current = 0;
            if (containerRef.current && window.Plotly) {
              window.Plotly.relayout(containerRef.current, {
                'scene.camera': {
                  eye: { x: 1.35, y: 1.35, z: 1.0 },
                  center: { x: 0, y: 0, z: 0 },
                  up: { x: 0, y: 0, z: 1 }
                }
              });
            }
          }}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800/80 border border-slate-700 text-slate-400 hover:border-cyan-500/50 transition-all"
        >
          ↺ Reset
        </button>
      </div>
    </div>
  );
};

const App = () => {
  const [view, setView] = useState<'landing' | 'app'>('landing');
  const [step, setStep] = useState<'upload' | 'analyze' | 'elbow' | 'processing' | 'dashboard'>('upload');
  const [processingMode, setProcessingMode] = useState<ProcessingRunMode>('manual');
  const [processingStartedAt, setProcessingStartedAt] = useState<number | null>(null);
  const [processingElapsedSec, setProcessingElapsedSec] = useState(0);
  const [processingWorkflow, setProcessingWorkflow] = useState<ProcessingStep[]>([]);

  // Data
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pathFileName, setPathFileName] = useState<string | null>(null);
  const [pathFile, setPathFile] = useState<File | null>(null); // New state
  const [pathStatus, setPathStatus] = useState<'none' | 'selected' | 'uploaded' | 'error'>('none'); // New state
  const [columns, setColumns] = useState<string[]>([]);
  const [degColumns, setDegColumns] = useState<string[]>([]); // New state
  const [previewData, setPreviewData] = useState<any[]>([]);

  // Config
  const [pathCol, setPathCol] = useState('');
  const [geneCol, setGeneCol] = useState('');
  const [scoreCol, setScoreCol] = useState('null');
  const [pathPadjCol, setPathPadjCol] = useState('null');
  const [pathSignificance, setPathSignificance] = useState<{
    checked: boolean;
    column: string | null;
    threshold: number;
    all_significant: boolean | null;
    n_total: number;
    n_kept: number;
    n_removed: number;
  } | null>(null);
  const [nClusters, setNClusters] = useState(5);
  const [nesDirection, setNesDirection] = useState<'all' | 'positive' | 'negative'>('all');
  const [reportName, setReportName] = useState('');

  // Elbow Analysis
  const [elbowData, setElbowData] = useState<any>(null);
  const [elbowLoading, setElbowLoading] = useState(false);
  const [suggestedK, setSuggestedK] = useState<number | null>(null);
  const [showMetricsTable, setShowMetricsTable] = useState(false);

  // Agent Config
  const [useAgent, setUseAgent] = useState(true);
  const [apiKey, setApiKey] = useState('');
  const [agentProvider, setAgentProvider] = useState<'openai' | 'gemini' | 'claude'>('openai');
  const [agentTurbo, setAgentTurbo] = useState(false);
  const [agentStatus, setAgentStatus] = useState<'idle' | 'running' | 'complete' | 'error'>('idle');

  // DEG Config
  const [degFile, setDegFile] = useState<File | null>(null);
  const [degStatus, setDegStatus] = useState<'none' | 'selected' | 'uploaded' | 'error'>('none');
  const [degErrorMessage, setDegErrorMessage] = useState<string | null>(null);
  const [degInfo, setDegInfo] = useState<{ n_genes: number, n_degs: number } | null>(null);
  const [degPValueCol, setDegPValueCol] = useState('null');
  const [degSignificance, setDegSignificance] = useState<{
    checked_col: string;
    threshold: number;
    all_significant: boolean;
    n_total: number;
    n_significant: number;
    n_non_significant: number;
    using_nominal_p: boolean;
    effective_col: string;
    effective_threshold: number;
  } | null>(null);
  const [degNeedsConfirmation, setDegNeedsConfirmation] = useState<{
    message: string;
    checked_col: string;
    threshold: number;
    n_total: number;
    n_significant: number;
    n_non_significant: number;
    p_value_col: string;
    n_nominal_p_below_0_05: number;
  } | null>(null);
  const [degConfig, setDegConfig] = useState({
    gene_col: 'gene',
    padj_col: 'fdr',
    lfc_col: 'log2FC',
    padj_threshold: 0.05,
    lfc_threshold: 0.25
  });

  // Auto-analyze
  const [autoAnalyzeLoading, setAutoAnalyzeLoading] = useState(false);
  const [autoKResult, setAutoKResult] = useState<any>(null);

  // Results
  const [results, setResults] = useState<any>(null);
  const [aiAnnotations, setAiAnnotations] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'3d' | '2d' | 'methods' | 'heatmap' | 'elbow' | 'nes' | 'table' | 'bubble' | 'bar' | 'manhattan' | 'mountain' | 'hexagon'>('3d');
  const [chatMessages, setChatMessages] = useState<ClusterChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [mountainData, setMountainData] = useState<MountainDataPayload | null>(null);
  const [mountainLoading, setMountainLoading] = useState(false);
  const [mountainError, setMountainError] = useState<string | null>(null);
  const [mountainPathSearch, setMountainPathSearch] = useState('');
  const [mountainModuleFilter, setMountainModuleFilter] = useState<string | null>(null);
  const [mountainModuleExpanded, setMountainModuleExpanded] = useState(false);
  const [selectedMountainPathId, setSelectedMountainPathId] = useState<number | null>(null);
  const [mountainHitInfo, setMountainHitInfo] = useState<{
    symbol: string;
    rank: number;
    logfc: string;
    fullName: string;
    functionText: string;
    sources: GeneMetadataSource[];
  } | null>(null);
  const [mountainHitExpanded, setMountainHitExpanded] = useState(false);
  const [mountainLockedGeneKey, setMountainLockedGeneKey] = useState<string | null>(null);
  const [mountainLockNotice, setMountainLockNotice] = useState('');
  const [mountainOriginLoading, setMountainOriginLoading] = useState(false);
  const [mountainOriginQuery, setMountainOriginQuery] = useState('');
  const [mountainOriginMessage, setMountainOriginMessage] = useState('');
  const [mountainOriginExact, setMountainOriginExact] = useState<boolean | null>(null);
  const [mountainOriginPapers, setMountainOriginPapers] = useState<TrailPaper[]>([]);
  const [mountainExpandedAbstracts, setMountainExpandedAbstracts] = useState<Record<string, boolean>>({});
  const [mountainOriginQueries, setMountainOriginQueries] = useState<string[]>([]);
  const [mountainNamingClue, setMountainNamingClue] = useState<TrailNamingClue | null>(null);
  const [mountainTrailSearchSpec, setMountainTrailSearchSpec] = useState<TrailSearchSpec | null>(null);
  const [trailTalkSteps, setTrailTalkSteps] = useState<TrailTalkStep[]>([]);
  const [trailTalkStartedAt, setTrailTalkStartedAt] = useState<number | null>(null);
  const [trailTalkElapsedSec, setTrailTalkElapsedSec] = useState(0);
  const [trailStatusExpanded, setTrailStatusExpanded] = useState(false);
  const [trailUseNamingClue, setTrailUseNamingClue] = useState(true);
  const [trailMsigdbGoOnly, setTrailMsigdbGoOnly] = useState(true);
  const [pendingPaperForContext, setPendingPaperForContext] = useState<TrailPaper | null>(null);
  const [pendingContextError, setPendingContextError] = useState('');
  const [mountainPaperAnalysisLoading, setMountainPaperAnalysisLoading] = useState<string | null>(null);
  const [mountainPaperAnalyses, setMountainPaperAnalyses] = useState<Record<string, {
    evidence: string;
    hypothesis: string;
    openAccessNote: string;
    nHits: number;
    nUp: number;
    nDown: number;
    moduleId: string;
    moduleName: string;
    geneHits: Array<{ gene: string; logfc: number; direction: string; module: string }>;
  }>>({});
  const [mountainExpandedAnalyses, setMountainExpandedAnalyses] = useState<Record<string, boolean>>({});
  const [mountainStudyContext, setMountainStudyContext] = useState<MountainStudyContext>({
    disease: '',
    tissue: '',
    organism: '',
    technology: '',
    cohort: '',
    notes: ''
  });
  const [replayLoading, setReplayLoading] = useState(false);
  const [aiActionLoading, setAiActionLoading] = useState(false);
  const [geneSearchQuery, setGeneSearchQuery] = useState('');
  const [copyNotice, setCopyNotice] = useState('');
  const [issueReportInfo, setIssueReportInfo] = useState<IssueReportInfo | null>(null);
  const [showIssueReportPrompt, setShowIssueReportPrompt] = useState(false);
  const [issueSummaryInput, setIssueSummaryInput] = useState('');
  const [issueReportLoading, setIssueReportLoading] = useState(false);
  const [issueReportCopyStatus, setIssueReportCopyStatus] = useState('');
  const [showRunningEs, setShowRunningEs] = useState(true);
  const [showPathwayGenes, setShowPathwayGenes] = useState(true);
  const [showRankStrip, setShowRankStrip] = useState(true);
  const [showRankStripLegend, setShowRankStripLegend] = useState(true);
  const [showMountainExportModal, setShowMountainExportModal] = useState(false);
  const [mountainExportConfig, setMountainExportConfig] = useState<{
    format: MountainExportFormat;
    width: number;
    height: number;
    quality: 'standard' | 'high' | 'ultra';
  }>({
    format: 'png',
    width: 800,
    height: 650,
    quality: 'high'
  });
  const [mountainExportGeneSearch, setMountainExportGeneSearch] = useState('');
  const [mountainExportSelectedGenes, setMountainExportSelectedGenes] = useState<string[]>([]);
  const [showResearchApiPrompt, setShowResearchApiPrompt] = useState(false);
  const [includeChatInHtml, setIncludeChatInHtml] = useState(false);
  const [includeChatInJson, setIncludeChatInJson] = useState(false);
  const [quickExportPrompt, setQuickExportPrompt] = useState<QuickExportPromptType | null>(null);

  const mountainPlotRef = useRef<HTMLDivElement | null>(null);
  const mountainSectionRef = useRef<HTMLElement | null>(null);
  const trailContextRef = useRef<HTMLDivElement | null>(null);
  const researchApiInputRef = useRef<HTMLInputElement | null>(null);
  const mountainGeneMetaCacheRef = useRef<Map<string, MountainGeneMetadata>>(new Map());
  const mountainGeneMetaInflightRef = useRef<Map<string, Promise<MountainGeneMetadata>>>(new Map());
  const mountainHitHoverSeqRef = useRef(0);
  const mountainLastHoverKeyRef = useRef('');
  const mountainHoverThrottleTsRef = useRef(0);
  const mountainLockedGeneKeyRef = useRef<string | null>(null);
  const mountainLockNoticeTimerRef = useRef<number | null>(null);
  const mountainLastPointClickTsRef = useRef(0);
  const mountainBgClickHandlerRef = useRef<((event: MouseEvent) => void) | null>(null);
  const replayManifestInputRef = useRef<HTMLInputElement | null>(null);
  const emergencyJsonInFlightRef = useRef(false);
  const emergencyJsonLastTriggerRef = useRef(0);
  const [pendingPathwayJump, setPendingPathwayJump] = useState<string | null>(null);
  const resultTabs = [
    { id: '3d', label: '3D Map' },
    { id: '2d', label: '2D Projections' },
    { id: 'methods', label: 'Methods' },
    { id: 'table', label: 'Table' }
  ] as const;
  const dashboardHeaderTitle = activeTab === '3d'
    ? '3D Map and Mountain Explorer'
    : activeTab === '2d'
      ? '2D Projections'
      : activeTab === 'methods'
        ? 'Methodology'
        : activeTab === 'table'
          ? 'Table'
          : 'Module Analysis';
  const dashboardHeaderNote = activeTab === '3d'
    ? 'Click modules for details'
    : activeTab === '2d'
      ? 'Review pairwise MDS projections'
      : activeTab === 'methods'
        ? 'Review Autopilot selection rationale'
        : 'Browse tabular module outputs';
  const showLegacyPanels = false;

  // Publication figures (generated on demand)
  const [pubFigures, setPubFigures] = useState<{ [key: string]: string }>({});

  // Export Modal
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportStatus, setExportStatus] = useState('');
  const [htmlExportLoading, setHtmlExportLoading] = useState(false);
  const [htmlExportProgress, setHtmlExportProgress] = useState(0);
  const [htmlExportStatus, setHtmlExportStatus] = useState('');
  const [xlsxExportLoading, setXlsxExportLoading] = useState(false);
  const [xlsxExportProgress, setXlsxExportProgress] = useState(0);
  const [xlsxExportStatus, setXlsxExportStatus] = useState('');
  const [jsonExportLoading, setJsonExportLoading] = useState(false);
  const [jsonExportProgress, setJsonExportProgress] = useState(0);
  const [jsonExportStatus, setJsonExportStatus] = useState('');
  const [exportConfig, setExportConfig] = useState({
    img_format: 'png',
    dpi: 300,
    include_3d: true,
    include_2d: true,
    include_elbow: true,
    include_heatmap: true,
    include_barplots: true,
    include_html: true,
    include_json: true,
    include_table: true,
    data_format: 'xlsx'
  });
  const hasSelectedPublicationFigure =
    exportConfig.include_3d ||
    exportConfig.include_2d ||
    exportConfig.include_elbow ||
    exportConfig.include_heatmap ||
    exportConfig.include_barplots;

  const moduleizeText = (value: string) =>
    String(value || '')
      .replace(/\bClusters\b/g, 'Modules')
      .replace(/\bCluster\b/g, 'Module')
      .replace(/\bclusters\b/g, 'modules')
      .replace(/\bcluster\b/g, 'module');

  const moduleizedPlotlyJson = useMemo(() => {
    const payload = results?.plotly_json;
    if (!payload || typeof payload !== 'object') return payload;
    const sanitize = (v: any): any => {
      if (typeof v === 'string') return moduleizeText(v);
      if (Array.isArray(v)) return v.map(sanitize);
      if (!v || typeof v !== 'object') return v;
      const out: any = {};
      Object.keys(v).forEach((k) => {
        out[k] = sanitize(v[k]);
      });
      return out;
    };
    try {
      return sanitize(payload);
    } catch {
      return payload;
    }
  }, [results?.plotly_json]);

  const effectiveAgentTurbo = useMemo(
    () => Boolean(agentTurbo && agentProvider === 'gemini'),
    [agentTurbo, agentProvider]
  );

  const turboModelLabel = useMemo(() => {
    if (agentProvider === 'gemini') {
      return effectiveAgentTurbo ? 'Gemini 3 Pro Preview' : 'Gemini 2.5 Flash';
    }
    if (agentProvider === 'openai') return 'Default OpenAI model';
    return 'Default Claude model';
  }, [agentProvider, effectiveAgentTurbo]);

  const beginProcessingWorkflow = (mode: ProcessingRunMode, includeAgentStep: boolean) => {
    const now = Date.now();
    setProcessingMode(mode);
    setProcessingStartedAt(now);
    setProcessingElapsedSec(0);
    setProcessingWorkflow(buildProcessingWorkflow(mode, includeAgentStep));
  };

  const updateProcessingWorkflowStep = (key: string, status: ProcessingStepStatus, note?: string) => {
    const now = Date.now();
    setProcessingWorkflow((prev) => prev.map((stepDef) => {
      if (stepDef.key !== key) return stepDef;
      return {
        ...stepDef,
        status,
        note: note && note.trim() ? note.trim() : stepDef.note,
        updatedAt: now
      };
    }));
  };

  const beginTrailTalkWorkflow = (pathway: string) => {
    const now = Date.now();
    setTrailTalkStartedAt(now);
    setTrailTalkElapsedSec(0);
    setTrailStatusExpanded(true);
    setTrailTalkSteps(buildTrailTalkWorkflow(pathway, trailUseNamingClue, trailMsigdbGoOnly, effectiveAgentTurbo));
  };

  const updateTrailTalkStep = (
    key: TrailTalkStep['key'],
    status: ProcessingStepStatus,
    detail?: string
  ) => {
    const now = Date.now();
    setTrailTalkSteps((prev) => prev.map((stepDef) => {
      if (stepDef.key !== key) return stepDef;
      return {
        ...stepDef,
        status,
        detail: detail && detail.trim() ? detail.trim() : stepDef.detail,
        updatedAt: now
      };
    }));
  };

  const trailTalkWorkflowCode = useMemo(() => {
    if (!trailTalkSteps.length) return '';
    const lines = [
      'type TrailTalkWorkflowStep = {',
      '  id: number;',
      '  step: string;',
      '  status: "pending" | "running" | "done" | "error" | "skipped";',
      '  detail: string;',
      '};',
      '',
      'const trailTalkWorkflow: TrailTalkWorkflowStep[] = ['
    ];
    trailTalkSteps.forEach((entry) => {
      lines.push(
        `  { id: ${entry.id}, step: "${escapeForTsLiteral(entry.label)}", status: "${entry.status}", detail: "${escapeForTsLiteral(entry.detail)}" },`
      );
    });
    lines.push('];');
    return lines.join('\n');
  }, [trailTalkSteps]);

  const trailTalkStatusSummary = useMemo(() => {
    let done = 0;
    let running = 0;
    let error = 0;
    let pending = 0;
    let skipped = 0;
    trailTalkSteps.forEach((stepInfo) => {
      if (stepInfo.status === 'done') done += 1;
      else if (stepInfo.status === 'running') running += 1;
      else if (stepInfo.status === 'error') error += 1;
      else if (stepInfo.status === 'skipped') skipped += 1;
      else pending += 1;
    });
    const total = trailTalkSteps.length;
    const completed = done + error + skipped;
    const progressPct = total > 0 ? Math.round((completed / total) * 100) : 0;
    return { done, running, error, pending, skipped, total, progressPct };
  }, [trailTalkSteps]);

  useEffect(() => {
    if (step !== 'processing' || !processingStartedAt) return;
    const updateElapsed = () => {
      const elapsed = Math.max(0, Math.floor((Date.now() - processingStartedAt) / 1000));
      setProcessingElapsedSec(elapsed);
    };
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [step, processingStartedAt]);

  useEffect(() => {
    if (!mountainOriginLoading || !trailTalkStartedAt) return;
    const updateElapsed = () => {
      const elapsed = Math.max(0, Math.floor((Date.now() - trailTalkStartedAt) / 1000));
      setTrailTalkElapsedSec(elapsed);
    };
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [mountainOriginLoading, trailTalkStartedAt]);

  useEffect(() => {
    if (!resultTabs.some(tab => tab.id === activeTab)) {
      setActiveTab('3d');
    }
  }, [activeTab]);

  useEffect(() => {
    if (showResearchApiPrompt && !apiKey.trim()) {
      researchApiInputRef.current?.focus();
    }
  }, [showResearchApiPrompt, apiKey]);

  const mountainModuleOptions = useMemo(() => {
    const pathways = mountainData?.pathways || [];
    const summary = new Map<string, { count: number; name: string; color: string }>();
    pathways.forEach((p) => {
      const id = String(p.cluster);
      const current = summary.get(id) || { count: 0, name: p.module || `Module ${id}`, color: colorForCluster(id) };
      summary.set(id, { ...current, count: current.count + 1, name: p.module || current.name, color: colorForCluster(id) });
    });
    return Array.from(summary.entries())
      .map(([id, meta]) => ({ id, ...meta }))
      .sort((a, b) => Number(a.id) - Number(b.id));
  }, [mountainData]);

  const visibleModuleOptions = useMemo(() => {
    const maxShown = 12;
    if (!mountainModuleExpanded) return mountainModuleOptions.slice(0, maxShown);
    return mountainModuleOptions;
  }, [mountainModuleOptions, mountainModuleExpanded]);

  const extraModuleCount = useMemo(() => {
    const maxShown = 12;
    return Math.max(mountainModuleOptions.length - maxShown, 0);
  }, [mountainModuleOptions]);

  const filteredMountainPathways = useMemo(() => {
    const pathways = mountainData?.pathways || [];
    const q = mountainPathSearch.trim().toLowerCase();
    const narrowed = mountainModuleFilter
      ? pathways.filter((p) => String(p.cluster) === String(mountainModuleFilter))
      : pathways;
    if (!q) return narrowed;
    return narrowed.filter((p) => {
      const name = String(p.pathway || '').toLowerCase();
      const moduleName = String(p.module || '').toLowerCase();
      const moduleId = `m${String(p.cluster)}`.toLowerCase();
      return name.includes(q) || moduleName.includes(q) || moduleId.includes(q);
    });
  }, [mountainData, mountainPathSearch, mountainModuleFilter]);

  useEffect(() => {
    setMountainModuleFilter(null);
    setMountainModuleExpanded(false);
  }, [mountainData?.pathways]);

  useEffect(() => {
    mountainLockedGeneKeyRef.current = mountainLockedGeneKey;
  }, [mountainLockedGeneKey]);

  useEffect(() => {
    return () => {
      if (mountainLockNoticeTimerRef.current) {
        window.clearTimeout(mountainLockNoticeTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!filteredMountainPathways.length) return;
    const stillVisible = filteredMountainPathways.some((p) => Number(p.idx) === Number(selectedMountainPathId));
    if (!stillVisible) {
      setSelectedMountainPathId(Number(filteredMountainPathways[0].idx));
      setMountainHitInfo(null);
      mountainLastHoverKeyRef.current = '';
      mountainHoverThrottleTsRef.current = 0;
      setMountainLockedGeneKey(null);
      mountainLockedGeneKeyRef.current = null;
      setMountainOriginPapers([]);
      setMountainExpandedAbstracts({});
      setMountainOriginQuery('');
      setMountainOriginMessage('');
      setMountainOriginExact(null);
      setMountainOriginQueries([]);
      setMountainNamingClue(null);
      setMountainTrailSearchSpec(null);
      setTrailTalkSteps([]);
      setTrailTalkStartedAt(null);
      setTrailTalkElapsedSec(0);
      setTrailStatusExpanded(false);
      setMountainPaperAnalyses({});
      setMountainExpandedAnalyses({});
      setMountainPaperAnalysisLoading(null);
    }
  }, [filteredMountainPathways, selectedMountainPathId]);

  const selectedMountainPathway = useMemo(() => {
    if (!selectedMountainPathId || !mountainData?.pathways) return null;
    return mountainData.pathways.find((p) => Number(p.idx) === Number(selectedMountainPathId)) || null;
  }, [mountainData, selectedMountainPathId]);

  const degGeneSets = useMemo(() => {
    const byModule = new Set<string>();
    const any = new Set<string>();
    if (!Array.isArray(results?.gene_stats)) return { byModule, any };
    results.gene_stats.forEach((row: any) => {
      const isDeg = Boolean(row?.DEG);
      if (!isDeg) return;
      const symbol = String(row?.Item || '').trim().toUpperCase();
      if (!symbol) return;
      const moduleId = String(row?.Cluster || '').trim();
      any.add(symbol);
      if (moduleId) byModule.add(`${symbol}::${moduleId}`);
    });
    return { byModule, any };
  }, [results?.gene_stats]);

  const selectedPathwayDegGeneCount = useMemo(() => {
    if (!selectedMountainPathway) return 0;
    const moduleId = String(selectedMountainPathway.cluster);
    const genes = Array.isArray(selectedMountainPathway.genes) ? selectedMountainPathway.genes : [];
    let count = 0;
    genes.forEach((geneRaw) => {
      const gene = String(geneRaw || '').trim().toUpperCase();
      if (!gene) return;
      if (degGeneSets.byModule.has(`${gene}::${moduleId}`) || degGeneSets.any.has(gene)) count += 1;
    });
    return count;
  }, [selectedMountainPathway, degGeneSets]);

  const mountainExportGeneOptions = useMemo(() => {
    if (!selectedMountainPathway) return [] as Array<{ gene: string; rank: number | null; logfc: number | null; isDeg: boolean }>;
    const rankedRows = Array.isArray(mountainData?.ranked_genes) ? mountainData.ranked_genes : [];
    const rankedMap = new Map<string, { rank: number; logfc: number }>();
    rankedRows.forEach((row, idx) => {
      const symbol = String(row?.gene || '').trim().toUpperCase();
      const logfc = Number(row?.logfc);
      if (!symbol || !Number.isFinite(logfc) || rankedMap.has(symbol)) return;
      rankedMap.set(symbol, { rank: idx + 1, logfc });
    });
    const uniqueGenes = Array.from(
      new Set((selectedMountainPathway.genes || []).map((g) => String(g || '').trim().toUpperCase()).filter(Boolean))
    );
    const moduleId = String(selectedMountainPathway.cluster);
    return uniqueGenes
      .map((gene) => {
        const ranked = rankedMap.get(gene);
        return {
          gene,
          rank: ranked ? ranked.rank : null,
          logfc: ranked ? ranked.logfc : null,
          isDeg: degGeneSets.byModule.has(`${gene}::${moduleId}`) || degGeneSets.any.has(gene)
        };
      })
      .sort((a, b) => {
        if (a.rank === null && b.rank === null) return a.gene.localeCompare(b.gene);
        if (a.rank === null) return 1;
        if (b.rank === null) return -1;
        return a.rank - b.rank;
      });
  }, [selectedMountainPathway, mountainData?.ranked_genes, degGeneSets]);

  const filteredMountainExportGenes = useMemo(() => {
    const q = mountainExportGeneSearch.trim().toUpperCase();
    if (!q) return mountainExportGeneOptions;
    return mountainExportGeneOptions.filter((g) => g.gene.includes(q));
  }, [mountainExportGeneSearch, mountainExportGeneOptions]);

  useEffect(() => {
    setMountainExportGeneSearch('');
    setMountainExportSelectedGenes([]);
  }, [selectedMountainPathway?.idx]);

  useEffect(() => {
    const allowed = new Set(
      mountainExportGeneOptions
        .filter((g) => g.rank !== null)
        .map((g) => g.gene)
    );
    setMountainExportSelectedGenes((prev) => prev.filter((gene) => allowed.has(gene)));
  }, [mountainExportGeneOptions]);

  const geneSearchUpper = useMemo(() => geneSearchQuery.trim().toUpperCase(), [geneSearchQuery]);

  const pathwayGeneMatches = useMemo(() => {
    const matchingPathways = new Set<string>();
    const matchingModules = new Set<string>();
    const pathwaysByGeneModule = new Map<string, Set<string>>();
    if (!geneSearchUpper || !Array.isArray(mountainData?.pathways)) {
      return { matchingPathways, matchingModules, pathwaysByGeneModule };
    }
    mountainData.pathways.forEach((path) => {
      const moduleId = String(path.cluster);
      const pathwayName = String(path.pathway || '');
      const genes = Array.isArray(path.genes) ? path.genes : [];
      genes.forEach((geneRaw) => {
        const symbol = String(geneRaw || '').trim().toUpperCase();
        if (!symbol || !symbol.includes(geneSearchUpper)) return;
        matchingModules.add(moduleId);
        matchingPathways.add(pathwayName);
        const key = `${symbol}::${moduleId}`;
        const current = pathwaysByGeneModule.get(key) || new Set<string>();
        current.add(pathwayName);
        pathwaysByGeneModule.set(key, current);
      });
    });
    return { matchingPathways, matchingModules, pathwaysByGeneModule };
  }, [mountainData, geneSearchUpper]);

  const geneSearchMatches = useMemo(() => {
    if (!geneSearchUpper || !Array.isArray(results?.gene_stats)) return [];
    const dedup = new Map<string, {
      gene: string;
      module: string;
      percentage: number;
      isDeg: boolean;
      pathways: string[];
    }>();
    results.gene_stats.forEach((row: any) => {
      const symbol = String(row?.Item || '').trim().toUpperCase();
      if (!symbol || !symbol.includes(geneSearchUpper)) return;
      const moduleId = String(row?.Cluster || '');
      const key = `${symbol}::${moduleId}`;
      const pathwaySet = pathwayGeneMatches.pathwaysByGeneModule.get(key) || new Set<string>();
      dedup.set(key, {
        gene: symbol,
        module: moduleId,
        percentage: Number(row?.Percentage || 0),
        isDeg: Boolean(row?.DEG),
        pathways: Array.from(pathwaySet).slice(0, 4),
      });
    });
    return Array.from(dedup.values())
      .sort((a, b) => b.percentage - a.percentage)
      .slice(0, 120);
  }, [results?.gene_stats, geneSearchUpper, pathwayGeneMatches]);

  const filteredTableRows = useMemo(() => {
    const rows = Array.isArray(results?.scatter_data) ? results.scatter_data : [];
    if (!geneSearchUpper) return rows;
    return rows.filter((row: any) => {
      const pathwayName = String(row?.[pathCol] || '');
      const moduleId = String(row?.Cluster || '');
      if (pathwayName.toUpperCase().includes(geneSearchUpper)) return true;
      if (pathwayGeneMatches.matchingPathways.has(pathwayName)) return true;
      if (pathwayGeneMatches.matchingModules.has(moduleId)) return true;
      return false;
    });
  }, [results?.scatter_data, pathCol, geneSearchUpper, pathwayGeneMatches]);

  const formatPValue = (value: number | null | undefined) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return 'NA';
    if (n <= 0) return '<1e-300';
    if (n < 1e-4) return n.toExponential(2);
    if (n < 0.01) return n.toExponential(2);
    return n.toFixed(4);
  };

  const requestResearchApiKey = (reason: 'annotation' | 'chat') => {
    setShowResearchApiPrompt(true);
    setChatMessages(prev => [
      ...prev,
      {
        role: 'system',
        text: reason === 'annotation'
          ? 'Please add an API key below in Research Chat to annotate or reannotate modules.'
          : 'Please add an API key below in Research Chat to continue assistant responses.',
        ts: Date.now()
      }
    ]);
  };

  const copyPathwayName = async (pathwayName: string) => {
    const value = String(pathwayName || '').trim();
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value);
      setCopyNotice(`Copied: ${value}`);
      window.setTimeout(() => setCopyNotice(''), 1800);
    } catch {
      setCopyNotice('Copy failed on this browser.');
      window.setTimeout(() => setCopyNotice(''), 1800);
    }
  };

  const showMountainLockToast = (message: string) => {
    if (mountainLockNoticeTimerRef.current) {
      window.clearTimeout(mountainLockNoticeTimerRef.current);
    }
    setMountainLockNotice(message);
    mountainLockNoticeTimerRef.current = window.setTimeout(() => {
      setMountainLockNotice('');
      mountainLockNoticeTimerRef.current = null;
    }, 1600);
  };

  const buildIssueReportBody = (info: IssueReportInfo) =>
    [
      'Dear GEMMAP Support Team,',
      '',
      `What happened: ${info.issueSummary || 'N/A'}`,
      'I have attached the downloaded GEMMAP report text file (.txt).',
      '',
      'Sincerely,',
      '[User]',
    ].join('\n');

  const buildIssueReportTemplate = (info: IssueReportInfo) =>
    [
      `To: ${info.email}`,
      `Subject: ${info.subject}`,
      '',
      buildIssueReportBody(info),
    ].join('\n');

  const copyIssueReportText = async (payload: string, label: string) => {
    try {
      await navigator.clipboard.writeText(payload);
      setIssueReportCopyStatus(`${label} copied.`);
      window.setTimeout(() => setIssueReportCopyStatus(''), 1800);
    } catch {
      setIssueReportCopyStatus('Copy failed on this browser.');
      window.setTimeout(() => setIssueReportCopyStatus(''), 1800);
    }
  };

  const copyIssueReportTemplate = async () => {
    if (!issueReportInfo) return;
    await copyIssueReportText(buildIssueReportTemplate(issueReportInfo), 'Email template');
  };

  const copyIssueReportEmail = async () => {
    if (!issueReportInfo) return;
    await copyIssueReportText(issueReportInfo.email, 'Email address');
  };

  const copyIssueReportSubject = async () => {
    if (!issueReportInfo) return;
    await copyIssueReportText(issueReportInfo.subject, 'Subject');
  };

  const copyIssueReportBody = async () => {
    if (!issueReportInfo) return;
    await copyIssueReportText(buildIssueReportBody(issueReportInfo), 'Body text');
  };

  const openIssueReportMailClient = () => {
    if (!issueReportInfo) return;
    const body = buildIssueReportBody(issueReportInfo);
    const mailto = `mailto:${encodeURIComponent(issueReportInfo.email)}?subject=${encodeURIComponent(issueReportInfo.subject)}&body=${encodeURIComponent(body)}`;
    window.location.href = mailto;
  };

  const restoreChatsFromReplay = (chatLogRaw: any) => {
    const moduleRestored: ClusterChatMessage[] = [];

    if (Array.isArray(chatLogRaw)) {
      chatLogRaw.forEach((entry: any, idx: number) => {
        if (!entry || typeof entry !== 'object') return;
        const userText = String(entry.user_message || '').trim();
        const assistantText = String(entry.assistant_reply || '').trim();
        const literature = Array.isArray(entry.literature) ? entry.literature : undefined;
        const parsedTs = Date.parse(String(entry.timestamp_utc || ''));
        const baseTs = Number.isFinite(parsedTs) ? parsedTs : (Date.now() + idx * 2);
        const lowerUser = userText.toLowerCase();
        const isTrailTalkEntry =
          lowerUser.startsWith('trail talk context') ||
          lowerUser.includes("analyze paper relevance for pathway");
        if (isTrailTalkEntry) return;

        if (userText) {
          moduleRestored.push({
            role: 'user',
            text: userText,
            ts: baseTs
          });
        }
        if (assistantText) {
          moduleRestored.push({
            role: 'assistant',
            text: assistantText,
            ts: baseTs + 1,
            literature
          });
        }
      });
    }

    setChatMessages(moduleRestored.length ? moduleRestored.slice(-80) : []);
  };

  const normalizePathwayName = (value: any) => {
    const raw = String(value || '')
      .replace(/<[^>]*>/g, ' ')
      .split(/\n|<br\s*\/?>|\|/i)[0]
      .replace(/[_\s]+/g, ' ')
      .replace(/[^a-zA-Z0-9 ]+/g, ' ')
      .toLowerCase()
      .trim();
    return raw;
  };

  const collectPathwayCandidatesFromPoint = (point: any) => {
    const values: string[] = [];
    const pushCandidate = (v: any) => {
      if (typeof v !== 'string') return;
      const first = String(v).split(/\n|<br\s*\/?>|\|/i)[0].trim();
      if (first.length > 2) values.push(first);
    };
    const cd = point?.customdata;
    if (Array.isArray(cd)) {
      cd.forEach(pushCandidate);
    } else {
      pushCandidate(cd);
    }
    pushCandidate(point?.text);
    pushCandidate(point?.hovertext);
    pushCandidate(point?.name);
    return Array.from(new Set(values));
  };

  const focusMountainExplorer = () => {
    setTimeout(() => {
      mountainSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 120);
  };

  const selectMountainPathwayByName = (pathwayName: string) => {
    if (!mountainData?.pathways?.length) return false;
    const target = normalizePathwayName(pathwayName);
    if (!target) return false;

    let match = mountainData.pathways.find((p) => normalizePathwayName(p.pathway) === target);
    if (!match) {
      match = mountainData.pathways.find((p) => {
        const name = normalizePathwayName(p.pathway);
        return name.includes(target) || target.includes(name);
      });
    }
    if (!match) return false;

    setMountainModuleFilter(String(match.cluster));
    setSelectedMountainPathId(Number(match.idx));
    setMountainHitInfo(null);
    mountainLastHoverKeyRef.current = '';
    mountainHoverThrottleTsRef.current = 0;
    setMountainLockedGeneKey(null);
    mountainLockedGeneKeyRef.current = null;
    setMountainOriginPapers([]);
    setMountainExpandedAbstracts({});
    setMountainOriginQuery('');
    setMountainOriginMessage('');
    setMountainOriginExact(null);
    setMountainOriginQueries([]);
    setMountainNamingClue(null);
    setMountainTrailSearchSpec(null);
    setMountainPaperAnalyses({});
    setMountainExpandedAnalyses({});
    setMountainPaperAnalysisLoading(null);
    focusMountainExplorer();
    return true;
  };

  const tryOpenPathwayFrom3DPoint = async (point: any) => {
    const directCandidates = collectPathwayCandidatesFromPoint(point);
    if (mountainData?.pathways?.length) {
      for (const candidate of directCandidates) {
        if (selectMountainPathwayByName(candidate)) return true;
      }
    }

    const x = Number(point?.x);
    const y = Number(point?.y);
    const z = Number(point?.z);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && Array.isArray(results?.scatter_data) && results.scatter_data.length > 0) {
      let best: any = null;
      let bestDist = Number.POSITIVE_INFINITY;
      for (const row of results.scatter_data) {
        const dx = Number(row?.Dim1) - x;
        const dy = Number(row?.Dim2) - y;
        const dz = Number(row?.Dim3) - z;
        if (!Number.isFinite(dx) || !Number.isFinite(dy) || !Number.isFinite(dz)) continue;
        const dist = (dx * dx) + (dy * dy) + (dz * dz);
        if (dist < bestDist) {
          bestDist = dist;
          best = row;
        }
      }
      const nearestPath = best ? String(best?.[pathCol] || '').trim() : '';
      if (nearestPath) {
        if (mountainData?.pathways?.length && selectMountainPathwayByName(nearestPath)) return true;
        directCandidates.push(nearestPath);
      }
    }

    const fallbackCandidate = directCandidates[0];
    if (fallbackCandidate && sessionId) {
      setPendingPathwayJump(fallbackCandidate);
      if (!mountainLoading) {
        await loadMountainData();
      }
    }
    return false;
  };

  const buildMountainSeries = (pathway: MountainPathway | null) => {
    if (!pathway || !mountainData?.ranked_genes?.length) return null;
    const ranked = mountainData.ranked_genes
      .map((g) => ({ gene: String(g.gene || '').toUpperCase(), logfc: Number(g.logfc) }))
      .filter((g) => g.gene && Number.isFinite(g.logfc));
    const geneSet = new Set((pathway.genes || []).map((g) => String(g).toUpperCase()));
    if (ranked.length === 0 || geneSet.size === 0) return null;

    const hitGenes = ranked.filter((g) => geneSet.has(g.gene));
    const hitWeightSum = hitGenes.reduce((acc, g) => acc + Math.abs(g.logfc || 0), 0) || Math.max(hitGenes.length, 1);
    const missPenalty = 1 / Math.max(ranked.length - hitGenes.length, 1);
    const x: number[] = [];
    const y: number[] = [];
    const hitMeta: Array<{ rank: number; gene: string; logfc: number }> = [];
    let running = 0;
    let maxEs = -Infinity;
    let minEs = Infinity;

    ranked.forEach((g, idx) => {
      if (geneSet.has(g.gene)) {
        running += Math.abs(g.logfc || 0) / hitWeightSum;
        hitMeta.push({ rank: idx + 1, gene: g.gene, logfc: g.logfc });
      } else {
        running -= missPenalty;
      }
      x.push(idx + 1);
      y.push(running);
      maxEs = Math.max(maxEs, running);
      minEs = Math.min(minEs, running);
    });

    const ymin = Math.min(...y, 0);
    const ymax = Math.max(...y, 0);
    const ypad = Math.max((ymax - ymin) * 0.12, 0.1);
    const hitY = ymin - ypad * 0.35;
    const peakEs = Math.abs(maxEs) >= Math.abs(minEs) ? maxEs : minEs;
    const rankMetric = ranked.map((g) => Number(g.logfc));
    const maxAbsLogfc = Math.max(...rankMetric.map((v) => Math.abs(v)), 1e-6);
    return { x, y, hitMeta, maxEs, minEs, ymin, ymax, ypad, hitY, peakEs, rankMetric, maxAbsLogfc };
  };

  const handle3DMapPointClick = async (point: any) => {
    if (activeTab !== '3d') return;
    await tryOpenPathwayFrom3DPoint(point);
  };

  const compactText = (value: string, maxLen: number) => {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    if (!text) return '';
    return text.length > maxLen ? `${text.slice(0, maxLen - 1)}...` : text;
  };

  const extractPrefixedLine = (text: string, prefix: string): string => {
    const escaped = prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const match = new RegExp(`^${escaped}:\\s*(.+)$`, 'im').exec(String(text || ''));
    return String(match?.[1] || '').trim();
  };

  const parseEvidenceAgentText = (text: string) => {
    const evidence = String(text || '');
    return {
      verdict: extractPrefixedLine(evidence, 'Verdict'),
      whyFound: extractPrefixedLine(evidence, 'Why found'),
      studyRelation: extractPrefixedLine(evidence, 'Study relation'),
      confidence: extractPrefixedLine(evidence, 'Confidence'),
      nextCheck: extractPrefixedLine(evidence, 'Next check')
    };
  };

  const stripHypothesisPrefix = (text: string) => {
    return String(text || '').replace(/^Hypothesis:\s*/i, '').trim();
  };

  const normalizeAnnotationSummary = (summary: any) => {
    return String(summary || '')
      .replace(/^Fast fallback annotation\.?\s*/i, '')
      .replace(/^Provisional annotation\.?\s*/i, '')
      .trim();
  };

  const buildStudyGeneFallback = (geneSymbol: string) => {
    const symbol = String(geneSymbol || '').trim().toUpperCase();
    if (!symbol || !mountainData?.pathways?.length) return null;
    const modules = new Set<string>();
    const pathways: string[] = [];
    mountainData.pathways.forEach((path) => {
      const genes = Array.isArray(path.genes) ? path.genes : [];
      const found = genes.some((g) => String(g || '').trim().toUpperCase() === symbol);
      if (!found) return;
      modules.add(`M${path.cluster}`);
      if (pathways.length < 4 && path.pathway) pathways.push(String(path.pathway));
    });
    if (modules.size === 0) return null;
    return `Observed in this study across ${modules.size} module(s) (${Array.from(modules).join(', ')}) and pathways such as ${pathways.join(', ')}.`;
  };

  const fetchGeneMetadata = async (geneSymbol: string): Promise<MountainGeneMetadata> => {
    const symbol = String(geneSymbol || '').trim().toUpperCase();
    if (!symbol) {
      return {
        fullName: 'Name unavailable',
        functionText: 'Function description unavailable',
        sources: [{ label: 'GEMMAP local fallback', note: 'Gene symbol was empty; no metadata lookup was possible.' }]
      };
    }

    const cached = mountainGeneMetaCacheRef.current.get(symbol);
    if (cached) return cached;
    const inflight = mountainGeneMetaInflightRef.current.get(symbol);
    if (inflight) return inflight;

    const fetchPromise = (async (): Promise<MountainGeneMetadata> => {
      const fallbackStudy = buildStudyGeneFallback(symbol);
      const myGeneEndpoint = 'https://mygene.info/v3/query';
      let meta: MountainGeneMetadata = {
        fullName: 'Name unavailable',
        functionText: 'Function description unavailable',
        sources: [{ label: 'GEMMAP local fallback', note: 'No external gene metadata source returned a usable record.' }]
      };

      try {
        const queryUrls = [
          `${myGeneEndpoint}?q=symbol:${encodeURIComponent(symbol)}&species=human,mouse&size=10&fields=symbol,name,summary,alias`,
          `${myGeneEndpoint}?q=${encodeURIComponent(symbol)}&species=human,mouse&size=10&fields=symbol,name,summary,alias`
        ];
        for (const url of queryUrls) {
          const res = await fetch(url);
          if (!res.ok) continue;
          const payload = await res.json();
          const hits = Array.isArray(payload?.hits) ? payload.hits : [];
          if (!hits.length) continue;
          const exactSymbolHit = hits.find((h: any) => String(h?.symbol || '').toUpperCase() === symbol);
          const aliasHit = hits.find((h: any) =>
            Array.isArray(h?.alias) && h.alias.some((a: any) => String(a || '').toUpperCase() === symbol)
          );
          const best = exactSymbolHit || aliasHit || hits[0];
          if (!best) continue;

          const summaryText = String(best.summary || '').trim();
          const functionText = String(summaryText || fallbackStudy || 'Function description unavailable')
            .replace(/\s+/g, ' ')
            .trim();
          const sourceNote = exactSymbolHit
            ? `Matched exact symbol ${symbol}; fields used: name + summary.`
            : aliasHit
              ? `Matched via alias for ${symbol}; fields used: name + summary.`
              : `Used best available MyGene hit for ${symbol}; fields used: name + summary.`;
          const entrezId = String(best.entrezgene || '').trim();
          const ncbiUrl = entrezId
            ? `https://www.ncbi.nlm.nih.gov/gene/${encodeURIComponent(entrezId)}`
            : `https://www.ncbi.nlm.nih.gov/gene/?term=${encodeURIComponent(symbol)}%5Bsym%5D`;
          const sources: GeneMetadataSource[] = [
            {
              label: 'NCBI Gene',
              note: entrezId
                ? `Curated NCBI gene record for ${symbol} (Entrez ${entrezId}).`
                : `NCBI gene search page for ${symbol}.`,
              url: ncbiUrl
            },
            {
              label: 'MyGene.info API',
              note: sourceNote
            }
          ];
          if (!summaryText && fallbackStudy) {
            sources.push({
              label: 'GEMMAP local study context',
              note: 'Summary fallback inferred from uploaded pathway/module membership in this session.'
            });
          }
          meta = {
            fullName: compactText(String(best.name || symbol), 160),
            functionText: compactText(functionText, 2400),
            sources
          };
          break;
        }
      } catch {
        // Keep graceful fallback when external metadata lookup is unavailable.
      }

      if (meta.functionText === 'Function description unavailable') {
        if (fallbackStudy) {
          meta = {
            fullName: meta.fullName === 'Name unavailable' ? symbol : meta.fullName,
            functionText: compactText(fallbackStudy, 2400),
            sources: [
              {
                label: 'GEMMAP local study context',
                note: 'No external metadata was available; function text inferred from this uploaded dataset.'
              }
            ]
          };
        } else {
          meta = {
            ...meta,
            sources: [
              {
                label: 'GEMMAP local fallback',
                note: 'No external metadata or study-derived fallback was available for this gene.'
              }
            ]
          };
        }
      }

      mountainGeneMetaCacheRef.current.set(symbol, meta);
      if (mountainGeneMetaCacheRef.current.size > 320) {
        const keys = Array.from(mountainGeneMetaCacheRef.current.keys());
        for (const key of keys.slice(0, Math.max(0, keys.length - 280))) {
          mountainGeneMetaCacheRef.current.delete(key);
        }
      }
      return meta;
    })();

    mountainGeneMetaInflightRef.current.set(symbol, fetchPromise);
    try {
      return await fetchPromise;
    } finally {
      mountainGeneMetaInflightRef.current.delete(symbol);
    }
  };

  const loadMountainData = async (options?: { preserveTrailTalk?: boolean; sessionOverride?: string }): Promise<boolean> => {
    const targetSessionId = options?.sessionOverride || sessionId;
    if (!targetSessionId) return false;
    const preserveTrailTalk = Boolean(options?.preserveTrailTalk);
    setMountainLoading(true);
    setMountainError(null);
    try {
      const res = await fetch(`${API_URL}/mountain-data`, { headers: { 'X-Session-ID': targetSessionId } });
      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        throw new Error(extractApiErrorMessage(data, 'Failed to load mountain data.'));
      }
      setMountainData(data);
      const firstPath = Array.isArray(data.pathways) && data.pathways.length > 0 ? data.pathways[0] : null;
      setSelectedMountainPathId(firstPath ? Number(firstPath.idx) : null);
      setMountainPathSearch('');
      setMountainHitInfo(null);
      mountainLastHoverKeyRef.current = '';
      mountainHoverThrottleTsRef.current = 0;
      setMountainLockedGeneKey(null);
      mountainLockedGeneKeyRef.current = null;
      setMountainOriginPapers([]);
      setMountainExpandedAbstracts({});
      setMountainOriginQuery('');
      setMountainOriginMessage('');
      setMountainOriginExact(null);
      setMountainOriginQueries([]);
      setMountainNamingClue(null);
      setMountainTrailSearchSpec(null);
      setMountainPaperAnalyses({});
      setMountainExpandedAnalyses({});
      setMountainPaperAnalysisLoading(null);
      if (!preserveTrailTalk) {
        setMountainStudyContext((prev) => ({ ...prev, notes: '' }));
      }
      return true;
    } catch (err: any) {
      setMountainData(null);
      setSelectedMountainPathId(null);
      setMountainError(err?.message || 'Failed to load mountain data.');
      return false;
    } finally {
      setMountainLoading(false);
    }
  };

  useEffect(() => {
    if (step === 'dashboard' && results && sessionId && !mountainData && !mountainLoading && !mountainError) {
      loadMountainData();
    }
  }, [step, results, sessionId]);

  useEffect(() => {
    if (!pendingPathwayJump || !mountainData?.pathways?.length) return;
    if (selectMountainPathwayByName(pendingPathwayJump)) {
      setPendingPathwayJump(null);
    }
  }, [pendingPathwayJump, mountainData]);

  useEffect(() => {
    setPendingPaperForContext(null);
    setPendingContextError('');
  }, [selectedMountainPathId]);

  useEffect(() => {
    setMountainHitExpanded(false);
  }, [selectedMountainPathId, mountainHitInfo?.symbol]);

  const requestPathwayOriginPapers = async () => {
    if (!sessionId || !selectedMountainPathway) return;
    beginTrailTalkWorkflow(selectedMountainPathway.pathway);
    if (!apiKey.trim()) {
      setMountainOriginMessage('Add API key to run agentic pathway-origin search.');
      updateTrailTalkStep('prepare', 'done', 'Pathway context prepared.');
      updateTrailTalkStep('configure', 'error', 'Missing API key. Configure Trail Talk API access to continue.');
      setTrailTalkElapsedSec(0);
      return;
    }
    setMountainOriginLoading(true);
    setMountainOriginPapers([]);
    setMountainExpandedAbstracts({});
    setMountainExpandedAnalyses({});
    setPendingPaperForContext(null);
    setPendingContextError('');
    setMountainOriginQuery('');
    setMountainOriginMessage('');
    setMountainOriginExact(null);
    setMountainOriginQueries([]);
    setMountainNamingClue(null);
    setMountainTrailSearchSpec(null);
    let activeStep: TrailTalkStep['key'] = 'prepare';
    try {
      updateTrailTalkStep('prepare', 'done', `Pathway context ready for ${selectedMountainPathway.pathway}.`);
      activeStep = 'configure';
      updateTrailTalkStep(
        'configure',
        'running',
        `${trailUseNamingClue ? 'Naming Clues enabled.' : 'Naming Clues disabled.'} ${trailMsigdbGoOnly ? 'Mode: Origin.' : 'Mode: Origin + Context.'} LLM tier: ${effectiveAgentTurbo ? 'Turbo.' : 'Standard.'}`
      );
      updateTrailTalkStep('configure', 'done');
      activeStep = 'local';
      updateTrailTalkStep('local', 'running', 'Submitting local deterministic retrieval request to Trail Talk API.');
      const res = await fetch(`${API_URL}/pathway-origin`, {
        method: 'POST',
        headers: {
          'X-Session-ID': sessionId,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          pathway: selectedMountainPathway.pathway,
          max_results: 6,
          api_key: apiKey.trim(),
          provider: agentProvider,
          turbo: effectiveAgentTurbo,
          use_naming_clue: trailUseNamingClue,
          msigdb_go_only: trailMsigdbGoOnly
        })
      });
      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        throw new Error(extractApiErrorMessage(data, 'Pathway-origin paper search failed.'));
      }
      const papers = Array.isArray(data.papers) ? data.papers : [];
      const candidatePreview = Array.isArray(data.candidate_preview)
        ? data.candidate_preview.map((p: any) => ({
            ...p,
            selection_source: String(p?.selection_source || 'candidate_pool_unvalidated')
          }))
        : [];
      const displayPapers = papers.length > 0 ? papers : candidatePreview;
      const exact = Boolean(data.exact_name_found);
      const searchSpec = data.search_spec && typeof data.search_spec === 'object'
        ? data.search_spec as TrailSearchSpec
        : null;
      const localMode = String(searchSpec?.local_search || '').trim();
      const localNotes = Array.isArray(searchSpec?.local_search_notes)
        ? searchSpec.local_search_notes.map((n) => String(n || '').trim()).filter(Boolean)
        : [];
      updateTrailTalkStep(
        'local',
        'done',
        [
          `Local retrieval completed with ${Number(data?.n_candidates || 0)} candidate papers.`,
          localMode ? `Local: ${localMode}` : '',
          localNotes.length > 0 ? localNotes.join(' | ') : ''
        ].filter(Boolean).join('\n')
      );
      activeStep = 'llm';
      updateTrailTalkStep('llm', 'running', 'Validating candidate pool relevance with strict no-out-of-pool policy.');
      setMountainOriginQuery(String(data.query || ''));
      setMountainOriginQueries(Array.isArray(data.queries) ? data.queries.map((q: any) => String(q || '')) : []);
      setMountainOriginExact(exact);
      setMountainOriginPapers(displayPapers);
      setMountainExpandedAbstracts({});
      setMountainNamingClue(data.naming_clue && typeof data.naming_clue === 'object' ? data.naming_clue as TrailNamingClue : null);
      setMountainTrailSearchSpec(searchSpec);
      const llmNoMatch = Boolean(searchSpec?.llm_no_match);
      const llmMode = String(searchSpec?.llm_search || '').trim();
      const llmConfidence = String(searchSpec?.llm_confidence || 'low').trim() || 'low';
      const llmTier = String(searchSpec?.llm_model_tier || (effectiveAgentTurbo ? 'turbo' : 'standard')).trim() || 'standard';
      updateTrailTalkStep(
        'llm',
        'done',
        [
          llmMode ? `LLM: ${llmMode}` : 'LLM: Candidate-pool relevance validation completed.',
          `LLM tier: ${llmTier}`,
          `LLM confidence: ${llmConfidence}`,
          llmNoMatch
            ? (
              candidatePreview.length > 0
                ? `Result: no direct origin paper validated; showing ${candidatePreview.length} unvalidated candidates.`
                : 'Result: no direct origin paper validated.'
            )
            : `Result: selected ${papers.length} evidence paper${papers.length === 1 ? '' : 's'}.`
        ].join('\n')
      );
      activeStep = 'finalize';
      updateTrailTalkStep('finalize', 'running', 'Rendering concise evidence panel output.');
      if (papers.length === 0) {
        setMountainOriginMessage(String(data.message || 'I could not find relevant evidence.'));
      } else {
      setMountainOriginMessage(String(data.message || ''));
    }
    setMountainPaperAnalyses({});
    setMountainExpandedAnalyses({});
    setMountainPaperAnalysisLoading(null);
    const taggedCount = displayPapers.filter((p: any) => Array.isArray(p?.source_tags) && p.source_tags.length > 0).length;
      updateTrailTalkStep(
        'finalize',
        'done',
        papers.length === 0
          ? (
            candidatePreview.length > 0
              ? `Completed with ${candidatePreview.length} unvalidated candidate paper${candidatePreview.length === 1 ? '' : 's'} for manual review.`
              : 'Completed with no reliable origin evidence.'
          )
          : `Completed with ${papers.length} evidence paper${papers.length === 1 ? '' : 's'} (${taggedCount} with provenance tags).`
      );
    } catch (err: any) {
      setMountainOriginPapers([]);
      setMountainExpandedAbstracts({});
      setMountainExpandedAnalyses({});
      setMountainOriginQuery('');
      setMountainOriginQueries([]);
      setMountainNamingClue(null);
      setMountainTrailSearchSpec(null);
      setMountainOriginMessage(err?.message || 'Pathway-origin paper lookup failed.');
      updateTrailTalkStep(activeStep, 'error', err?.message || 'Trail Talk step failed.');
      updateTrailTalkStep('finalize', 'error', 'Trail Talk stopped before final rendering.');
    } finally {
      setMountainOriginLoading(false);
      setTrailStatusExpanded(false);
    }
  };

  const analyzeOriginPaperWithAgents = async (paper: TrailPaper): Promise<boolean> => {
    if (!sessionId || !selectedMountainPathway) return false;
    if (!apiKey) {
      setMountainOriginMessage('Add API key in Trail Talk to run evidence analysis.');
      return false;
    }
    const paperKey = String(paper.pmid || paper.doi || paper.title || '').trim();
    if (!paperKey) return false;

    setMountainPaperAnalysisLoading(paperKey);
    try {
      const res = await fetch(`${API_URL}/pathway-paper-analyze`, {
        method: 'POST',
        headers: {
          'X-Session-ID': sessionId,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          api_key: apiKey,
          provider: agentProvider,
          turbo: effectiveAgentTurbo,
          pathway: selectedMountainPathway.pathway,
          module_id: String(selectedMountainPathway.cluster),
          module_name: selectedMountainPathway.module,
          paper,
          study_disease: mountainStudyContext.disease.trim() || null,
          study_tissue: mountainStudyContext.tissue.trim() || null,
          study_organism: mountainStudyContext.organism.trim() || null,
          study_technology: mountainStudyContext.technology.trim() || null,
          study_cohort: mountainStudyContext.cohort.trim() || null,
          study_notes: mountainStudyContext.notes.trim() || null
        })
      });
      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        throw new Error(extractApiErrorMessage(data, 'Paper relevance analysis failed.'));
      }
      setMountainPaperAnalyses((prev) => ({
        ...prev,
        [paperKey]: {
          evidence: String(data.evidence_agent || ''),
          hypothesis: String(data.hypothesis_agent || ''),
          openAccessNote: String(data.open_access_note || ''),
          nHits: Number(data.n_gene_hits || 0),
          nUp: Number(data.n_up || 0),
          nDown: Number(data.n_down || 0),
          moduleId: String(data.module_id || ''),
          moduleName: String(data.module_name || ''),
          geneHits: Array.isArray(data.gene_hits) ? data.gene_hits : []
        }
      }));
      return true;
    } catch (err: any) {
      setMountainPaperAnalyses((prev) => ({
        ...prev,
        [paperKey]: {
          evidence: err?.message || 'Paper relevance analysis failed.',
          hypothesis: '',
          openAccessNote: '',
          nHits: 0,
          nUp: 0,
          nDown: 0,
          moduleId: '',
          moduleName: '',
          geneHits: []
        }
      }));
      return false;
    } finally {
      setMountainPaperAnalysisLoading(null);
    }
  };

  const openAnalyzeRelevancePrompt = (paper: TrailPaper) => {
    setPendingPaperForContext(paper);
    setPendingContextError('');
    window.setTimeout(() => {
      trailContextRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
  };

  const runPendingPaperAnalysis = async () => {
    if (!pendingPaperForContext) return;
    const missing: string[] = [];
    if (!mountainStudyContext.disease.trim()) missing.push('Disease or phenotype');
    if (!mountainStudyContext.tissue.trim()) missing.push('Tissue or cell type');
    if (!mountainStudyContext.organism.trim()) missing.push('Organism');
    if (missing.length > 0) {
      setPendingContextError(`Please provide required context before analysis: ${missing.join(', ')}.`);
      window.setTimeout(() => {
        trailContextRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 80);
      return;
    }
    setPendingContextError('');
    const ok = await analyzeOriginPaperWithAgents(pendingPaperForContext);
    if (ok) {
      setPendingPaperForContext(null);
    }
  };

  const quickDebugLogExport = async (issueSummary: string): Promise<boolean> => {
    if (issueReportLoading) return false;
    setIssueReportLoading(true);
    try {
      const headers: Record<string, string> = {};
      if (sessionId) headers['X-Session-ID'] = sessionId;
      const params = new URLSearchParams();
      const issueSummaryClean = issueSummary.trim();
      if (issueSummaryClean) params.set('issue_summary', issueSummaryClean);
      const query = params.toString() ? `?${params.toString()}` : '';
      const res = await fetch(`${API_URL}/export-debug-log${query}`, { headers });
      if (!res.ok) throw new Error('Debug log export failed');
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename="?([^"]+)"?/i);
      const suggestedFilename = match?.[1] || 'gemmap_issue_report.txt';
      const fallbackIssueId = `GMM-${Date.now()}`;
      const issueId = res.headers.get('X-Report-Issue-ID') || fallbackIssueId;
      const issueEmail = res.headers.get('X-Report-Issue-Email') || DEFAULT_ISSUE_REPORT_EMAIL;
      const issueSubject =
        res.headers.get('X-Report-Issue-Subject')
        || `GEMMAP v${APP_VERSION} Debug Log Report | ${issueId}`;
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = suggestedFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      setIssueReportInfo({
        email: issueEmail,
        subject: issueSubject,
        issueSummary: issueSummaryClean,
      });
      setIssueReportCopyStatus('');
      return true;
    } catch (err) {
      console.error(err);
      alert('Debug log export failed. Please try again.');
      return false;
    } finally {
      setIssueReportLoading(false);
    }
  };

  const submitIssueReportExport = async () => {
    if (!issueSummaryInput.trim()) {
      alert('Please tell us what happened before exporting the debug log.');
      return;
    }
    const ok = await quickDebugLogExport(issueSummaryInput);
    if (ok) {
      setShowIssueReportPrompt(false);
      setIssueSummaryInput('');
    }
  };

  const triggerEmergencyReproJsonDownload = async (
    triggerSource: string,
    reason?: string
  ): Promise<boolean> => {
    if (!sessionId) return false;
    const now = Date.now();
    if (emergencyJsonInFlightRef.current) return false;
    if (now - emergencyJsonLastTriggerRef.current < 8000) return false;
    emergencyJsonInFlightRef.current = true;
    emergencyJsonLastTriggerRef.current = now;
    try {
      const res = await fetch(`${API_URL}/reproducibility?include_chat_history=true`, {
        headers: { 'X-Session-ID': sessionId }
      });
      if (!res.ok) {
        throw new Error(`Emergency reproducibility export failed (${res.status}).`);
      }
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename="?([^"]+)"?/i);
      const defaultStem = String(reportName || pathFileName || 'gemmap_session')
        .replace(/\.[^/.]+$/, '')
        .replace(/[^\w\-. ]+/g, '_')
        .trim()
        .slice(0, 80) || 'gemmap_session';
      const baseFilename = match?.[1] || `${defaultStem}_reproducibility.json`;
      const normalizedBase = String(baseFilename || '').trim() || `${defaultStem}_reproducibility.json`;
      const baseWithoutExt = normalizedBase.replace(/\.json$/i, '') || `${defaultStem}_reproducibility`;
      const stamp = new Date().toISOString().replace(/[:]/g, '-').replace(/\.\d{3}Z$/, 'Z');
      const emergencyFilename = `${baseWithoutExt}_emergency_${stamp}.json`;

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = emergencyFilename;
      a.setAttribute('data-emergency-repro-export', 'true');
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      console.warn(
        `[Emergency JSON backup] Auto-downloaded ${emergencyFilename} via ${triggerSource}.`,
        reason || 'No reason provided.'
      );
      return true;
    } catch (err) {
      console.error('Emergency JSON backup failed:', err);
      return false;
    } finally {
      emergencyJsonInFlightRef.current = false;
    }
  };

  const quickReproExport = async (includeChat?: boolean) => {
    if (!sessionId || jsonExportLoading) return;
    if (typeof includeChat !== 'boolean') {
      setQuickExportPrompt('json');
      return;
    }
    setIncludeChatInJson(includeChat);
    setJsonExportLoading(true);
    setJsonExportProgress(2);
    setJsonExportStatus('Stage 1/4: preparing reproducibility manifest export...');
    let progressTimer: number | null = null;
    try {
      progressTimer = window.setInterval(() => {
        setJsonExportProgress((prev) => {
          if (prev >= 93) return prev;
          if (prev < 40) return prev + 4;
          if (prev < 70) return prev + 2;
          return prev + 1;
        });
      }, 260);
      const query = includeChat ? '?include_chat_history=true' : '?include_chat_history=false';
      setJsonExportStatus(`Stage 2/4: collecting module settings and ${includeChat ? 'chat transcript' : 'non-chat'} replay data...`);
      const res = await fetch(`${API_URL}/reproducibility${query}`, { headers: { 'X-Session-ID': sessionId } });
      if (!res.ok) throw new Error('Reproducibility export failed');
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename="?([^"]+)"?/i);
      const suggestedFilename = match?.[1] || 'gemmap_reproducibility.json';
      setJsonExportStatus('Stage 3/4: serializing reproducibility JSON payload...');
      setJsonExportProgress((prev) => Math.max(prev, 74));
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = suggestedFilename;
      setJsonExportStatus('Stage 4/4: finalizing JSON and starting download...');
      setJsonExportProgress((prev) => Math.max(prev, 95));
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      setJsonExportProgress(100);
      setJsonExportStatus('Reproducibility JSON ready.');
    } catch (err) {
      console.error(err);
      setJsonExportStatus('Reproducibility JSON export failed.');
      alert('Reproducibility JSON export failed. Please try again.');
    } finally {
      if (progressTimer) window.clearInterval(progressTimer);
      window.setTimeout(() => {
        setJsonExportLoading(false);
        setJsonExportProgress(0);
        setJsonExportStatus('');
      }, 900);
    }
  };

  const triggerReproReplayPicker = () => {
    replayManifestInputRef.current?.click();
  };

  const extractBalancedJsonObject = (text: string, startIndex: number): string | null => {
    if (startIndex < 0 || startIndex >= text.length || text[startIndex] !== '{') return null;
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let i = startIndex; i < text.length; i += 1) {
      const ch = text[i];
      if (inString) {
        if (escaped) {
          escaped = false;
          continue;
        }
        if (ch === '\\') {
          escaped = true;
          continue;
        }
        if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') {
        inString = true;
        continue;
      }
      if (ch === '{') {
        depth += 1;
        continue;
      }
      if (ch === '}') {
        depth -= 1;
        if (depth === 0) return text.slice(startIndex, i + 1);
        if (depth < 0) return null;
      }
    }
    return null;
  };

  const extractManifestFromHtml = (htmlText: string): any => {
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(htmlText, 'text/html');
      const embedded = doc.getElementById('gemmap-repro-manifest');
      const embeddedText = String(embedded?.textContent || '').trim();
      if (embeddedText) {
        return JSON.parse(embeddedText);
      }
    } catch {
      // fall through to script extraction
    }

    const markerMatch = /const\s+reproManifest\s*=\s*/.exec(htmlText);
    if (!markerMatch) {
      throw new Error('No embedded reproducibility manifest found in the HTML file.');
    }
    const markerEnd = markerMatch.index + markerMatch[0].length;
    const objectStart = htmlText.indexOf('{', markerEnd);
    if (objectStart < 0) {
      throw new Error('Embedded reproducibility manifest is malformed.');
    }
    const objectJson = extractBalancedJsonObject(htmlText, objectStart);
    if (!objectJson) {
      throw new Error('Could not read embedded reproducibility manifest from HTML.');
    }
    return JSON.parse(objectJson);
  };

  const parseReproManifestFromFile = (file: File, text: string): any => {
    const name = String(file.name || '').toLowerCase();
    const mime = String(file.type || '').toLowerCase();
    const looksLikeHtml = name.endsWith('.html') || name.endsWith('.htm') || mime.includes('text/html');
    if (!looksLikeHtml) {
      try {
        return JSON.parse(text);
      } catch {
        // fall back to HTML manifest extraction
      }
    }
    return extractManifestFromHtml(text);
  };

  const handleReproReplayFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setReplayLoading(true);
    try {
      const text = await file.text();
      const manifest = parseReproManifestFromFile(file, text);
      const res = await fetch(`${API_URL}/reproducibility/replay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          manifest,
          provider: agentProvider,
          turbo: effectiveAgentTurbo,
          rerun_annotations: false
        })
      });
      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        throw new Error(extractApiErrorMessage(data, 'Repro replay failed.'));
      }
      const replaySessionId = data.session_id ? String(data.session_id) : (sessionId || '');
      if (data.session_id) {
        setSessionId(String(data.session_id));
      }
      const replayAnnotations = Array.isArray(data.ai_annotations) ? data.ai_annotations : [];
      setResults(data);
      setAiAnnotations(replayAnnotations);
      setUseAgent(replayAnnotations.length > 0);
      setAgentStatus(replayAnnotations.length > 0 ? 'complete' : 'idle');
      setApiKey('');
      restoreChatsFromReplay(data.chat_log);
      setGeneSearchQuery('');
      setActiveTab('3d');
      setStep('dashboard');
      await loadMountainData({ preserveTrailTalk: true, sessionOverride: replaySessionId || undefined });
    } catch (err: any) {
      alert(err?.message || 'Failed to replay reproducibility file. Upload a GEMMAP .json or .html report.');
    } finally {
      e.target.value = '';
      setReplayLoading(false);
    }
  };

  const passScrollToPage = (event: React.WheelEvent<HTMLDivElement>) => {
    const el = event.currentTarget;
    const atTop = el.scrollTop <= 0;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
    if ((event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom)) {
      event.preventDefault();
      window.scrollBy({ top: event.deltaY, behavior: 'auto' });
    }
  };

  const exportSelectedMountainPlot = async () => {
    if (!window.Plotly || !selectedMountainPathway) return;
    const series = buildMountainSeries(selectedMountainPathway);
    if (!series) {
      alert('Mountain plot export failed: data unavailable.');
      return;
    }
    const exportWidth = Math.max(300, Math.min(5200, Math.round(Number(mountainExportConfig.width) || 800)));
    const exportHeight = Math.max(300, Math.min(3600, Math.round(Number(mountainExportConfig.height) || 650)));
    const exportFormat: MountainExportFormat = mountainExportConfig.format || 'png';
    const exportScale =
      mountainExportConfig.quality === 'ultra'
        ? 3
        : mountainExportConfig.quality === 'high'
          ? 2
          : 1.4;
    let tempDiv: HTMLDivElement | null = null;
    try {
      const pValueText = formatPValue(selectedMountainPathway.p_value);
      const adjPValueText = formatPValue(selectedMountainPathway.adj_p_value);
      const clusterColor = colorForCluster(selectedMountainPathway.cluster);
      const ySpan = Math.max(series.ymax - series.ymin, 0.4);
      const stripTop = series.ymin - ySpan * 0.14;
      const stripHeight = Math.max(ySpan * 0.06, 0.03);
      const stripBottom = stripTop - stripHeight;
      const yFloor = stripBottom - ySpan * 0.08;
      const yCeil = series.ymax + ySpan * 0.24;
      const stripMid = (stripTop + stripBottom) / 2;
      const maxX = Math.max(...series.x, 1);
      const stripLabelY = stripBottom - ySpan * 0.02;
      const hitByGene = new Map(series.hitMeta.map((h) => [String(h.gene || '').toUpperCase(), h]));
      const selectedLabelPoints = mountainExportSelectedGenes
        .map((gene) => {
          const hit = hitByGene.get(String(gene || '').toUpperCase());
          if (!hit) return null;
          const yVal = series.y[Math.max(0, Math.min(series.y.length - 1, Number(hit.rank) - 1))];
          if (!Number.isFinite(yVal)) return null;
          return {
            gene: String(hit.gene || '').toUpperCase(),
            rank: Number(hit.rank),
            es: Number(yVal),
            logfc: Number(hit.logfc || 0)
          };
        })
        .filter((p): p is { gene: string; rank: number; es: number; logfc: number } => Boolean(p));
      const traces = [
        {
          x: series.x,
          y: series.y,
          type: 'scatter',
          mode: 'lines',
          name: 'Running ES',
          line: { color: '#334155', width: 2.4 },
          visible: showRunningEs ? true : 'legendonly'
        },
        {
          x: series.hitMeta.map((h) => h.rank),
          y: series.hitMeta.map(() => series.hitY),
          type: 'scatter',
          mode: 'markers',
          name: 'Module Hits',
          marker: { color: clusterColor, symbol: 'line-ns-open', size: 12, line: { width: 1, color: clusterColor } },
          customdata: series.hitMeta.map((h) => [h.gene, Number(h.logfc).toFixed(3)]),
          hovertemplate: '<b>%{customdata[0]}</b><br>Rank: %{x}<br>LogFC: %{customdata[1]}<extra></extra>',
          visible: showPathwayGenes ? true : 'legendonly'
        },
        {
          x: series.x,
          y: series.x.map(() => stripMid),
          type: 'scatter',
          mode: 'markers',
          name: 'Rank Metric',
          marker: {
            symbol: 'square',
            size: 8,
            color: series.rankMetric,
            cmin: -series.maxAbsLogfc,
            cmax: series.maxAbsLogfc,
            colorscale: [
              [0, '#1d4ed8'],
              [0.5, '#475569'],
              [1, '#dc2626']
            ],
            showscale: false,
            line: { width: 0 }
          },
          customdata: series.rankMetric.map((v: number) => Number(v).toFixed(3)),
          hovertemplate: 'Rank: %{x}<br>Rank metric: %{customdata}<extra></extra>',
          showlegend: false,
          visible: showRankStrip ? true : 'legendonly'
        },
        {
          x: series.x,
          y: series.x.map(() => stripMid),
          type: 'scatter',
          mode: 'markers',
          name: 'Rank Strip',
          marker: {
            symbol: 'square',
            size: 14,
            color: series.x.map((v) => v),
            colorscale: [
              [0, '#dc2626'],
              [0.5, '#4b5563'],
              [1, '#1d4ed8']
            ],
            cmin: 1,
            cmax: maxX,
            opacity: 0.35,
            showscale: false,
            line: { width: 0 }
          },
          hoverinfo: 'skip',
          showlegend: false,
          visible: showRankStrip ? true : 'legendonly'
        },
        ...(selectedLabelPoints.length > 0 ? [
          {
            x: selectedLabelPoints.map((p) => p.rank),
            y: selectedLabelPoints.map((p) => p.es),
            type: 'scatter',
            mode: 'markers+text',
            name: 'Selected Genes',
            marker: { color: '#f59e0b', size: 7, line: { width: 1, color: '#7c2d12' } },
            text: selectedLabelPoints.map((p) => p.gene),
            textposition: 'top center',
            textfont: { color: '#92400e', size: 12 },
            cliponaxis: false,
            customdata: selectedLabelPoints.map((p) => Number(p.logfc).toFixed(3)),
            hovertemplate: '<b>%{text}</b><br>Rank: %{x}<br>ES: %{y:.3f}<br>LogFC: %{customdata}<extra></extra>',
            showlegend: false
          }
        ] : [])
      ];
      const layout = {
        title: {
        text: `${selectedMountainPathway.pathway}<br><span style="font-size:13px;color:#64748b">Module M${selectedMountainPathway.cluster} | NES ${Number(selectedMountainPathway.nes || 0).toFixed(2)} | Pvalue ${pValueText} | Adjusted Pvalue ${adjPValueText} | Peak ES ${series.peakEs >= 0 ? '+' : ''}${series.peakEs.toFixed(3)}</span>`,
          x: 0.5,
          xanchor: 'center'
        },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        font: { color: '#0f172a', size: 15 },
        hovermode: 'closest',
        hoverlabel: {
          align: 'left',
          bgcolor: '#fef3c7',
          bordercolor: '#f59e0b',
          font: { size: 12, color: '#0f172a' },
          namelength: -1
        },
        margin: { l: 78, r: 34, t: 88, b: 74 },
        xaxis: { title: 'Rank in DEG Gene List (by logFC)', gridcolor: '#e2e8f0', linecolor: '#cbd5e1', titlefont: { size: 16 } },
        yaxis: { title: 'Enrichment Score (ES)', gridcolor: '#e2e8f0', zerolinecolor: '#94a3b8', linecolor: '#cbd5e1', titlefont: { size: 16 }, range: [yFloor, yCeil] },
        shapes: [
          {
            type: 'line',
            x0: 1,
            x1: maxX,
            y0: 0,
            y1: 0,
            line: { color: '#94a3b8', width: 1, dash: 'dash' }
          },
          ...(showRankStrip ? [
            {
              type: 'rect',
              x0: 1,
              x1: maxX,
              y0: stripBottom,
              y1: stripTop,
              fillcolor: 'rgba(226,232,240,0.35)',
              line: { width: 0 }
            }
          ] : [])
        ],
        annotations: [
          ...(showRankStrip && showRankStripLegend ? [
            {
              x: 1,
              y: stripLabelY,
              xref: 'x',
              yref: 'y',
              xanchor: 'left',
              yanchor: 'top',
              showarrow: false,
              font: { size: 12, color: '#ef4444', family: '"Avenir Next", "Segoe UI", sans-serif' },
              text: '<span style="font-weight:700;letter-spacing:0.4px;">Activated</span>'
            },
            {
              x: maxX,
              y: stripLabelY,
              xref: 'x',
              yref: 'y',
              xanchor: 'right',
              yanchor: 'top',
              showarrow: false,
              font: { size: 12, color: '#3b82f6', family: '"Avenir Next", "Segoe UI", sans-serif' },
              text: '<span style="font-weight:700;letter-spacing:0.4px;">Suppressed</span>'
            }
          ] : [])
        ],
        legend: { orientation: 'h', y: 1.03, x: 0, font: { size: 13 } }
      };

      tempDiv = document.createElement('div');
      tempDiv.style.position = 'fixed';
      tempDiv.style.left = '-10000px';
      tempDiv.style.top = '-10000px';
      tempDiv.style.width = `${exportWidth}px`;
      tempDiv.style.height = `${exportHeight}px`;
      document.body.appendChild(tempDiv);
      await window.Plotly.newPlot(tempDiv, traces, layout, {
        displaylogo: false,
        staticPlot: true,
        responsive: false,
      });

      let imageData = '';
      if (exportFormat === 'tiff') {
        const pngData = await window.Plotly.toImage(tempDiv, {
          format: 'png',
          width: exportWidth,
          height: exportHeight,
          scale: exportScale
        });
        const res = await fetch(`${API_URL}/export/image/tiff`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_data_url: pngData })
        });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok || payload?.status !== 'success' || !payload?.image_data_url) {
          throw new Error(payload?.detail || payload?.message || 'TIFF conversion failed.');
        }
        imageData = String(payload.image_data_url);
      } else {
        imageData = await window.Plotly.toImage(tempDiv, {
          format: exportFormat,
          width: exportWidth,
          height: exportHeight,
          scale: exportFormat === 'svg' ? 1 : exportScale
        });
      }
      const safeName = String(selectedMountainPathway.pathway || 'mountain_plot')
        .replace(/[^a-zA-Z0-9-_]+/g, '_')
        .slice(0, 120);
      const fileExt = exportFormat === 'jpeg' ? 'jpg' : exportFormat === 'tiff' ? 'tiff' : exportFormat;
      const a = document.createElement('a');
      a.href = imageData;
      a.download = `${safeName}_publication_mountain.${fileExt}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setShowMountainExportModal(false);
    } catch (err) {
      console.error(err);
      alert('Mountain plot export failed.');
    } finally {
      if (tempDiv) {
        try {
          window.Plotly.purge(tempDiv);
        } catch {
          // no-op
        }
        if (tempDiv.parentNode) {
          tempDiv.parentNode.removeChild(tempDiv);
        }
      }
    }
  };

  useEffect(() => {
    if (!selectedMountainPathway || !mountainPlotRef.current || !window.Plotly) {
      return;
    }
    const series = buildMountainSeries(selectedMountainPathway);
    if (!series) {
      mountainPlotRef.current.innerHTML = '<div class="text-slate-400 text-xs p-4">Mountain plot data unavailable.</div>';
      return;
    }
    const pValueText = formatPValue(selectedMountainPathway.p_value);
    const adjPValueText = formatPValue(selectedMountainPathway.adj_p_value);
    const clusterColor = colorForCluster(selectedMountainPathway.cluster);
    const ySpan = Math.max(series.ymax - series.ymin, 0.4);
    const stripTop = series.ymin - ySpan * 0.14;
    const stripHeight = Math.max(ySpan * 0.06, 0.03);
    const stripBottom = stripTop - stripHeight;
    const yFloor = stripBottom - ySpan * 0.08;
    const yCeil = series.ymax + ySpan * 0.24;
    const stripMid = (stripTop + stripBottom) / 2;
    const maxX = Math.max(...series.x, 1);
    const stripLabelY = stripBottom - ySpan * 0.02;

    const traces = [
      {
        x: series.x,
        y: series.y,
        type: 'scatter',
        mode: 'lines',
        name: 'Running ES',
        line: { color: '#f8fafc', width: 2.2 },
        visible: showRunningEs ? true : 'legendonly'
      },
      {
        x: series.hitMeta.map((h) => h.rank),
        y: series.hitMeta.map(() => series.hitY),
        type: 'scatter',
        mode: 'markers',
        name: 'Pathway Genes (module color)',
        marker: { color: clusterColor, symbol: 'line-ns-open', size: 12, line: { width: 1, color: clusterColor } },
        customdata: series.hitMeta.map((h) => [h.gene, Number(h.logfc).toFixed(3)]),
        hovertemplate: '<b>%{customdata[0]}</b><br>Rank: %{x}<br>LogFC: %{customdata[1]}<extra></extra>',
        visible: showPathwayGenes ? true : 'legendonly'
      },
      {
        x: series.x,
        y: series.x.map(() => stripMid),
        type: 'scatter',
        mode: 'markers',
        name: 'Rank Metric',
        marker: {
          symbol: 'square',
          size: 7,
          color: series.rankMetric,
          cmin: -series.maxAbsLogfc,
          cmax: series.maxAbsLogfc,
          colorscale: [
            [0, '#1d4ed8'],
            [0.5, '#1f2937'],
            [1, '#dc2626']
          ],
          showscale: false,
          line: { width: 0 }
        },
        customdata: series.rankMetric.map((v: number) => Number(v).toFixed(3)),
        hovertemplate: 'Rank: %{x}<br>Rank metric: %{customdata}<extra></extra>',
        showlegend: false,
        visible: showRankStrip ? true : 'legendonly'
      },
      {
        x: series.x,
        y: series.x.map(() => stripMid),
        type: 'scatter',
        mode: 'markers',
        name: 'Rank Strip',
        marker: {
          symbol: 'square',
          size: 12,
          color: series.x.map((v) => v),
          colorscale: [
            [0, '#dc2626'],
            [0.5, '#1f2937'],
            [1, '#1d4ed8']
          ],
          cmin: 1,
          cmax: maxX,
          opacity: 0.26,
          showscale: false,
          line: { width: 0 }
        },
        hoverinfo: 'skip',
        showlegend: false,
        visible: showRankStrip ? true : 'legendonly'
      }
    ];

    const layout = {
      title: {
        text: `${selectedMountainPathway.pathway}<br><span style="font-size:11px;color:#64748b">Module M${selectedMountainPathway.cluster} | NES ${Number(selectedMountainPathway.nes || 0).toFixed(2)} | Pvalue ${pValueText} | Adjusted Pvalue ${adjPValueText} | Peak ES ${series.peakEs >= 0 ? '+' : ''}${series.peakEs.toFixed(3)}</span>`,
        x: 0.5,
        xanchor: 'center'
      },
      paper_bgcolor: '#0f172a',
      plot_bgcolor: '#0f172a',
      font: { color: '#cbd5e1' },
      hovermode: 'closest',
      hoverlabel: {
        align: 'left',
        bgcolor: '#fef3c7',
        bordercolor: '#f59e0b',
        font: { size: 11, color: '#0f172a' },
        namelength: -1
      },
      autosize: true,
      margin: { l: 70, r: 34, t: 78, b: 64 },
      xaxis: { title: 'Rank in DEG Gene List (by logFC)', gridcolor: '#334155', linecolor: '#475569' },
      yaxis: { title: 'Enrichment Score (ES)', gridcolor: '#334155', zerolinecolor: '#64748b', linecolor: '#475569', range: [yFloor, yCeil] },
      shapes: [
        {
          type: 'line',
          x0: 1,
          x1: maxX,
          y0: 0,
          y1: 0,
          line: { color: '#64748b', width: 1, dash: 'dash' }
        },
        ...(showRankStrip ? [
          {
            type: 'rect',
            x0: 1,
            x1: maxX,
            y0: stripBottom,
            y1: stripTop,
            fillcolor: 'rgba(31,41,55,0.32)',
            line: { width: 0 }
          }
        ] : [])
      ],
      annotations: [
        ...(showRankStrip && showRankStripLegend ? [
          {
            x: 1,
            y: stripLabelY,
            xref: 'x',
            yref: 'y',
            xanchor: 'left',
            yanchor: 'top',
            showarrow: false,
            font: { size: 11, color: '#fb7185', family: '"Avenir Next", "Segoe UI", sans-serif' },
            text: '<span style="font-weight:700;letter-spacing:0.3px;">Activated</span>'
          },
          {
            x: maxX,
            y: stripLabelY,
            xref: 'x',
            yref: 'y',
            xanchor: 'right',
            yanchor: 'top',
            showarrow: false,
            font: { size: 11, color: '#60a5fa', family: '"Avenir Next", "Segoe UI", sans-serif' },
            text: '<span style="font-weight:700;letter-spacing:0.3px;">Suppressed</span>'
          }
        ] : [])
      ],
      legend: { orientation: 'h', y: 1.03, x: 0 }
    };

    window.Plotly.newPlot(mountainPlotRef.current, traces, layout, {
      responsive: true,
      displaylogo: false,
      scrollZoom: false,
      doubleClick: 'reset+autosize'
    });
    setTimeout(() => {
      if (!mountainPlotRef.current || !window.Plotly) return;
      try {
        window.Plotly.Plots.resize(mountainPlotRef.current);
      } catch {
        // no-op
      }
    }, 80);

    const plotNode: any = mountainPlotRef.current;
    const buildHitInfoFromPoint = (point: any, lockKey?: string) => {
      const geneSymbol = String(point?.customdata?.[0] || '').toUpperCase();
      const rank = Number(point?.x || 0);
      const logfc = String(point?.customdata?.[1] || 'NA');
      const hit = series.hitMeta.find((h) => h.rank === rank);
      const displaySymbol = String(hit?.gene || geneSymbol || '').toUpperCase();
      const hitKey = lockKey || `${displaySymbol}@${rank}`;
      const hoverToken = mountainHitHoverSeqRef.current + 1;
      mountainHitHoverSeqRef.current = hoverToken;
      const cachedMeta = mountainGeneMetaCacheRef.current.get(displaySymbol);
      setMountainHitInfo({
        symbol: displaySymbol,
        rank,
        logfc,
        fullName: cachedMeta?.fullName || 'Loading gene metadata...',
        functionText: cachedMeta?.functionText || 'Loading gene metadata...',
        sources: Array.isArray(cachedMeta?.sources) ? cachedMeta.sources : []
      });
      if (cachedMeta) return;
      void fetchGeneMetadata(displaySymbol).then((meta) => {
        if (mountainHitHoverSeqRef.current !== hoverToken) return;
        if (mountainLockedGeneKeyRef.current && mountainLockedGeneKeyRef.current !== hitKey) return;
        setMountainHitInfo({
          symbol: displaySymbol,
          rank,
          logfc,
          fullName: meta.fullName,
          functionText: meta.functionText,
          sources: Array.isArray(meta.sources) ? meta.sources : []
        });
      });
    };

    if (typeof plotNode.removeAllListeners === 'function') {
      plotNode.removeAllListeners('plotly_hover');
      plotNode.removeAllListeners('plotly_click');
    }
    if (mountainBgClickHandlerRef.current) {
      plotNode.removeEventListener('click', mountainBgClickHandlerRef.current);
      mountainBgClickHandlerRef.current = null;
    }

    plotNode.on('plotly_hover', (evt: any) => {
      if (mountainLockedGeneKeyRef.current) return;
      const now = Date.now();
      if (now - mountainHoverThrottleTsRef.current < 45) return;
      mountainHoverThrottleTsRef.current = now;
      const points = Array.isArray(evt?.points) ? evt.points : [];
      const point = points.find((p: any) => String(p?.data?.name || '').startsWith('Pathway Genes'));
      if (!point) return;
      const geneSymbol = String(point?.customdata?.[0] || '').toUpperCase();
      const rank = Number(point?.x || 0);
      const hit = series.hitMeta.find((h) => h.rank === rank);
      const displaySymbol = String(hit?.gene || geneSymbol || '').toUpperCase();
      const hoverKey = `${displaySymbol}@${rank}`;
      if (mountainLastHoverKeyRef.current === hoverKey) return;
      mountainLastHoverKeyRef.current = hoverKey;
      buildHitInfoFromPoint(point, hoverKey);
    });

    plotNode.on('plotly_click', (evt: any) => {
      const points = Array.isArray(evt?.points) ? evt.points : [];
      const point = points.find((p: any) => String(p?.data?.name || '').startsWith('Pathway Genes'));
      if (!point) return;
      const geneSymbol = String(point?.customdata?.[0] || '').toUpperCase();
      const rank = Number(point?.x || 0);
      const hit = series.hitMeta.find((h) => h.rank === rank);
      const displaySymbol = String(hit?.gene || geneSymbol || '').toUpperCase();
      const lockKey = `${displaySymbol}@${rank}`;
      mountainLastPointClickTsRef.current = Date.now();
      mountainLastHoverKeyRef.current = lockKey;
      mountainLockedGeneKeyRef.current = lockKey;
      setMountainLockedGeneKey(lockKey);
      setMountainHitExpanded(false);
      showMountainLockToast(`Locked on ${displaySymbol}. Click plot background to unlock.`);
      buildHitInfoFromPoint(point, lockKey);
    });

    const backgroundClickHandler = (event: MouseEvent) => {
      if (!mountainLockedGeneKeyRef.current) return;
      if (Date.now() - mountainLastPointClickTsRef.current < 160) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest('.point') || target?.closest('.modebar') || target?.closest('.legend')) return;
      mountainLastHoverKeyRef.current = '';
      mountainLockedGeneKeyRef.current = null;
      setMountainLockedGeneKey(null);
      showMountainLockToast('Unlocked gene focus.');
    };
    mountainBgClickHandlerRef.current = backgroundClickHandler;
    plotNode.addEventListener('click', backgroundClickHandler);

    return () => {
      try {
        if (typeof plotNode.removeAllListeners === 'function') {
          plotNode.removeAllListeners('plotly_hover');
          plotNode.removeAllListeners('plotly_click');
        }
      } catch {
        // no-op
      }
      if (mountainBgClickHandlerRef.current) {
        plotNode.removeEventListener('click', mountainBgClickHandlerRef.current);
        mountainBgClickHandlerRef.current = null;
      }
    };
  }, [mountainData, selectedMountainPathway, showRunningEs, showPathwayGenes, showRankStrip, showRankStripLegend]);

  useEffect(() => {
    const handleResize = () => {
      if (!mountainPlotRef.current || !window.Plotly) return;
      try {
        window.Plotly.Plots.resize(mountainPlotRef.current);
      } catch {
        // no-op
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // --- Handlers ---

  const handlePathFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setPathFile(file);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch(`${API_URL}/pathway/preview`, { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
          setColumns(data.columns);
          setPathSignificance(null);

          // Auto-detect columns
          const lowerCols = data.columns.map((c: string) => c.toLowerCase());
          const pCol = data.columns[lowerCols.findIndex((c: string) => c.includes('path') || c.includes('name') || c.includes('term'))] || data.columns[0];
          const gCol = data.columns[lowerCols.findIndex((c: string) => c.includes('gene') || c.includes('leading'))] || data.columns[1];
          const nesIdx = lowerCols.findIndex((c: string) => c.includes('nes') || c.includes('score') || c.includes('enrichment'));
          const padjIdx = lowerCols.findIndex((c: string) => c.includes('padj') || c.includes('fdr') || c.includes('qval') || c.includes('adj_p'));
          const sCol = nesIdx !== -1 ? data.columns[nesIdx] : 'null';
          const pAdjCol = padjIdx !== -1 ? data.columns[padjIdx] : 'null';

          setPathCol(pCol);
          setGeneCol(gCol);
          setScoreCol(sCol);
          setPathPadjCol(pAdjCol);

          setPathStatus('selected');
        } else {
          console.error(data);
          setPathStatus('error');
        }
      } catch (err) {
        console.error(err);
        setPathStatus('error');
      }
    }
  };

  const confirmPathUpload = async () => {
    if (pathFile) {
      const formData = new FormData();
      formData.append('file', pathFile);
      formData.append('path_col', pathCol);
      formData.append('gene_col', geneCol);
      formData.append('score_col', scoreCol);
      formData.append('padj_col', pathPadjCol);
      formData.append('padj_threshold', '0.05');

      try {
        const res = await fetch(`${API_URL}/pathway/upload`, { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
          setSessionId(data.session_id);
          setPathFileName(data.filename);
          setPreviewData(data.preview);
          setPathSignificance(data.pathway_significance || null);
          setPathStatus('uploaded');
        } else {
          setPathStatus('error');
        }
      } catch (err) {
        console.error(err);
        setPathStatus('error');
      }
    }
  };

  const loadDemo = async () => {
    const res = await fetch(`${API_URL}/demo`, { method: 'POST' });
    const data = await res.json();
    setSessionId(data.session_id);
    setColumns(data.columns);
    setPreviewData(data.preview);
    setPathCol('pathway');
    setGeneCol('leadingEdge');
    setScoreCol('NES');
    setPathPadjCol('null');
    setPathSignificance(null);
    setPathFileName('demo_data.csv');
    setPathStatus('uploaded');
    setStep('analyze');
  };

  const handleDegFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setDegFile(file);
      setDegErrorMessage(null);

      // Preview to get columns
      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch(`${API_URL}/deg/preview`, { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
          setDegColumns(data.columns);
          setDegNeedsConfirmation(null);
          setDegSignificance(null);

          // Auto-detect columns
          const lowerCols = data.columns.map((c: string) => c.toLowerCase());
          const gene = data.columns[lowerCols.findIndex((c: string) => c.includes('gene') || c.includes('symbol'))] || data.columns[0];
          const fdr = data.columns[lowerCols.findIndex((c: string) => c.includes('fdr') || c.includes('padj') || c.includes('qval'))] || data.columns[1];
          const pValue = data.columns[lowerCols.findIndex((c: string) => c === 'p_value' || c.includes('pvalue') || c.includes('p.value') || c.includes('p_val'))] || 'null';
          const lfc = data.columns[lowerCols.findIndex((c: string) => c.includes('log') || c.includes('lfc') || c.includes('fold'))] || data.columns[2];

          setDegConfig(prev => ({ ...prev, gene_col: gene, padj_col: fdr, lfc_col: lfc }));
          setDegPValueCol(pValue);
          setDegStatus('selected');
        } else {
          console.error("DEG Preview failed:", data);
          setDegErrorMessage(extractApiErrorMessage(data, 'Failed to preview DEG file.'));
          setDegStatus('error');
        }
      } catch (err) {
        console.error('DEG preview error:', err);
        setDegErrorMessage('Failed to preview DEG file. Please check format and try again.');
        setDegStatus('error');
      }
    }
  };

  const confirmDegUpload = async (useNominalP = false, confirmNonSignificant = false) => {
    if (degFile && sessionId) {
      setDegErrorMessage(null);
      const formData = new FormData();
      formData.append('file', degFile);
      formData.append('gene_col', degConfig.gene_col);
      formData.append('padj_col', degConfig.padj_col);
      formData.append('p_value_col', degPValueCol);
      formData.append('lfc_col', degConfig.lfc_col);
      formData.append('padj_threshold', degConfig.padj_threshold.toString());
      formData.append('lfc_threshold', degConfig.lfc_threshold.toString());
      formData.append('use_nominal_p', useNominalP.toString());
      formData.append('confirm_non_significant', confirmNonSignificant.toString());

      try {
        const res = await fetch(`${API_URL}/deg/upload`, {
          method: 'POST',
          body: formData,
          headers: { 'X-Session-ID': sessionId }
        });
        const data = await res.json();

        if (!res.ok) {
          setDegErrorMessage(extractApiErrorMessage(data, 'DEG upload failed.'));
          setDegStatus('error');
          return;
        }

        if (data.status === 'success') {
          setDegStatus('uploaded');
          setDegInfo({ n_genes: data.n_genes, n_degs: data.n_degs });
          setDegSignificance(data.deg_significance || null);
          setDegNeedsConfirmation(null);
        } else if (data.status === 'needs_confirmation') {
          if (data.deg_significance) {
            setDegNeedsConfirmation({
              message: data.message,
              ...data.deg_significance
            });
          } else {
            setDegNeedsConfirmation({
              message: data?.message || 'Some genes are not significant. Continue with nominal p_value < 0.05?',
              checked_col: degConfig.padj_col,
              threshold: degConfig.padj_threshold,
              n_total: 0,
              n_significant: 0,
              n_non_significant: 0,
              p_value_col: degPValueCol,
              n_nominal_p_below_0_05: 0
            });
          }
        } else {
          setDegErrorMessage(extractApiErrorMessage(data, 'DEG upload failed.'));
          setDegStatus('error');
        }
      } catch (err) {
        console.error('DEG upload error:', err);
        setDegErrorMessage('DEG upload request failed. Please retry.');
        setDegStatus('error');
      }
    }
  };

  const removeDeg = async () => {
    if (sessionId) {
      await fetch(`${API_URL}/deg`, {
        method: 'DELETE',
        headers: { 'X-Session-ID': sessionId }
      });
      setDegFile(null);
      setDegStatus('none');
      setDegInfo(null);
      setDegSignificance(null);
      setDegNeedsConfirmation(null);
      setDegErrorMessage(null);
    }
  };

  const runAutoAnalyze = async () => {
    if (!apiKey) {
      alert('API key required for Autopilot mode');
      return;
    }

    const shouldQueueAgent = Boolean(useAgent && apiKey.trim());
    beginProcessingWorkflow('auto', shouldQueueAgent);
    setAutoAnalyzeLoading(true);
    setStep('processing');
    setAgentStatus(shouldQueueAgent ? 'running' : 'complete');
    setAiAnnotations([]);
    setAutoKResult(null);
    setMountainData(null);
    setMountainError(null);
    setMountainOriginPapers([]);
    setMountainExpandedAbstracts({});
    setMountainOriginQuery('');
    setMountainOriginMessage('');
    setMountainOriginExact(null);
    setMountainOriginQueries([]);
    setMountainNamingClue(null);
    setMountainTrailSearchSpec(null);
    setMountainHitInfo(null);
    mountainLastHoverKeyRef.current = '';
    mountainHoverThrottleTsRef.current = 0;
    setMountainLockedGeneKey(null);
    mountainLockedGeneKeyRef.current = null;

    const formData = new FormData();
    formData.append('pathway_col', pathCol);
    formData.append('genes_col', geneCol);
    formData.append('score_col', scoreCol);
    formData.append('nes_direction', nesDirection);
    formData.append('api_key', apiKey);
    formData.append('provider', agentProvider);
    formData.append('turbo', effectiveAgentTurbo ? 'true' : 'false');
    formData.append('k_min', '2');
    formData.append('k_max', '10');
    formData.append('report_name', reportName.trim());

    let activeStep = 'validate_session';
    try {
      updateProcessingWorkflowStep('validate_session', 'done', 'Session and Autopilot inputs validated.');
      activeStep = 'auto_k_metrics';
      updateProcessingWorkflowStep(
        'auto_k_metrics',
        'running',
        `Running /auto-analyze with provider=${agentProvider}, tier=${effectiveAgentTurbo ? 'turbo' : 'standard'}; computing metrics for k=2..10.`
      );
      const res = await fetch(`${API_URL}/auto-analyze`, {
        method: 'POST',
        body: formData,
        headers: { 'X-Session-ID': sessionId! }
      });
      const data = await res.json();

      if (!res.ok || data.status !== 'success') {
        const errorMessage = extractApiErrorMessage(data, 'Auto-analyze failed.');
        updateProcessingWorkflowStep('auto_k_metrics', 'error', errorMessage);
        throw new Error(errorMessage);
      }
      const metrics = Array.isArray(data?.elbow?.metrics) ? data.elbow.metrics : [];
      const metricKs = metrics
        .map((m: any) => Number(m?.k))
        .filter((k: number) => Number.isFinite(k));
      const metricMinK = metricKs.length > 0 ? Math.min(...metricKs) : 2;
      const metricMaxK = metricKs.length > 0 ? Math.max(...metricKs) : 10;
      const elbowK = Number.isFinite(Number(data?.auto_k?.elbow_k)) ? Number(data.auto_k.elbow_k) : null;
      const silhouetteK = Number.isFinite(Number(data?.auto_k?.silhouette_k)) ? Number(data.auto_k.silhouette_k) : null;
      const statisticalK = Number.isFinite(Number(data?.auto_k?.statistical_k)) ? Number(data.auto_k.statistical_k) : null;
      const recommendedK = Number.isFinite(Number(data?.auto_k?.recommended_k)) ? Number(data.auto_k.recommended_k) : null;
      const rawConfidence = Number(data?.auto_k?.confidence);
      const confidenceText = Number.isFinite(rawConfidence)
        ? rawConfidence.toFixed(2)
        : String(data?.auto_k?.confidence || 'n/a');
      const reasoningText = compactWorkflowNote(data?.auto_k?.reasoning, 150);

      updateProcessingWorkflowStep(
        'auto_k_metrics',
        'done',
        `K sweep complete (${metrics.length} candidates, k=${metricMinK}..${metricMaxK}; elbow=${elbowK ?? 'n/a'}, silhouette=${silhouetteK ?? 'n/a'}).`
      );
      activeStep = 'auto_k_decision';
      updateProcessingWorkflowStep(
        'auto_k_decision',
        'running',
        `Selecting robust k from statistical + AI signals (${agentProvider}, ${effectiveAgentTurbo ? 'turbo' : 'standard'}).`
      );
      updateProcessingWorkflowStep(
        'auto_k_decision',
        'done',
        `Selected k=${recommendedK ?? 'n/a'} (statistical=${statisticalK ?? 'n/a'}, confidence=${confidenceText}).${reasoningText ? ` ${reasoningText}` : ''}`
      );
      activeStep = 'auto_module_mapping';
      updateProcessingWorkflowStep(
        'auto_module_mapping',
        'running',
        `Applying module mapping at k=${recommendedK ?? 'n/a'} and computing module summaries.`
      );
      setResults(data);
      setAutoKResult(data.auto_k);
      setNClusters(data.auto_k.recommended_k);
      setActiveTab('3d');
      updateProcessingWorkflowStep(
        'auto_module_mapping',
        'done',
        `Module mapping completed with ${Array.isArray(data?.clusters) ? data.clusters.length : 0} modules across ${Number(data?.total_pathways || 0)} pathways.`
      );

      activeStep = 'mountain_index';
      updateProcessingWorkflowStep('mountain_index', 'running', 'Fetching /mountain-data payload for Trail Talk evidence.');
      const mountainLoaded = await loadMountainData();
      if (mountainLoaded) {
        updateProcessingWorkflowStep('mountain_index', 'done', 'Mountain explorer data loaded.');
      } else {
        updateProcessingWorkflowStep('mountain_index', 'error', 'Mountain explorer data failed to load.');
      }

      if (shouldQueueAgent) {
        activeStep = 'agent_annotation';
        updateProcessingWorkflowStep(
          'agent_annotation',
          'running',
          `Submitting concise annotation request (${agentProvider}, ${effectiveAgentTurbo ? 'turbo' : 'standard'}).`
        );
        fetch(`${API_URL}/annotate`, {
          method: 'POST',
          headers: {
            'X-Session-ID': sessionId!,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ api_key: apiKey, provider: agentProvider, turbo: effectiveAgentTurbo })
        })
          .then(r => r.json())
          .then(aiData => {
            if (aiData.annotations) {
              setAiAnnotations(aiData.annotations);
              setAgentStatus('complete');
            } else {
              setAgentStatus('error');
            }
          })
          .catch(() => setAgentStatus('error'));
        updateProcessingWorkflowStep(
          'agent_annotation',
          'done',
          'Agent annotation was queued and continues in dashboard.'
        );
      } else {
        setAgentStatus('complete');
      }

      activeStep = 'finalize_dashboard';
      updateProcessingWorkflowStep('finalize_dashboard', 'running', 'Preparing dashboard panels and controls.');
      updateProcessingWorkflowStep('finalize_dashboard', 'done', 'Dashboard ready.');
      setStep('dashboard');
    } catch (err) {
      console.error('Auto-analyze error:', err);
      const message = (err as any)?.message || 'Auto-analyze request failed.';
      updateProcessingWorkflowStep(activeStep, 'error', message);
      updateProcessingWorkflowStep('finalize_dashboard', 'error', 'Stopped before dashboard could be prepared.');
      setStep('analyze');
      setAgentStatus('error');
    } finally {
      setAutoAnalyzeLoading(false);
    }
  };

  const fetchElbowData = async () => {
    setElbowLoading(true);
    setElbowData(null);
    setShowMetricsTable(false);

    const formData = new FormData();
    formData.append('pathway_col', pathCol);
    formData.append('genes_col', geneCol);
    formData.append('score_col', scoreCol);
    formData.append('nes_direction', nesDirection);
    formData.append('k_min', '2');
    formData.append('k_max', '10');

    try {
      const res = await fetch(`${API_URL}/elbow`, {
        method: 'POST',
        body: formData,
        headers: { 'X-Session-ID': sessionId! }
      });
      const data = await res.json();

      if (data.status === 'success') {
        setElbowData(data);
        setSuggestedK(data.elbow.optimal_k);
        setNClusters(data.elbow.optimal_k);
        setShowMetricsTable(false);
        setStep('elbow');
      } else {
        console.error('Elbow analysis failed:', data);
        setStep('analyze');
      }
    } catch (err) {
      console.error('Elbow fetch error:', err);
      setStep('analyze');
    } finally {
      setElbowLoading(false);
    }
  };

  const runAnalysis = async () => {
    const shouldQueueAgent = Boolean(useAgent && apiKey.trim());
    beginProcessingWorkflow('manual', shouldQueueAgent);
    setStep('processing');
    setAgentStatus('idle');
    setAiAnnotations([]);
    setMountainData(null);
    setMountainError(null);
    setMountainOriginPapers([]);
    setMountainExpandedAbstracts({});
    setMountainOriginQuery('');
    setMountainOriginMessage('');
    setMountainOriginExact(null);
    setMountainOriginQueries([]);
    setMountainNamingClue(null);
    setMountainTrailSearchSpec(null);
    setMountainHitInfo(null);
    mountainLastHoverKeyRef.current = '';
    mountainHoverThrottleTsRef.current = 0;
    setMountainLockedGeneKey(null);
    mountainLockedGeneKeyRef.current = null;
    setChatMessages([]);
    setChatInput('');
    setGeneSearchQuery('');

    const formData = new FormData();
    formData.append('pathway_col', pathCol);
    formData.append('genes_col', geneCol);
    formData.append('score_col', scoreCol);
    formData.append('nes_direction', nesDirection);
    formData.append('n_clusters', nClusters.toString());
    formData.append('report_name', reportName.trim());

    let activeStep = 'validate_session';
    try {
      updateProcessingWorkflowStep('validate_session', 'done', 'Manual analysis inputs validated.');
      activeStep = 'module_mapping';
      updateProcessingWorkflowStep(
        'module_mapping',
        'running',
        `Running /analyze with n_clusters=${nClusters}.`
      );
      const res = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
        headers: { 'X-Session-ID': sessionId! }
      });
      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        const errorMessage = extractApiErrorMessage(data, 'Module mapping failed.');
        updateProcessingWorkflowStep('module_mapping', 'error', errorMessage);
        throw new Error(errorMessage);
      }
      setResults(data);
      setActiveTab('3d');
      updateProcessingWorkflowStep(
        'module_mapping',
        'done',
        `Module mapping completed with ${Array.isArray(data?.clusters) ? data.clusters.length : 0} modules.`
      );

      activeStep = 'mountain_index';
      updateProcessingWorkflowStep('mountain_index', 'running', 'Fetching /mountain-data payload.');
      const mountainLoaded = await loadMountainData();
      if (mountainLoaded) {
        updateProcessingWorkflowStep('mountain_index', 'done', 'Mountain explorer data loaded.');
      } else {
        updateProcessingWorkflowStep('mountain_index', 'error', 'Mountain explorer data failed to load.');
      }

      if (shouldQueueAgent) {
        activeStep = 'agent_annotation';
        updateProcessingWorkflowStep(
          'agent_annotation',
          'running',
          `Submitting concise annotation request (${agentProvider}, ${effectiveAgentTurbo ? 'turbo' : 'standard'}).`
        );
        setAgentStatus('running');
        fetch(`${API_URL}/annotate`, {
          method: 'POST',
          headers: {
            'X-Session-ID': sessionId!,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ api_key: apiKey, provider: agentProvider, turbo: effectiveAgentTurbo })
        })
          .then(r => r.json())
          .then(aiData => {
            if (aiData.annotations) {
              setAiAnnotations(aiData.annotations);
              setAgentStatus('complete');
            } else {
              console.error('No annotations in response:', aiData);
              setAgentStatus('error');
            }
          })
          .catch(err => {
            console.error(err);
            setAgentStatus('error');
          });
        updateProcessingWorkflowStep(
          'agent_annotation',
          'done',
          'Agent annotation was queued and continues in dashboard.'
        );
      } else {
        updateProcessingWorkflowStep('agent_annotation', 'skipped', 'Agent annotation disabled or API key missing.');
      }

      activeStep = 'finalize_dashboard';
      updateProcessingWorkflowStep('finalize_dashboard', 'running', 'Preparing dashboard panels and controls.');
      updateProcessingWorkflowStep('finalize_dashboard', 'done', 'Dashboard ready.');
      setStep('dashboard');
    } catch (err) {
      console.error(err);
      const message = (err as any)?.message || 'Analysis failed.';
      updateProcessingWorkflowStep(activeStep, 'error', message);
      updateProcessingWorkflowStep('finalize_dashboard', 'error', 'Stopped before dashboard could be prepared.');
      setStep('analyze');
    }
  };

  const runAiAnnotationNow = async (): Promise<boolean> => {
    if (!sessionId) return false;
    if (!apiKey.trim()) {
      requestResearchApiKey('annotation');
      return false;
    }
    setUseAgent(true);
    setShowResearchApiPrompt(false);
    setAiActionLoading(true);
    setAgentStatus('running');
    try {
      const res = await fetch(`${API_URL}/annotate`, {
        method: 'POST',
        headers: {
          'X-Session-ID': sessionId,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          api_key: apiKey.trim(),
          provider: agentProvider,
          turbo: effectiveAgentTurbo
        })
      });
      const data = await res.json();
      if (!res.ok || !Array.isArray(data.annotations)) {
        throw new Error(extractApiErrorMessage(data, 'AI annotation failed.'));
      }
      setAiAnnotations(data.annotations);
      setAgentStatus('complete');
      return true;
    } catch (err: any) {
      setAgentStatus('error');
      alert(err?.message || 'AI annotation failed.');
      return false;
    } finally {
      setAiActionLoading(false);
    }
  };

  const clearAllChatHistory = async () => {
    if (!sessionId) return;
    try {
      const res = await fetch(`${API_URL}/chat/clear`, {
        method: 'POST',
        headers: { 'X-Session-ID': sessionId }
      });
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(extractApiErrorMessage(data, 'Failed to clear chat history.'));
      }
      setChatMessages([]);
      setMountainPaperAnalyses({});
      setMountainExpandedAnalyses({});
      setPendingPaperForContext(null);
      setPendingContextError('');
    } catch (err: any) {
      alert(err?.message || 'Failed to clear chat history.');
    }
  };

  const rerunFromResearchTools = async (nextK: number, nextDirection: 'all' | 'positive' | 'negative') => {
    if (!sessionId) throw new Error('Session not found.');
    const safeK = Math.max(2, Math.min(25, Math.round(nextK)));
    const formData = new FormData();
    formData.append('pathway_col', pathCol);
    formData.append('genes_col', geneCol);
    formData.append('score_col', scoreCol);
    formData.append('nes_direction', nextDirection);
    formData.append('n_clusters', String(safeK));
    formData.append('report_name', reportName.trim());

    const res = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      body: formData,
      headers: { 'X-Session-ID': sessionId }
    });
    const data = await res.json();
    if (!res.ok || !data) {
      throw new Error(extractApiErrorMessage(data, 'Research tool rerun failed.'));
    }
    setNClusters(safeK);
    setNesDirection(nextDirection);
    setResults(data);
    setAiAnnotations([]);
    setAgentStatus('idle');
    setActiveTab('3d');
    await loadMountainData({ preserveTrailTalk: true });
  };

  const handleResearchToolCommand = async (message: string): Promise<boolean> => {
    const normalized = message.trim().toLowerCase();
    if (!normalized) return false;

    const asksCapabilities = (
      /\b(what can you do|what can you change|capabilities|how can you help|help)\b/.test(normalized) ||
      /^\/?(help|examples?)$/.test(normalized)
    );
    if (asksCapabilities) {
      setChatMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: 'I can analyze your full study context, inspect pathway/gene patterns from uploaded results, search supporting literature, regenerate the map with updated module count or NES focus, re-run AI annotation, and clear chat history when requested.',
          ts: Date.now()
        }
      ]);
      return true;
    }

    const wantsClearChats = (
      /^\/?clear chats?$/.test(normalized) ||
      (/\b(clear|reset|erase|delete)\b/.test(normalized) && /\b(chat|history|conversation|messages)\b/.test(normalized))
    );
    if (wantsClearChats) {
      await clearAllChatHistory();
      return true;
    }

    const wantsReannotate = (
      /^\/?re-?annotate(?: modules?)?$/.test(normalized) ||
      /\b(re-?annotat|annotat(?:e|ion).*(again|refresh|redo|rerun)|refresh.*annotation|redo.*annotation)\b/.test(normalized)
    );
    if (wantsReannotate) {
      if (!apiKey.trim()) {
        requestResearchApiKey('annotation');
        return true;
      }
      const ok = await runAiAnnotationNow();
      setChatMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: ok ? 'Re-annotation completed for current modules.' : 'Re-annotation failed. Check API key/provider and retry.',
          ts: Date.now()
        }
      ]);
      return true;
    }

    const moduleCountMatch =
      normalized.match(/\b(?:modules?|clusters?)\s*=?\s*(\d{1,2})\b/) ||
      normalized.match(/\b(?:set|use|with|to|at)\s+(\d{1,2})\s*(?:modules?|clusters?)\b/) ||
      normalized.match(/\bk\s*=?\s*(\d{1,2})\b/);
    const requestedK = moduleCountMatch ? Number(moduleCountMatch[1]) : null;

    const asksPositiveNES = /\b(positive|upregulated|up-regulated|up)\b/.test(normalized) &&
      /\b(nes|pathway|pathways|enriched|focus|only|filter|keep)\b/.test(normalized);
    const asksNegativeNES = /\b(negative|downregulated|down-regulated|down)\b/.test(normalized) &&
      /\b(nes|pathway|pathways|enriched|focus|only|filter|keep)\b/.test(normalized);
    const asksAllNES = /\b(all|both)\b/.test(normalized) &&
      /\b(nes|pathway|pathways)\b/.test(normalized);

    const explicitRerunIntent = /\b(rerun|re-run|regenerate|rebuild|recompute|recluster|remap)\b/.test(normalized);
    const tuningIntent = /\b(change|adjust|set|update|tune|filter|focus|keep|only|trim|restrict)\b/.test(normalized) &&
      /\b(modules?|clusters?|nes|pathways?|positive|negative)\b/.test(normalized);
    const inferredRerunIntent = explicitRerunIntent || tuningIntent || requestedK !== null || asksPositiveNES || asksNegativeNES || asksAllNES;
    if (!inferredRerunIntent) return false;

    let nextDirection: 'all' | 'positive' | 'negative' = nesDirection;
    if (asksPositiveNES) nextDirection = 'positive';
    if (asksNegativeNES) nextDirection = 'negative';
    if (asksAllNES || /\bnes\s*=\s*all\b/.test(normalized)) nextDirection = 'all';
    const nextK = requestedK ?? nClusters;

    setChatLoading(true);
    try {
      await rerunFromResearchTools(nextK, nextDirection);
      setChatMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: `I updated the analysis and regenerated results from your request. Current settings: ${Math.max(2, Math.min(25, Math.round(nextK)))} modules, NES focus '${nextDirection}'.`,
          ts: Date.now()
        }
      ]);
    } catch (err: any) {
      setChatMessages(prev => [
        ...prev,
        {
          role: 'system',
          text: err?.message || 'Failed to regenerate results from research tool command.',
          ts: Date.now()
        }
      ]);
    } finally {
      setChatLoading(false);
    }
    return true;
  };

  const sendClusterChat = async () => {
    const message = chatInput.trim();
    if (!message || !sessionId) return;

    const userMsg: ClusterChatMessage = { role: 'user', text: message, ts: Date.now() };
    const nextHistory = [...chatMessages, userMsg].slice(-10);
    setChatMessages(nextHistory);
    setChatInput('');

    if (await handleResearchToolCommand(message)) {
      return;
    }

    if (!apiKey) {
      requestResearchApiKey('chat');
      return;
    }

    setChatLoading(true);
    try {
      const res = await fetch(`${API_URL}/cluster-chat`, {
        method: 'POST',
        headers: {
          'X-Session-ID': sessionId,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          api_key: apiKey,
          provider: agentProvider,
          turbo: effectiveAgentTurbo,
          message,
          cluster_id: null,
          history: nextHistory.map(m => ({ role: m.role, text: m.text })),
          include_literature: true
        })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(extractApiErrorMessage(data, 'Module chat failed'));
      }
      setChatMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: String(data.reply || 'No response from module chat agent.'),
          literature: Array.isArray(data.literature) ? data.literature : [],
          ts: Date.now()
        }
      ]);
    } catch (err: any) {
      setChatMessages(prev => [
        ...prev,
        {
          role: 'system',
          text: err?.message || 'Module chat error.',
          ts: Date.now()
        }
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const downloadReport = async () => {
    const res = await fetch(`${API_URL}/download`, { headers: { 'X-Session-ID': sessionId! } });
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "gemmap_analysis.xlsx";
    a.click();
  };

  const quickHtmlExport = async (includeChat?: boolean) => {
    if (!sessionId || htmlExportLoading) return;
    if (typeof includeChat !== 'boolean') {
      setQuickExportPrompt('html');
      return;
    }
    setIncludeChatInHtml(includeChat);
    setHtmlExportLoading(true);
    setHtmlExportProgress(2);
    setHtmlExportStatus('Stage 1/4: validating session and preparing HTML export request...');
    let progressTimer: number | null = null;
    try {
      progressTimer = window.setInterval(() => {
        setHtmlExportProgress((prev) => {
          if (prev >= 92) return prev;
          if (prev < 40) return prev + 4;
          if (prev < 70) return prev + 2;
          return prev + 1;
        });
      }, 260);

      const query = includeChat ? '?include_chat_history=true' : '?include_chat_history=false';
      setHtmlExportStatus('Stage 2/4: building interactive HTML sections and embedding figures...');
      const res = await fetch(`${API_URL}/export-html${query}`, {
        headers: { 'X-Session-ID': sessionId! }
      });
      if (!res.ok) throw new Error('Export failed');
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename="?([^"]+)"?/i);
      const suggestedFilename = match?.[1] || 'GEMMAP_Report.html';
      setHtmlExportStatus('Stage 3/4: streaming generated report payload...');
      setHtmlExportProgress((prev) => Math.max(prev, 72));

      let blob: Blob;
      const contentLength = Number(res.headers.get('Content-Length') || 0);
      if (res.body && typeof res.body.getReader === 'function') {
        const reader = res.body.getReader();
        const chunks: Uint8Array[] = [];
        let received = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            chunks.push(value);
            received += value.length;
            if (contentLength > 0) {
              const pct = Math.min(99, 72 + Math.round((received / contentLength) * 26));
              setHtmlExportProgress((prev) => Math.max(prev, pct));
            } else {
              setHtmlExportProgress((prev) => Math.min(99, prev + 1));
            }
          }
        }
        blob = new Blob(chunks, { type: 'text/html;charset=utf-8' });
      } else {
        blob = await res.blob();
      }

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = suggestedFilename;
      setHtmlExportStatus('Stage 4/4: finalizing file and starting download...');
      setHtmlExportProgress((prev) => Math.max(prev, 96));
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      setHtmlExportProgress(100);
      setHtmlExportStatus('HTML report ready.');
    } catch (err) {
      console.error('Quick HTML export failed:', err);
      setHtmlExportStatus('HTML export failed. Auto-downloading emergency JSON backup...');
      const errorReason = err instanceof Error ? err.message : String(err || 'unknown error');
      void triggerEmergencyReproJsonDownload('html_export_failed', errorReason);
      alert('HTML export failed. GEMMAP started an automatic emergency JSON backup download (with full chat history).');
    } finally {
      if (progressTimer) window.clearInterval(progressTimer);
      window.setTimeout(() => {
        setHtmlExportLoading(false);
        setHtmlExportProgress(0);
        setHtmlExportStatus('');
      }, 900);
    }
  };

  const quickXlsxExport = async () => {
    if (!sessionId || xlsxExportLoading) return;
    setXlsxExportLoading(true);
    setXlsxExportProgress(2);
    setXlsxExportStatus('Stage 1/3: preparing workbook export request...');
    let progressTimer: number | null = null;
    try {
      progressTimer = window.setInterval(() => {
        setXlsxExportProgress((prev) => {
          if (prev >= 92) return prev;
          if (prev < 45) return prev + 5;
          if (prev < 72) return prev + 3;
          return prev + 1;
        });
      }, 260);
      setXlsxExportStatus('Stage 2/3: assembling module tables and metadata sheets...');
      const res = await fetch(`${API_URL}/download`, {
        headers: { 'X-Session-ID': sessionId! }
      });
      if (!res.ok) throw new Error('Export failed');
      setXlsxExportStatus('Stage 3/3: downloading workbook and finalizing file...');
      setXlsxExportProgress((prev) => Math.max(prev, 80));
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = "gemmap_analysis.xlsx";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      setXlsxExportProgress(100);
      setXlsxExportStatus('XLSX export ready.');
    } catch (err) {
      console.error('Quick XLSX export failed:', err);
      setXlsxExportStatus('XLSX export failed.');
      alert('XLSX export failed. Please try again.');
    } finally {
      if (progressTimer) window.clearInterval(progressTimer);
      window.setTimeout(() => {
        setXlsxExportLoading(false);
        setXlsxExportProgress(0);
        setXlsxExportStatus('');
      }, 900);
    }
  };

  // Load publication figures on demand for tabs
  const loadPublicationFigure = async (figureKey: string) => {
    if (pubFigures[figureKey]) return; // Already loaded

    try {
      const formData = new FormData();
      formData.append('img_format', 'png');
      formData.append('dpi', '150');  // Lower DPI for web preview
      formData.append('include_3d', figureKey === '3d' ? 'true' : 'false');
      formData.append('include_2d', figureKey === '2d' ? 'true' : 'false');
      formData.append('include_elbow', figureKey === 'elbow' ? 'true' : 'false');
      formData.append('include_heatmap', figureKey === 'heatmap' ? 'true' : 'false');
      formData.append('include_barplots', figureKey === 'nes' ? 'true' : 'false');
      formData.append('include_html', 'false');
      formData.append('include_json', 'false');
      formData.append('include_table', 'false');
      formData.append('data_format', 'xlsx');

      const res = await fetch(`${API_URL}/export`, {
        method: 'POST',
        body: formData,
        headers: { 'X-Session-ID': sessionId! }
      });

      // This returns a ZIP, so we can't easily extract individual figures
      // Instead, we'll generate figures server-side via a new endpoint
    } catch (err) {
      console.error('Failed to load figure:', err);
    }
  };

  const exportPublicationPackage = async () => {
    if (exportLoading) return;
    setExportLoading(true);
    setExportProgress(2);
    const stageLabels = [
      'Step 1/6: Validating export options...',
      'Step 2/6: Generating publication figures...',
      exportConfig.include_table
        ? 'Step 3/6: Building data files...'
        : 'Step 3/6: Data file export skipped by options...',
      exportConfig.include_html
        ? 'Step 4/6: Generating HTML reports (with/without chat)...'
        : 'Step 4/6: HTML report generation skipped by options...',
      exportConfig.include_json
        ? 'Step 5/6: Packaging reproducibility JSON, logs, and ZIP archive...'
        : 'Step 5/6: Packaging logs and ZIP archive...',
      'Step 6/6: Downloading export package...'
    ];
    setExportStatus(stageLabels[0]);
    let simulatedProgress = 2;
    let progressTimer: number | null = null;
    const updateStageByProgress = () => {
      const idx =
        simulatedProgress < 18 ? 0 :
        simulatedProgress < 34 ? 1 :
        simulatedProgress < 52 ? 2 :
        simulatedProgress < 70 ? 3 :
        simulatedProgress < 90 ? 4 : 5;
      setExportStatus(stageLabels[idx]);
    };
    try {
      progressTimer = window.setInterval(() => {
        if (simulatedProgress >= 92) return;
        if (simulatedProgress < 30) simulatedProgress += 3;
        else if (simulatedProgress < 60) simulatedProgress += 2;
        else simulatedProgress += 1;
        simulatedProgress = Math.min(92, simulatedProgress);
        setExportProgress(simulatedProgress);
        updateStageByProgress();
      }, 320);

      const formData = new FormData();
      formData.append('img_format', exportConfig.img_format);
      formData.append('dpi', exportConfig.dpi.toString());
      formData.append('include_3d', exportConfig.include_3d.toString());
      formData.append('include_2d', exportConfig.include_2d.toString());
      formData.append('include_elbow', exportConfig.include_elbow.toString());
      formData.append('include_heatmap', exportConfig.include_heatmap.toString());
      formData.append('include_barplots', exportConfig.include_barplots.toString());
      formData.append('include_html', exportConfig.include_html.toString());
      formData.append('include_json', exportConfig.include_json.toString());
      formData.append('include_table', exportConfig.include_table.toString());
      formData.append('data_format', exportConfig.data_format);

      const res = await fetch(`${API_URL}/export`, {
        method: 'POST',
        body: formData,
        headers: { 'X-Session-ID': sessionId! }
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText);
      }

      const disposition = res.headers.get('Content-Disposition') || '';
      const encodedMatch = disposition.match(/filename\*=UTF-8''([^;]+)/i);
      const plainMatch = disposition.match(/filename="?([^\";]+)"?/i);
      const baseFallback = String(reportName || pathFileName || 'GEMMAP_Report')
        .replace(/\.[^/.]+$/, '')
        .replace(/[^\w\-. ]+/g, '_')
        .trim()
        .replace(/\.(html?|zip)$/i, '')
        .slice(0, 120);
      const fallbackZip = `${baseFallback || 'GEMMAP_Report'}.zip`;
      let suggestedFilename = fallbackZip;
      if (encodedMatch?.[1]) {
        try {
          suggestedFilename = decodeURIComponent(encodedMatch[1]);
        } catch {
          suggestedFilename = encodedMatch[1];
        }
      } else if (plainMatch?.[1]) {
        suggestedFilename = plainMatch[1].trim();
      }
      if (!/\.zip$/i.test(suggestedFilename)) {
        suggestedFilename = fallbackZip;
      }

      simulatedProgress = Math.max(simulatedProgress, 94);
      setExportProgress(simulatedProgress);
      setExportStatus(stageLabels[5]);

      let blob: Blob;
      const contentLength = Number(res.headers.get('Content-Length') || 0);
      if (res.body && typeof res.body.getReader === 'function') {
        const reader = res.body.getReader();
        const chunks: Uint8Array[] = [];
        let received = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            chunks.push(value);
            received += value.length;
            if (contentLength > 0) {
              const pct = Math.min(100, 94 + Math.round((received / contentLength) * 6));
              simulatedProgress = Math.max(simulatedProgress, pct);
              setExportProgress(simulatedProgress);
            } else {
              simulatedProgress = Math.min(99, simulatedProgress + 1);
              setExportProgress(simulatedProgress);
            }
          }
        }
        blob = new Blob(chunks, { type: 'application/zip' });
      } else {
        blob = await res.blob();
      }

      setExportProgress(100);
      setExportStatus('Export complete. Starting download...');
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = suggestedFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      setShowExportModal(false);
    } catch (err) {
      console.error('Export failed:', err);
      setExportStatus('Export failed. Please try again.');
      const errorReason = err instanceof Error ? err.message : String(err || 'unknown error');
      void triggerEmergencyReproJsonDownload('publication_export_failed', errorReason);
      alert('Export failed. Please try again.');
    } finally {
      if (progressTimer !== null) {
        window.clearInterval(progressTimer);
      }
      setExportLoading(false);
      window.setTimeout(() => {
        setExportProgress(0);
        setExportStatus('');
      }, 1000);
    }
  };

  useEffect(() => {
    if (!sessionId) return;
    const onUnhandledError = (event: ErrorEvent) => {
      const reason = event?.message || 'Unhandled runtime error';
      void triggerEmergencyReproJsonDownload('window_error', reason);
    };
    const onUnhandledRejection = (event: PromiseRejectionEvent) => {
      let reason = 'Unhandled promise rejection';
      try {
        if (event?.reason instanceof Error) {
          reason = event.reason.message || reason;
        } else if (typeof event?.reason === 'string') {
          reason = event.reason;
        } else if (event?.reason != null) {
          reason = JSON.stringify(event.reason);
        }
      } catch {
        reason = String(event?.reason || reason);
      }
      void triggerEmergencyReproJsonDownload('unhandled_rejection', reason);
    };
    window.addEventListener('error', onUnhandledError);
    window.addEventListener('unhandledrejection', onUnhandledRejection);
    return () => {
      window.removeEventListener('error', onUnhandledError);
      window.removeEventListener('unhandledrejection', onUnhandledRejection);
    };
  }, [sessionId, reportName, pathFileName]);

  // --- Views ---

  if (view === 'landing') {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex flex-col relative overflow-hidden" style={{ fontFamily: "'Space Grotesk', 'Sora', sans-serif" }}>
        <div className="absolute -top-16 left-0 w-72 h-72 rounded-full bg-cyan-700/15 blur-[100px] pointer-events-none" />
        <div className="absolute top-0 right-0 w-80 h-80 rounded-full bg-blue-900/20 blur-[110px] pointer-events-none" />

        <nav className="max-w-5xl mx-auto w-full px-6 py-6 flex items-center justify-between relative z-10">
          <div className="flex items-center gap-3">
            <img
              src={APP_ICON_URL}
              alt="GEMMAP"
              onError={withIconFallback}
              className="w-20 h-20 sm:w-24 sm:h-24 object-contain animate-gem-sparkle"
            />
            <div className="font-bold text-2xl sm:text-3xl tracking-tight text-slate-100">GEMMAP</div>
          </div>
          <a
            href="https://github.com/arashabadi/gemmap"
            target="_blank"
            rel="noreferrer"
            className="px-4 py-2 text-sm font-medium text-slate-300 border border-slate-700 rounded-lg hover:border-slate-500 transition-colors flex items-center gap-2"
          >
            <Github className="w-4 h-4" /> Source
          </a>
        </nav>

        <main className="flex-1 max-w-5xl mx-auto w-full px-6 pb-14 relative z-10">
          <section className="pt-8 sm:pt-12 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-7 text-center lg:text-left">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-950/60 border border-cyan-700/40 text-cyan-300 text-[11px] font-semibold mb-5">
                <Terminal className="w-3 h-3" /> v{APP_VERSION}
              </div>
              <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-slate-100 leading-[1.08] max-w-3xl mx-auto lg:mx-0">
                From Redundancy Reduction to Hypothesis Generation
              </h1>
              <p className="text-slate-300 text-base sm:text-lg mt-5 leading-relaxed max-w-2xl mx-auto lg:mx-0">
                Upload your pathway tables (with optional DEG data), let Jaccard-MDS geometry and LLM agents map enriched programs into modules, explore interactive 3D representations to surface core biology, and export clean, publication-ready outputs.
              </p>
              <div className="flex flex-wrap justify-center lg:justify-start gap-3 mt-8">
                <button
                  onClick={() => setView('app')}
                  className="px-7 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-slate-950 font-bold rounded-xl transition-all flex items-center gap-2 shadow-lg shadow-cyan-900/30"
                >
                  <Play className="w-4 h-4 fill-current" /> Start Analysis
                </button>
                <a
                  href="https://github.com/arashabadi/gemmap"
                  target="_blank"
                  rel="noreferrer"
                  className="px-7 py-3 border border-slate-700 hover:border-slate-500 text-slate-300 font-semibold rounded-xl transition-colors flex items-center gap-2 bg-slate-900/40"
                >
                  <BookOpen className="w-4 h-4" /> Documentation
                </a>
              </div>
            </div>

            <div className="lg:col-span-5">
              <section className="bg-slate-900/45 border border-slate-800 rounded-2xl p-4 sm:p-6">
                <h2 className="text-lg sm:text-xl font-bold text-slate-100 tracking-tight mb-4 text-center lg:text-left">How It Works</h2>
                <WorkflowSchematic />
              </section>
            </div>
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans flex flex-col">
      {/* Top Bar */}
      <div className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => setView('landing')}>
          <img
            src={APP_ICON_URL}
            alt="GEMMAP"
            onError={withIconFallback}
            className="w-8 h-8 object-contain animate-gem-sparkle"
          />
          <span className="font-bold text-lg tracking-tight">GEMMAP</span>
        </div>
        {step === 'dashboard' && (
          <div className="flex flex-col items-end gap-1">
            <div className="flex gap-2">
              <button onClick={() => setStep('upload')} className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-400 hover:text-white transition-colors">
                <RotateCcw className="w-4 h-4" /> Reset
              </button>
              <button
                onClick={quickHtmlExport}
                disabled={htmlExportLoading}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-slate-800 hover:bg-slate-700 text-white rounded-lg border border-slate-700 transition-all disabled:opacity-70"
              >
                <FileText className="w-4 h-4" /> {htmlExportLoading ? `Exporting... ${Math.max(1, Math.min(100, Math.round(htmlExportProgress)))}%` : 'HTML Report'}
              </button>
              <button
                onClick={() => {
                  setIssueReportCopyStatus('');
                  setIssueSummaryInput('');
                  setShowIssueReportPrompt(true);
                }}
                disabled={issueReportLoading}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-slate-800 hover:bg-slate-700 text-white rounded-lg border border-slate-700 transition-all disabled:opacity-70"
              >
                <AlertCircle className="w-4 h-4" /> {issueReportLoading ? 'Preparing...' : 'Report Issue'}
              </button>
              <button onClick={() => setShowExportModal(true)} className="flex items-center gap-2 px-4 py-2 text-sm font-bold bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg shadow-lg shadow-cyan-900/20">
                <Package className="w-4 h-4" /> Publication Export
              </button>
            </div>
            {(htmlExportLoading || xlsxExportLoading || jsonExportLoading) && (
              <div className="flex flex-col items-end text-[11px] leading-relaxed">
                {htmlExportLoading && (
                  <div className="text-cyan-300">
                    {htmlExportStatus || 'Exporting HTML report...'} {Math.max(1, Math.min(100, Math.round(htmlExportProgress)))}%
                  </div>
                )}
                {xlsxExportLoading && (
                  <div className="text-emerald-300">
                    {xlsxExportStatus || 'Exporting workbook...'} {Math.max(1, Math.min(100, Math.round(xlsxExportProgress)))}%
                  </div>
                )}
                {jsonExportLoading && (
                  <div className="text-violet-300">
                    {jsonExportStatus || 'Exporting reproducibility JSON...'} {Math.max(1, Math.min(100, Math.round(jsonExportProgress)))}%
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex-1 p-6 overflow-auto relative">
        {/* Background Gradients */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-900/20 rounded-full blur-[100px] pointer-events-none"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-900/10 rounded-full blur-[100px] pointer-events-none"></div>

        <div className="max-w-7xl mx-auto relative z-10">

          {step === 'upload' && (
            <div className="max-w-4xl mx-auto mt-12 animate-in fade-in slide-in-from-bottom-8 duration-700">
              <div className="mb-8 flex flex-col sm:flex-row items-center justify-between gap-3">
                <h1 className="text-3xl font-bold text-white text-center sm:text-left">Upload Your Data</h1>
                <button
                  onClick={triggerReproReplayPicker}
                  disabled={replayLoading}
                  className="flex items-center gap-2 px-4 py-2 text-sm font-semibold bg-slate-900 hover:bg-slate-800 text-cyan-200 rounded-lg border border-cyan-700/40 transition-all disabled:opacity-60"
                >
                  <RotateCcw className="w-4 h-4" /> {replayLoading ? 'Reproducing...' : 'Reproduce with HTML or JSON'}
                </button>
                <input
                  ref={replayManifestInputRef}
                  type="file"
                  accept=".json,.html,.htm,application/json,text/html"
                  onChange={handleReproReplayFile}
                  className="hidden"
                />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Pathway Data Upload */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-8 h-8 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                      <Layers className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-white">Pathway Data</h2>
                      <span className="text-xs text-cyan-400">Required</span>
                    </div>
                  </div>

                  {/* Upload Area */}
                  {/* Pathway Upload Area */}
                  {pathStatus === 'none' ? (
                    <div className="relative mb-4">
                      <input type="file" onChange={handlePathFileSelect} className="absolute inset-0 opacity-0 cursor-pointer z-10" accept=".csv, .xlsx, .xls, .txt, .tsv" />
                      <div className="border-2 border-dashed border-slate-700 hover:border-cyan-500/50 rounded-xl p-6 text-center transition-colors bg-slate-950/50">
                        <UploadCloud className="w-10 h-10 text-cyan-400 mx-auto mb-3" />
                        <p className="text-sm text-white font-medium">Drop Pathway CSV here</p>
                        <p className="text-[10px] text-slate-500 mt-1">supports .xlsx, .xls, .txt, .tsv (.csv preferred)</p>
                      </div>
                    </div>
                  ) : pathStatus === 'selected' ? (
                    <div className="mb-4 bg-cyan-950/20 border border-cyan-500/30 rounded-xl p-4 animate-in fade-in slide-in-from-top-2">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-cyan-400" />
                          <span className="text-sm text-cyan-300 font-medium">{pathFile?.name}</span>
                        </div>
                        <button
                          onClick={() => {
                            setPathStatus('none');
                            setPathFile(null);
                            setPathSignificance(null);
                          }}
                          className="text-xs text-slate-500 hover:text-white"
                        >
                          Cancel
                        </button>
                      </div>

                      {/* Pathway Column Mapping */}
                      <div className="space-y-3 mb-4">
                        <p className="text-[10px] uppercase text-cyan-500/70 font-bold">Map Pathway Columns</p>
                        <div>
                          <label className="block text-[10px] uppercase text-slate-500 font-bold mb-1">Pathway Name</label>
                          <select value={pathCol} onChange={e => setPathCol(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs focus:border-cyan-500 outline-none">
                            {columns.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <label className="block text-[10px] uppercase text-slate-500 font-bold mb-1">Gene List</label>
                          <select value={geneCol} onChange={e => setGeneCol(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs focus:border-cyan-500 outline-none">
                            {columns.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <label className="block text-[10px] uppercase text-slate-500 font-bold mb-1">NES Score</label>
                          <select value={scoreCol} onChange={e => setScoreCol(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs focus:border-cyan-500 outline-none">
                            <option value="null">Select NES Column...</option>
                            {columns.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <label className="block text-[10px] uppercase text-slate-500 font-bold mb-1">padj (Optional)</label>
                          <select value={pathPadjCol} onChange={e => setPathPadjCol(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs focus:border-cyan-500 outline-none">
                            <option value="null">No padj filtering</option>
                            {columns.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                          <p className="text-[9px] text-slate-600 mt-1">
                            If provided, pathways with padj &ge; 0.05 are removed automatically.
                          </p>
                        </div>
                      </div>
                      {scoreCol === 'null' && <p className="text-[10px] text-red-400 mb-2 font-medium">Please select NES column.</p>}

                      <button
                        onClick={confirmPathUpload}
                        disabled={scoreCol === 'null'}
                        className="w-full py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold rounded-lg text-xs shadow-lg shadow-cyan-900/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <Target className="w-3 h-3" /> Confirm & Upload
                      </button>
                    </div>
                  ) : pathStatus === 'uploaded' ? (
                    <div className="mb-4 bg-cyan-950/20 border border-cyan-500/30 rounded-xl p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                          <span className="text-sm text-cyan-300 font-medium">{pathFileName}</span>
                        </div>
                          <button
                            onClick={() => {
                              setSessionId(null);
                              setPathFileName(null);
                              setPathStatus('none');
                              setPathSignificance(null);
                            }}
                            className="text-xs text-red-400 hover:text-red-300"
                          >
                            Remove
                          </button>
                      </div>
                      {pathSignificance?.checked ? (
                        pathSignificance.all_significant ? (
                          <div className="text-xs text-green-300 flex items-center gap-2">
                            <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                            All {pathSignificance.n_total.toLocaleString()} pathways are significant ({pathSignificance.column} &lt; {pathSignificance.threshold}).
                          </div>
                        ) : (
                          <div className="text-xs text-amber-300">
                            {pathSignificance.n_removed.toLocaleString()} pathways were not significant ({pathSignificance.column} &ge; {pathSignificance.threshold}) and were removed automatically.
                            <span className="text-cyan-300"> {pathSignificance.n_kept.toLocaleString()} kept.</span>
                          </div>
                        )
                      ) : (
                        <div className="text-xs text-slate-400">
                          File uploaded successfully. padj check skipped (optional column not provided).
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="mb-4 bg-red-950/20 border border-red-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 text-red-400" />
                        <span className="text-xs text-red-400">Upload failed - check file format</span>
                        <button
                          onClick={() => {
                            setPathStatus('none');
                            setPathFile(null);
                            setPathSignificance(null);
                          }}
                          className="ml-auto text-xs text-slate-500 hover:text-white"
                        >
                          Retry
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Expected Format Table (Only shown if no session) */}
                  {!sessionId && (
                    <div className="bg-slate-950 rounded-lg p-3 border border-slate-800">
                      <p className="text-[10px] text-slate-500 uppercase font-bold mb-2">Expected Format</p>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-slate-800">
                              <th className="text-left py-1 px-2 text-cyan-400 font-mono">pathway</th>
                              <th className="text-left py-1 px-2 text-cyan-400 font-mono">padj</th>
                              <th className="text-left py-1 px-2 text-cyan-400 font-mono">NES</th>
                              <th className="text-left py-1 px-2 text-cyan-400 font-mono">leadingEdge</th>
                            </tr>
                          </thead>
                          <tbody className="text-slate-400">
                            <tr className="border-b border-slate-800/50">
                              <td className="py-1 px-2">GO_CELL_CYCLE</td>
                              <td className="py-1 px-2">0.004</td>
                              <td className="py-1 px-2">1.85</td>
                              <td className="py-1 px-2">CDK1,CCNB1,...</td>
                            </tr>
                            <tr className="border-b border-slate-800/50">
                              <td className="py-1 px-2">KEGG_APOPTOSIS</td>
                              <td className="py-1 px-2">0.021</td>
                              <td className="py-1 px-2">-1.42</td>
                              <td className="py-1 px-2">CASP3,BAX,...</td>
                            </tr>
                            <tr>
                              <td className="py-1 px-2 text-slate-600">...</td>
                              <td className="py-1 px-2 text-slate-600">...</td>
                              <td className="py-1 px-2 text-slate-600">...</td>
                              <td className="py-1 px-2 text-slate-600">...</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                      <p className="text-[9px] text-slate-600 mt-2">
                        Genes can be comma, pipe, or semicolon separated
                      </p>
                    </div>
                  )}
                </div>

                {/* DEG Data Upload */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-8 h-8 bg-orange-500/20 rounded-lg flex items-center justify-center">
                      <Target className="w-4 h-4 text-orange-400" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-white">DEG Data (Optional)</h2>
                      <span className="text-xs text-slate-500">Optional</span>
                    </div>
                  </div>

                  {/* Upload Area or Status */}
                  {degStatus === 'none' ? (
                    <>
                      <div className="relative mb-4">
                        <input
                          type="file"
                          onChange={handleDegFileSelect}
                          accept=".csv, .xlsx, .xls, .txt, .tsv"
                          className="absolute inset-0 opacity-0 cursor-pointer z-10"
                          disabled={!sessionId}
                        />
                        <div className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors bg-slate-950/50 ${sessionId ? 'border-orange-500/50 hover:border-orange-400' : 'border-slate-800 opacity-50'}`}>
                          <UploadCloud className="w-10 h-10 text-orange-400/50 mx-auto mb-3" />
                          <p className="text-sm text-slate-400 font-medium">
                            {sessionId ? 'Drop DEG CSV here' : 'Upload pathway data first'}
                          </p>
                          <p className="text-[10px] text-slate-600 mt-1">supports .xlsx, .xls, .txt, .tsv (.csv preferred)</p>
                        </div>
                      </div>

                      {/* Expected Format Table */}
                      <div className="bg-slate-950 rounded-lg p-3 border border-slate-800 mt-4">
                        <p className="text-[10px] text-slate-500 uppercase font-bold mb-2">Expected Format</p>
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="border-b border-slate-800">
                                <th className="text-left py-1 px-2 text-orange-400 font-mono">gene</th>
                                <th className="text-left py-1 px-2 text-orange-400 font-mono">p_value</th>
                                <th className="text-left py-1 px-2 text-orange-400 font-mono">log2FC</th>
                                <th className="text-left py-1 px-2 text-orange-400 font-mono">fdr</th>
                              </tr>
                            </thead>
                            <tbody className="text-slate-400">
                              <tr className="border-b border-slate-800/50">
                                <td className="py-1 px-2">TP53</td>
                                <td className="py-1 px-2">0.0002</td>
                                <td className="py-1 px-2">2.45</td>
                                <td className="py-1 px-2">0.001</td>
                              </tr>
                              <tr className="border-b border-slate-800/50">
                                <td className="py-1 px-2">MDM2</td>
                                <td className="py-1 px-2">0.012</td>
                                <td className="py-1 px-2">-1.82</td>
                                <td className="py-1 px-2">0.032</td>
                              </tr>
                              <tr>
                                <td className="py-1 px-2 text-slate-600">...</td>
                                <td className="py-1 px-2 text-slate-600">...</td>
                                <td className="py-1 px-2 text-slate-600">...</td>
                                <td className="py-1 px-2 text-slate-600">...</td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                        <p className="text-[9px] text-slate-600 mt-2">
                          DEG threshold: FDR &lt; 0.05 AND |log2FC| &gt; 0.25 (optional nominal p_value fallback)
                        </p>
                        <div className="mt-3 pt-2 border-t border-slate-800 space-y-1">
                          <p className="text-[10px] text-slate-400 font-semibold">Adds expression evidence across analysis outputs</p>
                          <p className="text-[10px] text-slate-500">1. Powers ranked-gene mountain plots and pathway DEG-hit tracing.</p>
                          <p className="text-[10px] text-slate-500">2. Adds DEG-overlap metrics to module cards, tables, and exports.</p>
                          <p className="text-[10px] text-slate-500">3. Improves Analyze Relevance evidence with up/down DEG hit counts.</p>
                        </div>
                      </div>
                    </>
                  ) : degStatus === 'selected' ? (
                    <div className="mb-4 bg-orange-950/20 border border-orange-500/30 rounded-xl p-4 animate-in fade-in slide-in-from-top-2">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-orange-400" />
                          <span className="text-sm text-orange-300 font-medium">{degFile?.name}</span>
                        </div>
                        <button
                          onClick={() => {
                            setDegStatus('none');
                            setDegFile(null);
                            setDegNeedsConfirmation(null);
                            setDegSignificance(null);
                            setDegErrorMessage(null);
                          }}
                          className="text-xs text-slate-500 hover:text-white"
                        >
                          Cancel
                        </button>
                      </div>

                      {/* DEG Column Mapping */}
                      <div className="space-y-3 mb-4">
                        <p className="text-[10px] uppercase text-orange-500/70 font-bold">Map DEG Columns</p>
                        <div>
                          <label className="block text-[10px] text-slate-500 font-bold mb-1">Gene Symbol</label>
                          <select value={degConfig.gene_col} onChange={e => setDegConfig({ ...degConfig, gene_col: e.target.value })} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs focus:border-orange-500 outline-none">
                            {degColumns.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <label className="block text-[10px] text-slate-500 font-bold mb-1">p_value (Optional)</label>
                          <select value={degPValueCol} onChange={e => setDegPValueCol(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs focus:border-orange-500 outline-none">
                            <option value="null">No p_value column</option>
                            {degColumns.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <label className="block text-[10px] text-slate-500 font-bold mb-1">Log2 Fold Change</label>
                          <select value={degConfig.lfc_col} onChange={e => setDegConfig({ ...degConfig, lfc_col: e.target.value })} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs focus:border-orange-500 outline-none">
                            {degColumns.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <label className="block text-[10px] text-slate-500 font-bold mb-1">FDR / padj</label>
                          <select value={degConfig.padj_col} onChange={e => setDegConfig({ ...degConfig, padj_col: e.target.value })} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs focus:border-orange-500 outline-none">
                            {degColumns.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                      </div>

                      {degNeedsConfirmation && (
                        <div className="mb-4 rounded-lg border border-amber-500/40 bg-amber-950/20 p-3">
                          <p className="text-[11px] text-amber-200 leading-relaxed">
                            {degNeedsConfirmation.message}
                          </p>
                          <p className="text-[10px] text-amber-300/80 mt-2">
                            Maybe nominal p-values are still useful for hypothesis generation.
                          </p>
                          <div className="flex gap-2 mt-3">
                            <button
                              onClick={() => confirmDegUpload(true, true)}
                              className="flex-1 py-2 text-[11px] font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700"
                            >
                              Continue with p_value &lt; 0.05
                            </button>
                            <button
                              onClick={() => confirmDegUpload(false, true)}
                              className="flex-1 py-2 text-[11px] font-bold text-white bg-amber-600 hover:bg-amber-500 rounded-lg"
                            >
                              Keep FDR and Continue
                            </button>
                          </div>
                        </div>
                      )}

                      <button
                        onClick={() => confirmDegUpload()}
                        className="w-full py-2.5 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 text-white font-bold rounded-lg text-xs shadow-lg shadow-orange-900/20 transition-all flex items-center justify-center gap-2"
                      >
                        <Target className="w-3 h-3" /> Confirm & Upload
                      </button>
                    </div>
                  ) : degStatus === 'uploaded' && degInfo ? (
                    <div className="mb-4 bg-orange-950/20 border border-orange-500/30 rounded-xl p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4 text-orange-400" />
                          <span className="text-sm text-orange-300 font-medium">{degFile?.name}</span>
                        </div>
                        <button onClick={removeDeg} className="text-xs text-red-400 hover:text-red-300">Remove</button>
                      </div>
                      <div className="text-xs text-slate-400">
                        {degInfo.n_genes.toLocaleString()} genes • <span className="text-orange-400 font-bold">{degInfo.n_degs.toLocaleString()} DEGs</span> detected
                      </div>
                      {degSignificance && (
                        <div className="mt-2 text-xs">
                          {degSignificance.all_significant && !degSignificance.using_nominal_p ? (
                            <div className="text-green-300 flex items-center gap-1.5">
                              <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                              All genes pass {degSignificance.checked_col} &lt; {degSignificance.threshold}.
                            </div>
                          ) : degSignificance.using_nominal_p ? (
                            <div className="text-amber-300">
                              Continued with nominal {degSignificance.effective_col} &lt; {degSignificance.effective_threshold} for hypothesis generation.
                            </div>
                          ) : (
                            <div className="text-amber-300">
                              {degSignificance.n_non_significant.toLocaleString()} genes are not significant at {degSignificance.checked_col} &lt; {degSignificance.threshold}.
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="mb-4 bg-red-950/20 border border-red-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 text-red-400" />
                        <span className="text-xs text-red-400">{degErrorMessage || 'Upload failed - check column names'}</span>
                        <button
                          onClick={() => {
                            setDegStatus('none');
                            setDegFile(null);
                            setDegNeedsConfirmation(null);
                            setDegSignificance(null);
                            setDegErrorMessage(null);
                          }}
                          className="ml-auto text-xs text-slate-500 hover:text-white"
                        >
                          Retry
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Demo Data Button */}
              <div className="text-center mt-8">
                <button onClick={loadDemo} className="text-slate-500 hover:text-cyan-400 text-sm font-medium transition-colors flex items-center justify-center gap-2 mx-auto">
                  <Sparkles className="w-4 h-4" /> Use Synthetic Demo Data Instead
                </button>
              </div>

              {/* Continue Button if pathway data uploaded */}
              {sessionId && (
                <div className="mt-8 text-center animate-in fade-in">
                  <button
                    onClick={() => setStep('analyze')}
                    className="px-8 py-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/20 flex items-center justify-center gap-2 mx-auto transition-all"
                  >
                    Continue to Analysis <ChevronRight className="w-4 h-4" />
                  </button>
                  <p className="text-xs text-slate-500 mt-2">
                    {degStatus === 'uploaded' ? 'Pathway + DEG data ready' : 'Pathway data ready (DEG optional)'}
                  </p>
                </div>
              )}
            </div>
          )}

          {step === 'analyze' && (
            <div className="max-w-5xl mx-auto animate-in fade-in zoom-in-95 duration-500">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h1 className="text-2xl font-bold text-white">Choose Your Analysis Mode</h1>
                  <p className="text-slate-400 text-sm mt-1">Configure your data columns and select how to run the analysis</p>
                </div>
                <button onClick={() => setStep('upload')} className="text-slate-500 hover:text-white text-sm flex items-center gap-1">
                  <RotateCcw className="w-3 h-3" /> Back to Upload
                </button>
              </div>

              {/* Status badges */}
              <div className="flex gap-2 mb-6">
                <span className="text-xs bg-cyan-950/50 text-cyan-400 px-3 py-1 rounded-full border border-cyan-500/30">
                  Pathway data loaded
                </span>
                {degStatus === 'uploaded' && degInfo && (
                  <span className="text-xs bg-orange-950/50 text-orange-400 px-3 py-1 rounded-full border border-orange-500/30">
                    {degInfo.n_degs} DEGs loaded
                  </span>
                )}
              </div>

              <div className="mb-6 bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                  <div className="lg:col-span-5">
                    <p className="text-[10px] uppercase font-bold text-slate-500 mb-2">NES Direction Filter (Optional)</p>
                    <div className="flex flex-wrap gap-2">
                      {([
                        { value: 'all', label: 'All NES' },
                        { value: 'positive', label: 'Positive NES Only' },
                        { value: 'negative', label: 'Negative NES Only' }
                      ] as const).map(option => (
                        <button
                          key={option.value}
                          onClick={() => setNesDirection(option.value)}
                          className={`px-3 py-2 text-xs font-medium rounded-lg border transition-all ${nesDirection === option.value
                            ? 'bg-cyan-600 border-cyan-500 text-white'
                            : 'bg-slate-950 border-slate-700 text-slate-400 hover:border-cyan-500/40'
                            }`}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                    <p className="text-[10px] text-slate-500 mt-2">
                      This applies to elbow, manual run, and Autopilot.
                    </p>
                  </div>

                  <div className="lg:col-span-7">
                    <p className="text-[10px] uppercase font-bold text-slate-500 mb-2">HTML Report Filename (Optional)</p>
                    <input
                      type="text"
                      value={reportName}
                      onChange={e => setReportName(e.target.value)}
                      placeholder={`Default: ${pathFileName ? pathFileName.replace(/\.[^/.]+$/, '') : 'uploaded file name'}`}
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-slate-200 focus:border-cyan-500 outline-none"
                    />
                    <p className="text-[10px] text-slate-500 mt-2">
                      If left empty, the uploaded pathway file name is used.
                    </p>
                  </div>
                </div>
              </div>

              {/* Column Mapping Removed - Moved to Upload Step */}

              {/* Two Column Layout: Autopilot vs Manual */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* LEFT: AUTO-PILOT MODE */}
                <div className="bg-gradient-to-br from-purple-950/50 to-pink-950/30 border-2 border-purple-500/40 rounded-2xl p-6 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-3xl"></div>

                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-purple-500/20 rounded-xl flex items-center justify-center">
                      <Zap className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white">Autopilot</h2>
                      <p className="text-purple-300 text-sm">AI analyzes everything & decides k</p>
                    </div>
                  </div>

                  <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                    Our AI will compute the elbow plot, find the
                    <span className="text-purple-300 font-medium"> optimal</span> number of modules, run the analysis,
                    and <span className="text-purple-300 font-medium">generate AI module names</span> automatically.
                  </p>

                  {/* API Key Section */}
                  <div className="space-y-4 mb-6">
                    <div>
                      <label className="block text-xs font-bold text-purple-300 uppercase mb-2">AI Provider</label>
                      <div className="flex gap-2">
                        {(['openai', 'gemini', 'claude'] as const).map(p => (
                          <button
                            key={p}
                            onClick={() => setAgentProvider(p)}
                            className={`flex-1 py-2 px-3 text-xs font-medium rounded-lg border transition-all ${agentProvider === p
                              ? 'bg-purple-600 border-purple-400 text-white'
                              : 'bg-slate-900/50 border-slate-700 text-slate-400 hover:border-purple-500/50'
                              }`}
                          >
                            {p === 'openai' ? 'ChatGPT (OpenAI)' : p === 'gemini' ? 'Gemini (Google)' : 'Claude (Anthropic)'}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-lg border border-purple-500/30 bg-slate-900/70 p-3 space-y-2">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-xs font-bold text-purple-200 uppercase">Autopilot Turbo</div>
                          <div className="text-[11px] text-slate-400">
                            Model tier: <span className="text-slate-200">{turboModelLabel}</span>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => setAgentTurbo(prev => !prev)}
                          className={`px-3 py-1.5 text-[11px] font-semibold rounded-md border transition-all ${agentTurbo
                            ? 'bg-amber-500/20 border-amber-400/60 text-amber-100'
                            : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-purple-500/40'
                            }`}
                        >
                          {agentTurbo ? 'Enabled' : 'Enable'}
                        </button>
                      </div>
                      {effectiveAgentTurbo && (
                        <div className="text-[11px] text-amber-200 bg-amber-950/30 border border-amber-700/40 rounded-md px-2 py-1.5">
                          Turbo uses a higher-cost model tier and can increase API spend.
                        </div>
                      )}
                      {agentTurbo && agentProvider !== 'gemini' && (
                        <div className="text-[11px] text-slate-400">
                          Turbo currently upgrades Gemini only; this provider stays on its default tier.
                        </div>
                      )}
                    </div>

                    <div>
                      <label className="block text-xs font-bold text-purple-300 uppercase mb-2">
                        API Key <span className="text-slate-500 font-normal">(required, never stored)</span>
                      </label>
                      <input
                        type="password"
                        value={apiKey}
                        onChange={e => setApiKey(e.target.value)}
                        placeholder={agentProvider === 'openai' ? 'sk-...' : agentProvider === 'gemini' ? 'AIza...' : 'sk-ant-...'}
                        className="w-full bg-slate-900/70 border border-purple-500/30 rounded-lg px-3 py-3 text-sm focus:border-purple-400 outline-none placeholder-slate-600"
                      />
                    </div>
                  </div>

                  {/* Autopilot Button */}
                  <button
                    onClick={runAutoAnalyze}
                    disabled={!apiKey || autoAnalyzeLoading || scoreCol === 'null'}
                    className={`w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-bold rounded-xl shadow-lg shadow-purple-900/30 flex items-center justify-center gap-2 transition-all transform active:scale-95 ${!apiKey || autoAnalyzeLoading || scoreCol === 'null' ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {autoAnalyzeLoading ? (
                      <><Loader2 className="w-5 h-5 animate-spin" /> AI is analyzing...</>
                    ) : (
                      <><BrainCircuit className="w-5 h-5" /> {effectiveAgentTurbo ? 'Run Autopilot Turbo' : 'Run Autopilot'}</>
                    )}
                  </button>

                  {!apiKey && (
                    <p className="text-xs text-purple-400/70 text-center mt-3">
                      Enter your API key above to enable Autopilot
                    </p>
                  )}
                </div>

                {/* RIGHT: MANUAL MODE */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-cyan-500/20 rounded-xl flex items-center justify-center">
                      <TrendingDown className="w-6 h-6 text-cyan-400" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white">Manual Mode</h2>
                      <p className="text-cyan-300 text-sm">You pick the number of modules</p>
                    </div>
                  </div>

                  <p className="text-slate-400 text-sm mb-6 leading-relaxed flex-1">
                    Compute the elbow and silhouette plots, review the metrics,
                    and manually choose your optimal <span className="text-cyan-400 font-medium">k</span> value.
                    Perfect for fine-tuning module counts based on your domain expertise.
                  </p>

                  {/* Find Optimal Modules Button */}
                  <button
                    onClick={fetchElbowData}
                    disabled={scoreCol === 'null' || elbowLoading}
                    className={`w-full py-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/20 flex items-center justify-center gap-2 transition-all transform active:scale-95 ${scoreCol === 'null' || elbowLoading ? 'opacity-50 cursor-not-allowed grayscale' : ''}`}
                  >
                    {elbowLoading ? (
                      <><Loader2 className="w-4 h-4 animate-spin" /> Computing Metrics...</>
                    ) : (
                      <><TrendingDown className="w-4 h-4" /> Find Optimal Modules →</>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}

          {step === 'elbow' && elbowData && (
            <div className="max-w-6xl mx-auto animate-in fade-in zoom-in-95 duration-500">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h1 className="text-2xl font-bold text-white">Manual Module Selection</h1>
                  <p className="text-slate-400 text-sm mt-1">Review the metrics and choose your k value</p>
                </div>
                <button onClick={() => setStep('analyze')} className="text-slate-500 hover:text-white text-sm flex items-center gap-1">
                  <RotateCcw className="w-3 h-3" /> Back to Mode Selection
                </button>
              </div>

              <div className="mb-4 bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="text-[10px] uppercase font-bold text-slate-500 mb-1">NES Direction</p>
                    <p className="text-xs text-slate-400">Change filter and recompute elbow if needed.</p>
                  </div>
                  <div className="flex gap-2">
                    {([
                      { value: 'all', label: 'All' },
                      { value: 'positive', label: 'Positive' },
                      { value: 'negative', label: 'Negative' }
                    ] as const).map(option => (
                      <button
                        key={option.value}
                        onClick={() => setNesDirection(option.value)}
                        className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${nesDirection === option.value
                          ? 'bg-cyan-600 border-cyan-500 text-white'
                          : 'bg-slate-950 border-slate-700 text-slate-400 hover:border-cyan-500/40'
                          }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={fetchElbowData}
                    className="px-3 py-1.5 text-xs font-medium rounded-lg border border-cyan-500/40 text-cyan-300 hover:bg-cyan-950/30"
                  >
                    Recompute Elbow
                  </button>
                </div>
              </div>

              {/* Manual Selection Layout: keep metrics visible while tuning k */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
                <div className="lg:col-span-5 space-y-4">
                  <div className="bg-slate-900 border border-slate-700 rounded-2xl p-5">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <span className="text-xs text-slate-500 uppercase font-bold">Suggestion</span>
                        <div className="flex items-center gap-2 mt-2">
                          <span className="text-3xl font-bold text-cyan-400">k = {elbowData.elbow.optimal_k}</span>
                          <span className={`text-xs px-2 py-0.5 rounded-full ${elbowData.elbow.confidence === 'high' ? 'bg-green-500/20 text-green-400' :
                            elbowData.elbow.confidence === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                              'bg-red-500/20 text-red-400'
                            }`}>
                            {elbowData.elbow.confidence}
                          </span>
                        </div>
                        <div className="text-[10px] text-slate-500 mt-2">
                          Elbow: k={elbowData.elbow.elbow_k} • Silhouette: k={elbowData.elbow.silhouette_k}
                        </div>
                      </div>
                      <div className="text-right text-[10px] text-slate-500">
                        <div className="uppercase font-bold tracking-wide">Selected</div>
                        <div className="text-cyan-300 font-mono font-bold text-lg">k={nClusters}</div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                    <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Your Choice</label>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min="2"
                        max="15"
                        value={nClusters}
                        onChange={e => setNClusters(parseInt(e.target.value))}
                        className="flex-1 accent-cyan-500 h-2"
                      />
                      <div className="w-14 h-14 bg-cyan-500/20 rounded-xl flex items-center justify-center border border-cyan-500/30">
                        <span className="font-mono text-2xl text-cyan-400 font-bold">{nClusters}</span>
                      </div>
                    </div>
                    <div className="flex gap-2 mt-3">
                      <button
                        onClick={() => setNClusters(elbowData.elbow.elbow_k)}
                        className="flex-1 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700"
                      >
                        Elbow (k={elbowData.elbow.elbow_k})
                      </button>
                      <button
                        onClick={() => setNClusters(elbowData.elbow.silhouette_k)}
                        className="flex-1 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700"
                      >
                        Silhouette (k={elbowData.elbow.silhouette_k})
                      </button>
                    </div>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                    <div className="flex flex-col gap-3">
                      <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col">
                        <div className="flex items-center justify-between gap-2">
                          <h3 className="text-white font-bold text-sm">Module Mapping</h3>
                          <span className="text-[10px] text-slate-500">no AI</span>
                        </div>
                        <p className="text-[10px] text-slate-500 mt-1 flex-1">
                          Pure module mapping and plots with your selected k.
                        </p>
                        <button
                          onClick={runAnalysis}
                          className="mt-3 w-full py-3.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/20 flex items-center justify-center gap-2 transition-all transform active:scale-95"
                        >
                          <Play className="w-4 h-4 fill-current" />
                          Run (k={nClusters})
                        </button>
                      </div>

                      <div className="bg-gradient-to-br from-purple-950/30 to-pink-950/20 border border-purple-500/30 rounded-xl p-4 flex flex-col">
                        <div className="flex items-center justify-between gap-2">
                          <h3 className="text-white font-bold text-sm">Semi-autopilot</h3>
                          <span className="text-[10px] text-purple-300">AI annotate</span>
                        </div>
                        <p className="text-[10px] text-slate-400 mt-1">
                          Run with your k choice and generate module names.
                        </p>
                        <div className="mt-2 flex gap-2">
                          <select
                            value={agentProvider}
                            onChange={e => setAgentProvider(e.target.value as 'openai' | 'gemini' | 'claude')}
                            className="w-44 bg-slate-900/70 border border-purple-500/30 rounded-lg px-2.5 py-2 text-xs focus:border-purple-400 outline-none text-slate-300"
                          >
                            <option value="openai">OpenAI</option>
                            <option value="gemini">Gemini</option>
                            <option value="claude">Claude</option>
                          </select>
                          <input
                            type="password"
                            value={apiKey}
                            onChange={e => setApiKey(e.target.value)}
                            placeholder="API key"
                            className="flex-1 bg-slate-900/70 border border-purple-500/30 rounded-lg px-2.5 py-2 text-xs focus:border-purple-400 outline-none placeholder-slate-600"
                          />
                        </div>
                        <div className="mt-2 flex items-center justify-between gap-2">
                          <span className="text-[10px] text-slate-400">Tier: {effectiveAgentTurbo ? 'Turbo' : 'Standard'} ({turboModelLabel})</span>
                          <button
                            type="button"
                            onClick={() => setAgentTurbo(prev => !prev)}
                            className={`text-[10px] px-2 py-1 rounded border transition-all ${agentTurbo
                              ? 'border-amber-500/60 bg-amber-500/15 text-amber-200'
                              : 'border-slate-700 bg-slate-900 text-slate-300'
                              }`}
                          >
                            {agentTurbo ? 'Turbo On' : 'Turbo Off'}
                          </button>
                        </div>
                        <button
                          onClick={() => {
                            setUseAgent(true);
                            runAnalysis();
                          }}
                          disabled={!apiKey}
                          className={`mt-3 w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white text-sm font-bold rounded-xl shadow-lg shadow-purple-900/20 flex items-center justify-center gap-2 transition-all transform active:scale-95 ${!apiKey ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <BrainCircuit className="w-5 h-5" />
                          Run + Annotate
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6">
                  <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                    <TrendingDown className="w-5 h-5 text-cyan-400" />
                    Module Selection Metrics
                  </h3>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
                      <h4 className="text-slate-300 font-medium text-sm mb-3">Elbow Plot (WCSS vs k)</h4>
                      <div className="h-52 relative">
                        <svg viewBox="0 0 400 180" className="w-full h-full">
                          {[0, 1, 2, 3].map(i => (
                            <line key={i} x1="40" y1={30 + i * 40} x2="380" y2={30 + i * 40} stroke="#334155" strokeWidth="1" />
                          ))}
                          <polyline
                            fill="none"
                            stroke="#22d3ee"
                            strokeWidth="3"
                            points={elbowData.elbow.metrics.map((m: any, i: number) => {
                              const maxInertia = Math.max(...elbowData.elbow.metrics.map((x: any) => x.inertia));
                              const x = 50 + (i / (elbowData.elbow.metrics.length - 1)) * 320;
                              const y = 150 - (m.inertia / maxInertia) * 120;
                              return `${x},${y}`;
                            }).join(' ')}
                          />
                          {elbowData.elbow.metrics.map((m: any, i: number) => {
                            const maxInertia = Math.max(...elbowData.elbow.metrics.map((x: any) => x.inertia));
                            const x = 50 + (i / (elbowData.elbow.metrics.length - 1)) * 320;
                            const y = 150 - (m.inertia / maxInertia) * 120;
                            const isSelected = m.k === nClusters;
                            const isSuggested = m.k === elbowData.elbow.optimal_k;
                            return (
                              <g key={i}>
                                <circle
                                  cx={x} cy={y}
                                  r={isSelected ? 10 : isSuggested ? 7 : 5}
                                  fill={isSelected ? '#22d3ee' : isSuggested ? '#a855f7' : '#0e7490'}
                                  stroke={isSelected ? '#fff' : 'none'}
                                  strokeWidth="2"
                                />
                                <text x={x} y={170} fill="#94a3b8" fontSize="11" textAnchor="middle" fontWeight={isSelected ? 'bold' : 'normal'}>
                                  {m.k}
                                </text>
                              </g>
                            );
                          })}
                          <text x="20" y="95" fill="#64748b" fontSize="9" transform="rotate(-90, 20, 95)">WCSS</text>
                          <text x="210" y="12" fill="#64748b" fontSize="9" textAnchor="middle">Number of Modules (k)</text>
                        </svg>
                      </div>
                      <p className="text-[10px] text-slate-500 text-center">
                        Lower WCSS = tighter modules. Look for the "elbow" where improvement slows.
                      </p>
                    </div>

                    <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
                      <h4 className="text-slate-300 font-medium text-sm mb-3">Silhouette Score vs k</h4>
                      <div className="h-52 relative">
                        <svg viewBox="0 0 400 180" className="w-full h-full">
                          {[0, 1, 2, 3].map(i => (
                            <line key={i} x1="40" y1={30 + i * 40} x2="380" y2={30 + i * 40} stroke="#334155" strokeWidth="1" />
                          ))}
                          <polyline
                            fill="none"
                            stroke="#f97316"
                            strokeWidth="3"
                            points={elbowData.elbow.metrics.map((m: any, i: number) => {
                              const x = 50 + (i / (elbowData.elbow.metrics.length - 1)) * 320;
                              const y = 150 - m.silhouette * 120;
                              return `${x},${y}`;
                            }).join(' ')}
                          />
                          {elbowData.elbow.metrics.map((m: any, i: number) => {
                            const x = 50 + (i / (elbowData.elbow.metrics.length - 1)) * 320;
                            const y = 150 - m.silhouette * 120;
                            const isSelected = m.k === nClusters;
                            const isBest = m.k === elbowData.elbow.silhouette_k;
                            return (
                              <g key={i}>
                                <circle
                                  cx={x} cy={y}
                                  r={isSelected ? 10 : isBest ? 7 : 5}
                                  fill={isSelected ? '#f97316' : isBest ? '#a855f7' : '#c2410c'}
                                  stroke={isSelected ? '#fff' : 'none'}
                                  strokeWidth="2"
                                />
                                <text x={x} y={170} fill="#94a3b8" fontSize="11" textAnchor="middle" fontWeight={isSelected ? 'bold' : 'normal'}>
                                  {m.k}
                                </text>
                              </g>
                            );
                          })}
                          <text x="20" y="95" fill="#64748b" fontSize="9" transform="rotate(-90, 20, 95)">Silhouette</text>
                          <text x="210" y="12" fill="#64748b" fontSize="9" textAnchor="middle">Number of Modules (k)</text>
                        </svg>
                      </div>
                      <p className="text-[10px] text-slate-500 text-center">
                        Higher silhouette = better separated modules. Look for peaks.
                      </p>
                    </div>
                  </div>

                  <div className="mt-4 bg-slate-950 border border-slate-800 rounded-xl">
                    <button
                      onClick={() => setShowMetricsTable((prev) => !prev)}
                      className="w-full text-left px-4 py-3 text-xs font-semibold text-slate-400 hover:text-white"
                    >
                      {showMetricsTable ? 'Hide metrics table' : 'View metrics table'}
                    </button>
                    {showMetricsTable && (
                      <div className="px-4 pb-4 overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="text-slate-500 uppercase border-b border-slate-800">
                              <th className="py-2 px-3 text-left">k</th>
                              <th className="py-2 px-3 text-right">WCSS</th>
                              <th className="py-2 px-3 text-right">Silhouette</th>
                              <th className="py-2 px-3 text-right">Pathways/Module</th>
                              <th className="py-2 px-3 text-center">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {elbowData.elbow.metrics.map((m: any) => {
                              const isSelected = m.k === nClusters;
                              const totalPathways = elbowData.mds_scree?.data?.length || 150;
                              return (
                                <tr
                                  key={m.k}
                                  className={`border-b border-slate-800/50 cursor-pointer hover:bg-slate-800/50 ${isSelected ? 'bg-cyan-950/30' : ''}`}
                                  onClick={() => setNClusters(m.k)}
                                >
                                  <td className={`py-2 px-3 font-mono font-bold ${isSelected ? 'text-cyan-400' : 'text-slate-300'}`}>
                                    {m.k}
                                  </td>
                                  <td className="py-2 px-3 text-right font-mono text-slate-400">{m.inertia.toFixed(1)}</td>
                                  <td className="py-2 px-3 text-right font-mono text-orange-400">{m.silhouette.toFixed(4)}</td>
                                  <td className="py-2 px-3 text-right font-mono text-slate-500">~{Math.round(totalPathways / m.k)}</td>
                                  <td className="py-2 px-3 text-center">
                                    {m.k === elbowData.elbow.elbow_k && <span className="text-[9px] bg-cyan-900/50 text-cyan-400 px-2 py-0.5 rounded">elbow</span>}
                                    {m.k === elbowData.elbow.silhouette_k && <span className="text-[9px] bg-orange-900/50 text-orange-400 px-2 py-0.5 rounded ml-1">best sil</span>}
                                    {isSelected && <span className="text-[9px] bg-green-900/50 text-green-400 px-2 py-0.5 rounded ml-1">selected</span>}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {step === 'processing' && (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 min-h-[70vh]">
              <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-8 flex flex-col items-center justify-center text-center">
                <ProcessingMascot />
                <h2 className="mt-8 text-2xl font-bold text-white">Processing Data</h2>
                <p className="text-slate-400 mt-2">
                  {processingMode === 'auto' ? 'Autopilot workflow is executing.' : 'Manual workflow is executing.'}
                </p>
                <div className="mt-4 inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-950/30 px-3 py-1.5">
                  <Terminal className="w-4 h-4 text-cyan-300" />
                  <span className="text-xs text-cyan-100 font-mono">Elapsed: {Math.floor(processingElapsedSec / 60)}m {String(processingElapsedSec % 60).padStart(2, '0')}s</span>
                </div>
              </div>

              <div className="space-y-4">
                <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-white text-sm font-semibold">Execution Log (Numbered)</h3>
                    <span className="text-[11px] text-slate-400">{processingWorkflow.length} steps</span>
                  </div>
                  <ol className="space-y-2">
                    {processingWorkflow.map((entry) => (
                      <li key={entry.key} className="rounded-xl border border-slate-800 bg-slate-950/70 px-3 py-2">
                        <div className="flex items-center justify-between gap-3">
                          <div className="flex items-center gap-3 min-w-0">
                            <div className="w-6 h-6 rounded-md bg-slate-800 text-slate-300 text-xs font-mono flex items-center justify-center">
                              {entry.id}
                            </div>
                            <ProcessingStatusIcon status={entry.status} />
                            <div className="min-w-0">
                              <p className="text-sm text-slate-100 font-medium truncate">{entry.label}</p>
                              <p className="text-[11px] text-slate-400">{entry.note}</p>
                            </div>
                          </div>
                          <div className="flex flex-col items-end gap-1 shrink-0">
                            <span className={`px-2 py-0.5 rounded border text-[10px] uppercase tracking-wide ${processingStatusClasses(entry.status)}`}>
                              {processingStatusLabel(entry.status)}
                            </span>
                            <span className="text-[10px] text-slate-500 font-mono">
                              t+{Math.max(0, Math.floor(((entry.updatedAt || Date.now()) - (processingStartedAt || entry.updatedAt || Date.now())) / 1000))}s
                            </span>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ol>
                </div>

              </div>
            </div>
          )}

          {step === 'dashboard' && results && (
            <div className="space-y-6 animate-in fade-in duration-700">

              {/* Header Info */}
              <div className="flex justify-between items-end">
                <div>
                  <h2 className="text-2xl font-bold text-white mb-1">{dashboardHeaderTitle}</h2>
                  <p className="text-slate-400 text-sm">
                    {useAgent && agentStatus === 'complete' && <span className="text-purple-400">AI Annotations Complete • </span>}
                    {useAgent && agentStatus === 'running' && <span className="text-purple-400 animate-pulse">AI Analyzing... • </span>}
                    {dashboardHeaderNote}
                  </p>
                </div>

                <div className="flex gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
                  {resultTabs.map((t) => (
                    <button
                      key={t.id}
                      onClick={() => setActiveTab(t.id as any)}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${activeTab === t.id ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Agent Status Bar if running */}
              {useAgent && agentStatus === 'running' && (
                <div className="bg-purple-900/20 border border-purple-500/30 rounded-xl p-4 flex items-center gap-4 animate-pulse">
                  <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
                  <span className="text-purple-200 text-sm font-medium">Agents are analyzing {results.clusters.length} modules in background. You can keep exploring while it runs.</span>
                </div>
              )}

              <div className={`grid grid-cols-1 ${activeTab === '3d' ? 'lg:grid-cols-12' : 'lg:grid-cols-4'} gap-6`}>
                {/* Left: Visualization - 3D Map */}
                <div className={`${activeTab === '3d' ? 'lg:col-span-8' : 'lg:col-span-3'} ${activeTab === '3d' ? '' : 'hidden'}`}>
                  <PlotlyGraph
                    data={moduleizedPlotlyJson?.data || results.plotly_json.data}
                    layout={moduleizedPlotlyJson?.layout || results.plotly_json.layout}
                    onPointClick={handle3DMapPointClick}
                  />
                </div>

                {/* 2D Projections View */}
                <div className={`lg:col-span-3 ${activeTab === '2d' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <Layers className="w-5 h-5 text-cyan-400" /> 2D MDS Projections
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {/* Dim1 vs Dim2 */}
                      <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                        <div className="text-xs text-slate-500 font-medium mb-3">Dimension 1 vs Dimension 2</div>
                        <div className="aspect-square relative">
                          <svg viewBox="0 0 200 200" className="w-full h-full">
                            {results.scatter_data?.map((p: any, i: number) => {
                              const x = 20 + ((p.Dim1 - Math.min(...results.scatter_data.map((d: any) => d.Dim1))) /
                                (Math.max(...results.scatter_data.map((d: any) => d.Dim1)) - Math.min(...results.scatter_data.map((d: any) => d.Dim1)) + 0.001)) * 160;
                              const y = 180 - ((p.Dim2 - Math.min(...results.scatter_data.map((d: any) => d.Dim2))) /
                                (Math.max(...results.scatter_data.map((d: any) => d.Dim2)) - Math.min(...results.scatter_data.map((d: any) => d.Dim2)) + 0.001)) * 160;
                              return <circle key={i} cx={x} cy={y} r="4" fill={colorForCluster(p.Cluster)} opacity="0.7" />;
                            })}
                          </svg>
                        </div>
                      </div>
                      {/* Dim1 vs Dim3 */}
                      <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                        <div className="text-xs text-slate-500 font-medium mb-3">Dimension 1 vs Dimension 3</div>
                        <div className="aspect-square relative">
                          <svg viewBox="0 0 200 200" className="w-full h-full">
                            {results.scatter_data?.map((p: any, i: number) => {
                              const x = 20 + ((p.Dim1 - Math.min(...results.scatter_data.map((d: any) => d.Dim1))) /
                                (Math.max(...results.scatter_data.map((d: any) => d.Dim1)) - Math.min(...results.scatter_data.map((d: any) => d.Dim1)) + 0.001)) * 160;
                              const y = 180 - ((p.Dim3 - Math.min(...results.scatter_data.map((d: any) => d.Dim3))) /
                                (Math.max(...results.scatter_data.map((d: any) => d.Dim3)) - Math.min(...results.scatter_data.map((d: any) => d.Dim3)) + 0.001)) * 160;
                              return <circle key={i} cx={x} cy={y} r="4" fill={colorForCluster(p.Cluster)} opacity="0.7" />;
                            })}
                          </svg>
                        </div>
                      </div>
                      {/* Dim2 vs Dim3 */}
                      <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                        <div className="text-xs text-slate-500 font-medium mb-3">Dimension 2 vs Dimension 3</div>
                        <div className="aspect-square relative">
                          <svg viewBox="0 0 200 200" className="w-full h-full">
                            {results.scatter_data?.map((p: any, i: number) => {
                              const x = 20 + ((p.Dim2 - Math.min(...results.scatter_data.map((d: any) => d.Dim2))) /
                                (Math.max(...results.scatter_data.map((d: any) => d.Dim2)) - Math.min(...results.scatter_data.map((d: any) => d.Dim2)) + 0.001)) * 160;
                              const y = 180 - ((p.Dim3 - Math.min(...results.scatter_data.map((d: any) => d.Dim3))) /
                                (Math.max(...results.scatter_data.map((d: any) => d.Dim3)) - Math.min(...results.scatter_data.map((d: any) => d.Dim3)) + 0.001)) * 160;
                              return <circle key={i} cx={x} cy={y} r="4" fill={colorForCluster(p.Cluster)} opacity="0.7" />;
                            })}
                          </svg>
                        </div>
                      </div>
                    </div>
                    {/* Legend */}
                    <div className="mt-4 flex flex-wrap gap-2 justify-center">
                      {results.clusters?.map((c: any) => {
                        const aiData = aiAnnotations.find(a => String(a.cluster_id) === String(c));
                        return (
                          <div key={c} className="flex items-center gap-1.5 text-xs">
                            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colorForCluster(c) }}></span>
                            <span className="text-slate-400">{aiData?.title || `Module ${c}`}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Methods View */}
                <div className={`lg:col-span-3 ${activeTab === 'methods' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <Zap className="w-5 h-5 text-purple-400" /> Methods
                    </h3>
                    {autoKResult ? (
                      <div className="space-y-2 text-xs text-slate-400">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-white font-semibold">Autopilot selected k = {autoKResult.recommended_k}</span>
                          <span className={`px-2 py-0.5 rounded-full ${autoKResult.confidence === 'high' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                            autoKResult.confidence === 'medium' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' :
                              'bg-red-500/20 text-red-400 border border-red-500/30'
                            }`}>
                            {autoKResult.confidence} confidence
                          </span>
                        </div>
                        <p>{moduleizeText(autoKResult.reasoning)}</p>
                        <p className="text-slate-500">
                          Statistical: k={autoKResult.statistical_k} • Elbow: k={autoKResult.elbow_k} • Silhouette: k={autoKResult.silhouette_k}
                        </p>
                      </div>
                    ) : (
                      <p className="text-xs text-slate-500">No Autopilot run details available yet.</p>
                    )}
                  </div>
                </div>

                {/* Legacy result panels intentionally disabled: only 3D, 2D, Table remain */}
                {showLegacyPanels && (
                  <>
                {/* Heatmap View */}
                <div className={`lg:col-span-3 ${activeTab === 'heatmap' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <Layers className="w-5 h-5 text-cyan-400" /> Gene Frequency Heatmap
                    </h3>
                    <div className="overflow-x-auto">
                      <div className="min-w-[600px]">
                        {/* Simple heatmap visualization */}
                        <div className="grid gap-0.5" style={{ gridTemplateColumns: `120px repeat(${results.clusters?.length || 0}, 1fr)` }}>
                          {/* Header row */}
                          <div className="text-xs text-slate-500 font-medium p-2">Gene</div>
                          {results.clusters?.map((c: any) => {
                            const aiData = aiAnnotations.find(a => String(a.cluster_id) === String(c));
                            return (
                              <div key={c} className="text-xs text-slate-400 font-medium p-2 text-center truncate">
                                {aiData?.title || `M${c}`}
                              </div>
                            );
                          })}

                          {/* Data rows - top genes */}
                          {(() => {
                            const topGenes = new Set<string>();
                            results.clusters?.forEach((c: any) => {
                              results.gene_stats?.filter((g: any) => String(g.Cluster) === String(c))
                                .sort((a: any, b: any) => b.Percentage - a.Percentage)
                                .slice(0, 8)
                                .forEach((g: any) => topGenes.add(g.Item));
                            });

                            return Array.from(topGenes).slice(0, 30).map(gene => (
                              <React.Fragment key={gene}>
                                <div className="text-xs text-slate-300 p-2 truncate font-mono">{gene}</div>
                                {results.clusters?.map((c: any) => {
                                  const geneData = results.gene_stats?.find((g: any) =>
                                    String(g.Cluster) === String(c) && g.Item === gene
                                  );
                                  const pct = geneData?.Percentage || 0;
                                  const intensity = Math.min(pct / 50, 1);
                                  return (
                                    <div
                                      key={c}
                                      className="p-2 text-center text-xs font-mono"
                                      style={{
                                        backgroundColor: pct > 0 ? `rgba(59, 130, 246, ${intensity * 0.8})` : 'transparent',
                                        color: intensity > 0.5 ? 'white' : '#94a3b8'
                                      }}
                                    >
                                      {pct > 0 ? `${Math.round(pct)}%` : '-'}
                                    </div>
                                  );
                                })}
                              </React.Fragment>
                            ));
                          })()}
                        </div>
                      </div>
                    </div>
                    <div className="mt-4 flex items-center justify-center gap-4 text-xs text-slate-500">
                      <span>0%</span>
                      <div className="w-32 h-3 rounded" style={{ background: 'linear-gradient(to right, transparent, rgba(59, 130, 246, 0.8))' }}></div>
                      <span>50%+</span>
                    </div>
                  </div>
                </div>

                {/* Elbow Plot View */}
                <div className={`lg:col-span-3 ${activeTab === 'elbow' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <TrendingDown className="w-5 h-5 text-cyan-400" /> Module Selection Metrics
                    </h3>
                    {elbowData ? (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Elbow Plot */}
                        <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                          <div className="text-sm text-slate-400 font-medium mb-3">Elbow Method (WCSS)</div>
                          <div className="h-48">
                            <svg viewBox="0 0 400 180" className="w-full h-full">
                              {[0, 1, 2, 3].map(i => (
                                <line key={i} x1="40" y1={30 + i * 40} x2="380" y2={30 + i * 40} stroke="#334155" strokeWidth="1" />
                              ))}
                              <polyline
                                fill="none"
                                stroke="#3498db"
                                strokeWidth="3"
                                points={elbowData.elbow.metrics.map((m: any, i: number) => {
                                  const maxInertia = Math.max(...elbowData.elbow.metrics.map((x: any) => x.inertia));
                                  const x = 50 + (i / (elbowData.elbow.metrics.length - 1)) * 320;
                                  const y = 150 - (m.inertia / maxInertia) * 120;
                                  return `${x},${y}`;
                                }).join(' ')}
                              />
                              {elbowData.elbow.metrics.map((m: any, i: number) => {
                                const maxInertia = Math.max(...elbowData.elbow.metrics.map((x: any) => x.inertia));
                                const x = 50 + (i / (elbowData.elbow.metrics.length - 1)) * 320;
                                const y = 150 - (m.inertia / maxInertia) * 120;
                                const isOptimal = m.k === elbowData.elbow.elbow_k;
                                return (
                                  <g key={i}>
                                    <circle cx={x} cy={y} r={isOptimal ? 8 : 5} fill={isOptimal ? '#e74c3c' : '#3498db'} />
                                    <text x={x} y={170} fill="#94a3b8" fontSize="10" textAnchor="middle">{m.k}</text>
                                  </g>
                                );
                              })}
                            </svg>
                          </div>
                          <p className="text-xs text-center text-slate-500 mt-2">
                            Optimal (elbow): <span className="text-cyan-400 font-bold">k = {elbowData.elbow.elbow_k}</span>
                          </p>
                        </div>

                        {/* Silhouette Plot */}
                        <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                          <div className="text-sm text-slate-400 font-medium mb-3">Silhouette Score</div>
                          <div className="h-48">
                            <svg viewBox="0 0 400 180" className="w-full h-full">
                              {[0, 1, 2, 3].map(i => (
                                <line key={i} x1="40" y1={30 + i * 40} x2="380" y2={30 + i * 40} stroke="#334155" strokeWidth="1" />
                              ))}
                              <polyline
                                fill="none"
                                stroke="#2ecc71"
                                strokeWidth="3"
                                points={elbowData.elbow.metrics.map((m: any, i: number) => {
                                  const x = 50 + (i / (elbowData.elbow.metrics.length - 1)) * 320;
                                  const y = 150 - m.silhouette * 120;
                                  return `${x},${y}`;
                                }).join(' ')}
                              />
                              {elbowData.elbow.metrics.map((m: any, i: number) => {
                                const x = 50 + (i / (elbowData.elbow.metrics.length - 1)) * 320;
                                const y = 150 - m.silhouette * 120;
                                const isBest = m.k === elbowData.elbow.silhouette_k;
                                return (
                                  <g key={i}>
                                    <circle cx={x} cy={y} r={isBest ? 8 : 5} fill={isBest ? '#e74c3c' : '#2ecc71'} />
                                    <text x={x} y={170} fill="#94a3b8" fontSize="10" textAnchor="middle">{m.k}</text>
                                  </g>
                                );
                              })}
                            </svg>
                          </div>
                          <p className="text-xs text-center text-slate-500 mt-2">
                            Optimal (silhouette): <span className="text-green-400 font-bold">k = {elbowData.elbow.silhouette_k}</span>
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12 text-slate-500">
                        <TrendingDown className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Elbow data not available</p>
                        <p className="text-xs mt-2">Run analysis from the config step to see metrics</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* NES Distribution View */}
                <div className={`lg:col-span-3 ${activeTab === 'nes' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <TrendingDown className="w-5 h-5 text-cyan-400" /> NES Distribution by Module
                    </h3>
                    <div className="space-y-4">
                      {results.clusters?.map((c: any) => {
                        const clusterData = results.scatter_data?.filter((r: any) => r.Cluster === c);
                        const nesValues = clusterData?.map((r: any) => r[scoreCol]).filter((v: any) => v != null) || [];
                        const avgNes = nesValues.length > 0 ? nesValues.reduce((a: number, b: number) => a + b, 0) / nesValues.length : 0;
                        const minNes = nesValues.length > 0 ? Math.min(...nesValues) : 0;
                        const maxNes = nesValues.length > 0 ? Math.max(...nesValues) : 0;
                        const aiData = aiAnnotations.find(a => String(a.cluster_id) === String(c));
                        const colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#795548'];

                        return (
                          <div key={c} className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center gap-2">
                                <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors[c % colors.length] }}></span>
                                <span className="text-sm font-medium text-white">{aiData?.title || `Module ${c}`}</span>
                              </div>
                              <span className={`text-sm font-mono font-bold ${avgNes > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {avgNes > 0 ? '↑' : '↓'} {avgNes.toFixed(2)}
                              </span>
                            </div>
                            {/* Simple box plot representation */}
                            <div className="relative h-8 bg-slate-800 rounded">
                              <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-px bg-slate-600"></div>
                              {/* Zero line */}
                              <div
                                className="absolute top-0 bottom-0 w-px bg-slate-500"
                                style={{ left: `${50 - (minNes / (Math.max(Math.abs(minNes), Math.abs(maxNes), 3)) * 25)}%` }}
                              ></div>
                              {/* Box */}
                              <div
                                className="absolute top-1 bottom-1 rounded"
                                style={{
                                  left: `${50 + (minNes / 6) * 50}%`,
                                  right: `${50 - (maxNes / 6) * 50}%`,
                                  backgroundColor: colors[c % colors.length] + '40',
                                  borderLeft: `2px solid ${colors[c % colors.length]}`,
                                  borderRight: `2px solid ${colors[c % colors.length]}`
                                }}
                              ></div>
                              {/* Mean line */}
                              <div
                                className="absolute top-0 bottom-0 w-0.5"
                                style={{
                                  left: `${50 + (avgNes / 6) * 50}%`,
                                  backgroundColor: colors[c % colors.length]
                                }}
                              ></div>
                            </div>
                            <div className="flex justify-between text-xs text-slate-500 mt-1">
                              <span>{minNes.toFixed(2)}</span>
                              <span>0</span>
                              <span>{maxNes.toFixed(2)}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* GSEA Bubble Plot View */}
                <div className={`lg:col-span-3 ${activeTab === 'bubble' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <Diamond className="w-5 h-5 text-cyan-400" /> GSEA Bubble Plot - Module Summary
                    </h3>
                    <p className="text-xs text-slate-500 mb-4">Modules on Y-axis, Mean NES on X-axis. Dot size = pathway count.</p>
                    <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                      <div className="relative">
                        <svg viewBox="0 0 400 300" className="w-full" style={{ maxHeight: '400px' }}>
                          {/* X-axis (NES) */}
                          <line x1="60" y1="260" x2="380" y2="260" stroke="#475569" strokeWidth="1" />
                          <text x="220" y="290" textAnchor="middle" fill="#94a3b8" fontSize="12">Mean NES</text>
                          {[-2, -1, 0, 1, 2].map(v => (
                            <g key={v}>
                              <line x1={60 + (v + 2) * 80} y1="260" x2={60 + (v + 2) * 80} y2="265" stroke="#475569" strokeWidth="1" />
                              <text x={60 + (v + 2) * 80} y="275" textAnchor="middle" fill="#94a3b8" fontSize="10">{v}</text>
                            </g>
                          ))}
                          {/* Zero line */}
                          <line x1="220" y1="20" x2="220" y2="260" stroke="#475569" strokeWidth="1" strokeDasharray="4" />
                          {/* Bubbles */}
                          {results.clusters?.map((c: any, idx: number) => {
                            const clusterData = results.scatter_data?.filter((r: any) => r.Cluster === c);
                            const nesValues = clusterData?.map((r: any) => r[scoreCol]).filter((v: any) => v != null) || [];
                            const avgNes = nesValues.length > 0 ? nesValues.reduce((a: number, b: number) => a + b, 0) / nesValues.length : 0;
                            const count = clusterData?.length || 1;
                            const aiData = aiAnnotations.find(a => String(a.cluster_id) === String(c));
                            const colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#795548'];
                            const y = 40 + idx * (220 / Math.max(results.clusters.length, 1));
                            const x = 220 + avgNes * 80;
                            const r = Math.min(Math.max(count * 0.8, 8), 30);
                            return (
                              <g key={c}>
                                <circle cx={x} cy={y} r={r} fill={colors[idx % colors.length]} opacity="0.8" stroke="white" strokeWidth="2" />
                                <text x="55" y={y + 4} textAnchor="end" fill="#e2e8f0" fontSize="10">
                                  {aiData?.title ? aiData.title.slice(0, 20) : `Module ${c}`}
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Mountain/Ridge Plot View */}
                <div className={`lg:col-span-3 ${activeTab === 'mountain' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <TrendingDown className="w-5 h-5 text-cyan-400" /> Mountain Plot - NES Distribution
                    </h3>
                    <p className="text-xs text-slate-500 mb-4">Ridge plot showing NES distribution for each module.</p>
                    <div className="space-y-1">
                      {results.clusters?.map((c: any, idx: number) => {
                        const clusterData = results.scatter_data?.filter((r: any) => r.Cluster === c);
                        const nesValues = clusterData?.map((r: any) => r[scoreCol]).filter((v: any) => v != null).sort((a: number, b: number) => a - b) || [];
                        const aiData = aiAnnotations.find(a => String(a.cluster_id) === String(c));
                        const colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#795548'];
                        const color = colors[idx % colors.length];

                        // Create a simple density-like path
                        const minNes = Math.min(-3, ...nesValues);
                        const maxNes = Math.max(3, ...nesValues);
                        const avgNes = nesValues.length > 0 ? nesValues.reduce((a: number, b: number) => a + b, 0) / nesValues.length : 0;
                        const std = nesValues.length > 1 ? Math.sqrt(nesValues.reduce((s: number, v: number) => s + (v - avgNes) ** 2, 0) / nesValues.length) : 0.5;

                        return (
                          <div key={c} className="relative">
                            <div className="flex items-center gap-3">
                              <span className="text-xs text-slate-400 w-24 truncate">{aiData?.title?.slice(0, 15) || `Module ${c}`}</span>
                              <div className="flex-1 h-12 bg-slate-950 rounded relative overflow-hidden">
                                <svg viewBox="0 0 400 50" className="w-full h-full" preserveAspectRatio="none">
                                  {/* Mountain shape using bezier curve */}
                                  <path
                                    d={`M 0,50 
                                        Q ${200 + avgNes * 30 - std * 60},50 ${200 + avgNes * 30 - std * 30},20 
                                        Q ${200 + avgNes * 30},0 ${200 + avgNes * 30 + std * 30},20
                                        Q ${200 + avgNes * 30 + std * 60},50 400,50 Z`}
                                    fill={color}
                                    opacity="0.6"
                                  />
                                  {/* Individual points */}
                                  {nesValues.slice(0, 30).map((v: number, i: number) => (
                                    <circle
                                      key={i}
                                      cx={200 + v * 30}
                                      cy={25 + (Math.random() - 0.5) * 10}
                                      r="2"
                                      fill={color}
                                      opacity="0.8"
                                    />
                                  ))}
                                  {/* Zero line */}
                                  <line x1="200" y1="0" x2="200" y2="50" stroke="#475569" strokeWidth="1" strokeDasharray="2" />
                                </svg>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                      <div className="flex justify-between text-xs text-slate-500 mt-2 px-28">
                        <span>-3</span>
                        <span>0</span>
                        <span>+3</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Top Terms Bar Chart View */}
                <div className={`lg:col-span-3 ${activeTab === 'bar' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <Target className="w-5 h-5 text-cyan-400" /> Top Enriched Pathways
                    </h3>
                    <p className="text-xs text-slate-500 mb-4">Top 15 pathways by absolute NES score.</p>
                    <div className="space-y-2">
                      {results.scatter_data
                        ?.slice()
                        .sort((a: any, b: any) => Math.abs(b[scoreCol] || 0) - Math.abs(a[scoreCol] || 0))
                        .slice(0, 15)
                        .map((p: any, i: number) => {
                          const nes = p[scoreCol] || 0;
                          const maxNes = 3;
                          const width = Math.min(Math.abs(nes) / maxNes * 100, 100);
                          const isPositive = nes > 0;
                          const pathName = String(p[pathCol] || '')
                            .replace(/^(GO|KEGG|REACTOME|HALLMARK)[A-Z]*_/i, '')
                            .replace(/_/g, ' ')
                            .slice(0, 45);

                          return (
                            <div key={i} className="flex items-center gap-2">
                              <span className="text-xs text-slate-400 w-48 truncate" title={p[pathCol]}>{pathName}</span>
                              <div className="flex-1 h-5 bg-slate-800 rounded relative">
                                <div
                                  className={`absolute top-0 h-full rounded ${isPositive ? 'bg-red-500/70 right-1/2' : 'bg-blue-500/70 left-1/2'}`}
                                  style={{ width: `${width / 2}%` }}
                                />
                                <div className="absolute top-0 left-1/2 h-full w-px bg-slate-600" />
                              </div>
                              <span className={`text-xs font-mono w-12 text-right ${isPositive ? 'text-red-400' : 'text-blue-400'}`}>
                                {nes.toFixed(2)}
                              </span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                </div>

                {/* Manhattan Plot View */}
                <div className={`lg:col-span-3 ${activeTab === 'manhattan' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-cyan-400" /> Manhattan Plot
                    </h3>
                    <p className="text-xs text-slate-500 mb-4">All pathways shown by |NES| score, colored by module.</p>
                    <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                      <svg viewBox="0 0 600 200" className="w-full" style={{ maxHeight: '300px' }}>
                        {/* Y-axis */}
                        <line x1="40" y1="20" x2="40" y2="170" stroke="#475569" strokeWidth="1" />
                        <text x="15" y="100" textAnchor="middle" fill="#94a3b8" fontSize="10" transform="rotate(-90, 15, 100)">|NES|</text>
                        {[0, 1, 2, 3].map(v => (
                          <g key={v}>
                            <line x1="35" y1={170 - v * 50} x2="40" y2={170 - v * 50} stroke="#475569" strokeWidth="1" />
                            <text x="30" y={173 - v * 50} textAnchor="end" fill="#94a3b8" fontSize="9">{v}</text>
                          </g>
                        ))}
                        {/* X-axis */}
                        <line x1="40" y1="170" x2="580" y2="170" stroke="#475569" strokeWidth="1" />
                        <text x="310" y="190" textAnchor="middle" fill="#94a3b8" fontSize="10">Pathway Index</text>
                        {/* Points */}
                        {results.scatter_data?.map((p: any, i: number) => {
                          const colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#795548'];
                          const nes = Math.abs(p[scoreCol] || 0);
                          const x = 50 + (i / Math.max(results.scatter_data.length - 1, 1)) * 520;
                          const y = 170 - Math.min(nes, 3) * 50;
                          return (
                            <circle
                              key={i}
                              cx={x}
                              cy={y}
                              r="3"
                              fill={colors[p.Cluster % colors.length]}
                              opacity="0.7"
                            />
                          );
                        })}
                      </svg>
                    </div>
                  </div>
                </div>

                {/* Hexagonal Canvas View */}
                <div className={`lg:col-span-3 ${activeTab === 'hexagon' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-xl p-6">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                      <Diamond className="w-5 h-5 text-cyan-400" /> Hexagonal Canvas - Pathway Similarity
                    </h3>
                    <p className="text-xs text-slate-500 mb-4">
                      Each hexagon is a pathway. Position = MDS similarity. Blue = enriched (|NES| ≥ 1.5). Brightness = gene set similarity.
                    </p>
                    <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 relative overflow-hidden">
                      <svg viewBox="0 0 500 400" className="w-full" style={{ maxHeight: '500px' }}>
                        {/* Background grid */}
                        <defs>
                          <pattern id="hexPattern" width="30" height="26" patternUnits="userSpaceOnUse">
                            <polygon points="15,0 30,7.5 30,22.5 15,26 0,22.5 0,7.5" fill="none" stroke="#1e293b" strokeWidth="0.5" opacity="0.3" />
                          </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill="url(#hexPattern)" opacity="0.5" />

                        {/* Pathway hexagons */}
                        {results.scatter_data?.map((p: any, i: number) => {
                          const colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#795548'];
                          const nes = Math.abs(p[scoreCol] || 0);
                          const isEnriched = nes >= 1.5;

                          // Map Dim1 and Dim2 to SVG coordinates
                          const minDim1 = Math.min(...results.scatter_data.map((d: any) => d.Dim1));
                          const maxDim1 = Math.max(...results.scatter_data.map((d: any) => d.Dim1));
                          const minDim2 = Math.min(...results.scatter_data.map((d: any) => d.Dim2));
                          const maxDim2 = Math.max(...results.scatter_data.map((d: any) => d.Dim2));

                          const x = 50 + ((p.Dim1 - minDim1) / (maxDim1 - minDim1 + 0.001)) * 400;
                          const y = 350 - ((p.Dim2 - minDim2) / (maxDim2 - minDim2 + 0.001)) * 300;

                          // Hexagon size based on enrichment
                          const size = isEnriched ? 12 : 8;
                          const hexPoints = (cx: number, cy: number, s: number) => {
                            return [0, 60, 120, 180, 240, 300].map(angle => {
                              const rad = (angle * Math.PI) / 180;
                              return `${cx + s * Math.cos(rad)},${cy + s * Math.sin(rad)}`;
                            }).join(' ');
                          };

                          return (
                            <g key={i}>
                              <polygon
                                points={hexPoints(x, y, size)}
                                fill={isEnriched ? '#3b82f6' : colors[p.Cluster % colors.length]}
                                opacity={isEnriched ? 0.9 : 0.6}
                                stroke={isEnriched ? '#1e40af' : 'white'}
                                strokeWidth={isEnriched ? 2 : 0.5}
                              />
                              <title>{`${p[pathCol]}\nNES: ${nes.toFixed(2)}\nModule: ${p.Cluster}`}</title>
                            </g>
                          );
                        })}

                        {/* Legend */}
                        <g transform="translate(400, 20)">
                          <polygon points="0,10,10,5,10,15,0,20,-10,15,-10,5" fill="#3b82f6" stroke="#1e40af" strokeWidth="2" />
                          <text x="20" y="15" fill="#e2e8f0" fontSize="10">Enriched</text>
                          <polygon points="0,35,8,30,8,40,0,45,-8,40,-8,30" fill="#64748b" stroke="white" strokeWidth="0.5" opacity="0.6" />
                          <text x="20" y="40" fill="#94a3b8" fontSize="10">Other</text>
                        </g>
                      </svg>
                    </div>
                  </div>
                </div>

                  </>
                )}

                {/* Table View (Full Width if active) */}
                <div className={`lg:col-span-4 ${activeTab === 'table' ? '' : 'hidden'}`}>
                  <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                    {geneSearchUpper && (
                      <div className="px-3 py-2 text-[11px] text-cyan-300 border-b border-slate-800 bg-slate-950/70">
                        Gene filter active: showing {filteredTableRows.length.toLocaleString()} matching pathway row(s).
                      </div>
                    )}
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left text-slate-400">
                        <thead className="bg-slate-950 text-slate-200 uppercase text-xs font-bold sticky top-0">
                          <tr>
                            <th className="p-3 w-16">Module</th>
                            <th className="p-3 min-w-[200px]">Pathway</th>
                            <th className="p-3">NES</th>
                            <th className="p-3 min-w-[150px]">AI Title</th>
                            <th className="p-3 min-w-[200px]">AI Summary</th>
                            <th className="p-3">Key Process</th>
                            <th className="p-3 min-w-[250px]">Core Genes</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                          {filteredTableRows.map((row: any, i: number) => {
                            const aiData = aiAnnotations.find(a => String(a.cluster_id) === String(row.Cluster));
                            const clusterGenes = results.gene_stats
                              ?.filter((g: any) => String(g.Cluster) === String(row.Cluster) && g.Percentage >= 25)
                              .sort((a: any, b: any) => b.Percentage - a.Percentage)
                              .slice(0, 8);

                            return (
                              <tr key={i} className="hover:bg-slate-800/50">
                                <td className="p-3 font-mono text-cyan-500 font-bold">M{row.Cluster}</td>
                                <td className="p-3 text-white text-xs">{row[pathCol]}</td>
                                <td className={`p-3 font-mono text-sm ${row[scoreCol] > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  {row[scoreCol] ? row[scoreCol].toFixed(2) : '-'}
                                </td>
                                <td className="p-3 text-purple-300 font-medium text-sm">{aiData?.title || '-'}</td>
                                <td className="p-3 text-xs text-slate-400 italic">{normalizeAnnotationSummary(aiData?.summary) || '-'}</td>
                                <td className="p-3">
                                  {aiData?.key_process && (
                                    <span className="text-[10px] bg-purple-900/30 text-purple-300 px-2 py-1 rounded-full">
                                      {aiData.key_process}
                                    </span>
                                  )}
                                </td>
                                <td className="p-3">
                                  <div className="flex flex-wrap gap-1">
                                    {clusterGenes?.map((g: any) => (
                                      <span
                                        key={g.Item}
                                        className={`text-[9px] px-1.5 py-0.5 rounded ${g.DEG
                                          ? 'text-orange-300 bg-orange-950/50 border border-orange-500/30'
                                          : 'text-slate-300 bg-slate-800'
                                          }`}
                                      >
                                        {g.Item}
                                      </span>
                                    ))}
                                  </div>
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>

                {/* Right: Insights Panel (Show for all visualization tabs except table) */}
                {activeTab !== 'table' && (
                  <div className={`${activeTab === '3d' ? 'lg:col-span-4' : ''} bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-4 h-[460px] flex flex-col overflow-hidden`}>
                    <div className="mb-3 flex items-center gap-2">
                      <h3 className="text-white font-bold flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-cyan-400" /> Module Insights
                      </h3>
                    </div>
                    <div className="space-y-3 flex-1 overflow-y-auto pr-1" onWheel={passScrollToPage}>
                      {results.clusters.map((c: string) => {
                        const aiData = aiAnnotations.find(a => String(a.cluster_id) === String(c));
                        const clusterStats = results.cluster_stats?.find((s: any) => String(s.cluster_id) === String(c));
                        const topGenes = results.gene_stats
                          .filter((g: any) => String(g.Cluster) === String(c) && g.Percentage > 20)
                          .slice(0, 5);

                        return (
                          <div key={c} className="bg-slate-950 border border-slate-800 hover:border-cyan-500/30 rounded-xl p-4 transition-all group">
                            <div className="flex justify-between items-start mb-2">
                              <span className="text-xs font-bold text-cyan-500 font-mono bg-cyan-950/30 px-2 py-1 rounded">M{c}</span>
                              {aiData && <span className="text-[10px] bg-purple-900/30 text-purple-300 px-2 py-1 rounded-full border border-purple-500/20">{Math.round(aiData.confidence * 100)}%</span>}
                            </div>

                            {/* AI Title or Fallback */}
                            <h4 className="text-white font-bold text-sm mb-1 leading-snug">
                              {aiData ? aiData.title : (
                                moduleizeText(String(results.scatter_data.find((r: any) => String(r.Cluster) === String(c))?.Cluster_Summary || '')) ||
                                <span className="text-slate-500 italic">{useAgent && agentStatus === 'running' ? 'Analyzing...' : 'Module ' + c}</span>
                              )}
                            </h4>

                            {aiData ? (
                              <p className="text-xs text-slate-400 mb-2 leading-relaxed">{moduleizeText(normalizeAnnotationSummary(aiData.summary))}</p>
                            ) : (
                              <p className="text-[10px] text-slate-500 mb-2 italic leading-tight">
                                {clusterStats?.top_pathway || results.scatter_data.find((r: any) => String(r.Cluster) === String(c))?.[pathCol]}
                              </p>
                            )}

                            {/* Module Statistics */}
                            {clusterStats && (
                              <div className="grid grid-cols-3 gap-2 mb-3 text-center">
                                <div className="bg-slate-900 rounded-lg p-2">
                                  <div className="text-sm font-bold text-cyan-400">{clusterStats.n_pathways}</div>
                                  <div className="text-[9px] text-slate-500">Pathways</div>
                                </div>
                                <div className="bg-slate-900 rounded-lg p-2">
                                  <div className="text-sm font-bold text-purple-400">{clusterStats.n_core_genes}</div>
                                  <div className="text-[9px] text-slate-500">Core Genes</div>
                                </div>
                                <div className="bg-slate-900 rounded-lg p-2">
                                  <div className={`text-sm font-bold ${clusterStats.mean_nes > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {clusterStats.mean_nes > 0 ? '↑' : '↓'} {Math.abs(clusterStats.mean_nes).toFixed(2)}
                                  </div>
                                  <div className="text-[9px] text-slate-500">Mean NES</div>
                                </div>
                              </div>
                            )}

                            {/* Top Genes - DEG highlighted */}
                            <div className="flex flex-wrap gap-1">
                              {topGenes.map((g: any) => (
                                <span
                                  key={g.Item}
                                  className={`text-[10px] px-1.5 py-0.5 rounded border ${g.DEG
                                    ? 'text-orange-300 bg-orange-950/50 border-orange-500/30'
                                    : 'text-slate-400 bg-slate-900 border-slate-800'
                                    }`}
                                >
                                  {g.Item} <span className={g.DEG ? "text-orange-500" : "text-slate-600"}>{Math.round(g.Percentage)}%</span>
                                </span>
                              ))}
                            </div>

                            {/* AI Key Process Badge */}
                            {aiData?.key_process && (
                              <div className="mt-2 pt-2 border-t border-slate-800">
                                <span className="text-[10px] text-purple-400 bg-purple-950/30 px-2 py-1 rounded-full border border-purple-800/30">
                                  {aiData.key_process}
                                </span>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2 -mt-2">
                <div className="flex flex-wrap items-center gap-2">
                  <div className="px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-900/60 flex items-baseline gap-1.5">
                    <span className="text-sm font-bold text-cyan-300">{results.clusters.length}</span>
                    <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wide">Modules</span>
                  </div>
                  <div className="px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-900/60 flex items-baseline gap-1.5">
                    <span className="text-sm font-bold text-blue-300">{(results.total_pathways || results.scatter_data?.length || 0).toLocaleString()}</span>
                    <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wide">Pathways</span>
                  </div>
                  <div className="px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-900/60 flex items-baseline gap-1.5">
                    <span className="text-sm font-bold text-purple-300">{Number(results.total_genes || 0) ? Number(results.total_genes).toLocaleString() : '—'}</span>
                    <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wide">Unique Genes</span>
                  </div>
                  <div className="px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-900/60 flex items-baseline gap-1.5">
                    <span className="text-sm font-bold text-green-300">{results.mds_gof ? `${Math.round(results.mds_gof[0] * 100)}%` : '—'}</span>
                    <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wide">MDS GOF</span>
                  </div>

                  {Boolean(results.has_deg) && Number(results.n_deg_genes) > 0 && (
                    <div className="px-2.5 py-1.5 rounded-lg border border-orange-500/30 bg-orange-950/20 flex items-center gap-2 text-[11px] text-orange-200">
                      <Target className="w-4 h-4 text-orange-400" />
                      <span className="font-bold text-orange-400">{results.n_deg_genes}</span>
                      <span className="hidden sm:inline">DEG genes in core programs</span>
                      <span className="text-orange-400 hidden md:inline">•</span>
                      <span className="hidden md:inline text-orange-300/80">orange highlight</span>
                    </div>
                  )}
                </div>

                <button
                  onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })}
                  className="text-[11px] px-2.5 py-1 rounded-md border border-slate-700 bg-slate-900 text-slate-300 hover:text-white hover:border-cyan-500/50 transition-all"
                >
                  More Info ↓
                </button>
              </div>

              {activeTab === '3d' && (
                <details className="mt-4 bg-slate-900/50 border border-slate-800 rounded-xl">
                  <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-slate-400 hover:text-white flex items-center gap-2">
                    <Info className="w-4 h-4" /> What do these metrics mean? (click to expand)
                  </summary>
                  <div className="px-4 pb-4 grid grid-cols-2 lg:grid-cols-4 gap-4 text-xs text-slate-400">
                    <div className="space-y-2">
                      <h4 className="font-bold text-slate-200">Per-Module Stats</h4>
                      <p><span className="text-cyan-400 font-bold">Pathways</span>: Number of gene sets in this module</p>
                      <p><span className="text-purple-400 font-bold">Core Genes</span>: Genes appearing in &gt;=25% of module pathways</p>
                      <p><span className="text-green-400 font-bold">Mean NES</span>: Average Normalized Enrichment Score</p>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-bold text-slate-200">Gene Metrics</h4>
                      <p><span className="text-orange-400 font-bold">Frequency %</span>: Percentage of pathways containing this gene</p>
                      <p><span className="text-blue-400 font-bold">Activation Score</span>: -1 to +1. Positive = upregulated pathways</p>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-bold text-slate-200">AI Annotations</h4>
                      <p><span className="text-purple-400 font-bold">Title</span>: AI-generated name for the biological program</p>
                      <p><span className="text-purple-400 font-bold">Confidence</span>: AI certainty based on pathway coherence</p>
                      <p><span className="text-purple-400 font-bold">Key Process</span>: Primary biological category</p>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-bold text-slate-200">Quality Metrics</h4>
                      <p><span className="text-green-400 font-bold">MDS GOF</span>: How well 3D coordinates preserve original distances</p>
                      <p><span className="text-cyan-400 font-bold">Silhouette</span>: Module quality (-1 to 1). Higher = better separated</p>
                      <p><span className="text-cyan-400 font-bold">WCSS</span>: Within-module sum of squares (lower = tighter)</p>
                    </div>
                  </div>
                </details>
              )}

              {activeTab === '3d' && (
                <section ref={mountainSectionRef} className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 sm:p-5 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      <TrendingDown className="w-4 h-4 text-cyan-400" />
                      Mountain Explorer
                    </h3>
                    <span className="text-[11px] text-slate-400">
                      {(mountainData?.pathways?.length || 0).toLocaleString()} pathways
                    </span>
                  </div>
                  {copyNotice && (
                    <div className="text-[11px] text-cyan-300">{copyNotice}</div>
                  )}
                  {mountainLockNotice && (
                    <div className="text-[11px] text-amber-300">{mountainLockNotice}</div>
                  )}

                  {mountainLoading && (
                    <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-400 flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
                      Building mountain explorer data...
                    </div>
                  )}

                  {mountainError && (
                    <div className="bg-red-950/20 border border-red-800/40 rounded-xl p-4 text-sm text-red-300">
                      {mountainError}
                    </div>
                  )}

                  {!mountainLoading && !mountainError && (!mountainData?.has_data || !mountainData?.pathways?.length) && (
                    <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-400">
                      Mountain Explorer is available after pathway + DEG ranked genes are loaded.
                    </div>
                  )}

                  {!mountainLoading && mountainData?.pathways?.length ? (
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                      <div className="lg:col-span-4 bg-slate-950 border border-slate-800 rounded-xl p-3">
                        <div className="relative">
                          <input
                            value={mountainPathSearch}
                            onChange={(e) => setMountainPathSearch(e.target.value)}
                            placeholder="Search pathways or modules..."
                            className="w-full bg-slate-900 border border-slate-700 rounded-md px-3 py-2 pr-8 text-xs text-slate-200 placeholder:text-slate-500"
                          />
                          {mountainPathSearch.trim() && (
                            <button
                              onClick={() => setMountainPathSearch('')}
                              className="absolute right-1.5 top-1/2 -translate-y-1/2 w-5 h-5 rounded-full border border-slate-700 bg-slate-800/90 text-slate-400 hover:text-slate-100 hover:border-cyan-500/50 transition-all duration-200 flex items-center justify-center"
                              aria-label="Clear pathway search"
                              title="Clear search"
                            >
                              <X className="w-3 h-3" />
                            </button>
                          )}
                        </div>
                        {mountainModuleOptions.length > 0 && (
                          <div className="mt-2 space-y-1">
                            <div className="text-[10px] uppercase tracking-wide text-slate-500">Filter modules</div>
                            <div className="flex flex-wrap gap-1.5">
                              <button
                                onClick={() => {
                                  setMountainModuleFilter(null);
                                  setMountainModuleExpanded(false);
                                }}
                                className={`px-2.5 py-1 rounded-md border text-[11px] transition-all ${
                                  mountainModuleFilter === null
                                    ? 'bg-slate-800 border-cyan-500/60 text-white'
                                    : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-600'
                                }`}
                              >
                                All
                              </button>
                              {visibleModuleOptions.map((opt) => (
                                <button
                                  key={opt.id}
                                  onClick={() => setMountainModuleFilter((prev) => (prev === opt.id ? null : opt.id))}
                                  className={`px-2.5 py-1 rounded-md border text-[11px] transition-all ${
                                    mountainModuleFilter === opt.id
                                      ? 'text-white shadow-sm'
                                      : 'text-slate-300 hover:text-white'
                                  }`}
                                  style={
                                    mountainModuleFilter === opt.id
                                      ? { borderColor: opt.color, boxShadow: `0 0 0 1px ${opt.color}33`, backgroundColor: '#0f172a' }
                                      : { borderColor: '#1f2937', backgroundColor: '#0b1220', color: opt.color }
                                  }
                                  title={`${opt.name} (${opt.count} pathways)`}
                                >
                                  M{opt.id}
                                </button>
                              ))}
                              {extraModuleCount > 0 && (
                                <button
                                  onClick={() => setMountainModuleExpanded((v) => !v)}
                                  className="px-2.5 py-1 rounded-md border border-slate-700 bg-slate-900 text-[11px] text-slate-400 hover:text-white"
                                >
                                  {mountainModuleExpanded ? 'Collapse' : `+${extraModuleCount} more`}
                                </button>
                              )}
                            </div>
                          </div>
                        )}
                        <div className="mt-3 max-h-[520px] overflow-y-auto space-y-1 pr-1" onWheel={passScrollToPage}>
                          {filteredMountainPathways.map((p) => {
                            const isActive = Number(p.idx) === Number(selectedMountainPathId);
                            return (
                              <button
                                key={p.idx}
                                onClick={() => {
                                  setSelectedMountainPathId(Number(p.idx));
                                  setMountainOriginPapers([]);
                                  setMountainExpandedAbstracts({});
                                  setMountainOriginQuery('');
                                  setMountainOriginMessage('');
                                  setMountainOriginExact(null);
                                  setMountainOriginQueries([]);
                                  setMountainNamingClue(null);
                                  setMountainTrailSearchSpec(null);
                                  setMountainPaperAnalyses({});
                                  setMountainExpandedAnalyses({});
                                  setMountainPaperAnalysisLoading(null);
                                  setMountainHitInfo(null);
                                  mountainLastHoverKeyRef.current = '';
                                  mountainHoverThrottleTsRef.current = 0;
                                  setMountainLockedGeneKey(null);
                                  mountainLockedGeneKeyRef.current = null;
                                }}
                                className={`w-full text-left rounded-lg border px-3 py-2 transition-all ${
                                  isActive
                                    ? 'bg-cyan-950/40 border-cyan-600/50'
                                    : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                                }`}
                              >
                                <div className="text-[11px] text-slate-300 font-semibold truncate">{p.pathway}</div>
                                <div className="mt-1 text-[10px] text-slate-500 flex gap-2">
                                  <span>M{p.cluster}</span>
                                  <span className={p.nes >= 0 ? 'text-green-400' : 'text-red-400'}>
                                    NES {p.nes >= 0 ? '+' : ''}{Number(p.nes || 0).toFixed(2)}
                                  </span>
                                  <span>p {formatPValue(p.p_value)}</span>
                                  <span>adj p {formatPValue(p.adj_p_value)}</span>
                                </div>
                              </button>
                            );
                          })}
                          {filteredMountainPathways.length === 0 && (
                            <div className="text-xs text-slate-500 py-4 text-center">No pathways match your search.</div>
                          )}
                        </div>
                      </div>

                      <div className="lg:col-span-8 space-y-3">
                        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300">
                          <span className="text-[10px] uppercase tracking-wide text-slate-500">Layers</span>
                          <button
                            onClick={() => setShowRunningEs((v) => !v)}
                            className={`px-2.5 py-1 rounded-md border text-[11px] transition-all ${
                              showRunningEs ? 'bg-slate-800 border-cyan-500/60 text-cyan-100' : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-cyan-500/40'
                            }`}
                          >
                            Running ES
                          </button>
                          <button
                            onClick={() => setShowPathwayGenes((v) => !v)}
                            className={`px-2.5 py-1 rounded-md border text-[11px] transition-all ${
                              showPathwayGenes ? 'bg-slate-800 border-cyan-500/60 text-cyan-100' : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-cyan-500/40'
                            }`}
                          >
                            Pathway Genes
                          </button>
                          <button
                            onClick={() => setShowRankStrip((v) => !v)}
                            className={`px-2.5 py-1 rounded-md border text-[11px] transition-all ${
                              showRankStrip ? 'bg-slate-800 border-cyan-500/60 text-cyan-100' : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-cyan-500/40'
                            }`}
                          >
                            Rank Strip
                          </button>
                          <button
                            onClick={() => setShowRankStripLegend((v) => !v)}
                            disabled={!showRankStrip}
                            className={`px-2.5 py-1 rounded-md border text-[11px] transition-all ${
                              !showRankStrip
                                ? 'bg-slate-900/70 border-slate-800 text-slate-600 cursor-not-allowed'
                                : showRankStripLegend
                                  ? 'bg-slate-800 border-cyan-500/60 text-cyan-100'
                                  : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-cyan-500/40'
                            }`}
                            title={showRankStrip ? 'Show or hide Activated/Suppressed labels' : 'Enable Rank Strip first'}
                          >
                            Rank Strip Legend
                          </button>
                        </div>
                        <div className="bg-slate-950 border border-slate-800 rounded-xl p-2">
                          <div ref={mountainPlotRef} className="w-full h-[430px]" />
                        </div>

                        {selectedMountainPathway && (
                          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-300 space-y-2">
                            <div className="flex flex-wrap items-start justify-between gap-2">
                              <div className="text-cyan-200 font-semibold break-all">{selectedMountainPathway.pathway}</div>
                              <div className="flex items-center gap-2">
                                <span className="inline-flex items-center gap-2 text-[10px] text-slate-300 bg-slate-900 border border-slate-700 rounded-md px-2 py-1">
                                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: colorForCluster(selectedMountainPathway.cluster) }} />
                                  <span className="truncate">Module M{selectedMountainPathway.cluster}</span>
                                </span>
                                <button
                                  onClick={() => copyPathwayName(selectedMountainPathway.pathway)}
                                  className="shrink-0 text-[10px] px-2 py-1 rounded border border-slate-700 bg-slate-900 text-slate-300 hover:text-white"
                                >
                                  Copy
                                </button>
                                <button
                                  onClick={() => setShowMountainExportModal(true)}
                                  className="shrink-0 text-[10px] px-2 py-1 rounded border border-cyan-700/50 bg-cyan-950/30 text-cyan-200 hover:bg-cyan-900/40"
                                >
                                  Export
                                </button>
                              </div>
                            </div>
                            <div className="flex flex-wrap gap-1.5 text-[10px]">
                              <span className="px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300">Module: M{selectedMountainPathway.cluster} ({selectedMountainPathway.module})</span>
                              <span className={`px-2 py-0.5 rounded border ${selectedMountainPathway.nes >= 0 ? 'border-green-700/50 bg-green-950/20 text-green-200' : 'border-red-700/50 bg-red-950/20 text-red-200'}`}>
                                NES {Number(selectedMountainPathway.nes || 0).toFixed(2)}
                              </span>
                              <span className="px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300">Pvalue {formatPValue(selectedMountainPathway.p_value)}</span>
                              <span className="px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300">Adj Pvalue {formatPValue(selectedMountainPathway.adj_p_value)}</span>
                              <span className="px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300">Genes {selectedMountainPathway.genes?.length || 0}</span>
                              <span className="px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300">DEGs {selectedPathwayDegGeneCount}</span>
                            </div>
                            <div className="text-slate-400 leading-relaxed">
                              {mountainHitInfo ? (
                                <div className="space-y-1">
                                  <div className="flex items-center gap-2">
                                    <div className="text-cyan-300 font-semibold">{mountainHitInfo.symbol}</div>
                                    {mountainLockedGeneKey && (
                                      <span className="text-[10px] px-1.5 py-0.5 rounded border border-amber-600/50 bg-amber-950/25 text-amber-200">
                                        Locked
                                      </span>
                                    )}
                                  </div>
                                  <div>Rank: {mountainHitInfo.rank} | LogFC: {mountainHitInfo.logfc}</div>
                                  <div><span className="text-slate-500">Full Name:</span> {mountainHitInfo.fullName}</div>
                                  <button
                                    onClick={() => setMountainHitExpanded((v) => !v)}
                                    className="text-[11px] px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-cyan-300 hover:text-cyan-200 hover:border-cyan-500/50 transition-all"
                                  >
                                    {mountainHitExpanded ? 'Hide Gene Details' : 'More Gene Info'}
                                  </button>
                                  {mountainHitExpanded && (
                                    <div className="pt-2 mt-2 border-t border-slate-800/80 space-y-2">
                                      <div className="space-y-1">
                                        <div className="text-slate-500">Function:</div>
                                        <div className="text-slate-300 whitespace-pre-wrap leading-relaxed">{mountainHitInfo.functionText}</div>
                                      </div>
                                      <div className="space-y-1">
                                        <div className="text-slate-500">Reference</div>
                                        {Array.isArray(mountainHitInfo.sources) && mountainHitInfo.sources.length > 0 ? (
                                          <div className="space-y-1">
                                            {mountainHitInfo.sources.map((src, idx) => (
                                              <div key={`${src.label}-${idx}`} className="text-[11px] text-slate-400">
                                                {idx + 1}. {src.label}: {src.note}{' '}
                                                {src.url ? (
                                                  <a
                                                    href={src.url}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    className="text-cyan-300 underline hover:text-cyan-200"
                                                  >
                                                    source
                                                  </a>
                                                ) : (
                                                  ''
                                                )}
                                              </div>
                                            ))}
                                          </div>
                                        ) : (
                                          <div className="text-[11px] text-slate-500">No retrieval source metadata available.</div>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              ) : (
                                'Hover over a pathway-gene hit marker to view gene symbol and full name. Open More Gene Info for function and references.'
                              )}
                            </div>
                          </div>
                        )}

                      </div>

                      <div className="lg:col-span-12">
                        <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 min-h-[560px] flex flex-col gap-3">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div>
                              <h4 className="text-sm font-semibold text-white">Trail Talk</h4>
                              <p className="text-[11px] text-slate-500">Trace a pathway name to real papers: fast local search first, then AI checks what truly matches.</p>
                            </div>
                            <button
                              onClick={requestPathwayOriginPapers}
                              disabled={!selectedMountainPathway || mountainOriginLoading}
                              className="inline-flex items-center gap-2 text-xs sm:text-sm font-semibold px-4 py-2 rounded-lg border border-cyan-500/60 bg-gradient-to-r from-cyan-600/25 to-blue-600/25 text-cyan-100 hover:from-cyan-500/30 hover:to-blue-500/30 disabled:opacity-50"
                            >
                              <BookOpen className="w-4 h-4" />
                              {mountainOriginLoading ? 'Searching...' : 'Find Papers'}
                            </button>
                          </div>

                          <details className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2 text-[10px] text-slate-300">
                            <summary className="cursor-pointer select-none text-[11px] font-semibold text-cyan-300">
                              Trail Talk guide (click to expand)
                            </summary>
                            <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
                              <div className="rounded-md border border-slate-800 bg-slate-950/70 p-2 space-y-1">
                                <div className="text-[10px] font-semibold text-slate-100">How it works</div>
                                <div><span className="text-cyan-300">1. Local retrieval:</span> deterministic queries from pathway title, GO/MSigDB clues, and linked references.</div>
                                <div><span className="text-cyan-300">2. LLM validation:</span> checks only local candidates and cannot add out-of-pool papers.</div>
                              </div>
                              <div className="rounded-md border border-slate-800 bg-slate-950/70 p-2 space-y-1">
                                <div className="text-[10px] font-semibold text-slate-100">Controls</div>
                                <div><span className="text-cyan-300">Naming clues:</span> uses GO/MSigDB IDs, term metadata, and clue-linked references.</div>
                                <div><span className="text-cyan-300">Search Mode:</span> <span className="text-slate-200">Origin</span> keeps retrieval strict to naming evidence; <span className="text-slate-200">Origin + Context</span> also includes broader pathway-context papers.</div>
                              </div>
                              <div className="rounded-md border border-slate-800 bg-slate-950/70 p-2 space-y-1">
                                <div className="text-[10px] font-semibold text-slate-100">Status panel</div>
                                <div><span className="text-emerald-300">Done</span> = completed step, <span className="text-cyan-300">Running</span> = in progress, <span className="text-rose-300">Error</span> = failed step.</div>
                                <div>Progress bar shows completed steps across the Trail Talk pipeline.</div>
                              </div>
                              <div className="rounded-md border border-slate-800 bg-slate-950/70 p-2 space-y-1">
                                <div className="text-[10px] font-semibold text-slate-100">Evidence terms</div>
                                <div><span className="text-cyan-300">Selection source:</span> why a paper was included (validated source, ontology reference, or unvalidated candidate).</div>
                                <div><span className="text-cyan-300">Tags:</span> <span className="text-slate-200">pathway text match</span>, <span className="text-slate-200">msigdb reference</span>, <span className="text-slate-200">quickgo term reference</span>, <span className="text-slate-200">quickgo annotation reference</span>, <span className="text-slate-200">amigo reference</span>, <span className="text-slate-200">go ref reference</span>, <span className="text-slate-200">doi resolved reference</span>, <span className="text-slate-200">query candidate</span>.</div>
                              </div>
                            </div>
                          </details>

                          <div className="grid grid-cols-2 gap-2">
                            <button
                              onClick={() => setTrailUseNamingClue((v) => !v)}
                              aria-pressed={trailUseNamingClue}
                              className={`text-left rounded-md border px-2 py-1.5 transition-colors ${
                                trailUseNamingClue
                                  ? 'border-cyan-600/60 bg-cyan-950/25'
                                  : 'border-slate-700 bg-slate-900/60'
                              }`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="text-[10px] font-semibold text-slate-100">Naming clues</div>
                                <div className={`text-[10px] px-2 py-0.5 rounded-full border ${trailUseNamingClue ? 'border-cyan-500/60 text-cyan-200 bg-cyan-900/40' : 'border-slate-600 text-slate-400 bg-slate-900'}`}>
                                  {trailUseNamingClue ? 'Enabled' : 'Disabled'}
                                </div>
                              </div>
                              <div className="mt-0.5 text-[10px] text-slate-400">
                                Add GO/MSigDB IDs and curated references.
                              </div>
                            </button>
                            <button
                              onClick={() => setTrailMsigdbGoOnly((v) => !v)}
                              aria-pressed={trailMsigdbGoOnly}
                              className={`text-left rounded-md border px-2 py-1.5 transition-colors ${
                                trailMsigdbGoOnly
                                  ? 'border-purple-600/60 bg-purple-950/25'
                                  : 'border-slate-700 bg-slate-900/60'
                              }`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="text-[10px] font-semibold text-slate-100">Search Mode</div>
                                <div className={`text-[10px] px-2 py-0.5 rounded-full border ${trailMsigdbGoOnly ? 'border-purple-500/60 text-purple-200 bg-purple-900/40' : 'border-slate-600 text-slate-400 bg-slate-900'}`}>
                                  {trailMsigdbGoOnly ? 'Origin' : 'Origin + Context'}
                                </div>
                              </div>
                              <div className="mt-0.5 text-[10px] text-slate-400">
                                {trailMsigdbGoOnly
                                  ? 'Origin: Strict naming evidence only (GO/MSigDB clues and source references).'
                                  : 'Origin + Context: Includes naming evidence plus broader study papers that mention the pathway.'}
                              </div>
                            </button>
                          </div>

                          {trailTalkSteps.length > 0 && (
                            <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-2.5 space-y-2">
                              <button
                                onClick={() => setTrailStatusExpanded((v) => !v)}
                                className="w-full flex items-center justify-between gap-2 text-left"
                              >
                                <div>
                                  <div className="text-[11px] font-semibold text-cyan-300">Trail Talk Status</div>
                                  <div className="text-[10px] text-slate-400">
                                    {mountainOriginLoading
                                      ? `Running • ${trailTalkElapsedSec}s elapsed`
                                      : trailTalkStartedAt
                                        ? `Latest run • ${trailTalkElapsedSec}s`
                                        : 'Ready'}
                                  </div>
                                </div>
                                <div className="flex items-center gap-2 text-[10px]">
                                  <span className="inline-flex items-center gap-1 text-emerald-300"><CheckCircle2 className="w-3 h-3" /> {trailTalkStatusSummary.done}</span>
                                  <span className="inline-flex items-center gap-1 text-cyan-300"><Loader2 className={`w-3 h-3 ${trailTalkStatusSummary.running > 0 ? 'animate-spin' : ''}`} /> {trailTalkStatusSummary.running}</span>
                                  <span className="inline-flex items-center gap-1 text-rose-300"><AlertCircle className="w-3 h-3" /> {trailTalkStatusSummary.error}</span>
                                  <ChevronRight className={`w-4 h-4 text-slate-400 transition-transform ${trailStatusExpanded ? 'rotate-90' : ''}`} />
                                </div>
                              </button>
                              <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
                                <div
                                  className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                                  style={{ width: `${trailTalkStatusSummary.progressPct}%` }}
                                />
                              </div>
                              {trailStatusExpanded && (
                                <div className="space-y-1.5">
                                  {trailTalkSteps.map((stepInfo) => (
                                    <div key={stepInfo.key} className={`rounded-md border px-2 py-1.5 ${processingStatusClasses(stepInfo.status)}`}>
                                      <div className="flex flex-wrap items-center gap-2">
                                        <span className="inline-flex items-center justify-center w-5 h-5 rounded-full border border-slate-600 bg-slate-950/70 text-[10px] font-semibold text-slate-300">
                                          {stepInfo.id}
                                        </span>
                                        <ProcessingStatusIcon status={stepInfo.status} />
                                        <span className="text-[11px] font-semibold text-slate-100">{stepInfo.label}</span>
                                        <span className="ml-auto text-[10px] uppercase tracking-wide text-slate-300">{processingStatusLabel(stepInfo.status)}</span>
                                      </div>
                                      <div className="mt-1 text-[10px] text-slate-300 whitespace-pre-wrap leading-relaxed">{stepInfo.detail}</div>
                                      {trailTalkStartedAt && (
                                        <div className="text-[10px] text-slate-500 mt-0.5">
                                          T+{Math.max(0, Math.floor((stepInfo.updatedAt - trailTalkStartedAt) / 1000))}s
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                  {!!trailTalkWorkflowCode && (
                                    <details className="text-[10px] text-slate-400">
                                      <summary className="cursor-pointer select-none">Show TypeScript state view</summary>
                                      <pre className="mt-1 p-2 rounded border border-slate-800 bg-slate-950/80 text-slate-300 whitespace-pre-wrap break-words font-mono">
                                        {trailTalkWorkflowCode}
                                      </pre>
                                    </details>
                                  )}
                                </div>
                              )}
                            </div>
                          )}

                          {mountainOriginQuery && (
                            <div className="text-[10px] text-slate-500 break-words">Query: {mountainOriginQuery}</div>
                          )}
                          {mountainOriginQueries.length > 1 && (
                            <details className="text-[10px] text-slate-500">
                              <summary className="cursor-pointer select-none">Show local query set ({mountainOriginQueries.length})</summary>
                              <div className="mt-1 space-y-1">
                                {mountainOriginQueries.filter((q, idx) => !(idx === 0 && q === mountainOriginQuery)).map((q, idx) => (
                                  <div key={`${q}-${idx}`} className="break-words">{q}</div>
                                ))}
                              </div>
                            </details>
                          )}
                          {mountainOriginMessage && (
                            <div className={`text-[11px] ${mountainOriginExact ? 'text-green-300' : 'text-amber-300'}`}>
                              {mountainOriginMessage}
                            </div>
                          )}
                          {mountainNamingClue && (
                            <div className="text-[10px] text-slate-300 rounded-md border border-slate-800 bg-slate-900/70 px-2.5 py-2 space-y-1">
                              <div className="text-cyan-300 font-semibold">Naming clue</div>
                              <div>
                                Namespace: {mountainNamingClue.namespace || 'N/A'}
                                {mountainNamingClue.go_id ? ` | ${mountainNamingClue.go_id}` : ''}
                                {mountainNamingClue.go_term_name ? ` (${mountainNamingClue.go_term_name})` : ''}
                              </div>
                              {mountainNamingClue.go_definition && (
                                <div className="text-slate-400">{mountainNamingClue.go_definition}</div>
                              )}
                              {(mountainNamingClue.definition_references?.length || 0) > 0 && (
                                <div className="text-slate-500">
                                  Definition refs: {mountainNamingClue.definition_references?.slice(0, 4).join(', ')}
                                </div>
                              )}
                              <div className="flex flex-wrap gap-2">
                                {mountainNamingClue.msigdb_card_url && (
                                  <a
                                    href={mountainNamingClue.msigdb_card_url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-cyan-300 underline hover:text-cyan-200"
                                  >
                                    MSigDB card
                                  </a>
                                )}
                                {mountainNamingClue.quickgo_url && (
                                  <a
                                    href={mountainNamingClue.quickgo_url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-cyan-300 underline hover:text-cyan-200"
                                  >
                                    QuickGO term
                                  </a>
                                )}
                                {mountainNamingClue.amigo_term_url && (
                                  <a
                                    href={mountainNamingClue.amigo_term_url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-cyan-300 underline hover:text-cyan-200"
                                  >
                                    AmiGO term
                                  </a>
                                )}
                              </div>
                              {(mountainNamingClue.reference_pmids?.length || 0) > 0 && (
                                <div className="text-slate-400">Reference PMIDs: {mountainNamingClue.reference_pmids?.slice(0, 6).join(', ')}</div>
                              )}
                              {(() => {
                                const clueReferenceUrls = Array.from(
                                  new Set(
                                    (mountainNamingClue.reference_urls || [])
                                      .map((url) => String(url || '').trim())
                                      .filter(Boolean)
                                  )
                                );
                                if (clueReferenceUrls.length === 0) return null;
                                return (
                                  <details className="text-[10px] text-slate-400">
                                    <summary className="cursor-pointer select-none">Clue reference links ({clueReferenceUrls.length})</summary>
                                    <div className="mt-1 space-y-1 max-h-40 overflow-y-auto pr-1">
                                      {clueReferenceUrls.map((url, idx) => (
                                        <a key={`${url}-${idx}`} href={url} target="_blank" rel="noreferrer" className="block break-all text-cyan-300 underline hover:text-cyan-200">
                                          {url}
                                        </a>
                                      ))}
                                    </div>
                                  </details>
                                );
                              })()}
                            </div>
                          )}
                          <div className="flex flex-col gap-3 flex-1 min-h-0">
                            {!apiKey.trim() && (
                              <div className="rounded-lg border border-cyan-800/40 bg-slate-900 p-2.5 space-y-2">
                                <div className="text-[11px] text-cyan-300 font-semibold">Add API key to run Trail Talk agent search and relevance analysis</div>
                                <div className="grid grid-cols-12 gap-2">
                                  <select
                                    value={agentProvider}
                                    onChange={e => setAgentProvider(e.target.value as 'openai' | 'gemini' | 'claude')}
                                    className="col-span-4 text-xs bg-slate-950 border border-slate-700 rounded-md px-2 py-2 text-slate-200"
                                  >
                                    <option value="openai">OpenAI</option>
                                    <option value="gemini">Gemini</option>
                                    <option value="claude">Claude</option>
                                  </select>
                                  <input
                                    type="password"
                                    value={apiKey}
                                    onChange={e => setApiKey(e.target.value)}
                                    placeholder="API key"
                                    className="col-span-8 text-xs bg-slate-950 border border-slate-700 rounded-md px-2 py-2 text-slate-200 placeholder:text-slate-500"
                                  />
                                </div>
                                <div className="flex items-center justify-between gap-2 text-[10px] text-slate-400">
                                  <span>Tier: {effectiveAgentTurbo ? 'Turbo' : 'Standard'} ({turboModelLabel})</span>
                                  <button
                                    type="button"
                                    onClick={() => setAgentTurbo(prev => !prev)}
                                    className={`px-2 py-1 rounded border transition-all ${agentTurbo
                                      ? 'border-amber-500/60 bg-amber-500/15 text-amber-200'
                                      : 'border-slate-700 bg-slate-900 text-slate-300'
                                      }`}
                                  >
                                    {agentTurbo ? 'Turbo On' : 'Turbo Off'}
                                  </button>
                                </div>
                                <button
                                  onClick={runAiAnnotationNow}
                                  disabled={!apiKey.trim() || aiActionLoading}
                                  className="text-[11px] px-2.5 py-1.5 rounded-md border border-purple-600/50 bg-purple-900/30 text-purple-200 hover:bg-purple-800/40 disabled:opacity-50"
                                >
                                  {aiActionLoading ? 'Annotating...' : 'Run AI Annotation'}
                                </button>
                              </div>
                            )}

                            {pendingPaperForContext && (
                              <div ref={trailContextRef} className="rounded-lg border border-cyan-700/40 bg-cyan-950/15 p-3 space-y-3">
                                <div>
                                  <div className="text-xs font-semibold text-cyan-200">Please provide context before relevance analysis</div>
                                  <div className="text-[11px] text-slate-400 mt-1">
                                    Selected paper: <span className="text-slate-200">{pendingPaperForContext.title || `PMID ${pendingPaperForContext.pmid}`}</span>
                                  </div>
                                </div>
                                {pendingContextError && (
                                  <div className="text-[11px] text-rose-300 bg-rose-950/30 border border-rose-700/40 rounded-md px-2.5 py-2">
                                    {pendingContextError}
                                  </div>
                                )}
                                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-2">
                                  <div className="space-y-1">
                                    <label className="text-[10px] text-slate-300 flex items-center gap-1.5">
                                      <span>1. Disease or phenotype</span>
                                      <span className="inline-flex rounded-full border border-rose-500/50 bg-rose-950/35 px-1.5 py-0.5 text-[9px] text-rose-200 uppercase tracking-wide">Required</span>
                                    </label>
                                    <input
                                      value={mountainStudyContext.disease}
                                      onChange={(e) => {
                                        setPendingContextError('');
                                        setMountainStudyContext((prev) => ({ ...prev, disease: e.target.value }));
                                      }}
                                      placeholder="e.g., rheumatoid arthritis"
                                      className="w-full text-xs bg-slate-950 border border-slate-700 rounded-md px-2 py-1.5 text-slate-200 placeholder:text-slate-500"
                                    />
                                  </div>
                                  <div className="space-y-1">
                                    <label className="text-[10px] text-slate-300 flex items-center gap-1.5">
                                      <span>2. Tissue or cell type</span>
                                      <span className="inline-flex rounded-full border border-rose-500/50 bg-rose-950/35 px-1.5 py-0.5 text-[9px] text-rose-200 uppercase tracking-wide">Required</span>
                                    </label>
                                    <input
                                      value={mountainStudyContext.tissue}
                                      onChange={(e) => {
                                        setPendingContextError('');
                                        setMountainStudyContext((prev) => ({ ...prev, tissue: e.target.value }));
                                      }}
                                      placeholder="e.g., PBMC, tumor microenvironment"
                                      className="w-full text-xs bg-slate-950 border border-slate-700 rounded-md px-2 py-1.5 text-slate-200 placeholder:text-slate-500"
                                    />
                                  </div>
                                  <div className="space-y-1">
                                    <label className="text-[10px] text-slate-300 flex items-center gap-1.5">
                                      <span>3. Organism</span>
                                      <span className="inline-flex rounded-full border border-rose-500/50 bg-rose-950/35 px-1.5 py-0.5 text-[9px] text-rose-200 uppercase tracking-wide">Required</span>
                                    </label>
                                    <input
                                      value={mountainStudyContext.organism}
                                      onChange={(e) => {
                                        setPendingContextError('');
                                        setMountainStudyContext((prev) => ({ ...prev, organism: e.target.value }));
                                      }}
                                      placeholder="e.g., human, mouse"
                                      className="w-full text-xs bg-slate-950 border border-slate-700 rounded-md px-2 py-1.5 text-slate-200 placeholder:text-slate-500"
                                    />
                                  </div>
                                  <div className="space-y-1">
                                    <label className="text-[10px] text-slate-300 flex items-center gap-1.5">
                                      <span>4. Technology</span>
                                      <span className="inline-flex rounded-full border border-slate-600 bg-slate-900 px-1.5 py-0.5 text-[9px] text-slate-300 uppercase tracking-wide">Optional</span>
                                    </label>
                                    <input
                                      value={mountainStudyContext.technology}
                                      onChange={(e) => {
                                        setPendingContextError('');
                                        setMountainStudyContext((prev) => ({ ...prev, technology: e.target.value }));
                                      }}
                                      placeholder="e.g., bulk RNA-seq, scRNA-seq, spatial"
                                      className="w-full text-xs bg-slate-950 border border-slate-700 rounded-md px-2 py-1.5 text-slate-200 placeholder:text-slate-500"
                                    />
                                  </div>
                                  <div className="space-y-1">
                                    <label className="text-[10px] text-slate-300 flex items-center gap-1.5">
                                      <span>5. Cohort or comparison</span>
                                      <span className="inline-flex rounded-full border border-slate-600 bg-slate-900 px-1.5 py-0.5 text-[9px] text-slate-300 uppercase tracking-wide">Optional</span>
                                    </label>
                                    <input
                                      value={mountainStudyContext.cohort}
                                      onChange={(e) => {
                                        setPendingContextError('');
                                        setMountainStudyContext((prev) => ({ ...prev, cohort: e.target.value }));
                                      }}
                                      placeholder="e.g., disease vs control"
                                      className="w-full text-xs bg-slate-950 border border-slate-700 rounded-md px-2 py-1.5 text-slate-200 placeholder:text-slate-500"
                                    />
                                  </div>
                                  <div className="md:col-span-2 xl:col-span-3 space-y-1">
                                    <label className="text-[10px] text-slate-300 flex items-center gap-1.5">
                                      <span>6. Main biological question</span>
                                      <span className="inline-flex rounded-full border border-slate-600 bg-slate-900 px-1.5 py-0.5 text-[9px] text-slate-300 uppercase tracking-wide">Optional</span>
                                    </label>
                                    <textarea
                                      value={mountainStudyContext.notes}
                                      onChange={(e) => {
                                        setPendingContextError('');
                                        setMountainStudyContext((prev) => ({ ...prev, notes: e.target.value }));
                                      }}
                                      placeholder="e.g., Is this module linked to antiviral signaling in activated B cells?"
                                      rows={2}
                                      className="w-full text-xs bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-200 placeholder:text-slate-500 resize-y"
                                    />
                                  </div>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  <button
                                    onClick={runPendingPaperAnalysis}
                                    disabled={mountainPaperAnalysisLoading === String(pendingPaperForContext.pmid || pendingPaperForContext.doi || pendingPaperForContext.title || '').trim()}
                                    className="text-[11px] px-3 py-1.5 rounded-md border border-purple-700/50 bg-purple-900/30 text-purple-200 hover:bg-purple-800/40 disabled:opacity-50"
                                  >
                                    {mountainPaperAnalysisLoading === String(pendingPaperForContext.pmid || pendingPaperForContext.doi || pendingPaperForContext.title || '').trim() ? 'Analyzing...' : 'Analyze Relevance'}
                                  </button>
                                  <button
                                    onClick={() => {
                                      setPendingPaperForContext(null);
                                      setPendingContextError('');
                                    }}
                                    className="text-[11px] px-3 py-1.5 rounded-md border border-slate-700 bg-slate-900 text-slate-300 hover:text-white"
                                  >
                                    Skip for now
                                  </button>
                                </div>
                              </div>
                            )}

                            <div className="rounded-lg border border-slate-800 bg-slate-900 p-2.5 min-h-[280px] flex flex-col">
                              <div className="text-xs font-semibold text-slate-200">Evidence papers</div>
                              <div className="text-[10px] text-slate-500 mt-1">
                                {mountainOriginPapers.some((p) => String(p.selection_source || '') === 'candidate_pool_unvalidated')
                                  ? 'Showing unvalidated candidate papers from local retrieval (LLM did not validate direct origin evidence).'
                                  : 'Validated origin candidates plus clue-linked references. If none are reliable, Trail Talk returns no match.'}
                              </div>
                              <details className="text-[10px] text-slate-500 mt-1">
                                <summary className="cursor-pointer select-none">Tag guide</summary>
                                <div className="mt-1 text-slate-400">
                                  <span className="text-slate-300">pathway text match</span> = mentions pathway term,
                                  <span className="text-slate-300"> msigdb/quickgo reference</span> = clue-linked source,
                                  <span className="text-slate-300"> amigo/go ref reference</span> = ontology reference-chain source,
                                  <span className="text-slate-300"> query candidate</span> = generic query hit.
                                </div>
                              </details>
                              <div className="mt-2 flex-1 overflow-y-auto pr-1 space-y-2" onWheel={passScrollToPage}>
                                {mountainOriginPapers.length === 0 ? (
                                  <div className="text-xs text-slate-500">
                                    Click <span className="text-slate-300">Find Papers for This Pathway</span> to retrieve PubMed links, DOI, and abstracts for this pathway naming context.
                                  </div>
                                ) : (
                                  mountainOriginPapers.map((paper) => {
                                    const paperKey = String(paper.pmid || paper.doi || paper.title || '').trim();
                                    const analysis = mountainPaperAnalyses[paperKey];
                                    const pendingKey = String(pendingPaperForContext?.pmid || pendingPaperForContext?.doi || pendingPaperForContext?.title || '').trim();
                                    const isPending = pendingKey === paperKey;
                                    const abstractExpanded = Boolean(mountainExpandedAbstracts[paperKey]);
                                    const analysisExpanded = Boolean(mountainExpandedAnalyses[paperKey]);
                                    const parsedEvidence = analysis ? parseEvidenceAgentText(analysis.evidence) : null;
                                    const hypothesisText = analysis ? stripHypothesisPrefix(analysis.hypothesis) : '';
                                    return (
                                      <div key={`${paper.pmid}-${paper.doi || ''}`} className={`rounded-lg border p-2.5 text-xs ${isPending ? 'border-cyan-600/50 bg-cyan-950/15' : 'border-slate-800 bg-slate-950'}`}>
                                        <div className="flex items-start justify-between gap-2">
                                          <a
                                            href={paper.url || '#'}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="text-cyan-300 hover:text-cyan-200 underline font-medium"
                                          >
                                            {paper.title || `PMID ${paper.pmid}`}
                                          </a>
                                          <button
                                            onClick={() => openAnalyzeRelevancePrompt(paper)}
                                            className="shrink-0 text-[10px] px-2.5 py-1 rounded-md border border-purple-600/50 bg-gradient-to-r from-purple-800/30 to-fuchsia-800/30 text-purple-100 hover:from-purple-700/40 hover:to-fuchsia-700/40"
                                          >
                                            {isPending ? 'Please Provide Context' : 'Analyze Relevance'}
                                          </button>
                                        </div>
                                        <div className="text-slate-500 mt-1">{paper.journal || 'Journal N/A'} {paper.year ? `(${paper.year})` : ''}</div>
                                        {paper.selection_source && (
                                          <div className="text-[10px] text-slate-400 mt-1">
                                            Selection source: {paper.selection_source.replace(/_/g, ' ')}
                                          </div>
                                        )}
                                        {Array.isArray(paper.source_tags) && paper.source_tags.length > 0 && (
                                          <div className="flex flex-wrap gap-1 mt-1">
                                            {paper.source_tags.slice(0, 8).map((tag, idx) => (
                                              <span key={`${paperKey}-tag-${tag}-${idx}`} className="text-[10px] px-1.5 py-0.5 rounded border border-cyan-700/40 bg-cyan-950/25 text-cyan-200">
                                                {String(tag || '').replace(/_/g, ' ')}
                                              </span>
                                            ))}
                                          </div>
                                        )}
                                        {paper.selection_reason && (
                                          <div className="text-[10px] text-cyan-200 mt-1">
                                            {String(paper.selection_source || '') === 'candidate_pool_unvalidated' ? 'Candidate note' : 'LLM reason'}: {paper.selection_reason}
                                          </div>
                                        )}
                                        {paper.doi && (
                                          <a href={paper.doi_url || '#'} target="_blank" rel="noreferrer" className="text-slate-400 hover:text-slate-200 underline mt-1 block">
                                            DOI: {paper.doi}
                                          </a>
                                        )}
                                        {paper.pmcid && (
                                          <div className="text-[10px] text-slate-500 mt-1">PMCID: {paper.pmcid}</div>
                                        )}
                                        {paper.abstract && (
                                          <div className="mt-1.5">
                                            <button
                                              onClick={() => setMountainExpandedAbstracts((prev) => ({ ...prev, [paperKey]: !Boolean(prev[paperKey]) }))}
                                              className="text-[10px] px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300 hover:text-white hover:border-cyan-500/40"
                                            >
                                              {abstractExpanded ? 'Hide Abstract' : 'Show Abstract'}
                                            </button>
                                            {abstractExpanded && (
                                              <p className="text-slate-400 mt-1.5 leading-relaxed">{paper.abstract}</p>
                                            )}
                                          </div>
                                        )}
                                        {analysis && (
                                          <div className="mt-2 p-2 rounded border border-purple-800/40 bg-purple-950/20 space-y-2">
                                            <div className="flex flex-wrap items-center justify-between gap-2">
                                              <div className="text-[10px] text-purple-200">
                                                Module {analysis.moduleId ? `M${analysis.moduleId}` : 'N/A'} {analysis.moduleName ? `(${analysis.moduleName})` : ''}
                                              </div>
                                              <button
                                                onClick={() => setMountainExpandedAnalyses((prev) => ({ ...prev, [paperKey]: !Boolean(prev[paperKey]) }))}
                                                className="text-[10px] px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-200 hover:text-white hover:border-cyan-500/40"
                                              >
                                                {analysisExpanded ? 'Hide Analysis' : 'Show Analysis'}
                                              </button>
                                            </div>

                                            <div className="flex flex-wrap gap-1.5">
                                              {parsedEvidence?.verdict && (
                                                <span className={`text-[10px] px-1.5 py-0.5 rounded border ${
                                                  parsedEvidence.verdict.toLowerCase() === 'direct'
                                                    ? 'border-emerald-600/50 bg-emerald-950/30 text-emerald-200'
                                                    : parsedEvidence.verdict.toLowerCase() === 'indirect'
                                                      ? 'border-amber-600/50 bg-amber-950/30 text-amber-200'
                                                      : 'border-slate-700 bg-slate-900 text-slate-300'
                                                }`}>
                                                  Verdict: {parsedEvidence.verdict}
                                                </span>
                                              )}
                                              {parsedEvidence?.confidence && (
                                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-cyan-700/40 bg-cyan-950/25 text-cyan-200">
                                                  Confidence: {parsedEvidence.confidence}
                                                </span>
                                              )}
                                              <span className="text-[10px] px-1.5 py-0.5 rounded border border-purple-700/40 bg-purple-900/30 text-purple-200">
                                                DEG hits: {analysis.nHits} (up {analysis.nUp}, down {analysis.nDown})
                                              </span>
                                            </div>

                                            {parsedEvidence?.whyFound && (
                                              <div className="text-[10px] text-slate-300 leading-relaxed">
                                                <span className="text-slate-400">Why found:</span> {compactText(parsedEvidence.whyFound, 240)}
                                              </div>
                                            )}

                                            {analysisExpanded && (
                                              <div className="pt-2 border-t border-slate-800/80 space-y-2">
                                                {analysis.openAccessNote && (
                                                  <div className="text-[10px] text-slate-400">{analysis.openAccessNote}</div>
                                                )}
                                                {parsedEvidence?.whyFound && (
                                                  <div>
                                                    <div className="text-[10px] font-semibold text-cyan-300 mb-0.5">Evidence</div>
                                                    <div className="text-[11px] text-slate-300 whitespace-pre-wrap leading-relaxed">{parsedEvidence.whyFound}</div>
                                                  </div>
                                                )}
                                                {parsedEvidence?.studyRelation && (
                                                  <div>
                                                    <div className="text-[10px] font-semibold text-cyan-300 mb-0.5">Study Relation</div>
                                                    <div className="text-[11px] text-slate-300 whitespace-pre-wrap leading-relaxed">{parsedEvidence.studyRelation}</div>
                                                  </div>
                                                )}
                                                {parsedEvidence?.nextCheck && (
                                                  <div>
                                                    <div className="text-[10px] font-semibold text-cyan-300 mb-0.5">Next Check</div>
                                                    <div className="text-[11px] text-slate-300 whitespace-pre-wrap leading-relaxed">{parsedEvidence.nextCheck}</div>
                                                  </div>
                                                )}
                                                {hypothesisText && (
                                                  <div>
                                                    <div className="text-[10px] font-semibold text-fuchsia-300 mb-0.5">Hypothesis</div>
                                                    <div className="text-[11px] text-slate-300 whitespace-pre-wrap leading-relaxed">{hypothesisText}</div>
                                                  </div>
                                                )}
                                                {analysis.geneHits?.length > 0 && (
                                                  <details className="pt-1">
                                                    <summary className="cursor-pointer select-none text-[10px] text-slate-400">
                                                      DEG hit genes ({analysis.geneHits.length})
                                                    </summary>
                                                    <div className="flex flex-wrap gap-1 pt-2">
                                                      {analysis.geneHits.slice(0, 24).map((h, idx) => (
                                                        <span key={`${paperKey}-${h.gene}-${idx}`} className={`text-[10px] px-1.5 py-0.5 rounded border ${h.direction === 'up' ? 'bg-green-950/30 border-green-700/40 text-green-200' : h.direction === 'down' ? 'bg-red-950/30 border-red-700/40 text-red-200' : 'bg-slate-900 border-slate-700 text-slate-300'}`}>
                                                          {h.gene} {Number(h.logfc).toFixed(2)} ({h.module})
                                                        </span>
                                                      ))}
                                                    </div>
                                                  </details>
                                                )}
                                              </div>
                                            )}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })
                                )}
                              </div>
                            </div>

                          </div>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </section>
              )}

              {activeTab === '3d' && (
                <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 sm:p-5">
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <h3 className="text-sm font-semibold text-white">Gene Search</h3>
                    {geneSearchUpper && (
                      <span className="text-[11px] text-cyan-300">
                        {geneSearchMatches.length} match{geneSearchMatches.length === 1 ? '' : 'es'}
                      </span>
                    )}
                  </div>
                  <input
                    value={geneSearchQuery}
                    onChange={(e) => setGeneSearchQuery(e.target.value)}
                    placeholder="Search gene symbols across modules/pathways (e.g., STAT1)"
                    className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-xs text-slate-200 placeholder:text-slate-500"
                  />
                  {geneSearchUpper && (
                    <div className="mt-2 max-h-36 overflow-y-auto pr-1 space-y-1" onWheel={passScrollToPage}>
                      {geneSearchMatches.length === 0 ? (
                        <div className="text-[11px] text-slate-500">No matching genes were found in current module results.</div>
                      ) : (
                        geneSearchMatches.slice(0, 14).map((m) => (
                          <div key={`${m.gene}-${m.module}`} className="rounded border border-slate-800 bg-slate-950 px-2 py-1.5 text-[11px] text-slate-300">
                            <span className="font-semibold text-cyan-300">{m.gene}</span>
                            <span className="text-slate-500"> | </span>
                            <span>M{m.module}</span>
                            <span className="text-slate-500"> | </span>
                            <span>{Math.round(m.percentage)}%</span>
                            <span className={`ml-2 ${m.isDeg ? 'text-orange-300' : 'text-slate-500'}`}>{m.isDeg ? 'DEG' : 'non-DEG'}</span>
                            {m.pathways.length > 0 && (
                              <div className="text-[10px] text-slate-500 mt-0.5 truncate" title={m.pathways.join(', ')}>
                                {m.pathways.join(', ')}
                              </div>
                            )}
                          </div>
                        ))
                      )}
                    </div>
                  )}

                </section>
              )}

              {activeTab === '3d' && (
                <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 sm:p-5">
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                      <BrainCircuit className="w-4 h-4 text-cyan-400" />
                      Research Chat
                    </h3>
                    <span className="text-[10px] px-2 py-0.5 rounded-full border border-slate-700 bg-slate-950 text-slate-400">
                      Disabled
                    </span>
                  </div>
                  <div className="w-full rounded-lg border border-slate-800 bg-slate-950 p-3 text-xs text-slate-400 leading-relaxed">
                    Research Chat is currently disabled and will be available in future updates.
                  </div>
                </section>
              )}
            </div>
          )}
        </div>
      </div>

      {showMountainExportModal && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[106] animate-in fade-in duration-200"
          onClick={() => setShowMountainExportModal(false)}
        >
          <div
            className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-2xl m-4 p-5 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <h3 className="text-white text-base font-semibold">Mountain Plot Export</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Choose format, size, quality, and optional gene labels for the current Mountain Explorer panel.
                </p>
              </div>
              <button
                onClick={() => setShowMountainExportModal(false)}
                className="text-slate-400 hover:text-white p-1 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-2">
              <label className="block text-[11px] font-semibold text-slate-300">Format</label>
              <div className="grid grid-cols-5 gap-2">
                {(['png', 'jpeg', 'webp', 'svg', 'tiff'] as MountainExportFormat[]).map((fmt) => (
                  <button
                    key={fmt}
                    onClick={() => setMountainExportConfig((prev) => ({ ...prev, format: fmt }))}
                    className={`py-2 text-xs font-semibold rounded-md border transition-all ${
                      mountainExportConfig.format === fmt
                        ? 'bg-cyan-700/40 border-cyan-500 text-cyan-100'
                        : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-cyan-500/50'
                    }`}
                  >
                    {fmt.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            <div className="text-[11px] text-slate-400">Default size: 800 x 650</div>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-[11px] font-semibold text-slate-300 space-y-1">
                <span>Width (px)</span>
                <input
                  type="number"
                  min={300}
                  max={5200}
                  step={10}
                  value={mountainExportConfig.width}
                  onChange={(e) => {
                    const parsed = Number(e.target.value);
                    if (!Number.isFinite(parsed) || parsed <= 0) return;
                    const nextWidth = Math.round(parsed);
                    setMountainExportConfig((prev) => ({ ...prev, width: nextWidth }));
                  }}
                  className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-2 text-xs text-slate-100"
                />
              </label>
              <label className="text-[11px] font-semibold text-slate-300 space-y-1">
                <span>Height (px)</span>
                <input
                  type="number"
                  min={300}
                  max={3600}
                  step={10}
                  value={mountainExportConfig.height}
                  onChange={(e) => {
                    const parsed = Number(e.target.value);
                    if (!Number.isFinite(parsed) || parsed <= 0) return;
                    const nextHeight = Math.round(parsed);
                    setMountainExportConfig((prev) => ({ ...prev, height: nextHeight }));
                  }}
                  className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-2 text-xs text-slate-100"
                />
              </label>
            </div>
            <div className="space-y-2">
              <label className="block text-[11px] font-semibold text-slate-300">Quality</label>
              <div className="grid grid-cols-3 gap-2">
                {(['standard', 'high', 'ultra'] as const).map((quality) => (
                  <button
                    key={quality}
                    onClick={() => setMountainExportConfig((prev) => ({ ...prev, quality }))}
                    className={`py-2 text-xs font-semibold rounded-md border transition-all ${
                      mountainExportConfig.quality === quality
                        ? 'bg-cyan-700/40 border-cyan-500 text-cyan-100'
                        : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-cyan-500/50'
                    }`}
                  >
                    {quality === 'standard' ? 'Standard' : quality === 'high' ? 'High' : 'Ultra'}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <label className="block text-[11px] font-semibold text-slate-300">
                  Add pathway gene labels to export
                </label>
                <span className="text-[10px] text-slate-500">
                  Selected: {mountainExportSelectedGenes.length}
                </span>
              </div>
              <input
                value={mountainExportGeneSearch}
                onChange={(e) => setMountainExportGeneSearch(e.target.value)}
                placeholder="Search pathway genes..."
                className="w-full bg-slate-800 border border-slate-700 rounded-md px-2.5 py-2 text-xs text-slate-100 placeholder:text-slate-500"
              />
              <div className="flex items-center justify-end gap-2">
                <button
                  onClick={() => {
                    const allPlottable = mountainExportGeneOptions.filter((g) => g.rank !== null).map((g) => g.gene);
                    setMountainExportSelectedGenes(Array.from(new Set(allPlottable)));
                  }}
                  className="px-2.5 py-1 text-[10px] rounded border border-slate-700 bg-slate-800 text-slate-300 hover:text-white"
                >
                  Select all
                </button>
                <button
                  onClick={() => setMountainExportSelectedGenes([])}
                  className="px-2.5 py-1 text-[10px] rounded border border-slate-700 bg-slate-800 text-slate-300 hover:text-white"
                >
                  Clear
                </button>
              </div>
              <div className="max-h-44 overflow-y-auto space-y-1 rounded-md border border-slate-700 bg-slate-950 p-2">
                {filteredMountainExportGenes.length === 0 && (
                  <div className="text-[11px] text-slate-500 py-2 text-center">No genes match your search.</div>
                )}
                {filteredMountainExportGenes.map((g) => {
                  const checked = mountainExportSelectedGenes.includes(g.gene);
                  const disabled = g.rank === null;
                  return (
                    <label
                      key={g.gene}
                      className={`flex items-center justify-between gap-2 rounded border px-2 py-1.5 text-[11px] ${
                        disabled
                          ? 'border-slate-800 bg-slate-900/60 text-slate-500'
                          : 'border-slate-700 bg-slate-900 text-slate-200'
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={disabled}
                          onChange={(e) => {
                            const isChecked = e.target.checked;
                            setMountainExportSelectedGenes((prev) => {
                              if (isChecked) return Array.from(new Set([...prev, g.gene]));
                              return prev.filter((item) => item !== g.gene);
                            });
                          }}
                        />
                        <span className={`font-semibold ${g.isDeg ? 'text-orange-300' : 'text-slate-200'}`}>{g.gene}</span>
                        {g.isDeg && <span className="text-[10px] text-orange-300">DEG</span>}
                      </div>
                      <div className="text-[10px] text-slate-400">
                        {g.rank !== null ? `rank ${g.rank} | logFC ${Number(g.logfc || 0).toFixed(3)}` : 'not in ranked list'}
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>
            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={() => setShowMountainExportModal(false)}
                className="px-3 py-2 text-xs rounded-md border border-slate-700 bg-slate-800 text-slate-300 hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={() => { void exportSelectedMountainPlot(); }}
                disabled={!selectedMountainPathway}
                className="px-3 py-2 text-xs rounded-md border border-cyan-500/70 bg-cyan-700/30 text-cyan-100 hover:bg-cyan-600/30 disabled:opacity-50"
              >
                Export
              </button>
            </div>
          </div>
        </div>
      )}

      {showIssueReportPrompt && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[107] animate-in fade-in duration-200"
          onClick={() => setShowIssueReportPrompt(false)}
        >
          <div
            className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-xl m-4 p-5 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <h3 className="text-white text-base font-semibold">Report Issue</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Share a short description of what happened. Then export the debug log.
                </p>
              </div>
              <button
                onClick={() => setShowIssueReportPrompt(false)}
                className="text-slate-400 hover:text-white p-1 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="rounded-xl border border-cyan-700/40 bg-gradient-to-br from-slate-950 to-slate-900 p-4 space-y-2">
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-200">What happened? (include what you already tried)</label>
                <textarea
                  value={issueSummaryInput}
                  onChange={(e) => setIssueSummaryInput(e.target.value)}
                  rows={5}
                  placeholder="Example: I uploaded my CSV, clicked HTML Report, and nothing downloaded. I retried once and restarted GEMMAP, but the same issue happened."
                  className="w-full text-xs bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-slate-100 placeholder:text-slate-500 resize-y"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowIssueReportPrompt(false)}
                className="px-3 py-2 text-xs rounded-md border border-slate-700 bg-slate-800 text-slate-300 hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={() => { void submitIssueReportExport(); }}
                disabled={issueReportLoading}
                className="px-3 py-2 text-xs rounded-md border border-cyan-600/60 bg-cyan-900/30 text-cyan-100 hover:bg-cyan-800/30 disabled:opacity-70"
              >
                {issueReportLoading ? 'Exporting...' : 'Next: Export Debug Log'}
              </button>
            </div>
          </div>
        </div>
      )}

      {issueReportInfo && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[106] animate-in fade-in duration-200"
          onClick={() => setIssueReportInfo(null)}
        >
          <div
            className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-xl m-4 p-5 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <h3 className="text-white text-base font-semibold">Report Issue</h3>
                <p className="text-sm text-cyan-100 mt-1 leading-relaxed">
                  Thank you for taking the time to report this issue.
                </p>
                <p className="text-sm font-bold text-red-300 mt-1 leading-relaxed">
                  Please attach the downloaded report text file (.txt) and send this email.
                </p>
              </div>
              <button
                onClick={() => setIssueReportInfo(null)}
                className="text-slate-400 hover:text-white p-1 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="rounded-xl border border-cyan-700/40 bg-gradient-to-br from-slate-950 to-slate-900 p-4 space-y-3">
              <div className="text-xs space-y-2">
                <div className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2">
                  <div className="text-[11px] text-slate-500">To</div>
                  <div className="text-cyan-300 break-all">{issueReportInfo.email}</div>
                </div>
                <div className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2">
                  <div className="text-[11px] text-slate-500">Subject</div>
                  <div className="text-slate-200 break-all">{issueReportInfo.subject}</div>
                </div>
              </div>

              <div className="space-y-1">
                <div className="text-[11px] text-slate-400">Email body</div>
                <textarea
                  readOnly
                  value={buildIssueReportBody(issueReportInfo)}
                  rows={8}
                  className="w-full text-xs font-mono bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-slate-200 resize-y"
                />
              </div>

              <div className="flex flex-wrap gap-2 pt-1">
                <button
                  onClick={() => { void copyIssueReportEmail(); }}
                  className="px-2 py-1 rounded border border-slate-600 bg-slate-800 text-slate-200 hover:text-white text-[11px]"
                >
                  Copy Email Address
                </button>
                <button
                  onClick={() => { void copyIssueReportSubject(); }}
                  className="px-2 py-1 rounded border border-slate-600 bg-slate-800 text-slate-200 hover:text-white text-[11px]"
                >
                  Copy Subject
                </button>
                <button
                  onClick={() => { void copyIssueReportBody(); }}
                  className="px-2 py-1 rounded border border-slate-600 bg-slate-800 text-slate-200 hover:text-white text-[11px]"
                >
                  Copy Body (Optional)
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between gap-2">
              <div className="text-[11px] text-slate-500">
                Keep it simple. Short notes are enough.
              </div>
              {issueReportCopyStatus && <div className="text-[11px] text-cyan-300">{issueReportCopyStatus}</div>}
            </div>

            <div className="flex justify-end gap-2">
              <button
                onClick={openIssueReportMailClient}
                className="px-3 py-2 text-xs rounded-md border border-cyan-600/60 bg-cyan-900/30 text-cyan-100 hover:bg-cyan-800/30"
              >
                Open Email App
              </button>
              <button
                onClick={() => setIssueReportInfo(null)}
                className="px-3 py-2 text-xs rounded-md border border-slate-700 bg-slate-800 text-slate-300 hover:text-white"
              >
                Close
              </button>
              <button
                onClick={() => { void copyIssueReportTemplate(); }}
                className="px-3 py-2 text-xs rounded-md border border-cyan-600/60 bg-cyan-900/30 text-cyan-100 hover:bg-cyan-800/30"
              >
                Copy Template
              </button>
            </div>
          </div>
        </div>
      )}

      {quickExportPrompt && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[105] animate-in fade-in duration-200"
          onClick={() => setQuickExportPrompt(null)}
        >
          <div
            className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-md m-4 p-5 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <h3 className="text-white text-base font-semibold">
                  {quickExportPrompt === 'html' ? 'HTML Export Options' : 'Repro JSON Export Options'}
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Choose whether to include agent chat history in this export.
                </p>
              </div>
              <button
                onClick={() => setQuickExportPrompt(null)}
                className="text-slate-400 hover:text-white p-1 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-2">
              <button
                onClick={() => {
                  const type = quickExportPrompt;
                  setQuickExportPrompt(null);
                  if (type === 'html') void quickHtmlExport(false);
                  if (type === 'json') void quickReproExport(false);
                }}
                className="w-full text-left px-3 py-2 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-100 text-sm"
              >
                Export without chats
              </button>
              <button
                onClick={() => {
                  const type = quickExportPrompt;
                  setQuickExportPrompt(null);
                  if (type === 'html') void quickHtmlExport(true);
                  if (type === 'json') void quickReproExport(true);
                }}
                className="w-full text-left px-3 py-2 rounded-lg border border-cyan-600/60 bg-cyan-900/30 hover:bg-cyan-800/30 text-cyan-100 text-sm"
              >
                Export with chats
              </button>
              <button
                onClick={() => setQuickExportPrompt(null)}
                className="w-full text-left px-3 py-2 rounded-lg border border-slate-700 bg-slate-900 hover:bg-slate-800 text-slate-300 text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Modal */}
      {
        showExportModal && (
          <div
            className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[100] animate-in fade-in duration-200"
            onClick={() => setShowExportModal(false)}
          >
            <div
              className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-auto m-4 animate-in zoom-in-95 duration-200"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-slate-800">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
                    <Package className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-white">Publication Export</h2>
                    <p className="text-xs text-slate-400">Generate print-ready figures & data</p>
                  </div>
                </div>
                <button onClick={() => setShowExportModal(false)} className="text-slate-400 hover:text-white p-2 rounded-lg hover:bg-slate-800 transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Content */}
              <div className="p-6 space-y-6">

                {/* Quick Exports */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 text-cyan-400">
                    <Download className="w-4 h-4" />
                    <h3 className="font-bold text-sm">Quick Exports</h3>
                  </div>
                  <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-3">
                    <div className="mt-1 flex flex-wrap gap-2">
                      <button
                        onClick={() => {
                          setShowExportModal(false);
                          void quickXlsxExport();
                        }}
                        disabled={xlsxExportLoading}
                        className="px-3 py-2 text-xs rounded-md border border-slate-700 bg-slate-800 text-slate-200 hover:text-white disabled:opacity-60 flex items-center gap-1.5"
                      >
                        <FileSpreadsheet className="w-3.5 h-3.5" />
                        {xlsxExportLoading ? 'Exporting XLSX...' : 'Quick XLSX'}
                      </button>
                      <button
                        onClick={() => {
                          setShowExportModal(false);
                          void quickReproExport();
                        }}
                        disabled={jsonExportLoading}
                        className="px-3 py-2 text-xs rounded-md border border-slate-700 bg-slate-800 text-slate-200 hover:text-white disabled:opacity-60 flex items-center gap-1.5"
                      >
                        <FileText className="w-3.5 h-3.5" />
                        {jsonExportLoading ? 'Preparing JSON...' : 'Repro JSON'}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Report and Data Files */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 text-green-400">
                    <FileSpreadsheet className="w-4 h-4" />
                    <h3 className="font-bold text-sm">Report and Data Files</h3>
                  </div>

                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { key: 'include_html', label: 'Include HTML Report', desc: 'Interactive standalone report file' },
                      { key: 'include_json', label: 'Include JSON', desc: 'Adds with/without-chat reproducibility JSON files' },
                      { key: 'include_table', label: 'Include Table File', desc: 'Adds result tables to the package' },
                    ].map(item => (
                      <label
                        key={item.key}
                        className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all ${(exportConfig as any)[item.key]
                          ? 'bg-green-950/30 border-green-500/50'
                          : 'bg-slate-800/50 border-slate-700 hover:border-green-500/30'
                          }`}
                      >
                        <input
                          type="checkbox"
                          checked={(exportConfig as any)[item.key]}
                          onChange={e => setExportConfig({ ...exportConfig, [item.key]: e.target.checked })}
                          className="mt-1 accent-green-500"
                        />
                        <div>
                          <div className="text-sm text-white font-medium">{item.label}</div>
                          <div className="text-[10px] text-slate-500">{item.desc}</div>
                        </div>
                      </label>
                    ))}
                  </div>

                  {exportConfig.include_table && (
                    <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-3">
                      <label className="block text-[11px] font-bold text-slate-400 uppercase mb-2">Data File Format</label>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {[
                          { key: 'xlsx', label: 'Excel (.xlsx)', desc: 'Multiple sheets, formatting' },
                          { key: 'csv', label: 'CSV', desc: 'Comma-separated values' },
                          { key: 'tsv', label: 'TSV', desc: 'Tab-separated values' },
                          { key: 'h5ad', label: 'AnnData (.h5ad)', desc: 'Cross-ecosystem single-file export' },
                          { key: 'h5seurat', label: 'Seurat (.h5seurat)', desc: 'Native Seurat v5 file export' },
                        ].map(item => (
                          <button
                            key={item.key}
                            onClick={() => setExportConfig({ ...exportConfig, data_format: item.key })}
                            className={`p-2 rounded-lg border text-left transition-all ${exportConfig.data_format === item.key
                              ? 'bg-green-950/30 border-green-500/50'
                              : 'bg-slate-800/50 border-slate-700 hover:border-green-500/30'
                              }`}
                          >
                            <div className="text-xs text-white font-medium">{item.label}</div>
                            <div className="text-[10px] text-slate-500 leading-tight mt-0.5">{item.desc}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Figures to Include */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 text-purple-400">
                    <Image className="w-4 h-4" />
                    <h3 className="font-bold text-sm">Figures to Generate</h3>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { key: 'include_3d', label: '3D MDS Module Map (Fig 1)', desc: 'Main module landscape' },
                      { key: 'include_2d', label: '2D Projections (Fig 2)', desc: 'Three axis-pair projections' },
                      { key: 'include_elbow', label: 'Elbow & Silhouette (Fig 5)', desc: 'Optimal k selection plot' },
                      { key: 'include_heatmap', label: 'Gene Heatmap (Fig 4)', desc: 'Core gene frequency map' },
                      { key: 'include_barplots', label: 'Module Summary Panels (Fig 3, 6-10)', desc: 'Composition + NES/term panels' },
                    ].map(item => (
                      <label
                        key={item.key}
                        className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all ${(exportConfig as any)[item.key]
                          ? 'bg-purple-950/30 border-purple-500/50'
                          : 'bg-slate-800/50 border-slate-700 hover:border-purple-500/30'
                          }`}
                      >
                        <input
                          type="checkbox"
                          checked={(exportConfig as any)[item.key]}
                          onChange={e => setExportConfig({ ...exportConfig, [item.key]: e.target.checked })}
                          className="mt-1 accent-purple-500"
                        />
                        <div>
                          <div className="text-sm text-white font-medium">{item.label}</div>
                          <div className="text-[10px] text-slate-500">{item.desc}</div>
                        </div>
                      </label>
                    ))}
                  </div>

                  {hasSelectedPublicationFigure && (
                    <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-3 space-y-3">
                      <div className="flex items-center gap-2 text-cyan-300">
                        <FileImage className="w-4 h-4" />
                        <h4 className="text-xs font-bold uppercase tracking-wide">Image Settings (Figure Files)</h4>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Format</label>
                          <div className="grid grid-cols-5 gap-1">
                            {['png', 'jpeg', 'pdf', 'tiff', 'svg'].map(fmt => (
                              <button
                                key={fmt}
                                onClick={() => setExportConfig({ ...exportConfig, img_format: fmt })}
                                className={`py-2 px-2 text-xs font-medium rounded-lg border transition-all ${exportConfig.img_format === fmt
                                  ? 'bg-cyan-600 border-cyan-500 text-white'
                                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-cyan-500/50'
                                  }`}
                              >
                                {fmt.toUpperCase()}
                              </button>
                            ))}
                          </div>
                        </div>
                        <div>
                          <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Resolution (DPI)</label>
                          <div className="grid grid-cols-3 gap-2">
                            {[150, 300, 600].map(dpi => (
                              <button
                                key={dpi}
                                onClick={() => setExportConfig({ ...exportConfig, dpi })}
                                className={`py-2 px-3 text-xs font-medium rounded-lg border transition-all ${exportConfig.dpi === dpi
                                  ? 'bg-cyan-600 border-cyan-500 text-white'
                                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-cyan-500/50'
                                  }`}
                              >
                                {dpi} DPI
                                {dpi === 300 && <span className="text-[9px] block text-cyan-300">Standard</span>}
                                {dpi === 600 && <span className="text-[9px] block text-cyan-300">High-res</span>}
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* What's Included */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <BookOpen className="w-4 h-4 text-slate-400" />
                    <span className="text-xs font-bold text-slate-400 uppercase">Export Package Contents</span>
                  </div>
                  <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-slate-400">
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full"></span>
                      <span>figures/ - All selected figures</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 bg-green-400 rounded-full"></span>
                      <span>data/ - Module & gene tables (if enabled)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 bg-purple-400 rounded-full"></span>
                      <span>HTML reports - Interactive files (if enabled)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 bg-orange-400 rounded-full"></span>
                      <span>reproducibility/ - JSON manifests (if enabled)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 bg-slate-300 rounded-full"></span>
                      <span>README.txt + logs/ - Usage notes and debug log</span>
                    </div>
                  </div>
                </div>
              </div>

              {exportLoading && (
                <div className="px-6 pt-4 border-t border-slate-800 bg-slate-950/30">
                  <div className="flex items-center justify-between text-[11px] text-cyan-300">
                    <span>{exportStatus || 'Preparing export...'}</span>
                    <span>{Math.max(1, Math.min(100, Math.round(exportProgress)))}%</span>
                  </div>
                  <div className="mt-2 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                      style={{ width: `${Math.max(2, Math.min(100, exportProgress))}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Footer: Cancel + Publication Export */}
              <div className={`flex flex-wrap items-center justify-end gap-3 p-6 bg-slate-950/50 ${exportLoading ? '' : 'border-t border-slate-800'}`}>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowExportModal(false)}
                    className="text-sm text-slate-400 hover:text-white flex items-center gap-2 transition-colors px-3 py-2"
                  >
                    <X className="w-4 h-4" />
                    Cancel
                  </button>
                  <button
                    onClick={exportPublicationPackage}
                    disabled={exportLoading}
                    className={`px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/30 flex items-center gap-2 transition-all ${exportLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {exportLoading ? (
                      <><Loader2 className="w-4 h-4 animate-spin" /> Generating... {Math.max(1, Math.min(100, Math.round(exportProgress)))}%</>
                    ) : (
                      <><Package className="w-4 h-4" /> Publication Export (ZIP)</>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )
      }
    </div >
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
