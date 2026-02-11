import uvicorn
import pandas as pd
import io
import os
import shutil
import platform
import socket
import uuid
import time
import re
import asyncio
import json
import hashlib
import logging
from urllib.parse import urlencode, quote, unquote
from urllib.request import urlopen, Request
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
import xml.etree.ElementTree as ET
from typing import Dict, Optional, List, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.analysis import PathwayAnalyzer
# Lazy import graph builder to avoid crash if deps missing
try:
    from ..agents.graph import (
        build_annotation_graph,
        build_cluster_decider_graph,
        _create_llm,
        normalize_provider,
        get_provider_from_key,
    )
except ImportError:
    build_annotation_graph = None
    build_cluster_decider_graph = None
    _create_llm = None
    normalize_provider = None
    get_provider_from_key = None

APP_VERSION = "0.4.4"
RUNTIME_LOG_BUFFER = deque(maxlen=6000)
_LOG_CAPTURE_READY = False
REPORT_ISSUE_EMAIL_ENV = "GEMMAP_REPORT_EMAIL"
DEFAULT_REPORT_ISSUE_EMAIL = "abaghera@uab.edu"
TERMINAL_LOG_FILE_ENV = "GEMMAP_TERMINAL_LOG_FILE"
TERMINAL_RUN_ID_ENV = "GEMMAP_RUN_ID"
MAX_EXPORTED_LOG_LINES = 2500
HTML_EXPORT_IMAGE_DPI = 300
HTML_EXPORT_IMAGE_FORMAT = "png"
HTML_EXPORT_SIZE_NOTE = (
    "Embedded report images are generated at 300 DPI. "
    "For publication, export high-quality figures in your desired format from Publication Export in the app."
)


class _RuntimeLogHandler(logging.Handler):
    """Capture runtime logs in-memory so users can export a debug bundle."""

    def emit(self, record: logging.LogRecord):
        try:
            message = self.format(record)
            RUNTIME_LOG_BUFFER.append(message)
        except Exception:
            pass


def _init_runtime_log_capture():
    global _LOG_CAPTURE_READY
    if _LOG_CAPTURE_READY:
        return
    handler = _RuntimeLogHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    candidate_loggers = [
        logging.getLogger(),
        logging.getLogger("uvicorn.error"),
        logging.getLogger("uvicorn.access"),
        logging.getLogger("gemmap"),
    ]
    for lg in candidate_loggers:
        already = any(isinstance(h, _RuntimeLogHandler) for h in lg.handlers)
        if not already:
            lg.addHandler(handler)
    _LOG_CAPTURE_READY = True


def _log_event(message: str, level: int = logging.INFO):
    try:
        logging.getLogger("gemmap.app").log(level, message)
    except Exception:
        pass


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_df(df: Optional[pd.DataFrame]) -> str:
    if df is None or df.empty:
        return ""
    serial = df.fillna("").astype(str).to_csv(index=False)
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


def _redact_sensitive_text(value: str) -> str:
    if not isinstance(value, str) or not value:
        return value

    def _redact_ip_match(match: re.Match) -> str:
        ip_text = match.group(0)
        if ip_text.startswith("127.") or ip_text in {"0.0.0.0"}:
            return ip_text
        return "[REDACTED_IP]"

    def _redact_filename_match(match: re.Match) -> str:
        ext = str(match.group(2) or "").lower()
        return f"[REDACTED_FILE].{ext}" if ext else "[REDACTED_FILE]"

    redacted = value
    home_dir = str(Path.home())
    if home_dir and home_dir not in {"/", "."}:
        redacted = redacted.replace(home_dir, "~")
    hostname = str(socket.gethostname() or "").strip()
    if hostname:
        redacted = re.sub(rf"\b{re.escape(hostname)}\b", "[REDACTED_HOST]", redacted)
    patterns = [
        (r"\bsk-[A-Za-z0-9\-_]{16,}\b", "[REDACTED_API_KEY]"),
        (r"\bAIza[0-9A-Za-z\-_]{20,}\b", "[REDACTED_API_KEY]"),
        (r"\b(xox[baprs]-[A-Za-z0-9\-]{10,})\b", "[REDACTED_TOKEN]"),
        (r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "[REDACTED_TOKEN]"),
        (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]"),
        (r"([A-Za-z]:\\Users\\)[^\\\s]+", r"\1[REDACTED_USER]"),
        (r"/Users/[^/\s]+", "/Users/[REDACTED_USER]"),
        (r"/home/[^/\s]+", "/home/[REDACTED_USER]"),
        (r"(?i)([?&](?:token|api[_\s-]?key|key|access_token|refresh_token|signature|sig|auth)=)[^&\s]+", r"\1[REDACTED]"),
        (
            r"(?i)\b(api[_\s-]?key|access[_\s-]?token|refresh[_\s-]?token|authorization|bearer)\s*[:=]\s*[A-Za-z0-9\-_\.]{8,}",
            lambda m: f"{m.group(1)}=[REDACTED]",
        ),
    ]
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)
    redacted = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", _redact_ip_match, redacted)
    redacted = re.sub(
        r"\b([A-Za-z0-9][A-Za-z0-9._-]{1,120})\.(csv|tsv|txt|xls|xlsx|json|html|zip|log)\b",
        _redact_filename_match,
        redacted,
    )
    return redacted


def _scrub_sensitive_fields(obj: Any) -> Any:
    sensitive_keys = {
        "api_key",
        "apikey",
        "x_api_key",
        "authorization",
        "auth",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "client_secret",
        "password",
    }
    if isinstance(obj, dict):
        clean: Dict[str, Any] = {}
        for key, value in obj.items():
            key_text = str(key)
            if key_text.strip().lower() in sensitive_keys:
                continue
            clean[key] = _scrub_sensitive_fields(value)
        return clean
    if isinstance(obj, list):
        return [_scrub_sensitive_fields(v) for v in obj]
    if isinstance(obj, tuple):
        return [_scrub_sensitive_fields(v) for v in obj]
    if isinstance(obj, str):
        return _redact_sensitive_text(obj)
    return obj


def _get_issue_report_email() -> str:
    raw = str(os.getenv(REPORT_ISSUE_EMAIL_ENV, DEFAULT_REPORT_ISSUE_EMAIL) or "").strip()
    safe = re.sub(r"[\r\n]+", "", raw)
    return safe or DEFAULT_REPORT_ISSUE_EMAIL


def _hash_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "N/A"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _safe_file_summary(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "N/A"
    ext = Path(text).suffix.lower() or "none"
    return f"hash={_hash_text(text)} ext={ext}"


def _safe_env_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "N/A"
    lower = text.lower()
    if lower in {"gemmap", "base"}:
        return text
    return f"custom:{_hash_text(text)}"


def _safe_log_label(path_text: Any) -> str:
    text = str(path_text or "").strip()
    if not text:
        return "N/A"
    path_obj = Path(text)
    return path_obj.name or "N/A"


def _prepare_sanitized_log_lines(raw_lines: List[str]) -> Dict[str, Any]:
    if not raw_lines:
        return {"lines": [], "truncated": False, "total": 0}
    total = len(raw_lines)
    truncated = total > MAX_EXPORTED_LOG_LINES
    candidate = raw_lines[-MAX_EXPORTED_LOG_LINES:] if truncated else raw_lines
    return {
        "lines": [_redact_sensitive_text(str(line)) for line in candidate],
        "truncated": truncated,
        "total": total,
    }


def _build_issue_report_id(session_id: Optional[str]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_fragment = re.sub(r"[^A-Za-z0-9]", "", str(session_id or "nosession"))[:8] or "nosession"
    run_fragment = re.sub(r"[^A-Za-z0-9]", "", str(os.getenv(TERMINAL_RUN_ID_ENV, "") or ""))[:8] or "norun"
    return f"GMM-{APP_VERSION.replace('.', '')}-{timestamp}-{session_fragment}-{run_fragment}"


def _build_issue_report_subject(issue_id: str) -> str:
    return f"GEMMAP v{APP_VERSION} Debug Log Report | {issue_id}"


def _collect_system_debug_info() -> Dict[str, str]:
    tz_offset = datetime.now().astimezone().strftime("%z") or "N/A"
    return {
        "os": f"{platform.system()} {platform.release()}".strip(),
        "machine": platform.machine() or "N/A",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "conda_env": _safe_env_name(os.getenv("CONDA_DEFAULT_ENV", "")),
        "timezone_offset": tz_offset,
    }


def _read_terminal_log_capture() -> Dict[str, Any]:
    raw_path = str(os.getenv(TERMINAL_LOG_FILE_ENV, "") or "").strip()
    if not raw_path:
        return {"status": "not_set", "path": "", "size_bytes": 0, "content": ""}
    path = Path(raw_path).expanduser()
    if not path.exists():
        return {"status": "missing", "path": str(path), "size_bytes": 0, "content": ""}
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return {
            "status": "ok",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "content": content,
        }
    except Exception as exc:
        return {
            "status": "error",
            "path": str(path),
            "size_bytes": 0,
            "content": "",
            "error": str(exc),
        }


def _build_debug_log_export(
    session_id: Optional[str],
    session: Optional[Dict[str, Any]],
    issue_summary: Optional[str] = None,
    attempted_action: Optional[str] = None,
) -> Dict[str, str]:
    issue_id = _build_issue_report_id(session_id)
    issue_subject = _build_issue_report_subject(issue_id)
    issue_email = _get_issue_report_email()
    session_ref = _hash_text(session_id)
    terminal_run_raw = str(os.getenv(TERMINAL_RUN_ID_ENV, "") or "").strip()
    terminal_run_ref = _hash_text(terminal_run_raw)
    runtime_export = _prepare_sanitized_log_lines([str(line) for line in list(RUNTIME_LOG_BUFFER)])
    terminal_log = _read_terminal_log_capture()
    terminal_export = _prepare_sanitized_log_lines(str(terminal_log.get("content") or "").splitlines())
    system_info = _collect_system_debug_info()
    issue_summary_clean = _redact_sensitive_text(str(issue_summary or "").strip())
    attempted_action_clean = _redact_sensitive_text(str(attempted_action or "").strip())
    issue_details_clean = issue_summary_clean or attempted_action_clean or "N/A"

    lines = [
        "# GEMMAP Debug Log Export",
        f"generated_at_utc: {_now_utc_iso()}",
        f"gemmap_version: {APP_VERSION}",
        f"issue_report_id: {issue_id}",
        "privacy_mode: strict_sanitized_export",
        "",
        "## Report Issue (Copy/Paste)",
        f"To: {issue_email}",
        f"Subject: {issue_subject}",
        "",
        "Body template:",
        f"Issue details: {issue_details_clean}",
        "",
        "Attach the downloaded report text file (.txt) when sending your issue report.",
        "",
        "## Runtime Summary",
        f"session_ref: {session_ref}",
        f"terminal_run_ref: {terminal_run_ref}",
        f"terminal_log_file: {_safe_log_label(terminal_log.get('path'))}",
        f"terminal_log_status: {terminal_log.get('status')}",
        f"terminal_log_size_bytes: {terminal_log.get('size_bytes', 0)}",
    ]

    if session is None:
        lines.append("session_snapshot: not available (no active session).")
    else:
        lines.append(f"input_file: {_safe_file_summary(session.get('filename'))}")
        lines.append(f"report_name_hash: {_hash_text(_session_report_basename(session))}")
        base_df = session.get("df")
        if isinstance(base_df, pd.DataFrame):
            lines.append(f"input_rows: {len(base_df)}")
            lines.append(f"input_columns: {len(base_df.columns)}")
        analyzer = session.get("analyzer")
        if analyzer is not None and analyzer.cluster_results is not None:
            lines.append(f"n_pathways: {len(analyzer.cluster_results)}")
            if "Cluster" in analyzer.cluster_results.columns:
                lines.append(f"n_clusters: {int(analyzer.cluster_results['Cluster'].nunique())}")
            lines.append(f"score_col: {analyzer.score_col}")
            lines.append(f"has_deg_overlay: {session.get('deg_df') is not None}")

    lines.append("")
    lines.append("## System Information")
    for key, value in system_info.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("## In-Memory Runtime Logs (sanitized)")
    if len(runtime_export["lines"]) == 0:
        lines.append("(No runtime logs captured yet.)")
    else:
        if runtime_export["truncated"]:
            lines.append(
                f"(Showing last {MAX_EXPORTED_LOG_LINES} of {runtime_export['total']} captured runtime lines.)"
            )
        lines.extend(runtime_export["lines"])

    lines.append("")
    lines.append("## Terminal Logs (sanitized)")
    if terminal_log.get("status") != "ok":
        lines.append(f"(Terminal log unavailable: {terminal_log.get('status')})")
        if terminal_log.get("error"):
            lines.append(f"error: {_redact_sensitive_text(str(terminal_log.get('error')))}")
    else:
        if len(terminal_export["lines"]) == 0:
            lines.append("(Terminal log file is empty.)")
        else:
            if terminal_export["truncated"]:
                lines.append(
                    f"(Showing last {MAX_EXPORTED_LOG_LINES} of {terminal_export['total']} captured terminal lines.)"
                )
            lines.extend(terminal_export["lines"])

    lines.append("")
    lines.append("## End")

    safe_issue_id = re.sub(r"[^A-Za-z0-9._-]+", "_", issue_id)
    filename = f"gemmap_issue_report_{safe_issue_id}.txt"
    payload = "\n".join(lines)
    return {
        "payload": payload,
        "filename": filename,
        "issue_id": issue_id,
        "issue_subject": issue_subject,
        "issue_email": issue_email,
    }


def _normalize_gene_list(raw_value) -> List[str]:
    if isinstance(raw_value, list):
        return [str(g).strip() for g in raw_value if str(g).strip()]
    if raw_value is None:
        return []
    text = str(raw_value)
    for sep in [";", "|"]:
        text = text.replace(sep, ",")
    return [g.strip() for g in text.split(",") if g.strip()]


def _normalize_col_name(col_name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col_name).lower().strip())


def _detect_adjusted_pvalue_column(df: pd.DataFrame) -> Optional[str]:
    adjusted_candidates = {
        "padj",
        "adjp",
        "adjpval",
        "adjpvalue",
        "adjustedp",
        "adjustedpvalue",
        "adjustedpval",
        "fdr",
        "fdrbh",
        "qvalue",
        "qval",
        "adjpvaluebh",
    }
    for col in df.columns:
        if _normalize_col_name(col) in adjusted_candidates:
            return col
    return None


def _detect_nominal_pvalue_column(df: pd.DataFrame, exclude_col: Optional[str] = None) -> Optional[str]:
    nominal_candidates = {
        "pvalue",
        "pval",
        "p",
        "nominalp",
        "rawp",
        "rawpvalue",
    }
    for col in df.columns:
        if exclude_col is not None and str(col) == str(exclude_col):
            continue
        if _normalize_col_name(col) in nominal_candidates:
            return col
    return None


def _build_mountain_data(
    analyzer: PathwayAnalyzer,
    cluster_labels: Dict[str, str],
    deg_df: Optional[pd.DataFrame] = None,
    deg_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build pathway payload + ranked DEG genes for mountain explorer."""
    cluster_df = analyzer.cluster_results if analyzer.cluster_results is not None else pd.DataFrame()
    if cluster_df.empty:
        return {"pathways": [], "ranked_genes": [], "has_data": False}

    # Ranked genes from DEG table (logFC descending)
    ranked_gene_payload: List[Dict[str, Any]] = []
    deg_gene_col = (deg_config or {}).get("gene_col", "gene")
    deg_lfc_col = (deg_config or {}).get("lfc_col", "log2FC")
    if isinstance(deg_df, pd.DataFrame) and deg_gene_col in deg_df.columns and deg_lfc_col in deg_df.columns:
        deg_plot_df = deg_df[[deg_gene_col, deg_lfc_col]].copy()
        deg_plot_df.columns = ["Gene", "LogFC"]
        deg_plot_df["Gene"] = deg_plot_df["Gene"].astype(str)
        deg_plot_df["LogFC"] = pd.to_numeric(deg_plot_df["LogFC"], errors="coerce")
        deg_plot_df = deg_plot_df.dropna(subset=["Gene", "LogFC"])
        if not deg_plot_df.empty:
            deg_plot_df = deg_plot_df.groupby("Gene", as_index=False)["LogFC"].mean()
            ranked_gene_payload = [
                {"gene": g, "logfc": float(v)}
                for g, v in deg_plot_df.sort_values("LogFC", ascending=False)[["Gene", "LogFC"]].itertuples(index=False)
            ]

    # Map pathway -> leading-edge genes
    pathway_gene_map: Dict[str, List[str]] = {}
    try:
        raw_subset = analyzer.raw_data[[analyzer.pathway_col, analyzer.genes_col]].drop_duplicates(
            subset=[analyzer.pathway_col]
        )
        for _, row in raw_subset.iterrows():
            pathway_gene_map[str(row[analyzer.pathway_col])] = _normalize_gene_list(row[analyzer.genes_col])
    except Exception:
        pathway_gene_map = {}

    score_col_name = analyzer.score_col if analyzer.score_col and analyzer.score_col in cluster_df.columns else None
    adj_p_value_col_name = _detect_adjusted_pvalue_column(cluster_df)
    nominal_p_value_col_name = _detect_nominal_pvalue_column(cluster_df, exclude_col=adj_p_value_col_name)
    p_value_col_name = nominal_p_value_col_name or adj_p_value_col_name
    pathway_payload = []
    for idx, row in cluster_df.reset_index(drop=True).iterrows():
        pathway_name = str(row[analyzer.pathway_col])
        cluster_id = int(row["Cluster"])
        module_name = cluster_labels.get(str(cluster_id), f"Module {cluster_id}")
        nes_val = float(row[score_col_name]) if score_col_name else 0.0
        p_val = None
        if p_value_col_name is not None:
            try:
                p_tmp = pd.to_numeric(pd.Series([row[p_value_col_name]]), errors="coerce").iloc[0]
                if pd.notna(p_tmp):
                    p_val = float(p_tmp)
            except Exception:
                p_val = None

        adj_p_val = None
        if adj_p_value_col_name is not None:
            try:
                adj_tmp = pd.to_numeric(pd.Series([row[adj_p_value_col_name]]), errors="coerce").iloc[0]
                if pd.notna(adj_tmp):
                    adj_p_val = float(adj_tmp)
            except Exception:
                adj_p_val = None

        pathway_payload.append({
            "idx": idx + 1,
            "pathway": pathway_name,
            "cluster": cluster_id,
            "module": module_name,
            "nes": nes_val,
            "p_value": p_val,
            "adj_p_value": adj_p_val,
            "genes": pathway_gene_map.get(pathway_name, []),
        })

    return {
        "pathways": pathway_payload,
        "ranked_genes": ranked_gene_payload,
        "has_data": bool(pathway_payload) and bool(ranked_gene_payload),
    }


def _pathway_label_to_query(pathway_name: str) -> str:
    raw = str(pathway_name or "").strip()
    if not raw:
        return ""
    cleaned = raw.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_html(raw_html: str) -> str:
    text = str(raw_html or "")
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_pmids_from_text(text: str) -> List[str]:
    matches = re.findall(r"PMID[:\s]*([0-9]{5,10})", str(text or ""), flags=re.IGNORECASE)
    return list(dict.fromkeys(matches))


def _extract_dois_from_text(text: str) -> List[str]:
    matches = re.findall(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", str(text or ""), flags=re.IGNORECASE)
    return list(dict.fromkeys(matches))


def _extract_hrefs_from_html(raw_html: str) -> List[str]:
    html = str(raw_html or "")
    urls: List[str] = []
    for match in re.finditer(r"""href\s*=\s*["']([^"']+)["']""", html, flags=re.IGNORECASE):
        href = str(match.group(1) or "").strip()
        if not href:
            continue
        href = href.replace("&amp;", "&")
        if href.startswith("//"):
            href = f"https:{href}"
        if href.startswith("http://") or href.startswith("https://"):
            urls.append(href)
    return list(dict.fromkeys(urls))


def _extract_pmids_from_urls(urls: List[str]) -> List[str]:
    pmids: List[str] = []
    for raw_url in urls or []:
        text = unquote(str(raw_url or ""))
        pmids.extend(re.findall(r"(?:pubmed\.ncbi\.nlm\.nih\.gov|/pubmed/)([0-9]{5,10})", text, flags=re.IGNORECASE))
        pmids.extend(re.findall(r"[?&]uid=([0-9]{5,10})", text, flags=re.IGNORECASE))
        pmids.extend(_extract_pmids_from_text(text))
    return list(dict.fromkeys([p for p in pmids if re.fullmatch(r"[0-9]{5,10}", str(p).strip())]))


def _extract_dois_from_urls(urls: List[str]) -> List[str]:
    dois: List[str] = []
    for raw_url in urls or []:
        text = unquote(str(raw_url or ""))
        dois.extend(_extract_dois_from_text(text))
    return list(dict.fromkeys([d for d in dois if str(d).strip()]))


def _is_relevant_reference_url(url: str) -> bool:
    text = str(url or "").strip().lower()
    if not text or text.startswith("javascript:") or text.startswith("mailto:"):
        return False
    if "docs.gsea-msigdb.org" in text:
        return False
    if "#contact" in text:
        return False
    # Exclude gene record pages from clue reference links; they are not citation evidence.
    if "/gene/" in text or "ncbi.nlm.nih.gov/gene" in text or "view.ncbi.nlm.nih.gov/gene" in text:
        return False

    citation_tokens = (
        "pubmed.ncbi.nlm.nih.gov",
        "/pubmed/",
        "doi.org/",
        "europepmc.org/article/",
        "pmc.ncbi.nlm.nih.gov/articles/",
        "geneontology.org/go_ref/",
        "purl.obolibrary.org/obo/go/references/",
        "amigo.geneontology.org/amigo/reference/",
    )
    if any(token in text for token in citation_tokens):
        return True

    # Keep URLs that encode a PMID/DOI even if host/path is unconventional.
    if _extract_pmids_from_text(unquote(text)):
        return True
    if _extract_dois_from_text(unquote(text)):
        return True
    return False


def _extract_go_ref_ids(raw_text: str, urls: Optional[List[str]] = None) -> List[str]:
    text = str(raw_text or "")
    ids: List[str] = []
    for m in re.findall(r"\bGO[_:\-\s]?REF[_:\-\s]?([0-9]{1,7})\b", text, flags=re.IGNORECASE):
        ids.append(str(m).zfill(7))
    for url in urls or []:
        u = unquote(str(url or ""))
        for m in re.findall(r"/GO_REF/([0-9]{1,7})", u, flags=re.IGNORECASE):
            ids.append(str(m).zfill(7))
        for m in re.findall(r"/go/references/([0-9]{1,7})", u, flags=re.IGNORECASE):
            ids.append(str(m).zfill(7))
    return [f"GO_REF:{gid}" for gid in dict.fromkeys([g for g in ids if str(g).strip()])]


def _fetch_go_ref_reference_payload(go_ref_id: str) -> Dict[str, Any]:
    go_ref_id = str(go_ref_id or "").strip().upper()
    digits = re.sub(r"[^0-9]", "", go_ref_id)
    digits = digits.zfill(7) if digits else ""
    canonical = f"GO_REF:{digits}" if digits else go_ref_id
    url_candidates = []
    if digits:
        url_candidates.append(f"https://geneontology.org/GO_REF/{digits}")
        url_candidates.append(f"https://purl.obolibrary.org/obo/go/references/{digits}")
    if go_ref_id and go_ref_id not in {canonical, digits}:
        url_candidates.append(f"https://geneontology.org/{quote(go_ref_id)}")
    url_candidates = list(dict.fromkeys([u for u in url_candidates if u]))

    out = {
        "go_ref_id": canonical,
        "go_ref_url": url_candidates[0] if url_candidates else "",
        "pmids": [],
        "dois": [],
        "reference_urls": [],
    }
    if not url_candidates:
        return out

    for ref_url in url_candidates:
        try:
            req = Request(ref_url, headers={"User-Agent": "GEMMAP/0.4"})
            with urlopen(req, timeout=10) as resp:
                html_raw = resp.read().decode("utf-8", errors="ignore")
            if not html_raw:
                continue
            hrefs = _extract_hrefs_from_html(html_raw)
            plain = _strip_html(html_raw)
            pmids = list(dict.fromkeys(
                _extract_pmids_from_text(plain) +
                _extract_pmids_from_urls(hrefs)
            ))
            dois = list(dict.fromkeys(
                _extract_dois_from_text(plain) +
                _extract_dois_from_urls(hrefs)
            ))
            ref_urls = [u for u in hrefs if _is_relevant_reference_url(u)]
            if ref_url not in ref_urls:
                ref_urls.insert(0, ref_url)
            out["pmids"] = pmids[:40]
            out["dois"] = dois[:20]
            out["reference_urls"] = list(dict.fromkeys(ref_urls))[:40]
            out["go_ref_url"] = ref_url
            if out["pmids"] or out["dois"] or out["reference_urls"]:
                break
        except Exception:
            continue
    return out


def _fetch_amigo_go_ref_clue(go_id: str, max_go_refs: int = 10) -> Dict[str, Any]:
    go_id = str(go_id or "").strip().upper()
    amigo_term_url = f"https://amigo.geneontology.org/amigo/term/{quote(go_id)}" if go_id else ""
    out: Dict[str, Any] = {
        "found": False,
        "amigo_term_url": amigo_term_url,
        "go_ref_ids": [],
        "go_ref_urls": [],
        "amigo_term_reference_pmids": [],
        "go_ref_reference_pmids": [],
        "reference_pmids": [],
        "reference_dois": [],
        "reference_urls": [],
    }
    if not go_id:
        return out
    try:
        req = Request(amigo_term_url, headers={"User-Agent": "GEMMAP/0.4"})
        with urlopen(req, timeout=10) as resp:
            html_raw = resp.read().decode("utf-8", errors="ignore")
        if not html_raw:
            return out
        hrefs = _extract_hrefs_from_html(html_raw)
        plain = _strip_html(html_raw)
        out["found"] = True

        amigo_pmids = list(dict.fromkeys(
            _extract_pmids_from_text(plain) +
            _extract_pmids_from_urls(hrefs)
        ))
        amigo_dois = list(dict.fromkeys(
            _extract_dois_from_text(plain) +
            _extract_dois_from_urls(hrefs)
        ))
        reference_urls = [u for u in hrefs if _is_relevant_reference_url(u)]

        go_ref_ids = _extract_go_ref_ids(f"{plain} {' '.join(hrefs)}", hrefs)[: max(1, min(int(max_go_refs or 10), 20))]
        go_ref_urls: List[str] = []
        go_ref_pmids: List[str] = []
        go_ref_dois: List[str] = []
        for ref_id in go_ref_ids:
            ref_payload = _fetch_go_ref_reference_payload(ref_id)
            ref_url = str(ref_payload.get("go_ref_url") or "").strip()
            if ref_url:
                go_ref_urls.append(ref_url)
            go_ref_pmids.extend([str(p).strip() for p in (ref_payload.get("pmids") or []) if str(p).strip()])
            go_ref_dois.extend([str(d).strip() for d in (ref_payload.get("dois") or []) if str(d).strip()])
            reference_urls.extend([str(u).strip() for u in (ref_payload.get("reference_urls") or []) if str(u).strip()])

        out["go_ref_ids"] = go_ref_ids
        out["go_ref_urls"] = list(dict.fromkeys(go_ref_urls))[:20]
        out["amigo_term_reference_pmids"] = list(dict.fromkeys(amigo_pmids))[:40]
        out["go_ref_reference_pmids"] = list(dict.fromkeys(go_ref_pmids))[:40]
        out["reference_pmids"] = list(dict.fromkeys(amigo_pmids + go_ref_pmids))[:60]
        out["reference_dois"] = list(dict.fromkeys(amigo_dois + go_ref_dois))[:40]
        out["reference_urls"] = list(dict.fromkeys(reference_urls))[:80]
        return out
    except Exception:
        return out


def _collect_reference_ids(payload: Any) -> Dict[str, List[str]]:
    """
    Recursively collect PMID/DOI references from nested JSON-like payloads.
    Handles common xref shapes used by ontology APIs.
    """
    pmids: List[str] = []
    dois: List[str] = []

    def _add_pmid(value: Any):
        text = str(value or "").strip()
        if re.fullmatch(r"[0-9]{5,10}", text):
            pmids.append(text)

    def _add_doi(value: Any):
        text = str(value or "").strip()
        if not text:
            return
        for doi in _extract_dois_from_text(text):
            dois.append(doi)

    def _walk(node: Any, parent_key: str = ""):
        if isinstance(node, dict):
            db_val = str(
                node.get("db")
                or node.get("database")
                or node.get("dbName")
                or node.get("source")
                or ""
            ).strip().lower()
            id_val = node.get("id", node.get("value", node.get("xref", node.get("accession", ""))))
            if db_val in {"pmid", "pubmed", "medline"}:
                _add_pmid(id_val)
            elif db_val in {"doi"}:
                _add_doi(id_val)
            for k, v in node.items():
                key = str(k or "").strip().lower()
                if key in {"pmid", "pubmed", "pubmedid"}:
                    _add_pmid(v)
                elif key in {"doi"}:
                    _add_doi(v)
                _walk(v, key)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item, parent_key)
            return
        if isinstance(node, tuple):
            for item in node:
                _walk(item, parent_key)
            return
        if isinstance(node, str):
            if parent_key in {"pmid", "pubmed", "pubmedid"}:
                _add_pmid(node)
            elif parent_key in {"doi"}:
                _add_doi(node)
            for pmid in _extract_pmids_from_text(node):
                pmids.append(pmid)
            for doi in _extract_dois_from_text(node):
                dois.append(doi)
            return
        if parent_key in {"pmid", "pubmed", "pubmedid"}:
            _add_pmid(node)
        elif parent_key in {"doi"}:
            _add_doi(node)

    _walk(payload)
    return {
        "pmids": list(dict.fromkeys([p for p in pmids if str(p).strip()])),
        "dois": list(dict.fromkeys([d for d in dois if str(d).strip()])),
    }


def _detect_pathway_namespace(pathway_name: str) -> Dict[str, Any]:
    raw = str(pathway_name or "").strip()
    upper = raw.upper()
    mapping = [
        ("GOBP_", "MSigDB GO Biological Process (C5:GO:BP)", True),
        ("GOCC_", "MSigDB GO Cellular Component (C5:GO:CC)", True),
        ("GOMF_", "MSigDB GO Molecular Function (C5:GO:MF)", True),
        ("KEGG_", "MSigDB KEGG", False),
        ("REACTOME_", "MSigDB Reactome", False),
        ("HALLMARK_", "MSigDB Hallmark", False),
    ]
    for prefix, namespace, is_go in mapping:
        if upper.startswith(prefix):
            remainder = raw[len(prefix):]
            return {
                "prefix": prefix.rstrip("_"),
                "namespace": namespace,
                "is_msigdb_go": is_go,
                "go_label_guess": _pathway_label_to_query(remainder),
            }
    return {
        "prefix": "",
        "namespace": "Unknown/Custom",
        "is_msigdb_go": False,
        "go_label_guess": _pathway_label_to_query(raw),
    }


def _search_quickgo_go_id(term_label: str) -> Dict[str, str]:
    label = _pathway_label_to_query(term_label)
    if not label:
        return {}
    try:
        params = urlencode({
            "query": label,
            "limit": 10,
            "page": 1,
        })
        url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/search?{params}"
        req = Request(
            url,
            headers={"User-Agent": "GEMMAP/0.4", "Accept": "application/json"},
        )
        with urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        results = (payload or {}).get("results") or []
        if not isinstance(results, list):
            return {}
        target = label.lower()
        best = None
        for rec in results:
            go_id = str((rec or {}).get("id") or "").strip().upper()
            name = str((rec or {}).get("name") or "").strip()
            if not go_id.startswith("GO:"):
                continue
            if _pathway_label_to_query(name).lower() == target:
                best = {"go_id": go_id, "term_name": name}
                break
            if best is None:
                best = {"go_id": go_id, "term_name": name}
        return best or {}
    except Exception:
        return {}


def _fetch_msigdb_card_clue(pathway_name: str) -> Dict[str, Any]:
    raw = str(pathway_name or "").strip().upper()
    card_url = f"https://www.gsea-msigdb.org/gsea/msigdb/cards/{quote(raw)}"
    clue = {
        "found": False,
        "card_url": card_url,
        "go_id": "",
        "brief_description": "",
        "reference_pmids": [],
        "reference_dois": [],
        "reference_urls": [],
        "reference_link_pmids": [],
        "reference_link_dois": [],
    }
    if not raw:
        return clue
    try:
        req = Request(card_url, headers={"User-Agent": "GEMMAP/0.4"})
        with urlopen(req, timeout=10) as resp:
            html_raw = resp.read().decode("utf-8", errors="ignore")
        if not html_raw:
            return clue
        href_urls = _extract_hrefs_from_html(html_raw)
        ref_urls = [u for u in href_urls if _is_relevant_reference_url(u)]
        plain = _strip_html(html_raw)
        clue["found"] = True
        clue["reference_urls"] = ref_urls[:30]
        m_exact_go = re.search(r"Exact source\s+(GO:\d{7})", plain, flags=re.IGNORECASE)
        if m_exact_go:
            clue["go_id"] = str(m_exact_go.group(1)).upper()
        else:
            m_any_go = re.search(r"\b(GO:\d{7})\b", plain)
            clue["go_id"] = str(m_any_go.group(1)).upper() if m_any_go else ""
        m_brief = re.search(
            r"Brief description\s+(.+?)\s+(Full description|Collection|External details|Description)",
            plain,
            flags=re.IGNORECASE,
        )
        if m_brief:
            clue["brief_description"] = str(m_brief.group(1)).strip()
        ref_text = clue["brief_description"] or plain
        link_pmids = _extract_pmids_from_urls(ref_urls)
        link_dois = _extract_dois_from_urls(ref_urls)
        text_pmids = _extract_pmids_from_text(ref_text)
        text_dois = _extract_dois_from_text(ref_text)
        clue["reference_link_pmids"] = link_pmids[:20]
        clue["reference_link_dois"] = link_dois[:15]
        clue["reference_pmids"] = list(dict.fromkeys(text_pmids + link_pmids))[:25]
        clue["reference_dois"] = list(dict.fromkeys(text_dois + link_dois))[:15]
        return clue
    except Exception:
        return clue


def _fetch_quickgo_term_summary(go_id: str) -> Dict[str, Any]:
    go_id = str(go_id or "").strip().upper()
    empty = {
        "found": False,
        "go_id": go_id,
        "term_name": "",
        "definition": "",
        "quickgo_url": f"https://www.ebi.ac.uk/QuickGO/term/{quote(go_id)}" if go_id else "",
        "reference_pmids": [],
        "reference_dois": [],
        "definition_references": [],
    }
    if not go_id:
        return empty
    try:
        url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{quote(go_id)}"
        req = Request(
            url,
            headers={"User-Agent": "GEMMAP/0.4", "Accept": "application/json"},
        )
        with urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        results = (payload or {}).get("results") or []
        if not isinstance(results, list) or not results:
            return empty
        term = results[0] or {}
        def_raw = term.get("definition")
        definition_refs: List[str] = []
        if isinstance(def_raw, dict):
            definition_text = str(def_raw.get("text") or "").strip()
            xrefs = def_raw.get("xrefs")
            if isinstance(xrefs, list):
                for xref in xrefs:
                    if isinstance(xref, dict):
                        db = str(xref.get("db") or xref.get("database") or "").strip()
                        xid = str(xref.get("id") or xref.get("value") or "").strip()
                        if db or xid:
                            definition_refs.append(f"{db}:{xid}".strip(":"))
                    elif xref:
                        definition_refs.append(str(xref).strip())
        else:
            definition_text = str(def_raw or "").strip()
        if not definition_refs:
            term_refs = term.get("definitionXrefs")
            if isinstance(term_refs, list):
                for x in term_refs:
                    if x:
                        definition_refs.append(str(x).strip())
        serialized = json.dumps(term)
        refs = _collect_reference_ids(term)
        pmid_list = list(dict.fromkeys(
            refs.get("pmids", []) + _extract_pmids_from_text(serialized)
        ))[:20]
        doi_list = list(dict.fromkeys(
            refs.get("dois", []) + _extract_dois_from_text(serialized)
        ))[:12]
        return {
            "found": True,
            "go_id": go_id,
            "term_name": str(term.get("name") or "").strip(),
            "definition": definition_text,
            "quickgo_url": f"https://www.ebi.ac.uk/QuickGO/term/{quote(go_id)}",
            "reference_pmids": pmid_list,
            "reference_dois": doi_list,
            "definition_references": list(dict.fromkeys([r for r in definition_refs if r]))[:20],
        }
    except Exception:
        return empty


def _fetch_quickgo_annotation_pmids(go_id: str, limit: int = 40) -> List[str]:
    go_id = str(go_id or "").strip().upper()
    if not go_id:
        return []
    try:
        params = urlencode({
            "goId": go_id,
            "limit": max(1, min(int(limit or 40), 100)),
            "page": 1,
        })
        url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?{params}"
        req = Request(
            url,
            headers={"User-Agent": "GEMMAP/0.4", "Accept": "application/json"},
        )
        with urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        refs = _collect_reference_ids(payload)
        pmids = refs.get("pmids", [])
        if not pmids:
            pmids = _extract_pmids_from_text(json.dumps(payload))
        return list(dict.fromkeys(pmids))[:30]
    except Exception:
        return []


def _search_pubmed_ids(term: str, n: int) -> List[str]:
    term = str(term or "").strip()
    if not term:
        return []
    try:
        esearch_params = urlencode({
            "db": "pubmed",
            "retmode": "json",
            "retmax": max(1, min(int(n or 10), 40)),
            "sort": "relevance",
            "term": term,
        })
        esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{esearch_params}"
        req = Request(esearch_url, headers={"User-Agent": "GEMMAP/0.4"})
        with urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        return payload.get("esearchresult", {}).get("idlist", []) or []
    except Exception:
        return []


def _fetch_pubmed_details(ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    clean_ids = [str(i).strip() for i in ids if str(i).strip()]
    if not clean_ids:
        return []
    try:
        efetch_params = urlencode({
            "db": "pubmed",
            "id": ",".join(clean_ids),
            "retmode": "xml",
        })
        efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{efetch_params}"
        req2 = Request(efetch_url, headers={"User-Agent": "GEMMAP/0.4"})
        with urlopen(req2, timeout=15) as resp:
            xml_raw = resp.read()
        root = ET.fromstring(xml_raw)
    except Exception:
        return []

    papers: List[Dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        title = " ".join(article.findtext(".//Article/ArticleTitle", "").split())

        abstract_parts = []
        for abs_node in article.findall(".//Article/Abstract/AbstractText"):
            text = " ".join("".join(abs_node.itertext()).split())
            if text:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts).strip()

        journal = " ".join(article.findtext(".//Article/Journal/Title", "").split())
        year = (article.findtext(".//Article/Journal/JournalIssue/PubDate/Year") or "").strip()
        if not year:
            year = " ".join(article.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate", "").split())

        doi = ""
        pmcid = ""
        for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            id_type = str(aid.attrib.get("IdType", "")).lower()
            if id_type == "doi":
                doi = (aid.text or "").strip()
            elif id_type in {"pmc", "pmcid"}:
                pmcid = (aid.text or "").strip()

        papers.append({
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "year": year,
            "doi": doi,
            "pmcid": pmcid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            "doi_url": f"https://doi.org/{doi}" if doi else "",
            "abstract": abstract,
        })
    return papers


def _extract_json_object_from_text(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        chunk = text[start:end + 1]
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _compact_text(value: Any, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _looks_like_generic_go_curation_paper(title: Any, abstract: Any) -> bool:
    text = f"{title or ''} {abstract or ''}".lower()
    if not text.strip():
        return False
    generic_markers = [
        "gene ontology",
        "go curation",
        "manual annotation",
        "annotation process",
        "annotation pipeline",
        "interpro approach",
        "ontology curation",
    ]
    specific_markers = [
        "signaling pathway",
        "metabolic process",
        "biosynthetic process",
        "immune response",
        "cell cycle",
        "regulation of",
    ]
    generic_hits = sum(1 for m in generic_markers if m in text)
    specific_hits = sum(1 for m in specific_markers if m in text)
    has_go_id = bool(re.search(r"\bgo:\d{7}\b", text))
    return generic_hits >= 2 and specific_hits == 0 and not has_go_id


def _paper_has_strict_origin_signal(
    paper: Dict[str, Any],
    pathway_raw_name: str,
    pathway_query_phrase: str,
    go_id: str,
    go_term_name: str,
) -> bool:
    """
    Strict Origin-mode acceptance:
    - Allow clue-linked ontology references, OR
    - Require direct term identity signal in title/text.
    """
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    title_lower = title.lower()
    text_lower = f"{title} {abstract}".lower()
    tags = {
        str(t).strip().lower()
        for t in (paper.get("source_tags") or [])
        if str(t).strip()
    }

    clue_reference_tags = {
        "msigdb_reference",
        "quickgo_term_reference",
        "quickgo_annotation_reference",
        "amigo_reference",
        "go_ref_reference",
        "doi_resolved_reference",
    }
    has_clue_reference = bool(tags.intersection(clue_reference_tags))
    if has_clue_reference:
        return True

    if _looks_like_generic_go_curation_paper(title, abstract):
        return False

    raw_lower = str(pathway_raw_name or "").strip().lower()
    phrase_lower = str(pathway_query_phrase or "").strip().lower()
    go_id_lower = str(go_id or "").strip().lower()
    go_term_lower = str(go_term_name or "").strip().lower()

    has_direct_title_signal = any(
        term and term in title_lower
        for term in (raw_lower, phrase_lower, go_term_lower, go_id_lower)
    )
    has_pathway_text_tag = "pathway_text_match" in tags
    has_go_id_signal = bool(go_id_lower and go_id_lower in text_lower)
    return bool(has_direct_title_signal or (has_pathway_text_tag and has_go_id_signal))


def _fetch_pubmed_origin_papers(
    pathway_name: str,
    retmax: int = 5,
    use_naming_clue: bool = True,
    msigdb_go_only: bool = False,
) -> Dict[str, Any]:
    """
    Local deterministic retrieval stage for pathway-origin evidence.
    This stage intentionally performs no semantic ranking; it only gathers
    candidate clues and candidate papers for a downstream LLM validation pass.
    """
    raw_name = str(pathway_name or "").strip()
    query_phrase = _pathway_label_to_query(raw_name)
    if not raw_name:
        return {
            "query": "",
            "queries": [],
            "candidate_papers": [],
            "match_type": "none",
            "exact_name_found": False,
            "naming_clue": {},
            "naming_note": "",
            "local_search_notes": [],
        }

    retmax = max(1, min(int(retmax or 5), 12))
    namespace_info = _detect_pathway_namespace(raw_name)
    naming_clue: Dict[str, Any] = {
        "namespace": namespace_info["namespace"],
        "prefix": namespace_info["prefix"],
        "is_msigdb_go": bool(namespace_info["is_msigdb_go"]),
        "go_id": "",
        "go_term_name": "",
        "go_definition": "",
        "msigdb_card_url": "",
        "quickgo_url": "",
        "amigo_term_url": "",
        "reference_pmids": [],
        "reference_dois": [],
        "definition_references": [],
        "brief_description": "",
        "reference_urls": [],
        "msigdb_reference_pmids": [],
        "quickgo_reference_pmids": [],
        "quickgo_annotation_pmids": [],
        "amigo_reference_pmids": [],
        "go_ref_reference_pmids": [],
        "go_ref_ids": [],
        "go_ref_urls": [],
        "doi_resolved_pmids": [],
    }
    local_search_notes: List[str] = []

    if use_naming_clue:
        msigdb_clue = _fetch_msigdb_card_clue(raw_name)
        naming_clue["msigdb_card_url"] = msigdb_clue.get("card_url", "")
        naming_clue["brief_description"] = msigdb_clue.get("brief_description", "")
        if msigdb_clue.get("go_id"):
            naming_clue["go_id"] = str(msigdb_clue.get("go_id"))
        msigdb_pmids = list(msigdb_clue.get("reference_pmids") or [])
        msigdb_dois = list(msigdb_clue.get("reference_dois") or [])
        naming_clue["msigdb_reference_pmids"] = msigdb_pmids[:30]
        naming_clue["reference_pmids"] = msigdb_pmids[:30]
        naming_clue["reference_dois"] = msigdb_dois[:20]
        naming_clue["reference_urls"] = list(msigdb_clue.get("reference_urls") or [])[:30]
        if msigdb_clue.get("found"):
            local_search_notes.append("MSigDB card metadata found for exact pathway title.")
        msigdb_link_pmids = list(msigdb_clue.get("reference_link_pmids") or [])
        msigdb_link_dois = list(msigdb_clue.get("reference_link_dois") or [])
        if msigdb_link_pmids or msigdb_link_dois:
            local_search_notes.append(
                f"MSigDB clue-link harvest: {len(msigdb_link_pmids)} PMID and {len(msigdb_link_dois)} DOI references."
            )

        if not naming_clue["go_id"] and bool(namespace_info["is_msigdb_go"]):
            quickgo_match = _search_quickgo_go_id(namespace_info.get("go_label_guess", ""))
            if quickgo_match.get("go_id"):
                naming_clue["go_id"] = str(quickgo_match.get("go_id"))
                naming_clue["go_term_name"] = str(quickgo_match.get("term_name") or "")
                local_search_notes.append("QuickGO term match inferred from pathway title.")

        go_id = str(naming_clue.get("go_id") or "").strip().upper()
        if go_id:
            quickgo_term = _fetch_quickgo_term_summary(go_id)
            if quickgo_term.get("found"):
                naming_clue["quickgo_url"] = quickgo_term.get("quickgo_url", "")
                if not naming_clue["go_term_name"]:
                    naming_clue["go_term_name"] = quickgo_term.get("term_name", "")
                naming_clue["go_definition"] = quickgo_term.get("definition", "")
                naming_clue["quickgo_reference_pmids"] = list(quickgo_term.get("reference_pmids") or [])[:30]
                pmid_union = list(dict.fromkeys(
                    list(naming_clue.get("reference_pmids") or []) +
                    list(quickgo_term.get("reference_pmids") or [])
                ))
                doi_union = list(dict.fromkeys(
                    list(naming_clue.get("reference_dois") or []) +
                    list(quickgo_term.get("reference_dois") or [])
                ))
                naming_clue["reference_pmids"] = pmid_union[:30]
                naming_clue["reference_dois"] = doi_union[:20]
                naming_clue["definition_references"] = list(quickgo_term.get("definition_references") or [])[:20]
                local_search_notes.append(f"QuickGO term metadata loaded for {go_id}.")
            ann_pmids = _fetch_quickgo_annotation_pmids(go_id, limit=40)
            if ann_pmids:
                naming_clue["quickgo_annotation_pmids"] = list(dict.fromkeys(ann_pmids))[:30]
                naming_clue["reference_pmids"] = list(dict.fromkeys(
                    list(naming_clue.get("reference_pmids") or []) + ann_pmids
                ))[:30]
                local_search_notes.append(f"QuickGO annotation references collected ({len(ann_pmids)} PMID hits).")
            elif naming_clue.get("definition_references"):
                refs_preview = ", ".join(list(naming_clue.get("definition_references") or [])[:3])
                if refs_preview:
                    local_search_notes.append(f"QuickGO definition references found (non-PMID): {refs_preview}.")

            amigo_clue = _fetch_amigo_go_ref_clue(go_id, max_go_refs=10)
            if amigo_clue.get("found"):
                naming_clue["amigo_term_url"] = str(amigo_clue.get("amigo_term_url") or "")
                naming_clue["go_ref_ids"] = list(amigo_clue.get("go_ref_ids") or [])[:20]
                naming_clue["go_ref_urls"] = list(amigo_clue.get("go_ref_urls") or [])[:20]
                naming_clue["amigo_reference_pmids"] = list(amigo_clue.get("amigo_term_reference_pmids") or [])[:40]
                naming_clue["go_ref_reference_pmids"] = list(amigo_clue.get("go_ref_reference_pmids") or [])[:40]
                naming_clue["reference_pmids"] = list(dict.fromkeys(
                    list(naming_clue.get("reference_pmids") or []) +
                    list(amigo_clue.get("reference_pmids") or [])
                ))[:60]
                naming_clue["reference_dois"] = list(dict.fromkeys(
                    list(naming_clue.get("reference_dois") or []) +
                    list(amigo_clue.get("reference_dois") or [])
                ))[:40]
                naming_clue["reference_urls"] = list(dict.fromkeys(
                    list(naming_clue.get("reference_urls") or []) +
                    list(amigo_clue.get("reference_urls") or []) +
                    list(amigo_clue.get("go_ref_urls") or [])
                ))[:80]
                local_search_notes.append(f"AmiGO term metadata loaded for {go_id}.")
                go_ref_n = len(list(naming_clue.get("go_ref_ids") or []))
                go_ref_pmid_n = len(list(naming_clue.get("go_ref_reference_pmids") or []))
                if go_ref_n > 0:
                    local_search_notes.append(
                        f"GO_REF references harvested from AmiGO ({go_ref_n} GO_REF entries, {go_ref_pmid_n} PMID hits)."
                    )

        doi_resolved_pmids: List[str] = []
        clue_dois = list(dict.fromkeys([str(d).strip() for d in naming_clue.get("reference_dois") or [] if str(d).strip()]))[:12]
        for doi in clue_dois:
            ids = _search_pubmed_ids(f"\"{doi}\"[AID]", 3)
            for pmid in ids:
                pmid_text = str(pmid).strip()
                if pmid_text:
                    doi_resolved_pmids.append(pmid_text)
        if doi_resolved_pmids:
            naming_clue["doi_resolved_pmids"] = list(dict.fromkeys(doi_resolved_pmids))[:20]
            naming_clue["reference_pmids"] = list(dict.fromkeys(
                list(naming_clue.get("reference_pmids") or []) +
                list(naming_clue.get("doi_resolved_pmids") or [])
            ))[:40]
            local_search_notes.append(
                f"Resolved {len(list(naming_clue.get('doi_resolved_pmids') or []))} PMID records from clue DOI references."
            )
    else:
        local_search_notes.append("Naming-clue lookup skipped by user option.")

    naming_note = ""
    if naming_clue.get("go_id"):
        term_name = naming_clue.get("go_term_name") or "GO term"
        naming_note = (
            f"This title maps to {naming_clue['go_id']} ({term_name}). "
            "GO terms are curated ontology concepts with definitions and source references; naming is not arbitrary."
        )
    elif bool(namespace_info["is_msigdb_go"]):
        naming_note = (
            "This looks like an MSigDB GO-derived term. GO naming is curated, but an exact GO ID was not resolved from local clues."
        )

    query_attempts: List[str] = [f"\"{raw_name}\"[Title]", f"\"{raw_name}\"[Title/Abstract]"]
    go_id = str(naming_clue.get("go_id") or "").strip().upper()
    go_term_name = str(naming_clue.get("go_term_name") or "").strip()
    if msigdb_go_only:
        if go_id:
            query_attempts.append(f"\"{go_id}\"[All Fields]")
        if go_term_name:
            query_attempts.append(f"\"{go_term_name}\"[Title/Abstract]")
            query_attempts.append(f"\"{go_term_name}\"[All Fields]")
        if go_id and go_term_name:
            query_attempts.append(f"(\"{go_id}\"[All Fields]) OR (\"{go_term_name}\"[All Fields])")
    else:
        if query_phrase and query_phrase.lower() != raw_name.lower():
            query_attempts.append(f"\"{query_phrase}\"[Title/Abstract]")
            query_attempts.append(f"\"{query_phrase}\"[All Fields]")
        if go_id:
            query_attempts.append(f"\"{go_id}\"[All Fields]")
            query_attempts.append(
                f"(\"{go_id}\"[All Fields]) AND (ontology[Title/Abstract] OR curation[Title/Abstract] OR review[Title/Abstract])"
            )
        if go_term_name:
            query_attempts.append(f"\"{go_term_name}\"[Title/Abstract]")
            query_attempts.append(
                f"(\"{go_term_name}\"[Title/Abstract]) AND (ontology[Title/Abstract] OR curation[Title/Abstract] OR review[Title/Abstract])"
            )
            query_attempts.append(
                f"(\"{raw_name}\"[Title/Abstract] OR \"{go_term_name}\"[Title/Abstract])"
            )

    seen_queries = set()
    dedup_queries: List[str] = []
    for q in query_attempts:
        key = str(q or "").strip().lower()
        if not key or key in seen_queries:
            continue
        seen_queries.add(key)
        dedup_queries.append(q)

    if msigdb_go_only and not bool(namespace_info["is_msigdb_go"]):
        local_search_notes.append("Origin-source mode enabled, but pathway prefix is not GO-derived. Used exact-title PubMed plus available clues.")
    elif msigdb_go_only:
        local_search_notes.append("Origin-source mode enabled: focused on GO/MSigDB clues, GO references, and exact-title queries.")
    else:
        local_search_notes.append("Origin+context mode enabled: included broader ontology-linked PubMed queries.")

    candidate_pmids: List[str] = list(naming_clue.get("reference_pmids") or [])
    for query in dedup_queries:
        ids = _search_pubmed_ids(query, max(retmax * 3, 8))
        for pmid in ids:
            candidate_pmids.append(str(pmid))
    if not candidate_pmids and go_term_name:
        fallback_ids = _search_pubmed_ids(go_term_name, max(retmax * 4, 10))
        for pmid in fallback_ids:
            candidate_pmids.append(str(pmid))
        if fallback_ids:
            local_search_notes.append(f"Fallback GO-term query recovered {len(fallback_ids)} PMID candidates.")
    if not candidate_pmids and go_id:
        fallback_ids = _search_pubmed_ids(go_id, max(retmax * 4, 10))
        for pmid in fallback_ids:
            candidate_pmids.append(str(pmid))
        if fallback_ids:
            local_search_notes.append(f"Fallback GO-ID query recovered {len(fallback_ids)} PMID candidates.")
    candidate_pmids = list(dict.fromkeys([p for p in candidate_pmids if str(p).strip()]))[:45]

    papers = _fetch_pubmed_details(candidate_pmids)
    seen_keys = set()
    dedup_papers: List[Dict[str, Any]] = []
    raw_lower = raw_name.lower()
    phrase_lower = query_phrase.lower()
    go_term_lower = str(naming_clue.get("go_term_name") or "").strip().lower()
    clue_pmid_set = {str(p).strip() for p in (naming_clue.get("reference_pmids") or []) if str(p).strip()}
    msigdb_ref_set = {str(p).strip() for p in (naming_clue.get("msigdb_reference_pmids") or []) if str(p).strip()}
    quickgo_ref_set = {str(p).strip() for p in (naming_clue.get("quickgo_reference_pmids") or []) if str(p).strip()}
    quickgo_ann_set = {str(p).strip() for p in (naming_clue.get("quickgo_annotation_pmids") or []) if str(p).strip()}
    amigo_ref_set = {str(p).strip() for p in (naming_clue.get("amigo_reference_pmids") or []) if str(p).strip()}
    go_ref_ref_set = {str(p).strip() for p in (naming_clue.get("go_ref_reference_pmids") or []) if str(p).strip()}
    doi_resolved_set = {str(p).strip() for p in (naming_clue.get("doi_resolved_pmids") or []) if str(p).strip()}
    for paper in papers:
        paper_key = (
            str(paper.get("pmid") or "").strip() or
            str(paper.get("doi") or "").strip().lower() or
            str(paper.get("title") or "").strip().lower()
        )
        if not paper_key or paper_key in seen_keys:
            continue
        seen_keys.add(paper_key)
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        is_exact = (
            bool(raw_lower and raw_lower in text)
            or bool(phrase_lower and phrase_lower in text)
            or bool(go_term_lower and go_term_lower in text)
        )
        pmid = str(paper.get("pmid") or "").strip()
        source_tags: List[str] = []
        if pmid and pmid in msigdb_ref_set:
            source_tags.append("msigdb_reference")
        if pmid and pmid in quickgo_ref_set:
            source_tags.append("quickgo_term_reference")
        if pmid and pmid in quickgo_ann_set:
            source_tags.append("quickgo_annotation_reference")
        if pmid and pmid in amigo_ref_set:
            source_tags.append("amigo_reference")
        if pmid and pmid in go_ref_ref_set:
            source_tags.append("go_ref_reference")
        if pmid and pmid in doi_resolved_set:
            source_tags.append("doi_resolved_reference")
        if is_exact:
            source_tags.append("pathway_text_match")
        if not source_tags:
            source_tags.append("query_candidate")
        out = dict(paper)
        out["_exact_title_hit"] = is_exact
        out["_clue_reference_hit"] = bool(pmid and pmid in clue_pmid_set)
        out["source_tags"] = list(dict.fromkeys(source_tags))
        dedup_papers.append(out)

    dedup_papers.sort(
        key=lambda p: (
            1 if bool(p.get("_clue_reference_hit")) else 0,
            1 if bool(p.get("_exact_title_hit")) else 0,
            1 if str(p.get("abstract") or "").strip() else 0,
        ),
        reverse=True
    )

    exact_name_found = any(bool(p.get("_exact_title_hit")) for p in dedup_papers)
    candidate_papers = []
    for paper in dedup_papers[: max(retmax * 6, 20)]:
        out = dict(paper)
        out.pop("_exact_title_hit", None)
        out.pop("_clue_reference_hit", None)
        candidate_papers.append(out)

    return {
        "query": dedup_queries[0] if dedup_queries else "",
        "queries": dedup_queries,
        "candidate_papers": candidate_papers,
        "match_type": "candidate_pool" if candidate_papers else "none",
        "exact_name_found": exact_name_found,
        "naming_clue": naming_clue,
        "naming_note": naming_note,
        "local_search_notes": local_search_notes,
        "clue_reference_pmids": list(naming_clue.get("reference_pmids") or []),
    }


def _normalize_pathway_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.split(r"\n|<br\s*/?>|\|", text, flags=re.IGNORECASE)[0]
    text = re.sub(r"[_\s]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", text)
    return text.lower().strip()


def _match_pathway_record(pathways: List[Dict[str, Any]], pathway_name: str) -> Optional[Dict[str, Any]]:
    if not pathways:
        return None
    target = _normalize_pathway_text(pathway_name)
    if not target:
        return None

    for rec in pathways:
        if _normalize_pathway_text(rec.get("pathway")) == target:
            return rec
    for rec in pathways:
        candidate = _normalize_pathway_text(rec.get("pathway"))
        if candidate and (candidate in target or target in candidate):
            return rec
    return None


def _fetch_open_access_fulltext_snippet(pmid: str, max_chars: int = 7000) -> Dict[str, Any]:
    """
    Best-effort open-access fulltext snippet retrieval via Europe PMC.
    Returns snippet text when available; otherwise a reason string.
    """
    pmid = str(pmid or "").strip()
    if not pmid:
        return {"used": False, "snippet": "", "pmcid": "", "note": "No PMID available for open-access lookup."}
    try:
        search_params = urlencode({
            "query": f"EXT_ID:{pmid} AND SRC:MED",
            "format": "json",
            "pageSize": 1,
        })
        search_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{search_params}"
        req = Request(search_url, headers={"User-Agent": "GEMMAP/0.4"})
        with urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        results = (((payload or {}).get("resultList") or {}).get("result") or [])
        rec = results[0] if results else {}
        pmcid = str(rec.get("pmcid") or "").strip()
        is_oa = str(rec.get("isOpenAccess") or "").upper() == "Y"
        if not (pmcid and is_oa):
            return {
                "used": False,
                "snippet": "",
                "pmcid": pmcid,
                "note": "Open-access full text was not available from Europe PMC for this paper."
            }

        fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        req2 = Request(fulltext_url, headers={"User-Agent": "GEMMAP/0.4"})
        with urlopen(req2, timeout=14) as resp2:
            xml_raw = resp2.read().decode("utf-8", errors="ignore")
        if not xml_raw:
            return {
                "used": False,
                "snippet": "",
                "pmcid": pmcid,
                "note": "Open-access record exists, but full text could not be retrieved."
            }
        text = re.sub(r"<[^>]+>", " ", xml_raw)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return {
                "used": False,
                "snippet": "",
                "pmcid": pmcid,
                "note": "Open-access full text was empty after parsing."
            }
        return {
            "used": True,
            "snippet": text[:max_chars],
            "pmcid": pmcid,
            "note": f"Open-access full text snippet retrieved from {pmcid}."
        }
    except Exception:
        return {
            "used": False,
            "snippet": "",
            "pmcid": "",
            "note": "Open-access full text lookup failed; analysis used PubMed metadata/abstract."
        }


def _build_reproducibility_manifest(
    session_id: str,
    session: Dict[str, Any],
    analyzer: PathwayAnalyzer,
    ai_results: Optional[List[Dict[str, Any]]] = None,
    include_chat_history: bool = True
) -> Dict[str, Any]:
    """
    Build a publication-grade reproducibility JSON supplement.
    """
    ai_results = ai_results or []
    cluster_df = analyzer.cluster_results if analyzer.cluster_results is not None else pd.DataFrame()
    gene_stats = analyzer.gene_stats if analyzer.gene_stats is not None else analyzer.analyze_gene_frequencies()
    cluster_stats = compute_cluster_statistics(analyzer) if not cluster_df.empty else []
    n_clusters = int(cluster_df["Cluster"].nunique()) if "Cluster" in cluster_df.columns else 0
    methodology = getattr(analyzer, "methodology", {}) or {}

    try:
        mds_gof = [float(x) for x in analyzer.get_mds_gof()]
    except Exception:
        mds_gof = [0.0, 0.0]

    unique_genes = set()
    try:
        for genes in analyzer.raw_data[analyzer.genes_col]:
            for gene in _normalize_gene_list(genes):
                unique_genes.add(gene)
    except Exception:
        pass

    human_overview = (
        "This reproducibility package captures the exact GEMMAP workflow used to generate the session results. "
        "The pipeline applies Jaccard similarity over pathway gene sets, projects distances with Classical MDS, "
        "modules pathways using K-means, computes core-gene statistics, and optionally overlays DEG evidence and "
        "agentic AI annotations. Replaying this manifest with the same input matrix and model/provider settings "
        "should reproduce equivalent outputs."
    )

    agentic_steps = [
        "Validate pathway table schema and parse pathway gene lists.",
        "Resolve NES/score column and optional filters (padj/nominal thresholds, NES direction).",
        "Compute pathway-by-pathway Jaccard similarity matrix.",
        "Project similarity distances into 3D using Classical MDS.",
        "Run K-means module assignment (deterministic random_state=42) on MDS coordinates.",
        "Compute per-module core genes and activation metrics.",
        "If DEG table is provided, annotate DEG overlap with configured thresholds.",
        "If AI annotations are enabled, summarize each module using top pathways and core genes.",
        "Export interactive report, tabular outputs, and reproducibility supplement.",
    ]

    replay_params = {
        "pathway_col": analyzer.pathway_col,
        "genes_col": analyzer.genes_col,
        "score_col": analyzer.score_col,
        "n_clusters": n_clusters,
        "nes_direction": str(getattr(analyzer, "_nes_direction", "all")),
        "deg_config": session.get("deg_config"),
        "ai_mode": methodology.get("mode", "Manual"),
        "ai_provider": methodology.get("ai_provider", ""),
        "ai_model": methodology.get("ai_model", ""),
    }

    pathway_table_records = []
    try:
        raw_df = analyzer.raw_data.copy() if isinstance(analyzer.raw_data, pd.DataFrame) else pd.DataFrame()
        if not raw_df.empty:
            pathway_table_records = json_clean(raw_df.to_dict(orient="records"))
    except Exception:
        pathway_table_records = []

    deg_table_records = []
    try:
        deg_df = session.get("deg_df")
        if isinstance(deg_df, pd.DataFrame) and not deg_df.empty:
            deg_table_records = json_clean(deg_df.to_dict(orient="records"))
    except Exception:
        deg_table_records = []

    chat_log_records: List[Dict[str, Any]] = []
    if include_chat_history:
        try:
            raw_chat = session.get("chat_log")
            if isinstance(raw_chat, list):
                chat_log_records = json_clean(raw_chat)
        except Exception:
            chat_log_records = []

    output_payload: Dict[str, Any] = {
        "cluster_results": json_clean(cluster_df.to_dict(orient="records")) if not cluster_df.empty else [],
        "gene_stats": json_clean(gene_stats.to_dict(orient="records")) if isinstance(gene_stats, pd.DataFrame) else [],
        "ai_annotations": json_clean(ai_results),
    }
    if include_chat_history:
        output_payload["chat_log"] = chat_log_records

    manifest = {
        "schema": "gemmap.reproducibility.v1",
        "generated_at_utc": _now_utc_iso(),
        "gemmap_version": APP_VERSION,
        "session_id": session_id,
        "report_name": _session_report_basename(session),
        "input": {
            "source_filename": session.get("filename"),
            "pathway_col": analyzer.pathway_col,
            "genes_col": analyzer.genes_col,
            "score_col": analyzer.score_col,
            "deg_config": session.get("deg_config"),
            "nes_direction": str(getattr(analyzer, "_nes_direction", "all")),
            "pathway_table": pathway_table_records,
            "deg_table": deg_table_records,
        },
        "human_overview": human_overview,
        "agentic_framework": {
            "purpose": "Deterministic pathway-mapping and explainable module annotation workflow.",
            "determinism_contract": {
                "clustering_random_state": 42,
                "mds_method": "Classical MDS (eigendecomposition)",
                "kmeans_n_init": 10,
            },
            "steps": agentic_steps,
            "agent_prompt_contract": (
                "When AI annotation is enabled, summarize each module from top pathways + top core genes, "
                "label key process, and report confidence while separating evidence from hypothesis."
            ),
        },
        "methodology": methodology,
        "replay": {
            "endpoint": "/api/reproducibility/replay",
            "parameters": replay_params,
            "notes": (
                "Replay on the same pathway dataset to reproduce coordinates, modules, DEG overlays, and AI annotations "
                "using the same provider/model settings."
            ),
        },
        "results_summary": {
            "n_clusters": n_clusters,
            "n_pathways": int(len(cluster_df)),
            "n_unique_genes": int(len(unique_genes)),
            "mds_gof": mds_gof,
            "cluster_stats": cluster_stats,
        },
        "checksums": {
            "cluster_results_sha256": _hash_df(cluster_df),
            "gene_stats_sha256": _hash_df(gene_stats),
            "ai_annotations_sha256": hashlib.sha256(
                json.dumps(json_clean(ai_results), sort_keys=True).encode("utf-8")
            ).hexdigest() if ai_results else "",
        },
        "security": {
            "api_keys_stored": False,
            "sensitive_fields_redacted": True,
            "chat_history_included": bool(include_chat_history),
            "note": "Reproducibility exports intentionally remove API keys, tokens, and auth secrets.",
        },
        "outputs": output_payload,
    }
    return json_clean(_scrub_sensitive_fields(manifest))

# --- MODELS ---
class AnnotationRequest(BaseModel):
    """
    Request model for AI annotation.
    
    SECURITY NOTE: API keys are used ONLY for the current request and are 
    NEVER stored in session data, logs, or any persistent storage.
    """
    api_key: str  # Transient - NOT stored
    provider: str = "openai"  # "openai", "gemini", or "claude"
    turbo: bool = False  # Higher-cost, higher-capability model tier


class AutoAnalyzeRequest(BaseModel):
    """
    Request model for fully automated analysis with AI-driven k selection.
    """
    api_key: str  # Transient - NOT stored
    provider: str = "openai"
    turbo: bool = False
    k_min: int = 2
    k_max: int = 10


class ClusterChatRequest(BaseModel):
    """
    Request model for interactive cluster chat assistant.
    API key is transient and never stored.
    """
    api_key: str
    provider: str = "openai"
    turbo: bool = False
    message: str
    cluster_id: Optional[str] = None
    history: List[Dict[str, str]] = []
    include_literature: bool = True


class PathwayOriginRequest(BaseModel):
    """Request model for pathway-origin literature search."""
    pathway: str
    max_results: int = 5
    api_key: Optional[str] = None
    provider: str = "openai"
    turbo: bool = False
    use_naming_clue: bool = True
    msigdb_go_only: bool = False


class PathwayPaperAnalysisRequest(BaseModel):
    """Request model for deep paper-pathway analysis in mountain explorer."""
    api_key: str
    provider: str = "openai"
    turbo: bool = False
    pathway: str
    module_id: Optional[str] = None
    module_name: Optional[str] = None
    paper: Dict[str, Any]
    study_disease: Optional[str] = None
    study_tissue: Optional[str] = None
    study_organism: Optional[str] = None
    study_technology: Optional[str] = None
    study_cohort: Optional[str] = None
    study_notes: Optional[str] = None


class ReproReplayRequest(BaseModel):
    """
    Replay request for reproducibility manifest.
    """
    manifest: Dict[str, Any]
    api_key: Optional[str] = None
    provider: str = "openai"
    turbo: bool = False
    rerun_annotations: bool = False


class ImageTiffConvertRequest(BaseModel):
    """Convert a PNG/JPEG/WebP data URL into TIFF for mountain export."""
    image_data_url: str


def compute_cluster_statistics(analyzer) -> List[Dict]:
    """
    Compute detailed statistics for each cluster.
    
    Returns list of dicts with:
    - cluster_id, n_pathways, n_unique_genes, n_core_genes
    - mean_nes, mean_activation, dominant_direction
    - top_pathway (highest absolute NES)
    """
    import numpy as np
    
    stats = []
    cluster_df = analyzer.cluster_results
    gene_stats = analyzer.gene_stats if analyzer.gene_stats is not None else analyzer.analyze_gene_frequencies()
    
    for cid in cluster_df['Cluster'].unique():
        # Pathways in this cluster
        c_paths = cluster_df[cluster_df['Cluster'] == cid]
        n_pathways = len(c_paths)
        
        # Genes in this cluster
        c_genes = gene_stats[gene_stats['Cluster'] == cid] if gene_stats is not None else None
        n_unique_genes = len(c_genes) if c_genes is not None else 0
        n_core_genes = len(c_genes[c_genes['Percentage'] >= 25]) if c_genes is not None and 'Percentage' in c_genes.columns else 0
        
        # NES statistics
        if analyzer.score_col and analyzer.score_col in c_paths.columns:
            nes_values = c_paths[analyzer.score_col].dropna()
            mean_nes = float(nes_values.mean()) if len(nes_values) > 0 else 0.0
            
            # Top pathway by absolute NES
            if len(nes_values) > 0:
                top_idx = nes_values.abs().idxmax()
                top_pathway = c_paths.loc[top_idx, analyzer.pathway_col] if top_idx in c_paths.index else ""
            else:
                top_pathway = c_paths[analyzer.pathway_col].iloc[0] if len(c_paths) > 0 else ""
        else:
            mean_nes = 0.0
            top_pathway = c_paths[analyzer.pathway_col].iloc[0] if len(c_paths) > 0 else ""
        
        # Activation statistics
        if c_genes is not None and 'Activation_Score' in c_genes.columns:
            core_genes = c_genes[c_genes['Percentage'] >= 25]
            mean_activation = float(core_genes['Activation_Score'].mean()) if len(core_genes) > 0 else 0.0
        else:
            mean_activation = 0.0
        
        # Dominant direction
        dominant_direction = "positive" if mean_nes > 0 else "negative"
        
        stats.append({
            "cluster_id": str(cid),
            "n_pathways": n_pathways,
            "n_unique_genes": n_unique_genes,
            "n_core_genes": n_core_genes,
            "mean_nes": round(mean_nes, 3),
            "mean_activation": round(mean_activation, 3),
            "dominant_direction": dominant_direction,
            "top_pathway": str(top_pathway)[:80]  # Truncate long names
        })
    
    return stats

def json_clean(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: json_clean(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_clean(x) for x in obj]
    elif hasattr(obj, 'tolist'):
        return json_clean(obj.tolist())
    elif isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj

def _load_df_from_file(content: bytes, filename: str) -> pd.DataFrame:
    """
    Load a pandas DataFrame from various file formats based on extension.
    Supports: .csv, .txt, .tsv, .xlsx, .xls
    """
    filename_lower = filename.lower()
    file_stream = io.BytesIO(content)
    
    try:
        if filename_lower.endswith(('.xlsx', '.xls', '.xlsb', '.xlsm', '.ods')):
            # Use python-calamine for unified, fast Excel reading
            return pd.read_excel(file_stream, engine='calamine')
        elif filename_lower.endswith('.tsv'):
            return pd.read_csv(file_stream, sep='\t')
        elif filename_lower.endswith('.txt'):
            # Try tab first (common for GSEA), then fallback to engine='python' sniffing
            try:
                # Reset stream for each attempt if needed, though read_csv usually handles it.
                # simpler: just try sniffing
                return pd.read_csv(file_stream, sep=None, engine='python')
            except:
                file_stream.seek(0)
                return pd.read_csv(file_stream, sep='\t')
        elif filename_lower.endswith('.csv'):
            return pd.read_csv(file_stream)
        else:
            # Default to CSV/table sniffing
            return pd.read_csv(file_stream, sep=None, engine='python')
            
    except Exception as e:
        raise ValueError(f"Failed to parse file '{filename}'. Error: {str(e)}")


def _resolve_score_column(df: pd.DataFrame, requested_score_col: Optional[str]) -> Optional[str]:
    """Resolve NES/score column from user input or common defaults."""
    if requested_score_col and requested_score_col != "null" and requested_score_col in df.columns:
        return requested_score_col

    for default_col in ["NES", "nes", "Score", "score", "enrichmentScore"]:
        if default_col in df.columns:
            return default_col
    return None


def _apply_nes_direction_filter(
    df: pd.DataFrame,
    score_col: str,
    nes_direction: str
) -> tuple[pd.DataFrame, Dict]:
    """
    Optionally filter pathways by NES direction.
    nes_direction: 'all' | 'positive' | 'negative'
    """
    direction = (nes_direction or "all").lower()
    if direction not in {"all", "positive", "negative"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid nes_direction '{nes_direction}'. Use one of: all, positive, negative."
        )

    total_before = len(df)
    if direction == "all":
        return df.copy(), {
            "nes_direction": "all",
            "total_before": total_before,
            "total_after": total_before,
            "removed": 0
        }

    score_numeric = pd.to_numeric(df[score_col], errors="coerce")
    if direction == "positive":
        mask = score_numeric > 0
    else:
        mask = score_numeric < 0

    filtered_df = df.loc[mask].copy()
    total_after = len(filtered_df)
    if total_after == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No pathways left after NES '{direction}' filter."
        )

    return filtered_df, {
        "nes_direction": direction,
        "total_before": total_before,
        "total_after": total_after,
        "removed": total_before - total_after
    }


def _sanitize_report_basename(raw_name: Optional[str]) -> Optional[str]:
    """
    Sanitize optional report name for safe filesystem/browser downloads.
    Returns None when no valid name remains.
    """
    if not raw_name:
        return None
    name = str(raw_name).strip()
    if not name:
        return None
    name = re.sub(r"\.(html?|zip)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\-. ]+", "_", name).strip(" ._-")
    return name[:120] if name else None


def _session_report_basename(session: Dict) -> str:
    """
    Resolve report base name in order:
    1) user-provided optional report name
    2) uploaded file stem
    3) fallback GEMMAP_Report
    """
    custom = _sanitize_report_basename(session.get("report_name"))
    if custom:
        return custom
    uploaded = _sanitize_report_basename(Path(str(session.get("filename", ""))).stem)
    return uploaded or "GEMMAP_Report"


def _extract_gene_like_tokens(text: str) -> List[str]:
    """Extract uppercase gene-like tokens from free text."""
    if not text:
        return []
    blacklist = {
        "GO", "KEGG", "NES", "FDR", "PADJ", "PVAL", "P_VALUE",
        "AND", "OR", "NOT", "THE", "WITH", "FROM", "CLUSTER",
    }
    tokens = re.findall(r"\b[A-Z][A-Z0-9\-]{1,11}\b", str(text).upper())
    cleaned = []
    for tok in tokens:
        if tok in blacklist or tok.isdigit():
            continue
        cleaned.append(tok)
    return list(dict.fromkeys(cleaned))[:12]


def _fetch_pubmed_summaries(query: str, retmax: int = 5) -> List[Dict[str, str]]:
    """
    Lightweight PubMed search (titles + metadata) for cluster-chat grounding.
    Returns empty list on network errors.
    """
    if not query.strip():
        return []
    try:
        esearch_params = urlencode({
            "db": "pubmed",
            "retmode": "json",
            "retmax": max(1, min(int(retmax), 8)),
            "sort": "relevance",
            "term": query,
        })
        esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{esearch_params}"
        req = Request(esearch_url, headers={"User-Agent": "GEMMAP/0.4"})
        with urlopen(req, timeout=7) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        ids = payload.get("esearchresult", {}).get("idlist", []) or []
        if not ids:
            return []

        esummary_params = urlencode({
            "db": "pubmed",
            "retmode": "json",
            "id": ",".join(ids),
        })
        esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{esummary_params}"
        req2 = Request(esummary_url, headers={"User-Agent": "GEMMAP/0.4"})
        with urlopen(req2, timeout=7) as resp:
            summary_payload = json.loads(resp.read().decode("utf-8", errors="ignore"))

        results = []
        result_map = summary_payload.get("result", {})
        for pmid in ids:
            rec = result_map.get(str(pmid), {})
            if not rec:
                continue
            results.append({
                "pmid": str(pmid),
                "title": str(rec.get("title", "")).strip(),
                "journal": str(rec.get("fulljournalname", "")).strip(),
                "pubdate": str(rec.get("pubdate", "")).strip(),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return results
    except Exception:
        return []


def _prepare_cluster_chat_context(
    analyzer: PathwayAnalyzer,
    ai_results: Optional[List[Dict[str, Any]]],
    cluster_id: Optional[str]
) -> Dict[str, Any]:
    """Build compact, LLM-ready context for interactive cluster chat."""
    cluster_df = analyzer.cluster_results if analyzer.cluster_results is not None else pd.DataFrame()
    gene_stats = analyzer.gene_stats if analyzer.gene_stats is not None else analyzer.analyze_gene_frequencies()
    if cluster_df.empty:
        return {"cluster_context": "", "global_context": "", "genes_for_search": [], "pathway_for_search": "", "selected_cluster": None}

    clusters = sorted(cluster_df["Cluster"].unique(), key=lambda x: int(x))
    ai_map = {}
    for row in ai_results or []:
        ai_map[str(row.get("cluster_id"))] = row

    global_lines = []
    for cid in clusters:
        c_paths = cluster_df[cluster_df["Cluster"] == cid]
        mean_nes = float(c_paths[analyzer.score_col].mean()) if analyzer.score_col and analyzer.score_col in c_paths.columns else 0.0
        title = ai_map.get(str(cid), {}).get("title", f"Module {cid}")
        global_lines.append(f"C{cid}: {title} | pathways={len(c_paths)} | mean_NES={mean_nes:+.2f}")

    selected = None
    if cluster_id is not None and str(cluster_id).strip():
        cluster_id_str = str(cluster_id).strip()
        for cid in clusters:
            if str(cid) == cluster_id_str:
                selected = cid
                break
        if selected is None:
            selected = clusters[0]

    if selected is None:
        c_paths = cluster_df.copy()
        if analyzer.score_col and analyzer.score_col in c_paths.columns:
            c_paths = c_paths.assign(_abs_nes=c_paths[analyzer.score_col].abs()).sort_values("_abs_nes", ascending=False)
        top_paths = c_paths[analyzer.pathway_col].astype(str).head(10).tolist()

        c_genes = gene_stats.copy()
        if "Percentage" in c_genes.columns:
            c_genes = c_genes.sort_values("Percentage", ascending=False)
        top_genes = c_genes["Item"].astype(str).head(18).tolist()
        deg_core_genes = []
        if "DEG" in c_genes.columns:
            deg_core_genes = c_genes[c_genes["DEG"] == True]["Item"].astype(str).head(12).tolist()  # noqa: E712

        cluster_context = (
            "Focused scope: all modules (study-wide)\n"
            f"Total modules: {len(clusters)}\n"
            f"Top pathways across modules: {', '.join(top_paths[:8])}\n"
            f"Top genes across modules: {', '.join(top_genes[:12])}\n"
            f"Core DEG overlap genes: {', '.join(deg_core_genes[:10]) if deg_core_genes else 'None detected'}"
        )
        selected_label = "all"
    else:
        c_paths = cluster_df[cluster_df["Cluster"] == selected].copy()
        if analyzer.score_col and analyzer.score_col in c_paths.columns:
            c_paths = c_paths.sort_values(analyzer.score_col, ascending=False)
        top_paths = c_paths[analyzer.pathway_col].astype(str).head(8).tolist()

        c_genes = gene_stats[gene_stats["Cluster"] == selected].copy()
        if "Percentage" in c_genes.columns:
            c_genes = c_genes.sort_values("Percentage", ascending=False)
        top_genes = c_genes["Item"].astype(str).head(14).tolist()
        deg_core_genes = []
        if "DEG" in c_genes.columns:
            deg_core_genes = c_genes[c_genes["DEG"] == True]["Item"].astype(str).head(10).tolist()  # noqa: E712

        ai_selected = ai_map.get(str(selected), {})
        cluster_title = ai_selected.get("title", f"Module {selected}")
        cluster_summary = ai_selected.get("summary", "")
        cluster_process = ai_selected.get("key_process", "")

        cluster_context = (
            f"Focused module: M{selected} ({cluster_title})\n"
            f"AI summary: {cluster_summary or 'N/A'}\n"
            f"Key process: {cluster_process or 'N/A'}\n"
            f"Top pathways: {', '.join(top_paths[:6])}\n"
            f"Top genes: {', '.join(top_genes[:10])}\n"
            f"Core DEG overlap genes: {', '.join(deg_core_genes[:8]) if deg_core_genes else 'None detected'}"
        )
        selected_label = str(selected)

    return {
        "selected_cluster": selected_label,
        "cluster_context": cluster_context,
        "global_context": "\n".join(global_lines),
        "genes_for_search": (deg_core_genes[:6] or top_genes[:6]),
        "pathway_for_search": top_paths[0] if top_paths else "",
        "pathways_for_search": top_paths[:4],
    }


def _load_icon_b64() -> str:
    """Load icon from known local paths and return base64 string."""
    import base64

    app_dir = Path(__file__).resolve().parent
    candidates = [
        app_dir / "static" / "icon.png",
        app_dir.parent.parent.parent / "icon" / "icon.png",
        app_dir.parent.parent.parent / "src" / "gemmap" / "web" / "static" / "icon.png",
    ]
    for path in candidates:
        try:
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode("utf-8")
        except Exception:
            continue
    return ""


def _ensure_static_icon(static_dir: Path):
    """Ensure /static/icon.png exists so favicon and UI logo use a stable path."""
    try:
        target = static_dir / "icon.png"
        if target.exists() and target.stat().st_size > 0:
            return

        app_dir = Path(__file__).resolve().parent
        candidates: List[Path] = [
            app_dir.parent.parent.parent / "icon" / "icon.png",
            app_dir.parent.parent.parent / "src" / "gemmap" / "web" / "static" / "icon.png",
        ]

        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            hashed_icons = sorted(
                assets_dir.glob("icon-*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            candidates.extend(hashed_icons)

        for candidate in candidates:
            try:
                if not candidate.exists():
                    continue
                if candidate.resolve() == target.resolve():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, target)
                _log_event(f"Restored static icon from {candidate}")
                return
            except Exception:
                continue
    except Exception:
        pass


# --- SESSION MANAGEMENT ---
class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Dict] = {}
        self._ttl = 3600

    def create_session(self, df: pd.DataFrame, filename: str) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "df": df,
            "filename": filename,
            "report_name": None,
            "analyzer": None,
            "ai_results": None,
            "deg_df": None,  # Optional DEG DataFrame
            "deg_config": None,  # DEG column configuration
            "pending_deg_df": None,  # Temporary DEG upload pending confirmation
            "auto_k": None,  # AI-suggested k
            "annotation_cache": {},  # provider+cluster fingerprint -> annotations
            "chat_log": [],  # Research + mountain agent conversation history
            "created_at": time.time()
        }
        self._cleanup()
        return session_id

    def get_session(self, session_id: str):
        if session_id not in self._sessions:
            return None
        self._sessions[session_id]["created_at"] = time.time()
        return self._sessions[session_id]

    def update_analyzer(self, session_id: str, analyzer: PathwayAnalyzer):
        if session_id in self._sessions:
            self._sessions[session_id]["analyzer"] = analyzer

    def update_ai_results(self, session_id: str, results: List[Dict]):
        if session_id in self._sessions:
            self._sessions[session_id]["ai_results"] = results

    def update_deg_data(self, session_id: str, deg_df: pd.DataFrame, config: Dict):
        """Store DEG data and configuration for a session."""
        if session_id in self._sessions:
            self._sessions[session_id]["deg_df"] = deg_df
            self._sessions[session_id]["deg_config"] = config

    def update_auto_k(self, session_id: str, k: int):
        """Store AI-suggested k for a session."""
        if session_id in self._sessions:
            self._sessions[session_id]["auto_k"] = k

    def update_report_name(self, session_id: str, report_name: Optional[str]):
        """Store optional user-defined report base name for HTML exports."""
        if session_id in self._sessions:
            self._sessions[session_id]["report_name"] = _sanitize_report_basename(report_name)

    def _cleanup(self):
        now = time.time()
        expired = [sid for sid, data in self._sessions.items() if now - data["created_at"] > self._ttl]
        for sid in expired:
            del self._sessions[sid]

session_manager = SessionManager()

# --- APP FACTORY ---
def create_app() -> FastAPI:
    _init_runtime_log_capture()
    _log_event("Initializing GEMMAP FastAPI app.")
    app = FastAPI(title="gemmap-server", description="Agentic Pathway Mapping API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- API ROUTES ---

    @app.post("/api/preview")
    async def preview_csv(file: UploadFile = File(...)):
        # Legacy endpoint - keeping for safety but logic is moving to /api/pathway/upload
        try:
            content = await file.read()
            df = _load_df_from_file(content, file.filename)
            session_id = session_manager.create_session(df, file.filename)
            return {
                "session_id": session_id,
                "filename": file.filename,
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records")
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/pathway/preview")
    async def preview_pathway_file(file: UploadFile = File(...)):
        """Preview Pathway file to get columns (Stateless)."""
        try:
            content = await file.read()
            df = _load_df_from_file(content, file.filename)
            return {"status": "success", "columns": list(df.columns)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")

    @app.post("/api/pathway/upload")
    async def upload_pathway_file(
        file: UploadFile = File(...),
        path_col: str = Form("pathway"),
        gene_col: str = Form("leadingEdge"),
        score_col: str = Form("NES"),
        padj_col: str = Form("null"),
        padj_threshold: float = Form(0.05)
    ):
        """Upload Pathway file and create session with specific config."""
        try:
            content = await file.read()
            df = _load_df_from_file(content, file.filename)

            pathway_significance = {
                "checked": False,
                "column": None,
                "threshold": padj_threshold,
                "all_significant": None,
                "n_total": len(df),
                "n_kept": len(df),
                "n_removed": 0
            }

            if padj_col and padj_col != "null":
                if padj_col not in df.columns:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Pathway padj column '{padj_col}' not found. Available columns: {list(df.columns)}"
                    )

                padj_values = pd.to_numeric(df[padj_col], errors="coerce")
                significant_mask = padj_values < padj_threshold

                n_total = int(len(df))
                n_kept = int(significant_mask.sum())
                n_removed = n_total - n_kept

                if n_kept == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No pathways pass {padj_col} < {padj_threshold}. Please check your threshold or column mapping."
                    )

                df = df.loc[significant_mask].copy()
                pathway_significance = {
                    "checked": True,
                    "column": padj_col,
                    "threshold": padj_threshold,
                    "all_significant": n_removed == 0,
                    "n_total": n_total,
                    "n_kept": n_kept,
                    "n_removed": n_removed
                }

            # Create session (filtered if padj was provided)
            session_id = session_manager.create_session(df, file.filename)

            return {
                "status": "success",
                "session_id": session_id,
                "filename": file.filename,
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
                "pathway_significance": pathway_significance
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/demo")
    async def load_demo_data():
        """Load synthetic demo data including pathways and DEGs."""
        try:
            # Generate synthetic pathway data
            df = PathwayAnalyzer.create_synthetic_data(n_pathways=150)
            session_id = session_manager.create_session(df, "demo_data.csv")
            
            # Generate synthetic DEG data based on pathway genes
            deg_df = PathwayAnalyzer.create_synthetic_deg_data(df, genes_col="leadingEdge")
            
            # Mark DEG status using standard thresholds
            deg_df["DEG"] = (
                (deg_df["padj"] < 0.05) & 
                (deg_df["log2FoldChange"].abs() > 0.25)
            )
            
            n_degs = int(deg_df["DEG"].sum())
            
            # Store DEG data in session
            deg_config = {
                "gene_col": "gene",
                "padj_col": "padj",
                "lfc_col": "log2FoldChange",
                "padj_threshold": 0.05,
                "lfc_threshold": 0.25
            }
            session_manager.update_deg_data(session_id, deg_df, deg_config)
            
            return {
                "session_id": session_id,
                "filename": "demo_biological_data.csv",
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
                "deg_info": {
                    "filename": "demo_deg_data.csv",
                    "n_genes": len(deg_df),
                    "n_degs": n_degs,
                    "columns": list(deg_df.columns)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/deg/upload")
    async def upload_deg_file(
        file: Optional[UploadFile] = File(None),
        x_session_id: Optional[str] = Header(None),
        gene_col: str = Form("gene"),
        padj_col: str = Form("fdr"),
        p_value_col: str = Form("null"),
        lfc_col: str = Form("log2FC"),
        padj_threshold: float = Form(0.05),
        lfc_threshold: float = Form(0.25),
        use_nominal_p: bool = Form(False),
        confirm_non_significant: bool = Form(False)
    ):
        """
        Upload DEG (Differentially Expressed Genes) file for annotation.
        
        The DEG file should contain at minimum:
        - Gene symbol column
        - Adjusted p-value column
        - Log2 fold change column
        
        DEG genes (padj < threshold AND abs(log2FC) > threshold) will be
        highlighted in results and exported with the analysis.
        """
        if not x_session_id:
            raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session expired")
        
        try:
            if file is not None:
                content = await file.read()
                deg_df = _load_df_from_file(content, file.filename)
                filename = file.filename
            else:
                # Allow confirmation without re-uploading the file by reusing pending DEG data.
                pending_df = session.get("pending_deg_df")
                if pending_df is None:
                    raise HTTPException(
                        status_code=400,
                        detail="DEG file is required for initial upload."
                    )
                deg_df = pending_df.copy()
                filename = "pending_deg_upload.csv"
            
            # Validate required columns exist
            required_cols = [gene_col, padj_col, lfc_col]
            missing = [c for c in required_cols if c not in deg_df.columns]
            if missing:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required columns: {missing}. Available: {list(deg_df.columns)}"
                )

            p_value_available = bool(p_value_col and p_value_col != "null" and p_value_col in deg_df.columns)
            if p_value_col and p_value_col != "null" and not p_value_available:
                raise HTTPException(
                    status_code=400,
                    detail=f"Optional p_value column '{p_value_col}' not found. Available: {list(deg_df.columns)}"
                )

            padj_values = pd.to_numeric(deg_df[padj_col], errors="coerce")
            n_total = int(len(deg_df))
            n_significant = int((padj_values < padj_threshold).sum())
            n_non_significant = n_total - n_significant
            all_significant = n_non_significant == 0

            # Require explicit user confirmation before switching to nominal p-values.
            if (not all_significant and p_value_available and not use_nominal_p and not confirm_non_significant):
                nominal_values = pd.to_numeric(deg_df[p_value_col], errors="coerce")
                n_nominal = int((nominal_values < 0.05).sum())
                session["pending_deg_df"] = deg_df
                return {
                    "status": "needs_confirmation",
                    "message": (
                        f"{n_non_significant} genes are not significant at {padj_col} < {padj_threshold}. "
                        f"Continue with nominal {p_value_col} < 0.05 for hypothesis generation?"
                    ),
                    "deg_significance": {
                        "checked_col": padj_col,
                        "threshold": padj_threshold,
                        "all_significant": False,
                        "n_total": n_total,
                        "n_significant": n_significant,
                        "n_non_significant": n_non_significant,
                        "p_value_col": p_value_col,
                        "n_nominal_p_below_0_05": n_nominal
                    }
                }

            effective_p_col = padj_col
            effective_threshold = padj_threshold
            using_nominal_p = False

            if use_nominal_p:
                if not p_value_available:
                    raise HTTPException(
                        status_code=400,
                        detail="Cannot use nominal p-value threshold because optional p_value column is not provided."
                    )
                effective_p_col = p_value_col
                effective_threshold = 0.05
                using_nominal_p = True

            deg_p_values = pd.to_numeric(deg_df[effective_p_col], errors="coerce")
            deg_lfc_values = pd.to_numeric(deg_df[lfc_col], errors="coerce")

            # Mark DEG status
            deg_df["DEG"] = (
                (deg_p_values < effective_threshold) &
                (deg_lfc_values.abs() > lfc_threshold)
            )

            n_degs = int(deg_df["DEG"].sum())
            
            # Store in session
            config = {
                "gene_col": gene_col,
                "padj_col": effective_p_col,
                "lfc_col": lfc_col,
                "padj_threshold": effective_threshold,
                "lfc_threshold": lfc_threshold,
                "source_padj_col": padj_col,
                "p_value_col": p_value_col if p_value_available else None,
                "using_nominal_p": using_nominal_p
            }
            session_manager.update_deg_data(x_session_id, deg_df, config)
            session["pending_deg_df"] = None
            
            return {
                "status": "success",
                "filename": filename,
                "n_genes": len(deg_df),
                "n_degs": n_degs,
                "columns": list(deg_df.columns),
                "preview": deg_df.head(5).to_dict(orient="records"),
                "deg_significance": {
                    "checked_col": padj_col,
                    "threshold": padj_threshold,
                    "all_significant": all_significant,
                    "n_total": n_total,
                    "n_significant": n_significant,
                    "n_non_significant": n_non_significant,
                    "using_nominal_p": using_nominal_p,
                    "effective_col": effective_p_col,
                    "effective_threshold": effective_threshold
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/deg/preview")
    async def preview_deg_file(file: UploadFile = File(...)):
        """Preview DEG file to get columns."""
        try:
            content = await file.read()
            df = _load_df_from_file(content, file.filename)
            return {"status": "success", "columns": list(df.columns)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")

    @app.delete("/api/deg")
    async def remove_deg_file(x_session_id: Optional[str] = Header(None)):
        """Remove uploaded DEG file from session."""
        if not x_session_id:
            raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session expired")
        
        session["deg_df"] = None
        session["deg_config"] = None
        session["pending_deg_df"] = None
        return {"status": "success", "message": "DEG data removed"}

    @app.post("/api/elbow")
    async def get_elbow_data(
        x_session_id: Optional[str] = Header(None),
        pathway_col: str = Form(...),
        genes_col: str = Form(...),
        score_col: str = Form(None),
        nes_direction: str = Form("all"),
        k_min: int = Form(2),
        k_max: int = Form(10)
    ):
        """
        Compute elbow plot data with automatic k suggestion.
        
        Returns metrics for determining optimal number of clusters
        before running the main analysis.
        """
        if not x_session_id: 
            raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session: 
            raise HTTPException(status_code=404, detail="Session expired")

        try:
            df = session["df"]
            
            # Determine score column
            effective_score_col = _resolve_score_column(df, score_col)
            if effective_score_col is None:
                raise HTTPException(status_code=400, detail="NES/score column is required.")

            filtered_df, nes_filter_summary = _apply_nes_direction_filter(
                df=df,
                score_col=effective_score_col,
                nes_direction=nes_direction
            )
            if len(filtered_df) < 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Need at least 3 pathways after NES filter (got {len(filtered_df)})."
                )

            effective_k_max = min(k_max, len(filtered_df) - 1)
            if effective_k_max < k_min:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"k_min={k_min} is too large for filtered data (n={len(filtered_df)}). "
                        f"Set k_min <= {max(2, len(filtered_df) - 1)}."
                    )
                )
            
            # Create analyzer just for elbow computation
            analyzer = PathwayAnalyzer(
                filtered_df,
                pathway_col=pathway_col, 
                genes_col=genes_col, 
                score_col=effective_score_col
            )
            
            # Compute MDS first
            analyzer.run_mds()
            
            # Get optimal k suggestion
            k_range = range(k_min, effective_k_max + 1)
            result = analyzer.suggest_optimal_k(k_range=k_range)
            
            # Also get MDS scree data
            scree_data = analyzer.compute_mds_scree(max_dims=min(10, len(df) - 1))
            mds_gof = analyzer.get_mds_gof()
            
            # Store analyzer for later use
            session_manager.update_analyzer(x_session_id, analyzer)
            
            return json_clean({
                "status": "success",
                "elbow": {
                    "optimal_k": result['optimal_k'],
                    "elbow_k": result['elbow_k'],
                    "silhouette_k": result['silhouette_k'],
                    "confidence": result['confidence'],
                    "metrics": result['metrics'].to_dict(orient="records")
                },
                "mds_scree": {
                    "suggested_dims": int(scree_data[scree_data['suggested']]['dimension'].values[0]) if scree_data['suggested'].any() else 3,
                    "gof": list(mds_gof),
                    "data": scree_data.to_dict(orient="records")
                },
                "nes_filter": nes_filter_summary
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/analyze")
    async def run_analysis(
        x_session_id: Optional[str] = Header(None),
        pathway_col: str = Form(...),
        genes_col: str = Form(...),
        score_col: str = Form(None),
        nes_direction: str = Form("all"),
        n_clusters: int = Form(5),
        report_name: str = Form("")
    ):
        if not x_session_id: raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session: raise HTTPException(status_code=404, detail="Session expired")

        try:
            df = session["df"]
            deg_df = session.get("deg_df")
            deg_config = session.get("deg_config")
            session_manager.update_report_name(x_session_id, report_name)
            
            # Check if we already have an analyzer from /api/elbow
            existing_analyzer = session.get("analyzer")
            
            # Determine score column - NES is required for full analysis
            effective_score_col = _resolve_score_column(df, score_col)
            if effective_score_col is None:
                raise HTTPException(
                    status_code=400, 
                    detail="NES/score column is required. Please specify a valid score column or ensure your data has 'NES' column."
                )

            filtered_df, nes_filter_summary = _apply_nes_direction_filter(
                df=df,
                score_col=effective_score_col,
                nes_direction=nes_direction
            )
            if n_clusters > len(filtered_df):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"n_clusters={n_clusters} exceeds filtered pathways ({len(filtered_df)}). "
                        "Lower k or use all NES directions."
                    )
                )
            
            # Reuse existing analyzer if available and compatible
            if (existing_analyzer is not None and 
                existing_analyzer.pathway_col == pathway_col and
                existing_analyzer.genes_col == genes_col and
                existing_analyzer.score_col == effective_score_col and
                getattr(existing_analyzer, "_nes_direction", "all") == nes_filter_summary["nes_direction"] and
                len(existing_analyzer.raw_data) == len(filtered_df)):
                analyzer = existing_analyzer
            else:
                analyzer = PathwayAnalyzer(
                    filtered_df,
                    pathway_col=pathway_col, 
                    genes_col=genes_col, 
                    score_col=effective_score_col
                )
                analyzer._nes_direction = nes_filter_summary["nes_direction"]
            
            analyzer.run_clustering(n_clusters=n_clusters)
            session_manager.update_analyzer(x_session_id, analyzer)
            
            # Prepare standard response
            plotly_dict = analyzer.plot.scatter_3d(as_dict=True)
            
            # Clean results for JSON compliance
            scatter_data = analyzer.cluster_results.to_dict(orient="records")
            
            # Analyze gene frequencies with optional DEG annotation
            if deg_df is not None and deg_config is not None:
                gene_stats_df = analyzer.analyze_gene_frequencies(
                    deg_object=deg_df,
                    deg_gene_col=deg_config["gene_col"],
                    deg_padj_col=deg_config["padj_col"],
                    deg_lfc_col=deg_config["lfc_col"],
                    padj_threshold=deg_config["padj_threshold"],
                    lfc_threshold=deg_config["lfc_threshold"]
                )
            else:
                gene_stats_df = analyzer.analyze_gene_frequencies()
            
            gene_stats = gene_stats_df.to_dict(orient="records")
            
            # Compute cluster statistics
            cluster_stats = compute_cluster_statistics(analyzer)
            
            # Get MDS goodness-of-fit if available
            try:
                mds_gof = list(analyzer.get_mds_gof())
            except:
                mds_gof = [0.0, 0.0]
            
            # Check if DEG data is available
            has_deg = deg_df is not None
            n_deg_genes = 0
            if has_deg and 'DEG' in gene_stats_df.columns:
                n_deg_genes = int(gene_stats_df['DEG'].sum())
            
            return json_clean({
                "status": "success",
                "scatter_data": scatter_data,
                "plotly_json": plotly_dict,
                "gene_stats": gene_stats,
                "clusters": sorted(list(analyzer.cluster_results['Cluster'].unique())),
                "cluster_stats": cluster_stats,
                "mds_gof": mds_gof,
                "total_pathways": len(analyzer.cluster_results),
                "total_genes": len(set(g for genes in analyzer.raw_data[analyzer.genes_col] for g in genes if genes)),
                "has_deg": has_deg,
                "n_deg_genes": n_deg_genes,
                "nes_filter": nes_filter_summary
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/auto-analyze")
    async def auto_analyze(
        x_session_id: Optional[str] = Header(None),
        pathway_col: str = Form(...),
        genes_col: str = Form(...),
        score_col: str = Form(None),
        nes_direction: str = Form("all"),
        api_key: str = Form(...),
        provider: str = Form("openai"),
        turbo: bool = Form(False),
        k_min: int = Form(2),
        k_max: int = Form(10),
        report_name: str = Form("")
    ):
        """
        Fully automated analysis with AI-driven optimal k selection.
        
        This endpoint:
        1. Computes elbow + silhouette metrics for k_min to k_max
        2. Uses AI agent to analyze the metrics and determine optimal k
        3. Runs clustering with AI-recommended k
        4. Optionally runs AI annotation
        
        Returns complete analysis results with AI-chosen parameters.
        """
        if not x_session_id:
            raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session expired")
        
        if build_cluster_decider_graph is None:
            raise HTTPException(
                status_code=501, 
                detail="AI module not installed. Install with: pip install gemmap[ai]"
            )
        
        try:
            df = session["df"]
            deg_df = session.get("deg_df")
            deg_config = session.get("deg_config")
            session_manager.update_report_name(x_session_id, report_name)
            
            # Determine score column
            effective_score_col = _resolve_score_column(df, score_col)
            if effective_score_col is None:
                raise HTTPException(
                    status_code=400,
                    detail="NES/score column is required."
                )

            filtered_df, nes_filter_summary = _apply_nes_direction_filter(
                df=df,
                score_col=effective_score_col,
                nes_direction=nes_direction
            )
            if len(filtered_df) < 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Need at least 3 pathways after NES filter for auto-analyze (got {len(filtered_df)})."
                )

            effective_k_max = min(k_max, len(filtered_df) - 1)
            if effective_k_max < k_min:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"k_min={k_min} is too large for filtered data (n={len(filtered_df)}). "
                        f"Set k_min <= {max(2, len(filtered_df) - 1)}."
                    )
                )
            
            # Create analyzer and compute MDS
            analyzer = PathwayAnalyzer(
                filtered_df,
                pathway_col=pathway_col,
                genes_col=genes_col,
                score_col=effective_score_col
            )
            analyzer._nes_direction = nes_filter_summary["nes_direction"]
            analyzer.run_mds()
            
            # Compute elbow metrics
            k_range = range(k_min, effective_k_max + 1)
            elbow_result = analyzer.suggest_optimal_k(k_range=k_range)
            
            # Prepare data for AI cluster decider
            metrics_data = elbow_result['metrics'].to_dict(orient="records")
            
            # Run AI cluster decider
            graph = build_cluster_decider_graph()
            ai_result = await graph.ainvoke({
                "metrics_data": metrics_data,
                "elbow_k": elbow_result['elbow_k'],
                "silhouette_k": elbow_result['silhouette_k'],
                "statistical_k": elbow_result['optimal_k'],
                "confidence": elbow_result['confidence'],
                "n_pathways": len(filtered_df),
                "api_key": api_key,
                "model_provider": provider,
                "turbo": bool(turbo),
            })
            
            # Get AI-recommended k
            ai_recommended_k = ai_result.get('recommended_k', elbow_result['optimal_k'])
            ai_reasoning = ai_result.get('reasoning', 'Statistical default used.')
            ai_confidence = ai_result.get('ai_confidence', elbow_result['confidence'])
            ai_provider = ai_result.get('model_provider', provider)
            ai_model = ai_result.get('model_name', f"AI Agent ({ai_provider})")
            
            # Store AI k
            session_manager.update_auto_k(x_session_id, ai_recommended_k)
            
            # Update analyzer metadata explicitly for Autopilot
            analyzer.add_ai_metadata(
                mode="Autopilot Turbo" if bool(turbo) else "Autopilot",
                model=ai_model,
                provider=ai_provider,
                reasoning=ai_reasoning
            )
            
            # Run clustering with AI-recommended k
            analyzer.run_clustering(n_clusters=ai_recommended_k)
            session_manager.update_analyzer(x_session_id, analyzer)
            
            # Prepare response
            plotly_dict = analyzer.plot.scatter_3d(as_dict=True)
            scatter_data = analyzer.cluster_results.to_dict(orient="records")
            
            # Gene stats with optional DEG
            if deg_df is not None and deg_config is not None:
                gene_stats_df = analyzer.analyze_gene_frequencies(
                    deg_object=deg_df,
                    deg_gene_col=deg_config["gene_col"],
                    deg_padj_col=deg_config["padj_col"],
                    deg_lfc_col=deg_config["lfc_col"],
                    padj_threshold=deg_config["padj_threshold"],
                    lfc_threshold=deg_config["lfc_threshold"]
                )
            else:
                gene_stats_df = analyzer.analyze_gene_frequencies()
            
            gene_stats = gene_stats_df.to_dict(orient="records")
            cluster_stats = compute_cluster_statistics(analyzer)
            
            try:
                mds_gof = list(analyzer.get_mds_gof())
            except:
                mds_gof = [0.0, 0.0]
            
            has_deg = deg_df is not None
            n_deg_genes = int(gene_stats_df['DEG'].sum()) if has_deg and 'DEG' in gene_stats_df.columns else 0
            
            return json_clean({
                "status": "success",
                "auto_k": {
                    "recommended_k": ai_recommended_k,
                    "reasoning": ai_reasoning,
                    "confidence": ai_confidence,
                    "elbow_k": elbow_result['elbow_k'],
                    "silhouette_k": elbow_result['silhouette_k'],
                    "statistical_k": elbow_result['optimal_k'],
                    "provider": ai_provider,
                    "model": ai_model,
                    "turbo": bool(turbo),
                },
                "scatter_data": scatter_data,
                "plotly_json": plotly_dict,
                "gene_stats": gene_stats,
                "clusters": sorted(list(analyzer.cluster_results['Cluster'].unique())),
                "cluster_stats": cluster_stats,
                "mds_gof": mds_gof,
                "total_pathways": len(analyzer.cluster_results),
                "total_genes": len(set(g for genes in analyzer.raw_data[analyzer.genes_col] for g in genes if genes)),
                "has_deg": has_deg,
                "n_deg_genes": n_deg_genes,
                "elbow": {
                    "metrics": elbow_result['metrics'].to_dict(orient="records")
                },
                "nes_filter": nes_filter_summary
            })
        except HTTPException:
            raise
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "key" in error_msg.lower():
                error_msg = "Authentication failed. Please check your API key."
            raise HTTPException(status_code=500, detail=f"Auto-analyze Error: {error_msg}")

    @app.post("/api/annotate")
    async def annotate_clusters(
        req: AnnotationRequest,
        x_session_id: Optional[str] = Header(None)
    ):
        """
        Agentic Endpoint: Triggers LangGraph to annotate clusters.
        """
        if not x_session_id: raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session or not session.get("analyzer"): 
            raise HTTPException(status_code=400, detail="Run analysis first")
            
        if build_annotation_graph is None:
             raise HTTPException(status_code=501, detail="AI module not installed. Install 'gemmap[ai]'.")

        analyzer = session["analyzer"]
        provider = req.provider
        if normalize_provider is not None:
            provider = normalize_provider(provider)
        if provider == "openai" and get_provider_from_key is not None:
            try:
                detected = get_provider_from_key(req.api_key)
                if detected:
                    provider = detected
            except Exception:
                pass
        
        # 1. Prepare Data for Agents
        # We need to extract top pathways and top genes for each cluster
        clusters_data = []
        cluster_ids = analyzer.cluster_results['Cluster'].unique()
        
        gene_stats = analyzer.gene_stats 
        cluster_df = analyzer.cluster_results
        
        for cid in cluster_ids:
            # Top pathways by NES (if available) or random sample
            c_paths = cluster_df[cluster_df['Cluster'] == cid]
            if analyzer.score_col in c_paths.columns:
                top_paths = c_paths.sort_values(analyzer.score_col, ascending=False).head(6)[analyzer.pathway_col].tolist()
            else:
                top_paths = c_paths.head(6)[analyzer.pathway_col].tolist()
                
            # Top genes by frequency
            c_genes = gene_stats[gene_stats['Cluster'] == cid].sort_values('Percentage', ascending=False).head(12)['Item'].tolist()
            
            clusters_data.append({
                "cluster_id": cid,
                "top_pathways": top_paths,
                "top_genes": c_genes
            })

        # Fast cache for repeated runs with identical cluster context.
        fingerprint_payload = [{
            "cluster_id": str(c["cluster_id"]),
            "top_pathways": c["top_pathways"],
            "top_genes": c["top_genes"],
        } for c in sorted(clusters_data, key=lambda x: int(x["cluster_id"]))]
        signature = hashlib.sha1(
            json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        tier = "turbo" if bool(req.turbo) else "standard"
        cache_key = f"{provider}:{tier}:{signature}"
        annotation_cache = session.get("annotation_cache") or {}
        if cache_key in annotation_cache:
            annotations = annotation_cache[cache_key]
            session_manager.update_ai_results(x_session_id, annotations)
            cached_mode = analyzer.methodology.get("mode", "Manual + semi-autopilot") if hasattr(analyzer, "methodology") else "Manual + semi-autopilot"
            if bool(req.turbo) and "Turbo" not in cached_mode:
                cached_mode = f"{cached_mode} (Turbo)"
            analyzer.add_ai_metadata(provider=provider, model="cached", mode=cached_mode)
            analyzer.add_ai_annotations(annotations)
            session_manager.update_analyzer(x_session_id, analyzer)
            return {"status": "success", "annotations": annotations, "cached": True, "turbo": bool(req.turbo)}

        # 2. Run LangGraph
        # SECURITY: API key is passed directly to LLM and is NOT stored anywhere
        try:
            graph = build_annotation_graph()
            result = await asyncio.wait_for(
                graph.ainvoke({
                    "clusters_data": clusters_data,
                    "api_key": req.api_key,  # Used transiently, never stored
                    "model_provider": provider,
                    "turbo": bool(req.turbo),
                    "annotations": []
                }),
                timeout=35
            )
            
            annotations = [a.dict() for a in result['annotations']]
            # Only annotations are stored, NOT the API key
            session_manager.update_ai_results(x_session_id, annotations)
            annotation_cache[cache_key] = annotations
            session["annotation_cache"] = annotation_cache
            
            # Update analyzer with metadata and annotations
            provider = result.get('model_provider', provider)
            model = result.get('model_name', 'unknown')
            current_mode = analyzer.methodology.get("mode", "Manual") if hasattr(analyzer, "methodology") else "Manual"
            if current_mode == "Autopilot":
                mode_to_set = "Autopilot Turbo" if bool(req.turbo) else "Autopilot"
            else:
                mode_to_set = "Manual + semi-autopilot (Turbo)" if bool(req.turbo) else "Manual + semi-autopilot"
            analyzer.add_ai_metadata(provider=provider, model=model, mode=mode_to_set)
            analyzer.add_ai_annotations(annotations)
            session_manager.update_analyzer(x_session_id, analyzer)
            
            return {"status": "success", "annotations": annotations, "turbo": bool(req.turbo)}
        except asyncio.TimeoutError:
            fallback_annotations = []
            for c in clusters_data:
                top_paths = ", ".join(c["top_pathways"][:3]) if c["top_pathways"] else "N/A"
                top_genes = ", ".join(c["top_genes"][:6]) if c["top_genes"] else "N/A"
                title_seed = str(c["top_pathways"][0]) if c["top_pathways"] else f"Module {c['cluster_id']}"
                title_clean = title_seed.replace("_", " ").strip()
                if len(title_clean) > 72:
                    title_clean = f"{title_clean[:69]}..."
                if not title_clean:
                    title_clean = f"Module {c['cluster_id']}"
                fallback_annotations.append({
                    "cluster_id": str(c["cluster_id"]),
                    "title": title_clean,
                    "summary": f"Top pathways: {top_paths}. Top genes: {top_genes}.",
                    "confidence": 0.58,
                    "key_process": "Pathway-driven summary"
                })
            session_manager.update_ai_results(x_session_id, fallback_annotations)
            annotation_cache[cache_key] = fallback_annotations
            session["annotation_cache"] = annotation_cache
            current_mode = analyzer.methodology.get("mode", "Manual") if hasattr(analyzer, "methodology") else "Manual"
            if current_mode == "Autopilot":
                mode_to_set = "Autopilot Turbo" if bool(req.turbo) else "Autopilot"
            else:
                mode_to_set = "Manual + semi-autopilot (Turbo)" if bool(req.turbo) else "Manual + semi-autopilot"
            analyzer.add_ai_metadata(provider=provider, model="timeout-fallback", mode=mode_to_set)
            analyzer.add_ai_annotations(fallback_annotations)
            session_manager.update_analyzer(x_session_id, analyzer)
            return {
                "status": "success",
                "annotations": fallback_annotations,
                "fallback": True,
                "message": "AI annotation timed out; showing pathway-driven module summaries.",
                "turbo": bool(req.turbo),
            }
            
        except Exception as e:
            # Don't expose full error which might contain sensitive info
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "key" in error_msg.lower():
                error_msg = "Authentication failed. Please check your API key."
            raise HTTPException(status_code=500, detail=f"Agent Error: {error_msg}")

    @app.post("/api/cluster-chat")
    async def cluster_chat(
        req: ClusterChatRequest,
        x_session_id: Optional[str] = Header(None)
    ):
        """
        Interactive research chat for cluster interpretation with optional PubMed grounding.
        """
        if not x_session_id:
            raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session or not session.get("analyzer"):
            raise HTTPException(status_code=400, detail="Run analysis first")
        if not req.message or not req.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        if _create_llm is None:
            raise HTTPException(status_code=501, detail="AI chat module not installed. Install 'gemmap[ai]'.")

        analyzer = session["analyzer"]
        ai_results = session.get("ai_results") or []
        provider = req.provider
        if normalize_provider is not None:
            provider = normalize_provider(provider)
        if provider == "openai" and get_provider_from_key is not None:
            try:
                detected = get_provider_from_key(req.api_key)
                if detected:
                    provider = detected
            except Exception:
                pass

        context = _prepare_cluster_chat_context(analyzer, ai_results, req.cluster_id)
        message_gene_tokens = _extract_gene_like_tokens(req.message)
        search_genes = list(dict.fromkeys((context.get("genes_for_search") or []) + message_gene_tokens))[:6]

        pathway_candidates_raw = [context.get("pathway_for_search", "")] + list(context.get("pathways_for_search") or [])
        pathway_candidates = []
        seen_pathway = set()
        for p in pathway_candidates_raw:
            norm = _pathway_label_to_query(str(p or ""))
            if not norm:
                continue
            key = norm.lower()
            if key in seen_pathway:
                continue
            seen_pathway.add(key)
            pathway_candidates.append(norm)

        message_tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-_]{2,20}", req.message)
        msg_stop = {
            "please", "could", "would", "should", "study", "module", "modules", "pathway", "pathways",
            "genes", "analysis", "result", "results", "show", "tell", "about", "with", "from", "that"
        }
        message_terms = []
        for tok in message_tokens:
            low = tok.lower()
            if low in msg_stop:
                continue
            if low.isdigit():
                continue
            message_terms.append(low)
        message_terms = list(dict.fromkeys(message_terms))[:8]

        query_candidates: List[str] = []
        for pathway_phrase in pathway_candidates[:3]:
            if search_genes:
                query_candidates.append(f"\"{pathway_phrase}\" AND ({' OR '.join(search_genes[:4])})")
            query_candidates.append(f"\"{pathway_phrase}\" AND (pathway OR mechanism OR enrichment)")
        if search_genes:
            query_candidates.append(f"({' OR '.join(search_genes)}) AND (transcriptome OR \"RNA-seq\" OR pathway)")
        if message_terms:
            query_candidates.append(" ".join(message_terms) + " pathway mechanism")
        if not query_candidates:
            query_candidates.append(req.message.strip())

        papers: List[Dict[str, str]] = []
        literature_queries_used: List[str] = []
        if req.include_literature:
            seen_papers = set()
            for query in query_candidates:
                hits = _fetch_pubmed_summaries(query, retmax=4)
                if hits:
                    literature_queries_used.append(query)
                for hit in hits:
                    key = str(hit.get("pmid") or hit.get("url") or hit.get("title") or "").strip().lower()
                    if not key or key in seen_papers:
                        continue
                    seen_papers.add(key)
                    papers.append(hit)
                    if len(papers) >= 8:
                        break
                if len(papers) >= 8:
                    break
        literature_query = " || ".join(literature_queries_used[:3]) if literature_queries_used else query_candidates[0]

        paper_context = "\n".join(
            f"- PMID {p['pmid']}: {p['title']} ({p['journal']}, {p['pubdate']})"
            for p in papers
        ) or "No paper snippets retrieved."

        history_items = req.history[-8:] if isinstance(req.history, list) else []
        history_text = "\n".join(
            f"{str(h.get('role', 'user')).upper()}: {str(h.get('text') or h.get('content') or '')[:300]}"
            for h in history_items
            if isinstance(h, dict)
        )

        system_prompt = (
            "You are a study-level research copilot for GEMMAP.\n"
            "Use only the provided context from uploaded pathway/DEG results and literature snippets.\n"
            "Be concise, scientific, and practical. Distinguish evidence vs hypothesis.\n"
            "If referencing provided papers, cite PMID in square brackets.\n"
            "If user asks to change analysis settings, propose exact changes (module count, NES focus, filters) in plain language."
        )
        user_prompt = (
            f"Global module snapshot:\n{context.get('global_context', '')}\n\n"
            f"Focused module context:\n{context.get('cluster_context', '')}\n\n"
            f"Recent chat history:\n{history_text or 'None'}\n\n"
            f"PubMed snippets:\n{paper_context}\n\n"
            f"User question:\n{req.message.strip()}\n\n"
            "Answer with:\n"
            "1) Interpretation\n"
            "2) Evidence and confidence\n"
            "3) Suggested next checks"
        )

        try:
            llm = _create_llm(provider, req.api_key, turbo=bool(req.turbo))
            reply = await llm.ainvoke(f"{system_prompt}\n\n{user_prompt}")
            content = getattr(reply, "content", str(reply))
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)
            chat_log = session.setdefault("chat_log", [])
            chat_log.append({
                "timestamp_utc": _now_utc_iso(),
                "provider": provider,
                "selected_module": context.get("selected_cluster"),
                "user_message": req.message.strip(),
                "assistant_reply": str(content),
                "literature": papers,
                "query": literature_query,
            })
            if len(chat_log) > 500:
                session["chat_log"] = chat_log[-500:]
            return {
                "status": "success",
                "reply": str(content),
                "literature": papers,
                "selected_cluster": context.get("selected_cluster"),
                "query": literature_query,
                "provider": provider,
                "turbo": bool(req.turbo),
            }
        except Exception as e:
            msg = str(e)
            if "api_key" in msg.lower() or "key" in msg.lower() or "auth" in msg.lower():
                msg = "Authentication failed. Please check your API key."
            raise HTTPException(status_code=500, detail=f"Module chat error: {msg}")

    @app.post("/api/chat/clear")
    async def clear_chat_history(x_session_id: Optional[str] = Header(None)):
        """
        Clear stored chat history for the current session.
        This affects reproducibility exports when chat inclusion is enabled.
        """
        if not x_session_id:
            raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session:
            raise HTTPException(status_code=400, detail="Invalid session.")
        session["chat_log"] = []
        return {"status": "success", "cleared": True}

    @app.get("/api/mountain-data")
    async def get_mountain_data(x_session_id: Optional[str] = Header(None)):
        """
        Provide ranked-gene + pathway payload for in-app Mountain Explorer.
        """
        if not x_session_id:
            raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session or not session.get("analyzer"):
            raise HTTPException(status_code=400, detail="Run analysis first")

        analyzer = session["analyzer"]
        ai_results = session.get("ai_results") or []
        cluster_df = analyzer.cluster_results if analyzer.cluster_results is not None else pd.DataFrame()
        if cluster_df.empty:
            raise HTTPException(status_code=400, detail="No clustered pathways available.")

        clusters = sorted(cluster_df["Cluster"].unique(), key=lambda x: int(x))
        cluster_labels = {str(c): f"Module {c}" for c in clusters}
        for row in ai_results:
            cid = str(row.get("cluster_id"))
            title = str(row.get("title") or "").strip()
            if cid in cluster_labels and title:
                cluster_labels[cid] = title

        payload = _build_mountain_data(
            analyzer=analyzer,
            cluster_labels=cluster_labels,
            deg_df=session.get("deg_df"),
            deg_config=session.get("deg_config"),
        )
        return {
            "status": "success",
            "n_pathways": len(payload.get("pathways", [])),
            "has_data": bool(payload.get("has_data", False)),
            "pathways": payload.get("pathways", []),
            "ranked_genes": payload.get("ranked_genes", []),
        }

    @app.post("/api/pathway-origin")
    async def pathway_origin_literature(
        req: PathwayOriginRequest,
        x_session_id: Optional[str] = Header(None)
    ):
        """
        Search PubMed for pathway naming/origin evidence (PMID/DOI/abstract).
        """
        if not req.pathway or not req.pathway.strip():
            raise HTTPException(status_code=400, detail="Pathway name is required.")
        if _create_llm is None:
            raise HTTPException(status_code=501, detail="AI analysis module not installed. Install 'gemmap[ai]'.")
        if not req.api_key or not str(req.api_key).strip():
            raise HTTPException(
                status_code=400,
                detail="API key is required for Trail Talk agentic paper search.",
            )

        provider = req.provider
        if normalize_provider is not None:
            provider = normalize_provider(provider)
        if provider == "openai" and get_provider_from_key is not None:
            try:
                detected = get_provider_from_key(req.api_key)
                if detected:
                    provider = detected
            except Exception:
                pass

        result = _fetch_pubmed_origin_papers(
            req.pathway,
            retmax=req.max_results,
            use_naming_clue=bool(req.use_naming_clue),
            msigdb_go_only=bool(req.msigdb_go_only),
        )
        normalized = _pathway_label_to_query(req.pathway)
        candidate_papers = list(result.get("candidate_papers") or [])
        naming_clue_payload = result.get("naming_clue", {}) or {}
        strict_go_id = str(naming_clue_payload.get("go_id") or "").strip().upper()
        strict_go_term = str(naming_clue_payload.get("go_term_name") or "").strip()

        llm_no_match = True
        llm_confidence = "low"
        llm_summary = "I could not find relevant evidence."
        llm_reason_by_pmid: Dict[str, str] = {}
        llm_selected_pmids: List[str] = []

        if candidate_papers:
            paper_rows = []
            for p in candidate_papers[:30]:
                paper_rows.append({
                    "pmid": str(p.get("pmid") or "").strip(),
                    "title": _compact_text(p.get("title"), 220),
                    "journal": _compact_text(p.get("journal"), 120),
                    "year": _compact_text(p.get("year"), 20),
                    "doi": _compact_text(p.get("doi"), 100),
                    "abstract": _compact_text(p.get("abstract"), 520),
                })
            naming_clue = result.get("naming_clue") or {}
            clue_rows = {
                "namespace": naming_clue.get("namespace"),
                "prefix": naming_clue.get("prefix"),
                "is_msigdb_go": naming_clue.get("is_msigdb_go"),
                "go_id": naming_clue.get("go_id"),
                "go_term_name": naming_clue.get("go_term_name"),
                "go_definition": _compact_text(naming_clue.get("go_definition"), 320),
                "definition_references": list(naming_clue.get("definition_references") or [])[:12],
                "amigo_term_url": naming_clue.get("amigo_term_url"),
                "go_ref_ids": list(naming_clue.get("go_ref_ids") or [])[:20],
                "go_ref_urls": list(naming_clue.get("go_ref_urls") or [])[:12],
                "reference_pmids": list(naming_clue.get("reference_pmids") or [])[:20],
                "reference_dois": list(naming_clue.get("reference_dois") or [])[:12],
                "reference_urls": list(naming_clue.get("reference_urls") or [])[:10],
                "msigdb_reference_pmids": list(naming_clue.get("msigdb_reference_pmids") or [])[:20],
                "quickgo_reference_pmids": list(naming_clue.get("quickgo_reference_pmids") or [])[:20],
                "quickgo_annotation_pmids": list(naming_clue.get("quickgo_annotation_pmids") or [])[:20],
                "amigo_reference_pmids": list(naming_clue.get("amigo_reference_pmids") or [])[:20],
                "go_ref_reference_pmids": list(naming_clue.get("go_ref_reference_pmids") or [])[:20],
                "doi_resolved_pmids": list(naming_clue.get("doi_resolved_pmids") or [])[:20],
                "brief_description": _compact_text(naming_clue.get("brief_description"), 280),
            }
            llm_system = (
                "You are a strict literature-origin validation agent.\n"
                "Task: identify papers that directly explain why this pathway/term name exists.\n"
                "Rules:\n"
                "- Use ONLY the provided candidate papers and naming clues.\n"
                "- Never invent papers or IDs.\n"
                "- If direct origin evidence is not clear, return no_match=true.\n"
                "- In GO-only mode, reject generic GO curation/method papers unless they are directly tied to the specific GO term identity (GO ID / exact term signal / clue-linked reference).\n"
                "- In GO-only mode, do NOT select broad ontology workflow papers just because they discuss GO annotation.\n"
                "- Keep reasons concise and factual.\n"
                "Return JSON only with schema:\n"
                "{\"no_match\": boolean, \"confidence\": \"high|medium|low\", "
                "\"summary\": \"short\", "
                "\"selected\": [{\"pmid\":\"\", \"reason\":\"short\"}]}"
            )
            llm_user = (
                f"Pathway title (exact): {req.pathway}\n"
                f"Pathway title (display): {normalized}\n"
                f"GO-only mode: {bool(req.msigdb_go_only)}\n"
                f"Naming clues: {json.dumps(clue_rows, ensure_ascii=False)}\n"
                f"Candidate papers: {json.dumps(paper_rows, ensure_ascii=False)}\n"
                "Select only strong origin/naming evidence. "
                "In GO-only mode, enforce strict term-specific evidence and reject generic GO process papers. "
                "If uncertain, set no_match=true and summary='I could not find relevant evidence.'."
            )
            try:
                llm = _create_llm(provider, req.api_key, turbo=bool(req.turbo))
                reply = await llm.ainvoke(f"{llm_system}\n\n{llm_user}")
                llm_raw = getattr(reply, "content", str(reply))
                if isinstance(llm_raw, list):
                    llm_raw = "\n".join(str(x) for x in llm_raw)
                parsed = _extract_json_object_from_text(str(llm_raw))
                candidate_pmid_set = {
                    str(p.get("pmid") or "").strip()
                    for p in candidate_papers
                    if str(p.get("pmid") or "").strip()
                }
                selected_rows = parsed.get("selected") if isinstance(parsed, dict) else []
                if not isinstance(selected_rows, list):
                    selected_rows = []
                for row in selected_rows:
                    if not isinstance(row, dict):
                        continue
                    pmid = str(row.get("pmid") or "").strip()
                    reason = _compact_text(row.get("reason"), 180)
                    if not pmid or pmid not in candidate_pmid_set:
                        continue
                    if pmid in llm_selected_pmids:
                        continue
                    llm_selected_pmids.append(pmid)
                    if reason:
                        llm_reason_by_pmid[pmid] = reason
                llm_no_match = bool(parsed.get("no_match", not llm_selected_pmids)) if isinstance(parsed, dict) else True
                llm_confidence = str(parsed.get("confidence") or "low").strip().lower() if isinstance(parsed, dict) else "low"
                if llm_confidence not in {"high", "medium", "low"}:
                    llm_confidence = "low"
                llm_summary = _compact_text(
                    parsed.get("summary") if isinstance(parsed, dict) else "",
                    220
                ) or ("I could not find relevant evidence." if llm_no_match else "")
                if not llm_selected_pmids:
                    llm_no_match = True
                if llm_no_match:
                    llm_confidence = "low"
                    llm_summary = "I could not find relevant evidence."
            except Exception:
                llm_no_match = True
                llm_confidence = "low"
                llm_summary = "I could not find relevant evidence."

        selected_papers: List[Dict[str, Any]] = []
        selection_source = "none"
        paper_by_pmid = {
            str(p.get("pmid") or "").strip(): p
            for p in candidate_papers
            if str(p.get("pmid") or "").strip()
        }
        for pmid in llm_selected_pmids[: max(1, min(int(req.max_results or 5), 12))]:
            paper = paper_by_pmid.get(pmid)
            if not paper:
                continue
            if bool(req.msigdb_go_only) and not _paper_has_strict_origin_signal(
                paper=paper,
                pathway_raw_name=req.pathway,
                pathway_query_phrase=normalized,
                go_id=strict_go_id,
                go_term_name=strict_go_term,
            ):
                continue
            out = dict(paper)
            reason = llm_reason_by_pmid.get(pmid)
            if reason:
                out["selection_reason"] = reason
            out["selection_source"] = "llm_validated"
            selected_papers.append(out)
        if selected_papers:
            selection_source = "llm_validated"

        # If strict LLM validation finds no direct naming paper, but GO curation
        # references exist, return those deterministic references as source clues.
        if not selected_papers:
            go_id = str((result.get("naming_clue") or {}).get("go_id") or "").strip().upper()
            clue_pmids = [
                str(p).strip()
                for p in (result.get("clue_reference_pmids") or [])
                if str(p).strip()
            ]
            if go_id and clue_pmids:
                for pmid in clue_pmids:
                    paper = paper_by_pmid.get(pmid)
                    if not paper:
                        continue
                    out = dict(paper)
                    out["selection_reason"] = (
                        f"Referenced by ontology curation metadata for {go_id}; used as source-context evidence."
                    )
                    out["selection_source"] = "ontology_reference"
                    selected_papers.append(out)
                    if len(selected_papers) >= max(1, min(int(req.max_results or 5), 12)):
                        break
                if selected_papers:
                    selection_source = "ontology_reference"
                    llm_summary = (
                        f"I could not validate a direct naming paper. Showing GO curation references for {go_id}."
                    )

        if selected_papers:
            llm_no_match = False
            if not llm_summary:
                llm_summary = "Origin papers found."
        else:
            llm_no_match = True
            llm_confidence = "low"
            llm_summary = "I could not find relevant evidence."

        exact_name_found = bool(result.get("exact_name_found", False)) and bool(selected_papers)
        message = llm_summary or (
            "Origin papers found." if selected_papers else "I could not find relevant evidence."
        )
        naming_note = str(result.get("naming_note") or "").strip()
        go_id = str(naming_clue_payload.get("go_id") or "").strip().upper()
        clue_pmids = list(result.get("clue_reference_pmids") or [])
        definition_refs = list(naming_clue_payload.get("definition_references") or [])
        if (not selected_papers) and go_id and (not clue_pmids) and definition_refs:
            refs_preview = ", ".join(definition_refs[:3])
            if refs_preview:
                local_notes_extra = f"GO definition references available (non-PMID): {refs_preview}."
            else:
                local_notes_extra = ""
        else:
            local_notes_extra = ""
        local_mode_summary = (
            "Origin mode: deterministic GO/MSigDB-focused retrieval (exact-title + GO clues + GO references)."
            if bool(req.msigdb_go_only)
            else "Origin + Context mode: deterministic retrieval (exact-title + GO clues + broader metadata-linked PubMed queries)."
        )
        local_notes = list(result.get("local_search_notes", []))
        if local_notes_extra:
            local_notes.append(local_notes_extra)
        if selection_source == "ontology_reference":
            local_notes.append("No direct naming paper validated; returned GO curation reference papers.")
        candidate_preview: List[Dict[str, Any]] = []
        if (not selected_papers) and candidate_papers and (not bool(req.msigdb_go_only)):
            max_preview = max(1, min(int(req.max_results or 5), 12))
            for p in candidate_papers[:max_preview]:
                out = dict(p)
                out["selection_source"] = "candidate_pool_unvalidated"
                if not out.get("selection_reason"):
                    out["selection_reason"] = "Candidate from deterministic local retrieval; not validated as direct origin evidence."
                candidate_preview.append(out)
        if selection_source == "ontology_reference":
            match_type = "ontology_reference"
        elif selected_papers:
            match_type = "llm_validated"
        elif candidate_preview:
            match_type = "candidate_context"
        else:
            match_type = "none"
        if not selected_papers:
            if candidate_preview:
                message = (
                    f"No direct origin evidence validated. "
                    f"Showing {len(candidate_preview)} candidate context papers."
                )
            else:
                message = "I could not find relevant evidence."
        return {
            "status": "success",
            "pathway": req.pathway,
            "normalized_pathway": normalized,
            "query": result.get("query", ""),
            "queries": result.get("queries", []),
            "n_candidates": len(candidate_papers),
            "n_results": len(selected_papers),
            "exact_name_found": exact_name_found,
            "match_type": match_type,
            "message": message,
            "papers": selected_papers,
            "candidate_preview": candidate_preview,
            "naming_clue": result.get("naming_clue", {}),
            "naming_note": naming_note,
            "search_spec": {
                "local_search": local_mode_summary,
                "local_search_notes": local_notes,
                "llm_search": "LLM validates origin relevance only within retrieved candidates; no out-of-pool papers allowed.",
                "llm_no_match": llm_no_match,
                "llm_confidence": llm_confidence,
                "llm_model_tier": "turbo" if bool(req.turbo) else "standard",
            },
            "provider": provider,
            "turbo": bool(req.turbo),
        }

    @app.post("/api/pathway-paper-analyze")
    async def pathway_paper_analyze(
        req: PathwayPaperAnalysisRequest,
        x_session_id: Optional[str] = Header(None)
    ):
        """
        Analyze pathway-paper relevance with concise, non-hallucinatory output.
        """
        if not x_session_id:
            raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session or not session.get("analyzer"):
            raise HTTPException(status_code=400, detail="Run analysis first")
        if _create_llm is None:
            raise HTTPException(status_code=501, detail="AI analysis module not installed. Install 'gemmap[ai]'.")
        if not req.pathway or not req.pathway.strip():
            raise HTTPException(status_code=400, detail="Pathway name is required.")
        if not req.paper or not isinstance(req.paper, dict):
            raise HTTPException(status_code=400, detail="Paper payload is required.")

        analyzer = session["analyzer"]
        ai_results = session.get("ai_results") or []

        provider = req.provider
        if normalize_provider is not None:
            provider = normalize_provider(provider)
        if provider == "openai" and get_provider_from_key is not None:
            try:
                detected = get_provider_from_key(req.api_key)
                if detected:
                    provider = detected
            except Exception:
                pass

        cluster_df = analyzer.cluster_results if analyzer.cluster_results is not None else pd.DataFrame()
        if cluster_df.empty:
            raise HTTPException(status_code=400, detail="No module results available.")
        clusters = sorted(cluster_df["Cluster"].unique(), key=lambda x: int(x))
        cluster_labels = {str(c): f"Module {c}" for c in clusters}
        for row in ai_results:
            cid = str(row.get("cluster_id"))
            title = str(row.get("title") or "").strip()
            if cid in cluster_labels and title:
                cluster_labels[cid] = title

        mountain_payload = _build_mountain_data(
            analyzer=analyzer,
            cluster_labels=cluster_labels,
            deg_df=session.get("deg_df"),
            deg_config=session.get("deg_config"),
        )
        pathways = mountain_payload.get("pathways", []) or []
        ranked_genes = mountain_payload.get("ranked_genes", []) or []
        pathway_rec = _match_pathway_record(pathways, req.pathway) or {}

        module_id = str(pathway_rec.get("cluster") or req.module_id or "")
        module_name = str(pathway_rec.get("module") or req.module_name or cluster_labels.get(module_id, f"Module {module_id}" if module_id else "Unknown module"))

        # Build pathway-gene hit table with expression direction and module context.
        rank_map: Dict[str, float] = {}
        for row in ranked_genes:
            g = str(row.get("gene", "")).upper().strip()
            if not g:
                continue
            try:
                rank_map[g] = float(row.get("logfc"))
            except Exception:
                continue
        pathway_genes = [str(g).upper().strip() for g in (pathway_rec.get("genes") or []) if str(g).strip()]
        hit_rows = []
        for gene in pathway_genes:
            if gene in rank_map:
                lfc = float(rank_map[gene])
                hit_rows.append({
                    "gene": gene,
                    "logfc": lfc,
                    "direction": "up" if lfc > 0 else ("down" if lfc < 0 else "neutral"),
                    "module": f"M{module_id}" if module_id else "N/A",
                })
        hit_rows.sort(key=lambda x: abs(float(x.get("logfc", 0.0))), reverse=True)
        top_hits = hit_rows[:25]
        n_up = sum(1 for h in hit_rows if h["direction"] == "up")
        n_down = sum(1 for h in hit_rows if h["direction"] == "down")

        # Paper metadata and best-effort open-access full-text lookup.
        paper = req.paper
        pmid = str(paper.get("pmid", "")).strip()
        paper_title = str(paper.get("title", "")).strip()
        paper_abstract = str(paper.get("abstract", "")).strip()
        paper_doi = str(paper.get("doi", "")).strip()
        paper_url = str(paper.get("url", "")).strip()
        oa_payload = _fetch_open_access_fulltext_snippet(pmid)
        oa_snippet = str(oa_payload.get("snippet") or "").strip()
        oa_note = str(oa_payload.get("note") or "")
        oa_used = bool(oa_payload.get("used"))
        pmcid = str(oa_payload.get("pmcid") or paper.get("pmcid") or "").strip()

        module_ai = next((a for a in ai_results if str(a.get("cluster_id")) == module_id), {})
        module_ai_title = str(module_ai.get("title") or module_name or f"Module {module_id}")
        module_ai_summary = str(module_ai.get("summary") or "")
        module_ai_process = str(module_ai.get("key_process") or "")
        module_conf = module_ai.get("confidence")
        try:
            module_conf_text = f"{float(module_conf):.2f}" if module_conf is not None else "N/A"
        except Exception:
            module_conf_text = "N/A"

        gene_stats_df = analyzer.gene_stats if analyzer.gene_stats is not None else analyzer.analyze_gene_frequencies()
        if (
            not gene_stats_df.empty and
            "Cluster" in gene_stats_df.columns and
            "Percentage" in gene_stats_df.columns and
            "Item" in gene_stats_df.columns
        ):
            core_genes_df = gene_stats_df[
                (gene_stats_df["Cluster"].astype(str) == str(module_id)) &
                (gene_stats_df["Percentage"] >= 25)
            ]
        else:
            core_genes_df = pd.DataFrame()
        core_genes = core_genes_df.sort_values("Percentage", ascending=False).head(20)["Item"].astype(str).tolist() if not core_genes_df.empty else []

        study_context_rows = [
            ("Disease/phenotype", req.study_disease),
            ("Tissue/cell type", req.study_tissue),
            ("Organism", req.study_organism),
            ("Technology", req.study_technology),
            ("Cohort/comparison", req.study_cohort),
            ("Extra notes", req.study_notes),
        ]
        study_context_lines = []
        for label, raw_value in study_context_rows:
            value = str(raw_value or "").strip()
            if value:
                study_context_lines.append(f"- {label}: {value}")
        study_context_block = "\n".join(study_context_lines) if study_context_lines else "No user-provided study context."

        hit_block = "\n".join(
            f"- {h['gene']}: logFC {h['logfc']:+.3f} ({h['direction']}), module {h['module']}"
            for h in top_hits[:20]
        ) or "- No ranked DEG hits found for genes in this pathway."

        fulltext_block = oa_snippet[:5000] if oa_snippet else "Open-access full text was not available."
        abstract_block = paper_abstract or "No abstract available."

        evidence_system = (
            "You are GEMMAP Evidence Agent.\n"
            "Use only provided paper metadata/text and pathway/module context.\n"
            "Never invent evidence. Be concise and literal.\n"
            "If direct relevance is unclear, return verdict='none' and why_found='I could not find relevant evidence.'.\n"
            "Return JSON only with schema:\n"
            "{\"verdict\":\"direct|indirect|none\","
            "\"why_found\":\"concise_complete\","
            "\"study_relation\":\"concise_complete\","
            "\"confidence\":\"high|medium|low\","
            "\"next_check\":\"concise_complete\","
            "\"hypothesis\":\"concise_complete or empty\"}"
        )
        evidence_user = (
            f"Selected pathway: {req.pathway}\n"
            f"Matched module: M{module_id} ({module_ai_title})\n"
            f"Module AI summary: {module_ai_summary or 'N/A'}\n"
            f"Module key process: {module_ai_process or 'N/A'} | Confidence: {module_conf_text}\n\n"
            f"Paper:\nTitle: {paper_title or 'N/A'}\nPMID: {pmid or 'N/A'}\nDOI: {paper_doi or 'N/A'}\nURL: {paper_url or 'N/A'}\nPMCID: {pmcid or 'N/A'}\n"
            f"Open-access note: {oa_note}\n\n"
            f"Study context from user:\n{study_context_block}\n\n"
            f"Abstract:\n{abstract_block}\n\n"
            f"Open-access full text snippet (if available):\n{fulltext_block}\n\n"
            f"Pathway genes in ranked DEG list: {len(hit_rows)} hits (up={n_up}, down={n_down})\n"
            f"Top hit genes:\n{hit_block}\n\n"
            f"Top core genes in module M{module_id}: {', '.join(core_genes[:15]) if core_genes else 'N/A'}\n\n"
            "Keep each field concise but complete; avoid incomplete phrases or trailing ellipses."
        )

        evidence_text = "I could not find relevant evidence."
        hypothesis_text = ""
        try:
            llm = _create_llm(provider, req.api_key, turbo=bool(req.turbo))
            evidence_reply = await llm.ainvoke(f"{evidence_system}\n\n{evidence_user}")
            evidence_raw = getattr(evidence_reply, "content", str(evidence_reply))
            if isinstance(evidence_raw, list):
                evidence_raw = "\n".join(str(x) for x in evidence_raw)

            parsed = _extract_json_object_from_text(str(evidence_raw))
            if parsed:
                verdict = str(parsed.get("verdict") or "none").strip().lower()
                why_found = _compact_text(parsed.get("why_found"), 720)
                study_relation = _compact_text(parsed.get("study_relation"), 720)
                confidence = str(parsed.get("confidence") or "low").strip().lower()
                if confidence not in {"high", "medium", "low"}:
                    confidence = "low"
                next_check = _compact_text(parsed.get("next_check"), 360)
                hypothesis_candidate = _compact_text(parsed.get("hypothesis"), 640)

                if verdict in {"none", "uncertain", "unknown"} or not (why_found or study_relation):
                    evidence_text = "I could not find relevant evidence."
                    hypothesis_text = ""
                else:
                    lines = [
                        f"Verdict: {verdict}",
                        f"Why found: {why_found or 'Not specified.'}",
                        f"Study relation: {study_relation or 'Not specified.'}",
                        f"Confidence: {confidence}",
                    ]
                    if next_check and next_check.lower() not in {"none", "n/a", "na"}:
                        lines.append(f"Next check: {next_check}")
                    evidence_text = "\n".join(lines)
                    if hypothesis_candidate and verdict != "none":
                        hypothesis_text = f"Hypothesis: {hypothesis_candidate}"
            else:
                fallback = _compact_text(evidence_raw, 1200)
                if fallback:
                    evidence_text = fallback
                if "could not find relevant evidence" in evidence_text.lower():
                    evidence_text = "I could not find relevant evidence."
        except Exception as e:
            msg = str(e)
            if "api_key" in msg.lower() or "key" in msg.lower() or "auth" in msg.lower():
                msg = "Authentication failed. Please check your API key."
            raise HTTPException(status_code=500, detail=f"Paper analysis error: {msg}")

        chat_log = session.setdefault("chat_log", [])
        assistant_reply_payload = f"Evidence Agent:\n{evidence_text}"
        if hypothesis_text:
            assistant_reply_payload = f"{assistant_reply_payload}\n\nHypothesis Agent:\n{hypothesis_text}"
        chat_log.append({
            "timestamp_utc": _now_utc_iso(),
            "provider": provider,
            "selected_module": f"M{module_id}" if module_id else "",
            "user_message": f"Analyze paper relevance for pathway '{req.pathway}' (PMID {pmid or 'N/A'}).",
            "assistant_reply": assistant_reply_payload,
            "literature": [paper],
            "query": req.pathway,
        })
        if len(chat_log) > 500:
            session["chat_log"] = chat_log[-500:]

        return {
            "status": "success",
            "provider": provider,
            "turbo": bool(req.turbo),
            "pathway": req.pathway,
            "module_id": module_id,
            "module_name": module_name,
            "paper": {
                "pmid": pmid,
                "title": paper_title,
                "doi": paper_doi,
                "url": paper_url,
                "pmcid": pmcid,
            },
            "open_access_used": oa_used,
            "open_access_note": oa_note,
            "gene_hits": top_hits,
            "n_gene_hits": len(hit_rows),
            "n_up": n_up,
            "n_down": n_down,
            "evidence_agent": str(evidence_text),
            "hypothesis_agent": str(hypothesis_text),
        }

    @app.get("/api/reproducibility")
    async def export_reproducibility_manifest(
        x_session_id: Optional[str] = Header(None),
        include_chat_history: bool = False
    ):
        """
        Export official reproducibility supplement as JSON.
        """
        if not x_session_id:
            raise HTTPException(status_code=401, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session or not session.get("analyzer"):
            raise HTTPException(status_code=400, detail="Run analysis first")

        analyzer = session["analyzer"]
        ai_results = session.get("ai_results") or []
        manifest = _build_reproducibility_manifest(
            x_session_id,
            session,
            analyzer,
            ai_results,
            include_chat_history=include_chat_history
        )
        report_basename = _session_report_basename(session)
        filename = f"{report_basename}_reproducibility.json"
        content = json.dumps(json_clean(manifest), ensure_ascii=False, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    @app.post("/api/reproducibility/replay")
    async def replay_from_repro_manifest(
        req: ReproReplayRequest,
        x_session_id: Optional[str] = Header(None)
    ):
        """
        Replay an analysis from reproducibility JSON on the active session dataset.
        """
        manifest = req.manifest or {}
        replay_params = (manifest.get("replay", {}) or {}).get("parameters", {}) or {}
        input_spec = manifest.get("input", {}) if isinstance(manifest.get("input"), dict) else {}
        outputs_spec = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), dict) else {}
        manifest_has_annotations_key = isinstance(outputs_spec, dict) and "ai_annotations" in outputs_spec
        manifest_annotations = outputs_spec.get("ai_annotations", []) if isinstance(outputs_spec.get("ai_annotations"), list) else []
        manifest_has_chat_key = isinstance(outputs_spec, dict) and "chat_log" in outputs_spec
        manifest_chat_log = outputs_spec.get("chat_log", []) if isinstance(outputs_spec.get("chat_log"), list) else []
        source_filename = str(input_spec.get("source_filename") or "replay_manifest.csv")

        session_id = x_session_id
        session = session_manager.get_session(session_id) if session_id else None

        if session is None:
            pathway_table = input_spec.get("pathway_table")
            if not isinstance(pathway_table, list) or len(pathway_table) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No active session and manifest has no embedded pathway_table for replay."
                )
            try:
                replay_df = pd.DataFrame(pathway_table)
                if replay_df.empty:
                    raise ValueError("Embedded pathway_table is empty.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid embedded pathway_table: {str(e)}")

            session_id = session_manager.create_session(replay_df, source_filename)
            report_name = manifest.get("report_name")
            if isinstance(report_name, str) and report_name.strip():
                session_manager.update_report_name(session_id, report_name.strip())
            session = session_manager.get_session(session_id)
            if session is None:
                raise HTTPException(status_code=500, detail="Failed to initialize replay session.")

        if manifest_has_chat_key:
            session["chat_log"] = _scrub_sensitive_fields(manifest_chat_log)[-500:]

        pathway_col = replay_params.get("pathway_col") or input_spec.get("pathway_col") or "pathway"
        genes_col = replay_params.get("genes_col") or input_spec.get("genes_col") or "leadingEdge"
        requested_score_col = replay_params.get("score_col") or input_spec.get("score_col")
        try:
            n_clusters = int(replay_params.get("n_clusters") or 5)
        except Exception:
            n_clusters = 5
        nes_direction = str(replay_params.get("nes_direction") or input_spec.get("nes_direction") or "all")

        df = session.get("df")
        if df is None or df.empty:
            pathway_table = input_spec.get("pathway_table")
            if isinstance(pathway_table, list) and len(pathway_table) > 0:
                try:
                    df = pd.DataFrame(pathway_table)
                    session["df"] = df
                except Exception:
                    pass
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No pathway data available for replay.")

        effective_score_col = _resolve_score_column(df, requested_score_col)
        if not effective_score_col:
            raise HTTPException(status_code=400, detail="NES/score column not found for replay.")

        filtered_df, nes_filter_summary = _apply_nes_direction_filter(
            df=df,
            score_col=effective_score_col,
            nes_direction=nes_direction
        )
        if len(filtered_df) < 3:
            raise HTTPException(status_code=400, detail=f"Only {len(filtered_df)} pathways after replay filters.")
        if n_clusters > len(filtered_df):
            n_clusters = len(filtered_df)
        if n_clusters < 2:
            n_clusters = 2

        analyzer = PathwayAnalyzer(
            filtered_df,
            pathway_col=pathway_col,
            genes_col=genes_col,
            score_col=effective_score_col
        )
        analyzer._nes_direction = nes_filter_summary["nes_direction"]
        analyzer.run_clustering(n_clusters=n_clusters)

        # Re-apply DEG overlay if session DEG is available, otherwise use embedded manifest DEG table.
        deg_df = session.get("deg_df")
        deg_config = session.get("deg_config")
        if (deg_df is None or deg_config is None) and isinstance(input_spec, dict):
            manifest_deg_table = input_spec.get("deg_table")
            manifest_deg_config = input_spec.get("deg_config")
            if isinstance(manifest_deg_table, list) and len(manifest_deg_table) > 0 and isinstance(manifest_deg_config, dict):
                try:
                    deg_df = pd.DataFrame(manifest_deg_table)
                    deg_config = manifest_deg_config
                    session_manager.update_deg_data(session_id, deg_df, deg_config)
                except Exception:
                    deg_df = None
                    deg_config = None
        if deg_df is not None and deg_config is not None:
            gene_stats_df = analyzer.analyze_gene_frequencies(
                deg_object=deg_df,
                deg_gene_col=deg_config["gene_col"],
                deg_padj_col=deg_config["padj_col"],
                deg_lfc_col=deg_config["lfc_col"],
                padj_threshold=deg_config["padj_threshold"],
                lfc_threshold=deg_config["lfc_threshold"],
            )
        else:
            gene_stats_df = analyzer.analyze_gene_frequencies()

        replay_annotations: List[Dict[str, Any]] = []
        can_rerun_annotations = bool(req.rerun_annotations and req.api_key and build_annotation_graph is not None)
        if manifest_has_annotations_key and not can_rerun_annotations:
            replay_annotations = _scrub_sensitive_fields(manifest_annotations)
            if replay_annotations:
                analyzer.add_ai_annotations(replay_annotations)
                methodology = manifest.get("methodology", {}) if isinstance(manifest.get("methodology"), dict) else {}
                replay_provider = str(methodology.get("ai_provider") or replay_params.get("ai_provider") or "manifest")
                replay_model = str(methodology.get("ai_model") or replay_params.get("ai_model") or "reproducibility-manifest")
                replay_mode = str(methodology.get("mode") or replay_params.get("ai_mode") or "Replayed from reproducibility JSON")
                analyzer.add_ai_metadata(provider=replay_provider, model=replay_model, mode=replay_mode)
            session_manager.update_ai_results(session_id, replay_annotations)

        if req.rerun_annotations and req.api_key and build_annotation_graph is not None:
            provider = req.provider or "openai"
            if normalize_provider is not None:
                provider = normalize_provider(provider)
            if provider == "openai" and get_provider_from_key is not None:
                try:
                    detected = get_provider_from_key(req.api_key)
                    if detected:
                        provider = detected
                except Exception:
                    pass

            clusters_data = []
            cluster_ids = analyzer.cluster_results["Cluster"].unique()
            for cid in cluster_ids:
                c_paths = analyzer.cluster_results[analyzer.cluster_results["Cluster"] == cid]
                top_paths = c_paths.sort_values(analyzer.score_col, ascending=False).head(6)[analyzer.pathway_col].tolist()
                c_genes = gene_stats_df[gene_stats_df["Cluster"] == cid]
                top_genes = c_genes.sort_values("Percentage", ascending=False)["Item"].head(12).tolist() if "Percentage" in c_genes.columns else c_genes["Item"].head(12).tolist()
                clusters_data.append({
                    "cluster_id": str(cid),
                    "top_pathways": [str(p) for p in top_paths],
                    "top_genes": [str(g) for g in top_genes],
                })
            graph = build_annotation_graph()
            ai_state = await graph.ainvoke({
                "clusters_data": clusters_data,
                "api_key": req.api_key,
                "model_provider": provider,
                "turbo": bool(req.turbo),
            })
            replay_annotations = ai_state.get("annotations", []) if isinstance(ai_state, dict) else []
            if replay_annotations:
                analyzer.add_ai_annotations(replay_annotations)
                replay_mode = "Replayed from reproducibility JSON (Turbo)" if bool(req.turbo) else "Replayed from reproducibility JSON"
                analyzer.add_ai_metadata(provider=provider, model="replay-manifest", mode=replay_mode)
                session_manager.update_ai_results(session_id, replay_annotations)

        session_manager.update_analyzer(session_id, analyzer)
        cluster_stats = compute_cluster_statistics(analyzer)

        try:
            mds_gof = list(analyzer.get_mds_gof())
        except Exception:
            mds_gof = [0.0, 0.0]

        plotly_dict = analyzer.plot.scatter_3d(as_dict=True)
        scatter_data = analyzer.cluster_results.to_dict(orient="records")
        gene_stats = gene_stats_df.to_dict(orient="records")
        annotations = replay_annotations or session.get("ai_results") or []
        chat_log = session.get("chat_log") if isinstance(session.get("chat_log"), list) else []
        chat_log = _scrub_sensitive_fields(chat_log)

        _log_event(f"Replayed analysis from reproducibility manifest for session={session_id}")
        return json_clean({
            "status": "success",
            "session_id": session_id,
            "replayed_from_manifest": True,
            "scatter_data": scatter_data,
            "plotly_json": plotly_dict,
            "gene_stats": gene_stats,
            "clusters": sorted(list(analyzer.cluster_results['Cluster'].unique())),
            "cluster_stats": cluster_stats,
            "mds_gof": mds_gof,
            "total_pathways": len(analyzer.cluster_results),
            "total_genes": len(set(g for genes in analyzer.raw_data[analyzer.genes_col] for g in genes if genes)),
            "has_deg": deg_df is not None,
            "n_deg_genes": int(gene_stats_df['DEG'].sum()) if isinstance(gene_stats_df, pd.DataFrame) and 'DEG' in gene_stats_df.columns else 0,
            "nes_filter": nes_filter_summary,
            "ai_annotations": annotations,
            "chat_log": chat_log,
        })

    @app.get("/api/export-debug-log")
    async def export_debug_log(
        x_session_id: Optional[str] = Header(None),
        issue_summary: Optional[str] = Query(None, max_length=2000),
        attempted_action: Optional[str] = Query(None, max_length=2000),
    ):
        """
        Export full debug report (session snapshot + runtime logs + terminal logs).
        """
        session = session_manager.get_session(x_session_id) if x_session_id else None
        report = _build_debug_log_export(
            x_session_id,
            session,
            issue_summary=issue_summary,
            attempted_action=attempted_action,
        )
        return StreamingResponse(
            io.BytesIO(report["payload"].encode("utf-8")),
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="{report["filename"]}"',
                "X-Report-Issue-ID": report["issue_id"],
                "X-Report-Issue-Subject": report["issue_subject"],
                "X-Report-Issue-Email": report["issue_email"],
            },
        )

    @app.get("/api/download")
    async def download_report(x_session_id: Optional[str] = Header(None)):
        if not x_session_id: raise HTTPException(status_code=400, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session or not session.get("analyzer"): raise HTTPException(status_code=404)
        
        analyzer = session["analyzer"]
        ai_results = session.get("ai_results")  # List of dicts
        deg_df = session.get("deg_df")
        deg_config = session.get("deg_config")
        auto_k = session.get("auto_k")
        
        # Re-analyze gene frequencies with DEG if available
        if deg_df is not None and deg_config is not None:
            gene_stats = analyzer.analyze_gene_frequencies(
                deg_object=deg_df,
                deg_gene_col=deg_config["gene_col"],
                deg_padj_col=deg_config["padj_col"],
                deg_lfc_col=deg_config["lfc_col"],
                padj_threshold=deg_config["padj_threshold"],
                lfc_threshold=deg_config["lfc_threshold"]
            )
        else:
            gene_stats = analyzer.gene_stats if analyzer.gene_stats is not None else analyzer.analyze_gene_frequencies()
        
        output = io.BytesIO()
        
        # If AI results exist, add SEPARATE columns for each field
        if ai_results:
            ai_title_map = {str(a['cluster_id']): a.get('title', '') for a in ai_results}
            ai_summary_map = {str(a['cluster_id']): a.get('summary', '') for a in ai_results}
            ai_process_map = {str(a['cluster_id']): a.get('key_process', '') for a in ai_results}
            ai_confidence_map = {str(a['cluster_id']): a.get('confidence', 0) for a in ai_results}
            
            analyzer.cluster_results['AI_Title'] = analyzer.cluster_results['Cluster'].astype(str).map(ai_title_map)
            analyzer.cluster_results['AI_Summary'] = analyzer.cluster_results['Cluster'].astype(str).map(ai_summary_map)
            analyzer.cluster_results['AI_Key_Process'] = analyzer.cluster_results['Cluster'].astype(str).map(ai_process_map)
            analyzer.cluster_results['AI_Confidence'] = analyzer.cluster_results['Cluster'].astype(str).map(ai_confidence_map)
        
        # Add Core Genes column (genes with >= 25% frequency in cluster)
        def get_core_genes(cluster_id: int) -> str:
            cluster_genes = gene_stats[gene_stats['Cluster'] == cluster_id]
            core = cluster_genes[cluster_genes['Percentage'] >= 25].sort_values('Percentage', ascending=False)
            return ', '.join(core['Item'].head(20).tolist())  # Top 20 core genes
        
        # Add DEG Core Genes column if DEG data available
        def get_deg_core_genes(cluster_id: int) -> str:
            if 'DEG' not in gene_stats.columns:
                return ''
            cluster_genes = gene_stats[gene_stats['Cluster'] == cluster_id]
            core_degs = cluster_genes[(cluster_genes['Percentage'] >= 25) & (cluster_genes['DEG'] == True)]
            core_degs = core_degs.sort_values('Percentage', ascending=False)
            return ', '.join(core_degs['Item'].head(20).tolist())
        
        analyzer.cluster_results['Core_Genes'] = analyzer.cluster_results['Cluster'].apply(get_core_genes)
        
        if deg_df is not None:
            analyzer.cluster_results['DEG_Core_Genes'] = analyzer.cluster_results['Cluster'].apply(get_deg_core_genes)
        
        # Add auto-k info as metadata if available
        if auto_k is not None:
            analyzer.cluster_results['Auto_K_Used'] = auto_k
        
        analyzer.save_excel(output)
        output.seek(0)
        filename = f"gemmap_{session['filename']}.xlsx"
        
        return StreamingResponse(
            output, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    def _generate_publication_figures(
        analyzer,
        ai_results,
        gene_stats,
        img_format: str,
        dpi: int,
        include_annotation_variants: bool = True,
    ):
        """
        Generate publication-ready figures using matplotlib and seaborn.
        
        Creates high-quality, aesthetically pleasing visualizations suitable
        for scientific publications with clean, modern styling.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from io import BytesIO
        import matplotlib.colors as mcolors
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        
        # Set publication-ready seaborn theme
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
        
        # Publication-quality matplotlib settings
        plt.rcParams.update({
            # Figure aesthetics
            'figure.facecolor': 'white',
            'figure.edgecolor': 'none',
            'figure.dpi': 100,
            
            # Axes styling
            'axes.facecolor': 'white',
            'axes.edgecolor': '#2d3436',
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'axes.titlepad': 12,
            'axes.spines.top': False,
            'axes.spines.right': False,
            
            # Grid styling
            'grid.color': '#dfe6e9',
            'grid.linewidth': 0.6,
            'grid.linestyle': '-',
            'grid.alpha': 0.7,
            
            # Tick styling
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.color': '#2d3436',
            'ytick.color': '#2d3436',
            
            # Legend styling
            'legend.fontsize': 9,
            'legend.frameon': True,
            'legend.facecolor': 'white',
            'legend.edgecolor': '#dfe6e9',
            'legend.framealpha': 0.95,
            'legend.borderpad': 0.5,
            
            # Font styling
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'],
            'font.size': 10,
            
            # Savefig settings
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
        })
        
        figures = {}
        
        # Prepare cluster labels
        cluster_labels = {}
        if ai_results:
            for a in ai_results:
                cluster_labels[str(a['cluster_id'])] = a.get('title', f"Module {a['cluster_id']}")
        
        # Publication-quality color palette (Nature/Science inspired)
        # Colorblind-friendly, high contrast, professional look
        colors = [
            '#3498db',  # Blue
            '#e74c3c',  # Red
            '#2ecc71',  # Green
            '#9b59b6',  # Purple
            '#f39c12',  # Orange
            '#1abc9c',  # Teal
            '#e91e63',  # Pink
            '#00bcd4',  # Cyan
            '#ff9800',  # Amber
            '#795548',  # Brown
        ]
        def _color_for_cluster(cluster_id):
            return colors[int(cluster_id) % len(colors)]
        
        cluster_df = analyzer.cluster_results
        clusters = sorted(cluster_df['Cluster'].unique())

        plain_cluster_labels = {str(c): f"Module {c}" for c in clusters}
        annotated_cluster_labels = plain_cluster_labels.copy()
        annotated_cluster_labels.update(cluster_labels)
        
        # Map format names
        fmt_map = {'jpeg': 'jpg', 'tiff': 'tif'}
        save_fmt = fmt_map.get(img_format, img_format)
        
        try:
            def _render_fig1(label_map, key):
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('white')
                for c in clusters:
                    mask = cluster_df['Cluster'] == c
                    label = label_map.get(str(c), f"Module {c}")
                    ax.scatter(
                        cluster_df.loc[mask, 'Dim1'],
                        cluster_df.loc[mask, 'Dim2'],
                        cluster_df.loc[mask, 'Dim3'],
                        c=_color_for_cluster(c),
                        label=label, s=70, alpha=0.85,
                        edgecolors='white', linewidth=0.6,
                        depthshade=True
                    )
                ax.set_xlabel('MDS Dimension 1', fontsize=11, labelpad=12)
                ax.set_ylabel('MDS Dimension 2', fontsize=11, labelpad=12)
                ax.set_zlabel('MDS Dimension 3', fontsize=11, labelpad=12)
                ax.set_title('Pathway Module Map (3D MDS)', fontsize=14, fontweight='bold', pad=20)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('#dfe6e9')
                ax.yaxis.pane.set_edgecolor('#dfe6e9')
                ax.zaxis.pane.set_edgecolor('#dfe6e9')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True, fancybox=False, edgecolor='#dfe6e9')
                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
                buf.seek(0)
                figures[key] = buf.getvalue()
                plt.close(fig)

            # Default Fig1 uses annotated labels when AI is available.
            _render_fig1(annotated_cluster_labels if ai_results else plain_cluster_labels, 'Fig1_3D_MDS_Module_Map')
            if ai_results and include_annotation_variants:
                _render_fig1(plain_cluster_labels, 'Fig1_3D_MDS_Module_Map_No_Annotations')
        except Exception as e:
            print(f"Error generating 3D figure: {e}")
        
        try:
            def _render_fig2(label_map, key):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                dim_pairs = [('Dim1', 'Dim2'), ('Dim1', 'Dim3'), ('Dim2', 'Dim3')]
                for ax, (dx, dy) in zip(axes, dim_pairs):
                    for c in clusters:
                        mask = cluster_df['Cluster'] == c
                        label = label_map.get(str(c), f"Module {c}")
                        ax.scatter(
                            cluster_df.loc[mask, dx],
                            cluster_df.loc[mask, dy],
                            c=_color_for_cluster(c),
                            label=label, s=55, alpha=0.8,
                            edgecolors='white', linewidth=0.5
                        )
                    ax.set_xlabel(dx.replace('Dim', 'MDS Dimension '), fontsize=11)
                    ax.set_ylabel(dy.replace('Dim', 'MDS Dimension '), fontsize=11)
                    ax.set_title(f'{dx} vs {dy}', fontsize=12, fontweight='bold')
                    for spine in ['top', 'right']:
                        ax.spines[spine].set_visible(False)
                    for spine in ['bottom', 'left']:
                        ax.spines[spine].set_color('#2d3436')
                        ax.spines[spine].set_linewidth(0.8)
                axes[2].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True, fancybox=False, edgecolor='#dfe6e9')
                fig.suptitle('2D MDS Projections', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
                buf.seek(0)
                figures[key] = buf.getvalue()
                plt.close(fig)

            # Default Fig2 uses annotated labels when AI is available.
            _render_fig2(annotated_cluster_labels if ai_results else plain_cluster_labels, 'Fig2_2D_MDS_Projections')
            if ai_results and include_annotation_variants:
                _render_fig2(plain_cluster_labels, 'Fig2_2D_MDS_Projections_No_Annotations')
        except Exception as e:
            print(f"Error generating 2D figure: {e}")
        
        try:
            # Figure 3: Module Composition with seaborn bar plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Pathways per module
            pathway_counts = cluster_df['Cluster'].value_counts().sort_index()
            cluster_names = [cluster_labels.get(str(c), f"Module {c}") for c in pathway_counts.index]
            cluster_colors = [_color_for_cluster(c) for c in pathway_counts.index]
            
            bars1 = axes[0].bar(cluster_names, pathway_counts.values, color=cluster_colors,
                               edgecolor='white', linewidth=1.5, width=0.7)
            axes[0].set_ylabel('Number of Pathways', fontsize=11, fontweight='medium')
            axes[0].set_xlabel('Module', fontsize=11, fontweight='medium')
            axes[0].set_title('Pathways per Module', fontsize=12, fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars1, pathway_counts.values):
                height = bar.get_height()
                axes[0].annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 4), textcoords='offset points',
                               ha='center', va='bottom', fontsize=10, fontweight='medium')
            
            # Clean spines
            for spine in ['top', 'right']:
                axes[0].spines[spine].set_visible(False)
            
            # Core genes per module
            core_gene_counts = []
            for c in clusters:
                n_core = len(gene_stats[(gene_stats['Cluster'] == c) & (gene_stats['Percentage'] >= 25)])
                core_gene_counts.append(n_core)
            
            cluster_names_genes = [cluster_labels.get(str(c), f"Module {c}") for c in clusters]
            cluster_colors_genes = [_color_for_cluster(c) for c in clusters]
            
            bars2 = axes[1].bar(cluster_names_genes, core_gene_counts, color=cluster_colors_genes,
                               edgecolor='white', linewidth=1.5, width=0.7)
            axes[1].set_ylabel('Number of Core Genes (≥25% freq)', fontsize=11, fontweight='medium')
            axes[1].set_xlabel('Module', fontsize=11, fontweight='medium')
            axes[1].set_title('Core Genes per Module', fontsize=12, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars2, core_gene_counts):
                height = bar.get_height()
                axes[1].annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 4), textcoords='offset points',
                               ha='center', va='bottom', fontsize=10, fontweight='medium')
            
            for spine in ['top', 'right']:
                axes[1].spines[spine].set_visible(False)
            
            fig.suptitle('Module Composition Summary', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            figures['Fig3_Module_Composition'] = buf.getvalue()
            plt.close(fig)
        except Exception as e:
            print(f"Error generating composition figure: {e}")
        
        try:
            # Figure 4: Clustered Heatmap using seaborn
            n_top = 10
            top_genes_per_cluster = {}
            all_top_genes = set()
            for c in clusters:
                c_genes = gene_stats[gene_stats['Cluster'] == c].nlargest(n_top, 'Percentage')
                top_genes_per_cluster[c] = c_genes['Item'].tolist()
                all_top_genes.update(top_genes_per_cluster[c])
            all_top_genes = sorted(all_top_genes)
            
            if len(all_top_genes) > 0 and len(clusters) > 0:
                matrix = np.zeros((len(all_top_genes), len(clusters)))
                for j, c in enumerate(clusters):
                    c_data = gene_stats[gene_stats['Cluster'] == c].set_index('Item')
                    for i, gene in enumerate(all_top_genes):
                        if gene in c_data.index:
                            matrix[i, j] = c_data.loc[gene, 'Percentage']
                
                # Create DataFrame for seaborn
                heatmap_df = pd.DataFrame(
                    matrix,
                    index=all_top_genes,
                    columns=[cluster_labels.get(str(c), f"Module {c}") for c in clusters]
                )
                
                # Create custom colormap (white to deep blue - publication friendly)
                cmap = sns.color_palette("Blues", as_cmap=True)
                
                # Use seaborn clustermap for hierarchical clustering
                g = sns.clustermap(
                    heatmap_df,
                    cmap=cmap,
                    figsize=(max(10, len(clusters)*1.2 + 3), max(12, len(all_top_genes)*0.35 + 2)),
                    linewidths=0.5,
                    linecolor='white',
                    dendrogram_ratio=(0.12, 0.12),
                    cbar_pos=(0.02, 0.8, 0.03, 0.15),
                    tree_kws={'linewidths': 1.0, 'colors': '#2d3436'},
                    xticklabels=True,
                    yticklabels=True,
                    vmin=0,
                    annot=False,
                    method='average'
                )
                
                # Style adjustments
                g.ax_heatmap.set_xlabel('Module', fontsize=11, fontweight='medium')
                g.ax_heatmap.set_ylabel('Gene', fontsize=11, fontweight='medium')
                g.ax_heatmap.tick_params(axis='x', rotation=45, labelsize=9)
                g.ax_heatmap.tick_params(axis='y', labelsize=8)
                
                # Colorbar label
                g.ax_cbar.set_ylabel('Frequency (%)', fontsize=10, fontweight='medium')
                g.ax_cbar.tick_params(labelsize=9)
                
                g.fig.suptitle('Gene Frequency Heatmap', fontsize=14, fontweight='bold', y=1.01)
                
                buf = BytesIO()
                g.fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight', 
                             facecolor='white', edgecolor='none')
                buf.seek(0)
                figures['Fig4_Gene_Frequency_Heatmap'] = buf.getvalue()
                plt.close(g.fig)
        except Exception as e:
            print(f"Error generating heatmap: {e}")
        
        try:
            # Figure 5: Elbow and Silhouette plots
            k_range = range(2, 11)
            elbow_result = analyzer.suggest_optimal_k(k_range=k_range)
            metrics = elbow_result['metrics']
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Elbow plot
            axes[0].plot(metrics['k'], metrics['inertia'], 'o-', 
                        color='#3498db', linewidth=2.5, markersize=10, 
                        markerfacecolor='white', markeredgewidth=2.5,
                        markeredgecolor='#3498db')
            axes[0].axvline(x=elbow_result['elbow_k'], color='#e74c3c', linestyle='--', 
                           linewidth=2, label=f"Optimal k = {elbow_result['elbow_k']}", alpha=0.9)
            axes[0].set_xlabel('Number of Modules (k)', fontsize=11, fontweight='medium')
            axes[0].set_ylabel('Within-Module Sum of Squares', fontsize=11, fontweight='medium')
            axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
            axes[0].legend(loc='upper right', fontsize=10, frameon=True, 
                          fancybox=False, edgecolor='#dfe6e9')
            for spine in ['top', 'right']:
                axes[0].spines[spine].set_visible(False)
            
            # Silhouette plot
            axes[1].plot(metrics['k'], metrics['silhouette'], 'o-',
                        color='#2ecc71', linewidth=2.5, markersize=10,
                        markerfacecolor='white', markeredgewidth=2.5,
                        markeredgecolor='#2ecc71')
            axes[1].axvline(x=elbow_result['silhouette_k'], color='#e74c3c', linestyle='--',
                           linewidth=2, label=f"Optimal k = {elbow_result['silhouette_k']}", alpha=0.9)
            axes[1].set_xlabel('Number of Modules (k)', fontsize=11, fontweight='medium')
            axes[1].set_ylabel('Silhouette Score', fontsize=11, fontweight='medium')
            axes[1].set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
            axes[1].legend(loc='upper right', fontsize=10, frameon=True,
                          fancybox=False, edgecolor='#dfe6e9')
            for spine in ['top', 'right']:
                axes[1].spines[spine].set_visible(False)
            
            fig.suptitle('Optimal Module Selection', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            figures['Fig5_Elbow_Silhouette'] = buf.getvalue()
            plt.close(fig)
        except Exception as e:
            print(f"Error generating elbow figure: {e}")
        
        try:
            # Figure 6: NES Distribution with seaborn boxplot
            if analyzer.score_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Prepare data for seaborn
                plot_data = []
                for c in clusters:
                    c_data = cluster_df[cluster_df['Cluster'] == c][analyzer.score_col].dropna()
                    for val in c_data.values:
                        plot_data.append({
                            'Cluster': cluster_labels.get(str(c), f"Module {c}"),
                            'NES': val,
                            'ClusterNum': c
                        })
                plot_df = pd.DataFrame(plot_data)
                
                # Create palette dict for seaborn
                palette = {cluster_labels.get(str(c), f"Module {c}"): _color_for_cluster(c)
                          for c in clusters}
                
                # Seaborn boxplot with strip overlay
                sns.boxplot(data=plot_df, x='Cluster', y='NES', palette=palette, 
                           width=0.6, linewidth=1.5, fliersize=0, ax=ax)
                sns.stripplot(data=plot_df, x='Cluster', y='NES', color='#2d3436',
                             alpha=0.3, size=4, jitter=0.2, ax=ax)
                
                # Reference line at zero
                ax.axhline(y=0, color='#636e72', linestyle='--', linewidth=1.5, alpha=0.7)
                
                ax.set_xlabel('Module', fontsize=11, fontweight='medium')
                ax.set_ylabel('Normalized Enrichment Score (NES)', fontsize=11, fontweight='medium')
                ax.set_title('NES Distribution by Module', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                
                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                figures['Fig6_NES_Distribution'] = buf.getvalue()
                plt.close(fig)
        except Exception as e:
            print(f"Error generating NES figure: {e}")
        
        try:
            # Figure 7: GSEA Bubble Plot (matplotlib)
            if analyzer.score_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bubble_data = []
                for c in clusters:
                    c_paths = cluster_df[cluster_df['Cluster'] == c]
                    mean_nes = c_paths[analyzer.score_col].mean()
                    n_paths = len(c_paths)
                    # Use -log10(pvalue) if available, else use |NES| as proxy
                    significance = abs(mean_nes) * 2  # proxy for significance
                    bubble_data.append({
                        'Cluster': cluster_labels.get(str(c), f"Module {c}"),
                        'Mean_NES': mean_nes,
                        'N_Pathways': n_paths,
                        'Significance': significance,
                        'Color': colors[int(c) % len(colors)]
                    })
                bubble_df = pd.DataFrame(bubble_data)

                counts = bubble_df['N_Pathways'].astype(float).values
                min_count = float(np.min(counts)) if len(counts) else 0.0
                max_count = float(np.max(counts)) if len(counts) else 0.0

                def _bubble_size_area(count_value: float) -> float:
                    # Matplotlib scatter size uses marker area (pt^2), so we normalize
                    # counts into a bounded area range to keep visual comparisons readable.
                    if max_count <= min_count:
                        return 360.0
                    min_area, max_area = 140.0, 920.0
                    return min_area + ((count_value - min_count) / (max_count - min_count)) * (max_area - min_area)

                bubble_sizes = np.array([_bubble_size_area(float(v)) for v in counts], dtype=float)
                
                ax.scatter(
                    bubble_df['Mean_NES'], 
                    range(len(bubble_df)),
                    s=bubble_sizes,
                    c=bubble_df['Color'],
                    alpha=0.7,
                    edgecolors='white',
                    linewidth=1.5
                )
                
                ax.axvline(x=0, color='#636e72', linestyle='--', linewidth=1, alpha=0.7)
                ax.set_yticks(range(len(bubble_df)))
                ax.set_yticklabels(bubble_df['Cluster'])
                ax.set_xlabel('Mean NES', fontsize=11, fontweight='medium')
                ax.set_ylabel('Module', fontsize=11, fontweight='medium')
                ax.set_title('GSEA Bubble Plot', fontsize=14, fontweight='bold')
                
                # Add size legend using actual pathway-count scale from current data.
                if len(counts):
                    rep_counts = sorted({
                        int(round(min_count)),
                        int(round(np.median(counts))),
                        int(round(max_count))
                    })
                    for count_val in rep_counts:
                        if count_val <= 0:
                            continue
                        ax.scatter(
                            [],
                            [],
                            s=_bubble_size_area(float(count_val)),
                            c='gray',
                            alpha=0.5,
                            label=f'{count_val} pathways',
                            edgecolors='white'
                        )
                ax.legend(loc='lower right', title='Pathway Count', frameon=True, 
                         fancybox=False, edgecolor='#dfe6e9')
                
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                
                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                figures['Fig7_GSEA_Bubble_Plot'] = buf.getvalue()
                plt.close(fig)
        except Exception as e:
            print(f"Error generating bubble plot: {e}")
        
        try:
            # Figure 8: Bar Chart of Top Terms (seaborn)
            if analyzer.score_col:
                fig, axes = plt.subplots(1, 2, figsize=(14, 8))
                
                # Split into positive and negative NES
                pos_paths = cluster_df[cluster_df[analyzer.score_col] > 0].nlargest(15, analyzer.score_col)
                neg_paths = cluster_df[cluster_df[analyzer.score_col] < 0].nsmallest(15, analyzer.score_col)
                
                # Clean pathway names
                def clean_name(name):
                    import re
                    name = re.sub(r'^(GO|KEGG|REACTOME|HALLMARK)[A-Z]*_', '', str(name))
                    return name.replace('_', ' ')[:40]
                
                # Positive NES (Activated)
                if len(pos_paths) > 0:
                    pos_paths = pos_paths.copy()
                    pos_paths['Clean_Name'] = pos_paths[analyzer.pathway_col].apply(clean_name)
                    pos_paths['Color'] = pos_paths['Cluster'].apply(lambda c: colors[int(c) % len(colors)])
                    
                    axes[0].barh(range(len(pos_paths)), pos_paths[analyzer.score_col].values,
                                color=pos_paths['Color'].values, edgecolor='white', linewidth=0.5)
                    axes[0].set_yticks(range(len(pos_paths)))
                    axes[0].set_yticklabels(pos_paths['Clean_Name'].values, fontsize=9)
                    axes[0].set_xlabel('NES', fontsize=11, fontweight='medium')
                    axes[0].set_title('Activated Pathways', fontsize=12, fontweight='bold', color='#2ecc71')
                    axes[0].invert_yaxis()
                    for spine in ['top', 'right']:
                        axes[0].spines[spine].set_visible(False)
                
                # Negative NES (Suppressed)
                if len(neg_paths) > 0:
                    neg_paths = neg_paths.copy()
                    neg_paths['Clean_Name'] = neg_paths[analyzer.pathway_col].apply(clean_name)
                    neg_paths['Color'] = neg_paths['Cluster'].apply(lambda c: colors[int(c) % len(colors)])
                    
                    axes[1].barh(range(len(neg_paths)), neg_paths[analyzer.score_col].values,
                                color=neg_paths['Color'].values, edgecolor='white', linewidth=0.5)
                    axes[1].set_yticks(range(len(neg_paths)))
                    axes[1].set_yticklabels(neg_paths['Clean_Name'].values, fontsize=9)
                    axes[1].set_xlabel('NES', fontsize=11, fontweight='medium')
                    axes[1].set_title('Suppressed Pathways', fontsize=12, fontweight='bold', color='#e74c3c')
                    axes[1].invert_yaxis()
                    for spine in ['top', 'right']:
                        axes[1].spines[spine].set_visible(False)
                
                fig.suptitle('Top Enriched Terms by NES', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                figures['Fig8_Top_Enriched_Terms'] = buf.getvalue()
                plt.close(fig)
        except Exception as e:
            print(f"Error generating bar chart: {e}")
        
        try:
            # Figure 9: Manhattan Plot (matplotlib)
            if analyzer.score_col:
                fig, ax = plt.subplots(figsize=(14, 5))
                
                plot_df = cluster_df.copy()
                plot_df = plot_df.reset_index(drop=True)
                plot_df['x_pos'] = plot_df.index + 1
                
                for c in clusters:
                    c_data = plot_df[plot_df['Cluster'] == c]
                    color = _color_for_cluster(c)
                    ax.scatter(c_data['x_pos'], c_data[analyzer.score_col].abs(),
                              c=color, s=30, alpha=0.8, label=cluster_labels.get(str(c), f"Module {c}"),
                              edgecolors='white', linewidth=0.3)
                
                ax.set_xlabel('Pathway Index', fontsize=11, fontweight='medium')
                ax.set_ylabel('|NES|', fontsize=11, fontweight='medium')
                ax.set_title('Manhattan Plot of Pathway Enrichment', fontsize=14, fontweight='bold')
                ax.legend(title='Module', loc='upper right', frameon=True, fancybox=False, edgecolor='#dfe6e9')
                
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                
                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                figures['Fig9_Manhattan_Plot'] = buf.getvalue()
                plt.close(fig)
        except Exception as e:
            print(f"Error generating manhattan plot: {e}")
        
        try:
            # Figure 10: Module NES Density Plot (seaborn KDE)
            if analyzer.score_col:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for i, c in enumerate(reversed(clusters)):
                    c_data = cluster_df[cluster_df['Cluster'] == c][analyzer.score_col].dropna()
                    if len(c_data) > 1:
                        color = colors[int(c) % len(colors)]
                        # KDE plot with fill
                        try:
                            sns.kdeplot(
                                data=c_data,
                                ax=ax,
                                fill=True,
                                alpha=0.4,
                                color=color,
                                linewidth=1.5,
                                label=cluster_labels.get(str(c), f"Module {c}")
                            )
                        except:
                            # Fallback to histogram if KDE fails
                            ax.hist(
                                c_data,
                                bins=15,
                                alpha=0.4,
                                color=color,
                                edgecolor='white',
                                label=cluster_labels.get(str(c), f"Module {c}")
                            )
                
                ax.axvline(x=0, color='#636e72', linestyle='--', linewidth=1, alpha=0.7)
                ax.set_xlabel('NES', fontsize=11, fontweight='medium')
                ax.set_ylabel('Density', fontsize=11, fontweight='medium')
                ax.set_title('NES Density by Module', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#dfe6e9')
                
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                
                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                figures['Fig10_Module_NES_Density'] = buf.getvalue()
                plt.close(fig)
        except Exception as e:
            print(f"Error generating mountain plot: {e}")
        
        # Reset matplotlib/seaborn to defaults
        sns.reset_defaults()
        plt.style.use('default')
        
        return figures, cluster_labels, colors

    @app.post("/api/export/image/tiff")
    async def convert_image_to_tiff(payload: ImageTiffConvertRequest):
        """Convert a browser image data URL to TIFF for publication export."""
        data_url = str(payload.image_data_url or "").strip()
        if not data_url.startswith("data:image/") or "," not in data_url:
            raise HTTPException(status_code=400, detail="Invalid image payload for TIFF conversion.")
        try:
            import base64
            header, encoded = data_url.split(",", 1)
            raw = base64.b64decode(encoded)
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as img:
                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGB")
                out_buf = io.BytesIO()
                img.save(out_buf, format="TIFF", compression="tiff_deflate")
                out_buf.seek(0)
                tiff_b64 = base64.b64encode(out_buf.read()).decode("ascii")
            return {"status": "success", "image_data_url": f"data:image/tiff;base64,{tiff_b64}"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TIFF conversion failed: {str(e)}")

    @app.post("/api/export")
    async def export_publication_package(
        x_session_id: Optional[str] = Header(None),
        img_format: str = Form("png"),
        dpi: int = Form(300),
        include_3d: bool = Form(True),
        include_2d: bool = Form(True),
        include_elbow: bool = Form(True),
        include_heatmap: bool = Form(True),
        include_barplots: bool = Form(True),
        data_format: str = Form("xlsx"),
        include_html: bool = Form(True),
        include_json: bool = Form(True),
        include_table: bool = Form(True),
        include_chat_history: bool = Form(False),  # backward compatibility (unused)
    ):
        """
        Generate publication-ready export package with figures and data.
        
        Supported image formats: png, jpeg, pdf, tiff, svg
        Supported DPI: 150, 300, 600
        Supported data formats: xlsx, csv, tsv, h5ad, h5seurat
        
        Returns a ZIP file containing all requested outputs.
        """
        import zipfile
        from io import BytesIO
        import base64
        import tempfile
        from urllib.parse import quote
        
        if not x_session_id:
            raise HTTPException(status_code=400, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session or not session.get("analyzer"):
            raise HTTPException(status_code=404, detail="No analysis found")
        
        analyzer = session["analyzer"]
        ai_results = session.get("ai_results")
        deg_df = session.get("deg_df")
        deg_config = session.get("deg_config")
        gene_stats = analyzer.gene_stats if analyzer.gene_stats is not None else analyzer.analyze_gene_frequencies()
        data_format = str(data_format or "xlsx").strip().lower()
        if data_format not in {"xlsx", "csv", "tsv", "h5ad", "h5seurat"}:
            data_format = "xlsx"
        report_basename = _session_report_basename(session)
        zip_filename = f"{report_basename}.zip"
        html_filename_with_chat = f"{report_basename}_with_chat.html"
        html_filename_no_chat = f"{report_basename}_no_chat.html"
        repro_manifest_with_chat = _build_reproducibility_manifest(
            x_session_id,
            session,
            analyzer,
            ai_results or [],
            include_chat_history=True
        )
        repro_manifest_no_chat = _build_reproducibility_manifest(
            x_session_id,
            session,
            analyzer,
            ai_results or [],
            include_chat_history=False
        )
        
        # Generate all figures
        figures, cluster_labels, colors = _generate_publication_figures(analyzer, ai_results, gene_stats, img_format, dpi)
        
        # Determine file extension
        ext_map = {'jpeg': 'jpg', 'tiff': 'tif'}
        file_ext = ext_map.get(img_format, img_format)
        
        # Create ZIP file
        zip_buffer = BytesIO()
        
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add all figures (export-all should always include full figure set)
                for fig_name, fig_data in figures.items():
                    if fig_data:
                        zf.writestr(f"figures/{fig_name}.{file_ext}", fig_data)
                
                # Prepare cluster results with AI annotations
                export_df = analyzer.cluster_results.copy()
                if ai_results:
                    ai_title_map = {str(a['cluster_id']): a.get('title', '') for a in ai_results}
                    ai_summary_map = {str(a['cluster_id']): a.get('summary', '') for a in ai_results}
                    export_df['AI_Title'] = export_df['Cluster'].astype(str).map(ai_title_map)
                    export_df['AI_Summary'] = export_df['Cluster'].astype(str).map(ai_summary_map)
                
                # Data files (optional, selected by format)
                if include_table:
                    if data_format == "xlsx":
                        xlsx_buf = BytesIO()
                        with pd.ExcelWriter(xlsx_buf, engine='openpyxl') as writer:
                            export_df.to_excel(writer, sheet_name='Cluster_Results', index=False)
                            gene_stats.to_excel(writer, sheet_name='Gene_Statistics', index=False)

                            summary_data = []
                            clusters = sorted(export_df['Cluster'].unique())
                            for c in clusters:
                                c_paths = export_df[export_df['Cluster'] == c]
                                c_genes = gene_stats[(gene_stats['Cluster'] == c) & (gene_stats['Percentage'] >= 25)]
                                ai_title = ''
                                if ai_results:
                                    for a in ai_results:
                                        if str(a['cluster_id']) == str(c):
                                            ai_title = a.get('title', '')
                                            break
                                summary_data.append({
                                    'Cluster': c,
                                    'AI_Title': ai_title,
                                    'N_Pathways': len(c_paths),
                                    'N_Core_Genes': len(c_genes),
                                    'Mean_NES': c_paths[analyzer.score_col].mean() if analyzer.score_col else None,
                                    'Top_Genes': ', '.join(c_genes.nlargest(10, 'Percentage')['Item'].tolist())
                                })
                            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Cluster_Summary', index=False)
                        xlsx_buf.seek(0)
                        zf.writestr("data/gemmap_results.xlsx", xlsx_buf.read())
                    elif data_format == "csv":
                        csv_buf = BytesIO()
                        export_df.to_csv(csv_buf, index=False)
                        csv_buf.seek(0)
                        zf.writestr("data/cluster_results.csv", csv_buf.read())

                        csv_gene_buf = BytesIO()
                        gene_stats.to_csv(csv_gene_buf, index=False)
                        csv_gene_buf.seek(0)
                        zf.writestr("data/gene_statistics.csv", csv_gene_buf.read())
                    elif data_format == "tsv":
                        tsv_buf = BytesIO()
                        export_df.to_csv(tsv_buf, index=False, sep='\t')
                        tsv_buf.seek(0)
                        zf.writestr("data/cluster_results.tsv", tsv_buf.read())

                        tsv_gene_buf = BytesIO()
                        gene_stats.to_csv(tsv_gene_buf, index=False, sep='\t')
                        tsv_gene_buf.seek(0)
                        zf.writestr("data/gene_statistics.tsv", tsv_gene_buf.read())
                    elif data_format == "h5ad":
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
                                tmp_path = tmp_file.name
                            analyzer.save_h5ad(tmp_path)
                            with open(tmp_path, "rb") as f:
                                zf.writestr("data/gemmap_results.h5ad", f.read())
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass
                    else:  # h5seurat
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".h5seurat", delete=False) as tmp_file:
                                tmp_path = tmp_file.name
                            analyzer.save_h5seurat(tmp_path)
                            with open(tmp_path, "rb") as f:
                                zf.writestr("data/gemmap_results.h5seurat", f.read())
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass

                # Reproducibility supplement JSON (with and without chat)
                repro_filename_chat = f"{report_basename}_reproducibility_with_chat.json"
                repro_filename_no_chat = f"{report_basename}_reproducibility_no_chat.json"
                if include_json:
                    repro_json_chat = json.dumps(json_clean(repro_manifest_with_chat), ensure_ascii=False, indent=2)
                    repro_json_no_chat = json.dumps(json_clean(repro_manifest_no_chat), ensure_ascii=False, indent=2)
                    zf.writestr(f"reproducibility/{repro_filename_chat}", repro_json_chat.encode("utf-8"))
                    zf.writestr(f"reproducibility/{repro_filename_no_chat}", repro_json_no_chat.encode("utf-8"))
                
                # Generate HTML report embeds at fixed high-quality settings for portability.
                if include_html:
                    html_figures = figures
                    html_cluster_labels = cluster_labels
                    html_colors = colors
                    html_img_format = img_format
                    try:
                        html_figures, html_cluster_labels, html_colors = _generate_publication_figures(
                            analyzer,
                            ai_results,
                            gene_stats,
                            HTML_EXPORT_IMAGE_FORMAT,
                            HTML_EXPORT_IMAGE_DPI,
                            include_annotation_variants=False,
                        )
                        html_img_format = HTML_EXPORT_IMAGE_FORMAT
                    except Exception as e:
                        print(f"HTML figure generation with default export settings failed, using main figures: {e}")

                    # Generate interactive plots once
                    interactive_3d = None
                    interactive_2d = None
                    try:
                        cols = analyzer.cluster_results.columns
                        if 'Dim3' in cols:
                             fig3d = analyzer.plot.scatter_3d(as_dict=False)
                             interactive_3d = fig3d.to_html(full_html=False, include_plotlyjs='cdn')
                        fig2d = analyzer.plot.scatter_2d_interactive(as_dict=False)
                        interactive_2d = fig2d.to_html(full_html=False, include_plotlyjs=False)
                    except Exception as e:
                        print(f"Interactive plot generation failed: {e}")

                    html_content_chat = _generate_html_report(
                        analyzer,
                        ai_results,
                        gene_stats,
                        html_figures,
                        html_cluster_labels,
                        html_colors,
                        html_img_format,
                        interactive_3d,
                        interactive_2d,
                        filename=session.get("filename", ""),
                        report_basename=report_basename,
                        deg_df=deg_df,
                        deg_config=deg_config,
                        repro_manifest=repro_manifest_with_chat,
                        chat_log=session.get("chat_log"),
                        chat_history_included=True
                    )
                    html_content_no_chat = _generate_html_report(
                        analyzer,
                        ai_results,
                        gene_stats,
                        html_figures,
                        html_cluster_labels,
                        html_colors,
                        html_img_format,
                        interactive_3d,
                        interactive_2d,
                        filename=session.get("filename", ""),
                        report_basename=report_basename,
                        deg_df=deg_df,
                        deg_config=deg_config,
                        repro_manifest=repro_manifest_no_chat,
                        chat_log=None,
                        chat_history_included=False
                    )
                    zf.writestr(html_filename_with_chat, html_content_chat.encode('utf-8'))
                    zf.writestr(html_filename_no_chat, html_content_no_chat.encode('utf-8'))

                # Debug log snapshot
                try:
                    report = _build_debug_log_export(x_session_id, session)
                    zf.writestr("logs/gemmap_debug_log.txt", report["payload"].encode("utf-8"))
                except Exception:
                    pass
                
                # README
                if include_table:
                    if data_format == "xlsx":
                        data_lines = [
                            "- data/gemmap_results.xlsx: Workbook with cluster results, gene stats, and cluster summary",
                        ]
                    elif data_format == "csv":
                        data_lines = [
                            "- data/cluster_results.csv + data/gene_statistics.csv: CSV table exports",
                        ]
                    elif data_format == "h5ad":
                        data_lines = [
                            "- data/gemmap_results.h5ad: AnnData export with pathway metadata, sparse pathway-gene matrix, MDS embedding, and reproducibility metadata",
                        ]
                    elif data_format == "h5seurat":
                        data_lines = [
                            "- data/gemmap_results.h5seurat: Native Seurat v5 export (generated via SeuratDisk Convert)",
                        ]
                    else:
                        data_lines = [
                            "- data/cluster_results.tsv + data/gene_statistics.tsv: TSV table exports",
                        ]
                else:
                    data_lines = ["- Table files excluded by export settings."]

                if include_html:
                    html_lines = [
                        f"- {html_filename_with_chat}: Interactive standalone report (agent chats included)",
                        f"- {html_filename_no_chat}: Interactive standalone report (agent chats omitted)",
                    ]
                else:
                    html_lines = ["- HTML report files excluded by export settings."]

                if include_json:
                    repro_lines = [
                        f"- reproducibility/{report_basename}_reproducibility_with_chat.json: Replay manifest with chat log",
                        f"- reproducibility/{report_basename}_reproducibility_no_chat.json: Replay manifest without chat log",
                    ]
                else:
                    repro_lines = ["- Reproducibility JSON files excluded by export settings."]

                readme_content = f"""# GEMMAP Publication Export
Generated by GEMMAP - Gene Expression Multi-dimensional Mapping & Agentic Profiling

## Figures ({img_format.upper()}, {dpi} DPI)

- Fig1_3D_MDS_Module_Map: 3D visualization of pathway modules in MDS space
- Fig2_2D_MDS_Projections: Three 2D projections (Dim1 vs Dim2, Dim1 vs Dim3, Dim2 vs Dim3)
- Fig3_Module_Composition: Bar charts showing pathways and core genes per module
- Fig4_Gene_Frequency_Heatmap: Heatmap of top genes across modules
- Fig5_Elbow_Silhouette: Optimal k selection plots (WCSS elbow + silhouette)
- Fig6_NES_Distribution: Box plots of NES by module
- Fig7_GSEA_Bubble_Plot: GSEA-style bubble plot (modules vs NES)
- Fig8_Top_Enriched_Terms: Bar chart of top enriched pathways
- Fig9_Manhattan_Plot: Manhattan plot showing pathway significance
- Fig10_Module_NES_Density: KDE density plot of NES distributions by module

## Data Files

{chr(10).join(data_lines)}

## HTML Report

{chr(10).join(html_lines)}
{chr(10).join(repro_lines)}
- Note: Embedded HTML figures are generated at 300 DPI. For publication, use Publication Export to choose high-quality output format.
- logs/gemmap_debug_log.txt: Session debug snapshot

## Figure Guidelines

- All figures are generated at {dpi} DPI for publication quality
- Colors are colorblind-friendly (based on Nature-style palette)
- Font sizes optimized for single-column (3.5") or double-column (7") width
"""
                zf.writestr("README.txt", readme_content.encode('utf-8'))
            
            zip_buffer.seek(0)
            
            return StreamingResponse(
                zip_buffer,
                media_type="application/x-zip-compressed",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{zip_filename}"; '
                        f"filename*=UTF-8''{quote(zip_filename)}"
                    ),
                    "Content-Type": "application/x-zip-compressed"
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    def _generate_html_report(
        analyzer,
        ai_results,
        gene_stats,
        figures,
        cluster_labels,
        colors,
        img_format,
        interactive_3d=None,
        interactive_2d=None,
        filename=None,
        report_basename=None,
        deg_df=None,
        deg_config=None,
        repro_manifest=None,
        chat_log=None,
        chat_history_included=True
    ):
        """
        Generate a publication-ready standalone HTML report.
        
        Creates a clean, professional report with embedded figures suitable
        for scientific documentation and publication supplementary materials.
        """
        import base64
        import json
        import html as html_lib
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        cluster_df = analyzer.cluster_results
        clusters = sorted(cluster_df['Cluster'].unique())

        # Create independent interactive plot HTML blocks so each tab has unique Plotly element IDs.
        interactive_3d_modules = interactive_3d
        interactive_2d_modules = interactive_2d
        try:
            cols = cluster_df.columns
            if interactive_3d is not None and 'Dim3' in cols:
                fig3d_copy = analyzer.plot.scatter_3d(as_dict=False)
                interactive_3d_modules = fig3d_copy.to_html(full_html=False, include_plotlyjs=False)
            if interactive_2d is not None:
                fig2d_copy = analyzer.plot.scatter_2d_interactive(as_dict=False)
                interactive_2d_modules = fig2d_copy.to_html(full_html=False, include_plotlyjs=False)
        except Exception:
            interactive_3d_modules = interactive_3d
            interactive_2d_modules = interactive_2d
        
        # Embed static figures directly so HTML quality matches generated export settings.
        fig_b64 = {}
        fig_mime = {}
        fallback_mime = 'image/png' if img_format == 'png' else f'image/{img_format}'

        for name, data in figures.items():
            if not data:
                continue
            if name.endswith("_No_Annotations"):
                continue
            fig_b64[name] = base64.b64encode(data).decode('utf-8')
            fig_mime[name] = fallback_mime
        
        # Methodology & Metadata
        # Methodology & Metadata
        # Start with optimal_k results, then update with analyzer.methodology to ensure
        # AI metadata (mode="Autopilot") takes precedence.
        methodology = {}
        if hasattr(analyzer, '_optimal_k_result') and analyzer._optimal_k_result:
            methodology.update(analyzer._optimal_k_result)
        methodology.update(getattr(analyzer, 'methodology', {}))
            
        version = f"v{APP_VERSION}"
        mode = methodology.get('mode', 'Manual')
        ai_model = methodology.get('ai_model', 'N/A')
        ai_provider = methodology.get('ai_provider', '')
        
        # Smart formatting for AI model to avoid redundant parenthesis
        if ai_model != 'N/A':
            # normalize strings
            model_lower = ai_model.lower()
            provider_lower = ai_provider.lower() if ai_provider else ""
            
            if not ai_provider:
                 ai_model_display = ai_model
            elif provider_lower in model_lower: 
                # e.g. model="AI Agent (Active)", provider="Active" -> don't repeat
                ai_model_display = ai_model
            else:
                ai_model_display = f"{ai_model} ({ai_provider})"
        else:
            ai_model_display = "None"
        
        # Report Title
        report_title = "GEMMAP Analysis Report"
        if report_basename:
            report_title = f"GEMMAP: {report_basename}"
        elif filename:
            report_title = f"GEMMAP: {Path(str(filename)).stem}"
        
        # Calculate Top Panel Stats
        n_pathways = len(cluster_df)
        n_clusters = len(clusters)
        n_genes = len(gene_stats['Item'].unique())
        avg_nes = cluster_df[analyzer.score_col].mean() if analyzer.score_col else 0
        avg_pathways_per_cluster = n_pathways / n_clusters
        
        # Build cluster cards with DEG highlighting
        cluster_cards = []
        for c in clusters:
            c_paths = cluster_df[cluster_df['Cluster'] == c]
            c_genes = gene_stats[(gene_stats['Cluster'] == c) & (gene_stats['Percentage'] >= 25)]
            
            ai_title = cluster_labels.get(str(c), f"Module {c}")
            ai_summary = ""
            ai_process = ""
            ai_confidence = None
            if ai_results:
                for a in ai_results:
                    if str(a['cluster_id']) == str(c):
                        ai_summary = a.get('summary', '')
                        ai_process = a.get('key_process', '')
                        ai_confidence = a.get('confidence', None)
                        break
            
            # Get top genes with DEG highlighting
            top_genes_df = c_genes.nlargest(8, 'Percentage')
            has_deg_col = 'DEG' in top_genes_df.columns
            
            # Build gene tags with DEG highlighting (orange for DEG, gray for non-DEG)
            gene_tags = []
            for _, g in top_genes_df.iterrows():
                is_deg = g.get('DEG', False) if has_deg_col else False
                pct = g.get('Percentage', 0)
                if is_deg:
                    gene_tags.append(f'<span class="gene-tag deg-gene">{g["Item"]} <small>{pct:.0f}%</small></span>')
                else:
                    gene_tags.append(f'<span class="gene-tag">{g["Item"]} <small>{pct:.0f}%</small></span>')
            
            # Count DEGs in this cluster
            n_degs_in_cluster = c_genes['DEG'].sum() if has_deg_col else 0
            
            mean_nes = c_paths[analyzer.score_col].mean() if analyzer.score_col else 0
            color = colors[int(c) % len(colors)]
            
            # Confidence badge
            confidence_html = ''
            if ai_confidence is not None:
                confidence_html = f'<span class="confidence-badge">{ai_confidence*100:.0f}%</span>'
            
            cluster_cards.append(f'''
            <div class="cluster-card" style="--accent-color: {color}">
                <div class="cluster-header">
                    <span class="cluster-id">M{c}</span>
                    {confidence_html}
                </div>
                <h3 class="cluster-title">{ai_title}</h3>
                <div class="cluster-summary-full">{ai_summary}</div>
                <div class="cluster-stats">
                    <div class="stat">
                        <span class="stat-value">{len(c_paths)}</span>
                        <span class="stat-label">Pathways</span>
                    </div>
                    <div class="stat">
                        <span class="stat-value">{len(c_genes)}</span>
                        <span class="stat-label">Core Genes</span>
                    </div>
                    <div class="stat">
                        <span class="stat-value deg-count">{n_degs_in_cluster}</span>
                        <span class="stat-label">DEGs</span>
                    </div>
                    <div class="stat">
                        <span class="stat-value nes-{'pos' if mean_nes > 0 else 'neg'}">{'+' if mean_nes > 0 else ''}{mean_nes:.2f}</span>
                        <span class="stat-label">Mean NES</span>
                    </div>
                </div>
                <div class="gene-tags">{" ".join(gene_tags)}</div>
                
                <!-- Expanded Content: All Core Genes -->
                <div class="cluster-extended-content">
                    <details class="extended-details">
                        <summary>View All Core Genes ({len(c_genes)})</summary>
                        <p class="gene-list-text-all">
                            {', '.join(c_genes.sort_values('Item')['Item'].tolist())}
                        </p>
                    </details>
                </div>

                {f'<div class="key-process"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg> {ai_process}</div>' if ai_process else ''}
            </div>''')
        
        # Build table rows with DEG highlighting
        table_rows = []
        for c in clusters:
            c_paths = cluster_df[cluster_df['Cluster'] == c]
            c_genes = gene_stats[(gene_stats['Cluster'] == c) & (gene_stats['Percentage'] >= 25)]
            mean_nes = c_paths[analyzer.score_col].mean() if analyzer.score_col else 0
            color = colors[int(c) % len(colors)]
            
            # Get top 5 genes with DEG formatting
            top_5_df = c_genes.nlargest(5, 'Percentage')
            has_deg_col = 'DEG' in top_5_df.columns
            top_5_formatted = []
            for _, g in top_5_df.iterrows():
                is_deg = g.get('DEG', False) if has_deg_col else False
                if is_deg:
                    top_5_formatted.append(f'<span class="deg-gene-inline">{g["Item"]}</span>')
                else:
                    top_5_formatted.append(g['Item'])
            
            # Count DEGs
            n_degs = c_genes['DEG'].sum() if has_deg_col else 0
            
            table_rows.append(f'''<tr>
                <td><span class="cluster-badge" style="--badge-color: {color}">M{c}</span></td>
                <td class="title-cell">{cluster_labels.get(str(c), f'Module {c}')}</td>
                <td class="num">{len(c_paths)}</td>
                <td class="num">{len(c_genes)}</td>
                <td class="num deg-count">{n_degs}</td>
                <td class="num nes-{'pos' if mean_nes > 0 else 'neg'}">{mean_nes:+.2f}</td>
                <td class="genes-cell">{', '.join(top_5_formatted)}</td>
            </tr>''')
        
        # Figure descriptions as per app
        fig_descriptions = {
            'Fig1_3D_MDS_Module_Map': 'Interactive 3D visualization showing the global landscape of pathway modules. Distances represent functional similarity (Jaccard index).',
            'Fig1_3D_MDS_Module_Map_No_Annotations': 'Same 3D MDS landscape without AI annotation labels (module IDs only).',
            'Fig2_2D_MDS_Projections': 'Pairwise projections of the MDS dimensions, providing different perspectives on module separation and density.',
            'Fig2_2D_MDS_Projections_No_Annotations': 'Same 2D MDS projections without AI annotation labels (module IDs only).',
            'Fig3_Module_Composition': 'Compositional analysis showing the number of pathways and core genes identified in each module.',
            'Fig4_Gene_Frequency_Heatmap': 'Hierarchical clustering of gene frequencies. High-frequency genes (red) define the core biological identity of each module.',
            'Fig5_Elbow_Silhouette': 'Statistical validation of optimal module count (k) using Elbow method (inertia) and Silhouette analysis.',
            'Fig6_NES_Distribution': 'Distribution of Normalized Enrichment Scores (NES), indicating whether modules are generally upregulated or downregulated.',
            'Fig7_GSEA_Bubble_Plot': 'GSEA-style bubble plot with modules on Y-axis and mean NES on X-axis. Dot size is scaled to pathway count and color denotes module identity.',
            'Fig8_Top_Enriched_Terms': 'Bar chart showing top enriched pathways by NES score. Red bars indicate upregulation, blue bars indicate downregulation.',
            'Fig9_Manhattan_Plot': 'Manhattan-style plot showing all pathways with height representing significance (|NES|), colored by module assignment.',
            'Fig10_Module_NES_Density': 'KDE density plot showing NES distribution for each module with optional histogram fallback.'
        }
        
        # Prepare Autopilot Reasoning HTML
        autopilot_reasoning_html = ""
        if methodology.get('ai_reasoning'):
            autopilot_reasoning_html = f'''<div class="method-item" style="grid-column: 1 / -1;">
                <h4>Autopilot Reasoning</h4>
                <p style="font-family: serif; font-style: italic; color: var(--color-text-dim);">
                    "{methodology['ai_reasoning']}"
                </p>
            </div>'''
        
        # Metrics Explanation HTML (Expanded by default)
        # Metrics Explanation HTML - REPLACED with DEG Summary
        # Calculate total core DEGs across all clusters (unique genes)
        if 'DEG' in gene_stats.columns:
            n_total_core_degs = gene_stats[(gene_stats['Percentage'] >= 25) & (gene_stats['DEG'] == True)]['Item'].nunique()
        else:
            n_total_core_degs = 0
            
        metrics_explanation_html = f'''
        <div style="margin-bottom: 2rem; padding: 1rem; background: rgba(251, 146, 60, 0.1); border: 1px solid rgba(251, 146, 60, 0.3); border-radius: 12px; color: #fb923c; font-weight: 500; display: flex; align-items: center; gap: 0.75rem;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
            {n_total_core_degs} DEG genes found in core programs • Highlighted in orange below
        </div>
        '''

        if not isinstance(repro_manifest, dict) or not repro_manifest:
            repro_manifest = {
                "schema": "gemmap.reproducibility.v1",
                "generated_at_utc": _now_utc_iso(),
                "gemmap_version": APP_VERSION,
                "human_overview": (
                    "This report includes a reproducibility supplement describing the deterministic "
                    "Jaccard-MDS-KMeans workflow and optional AI annotation steps."
                ),
                "agentic_framework": {
                    "steps": [
                        "Compute Jaccard similarity from pathway gene sets.",
                        "Project pathways into 3D coordinates using Classical MDS.",
                        "Detect modules with K-means (random_state=42).",
                        "Compute core-gene frequencies and optional DEG overlay.",
                        "Apply AI annotations on top pathways/core genes when enabled.",
                    ]
                }
            }
        repro_manifest_clean = json_clean(repro_manifest)
        repro_manifest_json = json.dumps(repro_manifest_clean, ensure_ascii=False, indent=2)
        repro_manifest_js = json.dumps(repro_manifest_clean, ensure_ascii=False)
        repro_overview = html_lib.escape(str(repro_manifest_clean.get("human_overview", "")))
        framework_steps = (repro_manifest_clean.get("agentic_framework", {}) or {}).get("steps", []) or []
        framework_steps_html = "".join(
            f"<li style='margin:0.2rem 0;'>{html_lib.escape(str(step))}</li>"
            for step in framework_steps
        )
        repro_manifest_json_escaped = html_lib.escape(repro_manifest_json)
        html_export_size_note = html_lib.escape(HTML_EXPORT_SIZE_NOTE)
        
        
        # Build second-tab content
        agent_reasoning_cards = []

        def _normalize_gene_list(raw_value):
            if isinstance(raw_value, list):
                return [str(g).strip() for g in raw_value if str(g).strip()]
            if raw_value is None:
                return []
            text = str(raw_value)
            for sep in [";", "|"]:
                text = text.replace(sep, ",")
            return [g.strip() for g in text.split(",") if g.strip()]

        # DEG lookup for expression-linked plots
        deg_gene_col = (deg_config or {}).get("gene_col", "gene")
        deg_lfc_col = (deg_config or {}).get("lfc_col", "log2FC")
        deg_expr_lookup = {}
        ranked_gene_payload = []
        if isinstance(deg_df, pd.DataFrame) and deg_gene_col in deg_df.columns and deg_lfc_col in deg_df.columns:
            deg_plot_df = deg_df[[deg_gene_col, deg_lfc_col]].copy()
            deg_plot_df.columns = ["Gene", "LogFC"]
            deg_plot_df["Gene"] = deg_plot_df["Gene"].astype(str)
            deg_plot_df["LogFC"] = pd.to_numeric(deg_plot_df["LogFC"], errors="coerce")
            deg_plot_df = deg_plot_df.dropna(subset=["Gene", "LogFC"])
            if not deg_plot_df.empty:
                deg_plot_df = deg_plot_df.groupby("Gene", as_index=False)["LogFC"].mean()
                deg_expr_lookup = dict(zip(deg_plot_df["Gene"].tolist(), deg_plot_df["LogFC"].tolist()))
                ranked_gene_payload = [
                    {"gene": g, "logfc": float(v)}
                    for g, v in deg_plot_df.sort_values("LogFC", ascending=False)[["Gene", "LogFC"]].itertuples(index=False)
                ]

        # Build map from pathway -> genes for mountain-plot explorer
        pathway_gene_map = {}
        try:
            raw_subset = analyzer.raw_data[[analyzer.pathway_col, analyzer.genes_col]].drop_duplicates(subset=[analyzer.pathway_col])
            for _, row in raw_subset.iterrows():
                pathway_gene_map[str(row[analyzer.pathway_col])] = _normalize_gene_list(row[analyzer.genes_col])
        except Exception:
            pathway_gene_map = {}

        pathway_payload = []
        pathway_table_rows = []
        score_col_name = analyzer.score_col if analyzer.score_col and analyzer.score_col in cluster_df.columns else None
        adj_p_value_col_name = _detect_adjusted_pvalue_column(cluster_df)
        nominal_p_value_col_name = _detect_nominal_pvalue_column(cluster_df, exclude_col=adj_p_value_col_name)
        p_value_col_name = nominal_p_value_col_name or adj_p_value_col_name
        for idx, row in cluster_df.reset_index(drop=True).iterrows():
            pathway_name = str(row[analyzer.pathway_col])
            cluster_id = int(row["Cluster"])
            module_name = cluster_labels.get(str(cluster_id), f"Module {cluster_id}")
            nes_val = float(row[score_col_name]) if score_col_name else 0.0
            p_val = None
            if p_value_col_name is not None:
                try:
                    p_tmp = pd.to_numeric(pd.Series([row[p_value_col_name]]), errors="coerce").iloc[0]
                    if pd.notna(p_tmp):
                        p_val = float(p_tmp)
                except Exception:
                    p_val = None
            adj_p_val = None
            if adj_p_value_col_name is not None:
                try:
                    adj_tmp = pd.to_numeric(pd.Series([row[adj_p_value_col_name]]), errors="coerce").iloc[0]
                    if pd.notna(adj_tmp):
                        adj_p_val = float(adj_tmp)
                except Exception:
                    adj_p_val = None
            genes_for_pathway = pathway_gene_map.get(pathway_name, [])
            dim1 = pd.to_numeric(pd.Series([row.get("Dim1", np.nan)]), errors="coerce").iloc[0]
            dim2 = pd.to_numeric(pd.Series([row.get("Dim2", np.nan)]), errors="coerce").iloc[0]
            dim3 = pd.to_numeric(pd.Series([row.get("Dim3", np.nan)]), errors="coerce").iloc[0]
            pathway_payload.append({
                "idx": idx + 1,
                "pathway": pathway_name,
                "cluster": cluster_id,
                "module": module_name,
                "nes": nes_val,
                "p_value": p_val,
                "adj_p_value": adj_p_val,
                "genes": genes_for_pathway,
                "dim1": float(dim1) if pd.notna(dim1) else None,
                "dim2": float(dim2) if pd.notna(dim2) else None,
                "dim3": float(dim3) if pd.notna(dim3) else None
            })
            p_value_display = (
                f"{p_val:.2e}" if (p_val is not None and p_val > 0) else ("<1e-300" if p_val == 0 else "NA")
            )
            adj_p_value_display = (
                f"{adj_p_val:.2e}" if (adj_p_val is not None and adj_p_val > 0) else ("<1e-300" if adj_p_val == 0 else "NA")
            )
            pathway_table_rows.append(
                f'<tr class="click-row" data-path-id="{idx + 1}" '
                f'data-pathway="{html_lib.escape(pathway_name).lower()}" '
                f'data-module="m{cluster_id}" data-module-name="{html_lib.escape(module_name).lower()}">'
                f'<td class="num">{idx + 1}</td>'
                f'<td class="title-cell">{html_lib.escape(pathway_name)}</td>'
                f'<td class="num">M{cluster_id}</td>'
                f'<td class="num nes-{"pos" if nes_val >= 0 else "neg"}">{nes_val:+.2f}</td>'
                f'<td class="num">{p_value_display}</td>'
                f'<td class="num">{adj_p_value_display}</td>'
                f'<td class="num">{len(genes_for_pathway)}</td></tr>'
            )

        core_deg_points = []
        for c in clusters:
            c_paths = cluster_df[cluster_df['Cluster'] == c]
            c_genes = gene_stats[(gene_stats['Cluster'] == c) & (gene_stats['Percentage'] >= 25)].copy()
            top_paths = c_paths[analyzer.pathway_col].astype(str).tolist()
            top_gene_names = c_genes.sort_values('Percentage', ascending=False)['Item'].astype(str).tolist()
            ai_data = next((a for a in (ai_results or []) if str(a.get('cluster_id')) == str(c)), {})
            cluster_color = colors[int(c) % len(colors)]
            cluster_name = cluster_labels.get(str(c), f"Module {c}")

            agent_reasoning_cards.append(f"""
            <div class="method-item" style="border:1px solid var(--color-border); border-radius: var(--radius-md); padding: 1rem; background: var(--color-surface-1);">
                <h4 style="color:{cluster_color};">Module M{c}: {cluster_name}</h4>
                <p><strong>AI annotation:</strong> {ai_data.get('summary', 'No annotation available.')}</p>
                <p><strong>Key process:</strong> {ai_data.get('key_process', 'N/A')}</p>
                <p><strong>Evidence genes:</strong> {', '.join(top_gene_names[:12]) if top_gene_names else 'N/A'}</p>
                <p><strong>Evidence pathways:</strong> {', '.join(top_paths[:8]) if top_paths else 'N/A'}</p>
            </div>
            """)

            if "DEG" in c_genes.columns and deg_expr_lookup:
                c_deg_core = c_genes[c_genes["DEG"] == True].copy()
                if not c_deg_core.empty:
                    c_deg_core["LogFC"] = c_deg_core["Item"].map(deg_expr_lookup)
                    c_deg_core = c_deg_core.dropna(subset=["LogFC"])
                    c_deg_core["abs_logfc"] = c_deg_core["LogFC"].abs()
                    c_deg_core = c_deg_core.sort_values(["abs_logfc", "Percentage"], ascending=[False, False]).head(16)
                    for _, g in c_deg_core.iterrows():
                        core_deg_points.append({
                            "cluster": int(c),
                            "module": cluster_name,
                            "gene": str(g["Item"]),
                            "logfc": float(g["LogFC"]),
                            "percentage": float(g.get("Percentage", 0)),
                            "color": cluster_color
                        })

        # Study-level gene context: modules + pathways containing each gene.
        pathway_to_module = {}
        try:
            for _, row in cluster_df[[analyzer.pathway_col, "Cluster"]].iterrows():
                pathway_to_module[str(row[analyzer.pathway_col])] = int(row["Cluster"])
        except Exception:
            pathway_to_module = {}

        gene_context = {}
        for path_name, genes in pathway_gene_map.items():
            module_id = pathway_to_module.get(path_name)
            for gene in genes:
                key = str(gene).strip()
                if not key:
                    continue
                slot = gene_context.setdefault(key, {"modules": set(), "pathways": []})
                if module_id is not None:
                    slot["modules"].add(int(module_id))
                if path_name not in slot["pathways"]:
                    slot["pathways"].append(path_name)

        gene_context_payload = {}
        for gene_name, ctx in gene_context.items():
            modules = sorted(list(ctx.get("modules", set())))
            pathways = ctx.get("pathways", [])
            gene_context_payload[str(gene_name).upper()] = {
                "modules": [f"M{m}" for m in modules],
                "pathway_count": len(pathways),
                "example_pathways": pathways[:4],
            }
        gene_context_payload_json = json.dumps(gene_context_payload)

        # Build searchable genes table rows for second tab
        searchable_gene_rows = []
        if not gene_stats.empty:
            for _, g in gene_stats.sort_values(["Item", "Percentage"], ascending=[True, False]).iterrows():
                gene_name = str(g.get("Item", "")).strip()
                gene_name_html = html_lib.escape(gene_name)
                gene_attr = html_lib.escape(gene_name.lower())
                gene_symbol_attr = html_lib.escape(gene_name.upper())
                cluster_id = str(g.get("Cluster", ""))
                pct = float(g.get("Percentage", 0))
                deg_flag = bool(g.get("DEG", False)) if "DEG" in gene_stats.columns else False
                context = gene_context.get(gene_name, {"modules": set(), "pathways": []})
                module_list = sorted(list(context.get("modules", set())))
                pathway_list = context.get("pathways", [])
                modules_text = ", ".join(f"M{m}" for m in module_list[:8]) if module_list else "N/A"
                pathways_text = ", ".join(pathway_list[:5]) if pathway_list else "N/A"
                context_text = (
                    f"Found in {len(pathway_list)} pathways across {len(module_list)} modules. "
                    f"Modules: {modules_text}. Pathways: {pathways_text}"
                )
                searchable_gene_rows.append(
                    f'<tr data-gene="{gene_attr}" data-gene-symbol="{gene_symbol_attr}"><td class="genes-cell">{gene_name_html}</td>'
                    f'<td class="num">M{cluster_id}</td>'
                    f'<td class="num">{pct:.1f}%</td>'
                    f'<td class="num {"deg-count" if deg_flag else ""}">{"Yes" if deg_flag else "No"}</td>'
                    f'<td class="genes-cell gene-fullname-cell" data-meta-full="{gene_symbol_attr}">Loading...</td>'
                    f'<td class="genes-cell gene-function-cell" data-meta-function="{gene_symbol_attr}">Loading...</td>'
                    f'<td class="genes-cell" style="white-space: normal; line-height: 1.4;">{html_lib.escape(context_text)}</td></tr>'
                )

        # Build chat history tab content.
        if not chat_history_included:
            chat_tab_html = (
                '<div class="publication-note">Agent chats were omitted from this export by privacy settings.</div>'
            )
        else:
            chat_entries = chat_log if isinstance(chat_log, list) else []
            chat_tab_blocks = []
            for entry in chat_entries:
                if not isinstance(entry, dict):
                    continue
                ts = html_lib.escape(str(entry.get("timestamp_utc", "")))
                provider = html_lib.escape(str(entry.get("provider", "")))
                selected_module = html_lib.escape(str(entry.get("selected_module", "")))
                user_msg = html_lib.escape(str(entry.get("user_message", "")))
                assistant_reply = html_lib.escape(str(entry.get("assistant_reply", "")))
                literature = entry.get("literature", []) if isinstance(entry.get("literature"), list) else []
                lit_links = []
                for p in literature[:5]:
                    if not isinstance(p, dict):
                        continue
                    pmid = html_lib.escape(str(p.get("pmid", "")))
                    title = html_lib.escape(str(p.get("title", "")))
                    url = html_lib.escape(str(p.get("url", "")))
                    if url:
                        lit_links.append(f'<a href="{url}" target="_blank" rel="noreferrer">PMID {pmid}: {title}</a>')
                lit_html = "<br>".join(lit_links) if lit_links else "No linked papers."
                chat_tab_blocks.append(
                    f'''
                    <div class="method-item" style="border:1px solid var(--color-border); border-radius: var(--radius-md); padding: 1rem; background: var(--color-surface-1); margin-bottom: 1rem;">
                        <h4 style="margin-bottom:0.5rem;">{ts} | provider: {provider} | module: {selected_module or "auto"}</h4>
                        <p><strong>User:</strong> {user_msg}</p>
                        <p style="white-space: pre-wrap;"><strong>Agent:</strong> {assistant_reply}</p>
                        <p><strong>Papers:</strong><br>{lit_html}</p>
                    </div>
                    '''
                )
            chat_tab_html = "".join(chat_tab_blocks) if chat_tab_blocks else (
                '<div class="publication-note">No agent chats were recorded in this session.</div>'
            )

        # Panel-C style DEG-overlap dot plot
        core_deg_dot_html = (
            '<div class="publication-note">'
            'No DEG overlap detected between core genes and uploaded DEG table.'
            '</div>'
        )
        if core_deg_points:
            try:
                import plotly.graph_objects as go
                fig_core = go.Figure()
                for c in clusters:
                    c_points = [p for p in core_deg_points if int(p["cluster"]) == int(c)]
                    if not c_points:
                        continue
                    fig_core.add_trace(go.Scatter(
                        x=[p["logfc"] for p in c_points],
                        y=[p["gene"] for p in c_points],
                        mode="markers",
                        name=cluster_labels.get(str(c), f"Module {c}"),
                        marker=dict(
                            color=colors[int(c) % len(colors)],
                            size=9,
                            line=dict(width=1, color="white"),
                            opacity=0.85
                        ),
                        customdata=[[p["module"], p["percentage"]] for p in c_points],
                        hovertemplate="<b>%{y}</b><br>LogFC: %{x:.3f}<br>Module: %{customdata[0]}<br>Core frequency: %{customdata[1]:.1f}%<extra></extra>"
                    ))

                fig_core.update_layout(
                    title="Core DEG Overlap by Module (Agent-guided)",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="#1e293b"),
                    xaxis=dict(title="DEG LogFC", gridcolor="#e2e8f0", zerolinecolor="#94a3b8"),
                    yaxis=dict(title="Core DEG Gene", gridcolor="#f1f5f9", automargin=True),
                    margin=dict(l=120, r=30, t=60, b=60),
                    height=min(780, max(460, 18 * len({p["gene"] for p in core_deg_points}) + 140)),
                    legend=dict(orientation="h", y=1.02, x=0, title="Module")
                )
                core_deg_dot_html = fig_core.to_html(full_html=False, include_plotlyjs=False)
            except Exception:
                pass

        pathway_payload_json = json.dumps(pathway_payload)
        ranked_gene_payload_json = json.dumps(ranked_gene_payload)
        has_mountain_data = bool(pathway_payload) and bool(ranked_gene_payload)
        if has_mountain_data:
            mountain_tab_html = f"""
            <div id="mountainSection" class="publication-layout">
                <div class="publication-table-wrapper">
                    <input id="pathwaySearchInput" class="publication-search" type="text" placeholder="Search pathways or modules (e.g., M4)..." />
                    <div id="moduleFilterContainer"></div>
                    <div style="max-height: 520px; overflow-y: auto;">
                        <table id="pathwayTable">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Pathway</th>
                                    <th>Module</th>
                                    <th>NES</th>
                                    <th>p-value</th>
                                    <th>Adjusted p-value</th>
                                    <th>Genes</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(pathway_table_rows)}</tbody>
                        </table>
                    </div>
                </div>
                <div class="publication-plot-panel">
                    <div id="mountainPlotContainer" class="map-container publication-plot-shell" style="height: 560px; margin-bottom: 0.75rem;"></div>
                    <div id="mountainMeta" style="color: var(--color-text-muted); font-size: 0.85rem;"></div>
                    <div id="mountainHitDetails" class="mountain-hit-details">
                        Hover over a pathway-gene hit marker to see full gene annotation.
                    </div>
                    <div id="mountainPathwayDetails" class="pathway-detail-card">
                        Select a pathway to view NES, Pvalue, Adjusted Pvalue, and module context.
                    </div>
                </div>
            </div>
            """
        else:
            mountain_tab_html = """
            <div id="mountainSection" class="publication-note">
                Mountain explorer is unavailable because DEG ranking data or pathway gene sets were not found.
            </div>
            """

        # Load and encode icon for standalone embedding
        icon_b64 = _load_icon_b64()
        logo_html = (
            f'<img src="data:image/png;base64,{icon_b64}" class="logo" alt="GEMMAP Logo">'
            if icon_b64 else
            '<svg class="logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            '<path d="M12 2L3 7l9 5 9-5-9-5z"/><path d="M3 17l9 5 9-5"/><path d="M3 12l9 5 9-5"/></svg>'
        )

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    {f'<link rel="icon" type="image/png" href="data:image/png;base64,{icon_b64}">' if icon_b64 else ''}
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    
    <style>
        :root {{
            /* Dark Theme Variables */
            --color-bg: #020617;       /* slate-950 */
            --color-surface-1: #0f172a; /* slate-900 */
            --color-surface-2: #1e293b; /* slate-800 */
            --color-surface-3: #334155; /* slate-700 */
            
            --color-text: #f8fafc;     /* slate-50 */
            --color-text-muted: #94a3b8; /* slate-400 */
            --color-text-dim: #64748b;   /* slate-500 */
            
            --color-primary: #06b6d4;  /* cyan-500 */
            --color-primary-dim: #0e7490; /* cyan-700 */
            --color-secondary: #8b5cf6; /* violet-500 */
            
            --color-positive: #4ade80; /* green-400 */
            --color-negative: #f87171; /* red-400 */
            
            --color-border: #1e293b;   /* slate-800 */
            
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.5);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.5);
            
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --radius-xl: 1rem;
        }}
        
        *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--color-bg);
            color: var(--color-text);
            line-height: 1.6;
            font-size: 15px;
            -webkit-font-smoothing: antialiased;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 0 2rem 4rem; }}
        
        /* Header */
        .header {{
            background: linear-gradient(to bottom, #020617, #0f172a);
            border-bottom: 1px solid var(--color-border);
            padding: 3rem 0;
            margin-bottom: 3rem;
        }}
        .header-content {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 2rem;
        }}
        .brand {{ display: flex; align-items: center; gap: 1rem; }}
        .logo {{
            width: 84px; height: 84px; color: var(--color-primary);
            animation: pulse 3s infinite;
        }}
        @keyframes pulse {{ 50% {{ opacity: 0.7; transform: scale(0.95); }} }}
        
        .title h1 {{
            font-size: 2.25rem; font-weight: 700; letter-spacing: -0.025em;
            background: linear-gradient(to right, #fff, #94a3b8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }}
        .title p {{
            color: var(--color-primary); font-weight: 500; letter-spacing: 0.05em;
            text-transform: uppercase; font-size: 0.75rem;
        }}
        
        .meta-tags {{ display: flex; gap: 0.75rem; flex-wrap: wrap; }}
        .meta-tag {{
            background: var(--color-surface-2); padding: 0.5rem 1rem;
            border-radius: 9999px; font-size: 0.875rem;
            color: var(--color-text-muted); border: 1px solid var(--color-border);
            display: flex; align-items: center; gap: 0.5rem;
        }}
        .meta-tag strong {{ color: var(--color-text); }}
        
        /* Analysis Metrics Bar */
        .metrics-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        .metric-card {{
            background: var(--color-surface-1);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        .metric-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--color-text-dim);
            font-weight: 600;
        }}
        .metric-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--color-text);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Section styling */
        .section-title {{
            font-size: 1.5rem; font-weight: 700; margin: 3rem 0 1.5rem;
            display: flex; align-items: center; gap: 0.75rem;
            color: var(--color-text);
        }}
        .section-title::before {{
            content: ''; display: block; width: 4px; height: 24px;
            background: linear-gradient(to bottom, var(--color-primary), var(--color-secondary));
            border-radius: 2px;
        }}
        .report-tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            background: var(--color-surface-1);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-lg);
            padding: 0.4rem;
        }}
        .report-tab {{
            border: 1px solid transparent;
            background: transparent;
            color: var(--color-text-muted);
            padding: 0.55rem 0.85rem;
            border-radius: 0.5rem;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
        }}
        .report-tab.active {{
            background: var(--color-surface-2);
            color: var(--color-text);
            border-color: var(--color-surface-3);
        }}
        .report-pane {{ display: none; }}
        .report-pane.active {{ display: block; }}
        .gene-collapsible {{
            background: var(--color-surface-1);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-lg);
            margin-bottom: 2rem;
            overflow: hidden;
        }}
        .gene-collapsible summary {{
            cursor: pointer;
            font-weight: 600;
            color: var(--color-text);
            padding: 0.9rem 1rem;
            background: rgba(30, 41, 59, 0.55);
            user-select: none;
        }}
        .gene-collapsible summary::-webkit-details-marker {{
            display: none;
        }}
        .gene-collapsible summary::after {{
            content: '+';
            float: right;
            color: var(--color-text-muted);
        }}
        .gene-collapsible[open] summary::after {{
            content: '-';
        }}
        .gene-collapsible-body {{
            padding: 1rem;
        }}
        
        /* Methodology Box */
        .methodology-box {{
            background: rgba(139, 92, 246, 0.05);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
        }}
        .methodology-box-rows {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        .methodology-box-rows .method-item {{
            width: 100%;
        }}
        .method-item h4 {{
            color: var(--color-secondary);
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        .method-item p {{
            font-size: 0.9rem;
            color: var(--color-text-muted);
        }}
        .method-item strong {{ color: var(--color-text); }}
        
        /* Interactive Maps */
        .map-container {{
            background: white; /* Plotly needs white usually, or we style plot to be dark */
            border-radius: var(--radius-xl);
            padding: 0.5rem;
            border: 1px solid var(--color-border);
            height: 520px;
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-md);
        }}
        
        /* Cluster Cards */
        .cluster-grid {{
            display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 1.5rem;
        }}
        .cluster-card {{
            background: linear-gradient(145deg, var(--color-surface-1), #0f172a 40%, #1e293b);
            border: 1px solid var(--color-border); border-top: 4px solid var(--accent-color);
            border-radius: var(--radius-lg); padding: 1.5rem;
            box-shadow: var(--shadow-md); transition: transform 0.2s, box-shadow 0.2s;
        }}
        .cluster-card:hover {{
            transform: translateY(-2px); box-shadow: var(--shadow-lg); border-color: var(--color-surface-3);
        }}
        .cluster-header {{ display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem; }}
        .cluster-id {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700;
            padding: 0.25rem 0.5rem; border-radius: var(--radius-md);
            background: rgba(15, 23, 42, 0.5); color: var(--accent-color); border: 1px solid var(--accent-color);
        }}
        .cluster-header h3 {{ font-size: 1.125rem; font-weight: 600; line-height: 1.4; color: var(--color-text); }}
        .cluster-summary {{
            font-size: 0.925rem; color: var(--color-text-muted); margin-bottom: 1.25rem;
            display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; height: 4.5em;
        }}
        .cluster-stats {{
            display: flex; gap: 0.75rem; margin-bottom: 1.25rem; padding: 1rem 0;
            border-top: 1px solid var(--color-border); border-bottom: 1px solid var(--color-border);
            background: rgba(2, 6, 23, 0.3); margin-left: -1.5rem; margin-right: -1.5rem; padding-left: 1.5rem; padding-right: 1.5rem;
        }}
        .stat {{ flex: 1; text-align: center; }}
        .stat-value {{
            display: block; font-size: 1.25rem; font-weight: 700; color: var(--color-text);
            font-family: 'JetBrains Mono', monospace;
        }}
        .stat-label {{
            font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--color-text-dim); font-weight: 600;
        }}
        .nes-pos {{ color: var(--color-positive); }} .nes-neg {{ color: var(--color-negative); }}
        .gene-tags {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }}
        .gene-tag {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--color-text-muted);
            background: var(--color-surface-2); padding: 0.25rem 0.625rem;
            border-radius: 99px; border: 1px solid transparent;
        }}
        .gene-tag.deg-gene {{
            color: #fb923c; background: rgba(251, 146, 60, 0.15);
            border: 1px solid rgba(251, 146, 60, 0.3);
        }}
        .gene-tag small {{ opacity: 0.6; margin-left: 0.25rem; }}
        .confidence-badge {{
            font-size: 0.65rem; background: rgba(147, 51, 234, 0.2);
            color: #c084fc; padding: 0.2rem 0.5rem; border-radius: 99px;
            border: 1px solid rgba(147, 51, 234, 0.3);
        }}
        .deg-count {{ color: #fb923c; }}
        .deg-gene-inline {{ color: #fb923c; font-weight: 600; }}
        .cluster-title {{ font-size: 1rem; font-weight: 600; margin: 0.5rem 0; color: var(--color-text); }}
        .key-process {{
            font-size: 0.8rem; color: var(--color-primary); background: rgba(6, 182, 212, 0.1);
            padding: 0.75rem; border-radius: var(--radius-md); display: flex; gap: 0.5rem;
            align-items: center; border: 1px solid rgba(6, 182, 212, 0.2);
        }}
        .key-process .icon {{ width: 16px; height: 16px; min-width: 16px; }}
        .key-process .icon {{ width: 16px; height: 16px; min-width: 16px; }}
        
        /* Metrics Legend */
        .metrics-legend {{
            background: rgba(15, 23, 42, 0.5); border: 1px solid var(--color-border);
            border-radius: var(--radius-xl); margin-bottom: 2rem; overflow: hidden;
        }}
        .metrics-legend summary {{
            padding: 1rem; cursor: pointer; font-size: 0.9rem; font-weight: 500;
            color: var(--color-text-muted); display: flex; align-items: center; gap: 0.5rem;
            user-select: none; background: rgba(30, 41, 59, 0.5);
        }}
        .metrics-legend summary:hover {{ color: var(--color-text); }}
        .metrics-legend summary .icon {{ width: 16px; height: 16px; }}
        .metrics-grid {{
            padding: 1.5rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 2rem; font-size: 0.8rem; color: var(--color-text-muted);
        }}
        .metric-column h4 {{
            color: var(--color-text); font-weight: 700; margin-bottom: 0.75rem; 
            font-size: 0.85rem;
        }}
        .metric-column p {{ margin-bottom: 0.5rem; line-height: 1.4; }}
        
        /* Extended Content */
        .cluster-summary-full {{
            font-size: 0.95rem; color: var(--color-text-muted); margin-bottom: 1.5rem;
            line-height: 1.6;
        }}
        .cluster-extended-content {{
            margin-top: 1.5rem; border-top: 1px solid var(--color-border); padding-top: 1rem;
            display: flex; flex-direction: column; gap: 1rem;
        }}
        .section-subtitle {{
            font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--color-text-dim); font-weight: 600; margin-bottom: 0.5rem;
        }}
        .gene-list-text {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--color-text);
            line-height: 1.5; word-wrap: break-word;
        }}
        .extended-details summary {{
            cursor: pointer; font-size: 0.85rem; color: var(--color-primary); font-weight: 500;
            padding: 0.5rem 0; user-select: none;
        }}
        .extended-details summary:hover {{ text-decoration: underline; }}
        .gene-list-text-all {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--color-text-muted);
            line-height: 1.5; background: var(--color-surface-2); padding: 1rem; border-radius: var(--radius-md);
            margin-top: 0.5rem; max-height: 200px; overflow-y: auto; word-break: break-all;
        }}
        
        /* Figures */
        .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 2rem; }}
        .figure {{
            background: var(--color-surface-1); border: 1px solid var(--color-border);
            border-radius: var(--radius-xl); overflow: hidden; box-shadow: var(--shadow-sm);
        }}
        .figure-content {{ background: white; padding: 0.5rem; display: flex; justify-content: center; }}
        .figure img {{ max-width: 100%; height: auto; display: block; }}
        .figure-caption {{
            padding: 1.5rem; border-top: 1px solid var(--color-border); background: var(--color-surface-1);
        }}
        .fig-title {{ font-weight: 600; color: var(--color-text); margin-bottom: 0.5rem; font-size: 1rem; }}
        .fig-desc {{ font-size: 0.875rem; color: var(--color-text-muted); }}
        
        /* Table */
        .table-wrapper {{
            background: var(--color-surface-1); border-radius: var(--radius-xl);
            border: 1px solid var(--color-border); overflow: hidden; box-shadow: var(--shadow-sm);
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{
            background: var(--color-surface-2); color: var(--color-text-muted); font-weight: 600;
            text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; padding: 1rem 1.5rem; text-align: left;
        }}
        td {{ padding: 1rem 1.5rem; border-bottom: 1px solid var(--color-border); color: var(--color-text); font-size: 0.9rem; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: rgba(255,255,255,0.02); }}
        
        .cluster-badge {{
            font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.75rem;
            padding: 0.2rem 0.5rem; border-radius: 4px; background: var(--badge-color);
            color: rgba(0,0,0,0.8);
        }}
        .num {{ font-family: 'JetBrains Mono', monospace; }}
        .title-cell {{ font-weight: 500; color: var(--color-primary); }}
        .genes-cell {{ color: var(--color-text-dim); font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; }}
        .search-input {{
            width: 100%;
            background: var(--color-surface-1);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
            color: var(--color-text);
            padding: 0.7rem 0.85rem;
            font-size: 0.9rem;
            margin-bottom: 0.9rem;
        }}
        .click-row {{ cursor: pointer; }}
        .click-row.active td {{
            background: rgba(6, 182, 212, 0.12);
            border-top: 1px solid rgba(6, 182, 212, 0.28);
            border-bottom: 1px solid rgba(6, 182, 212, 0.28);
        }}
        .publication-layout {{
            display: grid;
            grid-template-columns: minmax(340px, 1fr) minmax(520px, 2fr);
            gap: 1rem;
            align-items: start;
        }}
        .publication-table-wrapper {{
            background: white;
            border: 1px solid #dbe3ef;
            border-radius: var(--radius-xl);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            padding: 1rem;
        }}
        .publication-table-wrapper table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .publication-table-wrapper th {{
            background: #f8fafc;
            color: #334155;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.74rem;
            padding: 0.75rem 0.9rem;
        }}
        .publication-table-wrapper td {{
            color: #0f172a;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.84rem;
            padding: 0.7rem 0.9rem;
        }}
        .publication-table-wrapper tr:hover td {{
            background: #f8fafc;
        }}
        .publication-table-wrapper .title-cell {{
            color: #0f172a;
            font-weight: 500;
        }}
        .publication-table-wrapper .num {{
            color: #334155;
        }}
        .publication-table-wrapper .click-row.active td {{
            background: #e0f2fe;
            border-top: 1px solid #7dd3fc;
            border-bottom: 1px solid #7dd3fc;
        }}
        .publication-search {{
            width: 100%;
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 0.5rem;
            color: #0f172a;
            padding: 0.65rem 0.8rem;
            font-size: 0.88rem;
            margin-bottom: 0.9rem;
        }}
        .publication-search::placeholder {{
            color: #64748b;
        }}
        .module-filter-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 0.6rem;
        }}
        .module-chip {{
            border: 1px solid #cbd5e1;
            background: #0f172a;
            color: #cbd5e1;
            border-radius: 0.45rem;
            padding: 0.3rem 0.65rem;
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .module-chip:hover {{ opacity: 0.85; }}
        .module-chip.active {{
            background: #e2e8f0;
            color: #0f172a;
            border-color: #94a3b8;
            box-shadow: 0 0 0 1px rgba(148,163,184,0.35);
        }}
        .publication-plot-panel {{
            background: white;
            border: 1px solid #dbe3ef;
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-sm);
            padding: 1rem;
        }}
        .pathway-detail-card {{
            margin-top: 0.65rem;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 0.7rem;
            padding: 0.8rem 0.9rem;
            color: #0f172a;
            box-shadow: var(--shadow-sm);
            line-height: 1.45;
        }}
        .pathway-detail-card .title {{
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }}
        .pathway-detail-card .meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            font-size: 0.9rem;
            color: #334155;
            margin-bottom: 0.4rem;
        }}
        .pathway-detail-card .label {{
            font-weight: 600;
            color: #0f172a;
            margin-right: 0.25rem;
        }}
        .core-deg-figure {{
            max-width: 980px;
            margin: 0 auto;
        }}
        .publication-plot-shell {{
            background: white;
            border: 1px solid #e2e8f0;
            box-shadow: none;
            overflow: visible;
        }}
        .mountain-hit-details {{
            margin-top: 0.75rem;
            border: 1px solid #e2e8f0;
            border-radius: 0.6rem;
            background: #f8fafc;
            color: #0f172a;
            padding: 0.75rem 0.85rem;
            font-size: 0.84rem;
            line-height: 1.45;
            word-break: break-word;
            overflow-wrap: anywhere;
            min-height: 80px;
        }}
        .mountain-hit-details .label {{
            color: #334155;
            font-weight: 700;
            margin-right: 0.35rem;
        }}
        .mountain-hit-details .title {{
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }}
        .publication-note {{
            background: white;
            color: #475569;
            border: 1px solid #dbe3ef;
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-sm);
            padding: 1rem;
        }}
        .html-size-banner {{
            margin: 1rem 0 1.1rem;
            border: 1px solid rgba(249, 115, 22, 0.55);
            border-radius: 0.9rem;
            background: linear-gradient(135deg, rgba(255, 237, 213, 0.95), rgba(255, 247, 237, 0.98));
            color: #9a3412;
            padding: 0.8rem 0.95rem;
            font-size: 0.92rem;
            line-height: 1.45;
            font-weight: 600;
        }}
        .html-size-banner strong {{
            color: #7c2d12;
        }}
        
        /* Footer */
        footer {{
            text-align: center; padding: 4rem 2rem; margin-top: 4rem;
            border-top: 1px solid var(--color-border); color: var(--color-text-dim);
        }}
        
        @media (max-width: 768px) {{
            .header-content {{ flex-direction: column; text-align: center; }}
            .logo {{ width: 72px; height: 72px; }}
            .figure-grid, .cluster-grid {{ grid-template-columns: 1fr; }}
            .publication-layout {{ grid-template-columns: 1fr; }}
            .container {{ padding: 0 1rem 2rem; }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="brand">
                {logo_html}
                <div class="title">
                    <h1>{report_title}</h1>
                    <p>Gene Expression Multi-dimensional Mapping</p>
                </div>
            </div>
            
            <div class="meta-tags">
                <div class="meta-tag"><strong>{version}</strong></div>
                <div class="meta-tag"><strong>{datetime.now().strftime('%Y-%m-%d %H:%M')}</strong></div>
                <div class="meta-tag"><strong>{mode}</strong></div>
                <div class="meta-tag">AI Model: <strong>{ai_model_display}</strong></div>
            </div>
        </div>
    </header>
    
    <div class="container">
        
        <!-- Top Panel Stats -->
        <div class="metrics-bar">
            <div class="metric-card">
                <span class="metric-label">Total Pathways</span>
                <span class="metric-value">{n_pathways}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Total Modules</span>
                <span class="metric-value">{n_clusters}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Unique Genes</span>
                <span class="metric-value">{n_genes}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Average NES</span>
                <span class="metric-value nes-{'pos' if avg_nes > 0 else 'neg'}">{avg_nes:+.2f}</span>
            </div>
        </div>
        <div class="html-size-banner">
            <strong>Image Quality Note:</strong> {html_export_size_note}
        </div>
        <div class="report-tabs">
            <button class="report-tab active" data-pane="overview">Overview</button>
            <button class="report-tab" data-pane="chats">Agent Chats</button>
        </div>

        <section class="report-pane active" id="pane-overview">
        <h2 class="section-title">Full Results</h2>
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th style="text-align:center">Pathways</th>
                        <th style="text-align:center">Core Genes</th>
                        <th style="text-align:center">DEGs</th>
                        <th style="text-align:center">NES</th>
                        <th>Top Core Genes</th>
                    </tr>
                </thead>
                <tbody>{''.join(table_rows)}</tbody>
            </table>
        </div>
        
        <h2 class="section-title">Interactive Visualization</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 2rem;">
            {f'<div class="map-container">{interactive_3d}</div>' if interactive_3d else ''}
            {f'<div class="map-container">{interactive_2d}</div>' if interactive_2d else ''}
        </div>

        <h2 class="section-title">Mountain Explorer</h2>
        <p style="color: var(--color-text-muted); margin-bottom: 1rem;">
            Select a pathway to generate a publication-ready interactive GSEA-style mountain profile from ranked DEG logFC values.
        </p>
        {mountain_tab_html}

        <h2 class="section-title">Searchable Gene Names</h2>
        <p style="color: var(--color-text-muted); margin-bottom: 1rem;">
            Search genes and inspect module/pathway context from this study.
        </p>
        <details class="gene-collapsible">
            <summary>Open Searchable Gene Names Table</summary>
            <div class="gene-collapsible-body">
                <input id="geneSearchInput" class="search-input" type="text" placeholder="Search gene name (e.g. TP53)" />
                <div class="table-wrapper" id="geneSearchTable">
                    <table>
                        <thead>
                            <tr>
                                <th>Gene</th>
                                <th>Module</th>
                                <th>Frequency</th>
                                <th>DEG</th>
                                <th>Complete Name</th>
                                <th>Gene Function</th>
                                <th>Study Context</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(searchable_gene_rows)}</tbody>
                    </table>
                </div>
            </div>
        </details>

        <h2 class="section-title">Module Overview</h2>
        {metrics_explanation_html}
        <div class="cluster-grid">{"".join(cluster_cards)}</div>

        <!-- Methodology Section (Moved to bottom of Cluster Overview) -->
        <h2 class="section-title" style="margin-top: 3rem;">Methodology</h2>
        <div class="methodology-box">
            <div class="method-item">
                <h4>Optimal k Selection</h4>
                <p>
                    {f"Auto-selected k={methodology.get('optimal_k', 'N/A')}" if methodology else "Manual selection"} 
                    using <strong>{methodology.get('confidence', 'standard').upper()}</strong> confidence.
                </p>
                <p style="font-size: 0.8rem; margin-top:0.25rem; opacity:0.8">
                    Elbow suggested k={methodology.get('elbow_k', '?')}, Silhouette suggested k={methodology.get('silhouette_k', '?')}
                </p>
            </div>
            <div class="method-item">
                <h4>Dimensionality Reduction</h4>
                <p>Jaccard Similarity + Classical MDS (3D projection).</p>
            </div>
            <div class="method-item">
                <h4>Module Detection Algorithm</h4>
                <p>K-Means module assignment on MDS coordinates.</p>
            </div>
            {autopilot_reasoning_html}
        </div>

        <h2 class="section-title">Reproducibility Supplement</h2>
        <div class="publication-note" style="margin-bottom: 2rem;">
            <p style="margin-bottom: 0.75rem; color: #334155;">{repro_overview}</p>
            <button id="downloadReproFromHtml" style="margin-bottom: 0.75rem; background:#0f172a; color:#f8fafc; border:1px solid #334155; border-radius:0.45rem; padding:0.45rem 0.7rem; font-size:0.8rem; cursor:pointer;">
                Download Reproducibility JSON
            </button>
            <details style="margin-bottom:0.5rem;">
                <summary style="cursor:pointer; font-weight:600; color:#0f172a;">Agentic Framework Steps</summary>
                <ol style="margin:0.55rem 0 0 1.1rem; color:#334155; font-size:0.84rem;">
                    {framework_steps_html}
                </ol>
            </details>
            <details>
                <summary style="cursor:pointer; font-weight:600; color:#0f172a;">View Full JSON Manifest</summary>
                <pre style="margin-top:0.55rem; max-height:320px; overflow:auto; background:#0f172a; color:#cbd5e1; border-radius:0.5rem; padding:0.75rem; font-size:0.74rem;">{repro_manifest_json_escaped}</pre>
            </details>
        </div>
        
        <h2 class="section-title">Static Figures</h2>
        <div class="figure-grid">
            {''.join([f"""<div class="figure">
                <div class="figure-content">
                    <img src="data:{fig_mime.get(name, 'image/jpeg')};base64,{b64}" alt="{name}" loading="lazy">
                </div>
                <div class="figure-caption">
                    <div class="fig-title">{name.replace('_', ' ').replace('Fig', 'Figure ').replace('Cluster', 'Module').replace('Mountain Plot', 'Module NES Density')}</div>
                    <div class="fig-desc">{fig_descriptions.get(name, '')}</div>
                </div>
            </div>""" for name, b64 in fig_b64.items()])}
        </div>

        <h2 class="section-title">Figure 11: Core DEG Overlap</h2>
        <p style="color: var(--color-text-muted); margin-bottom: 1rem;">
            Dot plot of core genes overlapping uploaded DEGs, with x-axis showing DEG logFC.
        </p>
        <div class="publication-plot-panel core-deg-figure">
            <div class="map-container publication-plot-shell" style="height: auto; min-height: 420px;">
                {core_deg_dot_html}
            </div>
        </div>
        </section>

        <section class="report-pane" id="pane-chats">
            <h2 class="section-title">Agent Chats</h2>
            <p style="color: var(--color-text-muted); margin-bottom: 1rem;">
                Full transcript of module-chat and mountain-agent conversations captured in this session.
            </p>
            <h2 class="section-title">Agent Reasoning</h2>
            <div class="methodology-box methodology-box-rows">
                {autopilot_reasoning_html}
                {"".join(agent_reasoning_cards)}
            </div>
            <div class="methodology-box methodology-box-rows">
                {chat_tab_html}
            </div>
        </section>
        
        <footer>
            <div style="margin-bottom: 1rem; opacity: 0.5;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L2 7l10 5 10-5-10-5zm0 9l2-10 6 3M12 22l10-5V7l-10 5-10-5v10l10 5z"/></svg>
            </div>
            <p><strong>GEMMAP</strong> {version}</p>
            <p>Gene Expression Multi-dimensional Mapping</p>
        </footer>
    </div>
    <script id="gemmap-repro-manifest" type="application/json">{repro_manifest_json_escaped}</script>
    <script>
        (() => {{
            const tabButtons = Array.from(document.querySelectorAll('.report-tab'));
            const panes = {{
                overview: document.getElementById('pane-overview'),
                chats: document.getElementById('pane-chats')
            }};
            const reproManifest = {repro_manifest_js};
            const downloadReproFromHtml = document.getElementById('downloadReproFromHtml');
            tabButtons.forEach((btn) => {{
                btn.addEventListener('click', () => {{
                    tabButtons.forEach((b) => b.classList.remove('active'));
                    btn.classList.add('active');
                    const pane = btn.getAttribute('data-pane');
                    Object.entries(panes).forEach(([key, el]) => {{
                        if (!el) return;
                        el.classList.toggle('active', key === pane);
                    }});
                    setTimeout(() => {{
                        refreshPlotly();
                    }}, 120);
                }});
            }});
            if (downloadReproFromHtml) {{
                downloadReproFromHtml.addEventListener('click', () => {{
                    try {{
                        const payload = JSON.stringify(reproManifest, null, 2);
                        const blob = new Blob([payload], {{ type: 'application/json' }});
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'gemmap_reproducibility.json';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);
                    }} catch (_) {{}}
                }});
            }}

            const searchInput = document.getElementById('geneSearchInput');
            const geneRows = Array.from(document.querySelectorAll('#geneSearchTable tbody tr'));
            if (searchInput) {{
                searchInput.addEventListener('input', (e) => {{
                    const q = String(e.target.value || '').trim().toLowerCase();
                    geneRows.forEach((row) => {{
                        const gene = row.getAttribute('data-gene') || '';
                        row.style.display = !q || gene.includes(q) ? '' : 'none';
                    }});
                    hydrateGeneTableMetadata(180);
                }});
            }}

            const CLUSTER_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#795548'];
            const colorForCluster = (cluster) => {{
                const numeric = Math.abs(parseInt(cluster, 10)) || 0;
                return CLUSTER_COLORS[numeric % CLUSTER_COLORS.length];
            }};
            let activeModuleFilter = null;
            let showAllModules = false;

            const pathwayPayload = {pathway_payload_json};
            const pathwayNameToId = new Map();
            const normalizedPathwayEntries = [];
            const pathwayDetailsPanel = document.getElementById('mountainPathwayDetails');

            function normalizePathwayName(value) {{
                return String(value || '')
                    .replace(/<[^>]*>/g, ' ')
                    .split(/\\n|<br\\s*\\/?>|\\|/i)[0]
                    .replace(/[_\\s]+/g, ' ')
                    .replace(/[^a-zA-Z0-9 ]+/g, ' ')
                    .toLowerCase()
                    .trim();
            }}

            pathwayPayload.forEach((p) => {{
                const norm = normalizePathwayName(p && p.pathway);
                if (norm) pathwayNameToId.set(norm, Number(p.idx));
                normalizedPathwayEntries.push({{
                    idx: Number(p && p.idx),
                    norm,
                    dim1: Number(p && p.dim1),
                    dim2: Number(p && p.dim2),
                    dim3: Number(p && p.dim3)
                }});
            }});
            const moduleFilterContainer = document.getElementById('moduleFilterContainer');
            const moduleOptions = (() => {{
                const summary = new Map();
                (Array.isArray(pathwayPayload) ? pathwayPayload : []).forEach((p) => {{
                    const numericId = String(p && p.cluster);
                    const id = `m${{numericId}}`;
                    const current = summary.get(id) || {{ count: 0, name: p && p.module ? p.module : ('Module ' + numericId), color: colorForCluster(numericId) }};
                    summary.set(id, {{
                        count: current.count + 1,
                        name: p && p.module ? p.module : current.name,
                        color: colorForCluster(numericId)
                    }});
                }});
                return Array.from(summary.entries())
                    .map(([id, meta]) => ({{ id, ...meta }}))
                    .sort((a, b) => (parseInt(String(a.id).replace(/\\D/g, ''), 10) || 0) - (parseInt(String(b.id).replace(/\\D/g, ''), 10) || 0));
            }})();
            const rankedGenes = {ranked_gene_payload_json};
            const geneContextMap = {gene_context_payload_json};
            const pathwaySearchInput = document.getElementById('pathwaySearchInput');
            const pathwayRows = Array.from(document.querySelectorAll('#pathwayTable tbody tr.click-row'));
            const mountainPlotContainer = document.getElementById('mountainPlotContainer');
            const mountainMeta = document.getElementById('mountainMeta');
            const mountainHitDetails = document.getElementById('mountainHitDetails');
            const geneMetaCache = new Map();

            function formatPValue(value) {{
                const n = Number(value);
                if (!Number.isFinite(n)) return 'NA';
                if (n <= 0) return '<1e-300';
                if (n < 1e-4) return n.toExponential(2);
                if (n < 0.01) return n.toExponential(2);
                return n.toFixed(4);
            }}

            function compactText(value, maxLen = 180) {{
                const text = String(value || '').replace(/\\s+/g, ' ').trim();
                if (!text) return '';
                return text.length > maxLen ? `${{text.slice(0, maxLen - 1)}}...` : text;
            }}

            function applyPathwayFilters() {{
                const q = (pathwaySearchInput && pathwaySearchInput.value ? pathwaySearchInput.value : '').trim().toLowerCase();
                pathwayRows.forEach((row) => {{
                    const name = row.getAttribute('data-pathway') || '';
                    const moduleId = row.getAttribute('data-module') || '';
                    const moduleName = row.getAttribute('data-module-name') || '';
                    const matchesQuery = !q || name.includes(q) || moduleId.includes(q) || moduleName.includes(q);
                    const matchesModule = !activeModuleFilter || moduleId === activeModuleFilter;
                    row.style.display = matchesQuery && matchesModule ? '' : 'none';
                }});
            }}

            function renderModuleFilters() {{
                if (!moduleFilterContainer || !moduleOptions.length) return;
                const maxShown = 12;
                const opts = showAllModules ? moduleOptions : moduleOptions.slice(0, maxShown);
                const extra = Math.max(moduleOptions.length - maxShown, 0);
                const chips = [
                    '<button class="module-chip ' + (activeModuleFilter === null ? 'active' : '') + '" data-module="all">All</button>',
                    ...opts.map((opt) => {{
                        const isActive = activeModuleFilter === opt.id;
                        const style = 'border-color:' + opt.color + ';color:' + (isActive ? '#0f172a' : opt.color) + ';background:' + (isActive ? (opt.color + '26') : '#0f172a');
                        return '<button class="module-chip ' + (isActive ? 'active' : '') + '" data-module="' + opt.id + '" style="' + style + '" title="' + opt.name + ' (' + opt.count + ' pathways)">M' + opt.id + '</button>';
                    }}),
                    extra > 0 ? '<button class="module-chip" data-module="toggle">' + (showAllModules ? 'Collapse' : ('+' + extra + ' more')) + '</button>' : ''
                ].filter(Boolean).join('');
                moduleFilterContainer.innerHTML =
                    '<div style="font-size:11px;color:#64748b;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em;">Filter modules</div>' +
                    '<div class="module-filter-chips">' + chips + '</div>';
                Array.from(moduleFilterContainer.querySelectorAll('button')).forEach((btn) => {{
                    btn.addEventListener('click', () => {{
                        const val = btn.getAttribute('data-module');
                        if (val === 'toggle') {{
                            showAllModules = !showAllModules;
                            renderModuleFilters();
                            return;
                        }}
                        activeModuleFilter = val === 'all' ? null : String(val || '');
                        applyPathwayFilters();
                        ensureVisiblePathway();
                        renderModuleFilters();
                    }});
                }});
            }}

            function collectPathwayCandidates(point) {{
                const values = [];
                const pushCandidate = (v) => {{
                    if (typeof v !== 'string') return;
                    const first = v.split(/\\n|<br\\s*\\/?>|\\|/i)[0].trim();
                    if (first.length > 2) values.push(first);
                }};
                const cd = point && point.customdata;
                if (Array.isArray(cd)) cd.forEach(pushCandidate); else pushCandidate(cd);
                pushCandidate(point && point.text);
                pushCandidate(point && point.hovertext);
                pushCandidate(point && point.name);
                return Array.from(new Set(values));
            }}

            function findPathIdByCandidates(candidates) {{
                const normalizedCandidates = (candidates || [])
                    .map((cand) => normalizePathwayName(cand))
                    .filter(Boolean);

                for (const norm of normalizedCandidates) {{
                    if (pathwayNameToId.has(norm)) {{
                        return pathwayNameToId.get(norm);
                    }}
                }}

                let bestId = null;
                let bestOverlap = 0;
                for (const cand of normalizedCandidates) {{
                    for (const rec of normalizedPathwayEntries) {{
                        if (!rec || !rec.norm || !Number.isFinite(rec.idx)) continue;
                        if (rec.norm.includes(cand) || cand.includes(rec.norm)) {{
                            const overlap = Math.min(rec.norm.length, cand.length);
                            if (overlap > bestOverlap) {{
                                bestOverlap = overlap;
                                bestId = rec.idx;
                            }}
                        }}
                    }}
                }}
                if (bestId !== null) return bestId;
                return null;
            }}

            function findNearestPathIdByCoords(point) {{
                const x = Number(point && point.x);
                const y = Number(point && point.y);
                const z = Number(point && point.z);
                if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

                let bestId = null;
                let bestDist = Number.POSITIVE_INFINITY;
                normalizedPathwayEntries.forEach((rec) => {{
                    const px = Number(rec && rec.dim1);
                    const py = Number(rec && rec.dim2);
                    const pz = Number(rec && rec.dim3);
                    if (!Number.isFinite(px) || !Number.isFinite(py)) return;
                    const dz = Number.isFinite(z) && Number.isFinite(pz) ? (pz - z) : 0;
                    const dist = ((px - x) ** 2) + ((py - y) ** 2) + (dz ** 2);
                    if (dist < bestDist) {{
                        bestDist = dist;
                        bestId = rec.idx;
                    }}
                }});
                return bestId;
            }}

            function escapeHtml(value) {{
                return String(value || '')
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#39;');
            }}

            function renderHitDetails(hit) {{
                if (!mountainHitDetails) return;
                if (!hit) {{
                    mountainHitDetails.textContent = 'Hover over a pathway-gene hit marker to see full gene annotation.';
                    return;
                }}

                const symbol = escapeHtml(hit.symbol || 'NA');
                const rank = Number.isFinite(Number(hit.rank)) ? Number(hit.rank) : 'NA';
                const logfc = Number.isFinite(Number(hit.logfc)) ? Number(hit.logfc).toFixed(3) : 'NA';
                const fullName = escapeHtml(hit.fullName || 'Name unavailable');
                const funcText = escapeHtml(hit.function || 'Function description unavailable');

                mountainHitDetails.innerHTML = `
                    <div class="title">${{symbol}}</div>
                    <div><span class="label">Rank:</span>${{rank}} <span class="label" style="margin-left:0.8rem;">LogFC:</span>${{logfc}}</div>
                    <div><span class="label">Full Name:</span>${{fullName}}</div>
                    <div><span class="label">Function:</span>${{funcText}}</div>
                `;
            }}

            function buildStudyGeneFallback(gene) {{
                const key = String(gene || '').trim().toUpperCase();
                if (!key) return '';
                const rec = geneContextMap && geneContextMap[key];
                if (!rec) return '';
                const modules = Array.isArray(rec.modules) ? rec.modules : [];
                const examples = Array.isArray(rec.example_pathways) ? rec.example_pathways : [];
                const parts = [];
                if (modules.length) parts.push(`modules: ${{modules.join(', ')}}`);
                if (examples.length) parts.push(`example pathways: ${{examples.join(', ')}}`);
                return `Observed in this study across ${{Number(rec.pathway_count || 0)}} pathway(s); ${{parts.join(' | ')}}.`;
            }}

            function pickBestGeneHit(hits, gene) {{
                if (!Array.isArray(hits) || !hits.length) return null;
                const target = String(gene || '').toUpperCase();
                const exact = hits.find((h) => String((h && h.symbol) || '').toUpperCase() === target);
                if (exact) return exact;
                const aliasMatch = hits.find((h) =>
                    Array.isArray(h && h.alias) &&
                    h.alias.some((a) => String(a || '').toUpperCase() === target)
                );
                if (aliasMatch) return aliasMatch;
                return hits[0];
            }}

            async function fetchGeneMetadata(symbol) {{
                const gene = String(symbol || '').trim().toUpperCase();
                if (!gene) return null;
                if (geneMetaCache.has(gene)) return geneMetaCache.get(gene);

                let record = {{
                    symbol: gene,
                    fullName: gene,
                    function: 'Function description unavailable'
                }};
                const fallbackFromStudy = buildStudyGeneFallback(gene);
                const queryUrls = [
                    `https://mygene.info/v3/query?q=symbol:${{encodeURIComponent(gene)}}&species=human,mouse&size=10&fields=symbol,name,summary,alias`,
                    `https://mygene.info/v3/query?q=${{encodeURIComponent(gene)}}&species=human,mouse&size=10&fields=symbol,name,summary,alias`
                ];
                try {{
                    for (const url of queryUrls) {{
                        const response = await fetch(url);
                        if (!response.ok) continue;
                        const payload = await response.json();
                        const hit = pickBestGeneHit(Array.isArray(payload.hits) ? payload.hits : [], gene);
                        if (!hit) continue;
                        record = {{
                            symbol: String(hit.symbol || gene).toUpperCase(),
                            fullName: compactText(hit.name || gene, 180),
                            function: compactText(hit.summary || fallbackFromStudy || 'Function description unavailable', 320)
                        }};
                        break;
                    }}
                }} catch (_) {{}}

                if (!record.function || record.function === 'Function description unavailable') {{
                    record.function = compactText(fallbackFromStudy || 'Function description unavailable', 320);
                }}
                if (!record.fullName || record.fullName === 'Name unavailable') {{
                    record.fullName = gene;
                }}
                geneMetaCache.set(gene, record);
                return record;
            }}

            function applyGeneMetadataToRows(symbol, meta) {{
                const key = String(symbol || '').toUpperCase();
                if (!key) return;
                const safeKey = (window.CSS && typeof window.CSS.escape === 'function')
                    ? window.CSS.escape(key)
                    : key.replace(/"/g, '\\"');
                const fullNodes = Array.from(document.querySelectorAll(`[data-meta-full="${{safeKey}}"]`));
                const funcNodes = Array.from(document.querySelectorAll(`[data-meta-function="${{safeKey}}"]`));
                const full = escapeHtml((meta && meta.fullName) || key);
                const func = escapeHtml((meta && meta.function) || 'Function description unavailable');
                fullNodes.forEach((node) => {{
                    node.textContent = full;
                }});
                funcNodes.forEach((node) => {{
                    node.textContent = func;
                }});
            }}

            async function hydrateGeneTableMetadata(maxGenes = 260) {{
                if (!geneRows.length) return;
                const visibleRows = geneRows.filter((row) => row.style.display !== 'none');
                const rowsToRead = visibleRows.length > 0 ? visibleRows : geneRows;
                const symbols = Array.from(new Set(
                    rowsToRead
                        .map((row) => String(row.getAttribute('data-gene-symbol') || '').toUpperCase())
                        .filter(Boolean)
                )).slice(0, maxGenes);
                const missing = symbols.filter((s) => !geneMetaCache.has(s));
                const batchSize = 8;
                for (let i = 0; i < missing.length; i += batchSize) {{
                    const batch = missing.slice(i, i + batchSize);
                    await Promise.all(batch.map((s) => fetchGeneMetadata(s)));
                }}
                symbols.forEach((s) => {{
                    const meta = geneMetaCache.get(s);
                    if (meta) applyGeneMetadataToRows(s, meta);
                }});
            }}

            async function hydrateGeneMetadata(hitMeta) {{
                const symbols = Array.from(new Set(
                    (hitMeta || [])
                        .map((h) => String((h && h.gene) || '').toUpperCase())
                        .filter(Boolean)
                ));
                const missing = symbols.filter((s) => !geneMetaCache.has(s));
                const batchSize = 8;
                for (let i = 0; i < missing.length; i += batchSize) {{
                    const batch = missing.slice(i, i + batchSize);
                    await Promise.all(batch.map((s) => fetchGeneMetadata(s)));
                }}
                (hitMeta || []).forEach((h) => {{
                    const meta = geneMetaCache.get(String((h && h.gene) || '').toUpperCase());
                    h.symbol = meta && meta.symbol ? meta.symbol : String(h.gene || '').toUpperCase();
                    h.fullName = meta && meta.fullName ? meta.fullName : 'Name unavailable';
                    h.function = meta && meta.function ? meta.function : 'Function description unavailable';
                }});
            }}

            function setActivePathwayRow(pathId) {{
                pathwayRows.forEach((row) => {{
                    const rowId = Number(row.getAttribute('data-path-id') || 0);
                    row.classList.toggle('active', rowId === pathId);
                }});
            }}

            function ensureVisiblePathway() {{
                const activeRow = pathwayRows.find((row) => row.classList.contains('active') && row.style.display !== 'none');
                if (activeRow) return;
                const firstVisible = pathwayRows.find((row) => row.style.display !== 'none');
                if (firstVisible) {{
                    const firstId = Number(firstVisible.getAttribute('data-path-id') || 0);
                    if (firstId > 0) renderMountain(firstId);
                }}
            }}

            function scrollToMountain() {{
                const section = document.getElementById('mountainSection');
                if (section) {{
                    section.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }}

            function bindPlotClicks() {{
                const plots = Array.from(document.querySelectorAll('.map-container .js-plotly-plot'));
                plots.forEach((plot) => {{
                    if (typeof plot.on !== 'function') return;
                    if (plot.__gemmapPathwayClickBound) return;
                    plot.__gemmapPathwayClickBound = true;
                    plot.on('plotly_click', (evt) => {{
                        const point = evt && Array.isArray(evt.points) ? evt.points[0] : null;
                        if (!point) return;
                        const candidates = collectPathwayCandidates(point);
                        const matchId = findPathIdByCandidates(candidates) || findNearestPathIdByCoords(point);
                        if (matchId) {{
                            renderMountain(matchId);
                            scrollToMountain();
                        }}
                    }});
                }});
                return plots.length > 0;
            }}

            function bindPlotClicksWithRetry(attempt = 0) {{
                const ready = bindPlotClicks();
                if (!ready && attempt < 14) {{
                    setTimeout(() => bindPlotClicksWithRetry(attempt + 1), 220);
                }}
            }}

            function buildMountainSeries(pathRec) {{
                const ranked = Array.isArray(rankedGenes) ? rankedGenes
                    .map((g) => ({{
                        gene: String((g && g.gene) || '').toUpperCase(),
                        logfc: Number(g && g.logfc)
                    }}))
                    .filter((g) => g.gene && Number.isFinite(g.logfc))
                    : [];
                if (!ranked.length || !pathRec || !Array.isArray(pathRec.genes)) {{
                    return null;
                }}

                const geneSet = new Set(pathRec.genes.map((g) => String(g).toUpperCase()));
                const hitGenes = ranked.filter((g) => geneSet.has(g.gene));
                if (!hitGenes.length) {{
                    return {{
                        x: [1, ranked.length],
                        y: [0, 0],
                        hits: [],
                        hitMeta: [],
                        hitCount: 0,
                        total: ranked.length,
                        maxEs: 0,
                        minEs: 0
                    }};
                }}

                const hitWeightSum = hitGenes.reduce((acc, g) => acc + Math.abs(g.logfc || 0), 0) || hitGenes.length;
                const missPenalty = 1 / Math.max(ranked.length - hitGenes.length, 1);
                const x = [];
                const y = [];
                const hitTicks = [];
                const hitMeta = [];
                let running = 0;
                let maxEs = -Infinity;
                let minEs = Infinity;

                ranked.forEach((g, idx) => {{
                    if (geneSet.has(g.gene)) {{
                        running += Math.abs(g.logfc || 0) / hitWeightSum;
                        hitTicks.push(idx + 1);
                        hitMeta.push({{
                            rank: idx + 1,
                            gene: g.gene,
                            logfc: g.logfc
                        }});
                    }} else {{
                        running -= missPenalty;
                    }}
                    x.push(idx + 1);
                    y.push(running);
                    maxEs = Math.max(maxEs, running);
                    minEs = Math.min(minEs, running);
                }});

                return {{
                    x,
                    y,
                    hits: hitTicks,
                    hitMeta,
                    hitCount: hitTicks.length,
                    total: ranked.length,
                    maxEs,
                    minEs,
                    rankMetric: ranked.map((g) => Number(g.logfc)),
                    maxAbsLogfc: Math.max(...ranked.map((g) => Math.abs(Number(g.logfc) || 0)), 1e-6)
                }};
            }}

            async function renderMountain(pathId) {{
                if (!mountainPlotContainer || !window.Plotly) return;
                const rec = (Array.isArray(pathwayPayload) ? pathwayPayload : []).find((p) => Number(p.idx) === Number(pathId));
                if (!rec) return;

                const moduleId = `m${{rec.cluster}}`;
                if (activeModuleFilter && activeModuleFilter !== moduleId) {{
                    activeModuleFilter = moduleId;
                    renderModuleFilters();
                    applyPathwayFilters();
                }}

                const series = buildMountainSeries(rec);
                if (!series) {{
                    mountainPlotContainer.innerHTML = '<div style="padding:1rem;color:#94a3b8;">Mountain plot data unavailable.</div>';
                    if (mountainMeta) mountainMeta.textContent = '';
                    return;
                }}

                if (series.hitMeta && series.hitMeta.length) {{
                    await hydrateGeneMetadata(series.hitMeta);
                }}

                const ymin = Math.min(...series.y, 0);
                const ymax = Math.max(...series.y, 0);
                const ypad = Math.max((ymax - ymin) * 0.12, 0.1);
                const hitY = ymin - ypad * 0.35;
                const peakEs = Math.abs(series.maxEs) >= Math.abs(series.minEs) ? series.maxEs : series.minEs;
                const peakEsText = `${{peakEs >= 0 ? '+' : ''}}${{peakEs.toFixed(3)}}`;
                const pValueText = formatPValue(rec.p_value);
                const adjPValueText = formatPValue(rec.adj_p_value);
                const clusterColor = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#795548'][Number(rec.cluster) % 10];
                const ySpan = Math.max(ymax - ymin, 0.4);
                const stripTop = ymin - ySpan * 0.14;
                const stripHeight = Math.max(ySpan * 0.06, 0.03);
                const stripBottom = stripTop - stripHeight;
                const yFloor = stripBottom - ySpan * 0.08;
                const yCeil = ymax + ySpan * 0.24;
                const maxX = Math.max(...series.x, 1);
                const stripLabelY = stripBottom - ySpan * 0.02;

                const traces = [
                    {{
                        x: series.x,
                        y: series.y,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Running ES',
                        line: {{ color: '#334155', width: 2.3 }}
                    }},
                    {{
                        x: (series.hitMeta || []).map((h) => h.rank),
                        y: (series.hitMeta || []).map(() => hitY),
                        type: 'scatter',
                        mode: 'markers',
                        name: 'Pathway Genes (module color)',
                        marker: {{ color: clusterColor, symbol: 'line-ns-open', size: 12, line: {{ width: 1, color: clusterColor }} }},
                        customdata: (series.hitMeta || []).map((h) => [
                            h.symbol || h.gene,
                            Number.isFinite(h.logfc) ? Number(h.logfc).toFixed(3) : 'NA',
                            compactText(h.fullName || 'Name unavailable', 72),
                            compactText(h.function || 'Function description unavailable', 96),
                            h.fullName || 'Name unavailable',
                            h.function || 'Function description unavailable'
                        ]),
                        hovertemplate: '<b>%{{customdata[0]}}</b><br>Rank: %{{x}}<br>LogFC: %{{customdata[1]}}<br><i>Full annotation shown below</i><extra></extra>'
                    }},
                    {{
                        x: series.x,
                        y: series.x.map(() => (stripTop + stripBottom) / 2),
                        type: 'scatter',
                        mode: 'markers',
                        name: 'Rank Metric',
                        marker: {{
                            symbol: 'square',
                            size: 7,
                            color: series.rankMetric || [],
                            cmin: -Math.max(Number(series.maxAbsLogfc || 0), 1e-6),
                            cmax: Math.max(Number(series.maxAbsLogfc || 0), 1e-6),
                            colorscale: [
                                [0, '#1d4ed8'],
                                [0.5, '#4b5563'],
                                [1, '#dc2626']
                            ],
                            showscale: false,
                            line: {{ width: 0 }}
                        }},
                        customdata: (series.rankMetric || []).map((v) => Number(v).toFixed(3)),
                        hovertemplate: 'Rank: %{{x}}<br>Rank metric: %{{customdata}}<extra></extra>',
                        showlegend: false
                    }},
                    {{
                        x: series.x,
                        y: series.x.map(() => (stripTop + stripBottom) / 2),
                        type: 'scatter',
                        mode: 'markers',
                        name: 'Rank Strip',
                        marker: {{
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
                            opacity: 0.32,
                            showscale: false,
                            line: {{ width: 0 }}
                        }},
                        hoverinfo: 'skip',
                        showlegend: false
                    }}
                ];

                    const layout = {{
                        title: {{
                    text: `${{rec.pathway}}<br><span style="font-size:11px;color:#64748b">Module M${{rec.cluster}} | NES ${{Number(rec.nes || 0).toFixed(2)}} | Pvalue ${{pValueText}} | Adjusted Pvalue ${{adjPValueText}} | Peak ES ${{peakEsText}}</span>`
                    }},
                    paper_bgcolor: 'white',
                    plot_bgcolor: 'white',
                    font: {{ color: '#0f172a' }},
                    hovermode: 'closest',
                    hoverlabel: {{
                        align: 'left',
                        bgcolor: '#fef3c7',
                        bordercolor: '#f59e0b',
                        font: {{ size: 11, color: '#0f172a' }},
                        namelength: -1
                    }},
                    margin: {{ l: 70, r: 20, t: 70, b: 60 }},
                    xaxis: {{ title: 'Rank in DEG Gene List (by logFC)', gridcolor: '#e2e8f0', linecolor: '#cbd5e1' }},
                    yaxis: {{ title: 'Enrichment Score (ES)', gridcolor: '#e2e8f0', zerolinecolor: '#94a3b8', linecolor: '#cbd5e1', range: [yFloor, yCeil] }},
                    shapes: [
                        {{
                            type: 'line',
                            x0: 1, x1: maxX,
                            y0: 0, y1: 0,
                            line: {{ color: '#94a3b8', width: 1, dash: 'dash' }}
                        }},
                        {{
                            type: 'rect',
                            x0: 1, x1: maxX,
                            y0: stripBottom, y1: stripTop,
                            fillcolor: 'rgba(148,163,184,0.24)',
                            line: {{ width: 0 }}
                        }}
                    ],
                    annotations: [
                        {{
                            x: 0.99,
                            y: 0.99,
                            xref: 'paper',
                            yref: 'paper',
                            xanchor: 'right',
                            yanchor: 'top',
                            showarrow: false,
                            font: {{ size: 11, color: '#475569' }},
                            text: `Module M${{rec.cluster}} | NES: ${{Number(rec.nes || 0).toFixed(2)}} | Pvalue: ${{pValueText}} | Adjusted Pvalue: ${{adjPValueText}}`
                        }},
                        {{
                            x: 1,
                            y: stripLabelY,
                            xref: 'x',
                            yref: 'y',
                            xanchor: 'left',
                            yanchor: 'top',
                            showarrow: false,
                            font: {{ size: 10, color: '#dc2626' }},
                            text: 'activated'
                        }},
                        {{
                            x: maxX,
                            y: stripLabelY,
                            xref: 'x',
                            yref: 'y',
                            xanchor: 'right',
                            yanchor: 'top',
                            showarrow: false,
                            font: {{ size: 10, color: '#2563eb' }},
                            text: 'suppressed'
                        }}
                    ],
                    legend: {{ orientation: 'h', y: 1.02, x: 0 }}
                }};

                window.Plotly.newPlot(mountainPlotContainer, traces, layout, {{ responsive: true, displaylogo: false, scrollZoom: false }});
                renderHitDetails(null);
                if (typeof mountainPlotContainer.removeAllListeners === 'function') {{
                    mountainPlotContainer.removeAllListeners('plotly_hover');
                    mountainPlotContainer.removeAllListeners('plotly_unhover');
                }}
                mountainPlotContainer.on('plotly_hover', (evt) => {{
                    const points = evt && Array.isArray(evt.points) ? evt.points : [];
                    const point = points.find((p) => p && p.data && String(p.data.name || '').startsWith('Pathway Genes')) || points[0];
                    if (!point) return;
                    const cd = Array.isArray(point.customdata) ? point.customdata : [];
                    renderHitDetails({{
                        symbol: cd[0] || '',
                        rank: point.x,
                        logfc: cd[1],
                        fullName: cd[4] || cd[2] || '',
                        function: cd[5] || cd[3] || ''
                    }});
                }});
                mountainPlotContainer.on('plotly_unhover', () => {{
                    // Keep last hovered details visible; no reset on unhover.
                }});
                setActivePathwayRow(Number(pathId));
                if (pathwayDetailsPanel) {{
                    const geneCount = Array.isArray(rec.genes) ? rec.genes.length : 0;
                    const topGenes = Array.isArray(rec.genes) ? rec.genes.slice(0, 10).join(', ') : 'NA';
                    pathwayDetailsPanel.innerHTML = `
                        <div class="title">${{escapeHtml(rec.pathway || '')}}</div>
                        <div class="meta">
                            <span><span class="label">Module:</span>M${{rec.cluster}}</span>
                            <span><span class="label">NES:</span>${{Number(rec.nes || 0).toFixed(2)}}</span>
                            <span><span class="label">Pvalue:</span>${{pValueText}}</span>
                            <span><span class="label">Adjusted Pvalue:</span>${{adjPValueText}}</span>
                            <span><span class="label">Genes:</span>${{geneCount}}</span>
                        </div>
                        <div style="color:#475569;font-size:0.9rem;"><span class="label">Top genes:</span>${{escapeHtml(topGenes)}}</div>
                    `;
                }}
                if (mountainMeta) {{
                    mountainMeta.textContent = `Index: ${{rec.idx}} | Module: M${{rec.cluster}} | Pathway genes: ${{(rec.genes || []).length}} | Hits in ranked list: ${{series.hitCount}} / ${{series.total}} | NES: ${{Number(rec.nes || 0).toFixed(2)}} | Pvalue: ${{pValueText}} | Adjusted Pvalue: ${{adjPValueText}} | Peak ES: ${{peakEsText}} | ES range: ${{series.minEs.toFixed(3)}} to ${{series.maxEs.toFixed(3)}}`;
                }}
            }}

            renderModuleFilters();

            if (pathwaySearchInput && pathwayRows.length) {{
                pathwaySearchInput.addEventListener('input', () => {{
                    applyPathwayFilters();
                    ensureVisiblePathway();
                }});
            }}

            if (pathwayRows.length && mountainPlotContainer && window.Plotly) {{
                pathwayRows.forEach((row) => {{
                    row.addEventListener('click', () => {{
                        const pathId = Number(row.getAttribute('data-path-id') || 0);
                        if (pathId > 0) renderMountain(pathId);
                    }});
                }});
                applyPathwayFilters();
                ensureVisiblePathway();
            }}

            function refreshPlotly() {{
                if (!window.Plotly) return;
                const plots = Array.from(document.querySelectorAll('.map-container .js-plotly-plot'));
                plots.forEach((plot) => {{
                    try {{
                        window.Plotly.relayout(plot, {{
                            'scene.camera': {{
                                eye: {{ x: 1.45, y: 1.45, z: 1.05 }},
                                center: {{ x: 0, y: 0, z: 0 }},
                                up: {{ x: 0, y: 0, z: 1 }}
                            }},
                            'scene.aspectmode': 'cube',
                            'hoverlabel.font.size': 10,
                            'hoverlabel.namelength': 28
                        }});
                    }} catch (_) {{}}
                    try {{
                        window.Plotly.Plots.resize(plot);
                    }} catch (_) {{}}
                }});
            }}

            let resizeTimer = null;
            window.addEventListener('resize', () => {{
                if (resizeTimer) clearTimeout(resizeTimer);
                resizeTimer = setTimeout(refreshPlotly, 140);
            }});

            window.addEventListener('load', () => {{
                setTimeout(refreshPlotly, 180);
                setTimeout(refreshPlotly, 450);
                setTimeout(() => bindPlotClicksWithRetry(0), 260);
                setTimeout(() => hydrateGeneTableMetadata(220), 520);
            }});
            refreshPlotly();
        }})();
    </script>
</body>
</html>'''
        return html


    @app.get("/api/export-html")
    async def export_html_only(
        x_session_id: Optional[str] = Header(None),
        include_chat_history: bool = False
    ):
        """Quick export of just the HTML report for fast preview."""
        if not x_session_id:
            raise HTTPException(status_code=400, detail="Missing Session ID")
        session = session_manager.get_session(x_session_id)
        if not session or not session.get("analyzer"):
            raise HTTPException(status_code=404, detail="No analysis found")
        
        analyzer = session["analyzer"]
        ai_results = session.get("ai_results")
        deg_df = session.get("deg_df")
        deg_config = session.get("deg_config")
        gene_stats = analyzer.gene_stats if analyzer.gene_stats is not None else analyzer.analyze_gene_frequencies()
        report_basename = _session_report_basename(session)
        html_filename = f"{report_basename}.html"
        repro_manifest = _build_reproducibility_manifest(
            x_session_id,
            session,
            analyzer,
            ai_results or [],
            include_chat_history=include_chat_history
        )
        
        # Generate HTML figures using the fixed export defaults.
        figures, cluster_labels, colors = _generate_publication_figures(
            analyzer,
            ai_results,
            gene_stats,
            HTML_EXPORT_IMAGE_FORMAT,
            HTML_EXPORT_IMAGE_DPI,
            include_annotation_variants=False,
        )
        
        # Generate HTML
        interactive_3d = None
        interactive_2d = None
        try:
            cols = analyzer.cluster_results.columns
            if 'Dim3' in cols:
                    fig3d = analyzer.plot.scatter_3d(as_dict=False)
                    interactive_3d = fig3d.to_html(full_html=False, include_plotlyjs='cdn')
            
            fig2d = analyzer.plot.scatter_2d_interactive(as_dict=False)
            interactive_2d = fig2d.to_html(full_html=False, include_plotlyjs=False)
        except Exception as e:
            print(f"Interactive plot generation failed: {e}")

        html_content = _generate_html_report(
            analyzer,
            ai_results,
            gene_stats,
            figures,
            cluster_labels,
            colors,
            HTML_EXPORT_IMAGE_FORMAT,
            interactive_3d,
            interactive_2d,
            filename=session.get("filename", ""),
            report_basename=report_basename,
            deg_df=deg_df,
            deg_config=deg_config,
            repro_manifest=repro_manifest,
            chat_log=session.get("chat_log") if include_chat_history else None,
            chat_history_included=include_chat_history
        )
        
        return StreamingResponse(
            io.BytesIO(html_content.encode('utf-8')),
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="{html_filename}"'}
        )

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        _ensure_static_icon(static_dir)
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        @app.get("/{full_path:path}")
        async def serve_app(full_path: str):
            if full_path.startswith("api"): raise HTTPException(status_code=404)
            if full_path.startswith("static"): raise HTTPException(status_code=404)  # Let static mount handle it
            return FileResponse(static_dir / "index.html")

    return app
