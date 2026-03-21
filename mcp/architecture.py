"""
MCP-inspired responsible architecture for the Smart eCommerce Pipeline.

Implements the Model Context Protocol concepts from the dossier technique:
- MCP Host     : the main application environment (Streamlit dashboard)
- MCP Client   : the component that interacts with MCP Servers (LLM module)
- MCP Servers  : expose specific tools/data with controlled access
- Permissions  : enforce read-only access to analytics outputs
- Logs         : record all LLM and data interactions for accountability

Reference: https://modelcontextprotocol.io/specification/2025-03-26
Dossier   : FST Tanger — LSI 2M — DM & SID 2025/2026
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


# ── Config helpers ────────────────────────────────────────────────────────────

def _analytics_dir() -> Path:
    """Return the analytics output directory (relative to this file)."""
    base = Path(__file__).resolve().parent.parent
    return base / "analytics"

def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


# ── MCP Servers ───────────────────────────────────────────────────────────────

class AnalyticsReaderServer:
    """
    MCP Server — read-only access to pipeline analytics outputs.

    Enforces a strict whitelist of allowed files: the LLM layer can never
    access raw scraped data, model weights, or arbitrary filesystem paths.
    """

    ALLOWED_FILES = {
        # Top-K outputs
        "top_k_products.csv",
        "topk_per_category.csv",
        "shop_ranking.csv",
        # ML outputs
        "ml_classification.json",
        "feature_importance.csv",
        # Clustering outputs
        "clusters_kmeans.csv",
        "anomalies_dbscan.csv",
        "clustering_stats.json",
        # Association rules
        "association_rules.csv",
        "association_rules_summary.json",
        # Logs (read only)
        "llm_usage_log.jsonl",
        "mcp_access_log.jsonl",
    }

    def __init__(self):
        self._dir = _analytics_dir()

    def list_tools(self) -> list[str]:
        """Declare available tools — MCP tool declaration pattern."""
        return ["read_analytics_file", "list_available_files", "get_top_products"]

    def list_available_files(self) -> list[str]:
        """List analytics files the MCP Client is allowed to read."""
        if not self._dir.exists():
            return []
        return [
            f.name for f in self._dir.iterdir()
            if f.name in self.ALLOWED_FILES and f.is_file()
        ]

    def read_analytics_file(self, filename: str) -> str | None:
        """
        Read an analytics file — permission check enforced.
        Returns None and logs DENIED if the file is not whitelisted.
        """
        if filename not in self.ALLOWED_FILES:
            _log_access("DENIED", filename, "File not in allowed whitelist")
            return None
        path = self._dir / filename
        if not path.exists():
            _log_access("MISSING", filename, "File does not exist")
            return None
        _log_access("READ", filename, "OK")
        return path.read_text(encoding="utf-8")

    def get_top_products(self, limit: int = 5) -> str | None:
        """
        Securely extract only the Top-N products for LLM profiling.
        Returns aggregated JSON — never the full raw parquet dataset.
        """
        try:
            import pandas as pd
            path = _data_dir() / "top_k_products.csv"
            if not path.exists():
                return None
            df = pd.read_csv(path).nlargest(limit, "composite_score")
            # Return only business-relevant columns — no internal IDs
            cols = [c for c in ["title", "shop_name", "price", "rating_filled",
                                 "composite_score", "price_tier", "discount_pct"]
                    if c in df.columns]
            result = df[cols].to_json(orient="records", indent=2)
            _log_access("READ", "top_k_products.csv",
                        f"Extracted top {limit} products for LLM")
            return result
        except Exception as e:
            _log_access("ERROR", "top_k_products.csv", str(e)[:100])
            return None

    def get_shop_ranking(self) -> str | None:
        """Return shop ranking as structured JSON for LLM context."""
        try:
            import pandas as pd
            path = _analytics_dir() / "shop_ranking.csv"
            if not path.exists():
                return None
            df = pd.read_csv(path)
            _log_access("READ", "shop_ranking.csv", "Shop ranking for LLM")
            return df.to_json(orient="records", indent=2)
        except Exception as e:
            _log_access("ERROR", "shop_ranking.csv", str(e)[:100])
            return None


class SummaryGeneratorServer:
    """
    MCP Server — generate LLM summaries from structured analytics data only.

    This server never receives raw product rows — only aggregated metrics
    computed by the AnalyticsReaderServer. This enforces the isolation
    principle described in the MCP specification.
    """

    def list_tools(self) -> list[str]:
        return ["generate_executive_summary", "generate_strategy_report",
                "generate_product_profile", "chat_with_data"]

    def generate_executive_summary(self, structured_data: dict) -> str:
        """Generate a 3-5 sentence executive summary from aggregated metrics."""
        _log_access("GENERATE", "executive_summary",
                    f"input_keys={list(structured_data.keys())}")
        return _call_gemini_safe(
            f"Tu es un analyste eCommerce. Résume en 3-5 phrases ces données pour un décideur:\n"
            f"{json.dumps(structured_data, indent=2, ensure_ascii=False)[:2000]}"
        )

    def generate_strategy_report(self, structured_data: dict) -> str:
        """Generate a Chain-of-Thought strategic report."""
        _log_access("GENERATE", "strategy_report",
                    f"input_keys={list(structured_data.keys())}")
        return _call_gemini_safe(
            f"Analyse étape par étape et donne une recommandation stratégique basée sur:\n"
            f"{json.dumps(structured_data, indent=2, ensure_ascii=False)[:2000]}"
        )

    def generate_product_profile(self, top_products_json: str) -> str:
        """Generate a competitive profile for the Top-5 products."""
        _log_access("GENERATE", "product_profile",
                    f"payload_len={len(top_products_json)}")
        return _call_gemini_safe(
            f"Compare ces top produits et explique pourquoi ils performent bien:\n"
            f"{top_products_json[:2000]}"
        )

    def chat_with_data(self, query: str, context: dict,
                       history: list[dict]) -> str:
        """Handle interactive BI chat — context is aggregated metrics only."""
        _log_access("CHAT", "bi_assistant",
                    f"query_preview={query[:80]}")
        history_str = "\n".join([
            f"{m.get('role','').capitalize()}: {m.get('content','')}"
            for m in history[-3:]
        ])
        return _call_gemini_safe(
            f"Contexte analytique:\n{json.dumps(context, indent=2, ensure_ascii=False)[:2000]}\n\n"
            f"Historique:\n{history_str}\n\nQuestion: {query}"
        )



def _call_gemini_safe(prompt: str) -> str:
    """
    Internal Gemini API call — used by SummaryGeneratorServer.
    Reads GEMINI_API_KEY from environment (loaded via .env).
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        return "(LLM disabled — set GEMINI_API_KEY in .env)"
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for model in ["gemini-2.0-flash-lite", "gemini-2.0-flash"]:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return getattr(response, "text", "") or str(response)
            except Exception as e:
                if "429" in str(e) or "404" in str(e):
                    continue
                return str(e)
        return "(Quota exceeded — retry in 30s)"
    except ImportError:
        return "(google-genai not installed)"
    except Exception as e:
        return str(e)


# ── MCP Client ────────────────────────────────────────────────────────────────

class MCPClient:
    """
    MCP Client — routes requests from the Host to the appropriate Server.

    The Host (Streamlit dashboard) never calls analytics files or the LLM
    directly — it always goes through this Client, which enforces permissions
    and ensures all interactions are logged.
    """

    def __init__(self):
        self.analytics_server   = AnalyticsReaderServer()
        self.summary_server     = SummaryGeneratorServer()
        _log_access("INIT", "MCPClient", "Client initialized")

    # ── Delegated analytics access ────────────────────────────────────────────

    def get_analytics(self, filename: str) -> str | None:
        """Read a whitelisted analytics file."""
        return self.analytics_server.read_analytics_file(filename)

    def list_analytics(self) -> list[str]:
        """List available analytics files."""
        return self.analytics_server.list_available_files()

    def get_top_products(self, limit: int = 5) -> str | None:
        """Get top-K products as structured JSON for LLM."""
        return self.analytics_server.get_top_products(limit)

    def get_shop_ranking(self) -> str | None:
        """Get shop ranking as structured JSON for LLM."""
        return self.analytics_server.get_shop_ranking()

    # ── Delegated LLM generation ──────────────────────────────────────────────

    def generate_summary(self, data: dict) -> str:
        """Generate executive summary via LLM."""
        return self.summary_server.generate_executive_summary(data)

    def generate_strategy(self, data: dict) -> str:
        """Generate strategic report via Chain-of-Thought LLM."""
        return self.summary_server.generate_strategy_report(data)

    def generate_profile(self, top_products_json: str) -> str:
        """Generate product profile via LLM."""
        return self.summary_server.generate_product_profile(top_products_json)

    def chat(self, query: str, context: dict, history: list[dict]) -> str:
        """Handle BI chat query via LLM."""
        return self.summary_server.chat_with_data(query, context, history)


# ── MCP Host (documentation) ──────────────────────────────────────────────────
#
# The Streamlit dashboard (dashboard/app.py) acts as the MCP Host.
# It instantiates an MCPClient and uses it to:
#   1. Read analytics outputs    → via AnalyticsReaderServer (read-only, whitelisted)
#   2. Generate LLM summaries    → via SummaryGeneratorServer (aggregated data only)
#   3. Handle BI chat            → via SummaryGeneratorServer (context-bounded)
#
# The Host NEVER gives the LLM direct filesystem access or code execution rights.
# All LLM inputs are pre-filtered aggregated metrics — never raw product rows.


# ── Logging / Permissions ─────────────────────────────────────────────────────

def _log_access(action: str, resource: str, detail: str) -> None:
    """
    Append an access log entry for MCP accountability.
    Append-only — no entry is ever modified or deleted.
    """
    try:
        log_dir = _analytics_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action":    action,
            "resource":  resource,
            "detail":    detail[:200],
        }
        log_path = log_dir / "mcp_access_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Logging must never crash the main application


# ── Permissions summary ───────────────────────────────────────────────────────

PERMISSIONS = {
    "AnalyticsReaderServer": {
        "access":  "read-only",
        "scope":   "analytics/ (whitelisted files only — 14 files max)",
        "write":   False,
        "execute": False,
        "raw_data_access": False,
    },
    "SummaryGeneratorServer": {
        "access":  "read aggregated metrics + call external LLM API",
        "scope":   "structured aggregates only — no raw product rows",
        "write":   "append-only logs (mcp_access_log.jsonl, llm_usage_log.jsonl)",
        "execute": False,
        "raw_data_access": False,
    },
    "MCPClient": {
        "access":  "route requests between Host and Servers",
        "scope":   "full pipeline — but delegated, never direct",
        "write":   "via SummaryGeneratorServer only",
        "execute": False,
        "raw_data_access": False,
    },
}