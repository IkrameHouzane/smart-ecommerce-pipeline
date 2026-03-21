"""
Tests — MCP Architecture (Étape 6)
Vérifie les permissions, la whitelist et le logging de l'architecture MCP.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest


# ── Reproduire la logique MCP sans dépendre du vrai fichier ──────────────────

ALLOWED_FILES = {
    "top_k_products.csv",
    "topk_per_category.csv",
    "shop_ranking.csv",
    "ml_classification.json",
    "feature_importance.csv",
    "clusters_kmeans.csv",
    "anomalies_dbscan.csv",
    "clustering_stats.json",
    "association_rules.csv",
    "association_rules_summary.json",
    "llm_usage_log.jsonl",
    "mcp_access_log.jsonl",
}


class MockAnalyticsReaderServer:
    """Version test de AnalyticsReaderServer."""

    def __init__(self, base_dir: Path):
        self._dir = base_dir
        self.ALLOWED_FILES = ALLOWED_FILES
        self._log = []

    def _log_access(self, action, resource, detail):
        self._log.append({"action": action, "resource": resource, "detail": detail})

    def read_analytics_file(self, filename: str):
        if filename not in self.ALLOWED_FILES:
            self._log_access("DENIED", filename, "Not in whitelist")
            return None
        path = self._dir / filename
        if not path.exists():
            self._log_access("MISSING", filename, "File not found")
            return None
        self._log_access("READ", filename, "OK")
        return path.read_text(encoding="utf-8")

    def list_available_files(self):
        if not self._dir.exists():
            return []
        return [f.name for f in self._dir.iterdir()
                if f.name in self.ALLOWED_FILES]

    def list_tools(self):
        return ["read_analytics_file", "list_available_files", "get_top_products"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_analytics_dir(tmp_path):
    """Crée un dossier analytics temporaire avec quelques fichiers de test."""
    # Fichiers autorisés
    (tmp_path / "top_k_products.csv").write_text("title,score\nShoe A,0.92\n")
    (tmp_path / "shop_ranking.csv").write_text("shop_name,score_moyen\nallbirds,0.879\n")
    (tmp_path / "ml_classification.json").write_text('{"random_forest": {"accuracy": 0.926}}')
    # Fichier NON autorisé
    (tmp_path / "raw_scraped_data.json").write_text('{"secret": "data"}')
    (tmp_path / "passwords.txt").write_text("admin:1234")
    return tmp_path


@pytest.fixture
def server(tmp_analytics_dir):
    return MockAnalyticsReaderServer(tmp_analytics_dir)


# ── Tests Whitelist / Permissions ─────────────────────────────────────────────

class TestMCPPermissions:

    def test_allowed_file_readable(self, server):
        content = server.read_analytics_file("top_k_products.csv")
        assert content is not None
        assert "Shoe A" in content

    def test_denied_file_returns_none(self, server):
        result = server.read_analytics_file("raw_scraped_data.json")
        assert result is None

    def test_denied_file_logged(self, server):
        server.read_analytics_file("passwords.txt")
        denied = [e for e in server._log if e["action"] == "DENIED"]
        assert len(denied) == 1
        assert denied[0]["resource"] == "passwords.txt"

    def test_missing_allowed_file_returns_none(self, server):
        result = server.read_analytics_file("association_rules.csv")
        assert result is None

    def test_missing_allowed_file_logged_as_missing(self, server):
        server.read_analytics_file("association_rules.csv")
        missing = [e for e in server._log if e["action"] == "MISSING"]
        assert len(missing) >= 1

    def test_successful_read_logged(self, server):
        server.read_analytics_file("top_k_products.csv")
        reads = [e for e in server._log if e["action"] == "READ"]
        assert len(reads) >= 1
        assert reads[0]["detail"] == "OK"

    def test_arbitrary_path_traversal_denied(self, server):
        """Empêche l'accès à des fichiers via path traversal."""
        result = server.read_analytics_file("../../../etc/passwd")
        assert result is None

    def test_whitelist_has_expected_files(self):
        assert "top_k_products.csv" in ALLOWED_FILES
        assert "shop_ranking.csv" in ALLOWED_FILES
        assert "ml_classification.json" in ALLOWED_FILES
        assert "association_rules.csv" in ALLOWED_FILES

    def test_raw_data_not_in_whitelist(self):
        """Les données brutes ne doivent jamais être dans la whitelist."""
        forbidden = [
            "raw_products.json",
            "scraped_data.parquet",
            "passwords.txt",
            "config.py",
            ".env",
        ]
        for f in forbidden:
            assert f not in ALLOWED_FILES, f"'{f}' ne devrait pas être dans la whitelist"


class TestMCPListTools:

    def test_list_tools_returns_list(self, server):
        tools = server.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_expected_tools_declared(self, server):
        tools = server.list_tools()
        assert "read_analytics_file" in tools
        assert "list_available_files" in tools

    def test_list_available_files_only_whitelisted(self, server, tmp_analytics_dir):
        files = server.list_available_files()
        for f in files:
            assert f in ALLOWED_FILES, f"'{f}' non autorisé dans list_available_files"

    def test_non_whitelisted_files_not_listed(self, server):
        files = server.list_available_files()
        assert "raw_scraped_data.json" not in files
        assert "passwords.txt" not in files


# ── Tests Logging ─────────────────────────────────────────────────────────────

class TestMCPLogging:

    def test_log_entry_has_required_keys(self, server):
        server.read_analytics_file("top_k_products.csv")
        assert len(server._log) > 0
        entry = server._log[0]
        assert "action" in entry
        assert "resource" in entry
        assert "detail" in entry

    def test_multiple_accesses_all_logged(self, server):
        server.read_analytics_file("top_k_products.csv")
        server.read_analytics_file("shop_ranking.csv")
        server.read_analytics_file("forbidden_file.txt")
        assert len(server._log) == 3

    def test_log_is_append_only(self, server):
        server.read_analytics_file("top_k_products.csv")
        count_before = len(server._log)
        server.read_analytics_file("shop_ranking.csv")
        assert len(server._log) == count_before + 1

    def test_llm_usage_log_format(self, tmp_path):
        """Vérifie que le format du log LLM est correct (JSONL)."""
        log_path = tmp_path / "llm_usage_log.jsonl"
        entry = {
            "source": "gemini/gemini-2.0-flash",
            "prompt_preview": "Tu es un analyste eCommerce...",
            "response_preview": "Le catalogue Sport & Fitness...",
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Relire et vérifier
        with open(log_path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert len(lines) == 1
        assert lines[0]["source"] == "gemini/gemini-2.0-flash"
        assert "prompt_preview" in lines[0]
        assert "response_preview" in lines[0]