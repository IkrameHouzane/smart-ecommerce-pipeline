"""
Tests — Machine Learning & Data Mining
Vérifie les métriques ML, le clustering et les règles d'association.
"""
import pandas as pd
import pytest


# ── Tests Métriques ML ────────────────────────────────────────────────────────

class TestMLMetrics:

    def test_rf_accuracy_above_threshold(self, sample_ml_metrics):
        acc = sample_ml_metrics["random_forest"]["accuracy"]
        assert acc >= 0.85, f"RF accuracy {acc:.3f} inférieure au seuil de 0.85"

    def test_xgb_accuracy_above_threshold(self, sample_ml_metrics):
        acc = sample_ml_metrics["xgboost"]["accuracy"]
        assert acc >= 0.85, f"XGB accuracy {acc:.3f} inférieure au seuil de 0.85"

    def test_metrics_between_0_and_1(self, sample_ml_metrics):
        for model_name, metrics in sample_ml_metrics.items():
            for metric_name, value in metrics.items():
                assert 0 <= value <= 1, (
                    f"{model_name}.{metric_name} = {value} hors de [0, 1]"
                )

    def test_f1_close_to_accuracy(self, sample_ml_metrics):
        """F1 et accuracy ne doivent pas diverger de plus de 10%."""
        for model_name, metrics in sample_ml_metrics.items():
            diff = abs(metrics["accuracy"] - metrics["f1"])
            assert diff <= 0.10, (
                f"{model_name}: écart accuracy/F1 trop grand ({diff:.3f})"
            )

    def test_required_keys_present(self, sample_ml_metrics):
        required = ["accuracy", "f1", "precision", "recall"]
        for model_name, metrics in sample_ml_metrics.items():
            for key in required:
                assert key in metrics, f"{model_name} manque la clé '{key}'"

    def test_both_models_present(self, sample_ml_metrics):
        assert "random_forest" in sample_ml_metrics
        assert "xgboost" in sample_ml_metrics

    def test_precision_recall_consistent(self, sample_ml_metrics):
        """Precision et recall ne doivent pas diverger de plus de 15%."""
        for model_name, metrics in sample_ml_metrics.items():
            diff = abs(metrics["precision"] - metrics["recall"])
            assert diff <= 0.15, (
                f"{model_name}: écart précision/rappel trop grand ({diff:.3f})"
            )


# ── Tests Clustering ──────────────────────────────────────────────────────────

class TestClustering:

    def test_silhouette_score_positive(self, sample_clustering_stats):
        score = sample_clustering_stats["kmeans"]["silhouette_score"]
        assert score > 0, f"Silhouette score {score} doit être positif"

    def test_silhouette_score_max_1(self, sample_clustering_stats):
        score = sample_clustering_stats["kmeans"]["silhouette_score"]
        assert score <= 1.0

    def test_k_equals_4(self, sample_clustering_stats):
        k = sample_clustering_stats["kmeans"]["k"]
        assert k == 4, f"K attendu = 4, obtenu = {k}"

    def test_cluster_count_matches_k(self, sample_clustering_stats):
        km = sample_clustering_stats["kmeans"]
        assert len(km["clusters"]) == km["k"]

    def test_all_clusters_have_products(self, sample_clustering_stats):
        for cluster in sample_clustering_stats["kmeans"]["clusters"]:
            assert cluster["nb_produits"] > 0

    def test_cluster_prices_positive(self, sample_clustering_stats):
        for cluster in sample_clustering_stats["kmeans"]["clusters"]:
            assert cluster["prix_moyen"] > 0

    def test_dbscan_anomalies_positive(self, sample_clustering_stats):
        n = sample_clustering_stats["dbscan"]["nb_anomalies"]
        assert n >= 0

    def test_total_products_consistent(self, sample_clustering_stats):
        """La somme des produits par cluster doit être raisonnable."""
        total = sum(c["nb_produits"] for c in sample_clustering_stats["kmeans"]["clusters"])
        assert total > 0

    def test_cluster_notes_between_0_and_5(self, sample_clustering_stats):
        for cluster in sample_clustering_stats["kmeans"]["clusters"]:
            assert 0 <= cluster["note_moyenne"] <= 5


# ── Tests Règles d'Association ────────────────────────────────────────────────

class TestAssociationRules:

    def test_rules_not_empty(self, sample_rules_df):
        assert len(sample_rules_df) > 0

    def test_support_between_0_and_1(self, sample_rules_df):
        assert sample_rules_df["support"].between(0, 1).all()

    def test_confidence_between_0_and_1(self, sample_rules_df):
        assert sample_rules_df["confidence"].between(0, 1).all()

    def test_lift_positive(self, sample_rules_df):
        assert (sample_rules_df["lift"] > 0).all()

    def test_required_columns_present(self, sample_rules_df):
        required = ["antecedents", "consequents", "support", "confidence", "lift"]
        for col in required:
            assert col in sample_rules_df.columns

    def test_lift_above_1_means_positive_correlation(self, sample_rules_df):
        """Au moins une règle doit avoir lift > 1 (corrélation positive)."""
        assert (sample_rules_df["lift"] > 1).any()

    def test_high_confidence_rules_exist(self, sample_rules_df):
        """Au moins une règle doit avoir confidence >= 0.5."""
        assert (sample_rules_df["confidence"] >= 0.5).any()

    def test_antecedents_not_equal_consequents(self, sample_rules_df):
        """Antécédent et conséquent doivent être différents."""
        for _, row in sample_rules_df.iterrows():
            assert row["antecedents"] != row["consequents"]