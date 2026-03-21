"""
Shared pytest fixtures for Smart eCommerce Intelligence Pipeline tests.
FST Tanger — LSI 2M — 2025/2026
"""

from pathlib import Path

import pandas as pd
import pytest

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ANALYTICS = ROOT / "analytics"


# ── Sample DataFrames ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_products_df():
    """Minimal product DataFrame mimicking scored_products.parquet schema."""
    return pd.DataFrame(
        {
            "product_id": ["P001", "P002", "P003", "P004", "P005"],
            "title": ["Shoe A", "Shirt B", "Bag C", "Shoe D", "Top E"],
            "shop_name": ["allbirds", "allbirds", "born_primitive", "nobull", "nobull"],
            "price": [52.0, 39.0, 89.0, 120.0, 45.0],
            "rating": [4.4, 4.2, 3.8, 4.7, 4.1],
            "review_count": [120, 80, 30, 200, 60],
            "available": [True, True, False, True, True],
            "discount_pct": [30.0, 0.0, 15.0, 5.0, 0.0],
            "category": ["shoes", "shirts", "accessories", "shoes", "tops"],
            "source_platform": [
                "shopify",
                "shopify",
                "woocommerce",
                "shopify",
                "shopify",
            ],
            "composite_score": [0.92, 0.78, 0.55, 0.88, 0.71],
        }
    )


@pytest.fixture
def sample_shop_ranking_df():
    """Minimal shop ranking DataFrame."""
    return pd.DataFrame(
        {
            "shop_name": ["allbirds", "nobull", "born_primitive"],
            "score_moyen": [0.879, 0.805, 0.751],
            "prix_moyen": [70.59, 76.85, 46.39],
            "note_moyenne": [4.40, 4.52, 4.30],
            "nb_produits": [155, 98, 72],
        }
    )


@pytest.fixture
def sample_rules_df():
    """Minimal association rules DataFrame."""
    return pd.DataFrame(
        {
            "antecedents": ["shoes", "shirts", "accessories"],
            "consequents": ["accessories", "shoes", "shirts"],
            "support": [0.35, 0.28, 0.22],
            "confidence": [0.72, 0.65, 0.58],
            "lift": [2.45, 1.98, 1.67],
        }
    )


@pytest.fixture
def sample_ml_metrics():
    """Minimal ML metrics dict."""
    return {
        "random_forest": {
            "accuracy": 0.926,
            "f1": 0.918,
            "precision": 0.921,
            "recall": 0.915,
        },
        "xgboost": {
            "accuracy": 0.924,
            "f1": 0.916,
            "precision": 0.919,
            "recall": 0.913,
        },
    }


@pytest.fixture
def sample_clustering_stats():
    """Minimal clustering stats dict."""
    return {
        "kmeans": {
            "k": 4,
            "silhouette_score": 0.397,
            "clusters": [
                {
                    "cluster_kmeans": 0,
                    "nb_produits": 386,
                    "prix_moyen": 51.2,
                    "note_moyenne": 4.6,
                    "remise_moy": 12.0,
                },
                {
                    "cluster_kmeans": 1,
                    "nb_produits": 427,
                    "prix_moyen": 42.0,
                    "note_moyenne": 4.7,
                    "remise_moy": 40.0,
                },
                {
                    "cluster_kmeans": 2,
                    "nb_produits": 417,
                    "prix_moyen": 169.0,
                    "note_moyenne": 4.75,
                    "remise_moy": 1.6,
                },
                {
                    "cluster_kmeans": 3,
                    "nb_produits": 346,
                    "prix_moyen": 65.0,
                    "note_moyenne": 4.4,
                    "remise_moy": 16.0,
                },
            ],
        },
        "dbscan": {
            "nb_anomalies": 106,
            "nb_clusters": 3,
        },
    }
