"""
Tests — Dashboard & LLM Context
Vérifie la construction du contexte LLM et la cohérence des données dashboard.
"""

import json
import pytest


# ── Reproduire build_context() sans dépendre de Streamlit ────────────────────


def build_context(topk, shops, scored, rules, ml, cls_stats) -> dict:
    """Réplique la fonction build_context() de 7_LLM_Insights.py."""
    rf_acc = ml.get("random_forest", {}).get("accuracy", 0.0)
    xgb_acc = ml.get("xgboost", {}).get("accuracy", 0.0)

    top5p = topk.nlargest(5, "composite_score")[
        ["title", "shop_name", "price", "composite_score"]
    ].to_dict("records")

    cats = (
        scored["category"].value_counts().head(5).to_dict()
        if "category" in scored.columns
        else {}
    )

    km = cls_stats["kmeans"]
    cd = km["clusters"]
    top_rule = rules.nlargest(1, "lift").iloc[0]

    return {
        "catalogue": {
            "n_produits": len(scored),
            "n_boutiques": len(shops),
            "prix_median": round(float(scored["price"].median()), 2),
            "pct_disponibles": round(float(scored["available"].mean()) * 100, 1),
            "pct_en_promo": round(float((scored["discount_pct"] > 10).mean()) * 100, 1),
            "top_categories": cats,
        },
        "classement_boutiques": [
            {
                "rang": i + 1,
                "nom": row["shop_name"],
                "score_moyen": round(float(row["score_moyen"]), 3),
                "prix_moyen": round(float(row["prix_moyen"]), 2),
                "note_moyenne": round(float(row["note_moyenne"]), 2),
            }
            for i, (_, row) in enumerate(shops.head(5).iterrows())
        ],
        "top_5_produits": [
            {
                "titre": p["title"],
                "boutique": p["shop_name"],
                "prix": p["price"],
                "score": round(float(p["composite_score"]), 4),
            }
            for p in top5p
        ],
        "modeles_ml": {
            "random_forest_accuracy": round(rf_acc * 100, 1),
            "xgboost_accuracy": round(xgb_acc * 100, 1),
        },
        "segmentation_kmeans": {
            "silhouette_score": km["silhouette_score"],
            "k": km["k"],
            "clusters": [
                {
                    "cluster": d["cluster_kmeans"],
                    "prix_moyen": round(d["prix_moyen"], 0),
                    "note_moyenne": round(d["note_moyenne"], 2),
                    "remise_moy": round(d["remise_moy"], 0),
                    "nb_produits": d["nb_produits"],
                }
                for d in cd
            ],
            "anomalies_dbscan": cls_stats["dbscan"]["nb_anomalies"],
        },
        "regles_association": {
            "total_regles": len(rules),
            "regle_lift_max": {
                "antecedents": str(top_rule["antecedents"])[:60],
                "consequents": str(top_rule["consequents"])[:60],
                "lift": round(float(top_rule["lift"]), 2),
                "confidence_pct": round(float(top_rule["confidence"]) * 100, 1),
            },
        },
    }


@pytest.fixture
def full_context(
    sample_products_df,
    sample_shop_ranking_df,
    sample_rules_df,
    sample_ml_metrics,
    sample_clustering_stats,
):
    return build_context(
        topk=sample_products_df,
        shops=sample_shop_ranking_df,
        scored=sample_products_df,
        rules=sample_rules_df,
        ml=sample_ml_metrics,
        cls_stats=sample_clustering_stats,
    )


# ── Tests Structure du Contexte ───────────────────────────────────────────────


class TestContextStructure:
    def test_required_top_level_keys(self, full_context):
        required = [
            "catalogue",
            "classement_boutiques",
            "top_5_produits",
            "modeles_ml",
            "segmentation_kmeans",
            "regles_association",
        ]
        for key in required:
            assert key in full_context, f"Clé manquante : '{key}'"

    def test_catalogue_has_required_fields(self, full_context):
        cat = full_context["catalogue"]
        for field in [
            "n_produits",
            "n_boutiques",
            "prix_median",
            "pct_disponibles",
            "pct_en_promo",
        ]:
            assert field in cat

    def test_n_produits_positive(self, full_context):
        assert full_context["catalogue"]["n_produits"] > 0

    def test_n_boutiques_positive(self, full_context):
        assert full_context["catalogue"]["n_boutiques"] > 0

    def test_prix_median_positive(self, full_context):
        assert full_context["catalogue"]["prix_median"] > 0

    def test_pct_disponibles_between_0_and_100(self, full_context):
        pct = full_context["catalogue"]["pct_disponibles"]
        assert 0 <= pct <= 100

    def test_top_5_produits_max_5(self, full_context):
        assert len(full_context["top_5_produits"]) <= 5

    def test_top_5_produits_has_required_fields(self, full_context):
        for p in full_context["top_5_produits"]:
            assert "titre" in p
            assert "boutique" in p
            assert "prix" in p
            assert "score" in p

    def test_boutiques_have_required_fields(self, full_context):
        for b in full_context["classement_boutiques"]:
            assert "rang" in b
            assert "nom" in b
            assert "score_moyen" in b

    def test_boutiques_ranked_correctly(self, full_context):
        ranks = [b["rang"] for b in full_context["classement_boutiques"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_ml_accuracies_present(self, full_context):
        ml = full_context["modeles_ml"]
        assert "random_forest_accuracy" in ml
        assert "xgboost_accuracy" in ml

    def test_ml_accuracies_above_80(self, full_context):
        ml = full_context["modeles_ml"]
        assert ml["random_forest_accuracy"] >= 80
        assert ml["xgboost_accuracy"] >= 80


# ── Tests Isolation LLM (architecture MCP) ───────────────────────────────────


class TestLLMIsolation:
    def test_context_is_json_serializable(self, full_context):
        """Le contexte doit être sérialisable en JSON sans erreur."""
        try:
            serialized = json.dumps(full_context, ensure_ascii=False)
            assert len(serialized) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"Context non sérialisable : {e}")

    def test_context_contains_no_raw_rows(self, full_context):
        """Le contexte ne doit pas contenir toutes les colonnes brutes."""
        serialized = json.dumps(full_context)
        # Les clés de données brutes ne doivent pas apparaître
        forbidden = ["product_id", "review_count", "source_platform"]
        for key in forbidden:
            assert key not in serialized, (
                f"Clé brute '{key}' trouvée dans le contexte LLM"
            )

    def test_top_5_scores_between_0_and_1(self, full_context):
        for p in full_context["top_5_produits"]:
            assert 0 <= p["score"] <= 1

    def test_regles_association_fields(self, full_context):
        r = full_context["regles_association"]
        assert "total_regles" in r
        assert "regle_lift_max" in r
        assert r["total_regles"] > 0

    def test_lift_max_above_1(self, full_context):
        lift = full_context["regles_association"]["regle_lift_max"]["lift"]
        assert lift > 1, "La règle avec le lift max doit avoir lift > 1"

    def test_context_size_reasonable(self, full_context):
        """Le contexte tronqué à 3000 chars doit rester utile."""
        serialized = json.dumps(full_context, ensure_ascii=False)
        truncated = serialized[:3000]
        assert len(truncated) >= 500, "Contexte trop petit pour être utile"
