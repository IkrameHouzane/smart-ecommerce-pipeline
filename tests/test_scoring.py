"""
Tests — Scoring & Top-K ranking
Vérifie que le pipeline de scoring produit des résultats cohérents et reproductibles.
"""

import pandas as pd


# ── Helpers (simuler la logique de scoring sans dépendre des vrais fichiers) ──


def compute_score(df: pd.DataFrame) -> pd.Series:
    """
    Réplique la formule de scoring composite utilisée dans le pipeline.
    score = 0.35*rating_norm + 0.30*reviews_norm + 0.20*availability + 0.15*discount_norm
    """

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)

    rating_norm = norm(df["rating"].fillna(0))
    reviews_norm = norm(df["review_count"].fillna(0))
    availability = df["available"].fillna(False).astype(float)
    discount_norm = norm(df["discount_pct"].fillna(0))

    return (
        0.35 * rating_norm
        + 0.30 * reviews_norm
        + 0.20 * availability
        + 0.15 * discount_norm
    ).round(4)


def top_k(
    df: pd.DataFrame, k: int = 3, score_col: str = "composite_score"
) -> pd.DataFrame:
    """Retourne les k meilleurs produits triés par score décroissant."""
    return df.nlargest(k, score_col).reset_index(drop=True)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestScoringFormula:
    def test_scores_between_0_and_1(self, sample_products_df):
        scores = compute_score(sample_products_df)
        assert scores.between(0, 1).all(), "Tous les scores doivent être entre 0 et 1"

    def test_scores_not_all_equal(self, sample_products_df):
        scores = compute_score(sample_products_df)
        assert scores.nunique() > 1, "Les scores ne doivent pas être tous identiques"

    def test_high_rating_boosts_score(self, sample_products_df):
        """Un produit avec note 4.7 doit scorer plus haut qu'un produit note 3.8."""
        scores = compute_score(sample_products_df)
        idx_high = sample_products_df["rating"].idxmax()
        idx_low = sample_products_df["rating"].idxmin()
        assert scores[idx_high] > scores[idx_low]

    def test_unavailable_product_penalized(self):
        """Un produit indisponible doit scorer moins qu'un produit disponible, toutes choses égales."""
        df = pd.DataFrame(
            {
                "rating": [4.5, 4.5],
                "review_count": [100, 100],
                "available": [True, False],
                "discount_pct": [10.0, 10.0],
            }
        )
        scores = compute_score(df)
        assert scores[0] > scores[1], "Produit disponible doit scorer plus haut"

    def test_score_length_matches_input(self, sample_products_df):
        scores = compute_score(sample_products_df)
        assert len(scores) == len(sample_products_df)

    def test_handles_missing_values(self):
        """Le scoring doit fonctionner même avec des NaN."""
        df = pd.DataFrame(
            {
                "rating": [None, 4.2, 4.5],
                "review_count": [50, None, 200],
                "available": [True, True, None],
                "discount_pct": [0.0, 10.0, None],
            }
        )
        scores = compute_score(df)
        assert not scores.isna().any(), (
            "Aucun score NaN même avec des valeurs manquantes"
        )

    def test_discount_contributes_positively(self):
        """Une remise plus élevée doit améliorer le score."""
        df = pd.DataFrame(
            {
                "rating": [4.0, 4.0],
                "review_count": [100, 100],
                "available": [True, True],
                "discount_pct": [50.0, 0.0],
            }
        )
        scores = compute_score(df)
        assert scores[0] > scores[1], "Remise élevée doit booster le score"


class TestTopKRanking:
    def test_top_k_returns_correct_count(self, sample_products_df):
        result = top_k(sample_products_df, k=3)
        assert len(result) == 3

    def test_top_k_sorted_descending(self, sample_products_df):
        result = top_k(sample_products_df, k=3)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True), (
            "Top-K doit être trié par score décroissant"
        )

    def test_top_1_is_best_product(self, sample_products_df):
        result = top_k(sample_products_df, k=1)
        assert (
            result.iloc[0]["composite_score"]
            == sample_products_df["composite_score"].max()
        )

    def test_top_k_with_k_larger_than_df(self, sample_products_df):
        """k > len(df) ne doit pas lever d'erreur."""
        result = top_k(sample_products_df, k=100)
        assert len(result) == len(sample_products_df)

    def test_top_k_preserves_columns(self, sample_products_df):
        result = top_k(sample_products_df, k=2)
        for col in sample_products_df.columns:
            assert col in result.columns

    def test_top_k_index_reset(self, sample_products_df):
        result = top_k(sample_products_df, k=3)
        assert list(result.index) == [0, 1, 2], "Index doit être réinitialisé"


class TestShopRanking:
    def test_shop_ranking_sorted(self, sample_shop_ranking_df):
        """Le classement boutiques doit être trié par score décroissant."""
        scores = sample_shop_ranking_df["score_moyen"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_best_shop_is_first(self, sample_shop_ranking_df):
        best = sample_shop_ranking_df.iloc[0]["shop_name"]
        max_score = sample_shop_ranking_df["score_moyen"].max()
        assert sample_shop_ranking_df.iloc[0]["score_moyen"] == max_score
        assert isinstance(best, str) and len(best) > 0

    def test_shop_scores_positive(self, sample_shop_ranking_df):
        assert (sample_shop_ranking_df["score_moyen"] > 0).all()

    def test_shop_ranking_has_required_columns(self, sample_shop_ranking_df):
        required = ["shop_name", "score_moyen", "prix_moyen", "note_moyenne"]
        for col in required:
            assert col in sample_shop_ranking_df.columns
