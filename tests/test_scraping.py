"""
Tests — Scraping Output Validation
Vérifie la structure et la qualité des données extraites par les agents A2A.
"""

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

REQUIRED_PRODUCT_COLUMNS = [
    "product_id",
    "title",
    "shop_name",
    "price",
    "available",
    "source_platform",
]

VALID_PLATFORMS = {"shopify", "woocommerce"}


def validate_product_schema(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des erreurs de schéma."""
    errors = []
    for col in REQUIRED_PRODUCT_COLUMNS:
        if col not in df.columns:
            errors.append(f"Colonne manquante : '{col}'")
    return errors


def validate_platform_values(df: pd.DataFrame) -> pd.Series:
    """Retourne un masque des lignes avec plateforme invalide."""
    if "source_platform" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["source_platform"].isin(VALID_PLATFORMS)


def validate_prices(df: pd.DataFrame) -> pd.Series:
    """Retourne un masque des lignes avec prix valide (>=0 ou NaN acceptable)."""
    if "price" not in df.columns:
        return pd.Series(True, index=df.index)
    prices = pd.to_numeric(df["price"], errors="coerce")
    return prices.isna() | (prices >= 0)


# ── Tests Schéma ──────────────────────────────────────────────────────────────


class TestProductSchema:
    def test_required_columns_present(self, sample_products_df):
        errors = validate_product_schema(sample_products_df)
        assert errors == [], f"Erreurs de schéma : {errors}"

    def test_product_ids_unique(self, sample_products_df):
        assert sample_products_df["product_id"].nunique() == len(sample_products_df)

    def test_titles_not_empty(self, sample_products_df):
        assert sample_products_df["title"].notna().all()
        assert (sample_products_df["title"].str.strip() != "").all()

    def test_shop_names_not_empty(self, sample_products_df):
        assert sample_products_df["shop_name"].notna().all()

    def test_platform_values_valid(self, sample_products_df):
        mask = validate_platform_values(sample_products_df)
        invalid = sample_products_df[~mask]["source_platform"].tolist()
        assert len(invalid) == 0, f"Plateformes invalides : {invalid}"

    def test_prices_non_negative(self, sample_products_df):
        mask = validate_prices(sample_products_df)
        assert mask.all(), "Des prix négatifs ont été détectés"

    def test_availability_is_boolean(self, sample_products_df):
        """Accepte bool Python et numpy.bool_ (pandas retourne numpy.bool_)."""
        import numpy as np

        vals = sample_products_df["available"].dropna().unique()
        for v in vals:
            assert isinstance(v, (bool, int, np.bool_)), (
                f"Valeur disponibilité invalide : {v} (type: {type(v)})"
            )


# ── Tests Qualité des Données ─────────────────────────────────────────────────


class TestDataQuality:
    def test_no_fully_empty_rows(self, sample_products_df):
        """Aucune ligne ne doit être entièrement vide."""
        empty_mask = sample_products_df.isna().all(axis=1)
        assert not empty_mask.any()

    def test_price_coverage_acceptable(self, sample_products_df):
        """Au moins 70% des produits doivent avoir un prix."""
        coverage = sample_products_df["price"].notna().mean()
        assert coverage >= 0.70, f"Couverture prix insuffisante : {coverage:.1%}"

    def test_title_length_reasonable(self, sample_products_df):
        """Les titres doivent avoir entre 2 et 500 caractères."""
        lengths = sample_products_df["title"].dropna().str.len()
        assert (lengths >= 2).all(), "Certains titres sont trop courts"
        assert (lengths <= 500).all(), "Certains titres sont trop longs"

    def test_shop_diversity(self, sample_products_df):
        """Au moins 2 boutiques différentes."""
        assert sample_products_df["shop_name"].nunique() >= 2

    def test_platform_diversity(self, sample_products_df):
        """Au moins une plateforme Shopify ou WooCommerce."""
        platforms = set(sample_products_df["source_platform"].dropna().unique())
        assert len(platforms & VALID_PLATFORMS) >= 1

    def test_rating_range_valid(self, sample_products_df):
        """Les notes doivent être entre 0 et 5."""
        ratings = sample_products_df["rating"].dropna()
        assert (ratings >= 0).all() and (ratings <= 5).all()

    def test_discount_pct_range(self, sample_products_df):
        """Les remises doivent être entre 0 et 100%."""
        discounts = sample_products_df["discount_pct"].dropna()
        assert (discounts >= 0).all() and (discounts <= 100).all()


# ── Tests Multi-plateforme ────────────────────────────────────────────────────


class TestMultiPlatform:
    def test_shopify_products_present(self, sample_products_df):
        shopify = sample_products_df[sample_products_df["source_platform"] == "shopify"]
        assert len(shopify) > 0, "Aucun produit Shopify dans le dataset"

    def test_woocommerce_products_present(self, sample_products_df):
        woo = sample_products_df[sample_products_df["source_platform"] == "woocommerce"]
        assert len(woo) > 0, "Aucun produit WooCommerce dans le dataset"

    def test_shopify_has_valid_prices(self, sample_products_df):
        shopify = sample_products_df[sample_products_df["source_platform"] == "shopify"]
        prices = pd.to_numeric(shopify["price"], errors="coerce")
        assert (prices.dropna() >= 0).all()

    def test_both_platforms_have_titles(self, sample_products_df):
        for platform in ["shopify", "woocommerce"]:
            subset = sample_products_df[
                sample_products_df["source_platform"] == platform
            ]
            assert subset["title"].notna().all(), f"Titres manquants pour {platform}"
