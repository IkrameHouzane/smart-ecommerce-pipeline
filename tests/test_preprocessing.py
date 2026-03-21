"""
Tests — Preprocessing & Feature Engineering
Vérifie le nettoyage des données, la validation et la construction des features.
"""

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────


def clean_price(series: pd.Series) -> pd.Series:
    """Nettoie et normalise les prix."""
    return pd.to_numeric(series, errors="coerce").clip(lower=0)


def clean_rating(series: pd.Series) -> pd.Series:
    """Nettoie les notes — doit être entre 0 et 5."""
    return pd.to_numeric(series, errors="coerce").clip(lower=0, upper=5)


def fill_rating(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Remplace les NaN de rating par une valeur par défaut."""
    return series.fillna(default)


def compute_discount_pct(price: pd.Series, compare_price: pd.Series) -> pd.Series:
    """Calcule le pourcentage de remise."""
    mask = (compare_price > price) & (compare_price > 0)
    result = pd.Series(0.0, index=price.index)
    result[mask] = (
        (compare_price[mask] - price[mask]) / compare_price[mask] * 100
    ).round(2)
    return result


def flag_available(series: pd.Series) -> pd.Series:
    """Convertit la disponibilité en booléen."""
    return series.map(lambda x: bool(x) if pd.notna(x) else False)


def deduplicate(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Supprime les doublons."""
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def normalize_shop_name(series: pd.Series) -> pd.Series:
    """Normalise les noms de boutiques."""
    return series.str.lower().str.strip().str.replace(r"\s+", "_", regex=True)


# ── Tests Nettoyage ───────────────────────────────────────────────────────────


class TestPriceCleaning:
    def test_string_prices_converted(self):
        s = pd.Series(["52.99", "39.0", "invalid", None])
        result = clean_price(s)
        assert result[0] == 52.99
        assert result[1] == 39.0
        assert pd.isna(result[2])
        assert pd.isna(result[3])

    def test_negative_prices_clipped(self):
        s = pd.Series([-10.0, 0.0, 50.0])
        result = clean_price(s)
        assert result[0] == 0.0

    def test_valid_prices_unchanged(self):
        s = pd.Series([52.0, 39.0, 89.0])
        result = clean_price(s)
        pd.testing.assert_series_equal(result, s)

    def test_all_nan_returns_nan_series(self):
        s = pd.Series([None, None, None])
        result = clean_price(s)
        assert result.isna().all()


class TestRatingCleaning:
    def test_ratings_clipped_to_5(self):
        s = pd.Series([6.0, 4.5, 0.0, -1.0])
        result = clean_rating(s)
        assert result[0] == 5.0
        assert result[2] == 0.0
        assert result[3] == 0.0

    def test_valid_ratings_preserved(self):
        s = pd.Series([4.4, 4.2, 3.8])
        result = clean_rating(s)
        pd.testing.assert_series_equal(result, s)

    def test_fill_rating_replaces_nan(self):
        s = pd.Series([4.5, None, 3.8])
        result = fill_rating(s, default=0.0)
        assert result[1] == 0.0
        assert not result.isna().any()


class TestDiscountCalculation:
    def test_basic_discount(self):
        price = pd.Series([39.0])
        compare = pd.Series([65.0])
        result = compute_discount_pct(price, compare)
        expected = round((65 - 39) / 65 * 100, 2)
        assert abs(result[0] - expected) < 0.01

    def test_no_discount_when_price_equals_compare(self):
        price = pd.Series([50.0])
        compare = pd.Series([50.0])
        result = compute_discount_pct(price, compare)
        assert result[0] == 0.0

    def test_no_discount_when_compare_lower(self):
        price = pd.Series([80.0])
        compare = pd.Series([50.0])
        result = compute_discount_pct(price, compare)
        assert result[0] == 0.0

    def test_zero_compare_price_no_crash(self):
        price = pd.Series([50.0])
        compare = pd.Series([0.0])
        result = compute_discount_pct(price, compare)
        assert result[0] == 0.0

    def test_discount_pct_between_0_and_100(self):
        price = pd.Series([10.0, 50.0, 99.0])
        compare = pd.Series([100.0, 100.0, 100.0])
        result = compute_discount_pct(price, compare)
        assert (result >= 0).all() and (result <= 100).all()


class TestAvailability:
    def test_true_flagged_correctly(self):
        s = pd.Series([True, False, None, 1, 0])
        result = flag_available(s)
        assert result[0]
        assert not result[1]
        assert not result[2]


class TestDeduplication:
    def test_removes_exact_duplicates(self, sample_products_df):
        df_duped = pd.concat([sample_products_df, sample_products_df]).reset_index(
            drop=True
        )
        result = deduplicate(df_duped, subset=["product_id"])
        assert len(result) == len(sample_products_df)

    def test_preserves_unique_rows(self, sample_products_df):
        result = deduplicate(sample_products_df, subset=["product_id"])
        assert len(result) == len(sample_products_df)

    def test_index_reset_after_dedup(self, sample_products_df):
        df_duped = pd.concat([sample_products_df, sample_products_df]).reset_index(
            drop=True
        )
        result = deduplicate(df_duped, subset=["product_id"])
        assert list(result.index) == list(range(len(result)))


class TestShopNameNormalization:
    def test_lowercase(self):
        s = pd.Series(["AllBirds", "NOBULL", "Born Primitive"])
        result = normalize_shop_name(s)
        assert result[0] == "allbirds"
        assert result[1] == "nobull"

    def test_spaces_replaced_with_underscore(self):
        s = pd.Series(["Born Primitive"])
        result = normalize_shop_name(s)
        assert result[0] == "born_primitive"

    def test_strips_whitespace(self):
        s = pd.Series(["  allbirds  "])
        result = normalize_shop_name(s)
        assert result[0] == "allbirds"
