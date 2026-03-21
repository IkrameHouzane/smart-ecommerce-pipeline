# data_processing/scoring.py — VERSION FINALE COMPLÈTE
# ─────────────────────────────────────────────────────────
# Corrections v5 :
#   - Bris d'égalité double : catalogue_richness + prix_inverse
#     → différencie les produits avec même note ET mêmes variantes
#       (cas allbirds : tous 13 variantes, même note imputée)
#   - Déduplication par (title_racine, shop_name)
#   - Score composite pondéré (note 40%, pop 25%, dispo 20%, remise 15%)
#   - Quota par boutique : max 10 produits dans le Top-K
#   - generer_shop_ranking()       → analytics/shop_ranking.csv
#   - generer_topk_par_categorie() → analytics/topk_per_category.csv
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os

INPUT_PATH = "../data/featured_products.parquet"
OUTPUT_FULL = "../data/scored_products.parquet"
OUTPUT_TOPK = "../data/top_k_products.csv"

K = 50  # taille totale du Top-K
MAX_PAR_SHOP = 10  # max produits par boutique dans le Top-K


# ══════════════════════════════════════════════════════════
#  SCORE COMPOSITE
# ══════════════════════════════════════════════════════════


def calculer_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attribue un score composite à chaque produit.

    Poids principaux :
      note (40%)          — qualité perçue par les acheteurs
      popularité (25%)    — signal variantes plafonné à 10
      disponibilité (20%) — produit doit être en stock
      remise (15%)        — attractivité prix, plafonnée à 50%

    Double bris d'égalité (+0 à +0.04 max) :
      catalogue_richness (+0.02 max) — richesse catalogue (variantes + images)
      prix_inverse_norm  (+0.02 max) — favorise les prix bas à qualité égale

    Ce double bris d'égalité résout le cas fréquent où plusieurs
    produits d'une même boutique partagent la même note imputée
    ET le même nombre de variantes (ex: allbirds 13 variantes, 4.4★).
    """

    # ── Note normalisée (40%) ─────────────────────────────
    note_norm = df["rating_filled_norm"].fillna(0)

    # ── Popularité (25%) ──────────────────────────────────
    signal_variantes = (df["nb_variants"].clip(upper=10) / 10.0) * 5
    signal_avis = np.log1p(df["nb_reviews"].fillna(0))
    a_des_avis = (df["nb_reviews"].fillna(0) > 0).astype(float)

    signal_pop = (
        a_des_avis * (signal_avis * 0.7 + signal_variantes * 0.3)
        + (1 - a_des_avis) * signal_variantes
    )
    pop_max = signal_pop.max()
    pop_norm = (signal_pop / pop_max) if pop_max > 0 else signal_pop

    # ── Disponibilité (20%) ───────────────────────────────
    dispo = df["available"].astype(float)

    # ── Remise plafonnée (15%) ────────────────────────────
    remise_plafonnee = df["discount_pct"].clip(upper=50) / 50.0

    # ── Score principal ───────────────────────────────────
    score_principal = (
        note_norm * 0.40 + pop_norm * 0.25 + dispo * 0.20 + remise_plafonnee * 0.15
    )

    # ── Bris d'égalité 1 : richesse catalogue (+0 à +0.02) ──
    # catalogue_richness est entre 0 et 1 (calculé dans features.py)
    richesse_norm = df["catalogue_richness"].fillna(0)

    # ── Bris d'égalité 2 : prix inversé (+0 à +0.02) ─────
    # À qualité égale, un produit moins cher est préférable.
    # price_norm=0 → pas cher → prix_inverse=1.0 → bonus max
    # price_norm=1 → très cher → prix_inverse=0.0 → pas de bonus
    prix_norm = df["price_norm"].fillna(0.5)
    prix_inverse = 1.0 - prix_norm

    # ── Score final avec double bris d'égalité ───────────
    df["composite_score"] = (
        score_principal + richesse_norm * 0.02 + prix_inverse * 0.02
    ).round(4)

    return df


# ══════════════════════════════════════════════════════════
#  DÉDUPLICATION DES VARIANTES
# ══════════════════════════════════════════════════════════


def dedupliquer_produits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les doublons de variantes avant le scoring.

    Les titres Shopify incluent souvent la couleur :
      "Women's Wool Runner NZ - Rugged Beige"
      "Women's Wool Runner NZ - Dark Grey"

    On extrait la racine du titre (avant " - " ou " (") et on
    garde un seul produit par (title_racine, shop_name) :
      1. disponible en priorité
      2. plus de variantes
      3. prix le plus bas
    """
    avant = len(df)

    def extraire_racine(titre):
        titre = str(titre)
        titre = titre.split(" - ")[0]
        titre = titre.split(" (")[0]
        return titre.strip()

    df = df.copy()
    df["title_racine"] = df["title"].apply(extraire_racine)

    df_sorted = df.sort_values(
        ["available", "nb_variants", "price"], ascending=[False, False, True]
    )

    df_dedup = df_sorted.drop_duplicates(
        subset=["title_racine", "shop_name"], keep="first"
    )

    apres = len(df_dedup)
    print(
        f"  → Déduplication : {avant} → {apres} produits "
        f"({avant - apres} variantes supprimées)"
    )

    return df_dedup


# ══════════════════════════════════════════════════════════
#  EXTRACTION TOP-K DIVERSIFIÉ
# ══════════════════════════════════════════════════════════


def extraire_topk_diversifie(
    df: pd.DataFrame, k: int, max_par_shop: int
) -> pd.DataFrame:
    """
    Extrait le Top-K en garantissant la diversité par boutique.
    Une boutique ne peut pas dépasser max_par_shop produits.
    """
    df_trie = df.sort_values("composite_score", ascending=False)
    compteur = {}
    selectionnes = []

    for _, row in df_trie.iterrows():
        shop = row["shop_name"]
        if compteur.get(shop, 0) >= max_par_shop:
            continue
        selectionnes.append(row)
        compteur[shop] = compteur.get(shop, 0) + 1
        if len(selectionnes) == k:
            break

    top_k = pd.DataFrame(selectionnes).reset_index(drop=True)
    top_k["rank"] = range(1, len(top_k) + 1)
    return top_k


# ══════════════════════════════════════════════════════════
#  CLASSEMENT DES SHOPS
# ══════════════════════════════════════════════════════════


def generer_shop_ranking(df: pd.DataFrame, output_dir: str = "../analytics"):
    """
    Génère le classement des shops avec produit phare et géographie.
    Fichier : analytics/shop_ranking.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    shop_ranking = (
        df.groupby("shop_name")
        .agg(
            geography=("geography", lambda x: x.mode()[0] if len(x) > 0 else "US"),
            nb_produits=("title", "count"),
            score_moyen=("composite_score", "mean"),
            prix_moyen=("price", "mean"),
            note_moyenne=("rating_filled", "mean"),
            pct_dispo=("available", "mean"),
            remise_moyenne=("discount_pct", "mean"),
        )
        .round(3)
        .sort_values("score_moyen", ascending=False)
        .reset_index()
    )
    shop_ranking["rank_shop"] = range(1, len(shop_ranking) + 1)

    produit_phare = (
        df.sort_values("composite_score", ascending=False)
        .drop_duplicates(subset=["shop_name"])[
            ["shop_name", "title", "composite_score"]
        ]
        .rename(columns={"title": "produit_phare", "composite_score": "score_phare"})
    )
    shop_ranking = shop_ranking.merge(produit_phare, on="shop_name", how="left")

    cols = [
        "rank_shop",
        "shop_name",
        "geography",
        "nb_produits",
        "score_moyen",
        "note_moyenne",
        "prix_moyen",
        "pct_dispo",
        "remise_moyenne",
        "produit_phare",
    ]
    shop_ranking[[c for c in cols if c in shop_ranking.columns]].to_csv(
        f"{output_dir}/shop_ranking.csv", index=False, encoding="utf-8-sig"
    )

    print("\n  Classement des shops :")
    print(f"  {'─' * 65}")
    for _, row in shop_ranking.iterrows():
        phare = (
            str(row.get("produit_phare", "N/A"))[:30]
            if pd.notna(row.get("produit_phare"))
            else "N/A"
        )
        print(
            f"  #{int(row['rank_shop']):2d} | "
            f"{str(row['shop_name']):<20} | "
            f"{str(row.get('geography', '?')):<4} | "
            f"score: {row['score_moyen']:.3f} | "
            f"{phare}"
        )
    print(f"\n  Sauvegardé : {output_dir}/shop_ranking.csv")
    return shop_ranking


# ══════════════════════════════════════════════════════════
#  TOP-K PAR CATÉGORIE
# ══════════════════════════════════════════════════════════


def generer_topk_par_categorie(
    df: pd.DataFrame, k: int = 10, output_dir: str = "../analytics"
):
    """
    Génère le Top-K produits par catégorie.
    Fichier : analytics/topk_per_category.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    if "category_clean" not in df.columns or "composite_score" not in df.columns:
        print("  ⚠️  Colonnes manquantes — skipped")
        return pd.DataFrame()

    chunks = [
        group.nlargest(k, "composite_score")
        for _, group in df.groupby("category_clean", sort=False)
        if len(group) >= 3
    ]

    if not chunks:
        print("  ⚠️  Aucune catégorie avec assez de produits")
        return pd.DataFrame()

    topk_cat = pd.concat(chunks, ignore_index=True)

    cols = [
        "category_clean",
        "rank_global",
        "title",
        "shop_name",
        "source",
        "price",
        "price_tier",
        "rating_filled",
        "composite_score",
        "discount_pct",
        "available",
        "nb_reviews",
        "nb_variants",
        "product_url",
    ]
    cols_ok = [c for c in cols if c in topk_cat.columns]
    topk_cat[cols_ok].to_csv(
        f"{output_dir}/topk_per_category.csv", index=False, encoding="utf-8-sig"
    )

    nb_categories = topk_cat["category_clean"].nunique()
    print(
        f"  → topk_per_category.csv : "
        f"{nb_categories} catégories × top-{k} "
        f"({len(topk_cat)} lignes total)"
    )
    print(f"  Sauvegardé : {output_dir}/topk_per_category.csv")
    return topk_cat


# ══════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════


def scoring_topk(
    input_path=INPUT_PATH,
    output_full=OUTPUT_FULL,
    output_topk=OUTPUT_TOPK,
    k=K,
    max_par_shop=MAX_PAR_SHOP,
):
    """
    Lance le scoring complet :
      1. Déduplication des variantes
      2. Score composite avec double bris d'égalité
      3. Extraction Top-K diversifié
      4. Classement des shops
      5. Top-K par catégorie
      6. Rapport + sauvegarde
    """

    print("=" * 55)
    print("  ÉTAPE 2C — SCORING TOP-K")
    print("=" * 55)

    df = pd.read_parquet(input_path)
    print(f"\n  Produits chargés : {len(df)}")

    print("\n[1/6] Déduplication des variantes...")
    df = dedupliquer_produits(df)

    print("\n[2/6] Calcul du score composite...")
    df = calculer_score(df)

    df = df.sort_values("composite_score", ascending=False)
    df["rank_global"] = range(1, len(df) + 1)

    print(f"  → Score moyen  : {df['composite_score'].mean():.4f}")
    print(f"  → Score max    : {df['composite_score'].max():.4f}")
    print(f"  → Score min    : {df['composite_score'].min():.4f}")
    print(f"  → Écart-type   : {df['composite_score'].std():.4f}")

    nb_identiques = df.duplicated(subset=["composite_score"]).sum()
    print(
        f"  → Scores non-uniques : {nb_identiques} "
        f"({nb_identiques / len(df) * 100:.1f}%)"
    )

    print(f"\n[3/6] Extraction Top-{k} (max {max_par_shop} par boutique)...")
    top_k = extraire_topk_diversifie(df, k, max_par_shop)

    analytics_dir = os.path.normpath(
        os.path.join(os.path.dirname(output_full), "..", "analytics")
    )

    print("\n[4/6] Classement des shops...")
    generer_shop_ranking(df, output_dir=analytics_dir)

    print("\n[5/6] Top-K par catégorie...")
    generer_topk_par_categorie(df, k=10, output_dir=analytics_dir)

    print("\n[6/6] Rapport et sauvegarde...")

    colonnes_topk = [
        "rank",
        "title",
        "shop_name",
        "source",
        "category_clean",
        "price",
        "price_tier",
        "discount_pct",
        "rating_filled",
        "has_rating",
        "nb_reviews",
        "nb_variants",
        "available",
        "popularity_score",
        "catalogue_richness",
        "composite_score",
        "product_url",
    ]
    cols = [c for c in colonnes_topk if c in top_k.columns]
    top_k_propre = top_k[cols]

    print(f"\n  TOP 10 (sur {k}) :")
    print("  " + "─" * 72)
    for _, row in top_k_propre.head(10).iterrows():
        note_str = f"{row['rating_filled']:.1f}★"
        pop_str = (
            f"{int(row['nb_reviews'])} avis"
            if row.get("nb_reviews", 0) > 0
            else f"{int(row.get('nb_variants', 1))} var."
        )
        print(
            f"  #{int(row['rank']):2d} | "
            f"{str(row['title'])[:28]:<28} | "
            f"${row['price']:6.2f} | "
            f"{note_str} | "
            f"{pop_str:<8} | "
            f"score: {row['composite_score']:.4f}"
        )

    print(f"\n  Répartition Top-{k} par boutique :")
    print(top_k["shop_name"].value_counts().to_string())

    if "price_tier" in top_k.columns:
        print(f"\n  Répartition Top-{k} par tier de prix :")
        print(top_k["price_tier"].value_counts().to_string())

    print(f"\n  Métriques du Top-{k} :")
    print(f"    Prix moyen         : ${top_k['price'].mean():.2f}")
    print(f"    Prix médian        : ${top_k['price'].median():.2f}")
    print(f"    Note moyenne       : {top_k['rating_filled'].mean():.2f}/5")
    print(f"    Remise moyenne     : {top_k['discount_pct'].mean():.1f}%")
    print(f"    Nb variantes moyen : {top_k['nb_variants'].mean():.1f}")
    print(f"    Disponibilité      : {top_k['available'].mean() * 100:.1f}%")

    os.makedirs(os.path.dirname(output_full), exist_ok=True)
    df.to_parquet(output_full, index=False)
    top_k_propre.to_csv(output_topk, index=False, encoding="utf-8-sig")

    print("\n  Fichiers générés :")
    print(f"    {output_full}")
    print(f"    {output_topk}")
    print(f"    {analytics_dir}/shop_ranking.csv")
    print(f"    {analytics_dir}/topk_per_category.csv")
    print("=" * 55)

    return df, top_k_propre


if __name__ == "__main__":
    df, top_k = scoring_topk()
    print(f"\n✅ Scoring terminé — Top-{K} extrait dans top_k_products.csv")
