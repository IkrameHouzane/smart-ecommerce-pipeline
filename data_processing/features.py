# data_processing/features.py
# ─────────────────────────────────────────────────────────
# ÉTAPE 2B — Feature Engineering.
#
# CE FICHIER CRÉE de nouvelles colonnes (features) à partir
# des données propres. Ces features sont les "ingrédients"
# que les algorithmes ML vont utiliser.
#
# VERSION CORRIGÉE :
#   - Popularité utilise nb_variants comme proxy quand
#     nb_reviews = 0 (cas Shopify)
#   - Imputation des notes par médiane de vraies notes
#     collectées via scraping HTML
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

INPUT_PATH = "../data/clean_products.parquet"
OUTPUT_PATH = "../data/featured_products.parquet"


def engineer_features(input_path=INPUT_PATH, output_path=OUTPUT_PATH):

    print("=" * 55)
    print("  ÉTAPE 2B — FEATURE ENGINEERING")
    print("=" * 55)

    df = pd.read_parquet(input_path)
    print(f"\n  Données chargées : {len(df)} produits")

    # ── FEATURE 1 : Pourcentage de remise ─────────────────
    # Formule : (prix_barré - prix_actuel) / prix_barré × 100
    # Exemple : prix barré 60$, prix actuel 40$ → remise 33.3%
    print("\n[1/8] Calcul du pourcentage de remise...")

    df["discount_pct"] = np.where(
        # Condition : prix barré existe ET est supérieur au prix actuel
        (df["compare_price"].notna()) & (df["compare_price"] > df["price"]),
        # Si vrai → calcule la remise
        ((df["compare_price"] - df["price"]) / df["compare_price"] * 100).round(1),
        # Si faux → pas de remise
        0.0,
    )

    print(f"  → Remise moyenne       : {df['discount_pct'].mean():.1f}%")
    print(f"  → Produits avec remise : {(df['discount_pct'] > 0).sum()}")

    # ── FEATURE 2 : Niveau de prix ────────────────────────
    # Divise les produits en 3 segments selon leur prix.
    # pd.qcut() utilise des quantiles (33% dans chaque segment)
    # pour s'adapter à la vraie distribution des prix.
    print("\n[2/8] Calcul du niveau de prix (price_tier)...")

    df["price_tier"] = pd.qcut(
        df["price"], q=3, labels=["budget", "mid_range", "premium"], duplicates="drop"
    )
    print(df["price_tier"].value_counts().to_string())

    # ── FEATURE 3 : Gestion des notes ────────────────────
    # CONTEXTE :
    #   Les APIs Shopify publiques ne fournissent pas les notes.
    #   On a collecté de vraies notes via scraping HTML pour
    #   4 boutiques (allbirds, nobull, represent, goat_tape).
    #   Pour les autres boutiques, on impute la médiane
    #   des vraies notes collectées, par tier de prix.
    print("\n[3/8] Gestion des notes (rating)...")

    # Convertit en nombre — gère les strings "4.50", "0.00", etc.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # 0.0 exact = "pas encore de note" sur WooCommerce → on met NaN
    df.loc[df["rating"] == 0, "rating"] = np.nan

    nb_vraies = df["rating"].notna().sum()
    print(f"  → Vraies notes disponibles : {nb_vraies} produits")

    # has_rating : 1 si vraie note collectée, 0 si imputée
    # Cette colonne sera utilisée par le ML pour distinguer
    # les produits avec note fiable de ceux avec note estimée
    df["has_rating"] = df["rating"].notna().astype(int)

    # Calcule la médiane globale des vraies notes
    note_mediane_globale = df["rating"].median()
    if pd.isna(note_mediane_globale):
        # Fallback : 4.2 = médiane réaliste niche sport
        note_mediane_globale = 4.2
        print(f"  → Aucune vraie note — médiane fixée à {note_mediane_globale}")
    else:
        print(f"  → Médiane des vraies notes : {note_mediane_globale:.2f}")

    # Imputation par groupe (price_tier) :
    # chaque tier reçoit la médiane des vraies notes de ce tier.
    # Si un tier n'a aucune vraie note → médiane globale.
    df["rating_filled"] = df.groupby("price_tier", observed=True)["rating"].transform(
        lambda x: x.fillna(x.median() if x.notna().any() else note_mediane_globale)
    )
    # Remplacement final pour les cas non couverts
    df["rating_filled"] = df["rating_filled"].fillna(note_mediane_globale)

    print(f"  → Note moyenne après imputation : {df['rating_filled'].mean():.2f}/5")
    print("  → Répartition par tier :")
    print(
        df.groupby("price_tier", observed=True)["rating_filled"]
        .mean()
        .round(2)
        .to_string()
    )

    # ── FEATURE 4 : Score de popularité ──────────────────
    # PROBLÈME : nb_reviews = 0 pour tous les produits Shopify
    #   (les avis sont sur des systèmes tiers inaccessibles).
    #   popularity = note × log(0+1) = 0 → pas informatif.
    #
    # SOLUTION : combiner deux signaux complémentaires.
    #
    # Signal 1 — nb_reviews (vrais avis, disponibles WooCommerce)
    # Signal 2 — nb_variants comme PROXY de popularité Shopify.
    #   Justification : un produit avec 15 variantes
    #   (S/M/L/XL × 3 couleurs) est établi sur le marché.
    #   C'est un signal de demande réelle documenté en
    #   e-commerce analytics (Nielsen, 2022).
    print("\n[4/8] Calcul du score de popularité...")

    df["nb_reviews"] = pd.to_numeric(df["nb_reviews"], errors="coerce").fillna(0)
    df["nb_variants"] = pd.to_numeric(df["nb_variants"], errors="coerce").fillna(1)

    # Signal avis réels : log(nb_reviews + 1)
    # log() réduit l'effet des très grands nombres :
    # 10 avis → 2.4, 100 avis → 4.6, 1000 avis → 6.9
    signal_avis = np.log1p(df["nb_reviews"])

    # Signal variantes : normalisé entre 0 et 5 (même échelle que log)
    # Plafonné à 20 variantes (au-delà c'est rare et non représentatif)
    signal_variantes = (df["nb_variants"].clip(upper=20) / 20.0) * 5

    # Combinaison adaptative :
    # - Si le produit a des vrais avis → 70% avis + 30% variantes
    # - Si pas d'avis (Shopify) → 100% variantes
    a_des_avis = (df["nb_reviews"] > 0).astype(float)

    signal_popularite = (
        a_des_avis * (signal_avis * 0.7 + signal_variantes * 0.3)
        + (1 - a_des_avis) * signal_variantes
    )

    # Score final = note imputée × signal de popularité
    df["popularity_score"] = (df["rating_filled"] * signal_popularite).round(4)

    print(f"  → Popularité max               : {df['popularity_score'].max():.2f}")
    print(f"  → Popularité moyenne           : {df['popularity_score'].mean():.2f}")
    print(f"  → Produits avec vrais avis     : {(df['nb_reviews'] > 0).sum()}")
    print(f"  → Produits avec signal variant : {(df['nb_variants'] > 1).sum()}")

    # ── FEATURE 5 : Fraîcheur ─────────────────────────────
    # Calcule le nombre de jours depuis la création et la MAJ.
    # Utile pour détecter les produits récents (tendances).
    print("\n[5/8] Calcul de la fraîcheur...")

    maintenant = pd.Timestamp.now(tz="UTC")

    def calculer_jours(date_str):
        """Convertit une date en nb de jours depuis aujourd'hui."""
        if pd.isna(date_str) or str(date_str).strip() == "":
            return np.nan
        try:
            date = pd.to_datetime(date_str, utc=True, errors="coerce")
            if pd.isna(date):
                return np.nan
            return max(0, (maintenant - date).days)
        except Exception:
            return np.nan

    df["days_since_update"] = df["updated_at"].apply(calculer_jours)
    df["days_since_created"] = df["created_at"].apply(calculer_jours)

    # Remplace les manquants par la médiane
    df["days_since_update"] = df["days_since_update"].fillna(
        df["days_since_update"].median()
    )
    df["days_since_created"] = df["days_since_created"].fillna(
        df["days_since_created"].median()
    )

    print(f"  → Âge moyen des produits : {df['days_since_created'].mean():.0f} jours")
    print(f"  → MAJ moyenne            : {df['days_since_update'].mean():.0f} jours")

    # ── FEATURE 6 : Richesse catalogue ───────────────────
    # Combine nb_variants et nb_images en un score unique.
    # Un produit bien présenté (beaucoup de variantes et d'images)
    # est généralement de meilleure qualité.
    print("\n[6/8] Calcul de la richesse catalogue...")

    df["variants_norm"] = df["nb_variants"].clip(upper=20) / 20
    df["images_norm"] = df["nb_images"].clip(upper=10) / 10

    df["catalogue_richness"] = (
        df["variants_norm"] * 0.6  # variantes : 60% du score
        + df["images_norm"] * 0.4  # images    : 40% du score
    ).round(4)

    print(f"  → Richesse moyenne : {df['catalogue_richness'].mean():.3f}")

    # ── FEATURE 7 : Catégorie harmonisée ─────────────────
    # Les catégories sont incohérentes entre Shopify et WooCommerce.
    # On crée une colonne unifiée en cherchant dans l'ordre :
    # categories → product_type → tags → "unknown"
    print("\n[7/8] Harmonisation des catégories...")

    def harmoniser_categorie(row):
        """Extrait la catégorie principale d'un produit."""
        for champ in ["categories", "product_type", "tags"]:
            val = str(row.get(champ, "") or "").strip().lower()
            if val and val not in ("", "nan", "none", "woocommerce"):
                # Garde le premier mot-clé (avant la virgule)
                return val.split(",")[0].strip()[:50]
        return "unknown"

    df["category_clean"] = df.apply(harmoniser_categorie, axis=1)

    print(f"  → {df['category_clean'].nunique()} catégories distinctes")
    print(f"  → Top 5 : {df['category_clean'].value_counts().head(5).to_dict()}")

    # ── FEATURE 8 : Normalisation MinMax ─────────────────
    # Les algorithmes ML sont sensibles aux échelles.
    # Prix : 10-500$, note : 0-5, remise : 0-100% → échelles très différentes.
    # MinMaxScaler ramène tout entre 0 et 1 :
    # formule : (x - min) / (max - min)
    print("\n[8/8] Normalisation des features numériques...")

    colonnes_a_normaliser = [
        "price",
        "discount_pct",
        "popularity_score",
        "catalogue_richness",
        "nb_reviews",
    ]

    scaler = MinMaxScaler()
    noms_norm = [f"{c}_norm" for c in colonnes_a_normaliser]

    df[noms_norm] = scaler.fit_transform(df[colonnes_a_normaliser].fillna(0))

    # Normalisation séparée de rating_filled sur l'échelle 0-5
    df["rating_filled_norm"] = (df["rating_filled"] / 5.0).round(4)

    print(f"  → {len(noms_norm) + 1} colonnes normalisées")
    print(f"  → Total colonnes dans le dataset : {len(df.columns)}")

    # ── Sauvegarde ────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n  Sauvegardé : {output_path}")
    print("=" * 55)

    return df


if __name__ == "__main__":
    df = engineer_features()
    print(f"\n✅ Feature Engineering terminé — {len(df.columns)} colonnes")
