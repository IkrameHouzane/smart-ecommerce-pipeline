# data_processing/preprocess.py
# ─────────────────────────────────────────────────────────
# ÉTAPE 2A — Nettoyage et validation des données brutes.
#
# CE FICHIER FAIT QUOI ?
#   Il prend le CSV brut sorti du scraping (5051 lignes sales)
#   et le transforme en tableau propre, cohérent, utilisable
#   par les algorithmes ML de l'étape suivante.
#
# POURQUOI C'EST NÉCESSAIRE ?
#   Les données brutes ont plusieurs problèmes :
#   - Les prix sont du TEXTE ("29.99") pas des NOMBRES (29.99)
#     → impossible de faire des maths dessus
#   - Certaines colonnes sont vides (NaN)
#     → les algorithmes ML plantent sur les valeurs manquantes
#   - Les catégories sont incohérentes ("T-Shirt", "t-shirt", "tshirt")
#     → le clustering les traite comme 3 catégories différentes
#   - Il peut y avoir des doublons
#     → fausse les statistiques
#
# ANALOGIE :
#   C'est comme trier et nettoyer des légumes avant de cuisiner.
#   Tu ne mets pas des carottes avec la terre dans ta soupe.
# ─────────────────────────────────────────────────────────

import pandas as pd  # manipulation de tableaux de données
import numpy as np  # calculs numériques (NaN, inf, etc.)
import os

# ── Chemins des fichiers ──────────────────────────────────
# On remonte d'un niveau depuis data_processing/ vers smart_ecommerce/
# puis on va dans data/ pour lire le fichier final du scraping
INPUT_PATH = "../data/FINAL_sport_fitness_products.csv"
OUTPUT_PATH = "../data/clean_products.parquet"
# Parquet = format de fichier optimisé pour les données volumineuses.
# Plus rapide que CSV, préserve les types (un nombre reste un nombre).
# C'est le standard utilisé dans les pipelines ML professionnels.


# ══════════════════════════════════════
#  FONCTION PRINCIPALE
# ══════════════════════════════════════


def preprocess(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """
    Nettoie et valide le dataset brut.
    Retourne un DataFrame pandas propre.
    """

    print("=" * 55)
    print("  ÉTAPE 2A — PREPROCESSING")
    print("=" * 55)

    # ── 1. CHARGEMENT ─────────────────────────────────────
    print("\n[1/6] Chargement des données...")

    df = pd.read_csv(input_path, low_memory=False)
    # low_memory=False = lit tout le fichier d'un coup
    # (évite des erreurs de type sur les grandes colonnes mixtes)

    print(f"  → {len(df)} lignes, {len(df.columns)} colonnes chargées")
    print(f"  → Sources : {df['source'].value_counts().to_dict()}")

    # ── 2. SUPPRESSION DES DOUBLONS ───────────────────────
    print("\n[2/6] Suppression des doublons...")

    avant = len(df)
    # Un doublon = même boutique + même product_id
    # (peut arriver si on a relancé le scraping)
    df = df.drop_duplicates(subset=["shop_name", "product_id"])
    apres = len(df)

    print(f"  → {avant - apres} doublons supprimés ({apres} lignes restantes)")

    # ── 3. NETTOYAGE DES PRIX ─────────────────────────────
    print("\n[3/6] Nettoyage des prix...")

    # pd.to_numeric() essaie de convertir chaque valeur en nombre
    # errors="coerce" = si conversion impossible → met NaN
    # (au lieu de planter avec une erreur)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price_min"] = pd.to_numeric(df["price_min"], errors="coerce")
    df["price_max"] = pd.to_numeric(df["price_max"], errors="coerce")
    df["compare_price"] = pd.to_numeric(df["compare_price"], errors="coerce")

    # Supprimer les produits sans prix — inutilisables pour le ML
    # car le prix est notre variable principale
    avant = len(df)
    df = df[df["price"].notna() & (df["price"] > 0)]
    print(f"  → {avant - len(df)} produits sans prix supprimés")

    # Corriger les prix aberrants
    # Un produit à 0.01$ ou 99999$ est probablement une erreur de données
    df = df[(df["price"] >= 1) & (df["price"] <= 5000)]
    print(f"  → Prix valides : entre 1$ et 5000$ ({len(df)} produits)")

    # Calculer le prix moyen si price_min et price_max existent
    # (utile pour les produits avec variantes de prix)
    df["price_mean"] = df[["price_min", "price_max"]].mean(axis=1)
    # axis=1 = calculer la moyenne sur les colonnes (pas les lignes)
    # Si price_min ou price_max est NaN → price_mean = l'autre valeur
    df["price_mean"] = df["price_mean"].fillna(df["price"])
    # fillna() = remplace les NaN par une valeur de remplacement

    # ── 4. NETTOYAGE DES NOTES ET AVIS ───────────────────
    print("\n[4/6] Nettoyage des notes et avis...")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["nb_reviews"] = pd.to_numeric(df["nb_reviews"], errors="coerce")

    # Les notes doivent être entre 0 et 5
    # Une note de 47 ou -2 est clairement une erreur
    df.loc[df["rating"] > 5, "rating"] = np.nan
    df.loc[df["rating"] < 0, "rating"] = np.nan
    # .loc[condition, colonne] = modifier une colonne selon une condition

    # Remplacer les NaN de nb_reviews par 0
    # (un produit sans avis a bien 0 avis, pas une valeur inconnue)
    df["nb_reviews"] = df["nb_reviews"].fillna(0).astype(int)
    # astype(int) = convertit en entiers (0, 1, 2...) pas en décimaux (0.0, 1.0...)

    # Pour les ratings manquants (surtout Shopify qui ne les fournit pas),
    # on garde NaN — on ne DEVINE pas une note.
    # On les traitera dans le feature engineering.
    nb_sans_note = df["rating"].isna().sum()
    print(
        f"  → {nb_sans_note} produits sans note ({nb_sans_note / len(df) * 100:.1f}%)"
    )
    print(f"  → Note moyenne globale : {df['rating'].mean():.2f}/5")

    # ── 5. NETTOYAGE DES TEXTES ───────────────────────────
    print("\n[5/6] Nettoyage des textes...")

    # Nettoyer les titres
    df["title"] = (
        df["title"]
        .fillna("")  # remplace NaN par chaîne vide
        .str.strip()  # enlève espaces début/fin
        .str.replace(r"\s+", " ", regex=True)  # espaces multiples → un seul
    )

    # Supprimer les produits sans titre (inutilisables)
    avant = len(df)
    df = df[df["title"].str.len() > 1]
    print(f"  → {avant - len(df)} produits sans titre supprimés")

    # Normaliser les noms de boutiques (minuscules, sans espaces superflus)
    df["shop_name"] = df["shop_name"].str.strip().str.lower()

    # Normaliser la colonne "available" en booléen (True/False)
    # Elle peut contenir True, False, "True", "False", 1, 0, etc.
    df["available"] = df["available"].map(
        lambda x: str(x).lower() in ("true", "1", "yes", "instock")
    )
    # map(lambda) applique une fonction à chaque valeur de la colonne
    # On convertit tout en True/False proprement

    # Nettoyer la géographie — remplir les vides par "US" (majorité)
    df["geography"] = df["geography"].fillna("US").str.upper().str.strip()

    # Nettoyer les catégories
    df["product_type"] = (
        df["product_type"]
        .fillna("")
        .str.strip()
        .str.lower()
        # Supprimer les caractères spéciaux
        .str.replace(r"[^a-z0-9\s\-]", "", regex=True)
    )

    # ── 6. COLONNES DÉRIVÉES DE BASE ─────────────────────
    print("\n[6/6] Création des colonnes dérivées de base...")

    # Colonne "a_une_promo" : True si le produit a un prix barré
    # et que ce prix barré est supérieur au prix actuel
    df["a_une_promo"] = df["compare_price"].notna() & (
        df["compare_price"] > df["price"]
    )
    # .notna() = True si la valeur n'est PAS NaN

    # Colonne "nb_variants" nettoyée
    df["nb_variants"] = (
        pd.to_numeric(df["nb_variants"], errors="coerce").fillna(1).astype(int)
    )
    # Un produit a au minimum 1 variante

    # Colonne "nb_images" nettoyée
    df["nb_images"] = (
        pd.to_numeric(df["nb_images"], errors="coerce").fillna(0).astype(int)
    )

    # ── RAPPORT QUALITÉ ───────────────────────────────────
    print("\n" + "─" * 40)
    print("RAPPORT QUALITÉ")
    print("─" * 40)
    print(f"  Produits finaux      : {len(df)}")
    print(f"  Colonnes             : {len(df.columns)}")
    print(f"  Prix moyen           : ${df['price'].mean():.2f}")
    print(f"  Prix médian          : ${df['price'].median():.2f}")
    print(
        f"  Produits disponibles : {df['available'].sum()} ({df['available'].mean() * 100:.1f}%)"
    )
    print(
        f"  Produits en promo    : {df['a_une_promo'].sum()} ({df['a_une_promo'].mean() * 100:.1f}%)"
    )
    print(
        f"  Avec note            : {df['rating'].notna().sum()} ({df['rating'].notna().mean() * 100:.1f}%)"
    )

    # Par source
    print("\n  Répartition par plateforme :")
    print(
        df.groupby("source")["price"]
        .agg(["count", "mean", "median"])
        .round(2)
        .to_string()
    )

    # ── SAUVEGARDE ────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    # Parquet préserve les types automatiquement :
    # float reste float, int reste int, bool reste bool
    # Contrairement au CSV qui tout convertit en texte

    print(f"\n  Sauvegardé : {output_path}")
    print(f"  Taille     : {os.path.getsize(output_path) / 1024:.1f} Ko")
    print("=" * 55)

    return df


# ══════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════
if __name__ == "__main__":
    df = preprocess()
    print(f"\n✅ Preprocessing terminé — {len(df)} produits propres")
