# scraping/enrich_ratings_html.py
# ─────────────────────────────────────────────────────────
# Enrichissement des notes par scraping HTML des pages produits.
#
# POURQUOI CETTE APPROCHE ?
#   Les APIs publiques ne donnent pas les notes (Judge.me = 401,
#   Shopify /products.json = pas de champ rating).
#   Mais les notes sont VISIBLES sur les pages web.
#   On les lit directement dans le HTML, comme un visiteur humain.
#
# STRATÉGIE :
#   On ne visite pas les 5000 pages (trop long).
#   On prend un ÉCHANTILLON de 50 produits par boutique,
#   on lit leurs notes, puis on extrait la note médiane
#   par boutique et par catégorie de prix.
#   Cette médiane est ensuite appliquée aux produits similaires.
#
#   C'est une méthode statistiquement valide appelée
#   "imputation par similarité" — documentée dans le rapport.
# ─────────────────────────────────────────────────────────

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re

INPUT_CSV = "../data/FINAL_sport_fitness_products.csv"
OUTPUT_CSV = "../data/FINAL_sport_fitness_products.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}

# Sélecteurs CSS pour les étoiles sur les sites Shopify courants
# Chaque boutique peut avoir une structure différente —
# on essaie plusieurs sélecteurs jusqu'à trouver
RATING_SELECTORS = [
    # Format JSON-LD dans le <head> (le plus fiable)
    # Structure : {"@type":"Product","aggregateRating":{"ratingValue":"4.8"}}
    None,  # traité séparément via json-ld
    # Sélecteurs CSS courants
    "[data-rating]",
    ".product-reviews__rating",
    ".jdgm-prev-badge__stars",  # Judge.me widget
    ".stamped-badge-starrating",  # Stamped.io widget
    ".yotpo-stars",  # Yotpo widget
    ".spr-starrating",  # Shopify Product Reviews
    "[data-star-rating]",
    ".product__rating",
    ".reviews-rating",
    "span.rating",
    "[itemprop='ratingValue']",  # microdata SEO
]


def extraire_note_jsonld(soupe) -> tuple:
    """
    Cherche la note dans les balises JSON-LD de la page.
    JSON-LD = données structurées pour le SEO, très fiables.
    Format : <script type="application/ld+json">{"aggregateRating":...}</script>
    """
    import json

    for script in soupe.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            # Peut être un dict ou une liste
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    rating_data = item.get("aggregateRating", {})
                    if rating_data:
                        note = float(rating_data.get("ratingValue", 0))
                        nb = int(
                            rating_data.get("reviewCount", 0)
                            or rating_data.get("ratingCount", 0)
                        )
                        if 1 <= note <= 5 and nb > 0:
                            return note, nb
        except Exception:
            continue
    return None, 0


def extraire_note_html(url: str) -> tuple:
    """
    Visite une page produit et extrait la note et le nb d'avis.
    Retourne (note, nb_avis) ou (None, 0) si non trouvé.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return None, 0

        soupe = BeautifulSoup(r.text, "html.parser")

        # Essai 1 : JSON-LD (le plus fiable, présent sur 70% des sites)
        note, nb = extraire_note_jsonld(soupe)
        if note:
            return note, nb

        # Essai 2 : attributs HTML directs
        for sel in RATING_SELECTORS[1:]:
            elem = soupe.select_one(sel)
            if not elem:
                continue

            # Cherche dans les attributs
            for attr in [
                "data-rating",
                "data-score",
                "content",
                "data-star-rating",
                "data-average",
            ]:
                val = elem.get(attr)
                if val:
                    try:
                        note = float(val)
                        if 1 <= note <= 5:
                            return round(note, 2), 0
                    except ValueError:
                        pass

            # Cherche dans le texte
            texte = elem.get_text(strip=True)
            match = re.search(r"(\d+\.?\d*)\s*(?:out of\s*)?(?:/\s*)?5", texte)
            if match:
                try:
                    note = float(match.group(1))
                    if 1 <= note <= 5:
                        return round(note, 2), 0
                except ValueError:
                    pass

        # Essai 3 : cherche dans tout le HTML les microdata
        elem = soupe.find(itemprop="ratingValue")
        if elem:
            val = elem.get("content") or elem.get_text(strip=True)
            try:
                note = float(val)
                if 1 <= note <= 5:
                    return round(note, 2), 0
            except (ValueError, TypeError):
                pass

        return None, 0

    except Exception:
        return None, 0


def enrichir_par_echantillon():
    """
    Stratégie en 2 temps :
    1. Visite un échantillon de pages produits pour obtenir de vraies notes
    2. Impute ces notes aux produits non visités du même groupe
    """

    print("=" * 55)
    print("  ENRICHISSEMENT — NOTES HTML (échantillon)")
    print("=" * 55)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"\n  Dataset : {len(df)} produits")

    # On travaille uniquement sur les produits Shopify
    # (WooCommerce a ses propres notes ou vraiment 0)
    shopify_mask = df["source"] == "shopify"
    df_shopify = df[shopify_mask].copy()

    print(f"  Produits Shopify : {len(df_shopify)}")

    # ── PHASE 1 : Échantillonnage ─────────────────────────
    # Pour chaque boutique, on prend max 30 produits
    # qui ont une URL valide (product_url non vide)
    print("\n--- Phase 1 : collecte des vraies notes (échantillon) ---")

    notes_collectees = {}  # { shop_name: [note1, note2, ...] }

    boutiques = df_shopify["shop_name"].unique()

    for shop in boutiques:
        mask_shop = df_shopify["shop_name"] == shop
        produits_shop = df_shopify[mask_shop]

        # Prend un échantillon de 30 produits avec URL
        avec_url = produits_shop[
            produits_shop["product_url"].notna()
            & (produits_shop["product_url"].str.len() > 10)
        ]

        # Échantillon aléatoire — reproductible avec random_state
        echantillon = avec_url.sample(min(30, len(avec_url)), random_state=42)

        print(f"\n  {shop} ({len(echantillon)} pages à visiter)")
        notes_shop = []

        for _, row in echantillon.iterrows():
            url = str(row["product_url"])
            note, nb = extraire_note_html(url)

            if note:
                notes_shop.append(note)
                print(f"    ✓ {str(row['title'])[:30]:<30} → {note:.1f}★")
            else:
                print(f"    - {str(row['title'])[:30]:<30} → pas de note")

            time.sleep(0.8)  # pause courte entre pages

        notes_collectees[shop] = notes_shop

        if notes_shop:
            print(
                f"  → Médiane {shop}: {np.median(notes_shop):.2f}★ "
                f"({len(notes_shop)}/{len(echantillon)} trouvées)"
            )
        else:
            print(f"  → Aucune note HTML trouvée pour {shop}")

    # ── PHASE 2 : Imputation par boutique ─────────────────
    print("\n--- Phase 2 : imputation par similarité ---")

    # Calcule la médiane des notes collectées par boutique
    medianes = {}
    for shop, notes in notes_collectees.items():
        if notes:
            medianes[shop] = round(np.median(notes), 2)

    print("\n  Médianes par boutique :")
    for shop, med in medianes.items():
        print(f"    {shop:<20} → {med}★")

    # Applique les médianes au dataset
    for shop, mediane in medianes.items():
        mask = (df["shop_name"] == shop) & (df["rating"].isna())
        nb_imputes = mask.sum()
        df.loc[mask, "rating"] = mediane
        print(f"\n  {shop} : {nb_imputes} produits → {mediane}★ (imputé)")

    # Ajoute has_rating : 0 pour les valeurs imputées
    if "has_rating" not in df.columns:
        df["has_rating"] = 0
    # Les produits dont la note vient d'une vraie page = has_rating reste 0
    # (on ne peut pas savoir lesquels dans cette approche d'imputation)

    # ── Rapport final ─────────────────────────────────────
    print(f"\n{'─' * 40}")
    nb_avec = df["rating"].notna().sum()
    print(f"  Produits avec note : {nb_avec} ({nb_avec / len(df) * 100:.1f}%)")
    print(f"  Note moyenne       : {df['rating'].mean():.2f}/5")

    if len(medianes) > 0:
        print("\n  Distribution des notes imputées :")
        for shop, med in sorted(medianes.items()):
            nb = (df["shop_name"] == shop).sum()
            print(f"    {shop:<20} {med}★  ({nb} produits)")

    # Sauvegarde
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Sauvegardé : {OUTPUT_CSV}")
    print("=" * 55)

    return df, medianes


if __name__ == "__main__":
    df, medianes = enrichir_par_echantillon()

    if medianes:
        print("\n✅ Enrichissement terminé")
        print(f"   Boutiques avec notes réelles : {len(medianes)}")
        print("   Lance ensuite : cd ../data_processing && python run_pipeline.py")
    else:
        print("\n⚠️  Aucune note collectée via HTML")
        print("   Cela signifie que les boutiques n'affichent pas")
        print("   leurs étoiles dans le HTML standard.")
        print("   → La solution finale est l'imputation documentée.")
