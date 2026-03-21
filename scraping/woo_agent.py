# scraping/woo_agent.py
# ─────────────────────────────────────────────────────────
# Agent WooCommerce via la Store API publique.
#
# POINT CLÉ :
#   On utilise /wp-json/wc/store/v1/products (Store API)
#   et NON /wp-json/wc/v3/products (REST API classique).
#   La Store API est PUBLIQUE — aucune clé requise.
#
# Toute la configuration vient de config.py.
# ─────────────────────────────────────────────────────────

import requests
import pandas as pd
import time
import os
from bs4 import BeautifulSoup

# Tout vient de config.py — cohérence totale
from config import WOO_STORES, WOO_PER_PAGE, WOO_MAX_PAGES, OUTPUT_DIR


# ══════════════════════════════════════
#  EN-TÊTES HTTP
# ══════════════════════════════════════
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ══════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ══════════════════════════════════════

def nettoyer_html(texte):
    """Enlève les balises HTML avec BeautifulSoup."""
    if not texte:
        return ""
    propre = BeautifulSoup(texte, "html.parser").get_text(separator=" ").strip()
    propre = " ".join(propre.split())
    return propre[:500]

def safe(valeur, defaut=""):
    """Retourne defaut si valeur est None."""
    return defaut if valeur is None else valeur

def extraire_prix(product):
    """
    Extrait le prix depuis la structure "prices" de la Store API.
    Les prix sont en centimes → on divise par 10^currency_minor_unit.
    Exemple : price="999", currency_minor_unit=2 → 9.99$
    """
    prices    = product.get("prices") or {}
    try:
        minor_unit = int(prices.get("currency_minor_unit", 2))
    except (TypeError, ValueError):
        minor_unit = 2
    diviseur = 10 ** minor_unit

    def to_float(v):
        try:
            return float(v) / diviseur if v is not None else None
        except (TypeError, ValueError):
            return None

    prix_actuel   = to_float(prices.get("price"))
    prix_regulier = to_float(prices.get("regular_price"))
    prix_solde    = to_float(prices.get("sale_price"))

    prix        = prix_actuel or prix_solde or prix_regulier
    ancien_prix = (
        prix_regulier
        if prix_solde and prix_regulier and prix_solde < prix_regulier
        else None
    )
    return prix, ancien_prix

def extraire_categorie(product):
    """Extrait la catégorie : categories → tags → attributes."""
    for source in ["categories", "tags"]:
        items = product.get(source) or []
        if items:
            nom = items[0].get("name")
            if isinstance(nom, str) and nom.strip():
                return nom.strip()
    attrs = product.get("attributes") or []
    if attrs:
        options = attrs[0].get("options") or []
        if options and isinstance(options[0], str):
            return options[0].strip()
    return ""

def extraire_note(product):
    """Extrait la note moyenne et le nombre d'avis."""
    note    = product.get("average_rating") or product.get("rating")
    nb_avis = product.get("review_count") or product.get("rating_count") or 0
    try:
        note = float(note) if note not in (None, "", "0") else None
    except (TypeError, ValueError):
        note = None
    try:
        nb_avis = int(nb_avis) if nb_avis else 0
    except (TypeError, ValueError):
        nb_avis = 0
    return note, nb_avis


# ══════════════════════════════════════
#  FONCTION PRINCIPALE
# ══════════════════════════════════════

def scraper_boutique_woo(store):
    """
    Scrape une boutique WooCommerce via la Store API publique.
    Endpoint : /wp-json/wc/store/v1/products?per_page=40&page=1
    """

    tous_les_produits = []
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"\n🛒 WooCommerce — {store['name']}")
    print(f"   URL : {store['url']}")

    for numero_page in range(1, WOO_MAX_PAGES + 1):

        url = (
            f"{store['url']}/wp-json/wc/store/v1/products"
            f"?per_page={WOO_PER_PAGE}&page={numero_page}"
        )

        try:
            reponse = session.get(url, timeout=15)

            if reponse.status_code == 404:
                print(f"   ⚠️  404 — Store API non disponible")
                break
            if reponse.status_code == 403:
                print(f"   ⚠️  403 — accès refusé")
                break
            if reponse.status_code != 200:
                print(f"   ⚠️  HTTP {reponse.status_code}")
                break

            try:
                data = reponse.json()
            except ValueError:
                print(f"   ⚠️  Réponse invalide")
                break

            if not isinstance(data, list) or not data:
                print(f"   ✅ Fin — plus de produits")
                break

            for p in data:
                product_id  = str(p.get("id") or "")
                product_url = str(p.get("permalink") or p.get("link") or "")
                titre       = str(p.get("name") or p.get("title") or "").strip()

                if not product_id or not titre or not product_url:
                    continue

                prix, ancien_prix = extraire_prix(p)
                categorie         = extraire_categorie(p)
                note, nb_avis     = extraire_note(p)
                description       = nettoyer_html(
                                        p.get("description") or
                                        p.get("short_description") or ""
                                    )
                stock_status = p.get("stock_status") or ""
                disponible   = (
                    stock_status == "instock" or
                    p.get("is_in_stock") is True
                )
                variantes = p.get("variations") or []

                produit = {
                    "source"        : "woocommerce",
                    "shop_name"     : store["name"],
                    "shop_url"      : store["url"],
                    "geography"     : store.get("geography", "US"),
                    "product_id"    : product_id,
                    "product_url"   : product_url,
                    "title"         : titre,
                    "product_type"  : "woocommerce",
                    "vendor"        : store["name"],
                    "categories"    : categorie,
                    "tags"          : ", ".join([
                                        t.get("name", "")
                                        for t in (p.get("tags") or [])
                                      ]),
                    "price"         : prix,
                    "price_min"     : prix,
                    "price_max"     : prix,
                    "compare_price" : ancien_prix,
                    "on_sale"       : ancien_prix is not None,
                    "available"     : disponible,
                    "nb_variants"   : len(variantes),
                    "rating"        : note,
                    "nb_reviews"    : nb_avis,
                    "description"   : description,
                    "has_image"     : True,
                    "nb_images"     : 1,
                    "created_at"    : safe(p.get("date_created")),
                    "updated_at"    : safe(p.get("date_modified")),
                }
                tous_les_produits.append(produit)

            print(
                f"   → Page {numero_page} : "
                f"{len(data)} produits "
                f"(total : {len(tous_les_produits)})"
            )

            if len(data) < WOO_PER_PAGE:
                print(f"   ✅ Dernière page atteinte")
                break

            time.sleep(1.5)

        except requests.exceptions.ConnectionError:
            print(f"   ❌ Impossible de joindre {store['url']}")
            break
        except Exception as e:
            print(f"   ❌ Erreur inattendue : {e}")
            break

    print(f"   🎯 TOTAL : {len(tous_les_produits)} produits pour {store['name']}")
    return tous_les_produits


# ══════════════════════════════════════
#  FONCTION : sauvegarder en CSV
# ══════════════════════════════════════

def sauvegarder_csv(produits, nom_fichier):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df     = pd.DataFrame(produits)
    chemin = f"{OUTPUT_DIR}/{nom_fichier}_products.csv"
    df.to_csv(chemin, index=False, encoding="utf-8-sig")
    print(f"   💾 Sauvegardé : {chemin}")
    print(f"      {len(df)} lignes × {len(df.columns)} colonnes")
    return chemin


# ══════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════

if __name__ == "__main__":
    tous = []
    for boutique in WOO_STORES:
        produits = scraper_boutique_woo(boutique)
        if produits:
            sauvegarder_csv(produits, f"woo_{boutique['name']}")
            tous.extend(produits)
        time.sleep(2)
    if tous:
        sauvegarder_csv(tous, "woo_ALL")
        print(f"\n✅ WooCommerce terminé — {len(tous)} produits au total")