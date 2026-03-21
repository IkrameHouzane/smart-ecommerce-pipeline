# scraping/shopify_agent.py
# ─────────────────────────────────────────────────────────
# Agent de scraping pour les boutiques Shopify.
# Visite chaque boutique page par page via /products.json
# et sauvegarde les données dans des fichiers CSV.
# ─────────────────────────────────────────────────────────

import requests
import pandas as pd
import time
import os
import re

# On importe TOUT depuis config.py — source unique de vérité
from config import SHOPIFY_STORES, PRODUCTS_PER_PAGE, OUTPUT_DIR


# ══════════════════════════════════════
#  EN-TÊTES HTTP
# ══════════════════════════════════════
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ══════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ══════════════════════════════════════

def nettoyer_html(texte):
    """Enlève les balises HTML d'un texte."""
    if not texte:
        return ""
    texte = re.sub(r"<[^>]+>", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte

def safe(valeur, defaut=""):
    """Retourne defaut si valeur est None."""
    if valeur is None:
        return defaut
    return valeur


# ══════════════════════════════════════
#  FONCTION PRINCIPALE : scraper UNE boutique
# ══════════════════════════════════════

def scraper_boutique_shopify(store):
    """
    Scrape tous les produits d'une boutique Shopify.
    Paramètre : store = dictionnaire avec "name" et "url"
    Retourne  : liste de dictionnaires (un par produit)
    """

    tous_les_produits = []
    numero_page       = 1

    print(f"\n🔍 Shopify — {store['name']}")
    print(f"   URL : {store['url']}")

    while True:

        url = (
            f"{store['url']}/products.json"
            f"?limit={PRODUCTS_PER_PAGE}&page={numero_page}"
        )

        try:
            reponse = requests.get(url, headers=HEADERS, timeout=20)

            if reponse.status_code == 403:
                print(f"   ⚠️  403 — accès refusé par la boutique")
                break
            if reponse.status_code == 404:
                print(f"   ⚠️  404 — boutique non Shopify ou URL incorrecte")
                break
            if reponse.status_code != 200:
                print(f"   ⚠️  Erreur HTTP {reponse.status_code}")
                break

            donnees  = reponse.json()
            produits = donnees.get("products", [])

            if not produits:
                print(f"   ✅ Dernière page atteinte — scraping terminé")
                break

            for p in produits:

                variantes = p.get("variants") or []
                images    = p.get("images")   or []
                tags      = p.get("tags")     or []
                prem_var  = variantes[0] if variantes else {}

                prix_liste = []
                for v in variantes:
                    try:
                        prix_liste.append(float(v.get("price") or 0))
                    except:
                        pass

                produit = {
                    "source"        : "shopify",
                    "shop_name"     : store["name"],
                    "shop_url"      : store["url"],
                    "geography"     : store.get("geography", "US"),
                    "product_id"    : safe(p.get("id")),
                    "product_url"   : f"{store['url']}/products/{safe(p.get('handle'))}",
                    "title"         : safe(p.get("title")).strip(),
                    "product_type"  : safe(p.get("product_type")).strip(),
                    "vendor"        : safe(p.get("vendor")).strip(),
                    "tags"          : ", ".join([str(t) for t in tags]),
                    "price"         : safe(prem_var.get("price")),
                    "price_min"     : min(prix_liste) if prix_liste else None,
                    "price_max"     : max(prix_liste) if prix_liste else None,
                    "compare_price" : safe(prem_var.get("compare_at_price")),
                    # on_sale = True si compare_price existe (= prix barré)
                    "on_sale"       : bool(prem_var.get("compare_at_price")),
                    "available"     : prem_var.get("available") or False,
                    "nb_variants"   : len(variantes),
                    "description"   : nettoyer_html(
                                        safe(p.get("body_html"))
                                      )[:500],
                    "has_image"     : len(images) > 0,
                    "nb_images"     : len(images),
                    "created_at"    : safe(p.get("created_at")),
                    "updated_at"    : safe(p.get("updated_at")),
                }

                tous_les_produits.append(produit)

            print(f"   → Page {numero_page} : {len(produits)} produits récupérés")
            numero_page += 1
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
    for boutique in SHOPIFY_STORES:
        produits = scraper_boutique_shopify(boutique)
        if produits:
            sauvegarder_csv(produits, f"shopify_{boutique['name']}")
            tous.extend(produits)
        time.sleep(2)
    if tous:
        sauvegarder_csv(tous, "shopify_ALL")
        print(f"\n✅ Shopify terminé — {len(tous)} produits au total")