# scraping/html_fallback.py
# ─────────────────────────────────────────────────────────
# Agent de scraping HTML — fallback universel.
#
# QUAND l'utiliser ?
#   Quand un site ne répond pas à /products.json (Shopify)
#   ni à /wp-json/wc/store/v1/products (WooCommerce).
#   On scrape alors directement le HTML public de la page.
#
# COMMENT ?
#   requests télécharge le HTML de la page.
#   BeautifulSoup parse ce HTML et extrait les données.
#   On cherche les balises communes aux boutiques WooCommerce.
#
# Hérite de BaseScrapingAgent — respecte le contrat A2A.
# ─────────────────────────────────────────────────────────

import requests
from bs4 import BeautifulSoup  # parser HTML
import time
import re

from base import BaseScrapingAgent


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
#  SÉLECTEURS CSS — structures WooCommerce courantes
# ══════════════════════════════════════
# Une boutique WooCommerce typique utilise ces classes CSS.
# On essaie chaque sélecteur dans l'ordre jusqu'à trouver.
SELECTORS = {
    # Conteneur d'un produit sur la page catalogue
    "product_item": [
        "li.product",
        "li.product-item",
        ".product-grid-item",
        "article.product",
    ],
    # Titre du produit
    "title": [
        "h2.woocommerce-loop-product__title",
        ".product-title",
        "h2.product-name",
        "h3.product-title",
        ".card__heading a",
        "h2 a",
    ],
    # Prix
    "price": [
        ".price .amount",
        ".woocommerce-Price-amount",
        ".price-item",
        "span.price",
        ".product-price",
    ],
    # Note moyenne
    "rating": [
        ".star-rating",
        "[data-rating]",
        ".product-rating",
    ],
    # Lien vers la page produit
    "link": [
        "a.woocommerce-loop-product__link",
        "a.product-link",
        "h2 a",
        "a.card__heading",
        ".product-item__title a",
    ],
    # Bouton page suivante (pagination)
    "next_page": [
        "a.next.page-numbers",
        ".next-page",
        "a[aria-label='Next page']",
        "a[rel='next']",
    ],
}


class HTMLFallbackAgent(BaseScrapingAgent):
    """
    Agent de scraping HTML universel.
    Hérite de BaseScrapingAgent — doit implémenter scrape() et is_available().
    """

    def __init__(self, store: dict):
        # super().__init__ appelle le constructeur de la classe parente
        super().__init__(store)
        # URL du catalogue produits (ex: /shop ou /collections/all)
        self.catalogue_url = store.get("catalogue_url", store.get("url", "") + "/shop")

    # ── Implémentation de is_available() ─────────────────
    def is_available(self) -> bool:
        """
        Vérifie si le site répond avec HTTP 200.
        """
        try:
            r = requests.get(self.catalogue_url, headers=HEADERS, timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    # ── Implémentation de scrape() ────────────────────────
    def scrape(self) -> list:
        """
        Scrape le catalogue HTML page par page.
        1. Télécharge la page catalogue
        2. Parse le HTML avec BeautifulSoup
        3. Extrait titre, prix, lien de chaque produit
        4. Passe à la page suivante
        """
        self.log("Démarrage scraping HTML (fallback)")
        self.log(f"URL catalogue : {self.catalogue_url}")

        tous_produits = []
        url_courante = self.catalogue_url
        numero_page = 1
        max_pages = 20  # limite de sécurité

        while url_courante and numero_page <= max_pages:
            try:
                # ── Téléchargement de la page ─────────────
                reponse = requests.get(url_courante, headers=HEADERS, timeout=15)

                if reponse.status_code != 200:
                    self.log(f"HTTP {reponse.status_code} — arrêt")
                    break

                # ── Parsing HTML ──────────────────────────
                # BeautifulSoup analyse le HTML et crée un arbre navigable
                soupe = BeautifulSoup(reponse.text, "html.parser")
                # "html.parser" = parseur HTML intégré à Python (pas besoin de lxml)

                # ── Trouver les conteneurs produits ───────
                items = []
                for sel in SELECTORS["product_item"]:
                    items = soupe.select(sel)
                    # .select() = cherche TOUS les éléments correspondant au sélecteur CSS
                    if items:
                        break

                if not items:
                    self.log(
                        "Aucun produit trouvé — sélecteur incompatible avec ce site"
                    )
                    break

                self.log(f"Page {numero_page} : {len(items)} produits trouvés")

                # ── Extraction des données ────────────────
                for item in items:
                    produit = self._extraire_produit(item, url_courante)
                    if produit:
                        tous_produits.append(produit)

                # ── Pagination ────────────────────────────
                url_courante = self._trouver_page_suivante(soupe, url_courante)
                numero_page += 1
                time.sleep(1.5)  # pause respectueuse

            except requests.exceptions.ConnectionError:
                self.log(f"Connexion impossible à {url_courante}")
                break
            except Exception as e:
                self.log(f"Erreur : {e}")
                break

        self.log(f"TOTAL : {len(tous_produits)} produits")
        self.products = tous_produits
        return tous_produits

    # ── Méthode privée : extraire UN produit ─────────────
    def _extraire_produit(self, item, url_base: str) -> dict:
        """
        Extrait les données d'un élément produit HTML.
        Paramètre : item = élément BeautifulSoup (un <li> ou <div> produit)
        Retourne  : dictionnaire ou None si produit incomplet
        """

        # ── Titre ─────────────────────────────────────────
        titre = ""
        for sel in SELECTORS["title"]:
            elem = item.select_one(sel)
            # .select_one() = cherche LE PREMIER élément correspondant
            if elem:
                titre = elem.get_text(strip=True)
                # get_text() = extrait le texte visible (sans balises HTML)
                # strip=True = enlève les espaces au début et à la fin
                break

        if not titre:
            return None  # produit sans titre = on ignore

        # ── Prix ──────────────────────────────────────────
        prix_texte = ""
        for sel in SELECTORS["price"]:
            elem = item.select_one(sel)
            if elem:
                prix_texte = elem.get_text(strip=True)
                break

        # Nettoie le prix : "$29.99" ou "29,99 €" → "29.99"
        prix = self._nettoyer_prix(prix_texte)

        # ── Note ──────────────────────────────────────────
        note = ""
        for sel in SELECTORS["rating"]:
            elem = item.select_one(sel)
            if elem:
                texte = elem.get_text(strip=True)
                # Cherche un nombre décimal dans le texte
                match = re.search(r"(\d+\.?\d*)", texte)
                if match:
                    note = match.group(1)
                break

        # ── Lien ──────────────────────────────────────────
        lien = ""
        for sel in SELECTORS["link"]:
            elem = item.select_one(sel)
            if elem:
                href = elem.get("href", "")
                # .get("href") = récupère l'attribut href du tag <a>
                if href:
                    # Transforme les liens relatifs en absolus
                    if href.startswith("http"):
                        lien = href
                    else:
                        # Extrait la base de l'URL (ex: "https://site.com")
                        base = "/".join(url_base.split("/")[:3])
                        lien = base + href
                break

        return {
            "source": "html_fallback",
            "shop_name": self.name,
            "shop_url": self.url,
            "geography": self.store.get("geography", ""),
            "product_id": lien.split("/")[-1] if lien else titre[:30],
            "product_url": lien,
            "title": titre,
            "product_type": "woocommerce_html",
            "vendor": self.name,
            "categories": "",
            "tags": "",
            "price": prix,
            "price_min": prix,
            "price_max": prix,
            "compare_price": "",
            "on_sale": False,
            "available": True,  # on suppose disponible
            "nb_variants": 1,
            "rating": note,
            "nb_reviews": 0,
            "description": "",
            "has_image": bool(item.select_one("img")),
            "nb_images": len(item.select("img")),
            "created_at": "",
            "updated_at": "",
        }

    # ── Méthode privée : trouver la page suivante ─────────
    def _trouver_page_suivante(self, soupe, url_courante: str):
        """
        Cherche le lien "page suivante" dans le HTML.
        Retourne l'URL de la page suivante, ou None si dernière page.
        """
        for sel in SELECTORS["next_page"]:
            elem = soupe.select_one(sel)
            if elem:
                href = elem.get("href", "")
                if href:
                    if href.startswith("http"):
                        return href
                    base = "/".join(url_courante.split("/")[:3])
                    return base + href
        return None  # plus de page suivante

    # ── Méthode privée : nettoyer un prix ─────────────────
    def _nettoyer_prix(self, texte: str) -> str:
        """
        Extrait le nombre d'un texte de prix.
        "$29.99 USD" → "29.99"
        "€ 45,00"   → "45.00"
        """
        if not texte:
            return ""
        # Enlève tout sauf chiffres, virgule, point
        prix = re.sub(r"[^\d.,]", "", texte)
        prix = prix.replace(",", ".")
        parties = prix.split(".")
        if len(parties) > 2:
            prix = "".join(parties[:-1]) + "." + parties[-1]
        return prix
