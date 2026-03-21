# scraping/coordinator.py
# ─────────────────────────────────────────────────────────
# Agent Coordinateur A2A — orchestrateur du scraping.
#
# CONCEPT A2A (Agent-to-Agent) :
#   Dans une architecture multi-agents, le coordinateur
#   reçoit une liste de tâches et les délègue aux agents
#   workers appropriés.
#
#   Schéma :
#     Coordinateur
#         ├── ShopifyWorker  → scrape boutique Shopify
#         ├── WooWorker      → scrape boutique WooCommerce
#         └── HTMLWorker     → fallback HTML si API inaccessible
#
#   Le coordinateur ne scrape pas lui-même — il coordonne.
#   Chaque worker est autonome et hérite de BaseScrapingAgent.
#
# PRINCIPE :
#   1. Pour chaque boutique Shopify → crée un ShopifyWorker
#   2. Pour chaque boutique WooCommerce → crée un WooWorker
#      Si l'API WooCommerce échoue → bascule sur HTMLWorker
#   3. Lance tous les workers (séquentiellement ici,
#      peut être parallélisé plus tard)
#   4. Collecte et fusionne tous les résultats
# ─────────────────────────────────────────────────────────

import pandas as pd
import time
import os
from datetime import datetime

# ── Importation des agents workers ───────────────────────
# On importe les fonctions existantes de shopify_agent et woo_agent
# en les ENVELOPPANT dans des classes qui héritent de BaseScrapingAgent
# Ainsi on réutilise tout le code déjà testé et fonctionnel

from base import BaseScrapingAgent
from shopify_agent import scraper_boutique_shopify, sauvegarder_csv
from woo_agent import scraper_boutique_woo
from html_fallback import HTMLFallbackAgent
from config import SHOPIFY_STORES, WOO_STORES, OUTPUT_DIR


# ══════════════════════════════════════
#  AGENT WORKER SHOPIFY
# ══════════════════════════════════════
# Encapsule la fonction scraper_boutique_shopify dans une classe.


class ShopifyWorker(BaseScrapingAgent):
    """
    Worker agent pour les boutiques Shopify.
    Utilise la Storefront API publique (/products.json).
    """

    def is_available(self) -> bool:
        """Vérifie si l'API Shopify répond."""
        import requests

        try:
            r = requests.get(f"{self.url}/products.json?limit=1", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def scrape(self) -> list:
        """Délègue au code Shopify existant et testé."""
        self.log("Démarrage (Shopify API)")
        # On réutilise la fonction déjà testée — pas de duplication
        produits = scraper_boutique_shopify(self.store)
        self.products = produits
        return produits


# ══════════════════════════════════════
#  AGENT WORKER WOOCOMMERCE
# ══════════════════════════════════════


class WooWorker(BaseScrapingAgent):
    """
    Worker agent pour les boutiques WooCommerce.
    Utilise la Store API publique (/wp-json/wc/store/v1/products).
    Si l'API échoue → le coordinateur bascule sur HTMLFallbackAgent.
    """

    def is_available(self) -> bool:
        """Vérifie si la Store API WooCommerce répond."""
        import requests

        try:
            r = requests.get(
                f"{self.url}/wp-json/wc/store/v1/products?per_page=1", timeout=10
            )
            return r.status_code == 200
        except Exception:
            return False

    def scrape(self) -> list:
        """Délègue au code WooCommerce existant et testé."""
        self.log("Démarrage (WooCommerce Store API)")
        produits = scraper_boutique_woo(self.store)
        self.products = produits
        return produits


# ══════════════════════════════════════
#  AGENT COORDINATEUR
# ══════════════════════════════════════


class ScrapingCoordinator:
    """
    Coordinateur A2A — orchestre tous les agents workers.

    Responsabilités :
    1. Créer les agents workers appropriés pour chaque boutique
    2. Vérifier la disponibilité de chaque source
    3. Choisir la bonne stratégie (API ou HTML fallback)
    4. Lancer les workers et collecter les résultats
    5. Générer un rapport d'exécution
    """

    def __init__(self):
        # Listes des agents créés
        self.workers = []  # tous les agents
        self.resultats = {}  # {nom_boutique: [produits]}
        self.rapport = []  # log des événements

        # Compteurs pour le rapport final
        self.nb_reussis = 0
        self.nb_echoues = 0
        self.nb_fallbacks = 0  # combien de fois le fallback HTML a été utilisé

    def _log(self, message: str):
        """Enregistre un événement dans le rapport."""
        horodatage = datetime.now().strftime("%H:%M:%S")
        entree = f"[{horodatage}] {message}"
        self.rapport.append(entree)
        print(entree)

    def _creer_workers_shopify(self):
        """
        Crée un ShopifyWorker pour chaque boutique dans SHOPIFY_STORES.
        """
        workers = []
        for store in SHOPIFY_STORES:
            worker = ShopifyWorker(store)
            workers.append(worker)
            self._log(f"Worker créé : {worker}")
        return workers

    def _creer_workers_woo(self):
        """
        Crée un WooWorker pour chaque boutique dans WOO_STORES.
        Si la Store API est indisponible → crée un HTMLFallbackAgent.
        C'est la logique A2A : choisir le bon agent selon la situation.
        """
        workers = []
        for store in WOO_STORES:
            # On crée d'abord un WooWorker pour tester la disponibilité
            woo_worker = WooWorker(store)

            self._log(f"Test disponibilité API : {store['name']}...")
            if woo_worker.is_available():
                # L'API répond → on utilise WooWorker
                self._log("  → API disponible : WooWorker assigné")
                workers.append(woo_worker)
            else:
                # L'API ne répond pas → on bascule sur HTMLFallbackAgent
                self._log("  → API indisponible : HTMLFallbackAgent assigné")
                fallback_store = {**store, "catalogue_url": store["url"] + "/shop"}
                # {**store} = copie du dictionnaire + on ajoute catalogue_url
                html_worker = HTMLFallbackAgent(fallback_store)
                workers.append(html_worker)
                self.nb_fallbacks += 1

        return workers

    def run(self) -> pd.DataFrame:
        """
        Lance le scraping complet.
        C'est la méthode principale — elle orchestre tout.
        Retourne : DataFrame pandas avec tous les produits
        """

        self._log("=" * 50)
        self._log("COORDINATEUR A2A — SCRAPING LANCÉ")
        self._log(f"Shopify    : {len(SHOPIFY_STORES)} boutiques")
        self._log(f"WooCommerce: {len(WOO_STORES)} boutiques")
        self._log("=" * 50)

        tous_produits = []

        # ── PHASE 1 : Workers Shopify ─────────────────────
        self._log("\n--- PHASE 1 : Workers Shopify ---")
        shopify_workers = self._creer_workers_shopify()

        for worker in shopify_workers:
            self._log(f"\nLancement : {worker}")
            try:
                produits = worker.scrape()
                if produits:
                    sauvegarder_csv(produits, f"shopify_{worker.name}")
                    tous_produits.extend(produits)
                    self.resultats[worker.name] = produits
                    self.nb_reussis += 1
                    self._log(f"  ✓ {len(produits)} produits collectés")
                else:
                    self.nb_echoues += 1
                    self._log("  ✗ Aucun produit")
            except Exception as e:
                self.nb_echoues += 1
                self._log(f"  ✗ Erreur : {e}")

            time.sleep(2)  # pause entre boutiques

        # ── PHASE 2 : Workers WooCommerce ─────────────────
        self._log("\n--- PHASE 2 : Workers WooCommerce ---")
        woo_workers = self._creer_workers_woo()

        for worker in woo_workers:
            self._log(f"\nLancement : {worker}")
            try:
                produits = worker.scrape()
                # Détermine le préfixe selon le type d'agent utilisé
                prefixe = "woo_html" if isinstance(worker, HTMLFallbackAgent) else "woo"
                if produits:
                    sauvegarder_csv(produits, f"{prefixe}_{worker.name}")
                    tous_produits.extend(produits)
                    self.resultats[worker.name] = produits
                    self.nb_reussis += 1
                    self._log(f"  ✓ {len(produits)} produits collectés")
                else:
                    self.nb_echoues += 1
                    self._log("  ✗ Aucun produit")
            except Exception as e:
                self.nb_echoues += 1
                self._log(f"  ✗ Erreur : {e}")

            time.sleep(2)

        # ── PHASE 3 : Fusion et sauvegarde finale ─────────
        self._log("\n--- PHASE 3 : Fusion finale ---")

        if not tous_produits:
            self._log("ERREUR : aucun produit collecté")
            return pd.DataFrame()

        df = pd.DataFrame(tous_produits)

        # Nettoyage de base
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        avant = len(df)
        df = df.drop_duplicates(subset=["shop_name", "product_id"])
        if len(df) < avant:
            self._log(f"Doublons supprimés : {avant - len(df)}")

        # Sauvegarde
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        chemin = f"{OUTPUT_DIR}/FINAL_sport_fitness_products.csv"
        df.to_csv(chemin, index=False, encoding="utf-8-sig")

        # ── Rapport final ─────────────────────────────────
        self._log("\n" + "=" * 50)
        self._log("SCRAPING A2A TERMINÉ")
        self._log(f"Total produits     : {len(df)}")
        self._log(f"Workers réussis    : {self.nb_reussis}")
        self._log(f"Workers échoués    : {self.nb_echoues}")
        self._log(f"Fallbacks HTML     : {self.nb_fallbacks}")
        self._log(f"Fichier final      : {chemin}")
        self._log("=" * 50)

        # Résumé par boutique
        self._log("\nRépartition par boutique :")
        resume = (
            df.groupby(["source", "shop_name"]).size().reset_index(name="nb_produits")
        )
        print(resume.to_string(index=False))

        return df
