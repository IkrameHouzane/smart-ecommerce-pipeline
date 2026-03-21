# scraping/base.py
# ─────────────────────────────────────────────────────────
# Classe de base abstraite pour tous les agents de scraping.
#
# POURQUOI une classe abstraite ?
#   Dans l'architecture A2A, tous les agents (Shopify, WooCommerce,
#   HTML) doivent avoir la MÊME interface — les mêmes méthodes.
#   Ainsi le coordinateur peut les appeler de façon uniforme
#   sans se soucier de leur implémentation interne.
#
#   C'est le principe de polymorphisme en POO :
#   le coordinateur dit "scrape()" à n'importe quel agent
#   et chacun sait quoi faire à sa façon.
# ─────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
# ABC = Abstract Base Class — outil Python pour créer des classes abstraites
# abstractmethod = décorateur qui rend une méthode obligatoire à implémenter

import pandas as pd
import os
from config import OUTPUT_DIR


class BaseScrapingAgent(ABC):
    """
    Classe de base pour tous les agents de scraping.
    Tout agent DOIT implémenter les méthodes marquées @abstractmethod.
    Si un agent hérite de cette classe sans les implémenter → erreur Python.
    """

    def __init__(self, store: dict):
        """
        Constructeur — appelé quand on crée un agent.
        Paramètre : store = dictionnaire avec les infos de la boutique
        """
        # self = l'instance de l'agent (comme "moi" en Python)
        self.store = store
        self.name = store.get("name", "unknown")
        self.url = store.get("url", "")
        self.products = []  # liste qui stockera les produits collectés

    # ── Méthode abstraite 1 ───────────────────────────────
    @abstractmethod
    def scrape(self) -> list:
        """
        Lance le scraping de la boutique.
        DOIT être implémentée par chaque agent enfant.
        Retourne : liste de dictionnaires produit
        """
        pass  # "pass" = le corps sera défini dans la classe enfant

    # ── Méthode abstraite 2 ───────────────────────────────
    @abstractmethod
    def is_available(self) -> bool:
        """
        Vérifie si la boutique est accessible.
        DOIT être implémentée par chaque agent enfant.
        Retourne : True si accessible, False sinon
        """
        pass

    # ── Méthode concrète (partagée par tous les agents) ───
    def save(self, products: list, suffix: str = "") -> str:
        """
        Sauvegarde les produits en CSV.
        Cette méthode est IDENTIQUE pour tous les agents —
        pas besoin de la réécrire dans chaque classe enfant.
        """
        if not products:
            return ""

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Nom du fichier : ex "shopify_allbirds_products.csv"
        nom = f"{suffix}_{self.name}" if suffix else self.name
        chemin = f"{OUTPUT_DIR}/{nom}_products.csv"

        df = pd.DataFrame(products)
        df.to_csv(chemin, index=False, encoding="utf-8-sig")

        print(f"   💾 {chemin}  ({len(df)} lignes × {len(df.columns)} colonnes)")
        return chemin

    # ── Méthode concrète utilitaire ───────────────────────
    def log(self, message: str):
        """
        Affiche un message avec le nom de l'agent en préfixe.
        Exemple : "[ShopifyAgent:allbirds] Scraping démarré"
        """
        agent_type = self.__class__.__name__
        # __class__.__name__ = nom de la classe (ex: "ShopifyAgent")
        print(f"   [{agent_type}:{self.name}] {message}")

    def __repr__(self):
        """
        Représentation textuelle de l'agent (utile pour le débogage).
        Exemple : ShopifyAgent(allbirds @ https://www.allbirds.com)
        """
        return f"{self.__class__.__name__}({self.name} @ {self.url})"
