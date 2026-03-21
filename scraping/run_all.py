# scraping/run_all.py
# ─────────────────────────────────────────────────────────
# Point d'entrée principal.
# Lance le coordinateur A2A qui orchestre tous les agents.
#
# VERSION 4 — architecture A2A complète :
#   coordinator.py → ShopifyWorker + WooWorker + HTMLFallbackAgent
#   Tous les agents héritent de BaseScrapingAgent (base.py)
# ─────────────────────────────────────────────────────────

from coordinator import ScrapingCoordinator


def main():
    """
    Lance le scraping A2A complet.
    Le coordinateur gère tout :
    - création des workers appropriés
    - choix API ou fallback HTML
    - collecte et fusion des résultats
    """
    coordinateur = ScrapingCoordinator()
    df = coordinateur.run()

    if not df.empty:
        print(f"\n✅ Pipeline A2A terminé — {len(df)} produits dans le dataset final")
    else:
        print("\n❌ Aucun produit collecté")


if __name__ == "__main__":
    main()
