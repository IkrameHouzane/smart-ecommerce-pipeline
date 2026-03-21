# data_processing/run_pipeline.py
# ─────────────────────────────────────────────────────────
# Lance toute l'Étape 2 dans l'ordre :
#   preprocess → features → scoring
#
# C'est le seul fichier que tu lances :
#   python run_pipeline.py
# ─────────────────────────────────────────────────────────

from preprocess import preprocess
from features import engineer_features
from scoring import scoring_topk


def run():
    print("\n" + "★" * 55)
    print("  PIPELINE ÉTAPE 2 — PREPROCESSING & SCORING")
    print("★" * 55 + "\n")

    # Étape 2A : nettoyage
    df_clean = preprocess()

    # Étape 2B : feature engineering
    df_featured = engineer_features()

    # Étape 2C : scoring Top-K
    df_scored, top_k = scoring_topk()

    print("\n" + "★" * 55)
    print("  ÉTAPE 2 TERMINÉE !")
    print(f"  Produits nettoyés   : {len(df_clean)}")
    print(f"  Features créées     : {len(df_featured.columns)}")
    print(f"  Produits scorés     : {len(df_scored)}")
    print(f"  Top-K sélectionnés  : {len(top_k)}")
    print("★" * 55)


if __name__ == "__main__":
    run()
