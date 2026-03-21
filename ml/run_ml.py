# ml/run_ml.py
# ─────────────────────────────────────────────────────────
# Point d'entrée Partie A — lance les 3 algos dans l'ordre.
# ─────────────────────────────────────────────────────────

from classifier        import run_classifier
from clustering        import run_clustering
from association_rules import run_association_rules


def run():
    print("\n" + "★" * 55)
    print("  PIPELINE ML — PARTIE A")
    print("★" * 55 + "\n")

    # 3A — Classification supervisée
    rf_model, xgb_model, classif = run_classifier()

    # 3B — Clustering non supervisé
    df_clusters, cluster_stats = run_clustering()

    # 3C — Règles d'association
    itemsets, regles = run_association_rules()

    print("\n" + "★" * 55)
    print("  PARTIE A TERMINÉE")
    if classif:
        print(f"  RF  accuracy : {classif['random_forest']['accuracy']*100:.1f}%")
        print(f"  XGB accuracy : {classif['xgboost']['accuracy']*100:.1f}%")
    if cluster_stats:
        sil = cluster_stats['kmeans']['silhouette_score']
        anom = cluster_stats['dbscan']['nb_anomalies']
        print(f"  Silhouette   : {sil:.4f}")
        print(f"  Anomalies    : {anom}")
    if regles is not None:
        print(f"  Règles Apriori : {len(regles)}")
    print("★" * 55)


if __name__ == "__main__":
    run()