# ml/clustering.py
# ─────────────────────────────────────────────────────────
# PARTIE A — Clustering non supervisé KMeans + DBSCAN.
#
# CE QUE L'ÉNONCÉ DEMANDE :
#   "KMeans, DBSCAN pour segmenter les produits"
#   "Silhouette score pour évaluer la qualité des clusters"
#   "Interprétation visuelle (PCA 2D)"
#   "Détection d'anomalies avec DBSCAN"
#
# KMeans  = divise en K groupes sphériques (K=4)
# DBSCAN  = trouve des groupes de forme libre
#           + étiquette -1 les anomalies
# PCA     = réduit N dimensions → 2 pour visualisation
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json
import os

from sklearn.cluster      import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics       import silhouette_score

INPUT_PATH  = "../data/scored_products.parquet"
OUTPUT_DIR  = "../analytics"
K_CLUSTERS  = 4


def preparer_features_clustering(df: pd.DataFrame):
    """
    Sélectionne les features numériques pour le clustering.
    Pas de variable cible — c'est non supervisé.
    StandardScaler : centre et réduit (moyenne=0, écart-type=1)
    car KMeans et DBSCAN sont sensibles aux différences d'échelle.
    """
    features = [
        "price_norm",
        "discount_pct_norm",
        "rating_filled_norm",
        "nb_reviews_norm",
        "catalogue_richness",
        "popularity_score_norm",
    ]
    features_ok = [f for f in features if f in df.columns]
    df_clean    = df[features_ok].dropna()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    print(f"  → Features : {features_ok}")
    print(f"  → Produits : {len(df_clean)}")

    return X_scaled, df_clean.index, features_ok


def run_kmeans(X_scaled, k=K_CLUSTERS):
    """
    KMeans avec k=4 — segmentation business naturelle :
      Cluster 0 : budget populaires
      Cluster 1 : milieu de gamme qualité
      Cluster 2 : premium exclusif
      Cluster 3 : promotions opportunistes
    (les libellés réels dépendent des données)
    """
    print(f"\n  --- KMeans (K={k}) ---")

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    labels = kmeans.fit_predict(X_scaled)

    # Silhouette score : entre -1 et 1, > 0.5 = bon clustering
    sil = silhouette_score(X_scaled, labels)

    print(f"  → Inertie         : {kmeans.inertia_:.1f}")
    print(f"  → Silhouette score: {sil:.4f}  "
          f"({'bon' if sil > 0.4 else 'acceptable' if sil > 0.2 else 'faible'})")
    print(f"  → Répartition :")
    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"     Cluster {c} : {n} produits ({n/len(labels)*100:.1f}%)")

    return kmeans, labels, sil


def run_dbscan(X_scaled):
    """
    DBSCAN : trouve les groupes denses et étiquette -1 les anomalies.
    eps=0.5 = rayon de voisinage en unités StandardScaler.
    min_samples=5 = minimum de voisins pour être un core point.
    """
    print(f"\n  --- DBSCAN ---")

    dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
    labels = dbscan.fit_predict(X_scaled)

    nb_clusters  = len(set(labels) - {-1})
    nb_anomalies = (labels == -1).sum()
    pct_anomalies = nb_anomalies / len(labels) * 100

    print(f"  → Clusters trouvés : {nb_clusters}")
    print(f"  → Anomalies        : {nb_anomalies} ({pct_anomalies:.1f}%)")

    return dbscan, labels, nb_anomalies, pct_anomalies


def run_clustering(input_path=INPUT_PATH, output_dir=OUTPUT_DIR):

    print("=" * 55)
    print("  3B — CLUSTERING KMEANS + DBSCAN")
    print("=" * 55)

    df = pd.read_parquet(input_path)
    print(f"\n  Produits chargés : {len(df)}")

    print("\n[1/4] Préparation...")
    X_scaled, indices_valides, feature_names = \
        preparer_features_clustering(df)

    print("\n[2/4] KMeans...")
    kmeans, km_labels, silhouette = run_kmeans(X_scaled)

    print("\n[3/4] DBSCAN...")
    dbscan, db_labels, nb_anom, pct_anom = run_dbscan(X_scaled)

    print("\n[4/4] PCA + sauvegarde...")

    # PCA 2D pour visualisation dans le dashboard
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    variance = pca.explained_variance_ratio_.sum()
    print(f"  → PCA variance expliquée : {variance*100:.1f}%")

    # Ajoute les résultats au dataframe
    df_result = df.loc[indices_valides].copy()
    df_result["cluster_kmeans"] = km_labels
    df_result["cluster_dbscan"] = db_labels
    df_result["is_anomaly"]     = (db_labels == -1).astype(int)
    df_result["pca_x"]          = X_2d[:, 0]
    df_result["pca_y"]          = X_2d[:, 1]

    # Statistiques par cluster pour interprétation business
    stats = (
        df_result.groupby("cluster_kmeans")
        .agg(
            nb_produits  = ("title",           "count"),
            prix_moyen   = ("price",           "mean"),
            note_moyenne = ("rating_filled",   "mean"),
            score_moyen  = ("composite_score", "mean"),
            remise_moy   = ("discount_pct",    "mean"),
        )
        .round(2)
    )
    print(f"\n  Stats par cluster :")
    print(stats.to_string())

    os.makedirs(output_dir, exist_ok=True)

    # CSV principal avec clusters
    cols = [
        "title", "shop_name", "source", "category_clean",
        "price", "price_tier", "rating_filled", "nb_reviews",
        "composite_score", "discount_pct",
        "cluster_kmeans", "cluster_dbscan", "is_anomaly",
        "pca_x", "pca_y"
    ]
    cols_ok = [c for c in cols if c in df_result.columns]
    df_result[cols_ok].to_csv(
        f"{output_dir}/clusters_kmeans.csv",
        index=False, encoding="utf-8-sig"
    )

    # CSV anomalies uniquement
    df_result[df_result["is_anomaly"] == 1][cols_ok].to_csv(
        f"{output_dir}/anomalies_dbscan.csv",
        index=False, encoding="utf-8-sig"
    )

    # JSON métriques
    stats_json = {
        "kmeans": {
            "k"               : K_CLUSTERS,
            "inertie"         : round(float(kmeans.inertia_), 2),
            "silhouette_score": round(float(silhouette), 4),
            "clusters"        : stats.reset_index().to_dict("records")
        },
        "dbscan": {
            "eps"          : 0.5,
            "min_samples"  : 5,
            "nb_clusters"  : int(len(set(db_labels) - {-1})),
            "nb_anomalies" : int(nb_anom),
            "pct_anomalies": round(float(pct_anom), 1)
        },
        "pca": {
            "n_components"       : 2,
            "variance_expliquee" : round(float(variance), 4)
        }
    }

    with open(f"{output_dir}/clustering_stats.json", "w") as f:
        json.dump(stats_json, f, indent=2, default=str)

    print(f"\n  Fichiers sauvegardés :")
    print(f"    {output_dir}/clusters_kmeans.csv")
    print(f"    {output_dir}/anomalies_dbscan.csv")
    print(f"    {output_dir}/clustering_stats.json")
    print("=" * 55)

    return df_result, stats_json


if __name__ == "__main__":
    run_clustering()
    print("\n✅ Clustering terminé")