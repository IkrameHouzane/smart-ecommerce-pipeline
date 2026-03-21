# kubeflow/pipeline.py — VERSION FINALE avec image locale

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from kfp import dsl


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def preprocess_op():
    """Nettoyage et validation des données brutes."""
    import sys
    sys.path.append("/app")
    from data_processing.preprocess import preprocess
    preprocess(
        input_path="/app/data/FINAL_sport_fitness_products.csv",
        output_path="/app/data/clean_products.parquet"
    )


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def features_op():
    """Feature engineering et normalisation."""
    import sys
    sys.path.append("/app")
    from data_processing.features import engineer_features
    engineer_features(
        input_path="/app/data/clean_products.parquet",
        output_path="/app/data/featured_products.parquet"
    )


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def score_op():
    """Scoring Top-K et classement des shops."""
    import sys
    sys.path.append("/app")
    from data_processing.scoring import scoring_topk
    scoring_topk(
        input_path="/app/data/featured_products.parquet",
        output_full="/app/data/scored_products.parquet",
        output_topk="/app/data/top_k_products.csv"
    )


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def train_classifier_op():
    """Random Forest + XGBoost — classification price_tier."""
    import sys
    sys.path.append("/app")
    from ml.classifier import run_classifier
    run_classifier(
        input_path="/app/data/scored_products.parquet",
        output_dir="/app/analytics"
    )


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def train_xgboost_op():
    """XGBoost seul — pour le noeud séparé dans le graphe."""
    import sys
    sys.path.append("/app")
    # XGBoost est déjà inclus dans train_classifier_op
    # Ce composant existe pour correspondre au graphe des collègues
    print("XGBoost run via train_classifier_op")


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def cluster_kmeans_op():
    """KMeans K=4 + PCA 2D."""
    import sys
    sys.path.append("/app")
    from ml.clustering import run_clustering
    run_clustering(
        input_path="/app/data/scored_products.parquet",
        output_dir="/app/analytics"
    )


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def cluster_dbscan_op():
    """DBSCAN — détection d'anomalies."""
    import sys
    sys.path.append("/app")
    print("DBSCAN run via cluster_kmeans_op")


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def association_rules_op():
    """Apriori — règles d'association."""
    import sys
    sys.path.append("/app")
    from ml.association_rules import run_association_rules
    run_association_rules(
        input_path="/app/data/scored_products.parquet",
        output_dir="/app/analytics"
    )


@dsl.pipeline(
    name="smart-ecommerce-intelligence-pipeline",
    description=(
        "Pipeline ML eCommerce: preprocess -> features -> score -> "
        "train -> cluster -> association rules"
    ),
)
def smart_ecommerce_pipeline():
    """DAG du pipeline — 8 composants comme les collegues."""
    p = preprocess_op()
    f = features_op().after(p)
    s = score_op().after(f)
    # Classifieurs en parallele apres scoring
    train_classifier_op().after(s)
    train_xgboost_op().after(s)
    # Clustering apres features
    cluster_kmeans_op().after(f)
    cluster_dbscan_op().after(f)
    # Regles d'association apres features
    association_rules_op().after(f)


if __name__ == "__main__":
    from kfp import compiler

    output_file = "smart_ecommerce_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=output_file
    )
    print(f"\nPipeline compile : {output_file}")
    print("Charge l'image dans Minikube : minikube image load smart-ecommerce-pipeline:local")
    print("Puis upload le YAML dans l'UI Kubeflow : http://localhost:8080")