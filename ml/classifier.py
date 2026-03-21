# ml/classifier.py
# ─────────────────────────────────────────────────────────
# PARTIE A — Classification supervisée RF + XGBoost.
#
# CE QUE L'ÉNONCÉ DEMANDE :
#   "Random Forest, XGBoost pour classer les produits"
#   "Train/test split ou validation croisée"
#   "Accuracy, précision, rappel, F1, matrice de confusion"
#   "Feature importance"
#
# VARIABLE CIBLE : price_tier (budget / mid_range / premium)
# On prédit la catégorie de prix d'un produit à partir
# de ses autres caractéristiques — sans utiliser le prix
# lui-même (sinon le modèle tricherait).
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json
import os

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
import xgboost as xgb

INPUT_PATH  = "../data/scored_products.parquet"
OUTPUT_DIR  = "../analytics"


def preparer_features(df: pd.DataFrame):
    """
    Sélectionne les colonnes pour la classification.

    On N'utilise PAS : price, price_min, price_max, compare_price
    → trop corrélés à price_tier, le modèle apprendrait juste
      "prix < 30 = budget" sans comprendre les vraies features.
    """
    features = [
        "discount_pct",        # % de remise
        "rating_filled",       # note (imputée si manquante)
        "has_rating",          # 1 si vraie note, 0 si imputée
        "nb_reviews",          # nombre d'avis
        "nb_variants",         # nombre de variantes
        "nb_images",           # nombre d'images produit
        "available",           # en stock ou non
        "catalogue_richness",  # score richesse catalogue
        "popularity_score",    # popularité combinée
        "composite_score",     # score final étape 2
    ]
    target = "price_tier"

    features_ok = [f for f in features if f in df.columns]
    df_ml = df[features_ok + [target]].dropna()
    df_ml["available"] = df_ml["available"].astype(float)

    print(f"  → {len(features_ok)} features · {len(df_ml)} produits")
    print(f"  → Distribution cible :")
    print(df_ml[target].value_counts().to_string())

    X = df_ml[features_ok].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_ml[target].astype(str).values)

    return X, y, features_ok, encoder


def entrainer_random_forest(X_train, X_test, y_train, y_test,
                             feature_names, encoder):
    """
    Random Forest : forêt de 300 arbres de décision.
    Chaque arbre vote, la majorité l'emporte.
    Robuste, résistant au surapprentissage.
    """
    print("\n  --- Random Forest ---")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"  → Accuracy : {acc*100:.1f}%")
    print(classification_report(
        y_test, y_pred, target_names=encoder.classes_
    ))
    print(f"  → Matrice de confusion :\n{cm}")

    # Validation croisée 5-fold (comme demandé dans l'énoncé)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5,
                                scoring="accuracy", n_jobs=-1)
    print(f"  → Cross-val (5-fold) : "
          f"{cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # Feature importance : quelles variables influencent le plus ?
    importances = pd.DataFrame({
        "feature"   : feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  → Top 5 features :")
    print(importances.head(5).to_string(index=False))

    rapport = classification_report(
        y_test, y_pred,
        target_names=encoder.classes_,
        output_dict=True
    )

    return rf, acc, rapport, importances, cm.tolist(), cv_scores.mean()


def entrainer_xgboost(X_train, X_test, y_train, y_test,
                      feature_names, encoder):
    """
    XGBoost : chaque arbre corrige les erreurs du précédent.
    Principe du boosting = renforcer les faibles.
    Plus précis que RF, standard dans les compétitions ML.
    """
    print("\n  --- XGBoost ---")

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"  → Accuracy : {acc*100:.1f}%")
    print(classification_report(
        y_test, y_pred, target_names=encoder.classes_
    ))
    print(f"  → Matrice de confusion :\n{cm}")

    importances = pd.DataFrame({
        "feature"   : feature_names,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  → Top 5 features :")
    print(importances.head(5).to_string(index=False))

    rapport = classification_report(
        y_test, y_pred,
        target_names=encoder.classes_,
        output_dict=True
    )

    return xgb_model, acc, rapport, importances, cm.tolist()


def run_classifier(input_path=INPUT_PATH, output_dir=OUTPUT_DIR):

    print("=" * 55)
    print("  3A — CLASSIFICATION RF + XGBOOST")
    print("=" * 55)

    df = pd.read_parquet(input_path)
    print(f"\n  Produits chargés : {len(df)}")

    print("\n[1/4] Préparation des features...")
    X, y, feature_names, encoder = preparer_features(df)

    print("\n[2/4] Split train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  → Train : {len(X_train)}  |  Test : {len(X_test)}")

    print("\n[3/4] Entraînement...")
    rf_model, rf_acc, rf_rapport, rf_imp, rf_cm, rf_cv = \
        entrainer_random_forest(
            X_train, X_test, y_train, y_test, feature_names, encoder
        )

    xgb_model, xgb_acc, xgb_rapport, xgb_imp, xgb_cm = \
        entrainer_xgboost(
            X_train, X_test, y_train, y_test, feature_names, encoder
        )

    print("\n[4/4] Sauvegarde...")
    os.makedirs(output_dir, exist_ok=True)

    # JSON avec toutes les métriques
    resultats = {
        "random_forest": {
            "accuracy"               : round(rf_acc, 4),
            "cross_val_mean_accuracy": round(rf_cv, 4),
            "confusion_matrix"       : rf_cm,
            "classification_report"  : rf_rapport
        },
        "xgboost": {
            "accuracy"              : round(xgb_acc, 4),
            "confusion_matrix"      : xgb_cm,
            "classification_report" : xgb_rapport
        },
        "meilleur_modele" : "xgboost" if xgb_acc >= rf_acc else "random_forest",
        "classes"         : list(encoder.classes_),
        "nb_features"     : len(feature_names),
        "features"        : feature_names
    }

    with open(f"{output_dir}/ml_classification.json", "w") as f:
        json.dump(resultats, f, indent=2, default=str)

    # Feature importances des deux modèles en CSV
    rf_imp["model"]  = "random_forest"
    xgb_imp["model"] = "xgboost"
    pd.concat([rf_imp, xgb_imp]).to_csv(
        f"{output_dir}/feature_importance.csv",
        index=False, encoding="utf-8-sig"
    )

    print(f"\n  RF  accuracy : {rf_acc*100:.1f}%")
    print(f"  XGB accuracy : {xgb_acc*100:.1f}%")
    print(f"  Meilleur     : {resultats['meilleur_modele']}")
    print(f"  Sauvegardé   : {output_dir}/ml_classification.json")
    print("=" * 55)

    return rf_model, xgb_model, resultats


if __name__ == "__main__":
    run_classifier()
    print("\n✅ Classification terminée")