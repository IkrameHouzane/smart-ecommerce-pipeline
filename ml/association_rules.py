# ml/association_rules.py
# ─────────────────────────────────────────────────────────
# PARTIE A — Règles d'association (algorithme Apriori).
#
# CE QUE L'ÉNONCÉ DEMANDE :
#   "Règles d'association validées avec support,
#    confidence et lift"
#
# EXEMPLE DE RÈGLE :
#   [prix_premium, bcp_variants] → [score_excellent]
#   confidence=0.78, lift=2.1, support=0.12
#   Lecture : "78% des produits premium avec beaucoup
#   de variantes ont un excellent score. C'est 2.1×
#   plus fréquent que par hasard."
#
# TERMINOLOGIE :
#   Support    = fréquence de la combinaison (ex: 12%)
#   Confidence = P(B sachant A)              (ex: 78%)
#   Lift       = amélioration vs hasard      (ex: 2.1×)
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json
import os

from mlxtend.frequent_patterns import apriori, association_rules

INPUT_PATH  = "../data/scored_products.parquet"
OUTPUT_DIR  = "../analytics"

MIN_SUPPORT    = 0.05
MIN_CONFIDENCE = 0.50
MIN_LIFT       = 1.20


def discretiser_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes numériques en items discrets booléens.
    Apriori travaille sur des items binaires (vrai/faux),
    pas sur des nombres continus.

    On crée des items lisibles :
      prix_budget / prix_mid / prix_premium
      note_haute / note_moyenne / note_basse
      en_promo / sans_promo
      dispo / indispo
      bcp_variants / peu_variants
      bcp_images / peu_images
      score_excellent / score_bon / score_faible
    """
    items = pd.DataFrame(index=df.index)

    if "price_tier" in df.columns:
        tier = df["price_tier"].astype(str)
        items["prix_budget"]  = (tier == "budget").astype(bool)
        items["prix_mid"]     = (tier == "mid_range").astype(bool)
        items["prix_premium"] = (tier == "premium").astype(bool)

    if "rating_filled" in df.columns:
        r = df["rating_filled"]
        items["note_haute"]   = (r >= 4.5).astype(bool)
        items["note_moyenne"] = ((r >= 3.5) & (r < 4.5)).astype(bool)
        items["note_basse"]   = (r < 3.5).astype(bool)

    if "discount_pct" in df.columns:
        items["en_promo"]   = (df["discount_pct"] > 10).astype(bool)
        items["sans_promo"] = (df["discount_pct"] <= 10).astype(bool)

    if "available" in df.columns:
        dispo = df["available"].astype(float)
        items["dispo"]   = (dispo == 1).astype(bool)
        items["indispo"] = (dispo == 0).astype(bool)

    if "nb_variants" in df.columns:
        items["bcp_variants"] = (df["nb_variants"] > 5).astype(bool)
        items["peu_variants"] = (df["nb_variants"] <= 5).astype(bool)

    if "nb_images" in df.columns:
        items["bcp_images"] = (df["nb_images"] > 3).astype(bool)
        items["peu_images"] = (df["nb_images"] <= 3).astype(bool)

    if "composite_score" in df.columns:
        s = df["composite_score"]
        items["score_excellent"] = (s > 0.80).astype(bool)
        items["score_bon"]       = ((s >= 0.60) & (s <= 0.80)).astype(bool)
        items["score_faible"]    = (s < 0.60).astype(bool)

    return items


def run_association_rules(input_path=INPUT_PATH, output_dir=OUTPUT_DIR):

    print("=" * 55)
    print("  3C — RÈGLES D'ASSOCIATION (APRIORI)")
    print("=" * 55)

    df = pd.read_parquet(input_path)
    print(f"\n  Produits chargés : {len(df)}")

    print("\n[1/4] Discrétisation des features...")
    items_df = discretiser_features(df)
    print(f"  → {len(items_df.columns)} items : {list(items_df.columns)}")

    print(f"\n[2/4] Apriori (support min={MIN_SUPPORT})...")
    frequent_itemsets = apriori(
        items_df,
        min_support=MIN_SUPPORT,
        use_colnames=True
    )
    print(f"  → {len(frequent_itemsets)} itemsets fréquents")

    if len(frequent_itemsets) == 0:
        print("  ⚠️  Aucun itemset — essaye de baisser MIN_SUPPORT")
        return None, None

    print(f"\n[3/4] Génération des règles "
          f"(confidence≥{MIN_CONFIDENCE}, lift≥{MIN_LIFT})...")
    regles = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )
    regles = regles[regles["lift"] >= MIN_LIFT]
    regles = regles.sort_values("confidence", ascending=False)
    print(f"  → {len(regles)} règles trouvées")

    if len(regles) > 0:
        print(f"\n  TOP 10 RÈGLES :")
        print("  " + "─" * 55)
        for _, row in regles.head(10).iterrows():
            ant = ", ".join(list(row["antecedents"]))
            con = ", ".join(list(row["consequents"]))
            print(
                f"  [{ant}]"
                f"\n    → [{con}]"
                f"  conf={row['confidence']:.2f}"
                f"  lift={row['lift']:.2f}"
                f"  supp={row['support']:.2f}"
            )

    print(f"\n[4/4] Sauvegarde...")
    os.makedirs(output_dir, exist_ok=True)

    # Formate les frozensets en strings pour le CSV
    regles_csv = regles.copy()
    regles_csv["antecedents"] = regles_csv["antecedents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    regles_csv["consequents"] = regles_csv["consequents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    regles_csv[[
        "antecedents", "consequents",
        "support", "confidence", "lift",
        "leverage", "conviction"
    ]].round(4).to_csv(
        f"{output_dir}/association_rules.csv",
        index=False, encoding="utf-8-sig"
    )

    summary = {
        "parametres": {
            "min_support"   : MIN_SUPPORT,
            "min_confidence": MIN_CONFIDENCE,
            "min_lift"      : MIN_LIFT
        },
        "n_itemsets" : int(len(frequent_itemsets)),
        "n_regles"   : int(len(regles)),
        "top_regles" : regles_csv.head(10).to_dict("records")
    }
    with open(f"{output_dir}/association_rules_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  → {output_dir}/association_rules.csv")
    print(f"  → {output_dir}/association_rules_summary.json")
    print("=" * 55)

    return frequent_itemsets, regles


if __name__ == "__main__":
    run_association_rules()
    print("\n✅ Règles d'association terminées")