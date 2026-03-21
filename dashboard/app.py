"""Smart eCommerce Intelligence Pipeline — Home"""

import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:"DM Sans",sans-serif;background:#0c0c12;color:#c8c8d8;}
.stApp{background:#0c0c12;}
section[data-testid="stSidebar"]{background:#08080e!important;border-right:1px solid #1a1a28!important;}
section[data-testid="stSidebar"] *{color:#7070a0!important;}
section[data-testid="stSidebar"] a{color:#7070a0!important;font-size:0.85rem!important;padding:8px 16px!important;border-radius:4px!important;display:block!important;margin:1px 8px!important;text-decoration:none!important;}
section[data-testid="stSidebar"] a:hover{color:#c8c8d8!important;background:#14141e!important;}
section[data-testid="stSidebar"] a[aria-current="page"]{color:#e8d5a3!important;background:#1a1820!important;border-left:2px solid #e8d5a3!important;padding-left:14px!important;font-weight:500!important;}
section[data-testid="stSidebar"] span,section[data-testid="stSidebar"] li{color:#7070a0!important;}
h1,h2,h3{font-family:"Playfair Display",serif!important;color:#e8d5a3!important;}
.kpi-card{background:#0f0f18;border:1px solid #1e1e2e;border-top:2px solid #e8d5a3;border-radius:2px;padding:24px 20px 20px;margin-bottom:8px;}
.kpi-val{font-family:"DM Mono",monospace;font-size:2.4rem;font-weight:500;line-height:1;margin-bottom:6px;}
.kpi-label{font-family:"DM Mono",monospace;font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;color:#4a4a6a;}
.ins-card{background:#0f0f18;border:1px solid #1e1e2e;border-left:3px solid #4ecdc4;border-radius:2px;padding:16px 20px;margin-bottom:16px;}
.ins-lbl{font-family:"DM Mono",monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:2px;color:#4ecdc4;margin-bottom:6px;}
.ins-txt{font-size:0.88rem;color:#9090b0;line-height:1.6;}
.ins-txt strong{color:#c8c8d8;}
.nav-card{background:#0f0f18;border:1px solid #1e1e2e;border-radius:2px;padding:20px;margin-bottom:8px;}
.nav-title{font-family:"Playfair Display",serif;font-size:1rem;font-weight:600;color:#e8d5a3;margin-bottom:4px;}
.nav-desc{font-size:0.78rem;color:#4a4a6a;}
</style>"""


def kpi(v, lbl, col="#e8d5a3"):
    return f'<div class="kpi-card"><div class="kpi-val" style="color:{col}">{v}</div><div class="kpi-label">{lbl}</div></div>'


def ins(lbl, txt, col="#4ecdc4"):
    return f'<div class="ins-card" style="border-left-color:{col}"><div class="ins-lbl" style="color:{col}">{lbl}</div><div class="ins-txt">{txt}</div></div>'


st.markdown(CSS, unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
ANALYTICS = os.path.join(BASE, "..", "analytics")
DATA = os.path.join(BASE, "..", "data")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="padding:48px 0 32px 0;border-bottom:1px solid #1a1a28;margin-bottom:36px;">
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3a3a5c;
                text-transform:uppercase;letter-spacing:3px;margin-bottom:16px;">
        FST Tanger &nbsp;·&nbsp; LSI 2M &nbsp;·&nbsp; 2025/2026
    </div>
    <div style="font-family:'Playfair Display',serif;font-size:3.2rem;font-weight:900;
                color:#e8d5a3;line-height:1.05;margin-bottom:12px;">
        Smart eCommerce<br>Intelligence Pipeline
    </div>
    <div style="font-size:0.9rem;color:#4a4a6a;max-width:580px;line-height:1.7;">
        An end-to-end data engineering pipeline collecting Sport &amp; Fitness product
        data from Shopify and WooCommerce — enriched with ML models, Kubeflow
        orchestration, and real-time analytics across 11 boutiques.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
try:
    topk = pd.read_csv(os.path.join(DATA, "top_k_products.csv"))
    shops = pd.read_csv(os.path.join(ANALYTICS, "shop_ranking.csv"))
    rules = pd.read_csv(os.path.join(ANALYTICS, "association_rules.csv"))
    with open(os.path.join(ANALYTICS, "ml_classification.json")) as f:
        ml = json.load(f)
    scored = pd.read_parquet(os.path.join(DATA, "scored_products.parquet"))
    rf_acc = ml.get("random_forest", {}).get("accuracy", ml.get("rf_accuracy", 0.926))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        kpi(f"{len(scored):,}", "Produits analysés", "#e8d5a3"), unsafe_allow_html=True
    )
    c2.markdown(kpi(f"{len(shops)}", "Boutiques", "#4ecdc4"), unsafe_allow_html=True)
    c3.markdown(
        kpi(f"{rf_acc * 100:.1f}%", "RF Accuracy", "#a8e6cf"), unsafe_allow_html=True
    )
    c4.markdown(
        kpi(f"{len(topk)}", "Top-K Produits", "#ffd93d"), unsafe_allow_html=True
    )
    c5.markdown(
        kpi(f"{len(rules):,}", "Règles Apriori", "#ff6b6b"), unsafe_allow_html=True
    )

    st.markdown(
        "<div style='margin:28px 0;border-top:1px solid #1a1a28'></div>",
        unsafe_allow_html=True,
    )

    best_shop = shops.iloc[0]["shop_name"]
    best_score = shops.iloc[0]["score_moyen"]
    avg_price = scored["price"].median()
    st.markdown(
        ins(
            "Pipeline State",
            f"The catalog is currently led by <strong>{best_shop}</strong> with a composite score of "
            f"<strong>{best_score:.3f}</strong>. The dataset spans <strong>{len(scored):,} products</strong> "
            f"across <strong>{len(shops)} shops</strong>, with a median price of <strong>${avg_price:.0f}</strong>. "
            f"ML models achieve <strong>{rf_acc * 100:.1f}% accuracy</strong> on price tier classification.",
        ),
        unsafe_allow_html=True,
    )

except Exception as e:
    st.error(f"Erreur chargement : {e}")

# ── Navigation ────────────────────────────────────────────────────────────────
st.markdown(
    """<div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#3a3a5c;
    text-transform:uppercase;letter-spacing:3px;margin-bottom:20px;">Navigation</div>""",
    unsafe_allow_html=True,
)

pages = [
    ("📊", "Overview", "KPIs globaux, distribution des prix, statut Kubeflow"),
    ("🏆", "Product Rankings", "Top-K produits scorés, analyse par catégorie"),
    ("🏪", "Shop Analysis", "Classement des 11 boutiques, métriques comparatives"),
    ("🤖", "ML Models", "RF 92.6% · XGBoost 92.4% · Feature importance"),
    ("◉", "Clustering", "KMeans K=4 · DBSCAN 106 anomalies · PCA 2D"),
    ("⊗", "Association Rules", "4 471 règles Apriori · lift max 12.09"),
]
c1, c2, c3 = st.columns(3)
for i, (icon, title, desc) in enumerate(pages):
    [c1, c2, c3][i % 3].markdown(
        f"""
    <div class="nav-card">
        <div style="font-size:1.4rem;margin-bottom:8px">{icon}</div>
        <div class="nav-title">{title}</div>
        <div class="nav-desc">{desc}</div>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    """<div style="margin-top:48px;padding-top:20px;border-top:1px solid #1a1a28;
    font-family:'DM Mono',monospace;font-size:0.62rem;color:#2a2a3a;">
    Kubeflow KFP 2.4.1 &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp;
    mlxtend &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; A2A Scraping Architecture
</div>""",
    unsafe_allow_html=True,
)
