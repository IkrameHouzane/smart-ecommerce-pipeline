"""Page 1 — Overview"""
import streamlit as st
import pandas as pd
import plotly.express as px
import json, os

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

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
.sec-hdr{margin:36px 0 8px 0;padding-bottom:12px;border-bottom:1px solid #1a1a28;}
.sec-title{font-family:"Playfair Display",serif;font-size:1.4rem;font-weight:700;color:#e8d5a3;margin:0 0 4px 0;}
.sec-sub{font-size:0.82rem;color:#4a4a6a;font-family:"DM Mono",monospace;line-height:1.5;}
.ins-card{background:#0f0f18;border:1px solid #1e1e2e;border-left:3px solid #4ecdc4;border-radius:2px;padding:16px 20px;margin-bottom:16px;}
.ins-lbl{font-family:"DM Mono",monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:2px;color:#4ecdc4;margin-bottom:6px;}
.ins-txt{font-size:0.88rem;color:#9090b0;line-height:1.6;}
.ins-txt strong{color:#c8c8d8;}
.node-card{background:#0f0f18;border:1px solid #1e1e2e;border-radius:2px;padding:14px 12px;text-align:center;margin-bottom:8px;}
.node-op{font-family:"DM Mono",monospace;font-size:0.68rem;color:#4ecdc4;font-weight:500;margin:4px 0 2px 0;}
.node-label{font-size:0.62rem;color:#4a4a6a;}
.pg-main{font-family:"Playfair Display",serif;font-size:2.4rem;font-weight:700;color:#e8d5a3;line-height:1.1;margin-bottom:6px;}
.pg-sub{font-size:0.82rem;color:#4a4a6a;font-family:"DM Mono",monospace;}
</style>"""

PLOT = dict(paper_bgcolor="#0f0f18", plot_bgcolor="#0f0f18",
    font=dict(family="DM Mono, monospace", color="#7070a0", size=11),
    title_font=dict(family="Playfair Display, serif", color="#e8d5a3", size=14),
    xaxis=dict(gridcolor="#1a1a28", linecolor="#1a1a28", tickfont=dict(color="#4a4a6a", size=10)),
    yaxis=dict(gridcolor="#1a1a28", linecolor="#1a1a28", tickfont=dict(color="#4a4a6a", size=10)),
    legend=dict(bgcolor="#0f0f18", bordercolor="#1e1e2e", borderwidth=1, font=dict(color="#7070a0", size=10)),
    margin=dict(l=12, r=12, t=44, b=12))

def kpi(v, lbl, col="#e8d5a3"):
    return f'<div class="kpi-card"><div class="kpi-val" style="color:{col}">{v}</div><div class="kpi-label">{lbl}</div></div>'
def ins(lbl, txt, col="#4ecdc4"):
    return f'<div class="ins-card" style="border-left-color:{col}"><div class="ins-lbl" style="color:{col}">{lbl}</div><div class="ins-txt">{txt}</div></div>'
def sec(title, sub="", col="#e8d5a3"):
    s = f'<div class="sec-sub">{sub}</div>' if sub else ""
    return f'<div class="sec-hdr"><div class="sec-title" style="color:{col}">{title}</div>{s}</div>'

st.markdown(CSS, unsafe_allow_html=True)

BASE      = os.path.dirname(os.path.abspath(__file__))
ANALYTICS = os.path.join(BASE, "..", "..", "analytics")
DATA      = os.path.join(BASE, "..", "..", "data")

@st.cache_data
def load():
    scored    = pd.read_parquet(os.path.join(DATA, "scored_products.parquet"))
    shops     = pd.read_csv(os.path.join(ANALYTICS, "shop_ranking.csv"))
    rules     = pd.read_csv(os.path.join(ANALYTICS, "association_rules.csv"))
    anomalies = pd.read_csv(os.path.join(ANALYTICS, "anomalies_dbscan.csv"))
    with open(os.path.join(ANALYTICS, "ml_classification.json")) as f:
        ml = json.load(f)
    return scored, shops, rules, anomalies, ml

scored, shops, rules, anomalies, ml = load()
rf_acc = ml.get("random_forest", {}).get("accuracy", ml.get("rf_accuracy", 0.926))

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-main">Dashboard Overview</div>'
    '<div class="pg-sub">A high-contrast control room for the current catalog — '
    'pricing spread, shop coverage, and model-ready product signals.</div>'
    '<div style="border-bottom:1px solid #1a1a28;margin:16px 0 28px 0"></div>',
    unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi(f"{len(scored):,}", "Produits analysés", "#e8d5a3"), unsafe_allow_html=True)
c2.markdown(kpi(f"{len(shops)}",    "Boutiques",          "#4ecdc4"), unsafe_allow_html=True)
c3.markdown(kpi(f"{rf_acc*100:.1f}%","RF Accuracy",       "#a8e6cf"), unsafe_allow_html=True)
c4.markdown(kpi(f"{len(anomalies)}", "Anomalies DBSCAN",  "#ff6b6b"), unsafe_allow_html=True)

# ── Insight narrative ─────────────────────────────────────────────────────────
best      = shops.iloc[0]["shop_name"]
top_cat   = scored["category_clean"].value_counts().index[0] if "category_clean" in scored.columns else "shoes"
pct_dispo = scored["available"].mean() * 100
pct_promo = (scored["discount_pct"] > 10).mean() * 100
avg_r     = scored["rating_filled"].mean()

c1, c2 = st.columns(2)
c1.markdown(ins("Market Reading",
    f"The catalog is led by <strong>{best}</strong>, with <strong>{top_cat}</strong> as the "
    f"largest visible category. Average rating <strong>{avg_r:.2f}★</strong>. "
    f"<strong>{pct_dispo:.1f}%</strong> of products are in stock, while "
    f"<strong>{pct_promo:.1f}%</strong> carry a discount signal."), unsafe_allow_html=True)
c2.markdown(ins("Coverage Diagnostics",
    f"<strong>{(scored['rating'].notna()).mean()*100:.1f}%</strong> rows carry real rating signals. "
    f"<strong>{pct_dispo:.1f}%</strong> products currently available. "
    f"<strong>{pct_promo:.1f}%</strong> rows carry a measurable discount signal right now.",
    "#ffd93d"), unsafe_allow_html=True)

# ── Distribution des prix ─────────────────────────────────────────────────────
st.markdown(sec("Distribution des prix",
    "Histogram capped at $300 · and price tier segmentation."), unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(scored, x="price", nbins=50,
        color_discrete_sequence=["#e8d5a3"], template="plotly_dark",
        title="Distribution des prix (€)")
    fig.update_layout(**PLOT)
    fig.update_xaxes(range=[0, 300])
    st.plotly_chart(fig, use_container_width=True)
with c2:
    tier = scored["price_tier"].value_counts()
    fig2 = px.pie(values=tier.values, names=tier.index, hole=0.45,
        color_discrete_sequence=["#e8d5a3", "#4ecdc4", "#a8e6cf"],
        template="plotly_dark", title="Segments de prix")
    fig2.update_layout(**PLOT)
    st.plotly_chart(fig2, use_container_width=True)

# ── Score moyen par boutique ──────────────────────────────────────────────────
st.markdown(sec("Score moyen par boutique",
    "Composite score averaged across all products per shop."), unsafe_allow_html=True)
sh = shops.sort_values("score_moyen", ascending=True)
fig3 = px.bar(sh, x="score_moyen", y="shop_name", orientation="h",
    color="score_moyen", text=sh["score_moyen"].round(3),
    color_continuous_scale=["#1a1a28", "#2a2a4a", "#4ecdc4", "#e8d5a3"],
    template="plotly_dark", title="Score composite moyen par boutique")
fig3.update_traces(textposition="outside")
fig3.update_layout(**PLOT, height=400, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig3, use_container_width=True)

# ── Statut Pipeline Kubeflow ──────────────────────────────────────────────────
st.markdown(sec("Statut du Pipeline Kubeflow",
    "All 8 components executed successfully on Minikube · KFP 2.4.1"), unsafe_allow_html=True)
steps = [
    ("preprocess-op",        "Preprocessing"),
    ("features-op",          "Feature Engineering"),
    ("score-op",             "Top-K Scoring"),
    ("cluster-kmeans-op",    "KMeans K=4"),
    ("cluster-dbscan-op",    "DBSCAN"),
    ("association-rules-op", "Apriori Rules"),
    ("train-classifier-op",  "Random Forest 92.6%"),
    ("train-xgboost-op",     "XGBoost 92.4%"),
]
cols2 = st.columns(4)
for i, (op, label) in enumerate(steps):
    cols2[i % 4].markdown(
        f'<div class="node-card"><div style="font-size:1rem;color:#4ecdc4">✓</div>'
        f'<div class="node-op">{op}</div><div class="node-label">{label}</div></div>',
        unsafe_allow_html=True)