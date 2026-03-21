"""Page 2 — Product Rankings"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Product Rankings", page_icon="🏆", layout="wide")

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
.met-row{background:#0f0f18;border:1px solid #1e1e2e;border-radius:2px;padding:16px 20px;text-align:center;margin-bottom:8px;}
.met-val{font-family:"DM Mono",monospace;font-size:1.6rem;font-weight:500;line-height:1.1;}
.met-lbl{font-family:"DM Mono",monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:2px;color:#4a4a6a;margin-top:4px;}
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

def ins(lbl, txt, col="#4ecdc4"):
    return f'<div class="ins-card" style="border-left-color:{col}"><div class="ins-lbl" style="color:{col}">{lbl}</div><div class="ins-txt">{txt}</div></div>'
def sec(title, sub="", col="#e8d5a3"):
    s = f'<div class="sec-sub">{sub}</div>' if sub else ""
    return f'<div class="sec-hdr"><div class="sec-title" style="color:{col}">{title}</div>{s}</div>'
def met(v, lbl, col="#e8d5a3"):
    return f'<div class="met-row"><div class="met-val" style="color:{col}">{v}</div><div class="met-lbl">{lbl}</div></div>'

st.markdown(CSS, unsafe_allow_html=True)

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA      = os.path.join(BASE, "..", "..", "data")
ANALYTICS = os.path.join(BASE, "..", "..", "analytics")

@st.cache_data
def load():
    topk   = pd.read_csv(os.path.join(DATA, "top_k_products.csv"))
    scored = pd.read_parquet(os.path.join(DATA, "scored_products.parquet"))
    catk   = pd.read_csv(os.path.join(ANALYTICS, "topk_per_category.csv"))
    return topk, scored, catk

topk, scored, catk = load()

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-main">Product Rankings</div>'
    '<div class="pg-sub">Showing ranked products from the current filtered view — '
    'scored on quality, availability, popularity and discount attractiveness.</div>'
    '<div style="border-bottom:1px solid #1a1a28;margin:16px 0 28px 0"></div>',
    unsafe_allow_html=True)

# ── Filtre boutique ───────────────────────────────────────────────────────────
boutiques   = ["Toutes"] + sorted(topk["shop_name"].unique().tolist())
shop_filter = st.selectbox("Filtrer par boutique", boutiques)
df = topk if shop_filter == "Toutes" else topk[topk["shop_name"] == shop_filter]

# ── Insight narrative ─────────────────────────────────────────────────────────
avg_s    = df["composite_score"].mean()
avg_r    = df["rating_filled"].mean()
n_shops  = df["shop_name"].nunique()
n_cats   = df["category_clean"].nunique() if "category_clean" in df.columns else "—"
pct_disc = (df["discount_pct"] > 10).mean() * 100

c1, c2, c3 = st.columns(3)
c1.markdown(ins("Commercial Reading",
    f"The current shortlist averages <strong>{avg_s:.3f}</strong> in score and "
    f"<strong>{avg_r:.2f}</strong> in rating."), unsafe_allow_html=True)
c2.markdown(ins("Availability Posture",
    f"<strong>{(df['available'].mean()*100):.1f}%</strong> of visible products are in stock, "
    f"while <strong>{pct_disc:.1f}%</strong> carry a discount signal.", "#ffd93d"), unsafe_allow_html=True)
c3.markdown(ins("Filter Impact",
    f"This view spans <strong>{n_shops} shops</strong> and <strong>{n_cats} categories</strong>, "
    f"useful as a broad leaderboard or a niche buying slice.", "#ff6b6b"), unsafe_allow_html=True)

# ── Top 10 produits ───────────────────────────────────────────────────────────
st.markdown(sec("Top 10 Produits",
    "Ordered by composite score — 4 decimal precision to show real differentiation."), unsafe_allow_html=True)
top10 = df.nlargest(10, "composite_score").copy()
top10["titre_court"] = top10["title"].str[:50]
top10["score_label"] = top10["composite_score"].apply(lambda x: f"{x:.4f}")

fig = px.bar(top10, x="composite_score", y="titre_court", orientation="h",
    color="composite_score", text="score_label",
    color_continuous_scale=["#1a1a28", "#2a2a4a", "#4ecdc4", "#e8d5a3"],
    template="plotly_dark", title="Top 10 produits par score composite",
    hover_data=["shop_name", "price", "rating_filled", "nb_variants", "discount_pct"],
    labels={"titre_court": "", "composite_score": "Score composite"})
fig.update_traces(textposition="outside")
plot2 = {**PLOT, "yaxis": dict(autorange="reversed", title="", gridcolor="#1a1a28",
    tickfont=dict(size=10, color="#7070a0"))}
fig.update_layout(**plot2, height=440, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# ── Métriques commerciales du Top-10 ─────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.markdown(met(f"${top10['price'].mean():.2f}", "Prix moyen Top-10", "#e8d5a3"), unsafe_allow_html=True)
c2.markdown(met(f"{top10['rating_filled'].mean():.2f} ★", "Note moyenne Top-10", "#4ecdc4"), unsafe_allow_html=True)
c3.markdown(met(f"{top10['discount_pct'].mean():.1f}%", "Remise moyenne Top-10", "#ffd93d"), unsafe_allow_html=True)

# ── Score vs Prix ─────────────────────────────────────────────────────────────
st.markdown(sec("Score vs Prix",
    "Each dot is a product — ideal products cluster top-left: high score, low price."), unsafe_allow_html=True)
sample = scored.sample(min(600, len(scored)), random_state=42)
fig2 = px.scatter(sample, x="price", y="composite_score", color="shop_name",
    template="plotly_dark", opacity=0.7,
    title="Score composite vs Prix par boutique",
    hover_data=["title", "price_tier"])
fig2.update_layout(**PLOT, xaxis_range=[0, 300], height=420)
st.plotly_chart(fig2, use_container_width=True)

# ── Top par catégorie ─────────────────────────────────────────────────────────
st.markdown(sec("Top produits par catégorie",
    "Select a segment to inspect the top ranked products."), unsafe_allow_html=True)
cats = sorted(catk["category_clean"].unique().tolist()) if "category_clean" in catk.columns else []
if cats:
    cat_sel = st.selectbox("Catégorie", cats)
    cat_df  = catk[catk["category_clean"] == cat_sel].nlargest(10, "composite_score")
    st.dataframe(cat_df[["title", "shop_name", "price", "rating_filled", "composite_score"]]
        .reset_index(drop=True), use_container_width=True, height=300)

# ── Tableau Top-K complet ─────────────────────────────────────────────────────
st.markdown(sec("Tableau Top-K complet"), unsafe_allow_html=True)
st.dataframe(df[["title", "shop_name", "price", "rating_filled", "composite_score"]]
    .sort_values("composite_score", ascending=False).reset_index(drop=True),
    use_container_width=True, height=400)