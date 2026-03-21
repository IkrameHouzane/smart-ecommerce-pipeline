"""Page 5 — Segmentation : KMeans, DBSCAN, PCA"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os

st.set_page_config(page_title="Segmentation", page_icon="◉", layout="wide")

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
.cl-card{background:#0f0f18;border:1px solid #1e1e2e;border-radius:2px;padding:18px 16px;text-align:center;margin-bottom:8px;}
.cl-name{font-family:"DM Mono",monospace;font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;}
.cl-price{font-family:"DM Mono",monospace;font-size:1.8rem;font-weight:500;line-height:1;margin-bottom:6px;}
.cl-type{font-size:0.75rem;font-weight:600;margin-bottom:4px;}
.cl-desc{font-size:0.7rem;color:#4a4a6a;}
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
PAL_CLU = ["#e8d5a3", "#4ecdc4", "#ff6b6b", "#ffd93d"]

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

@st.cache_data
def load():
    clusters  = pd.read_csv(os.path.join(ANALYTICS, "clusters_kmeans.csv"))
    anomalies = pd.read_csv(os.path.join(ANALYTICS, "anomalies_dbscan.csv"))
    with open(os.path.join(ANALYTICS, "clustering_stats.json")) as f:
        stats = json.load(f)
    return clusters, anomalies, stats

clusters, anomalies, stats = load()
kmeans_stats = stats["kmeans"]
dbscan_stats = stats["dbscan"]
pca_stats    = stats["pca"]
cluster_data = kmeans_stats["clusters"]

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown(f'<div class="pg-main">Segmentation</div>'
    f'<div class="pg-sub">KMeans K=4 · DBSCAN {dbscan_stats["nb_anomalies"]} anomalies · '
    f'PCA variance expliquée {pca_stats["variance_expliquee"]*100:.1f}%</div>'
    f'<div style="border-bottom:1px solid #1a1a28;margin:16px 0 28px 0"></div>',
    unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi(f"{kmeans_stats['silhouette_score']:.3f}", "Silhouette Score", "#4ecdc4"), unsafe_allow_html=True)
c2.markdown(kpi(f"K = {kmeans_stats['k']}",               "Clusters KMeans",  "#e8d5a3"), unsafe_allow_html=True)
c3.markdown(kpi(f"{dbscan_stats['nb_anomalies']}",         "Anomalies DBSCAN", "#ff6b6b"), unsafe_allow_html=True)
c4.markdown(kpi(f"{pca_stats['variance_expliquee']*100:.1f}%","Variance PCA",  "#a8e6cf"), unsafe_allow_html=True)

# ── Insight ───────────────────────────────────────────────────────────────────
cd = cluster_data
st.markdown(ins("Segmentation Reading",
    f"KMeans partitioned <strong>{sum(d['nb_produits'] for d in cd):,} products</strong> into 4 natural segments. "
    f"Cluster 2 is the <strong>premium segment</strong> (avg ${cd[2]['prix_moyen']:.0f}, {cd[2]['remise_moy']:.0f}% discount). "
    f"Cluster 1 is the <strong>promo segment</strong> (avg ${cd[1]['prix_moyen']:.0f}, {cd[1]['remise_moy']:.0f}% discount). "
    f"DBSCAN flagged <strong>{dbscan_stats['nb_anomalies']} anomalies</strong> "
    f"({dbscan_stats['pct_anomalies']}% of catalog) — products with no dense neighborhood."
), unsafe_allow_html=True)

# ── PCA 2D scatter ────────────────────────────────────────────────────────────
st.markdown(sec("Visualisation KMeans — PCA 2D",
    f"2D projection — {pca_stats['variance_expliquee']*100:.1f}% variance explained. "
    f"Each dot is a product, colored by cluster assignment."), unsafe_allow_html=True)
clusters["cluster_label"] = "Cluster " + clusters["cluster_kmeans"].astype(str)
fig = px.scatter(clusters, x="pca_x", y="pca_y", color="cluster_label",
    color_discrete_sequence=PAL_CLU, template="plotly_dark",
    title="Clusters KMeans en espace PCA 2D", opacity=0.65,
    hover_data=["title", "price", "composite_score", "price_tier"])
fig.update_layout(**PLOT, height=480, legend_title="Cluster")
fig.update_traces(marker_size=4)
st.plotly_chart(fig, use_container_width=True)

# ── Profil des clusters ───────────────────────────────────────────────────────
st.markdown(sec("Profil des clusters",
    "Business interpretation of each segment — price, type, and discount signal."), unsafe_allow_html=True)
CTYPES  = ["Qualité moyenne", "Promos ~40%", "Premium", "Budget+"]
cols = st.columns(4)
for col, d, ctype, color in zip(cols, cluster_data, CTYPES, PAL_CLU):
    col.markdown(f"""<div class="cl-card">
        <div class="cl-name" style="color:{color}">Cluster {d['cluster_kmeans']}</div>
        <div class="cl-price" style="color:{color}">{d['prix_moyen']:.0f}€</div>
        <div class="cl-type" style="color:{color}">{ctype}</div>
        <div class="cl-desc">{d['nb_produits']} produits · remise {d['remise_moy']:.0f}%</div>
    </div>""", unsafe_allow_html=True)

# ── Distribution des clusters ─────────────────────────────────────────────────
st.markdown(sec("Distribution des clusters", "Product count and positioning by segment."), unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    dist = clusters["cluster_kmeans"].value_counts().sort_index()
    fig2 = px.bar(x=[f"Cluster {i}" for i in dist.index], y=dist.values,
        color=dist.values, text=dist.values,
        color_continuous_scale=["#1a1a28", "#4ecdc4", "#e8d5a3"],
        template="plotly_dark", title="Nombre de produits par cluster",
        labels={"x": "", "y": "Count"})
    fig2.update_traces(textposition="outside")
    fig2.update_layout(**PLOT, height=340, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    cp_df = pd.DataFrame([{
        "Cluster": f"Cluster {d['cluster_kmeans']}",
        "Prix moyen": d["prix_moyen"], "Score moyen": d["score_moyen"], "Note": d["note_moyenne"]
    } for d in cluster_data])
    fig3 = px.scatter(cp_df, x="Prix moyen", y="Score moyen", size="Note",
        color="Cluster", text="Cluster", color_discrete_sequence=PAL_CLU,
        template="plotly_dark", title="Prix moyen vs Score moyen par cluster", size_max=40)
    fig3.update_traces(textposition="top center")
    fig3.update_layout(**PLOT, height=340)
    st.plotly_chart(fig3, use_container_width=True)

# ── Anomalies DBSCAN ──────────────────────────────────────────────────────────
st.markdown(sec(
    f"Anomalies détectées par DBSCAN ({dbscan_stats['nb_anomalies']} produits)",
    f"{dbscan_stats['pct_anomalies']}% of catalog — products with no dense neighborhood."),
    unsafe_allow_html=True)
disp = [c for c in ["title", "shop_name", "price", "composite_score"] if c in anomalies.columns]
st.dataframe(anomalies[disp].sort_values("price", ascending=False).reset_index(drop=True),
    use_container_width=True, height=320)