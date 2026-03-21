"""Page 3 — Shop Analysis"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Shop Analysis", page_icon="🏪", layout="wide")

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
.pg-main{font-family:"Playfair Display",serif;font-size:2.4rem;font-weight:700;color:#e8d5a3;line-height:1.1;margin-bottom:6px;}
.pg-sub{font-size:0.82rem;color:#4a4a6a;font-family:"DM Mono",monospace;}
</style>"""

PLOT = dict(
    paper_bgcolor="#0f0f18",
    plot_bgcolor="#0f0f18",
    font=dict(family="DM Mono, monospace", color="#7070a0", size=11),
    title_font=dict(family="Playfair Display, serif", color="#e8d5a3", size=14),
    xaxis=dict(
        gridcolor="#1a1a28",
        linecolor="#1a1a28",
        tickfont=dict(color="#4a4a6a", size=10),
    ),
    yaxis=dict(
        gridcolor="#1a1a28",
        linecolor="#1a1a28",
        tickfont=dict(color="#4a4a6a", size=10),
    ),
    legend=dict(
        bgcolor="#0f0f18",
        bordercolor="#1e1e2e",
        borderwidth=1,
        font=dict(color="#7070a0", size=10),
    ),
    margin=dict(l=12, r=12, t=44, b=12),
)


def kpi(v, lbl, col="#e8d5a3"):
    return f'<div class="kpi-card"><div class="kpi-val" style="color:{col}">{v}</div><div class="kpi-label">{lbl}</div></div>'


def ins(lbl, txt, col="#4ecdc4"):
    return f'<div class="ins-card" style="border-left-color:{col}"><div class="ins-lbl" style="color:{col}">{lbl}</div><div class="ins-txt">{txt}</div></div>'


def sec(title, sub="", col="#e8d5a3"):
    s = f'<div class="sec-sub">{sub}</div>' if sub else ""
    return f'<div class="sec-hdr"><div class="sec-title" style="color:{col}">{title}</div>{s}</div>'


st.markdown(CSS, unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
ANALYTICS = os.path.join(BASE, "..", "..", "analytics")
DATA = os.path.join(BASE, "..", "..", "data")


@st.cache_data
def load():
    shops = pd.read_csv(os.path.join(ANALYTICS, "shop_ranking.csv"))
    scored = pd.read_parquet(os.path.join(DATA, "scored_products.parquet"))
    return shops, scored


shops, scored = load()
top1 = shops.iloc[0]

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="pg-main">Shop Analysis</div>'
    f'<div class="pg-sub">Analyse comparative des {len(shops)} boutiques sport &amp; fitness — '
    f"score, pricing, ratings, and discount posture.</div>"
    f'<div style="border-bottom:1px solid #1a1a28;margin:16px 0 28px 0"></div>',
    unsafe_allow_html=True,
)

# ── KPIs boutiques ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    kpi(top1["shop_name"], "Meilleure boutique", "#e8d5a3"), unsafe_allow_html=True
)
c2.markdown(
    kpi(f"{top1['score_moyen']:.3f}", "Score #1", "#4ecdc4"), unsafe_allow_html=True
)
c3.markdown(
    kpi(f"{shops['nb_produits'].sum():,}", "Produits total", "#a8e6cf"),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi(f"{shops['note_moyenne'].mean():.2f} ★", "Note moyenne", "#ffd93d"),
    unsafe_allow_html=True,
)

# ── Insight narrative ─────────────────────────────────────────────────────────
gap = top1["score_moyen"] - shops.iloc[-1]["score_moyen"]
top3 = ", ".join(shops.head(3)["shop_name"].tolist())
st.markdown(
    ins(
        "Competitive Landscape",
        f"<strong>{top1['shop_name']}</strong> leads with a score of <strong>{top1['score_moyen']:.3f}</strong>. "
        f"The top 3 boutiques — <strong>{top3}</strong> — show a clear quality-availability-value advantage. "
        f"The gap between first and last place is <strong>{gap:.3f} points</strong>, "
        f"indicating meaningful spread in catalog quality across the niche.",
    ),
    unsafe_allow_html=True,
)

# ── Classement global ─────────────────────────────────────────────────────────
st.markdown(
    sec(
        "Classement global des boutiques",
        "Composite score averaged across all products per shop — ordered best to worst.",
    ),
    unsafe_allow_html=True,
)
sh = shops.sort_values("score_moyen", ascending=True)
fig = px.bar(
    sh,
    x="score_moyen",
    y="shop_name",
    orientation="h",
    color="score_moyen",
    text=sh["score_moyen"].round(3),
    color_continuous_scale=["#1a1a28", "#3d2b1a", "#ffd93d", "#ff6b6b"],
    template="plotly_dark",
    title="Score moyen par boutique",
    hover_data=["nb_produits", "note_moyenne", "prix_moyen"],
)
fig.update_traces(textposition="outside")
fig.update_layout(**PLOT, height=420, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# ── Métriques détaillées ──────────────────────────────────────────────────────
st.markdown(
    sec("Métriques détaillées", "Average price and product distribution."),
    unsafe_allow_html=True,
)
c1, c2 = st.columns(2)
with c1:
    sh2 = shops.sort_values("prix_moyen", ascending=False)
    fig2 = px.bar(
        sh2,
        x="shop_name",
        y="prix_moyen",
        color="prix_moyen",
        text=sh2["prix_moyen"].round(1),
        color_continuous_scale=["#1a1a28", "#2a2a4a", "#4ecdc4", "#e8d5a3"],
        template="plotly_dark",
        title="Prix moyen par boutique (€)",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        **PLOT,
        height=360,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    fig3 = px.pie(
        shops,
        values="nb_produits",
        names="shop_name",
        title="Répartition des produits par boutique",
        template="plotly_dark",
        hole=0.45,
    )
    fig3.update_layout(**PLOT, height=360)
    st.plotly_chart(fig3, use_container_width=True)

# ── Notes moyennes ────────────────────────────────────────────────────────────
st.markdown(
    sec(
        "Notes moyennes par boutique",
        "Mean rating — real where available, imputed by tier median elsewhere.",
    ),
    unsafe_allow_html=True,
)
sh3 = shops.sort_values("note_moyenne", ascending=False)
fig4 = go.Figure(
    go.Bar(
        x=sh3["shop_name"],
        y=sh3["note_moyenne"],
        marker_color=[
            f"rgba(232,213,163,{0.3 + 0.7 * (v - 4.0) / 1.2})"
            for v in sh3["note_moyenne"]
        ],
        text=[f"★ {v:.2f}" for v in sh3["note_moyenne"]],
        textposition="outside",
    )
)
fig4.update_layout(
    **{**PLOT, "yaxis": dict(range=[0, 5.5], gridcolor="#1a1a28")},
    title="Notes moyennes (★/5)",
    height=350,
    xaxis_tickangle=-30,
)
st.plotly_chart(fig4, use_container_width=True)

# ── Remises ───────────────────────────────────────────────────────────────────
st.markdown(
    sec(
        "Remises moyennes par boutique (%)",
        "High discount can signal promotions or pricing strategy.",
    ),
    unsafe_allow_html=True,
)
sh4 = shops.sort_values("remise_moyenne", ascending=False)
fig5 = px.bar(
    sh4,
    x="shop_name",
    y="remise_moyenne",
    color="remise_moyenne",
    text=sh4["remise_moyenne"].round(1),
    color_continuous_scale=["#1a1a28", "#3d2b1a", "#ffd93d"],
    template="plotly_dark",
    title="Remise moyenne (%)",
)
fig5.update_traces(textposition="outside")
fig5.update_layout(
    **PLOT, height=350, showlegend=False, coloraxis_showscale=False, xaxis_tickangle=-30
)
st.plotly_chart(fig5, use_container_width=True)

# ── Tableau complet ───────────────────────────────────────────────────────────
st.markdown(
    sec("Tableau récapitulatif complet", "Complete boutique data — all metrics."),
    unsafe_allow_html=True,
)
st.dataframe(
    shops.sort_values("score_moyen", ascending=False).reset_index(drop=True),
    use_container_width=True,
    height=380,
)
