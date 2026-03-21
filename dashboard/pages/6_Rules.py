"""Page 6 — Association Rules"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

st.set_page_config(page_title="Association Rules", page_icon="⊗", layout="wide")

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


@st.cache_data
def load():
    rules = pd.read_csv(os.path.join(ANALYTICS, "association_rules.csv"))
    with open(os.path.join(ANALYTICS, "association_rules_summary.json")) as f:
        summary = json.load(f)
    return rules, summary


rules, summary = load()
n_itemsets = summary.get("n_itemsets", summary.get("n_frequent_itemsets", 817))

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="pg-main">Association Rules</div>'
    '<div class="pg-sub">Apriori — support ≥ 0.05 · confiance ≥ 0.5 · lift ≥ 1.2. '
    "Rules describe co-occurrence patterns in product feature combinations.</div>"
    '<div style="border-bottom:1px solid #1a1a28;margin:16px 0 28px 0"></div>',
    unsafe_allow_html=True,
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi(f"{len(rules):,}", "Règles totales", "#ff6b6b"), unsafe_allow_html=True)
c2.markdown(
    kpi(f"{n_itemsets:,}", "Itemsets fréquents", "#ffd93d"), unsafe_allow_html=True
)
c3.markdown(
    kpi(f"{rules['lift'].max():.2f}", "Lift maximum", "#4ecdc4"), unsafe_allow_html=True
)
c4.markdown(
    kpi(f"{rules['confidence'].mean():.1%}", "Confiance moy.", "#e8d5a3"),
    unsafe_allow_html=True,
)

# ── Insight narrative ─────────────────────────────────────────────────────────
top_rule = rules.nlargest(1, "lift").iloc[0]
st.markdown(
    ins(
        "Strongest Association",
        f"The highest-lift rule: <strong>[{top_rule['antecedents']}]</strong> → "
        f"<strong>[{top_rule['consequents']}]</strong> with lift <strong>{top_rule['lift']:.2f}</strong> "
        f"and confidence <strong>{top_rule['confidence']:.1%}</strong>. "
        f"This combination occurs <strong>{top_rule['lift']:.1f}×</strong> more often than random chance.",
        "#ff6b6b",
    ),
    unsafe_allow_html=True,
)

# ── Filtres interactifs ───────────────────────────────────────────────────────
st.markdown(
    sec("Filtres interactifs", "Adjust thresholds to explore the rule space."),
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns(3)
min_lift = c1.slider("Lift minimum", 1.0, float(rules["lift"].max()), 1.2, 0.1)
min_conf = c2.slider("Confiance minimum", 0.0, 1.0, 0.5, 0.05)
n_top = c3.slider("Nombre de règles", 10, 100, 25)

filtered = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_conf)]
filtered = filtered.nlargest(n_top, "lift")
st.caption(f"**{len(filtered)}** règles affichées sur {len(rules):,} totales")

# ── Scatter Lift vs Confiance ─────────────────────────────────────────────────
st.markdown(
    sec(
        "Lift vs Confiance",
        "Each bubble = one rule. Size = support (frequency). Top-right = most interesting rules.",
    ),
    unsafe_allow_html=True,
)
fig = px.scatter(
    filtered,
    x="confidence",
    y="lift",
    size="support",
    color="lift",
    color_continuous_scale=["#1a1a28", "#3d1a1a", "#ff6b6b", "#ffd93d"],
    template="plotly_dark",
    title=f"Top {n_top} règles — Lift vs Confiance (taille = support)",
    hover_data=["antecedents", "consequents", "support", "confidence", "lift"],
    opacity=0.85,
    labels={"confidence": "Confidence", "lift": "Lift"},
)
fig.update_layout(**PLOT, height=440, coloraxis_showscale=False)
fig.update_traces(marker_line_width=0.5, marker_line_color="#3a1a1a")
st.plotly_chart(fig, use_container_width=True)

# ── Top règles par lift ───────────────────────────────────────────────────────
st.markdown(
    sec("Top règles par lift", "Sorted by lift descending."), unsafe_allow_html=True
)
st.dataframe(
    filtered[["antecedents", "consequents", "support", "confidence", "lift"]]
    .sort_values("lift", ascending=False)
    .reset_index(drop=True),
    use_container_width=True,
    height=380,
)

# ── Distribution du lift et de la confiance ───────────────────────────────────
st.markdown(
    sec(
        "Distribution du lift et de la confiance",
        "Histogram of all rules — most cluster at lift 2–4 with high confidence.",
    ),
    unsafe_allow_html=True,
)
c1, c2 = st.columns(2)
with c1:
    fig2 = px.histogram(
        rules,
        x="lift",
        nbins=40,
        color_discrete_sequence=["#ff6b6b"],
        template="plotly_dark",
        title="Distribution du lift (toutes règles)",
        labels={"lift": "Lift"},
    )
    fig2.update_layout(**PLOT, height=320)
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    fig3 = px.histogram(
        rules,
        x="confidence",
        nbins=30,
        color_discrete_sequence=["#ffd93d"],
        template="plotly_dark",
        title="Distribution de la confiance",
        labels={"confidence": "Confidence"},
    )
    fig3.update_layout(**PLOT, height=320)
    st.plotly_chart(fig3, use_container_width=True)
