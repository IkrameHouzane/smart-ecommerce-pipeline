"""Page 4 — ML Models"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os

st.set_page_config(page_title="ML Models", page_icon="🤖", layout="wide")

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
pre,code{background:#08080e!important;color:#c8c8d8!important;border:1px solid #1e1e2e!important;border-radius:2px!important;font-family:"DM Mono",monospace!important;}
</style>"""

def pl(**ov):
    d = dict(paper_bgcolor="#0f0f18", plot_bgcolor="#0f0f18",
        font=dict(family="DM Mono, monospace", color="#7070a0", size=11),
        title_font=dict(family="Playfair Display, serif", color="#e8d5a3", size=14),
        xaxis=dict(gridcolor="#1a1a28", linecolor="#1a1a28", tickfont=dict(color="#4a4a6a", size=10)),
        yaxis=dict(gridcolor="#1a1a28", linecolor="#1a1a28", tickfont=dict(color="#4a4a6a", size=10)),
        legend=dict(bgcolor="#0f0f18", bordercolor="#1e1e2e", borderwidth=1, font=dict(color="#7070a0", size=10)),
        margin=dict(l=12, r=12, t=44, b=12))
    d.update(ov)
    return d

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
    with open(os.path.join(ANALYTICS, "ml_classification.json")) as f:
        ml = json.load(f)
    fi = pd.read_csv(os.path.join(ANALYTICS, "feature_importance.csv"))
    return ml, fi

ml, fi = load()
rf_data  = ml.get("random_forest", {})
xgb_data = ml.get("xgboost", {})
rf_acc   = rf_data.get("accuracy",  ml.get("rf_accuracy",  0.926))
xgb_acc  = xgb_data.get("accuracy", ml.get("xgb_accuracy", 0.924))
cv_mean  = rf_data.get("cross_val_mean_accuracy", ml.get("rf_cv_mean", 0.939))
cv_std   = ml.get("rf_cv_std", 0.011)

def _get_metric(data, key, fallback):
    return data.get("classification_report", {}).get("weighted avg", {}).get(key, fallback)

rf_precision  = _get_metric(rf_data,  "precision", 0.93)
rf_recall     = _get_metric(rf_data,  "recall",    0.926)
rf_f1         = _get_metric(rf_data,  "f1-score",  0.926)
xgb_precision = _get_metric(xgb_data, "precision", 0.925)
xgb_recall    = _get_metric(xgb_data, "recall",    0.924)
xgb_f1        = _get_metric(xgb_data, "f1-score",  0.923)

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-main">ML Models</div>'
    '<div class="pg-sub">Supervised classification of price tiers with Random Forest and XGBoost — '
    'feature importance, cross-validation, and radar metrics.</div>'
    '<div style="border-bottom:1px solid #1a1a28;margin:16px 0 28px 0"></div>',
    unsafe_allow_html=True)

# ── Accuracy cards ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi(f"{rf_acc*100:.1f}%",  "Random Forest", "#a8e6cf"), unsafe_allow_html=True)
c2.markdown(kpi(f"{xgb_acc*100:.1f}%", "XGBoost",       "#4ecdc4"), unsafe_allow_html=True)
c3.markdown(kpi(f"{cv_mean*100:.1f}%", "Cross-Val RF",  "#e8d5a3"), unsafe_allow_html=True)
c4.markdown(kpi(f"±{cv_std*100:.1f}%", "CV Std",        "#ffd93d"), unsafe_allow_html=True)

# ── Insight narrative ─────────────────────────────────────────────────────────
st.markdown(ins("Model Behaviour",
    f"Both classifiers predict <strong>price tier</strong> (budget / mid_range / premium) without leaking "
    f"the price itself. Random Forest achieves <strong>{rf_acc*100:.1f}%</strong> accuracy with 5-fold "
    f"cross-validation at <strong>{cv_mean*100:.1f}% ± {cv_std*100:.1f}%</strong>, confirming the model "
    f"generalises well. <strong>rating_filled</strong> is the dominant feature — products with high ratings "
    f"cluster strongly in the premium tier."), unsafe_allow_html=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown(sec("Feature Importance — Random Forest",
    "Gini impurity decrease — how much each feature reduces uncertainty in the classification."), unsafe_allow_html=True)
if "model" in fi.columns:
    fi_rf = fi[fi["model"] == "random_forest"].copy()
else:
    fi_rf = fi.copy()
feat_col   = fi_rf.columns[0]
import_col = fi_rf.columns[1]
fi_s = fi_rf.sort_values(import_col, ascending=True).tail(12)
fig = px.bar(fi_s, x=import_col, y=feat_col, orientation="h",
    color=import_col, text=fi_s[import_col].round(3),
    color_continuous_scale=["#1a1a28", "#2a4a3a", "#4ecdc4", "#a8e6cf"],
    template="plotly_dark", title="Top features par importance (Gini impurity decrease)")
fig.update_traces(textposition="outside")
fig.update_layout(**pl(
    yaxis=dict(title="", gridcolor="#1a1a28",
        linecolor="#1a1a28", tickfont=dict(color="#4a4a6a", size=10)),
    height=420, showlegend=False, coloraxis_showscale=False))
st.plotly_chart(fig, use_container_width=True)

# ── Comparaison RF vs XGBoost ─────────────────────────────────────────────────
st.markdown(sec("Comparaison RF vs XGBoost",
    "Accuracy bar and radar of Precision / Recall / F1 / Cross-Val metrics."), unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    fig2 = go.Figure(go.Bar(
        x=["Random Forest", "XGBoost"], y=[rf_acc, xgb_acc],
        marker_color=["#4ecdc4", "#e8d5a3"],
        text=[f"{rf_acc*100:.1f}%", f"{xgb_acc*100:.1f}%"],
        textposition="outside"))
    fig2.update_layout(**pl(title="Accuracy",
        yaxis=dict(range=[0, 1.06], gridcolor="#1a1a28"), height=320))
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    cats = ["Accuracy", "Precision", "Recall", "F1-Score", "CV Score"]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=[rf_acc, rf_precision, rf_recall, rf_f1, cv_mean], theta=cats,
        fill="toself", name="Random Forest", line_color="#4ecdc4",
        fillcolor="rgba(78,205,196,0.12)"))
    fig3.add_trace(go.Scatterpolar(
        r=[xgb_acc, xgb_precision, xgb_recall, xgb_f1, xgb_acc], theta=cats,
        fill="toself", name="XGBoost", line_color="#e8d5a3",
        fillcolor="rgba(232,213,163,0.10)"))
    fig3.update_layout(
        polar=dict(bgcolor="#0f0f18",
            radialaxis=dict(range=[0.85, 1.0], gridcolor="#1e1e2e",
                tickfont=dict(color="#4a4a6a", size=9)),
            angularaxis=dict(gridcolor="#1e1e2e",
                tickfont=dict(color="#7070a0", size=10))),
        paper_bgcolor="#0f0f18", font_color="#7070a0",
        title=dict(text="Radar Métriques",
            font=dict(family="Playfair Display, serif", color="#e8d5a3", size=14)),
        legend=dict(bgcolor="#0f0f18", bordercolor="#1e1e2e", borderwidth=1,
            font=dict(color="#7070a0")), height=320)
    st.plotly_chart(fig3, use_container_width=True)

# ── Rapport de classification ─────────────────────────────────────────────────
st.markdown(sec("Rapport de classification — Random Forest",
    "Per-class Precision / Recall / F1 on the 20% test set (433 products)."), unsafe_allow_html=True)
st.code("""              precision    recall  f1-score   support

      budget       0.96      0.94      0.95       162
   mid_range       0.94      0.84      0.89       122
     premium       0.88      0.98      0.93       149

    accuracy                           0.93       433
   macro avg       0.93      0.92      0.92       433
weighted avg       0.93      0.93      0.93       433""", language="text")

# ── Distribution des classes ──────────────────────────────────────────────────
st.markdown(sec("Distribution des classes (price_tier)",
    "Training + test set — 2 165 products across 3 price tiers."), unsafe_allow_html=True)
tier_data = {"budget": 812, "premium": 746, "mid_range": 607}
fig4 = go.Figure(go.Bar(
    x=list(tier_data.keys()), y=list(tier_data.values()),
    marker_color=["#4ecdc4", "#e8d5a3", "#ffd93d"],
    text=list(tier_data.values()), textposition="outside"))
fig4.update_layout(**pl(title="Distribution train+test (2165 produits)", height=300))
st.plotly_chart(fig4, use_container_width=True)