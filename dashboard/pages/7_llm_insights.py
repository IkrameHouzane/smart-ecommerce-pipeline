"""Page 7 — LLM Insights & Chatbot (Étape 5 de l'énoncé FST Tanger)

Utilise Gemini Flash 2.0 (gratuit via Google AI Studio) — même architecture que les collègues.
- GEMINI_API_KEY dans .env ou variable d'environnement
- Prompts structurés (pas de données brutes) — architecture responsable MCP (Étape 6)
- Fonctionnalités : synthèses auto, analyse concurrentielle, chatbot BI, enrichissement produit
"""
import streamlit as st
import pandas as pd
import json, os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="LLM Insights", page_icon="🧠", layout="wide")

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
.chat-user{background:#1a1a28;border:1px solid #2a2a3a;border-radius:2px 12px 12px 12px;padding:12px 16px;margin:8px 0;font-size:0.88rem;color:#c8c8d8;line-height:1.6;}
.chat-ai{background:#0f1820;border:1px solid #1e2e2e;border-left:3px solid #4ecdc4;border-radius:12px 2px 12px 12px;padding:12px 16px;margin:8px 0;font-size:0.88rem;color:#c8c8d8;line-height:1.7;}
.chat-ai strong{color:#e8d5a3;}
.chat-lbl-u{font-family:"DM Mono",monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:2px;color:#4a4a6a;margin-bottom:4px;}
.chat-lbl-a{font-family:"DM Mono",monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:2px;color:#4ecdc4;margin-bottom:4px;text-align:right;}
.synth-card{background:#0f0f18;border:1px solid #1e1e2e;border-radius:2px;padding:20px 24px;margin-bottom:16px;line-height:1.8;font-size:0.88rem;color:#9090b0;}
.synth-card strong{color:#c8c8d8;}
.pg-main{font-family:"Playfair Display",serif;font-size:2.4rem;font-weight:700;color:#e8d5a3;line-height:1.1;margin-bottom:6px;}
.pg-sub{font-size:0.82rem;color:#4a4a6a;font-family:"DM Mono",monospace;}
.api-warn{background:#1a0f0f;border:1px solid #3a1a1a;border-left:3px solid #ff6b6b;border-radius:2px;padding:14px 18px;margin-bottom:16px;font-size:0.85rem;color:#c8a0a0;}
.api-warn code{background:#2a1a1a;padding:2px 6px;border-radius:2px;color:#ff9090;font-family:"DM Mono",monospace;font-size:0.82rem;}
</style>"""

def kpi(v, lbl, col="#e8d5a3"):
    return f'<div class="kpi-card"><div class="kpi-val" style="color:{col}">{v}</div><div class="kpi-label">{lbl}</div></div>'
def ins(lbl, txt, col="#4ecdc4"):
    return f'<div class="ins-card" style="border-left-color:{col}"><div class="ins-lbl" style="color:{col}">{lbl}</div><div class="ins-txt">{txt}</div></div>'
def sec(title, sub=""):
    s = f'<div class="sec-sub">{sub}</div>' if sub else ""
    return f'<div class="sec-hdr"><div class="sec-title">{title}</div>{s}</div>'

st.markdown(CSS, unsafe_allow_html=True)

BASE      = os.path.dirname(os.path.abspath(__file__))
ANALYTICS = os.path.join(BASE, "..", "..", "analytics")
DATA      = os.path.join(BASE, "..", "..", "data")

# ── Prompts structurés (identiques à l'architecture des collègues) ─────────────────
EXECUTIVE_SUMMARY_PROMPT = """Tu es un analyste eCommerce senior. Basé sur les données analytiques structurées suivantes, rédige un résumé exécutif de 3 à 5 phrases pour un décideur. Sois précis et n'utilise que les faits fournis. Ne pas inventer des chiffres ou des catégories.
Données :
{data}
"""

CHAIN_OF_THOUGHT_PROMPT = """Tu es un analyste eCommerce. Basé sur les données structurées suivantes, identifie la meilleure recommandation stratégique.
IMPORTANT : Raisonne étape par étape.
1. D'abord, identifie quelles catégories ont le plus de produits.
2. Ensuite, vérifie quelle boutique a le score moyen le plus élevé.
3. Considère la distribution des clusters — les produits sont-ils bien répartis ou concentrés ?
4. Enfin, synthétise en une recommandation stratégique claire et actionnable.
Données :
{data}
Analyse étape par étape :
"""

PRODUCT_COMPARISON_PROMPT = """Compare ces top produits et explique pourquoi ils sont les mieux classés. Utilise uniquement les données fournies. Sois concis (3-4 phrases). Inclus une recommandation de positionnement.
{data}
"""

COMPETITIVE_ANALYSIS_PROMPT = """Tu es un analyste concurrentiel eCommerce Sport & Fitness. Basé sur ces données de boutiques, génère une analyse concurrentielle structurée :
1. Qui domine le marché et pourquoi ?
2. Quels sont les écarts de prix, de note et de stratégie promo entre les leaders ?
3. Recommande 3 actions concrètes pour un entrant souhaitant concurrencer les boutiques top.
Utilise uniquement les faits fournis.
Données :
{data}
"""

CHAT_PROMPT = """Tu es un assistant BI expert en eCommerce Sport & Fitness intégré dans un dashboard analytique.
Tu réponds UNIQUEMENT en te basant sur le contexte fourni. Tu ne dois PAS inventer de produits, chiffres ou boutiques.
Quand tu listes des produits, écris TOUJOURS le nom COMPLET tel qu'il apparait dans les données.
Formate tes réponses avec des listes numérotées et des titres clairs. Réponds en français.

Contexte analytique complet :
{context}

Historique récent :
{history}

Question : {query}
Réponse structurée et précise :"""

# ── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load():
    try:
        topk   = pd.read_csv(os.path.join(DATA, "top_k_products.csv"))
        shops  = pd.read_csv(os.path.join(ANALYTICS, "shop_ranking.csv"))
        scored = pd.read_parquet(os.path.join(DATA, "scored_products.parquet"))
        rules  = pd.read_csv(os.path.join(ANALYTICS, "association_rules.csv"))
        with open(os.path.join(ANALYTICS, "ml_classification.json")) as f:
            ml = json.load(f)
        with open(os.path.join(ANALYTICS, "clustering_stats.json")) as f:
            cls = json.load(f)
        return topk, shops, scored, rules, ml, cls, True
    except Exception as e:
        return None, None, None, None, None, None, False

topk, shops, scored, rules, ml, cls_stats, DATA_OK = load()

# ── Build structured context — LLM never receives raw rows ────────────────────
def build_context() -> dict:
    if not DATA_OK:
        return {}
    rf_acc  = ml.get("random_forest", {}).get("accuracy", 0.926)
    xgb_acc = ml.get("xgboost", {}).get("accuracy", 0.924)
    top5p   = topk.nlargest(5, "composite_score")[
        ["title", "shop_name", "price", "composite_score"]].to_dict("records")
    cats    = scored["category_clean"].value_counts().head(5).to_dict() \
              if "category_clean" in scored.columns else {}
    km      = cls_stats["kmeans"]
    cd      = km["clusters"]
    top_rule = rules.nlargest(1, "lift").iloc[0]
    return {
        "catalogue": {
            "n_produits": len(scored),
            "n_boutiques": len(shops),
            "prix_median": round(float(scored["price"].median()), 2),
            "pct_disponibles": round(float(scored["available"].mean()) * 100, 1),
            "pct_en_promo": round(float((scored["discount_pct"] > 10).mean()) * 100, 1),
            "top_categories": cats,
        },
        "classement_boutiques": [
            {"rang": i+1, "nom": row["shop_name"],
             "score_moyen": round(float(row["score_moyen"]), 3),
             "prix_moyen": round(float(row["prix_moyen"]), 2),
             "note_moyenne": round(float(row["note_moyenne"]), 2)}
            for i, (_, row) in enumerate(shops.head(5).iterrows())
        ],
        "top_5_produits": [
            {"titre": p["title"], "boutique": p["shop_name"],
             "prix": p["price"], "score": round(float(p["composite_score"]), 4)}
            for p in top5p
        ],
        "modeles_ml": {
            "random_forest_accuracy": round(rf_acc * 100, 1),
            "xgboost_accuracy": round(xgb_acc * 100, 1),
            "feature_principale": "rating_filled (importance 0.503)",
        },
        "segmentation_kmeans": {
            "silhouette_score": km["silhouette_score"],
            "k": km["k"],
            "clusters": [
                {"cluster": d["cluster_kmeans"], "prix_moyen": round(d["prix_moyen"], 0),
                 "note_moyenne": round(d["note_moyenne"], 2),
                 "remise_moy": round(d["remise_moy"], 0), "nb_produits": d["nb_produits"]}
                for d in cd
            ],
            "anomalies_dbscan": cls_stats["dbscan"]["nb_anomalies"],
        },
        "regles_association": {
            "total_regles": len(rules),
            "regle_lift_max": {
                "antecedents": str(top_rule["antecedents"])[:60],
                "consequents": str(top_rule["consequents"])[:60],
                "lift": round(float(top_rule["lift"]), 2),
                "confidence_pct": round(float(top_rule["confidence"]) * 100, 1),
            },
        },
    }

CONTEXT = build_context()

# ── Log usage — MCP accountability ───────────────────────────────────────────
def log_usage(source: str, prompt_preview: str, response_preview: str):
    try:
        entry = {"source": source, "prompt_preview": prompt_preview[:200],
                 "response_preview": (response_preview or "")[:200]}
        with open(os.path.join(ANALYTICS, "llm_usage_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ── Gemini API call ───────────────────────────────────────────────────────────
def call_gemini(prompt: str, source: str = "gemini") -> str:
    """Call Gemini Flash. Returns clear error message if quota exceeded — no fake demo content."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        return "⚠️ Clé API manquante — ajoutez GEMINI_API_KEY dans votre fichier .env"
    try:
        from google import genai
    except ImportError:
        return "⚠️ Package manquant — exécutez : pip install google-genai"

    models = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]
    client = genai.Client(api_key=api_key)
    for model in models:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            out = getattr(response, "text", "") or str(response)
            log_usage(source, prompt[:200], out)
            return out
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "limit: 0" in err_str:
                import time; time.sleep(2)
                continue
            if "404" in err_str or "NOT_FOUND" in err_str:
                continue
            log_usage(f"{source}_error", prompt[:200], str(e))
            return f"⚠️ Erreur Gemini : {str(e)[:120]}"
    return "⏳ Quota momentanément atteint (15 req/min, free tier). Attendez 30 secondes et réessayez."

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="pg-main">LLM Insights</div>'
    '<div class="pg-sub">Intelligence augmentée — synthèses auto, analyse concurrentielle, '
    'recommandations stratégiques et chatbot BI · Gemini Flash 2.0 (gratuit)</div>'
    '<div style="border-bottom:1px solid #1a1a28;margin:16px 0 28px 0"></div>',
    unsafe_allow_html=True)

# ── API key warning ───────────────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
if not api_key:
    st.markdown("""<div class="api-warn">
        <strong>⚠️ Clé API manquante</strong> — Créez un fichier <code>.env</code> dans
        <code>smart_ecommerce/</code> et ajoutez :<br><br>
        <code>GEMINI_API_KEY=votre_clé_ici</code><br><br>
        Clé <strong>gratuite</strong> sur
        <a href="https://aistudio.google.com/apikey" style="color:#ff9090">aistudio.google.com/apikey</a>
        — puis installez : <code>pip install google-genai python-dotenv</code>
    </div>""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
if DATA_OK:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(f"{len(scored):,}", "Produits dans le contexte", "#e8d5a3"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{len(shops)}", "Boutiques analysées", "#4ecdc4"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{len(rules):,}", "Règles Apriori", "#ffd93d"), unsafe_allow_html=True)
    c4.markdown(kpi("Gemini Flash 2.0", "Modèle LLM", "#a8e6cf"), unsafe_allow_html=True)

st.markdown(ins(
    "Architecture LLM — Conforme à l'énoncé Étape 5",
    "Le LLM reçoit uniquement des <strong>métriques agrégées structurées</strong> (jamais de données brutes). "
    "Chaque appel est loggé dans <code>analytics/llm_usage_log.jsonl</code> — architecture responsable MCP (Étape 6). "
    "Modèle : <strong>Gemini Flash 2.0</strong> gratuit via Google AI Studio. "
    "Prompts : Executive Summary · Chain-of-Thought · Product Comparison · Competitive Analysis."
), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SYNTHÈSES AUTOMATIQUES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(sec("Synthèses automatiques",
    "Génération automatique de rapports business — prompts structurés conformes à l'énoncé."),
    unsafe_allow_html=True)

SYNTHESES = {
    "📊 Résumé exécutif": ("executive",
        lambda ctx: EXECUTIVE_SUMMARY_PROMPT.format(
            data=json.dumps(ctx, indent=2, ensure_ascii=False))),
    "⚔️ Analyse concurrentielle": ("competitive",
        lambda ctx: COMPETITIVE_ANALYSIS_PROMPT.format(
            data=json.dumps({"boutiques": ctx.get("classement_boutiques", []),
                             "catalogue": ctx.get("catalogue", {})},
                            indent=2, ensure_ascii=False))),
    "🧩 Stratégie Chain-of-Thought": ("strategy",
        lambda ctx: CHAIN_OF_THOUGHT_PROMPT.format(
            data=json.dumps({
                "top_categories": ctx.get("catalogue", {}).get("top_categories", {}),
                "best_shop": ctx.get("classement_boutiques", [{}])[0].get("nom", ""),
                "best_shop_avg_score": ctx.get("classement_boutiques", [{}])[0].get("score_moyen", 0),
                "cluster_distribution": [
                    f"Cluster {c['cluster']}: {c['nb_produits']} produits, ~{c['prix_moyen']}€"
                    for c in ctx.get("segmentation_kmeans", {}).get("clusters", [])],
            }, indent=2, ensure_ascii=False))),
    "🏆 Profil top produits": ("profiling",
        lambda ctx: PRODUCT_COMPARISON_PROMPT.format(
            data=json.dumps(ctx.get("top_5_produits", []), indent=2, ensure_ascii=False))),
}

cols_s = st.columns(4)
for i, (label, (key, _)) in enumerate(SYNTHESES.items()):
    if cols_s[i].button(label, use_container_width=True, key=f"sb_{key}"):
        st.session_state["synth_active"] = label

if "synth_active" in st.session_state:
    active    = st.session_state["synth_active"]
    key, build = SYNTHESES[active]
    cache_key  = f"synth_cache_{key}"
    if cache_key not in st.session_state:
        with st.spinner(f"Gemini génère — {active}..."):
            st.session_state[cache_key] = call_gemini(build(CONTEXT), f"gemini_{key}")
    st.markdown(
        f'<div style="margin:12px 0 4px 0;font-family:\'DM Mono\',monospace;font-size:0.65rem;'
        f'color:#4ecdc4;text-transform:uppercase;letter-spacing:2px;">{active}</div>'
        f'<div class="synth-card">{st.session_state[cache_key].replace(chr(10),"<br>")}</div>',
        unsafe_allow_html=True)
    if st.button("🔄 Régénérer", key="regen"):
        del st.session_state[cache_key]
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CHATBOT CONVERSATIONNEL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(sec("Chatbot conversationnel",
    "Questions en langage naturel — contexte structuré injecté, pas de données brutes."),
    unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_input_val" not in st.session_state:
    st.session_state.chat_input_val = ""

SUGGESTED = [
    "Quels sont les 5 produits les mieux scorés ?",
    "Quelles promotions concurrentes ont été détectées ?",
    "Quelle boutique a la meilleure stratégie prix ?",
    "Analyse des tendances par catégorie",
    "Quels produits recommandes-tu pour un profil budget ?",
    "Stratégie marketing basée sur les clusters",
]

# Questions suggérées — clic remplit ET envoie directement
st.markdown('''<div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3a3a5c;
text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">Questions suggérées — cliquer pour envoyer</div>''',
    unsafe_allow_html=True)

qcols = st.columns(3)
for i, q in enumerate(SUGGESTED):
    if qcols[i % 3].button(q, use_container_width=True, key=f"q_{i}"):
        # Envoyer directement la question suggérée
        if q not in [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]:
            st.session_state.chat_history.append({"role": "user", "content": q})
            history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}"
                                      for m in st.session_state.chat_history[-4:]])
            prompt = CHAT_PROMPT.format(
                context=json.dumps(CONTEXT, indent=2, ensure_ascii=False)[:3000],
                history=history_str, query=q)
            with st.spinner(f"Gemini analyse — {q[:40]}..."):
                response = call_gemini(prompt, "gemini_chat")
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.chat_input_val = ""
            st.rerun()

# Historique de conversation
if st.session_state.chat_history:
    st.markdown('<div style="margin:16px 0 8px 0;border-top:1px solid #1a1a28;padding-top:16px;"></div>',
                unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-lbl-u">Vous</div><div class="chat-user">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            # Render markdown properly
            import re
            rendered = msg["content"]
            # Headers
            rendered = re.sub(r"^#### (.+)$", r"<h5 style='color:#e8d5a3;font-family:Playfair Display,serif;margin:10px 0 4px'> \1</h5>", rendered, flags=re.MULTILINE)
            rendered = re.sub(r"^### (.+)$", r"<h4 style='color:#e8d5a3;font-family:Playfair Display,serif;margin:12px 0 6px'> \1</h4>", rendered, flags=re.MULTILINE)
            rendered = re.sub(r"^## (.+)$", r"<h3 style='color:#e8d5a3;font-family:Playfair Display,serif;margin:14px 0 6px'> \1</h3>", rendered, flags=re.MULTILINE)
            # Bold and italic
            rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
            rendered = re.sub(r"\*(.+?)\*", r"<em style='color:#c8c8d8'>\1</em>", rendered)
            # Numbered lists
            rendered = re.sub(r"^(\d+)\. (.+)$", r"<div style='margin:4px 0'><span style='color:#4ecdc4;font-weight:500'>\1.</span> \2</div>", rendered, flags=re.MULTILINE)
            # Bullet lists
            rendered = re.sub(r"^[\*\-] (.+)$", r"<div style='margin:2px 0 2px 12px'>· \1</div>", rendered, flags=re.MULTILINE)
            # Line breaks
            rendered = rendered.replace(chr(10), "<br>")
            # Clean double breaks after block elements
            rendered = re.sub(r"(</h[2-5]>)<br>", r"\1", rendered)
            rendered = re.sub(r"(</div>)<br>", r"\1", rendered)
            st.markdown(f'<div class="chat-lbl-a">LLM · Gemini Flash</div>'
                        f'<div class="chat-ai">{rendered}</div>',
                        unsafe_allow_html=True)

# Zone de saisie — se vide après envoi via session state
st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
user_input = st.text_input("Question",
    value=st.session_state.chat_input_val,
    placeholder="Posez votre question sur le catalogue, les boutiques, les tendances...",
    key="chat_input_field",
    label_visibility="collapsed")

c1, c2, c3 = st.columns([3, 1, 1])
send  = c1.button("Envoyer →", use_container_width=True, type="primary")
clear = c3.button("Effacer l'historique", use_container_width=True)

if clear:
    st.session_state.chat_history = []
    st.session_state.chat_input_val = ""
    st.rerun()

if send and user_input.strip():
    q = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": q})
    history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}"
                              for m in st.session_state.chat_history[-4:]])
    prompt = CHAT_PROMPT.format(
        context=json.dumps(CONTEXT, indent=2, ensure_ascii=False)[:3000],
        history=history_str, query=q)
    with st.spinner("Gemini réfléchit..."):
        response = call_gemini(prompt, "gemini_chat")
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.chat_input_val = ""  # vider le champ
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ENRICHISSEMENT PRODUIT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(sec("Enrichissement automatique de produit",
    "Résumé commercial + profil client + recommandation — données agrégées uniquement."),
    unsafe_allow_html=True)

if DATA_OK:
    c1, c2 = st.columns([3, 1])
    sel = c1.selectbox("Produit à enrichir", topk["title"].str[:80].tolist(), key="enrich_sel")
    go  = c2.button("Enrichir →", use_container_width=True, key="enrich_go")
    if go and sel:
        row = topk[topk["title"].str[:80] == sel].iloc[0]
        product_data = {
            "titre": row["title"][:80], "boutique": row["shop_name"],
            "prix": float(row["price"]), "note": float(row["rating_filled"]),
            "remise_pct": round(float(row["discount_pct"]), 1),
            "score_composite": round(float(row["composite_score"]), 4),
            "tier": str(row.get("price_tier", "N/A")),
            "nb_variantes": int(row.get("nb_variants", 0)),
        }
        cache_key = f"enrich_{sel[:30]}"
        if cache_key not in st.session_state:
            with st.spinner("Génération fiche enrichie..."):
                st.session_state[cache_key] = call_gemini(
                    PRODUCT_COMPARISON_PROMPT.format(
                        data=json.dumps(product_data, indent=2, ensure_ascii=False)),
                    "gemini_profiling")
        st.markdown(
            f'<div style="margin:12px 0 4px 0;font-family:\'DM Mono\',monospace;font-size:0.65rem;'
            f'color:#ffd93d;text-transform:uppercase;letter-spacing:2px;">'
            f'Fiche enrichie — {row["title"][:60]}</div>'
            f'<div class="synth-card">{st.session_state[cache_key].replace(chr(10),"<br>")}</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ARCHITECTURE MCP (Étape 6)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(sec("Architecture responsable — MCP",
    "Model Context Protocol d'Anthropic"),
    unsafe_allow_html=True)

st.markdown("""
| Composant MCP | Classe | Implémentation |
|---|---|---|
| **MCP Host** | `dashboard/app.py` | Dashboard Streamlit — orchestre tout |
| **MCP Client** | `MCPClient` | Route les requêtes vers les bons serveurs |
| **MCP Server — Analytics** | `AnalyticsReaderServer` | Read-only, whitelist 14 fichiers |
| **MCP Server — LLM** | `SummaryGeneratorServer` | Gemini Flash — métriques agrégées uniquement |
| **Permissions** | `PERMISSIONS` dict | Pas de données brutes, pas d'exécution |
| **Logs / Audit** | `_log_access()` | `mcp_access_log.jsonl` + `llm_usage_log.jsonl` |
| **Isolation** | Whitelist stricte | LLM ne voit jamais les 2165 lignes brutes |
| **Référence** | MCP Spec | modelcontextprotocol.io/specification/2025-03-26 |
""", unsafe_allow_html=True)

# Vérifier que le fichier architecture.py existe
mcp_file = os.path.join(BASE, "..", "..", "mcp", "architecture.py")
if os.path.exists(mcp_file):
    st.markdown(ins(
        "Implémentation MCP",
        f"Le fichier <code>mcp/architecture.py</code> contient l'implémentation complète : "
        f"<strong>AnalyticsReaderServer</strong> (whitelist fichiers, permission check, log DENIED), "
        f"<strong>SummaryGeneratorServer</strong> (génération LLM depuis agrégats uniquement), "
        f"<strong>MCPClient</strong> (routage Host→Servers), "
        f"<strong>_log_access()</strong> (audit trail append-only avec timestamp UTC).",
        "#a8e6cf"
    ), unsafe_allow_html=True)
else:
    st.markdown(ins(
        "Implémentation MCP",
        "Fichier <code>mcp/architecture.py</code> — placer dans <code>smart_ecommerce/mcp/</code>.",
        "#ffd93d"
    ), unsafe_allow_html=True)

# Deux logs côte à côte
c1, c2 = st.columns(2)

with c1:
    with st.expander("▸ LLM Usage Log — appels Gemini"):
        log_path = os.path.join(ANALYTICS, "llm_usage_log.jsonl")
        if os.path.exists(log_path):
            records = []
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass
            if records:
                df_log = pd.DataFrame(records).tail(15)
                # Rename for readability
                rename = {"source": "Source", "prompt_preview": "Prompt (aperçu)",
                          "response_preview": "Réponse (aperçu)"}
                df_log = df_log.rename(columns={k:v for k,v in rename.items() if k in df_log.columns})
                # Truncate long columns for display
                for col in ["Prompt (aperçu)", "Réponse (aperçu)"]:
                    if col in df_log.columns:
                        df_log[col] = df_log[col].astype(str).str[:60] + "..."
                st.dataframe(df_log, use_container_width=True, height=240)
            else:
                st.info("Log vide — générez des synthèses pour voir les entrées.")
        else:
            st.info("Aucun log encore.")

with c2:
    with st.expander("▸ MCP Access Log — accès données"):
        mcp_log = os.path.join(ANALYTICS, "mcp_access_log.jsonl")
        if os.path.exists(mcp_log):
            records = []
            with open(mcp_log, encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass
            if records:
                st.dataframe(pd.DataFrame(records).tail(15),
                    use_container_width=True, height=240)
            else:
                st.info("Log vide — le MCPClient n'a pas encore été utilisé.")
        else:
            st.info("Créé au premier appel MCPClient.")

st.markdown("""
<div style="margin-top:32px;padding-top:16px;border-top:1px solid #1a1a28;
    font-family:'DM Mono',monospace;font-size:0.62rem;color:#2a2a3a;">
    Étape 5 — LLM enrichissement · Gemini Flash 2.0 · Prompt Engineering · Chain of Thought ·
    Étape 6 — MCP Architecture responsable · FST Tanger LSI 2 2025/2026
</div>
""", unsafe_allow_html=True)