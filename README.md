# Smart eCommerce Intelligence Pipeline
### Top-K Product Selection — FST Tanger · LSI · 2025/2026

Un pipeline end-to-end qui collecte des données produits eCommerce depuis Shopify et WooCommerce via scraping multi-agents A2A, score et classe les produits avec une formule explicable, applique des modèles ML/Data Mining pour la segmentation et la découverte de patterns, et expose tous les insights via un dashboard BI Streamlit multi-pages avec synthèse LLM (Gemini) et accès analytics gouverné par MCP.

---

## Résultats clés

| Métrique | Valeur |
|---|---|
| Produits scrapés | 5 051 (4 741 Shopify + 310 WooCommerce) |
| Produits scorés | 2 165 (après déduplication) |
| Boutiques analysées | 11 |
| Random Forest accuracy | **92.6%** |
| XGBoost accuracy | **92.4%** |
| Cross-validation F1 | 93.9% ± 1.1% |
| Silhouette KMeans (k=4) | 0.397 |
| Anomalies DBSCAN | 106 |
| Règles Apriori générées | 4 471 |
| Nœuds Kubeflow verts | 8 / 8 |
| Tests pytest | **113 passed** |

---

## Architecture

```
smart_ecommerce/
├── scraping/           # Agents A2A — Shopify + WooCommerce + HTML fallback
├── data_processing/    # Preprocessing, feature engineering, scoring Top-K
├── ml/                 # RF, XGBoost, KMeans, DBSCAN, règles Apriori
├── kubeflow/           # Pipeline KFP v2, YAML compilé, Minikube
├── dashboard/          # App Streamlit multi-pages (7 pages)
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Products.py
│       ├── 3_Shops.py
│       ├── 4_ML.py
│       ├── 5_Clustering.py
│       ├── 6_Rules.py
│       └── 7_LLM_Insights.py
├── mcp/                # Architecture MCP responsable (Étape 6)
│   └── architecture.py
├── data/               # CSV + Parquet (raw → processed → scored)
├── analytics/          # Outputs ML, rankings, logs JSONL
├── tests/              # 113 tests pytest
└── .env                # GEMINI_API_KEY (non versionné)
```

---

## Stack

| Couche | Outils |
|---|---|
| Scraping | Playwright, requests, BeautifulSoup, agents A2A |
| Stockage | JSON (raw), Parquet + CSV + JSONL (processed/analytics) |
| ML / Data Mining | scikit-learn, XGBoost, mlxtend (Apriori) |
| Visualisation | Streamlit, Plotly, Altair, Matplotlib, Seaborn |
| LLM | Google Gemini Flash 2.0 (google-genai) |
| Accès analytics | MCP allowlist (`mcp/architecture.py`) — read-only |
| Orchestration | Kubeflow Pipelines v2 (KFP), Minikube, Kustomize |
| Packaging | Docker, Dockerfile |
| Tests | pytest — 113 tests |

---

## Démarrage rapide

### 1. Installation

```bash
git clone <repo-url>
cd smart_ecommerce

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Éditer .env et ajouter votre clé API Gemini
```

Contenu du `.env` :
```
GEMINI_API_KEY=votre_clé_ici
```

Clé gratuite sur [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 3. Pipeline complet

```bash
# Étape 1 — Scraping A2A
python scraping/run_scraping.py

# Étape 2 — Preprocessing + Scoring
python data_processing/run_pipeline.py

# Étape 3 — Machine Learning
python ml/run_ml.py

# Étape 4 — Dashboard BI
streamlit run dashboard/app.py
# → http://localhost:8501
```

### 4. Kubeflow (optionnel)

```bash
# Démarrer Minikube
minikube start --driver=docker --memory=4096 --cpus=4

# Port-forward Kubeflow UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
# → http://localhost:8080
```

### 5. Tests

```bash
python -m pytest tests/ -v
# 113 passed in ~0.5s
```

---

## Étapes du projet

### Étape 1 — Scraping A2A multi-agents
- **CoordinatorAgent** distribue les cibles aux **WorkerAgents** concurrents
- Capture JSON-LD, breadcrumbs, meta WooCommerce, fallback HTML
- 11 boutiques Sport & Fitness (Shopify + WooCommerce)
- Output : `data/FINAL_sport_fitness_products.csv` — 5 051 produits

### Étape 2 — Preprocessing & Scoring Top-K
- Nettoyage prix, notes, remises, disponibilité + déduplication
- Formule de score composite explicable :
```
score = 0.35×rating_norm + 0.30×reviews_norm + 0.20×availability + 0.15×discount_norm
```
- Output : `data/scored_products.parquet`, `data/top_k_products.csv`

### Étape 3 — Machine Learning & Data Mining
- **Classification supervisée** : RandomForest 92.6% + XGBoost 92.4%
- **Clustering** : KMeans k=4 (silhouette=0.397) + DBSCAN (106 anomalies)
- **Règles d'association** : Apriori — 4 471 règles (lift max > 2.0)
- **PCA** : visualisation 2D des clusters produits

### Étape 3B — Kubeflow Pipelines
- Pipeline compilé `kubeflow/smart_ecommerce_pipeline.yaml`
- Image Docker `smart-ecommerce-pipeline:local` dans Minikube
- 8 nœuds verts confirmés sur `localhost:8080`

### Étape 4 — Dashboard BI (7 pages)

| Page | Contenu |
|---|---|
| Overview | KPIs catalogue, distributions prix/notes, heatmap corrélations |
| Product Rankings | Leaderboard filtrable, carte d'opportunités, export CSV |
| Shop Analysis | Radar comparatif, score et prix par boutique |
| ML Models | Métriques RF/XGB, feature importance, matrices de confusion |
| Segmentation | PCA 2D clusters, profils KMeans, anomalies DBSCAN |
| Association Rules | Carte support/confiance/lift, top règles par lift |
| LLM Insights | Synthèses Gemini, chatbot BI, enrichissement produit, MCP |

### Étape 5 — Intelligence LLM (Gemini Flash 2.0)
- 4 synthèses automatiques : Executive Summary, Analyse concurrentielle, Chain-of-Thought, Profil top produits
- Chatbot BI conversationnel — contexte structuré injecté (jamais de données brutes)
- Enrichissement produit automatique : résumé commercial + profil client + recommandation
- Logging de chaque appel dans `analytics/llm_usage_log.jsonl`

### Étape 6 — Architecture MCP responsable

```
MCP Host (Dashboard Streamlit)
    └── MCPClient
            ├── AnalyticsReaderServer   → whitelist 14 fichiers, read-only, log DENIED
            └── SummaryGeneratorServer  → LLM reçoit métriques agrégées uniquement
```

- Audit trail append-only : `analytics/mcp_access_log.jsonl`
- Isolation totale : le LLM ne voit jamais les 2 165 lignes brutes
- Référence : [modelcontextprotocol.io/specification/2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26)

---

## Structure des données

```
data/
├── FINAL_sport_fitness_products.csv   # 5 051 produits bruts
├── clean_products.parquet             # après nettoyage
├── featured_products.parquet          # après feature engineering
├── scored_products.parquet            # avec score composite
└── top_k_products.csv                 # Top-K final

analytics/
├── shop_ranking.csv                   # classement 11 boutiques
├── topk_per_category.csv              # Top-K par catégorie
├── ml_classification.json            # métriques RF + XGB
├── clustering_stats.json             # KMeans + DBSCAN
├── association_rules.csv             # 4 471 règles Apriori
├── llm_usage_log.jsonl               # audit appels LLM
└── mcp_access_log.jsonl              # audit accès données MCP
```

---

## Variables d'environnement

```bash
# .env.example
GEMINI_API_KEY=your_key_here    # Google AI Studio — gratuit
```

---

## Tests (113 passed)

| Fichier | Tests | Couverture |
|---|---|---|
| `test_scoring.py` | 17 | Formule composite, Top-K, ranking boutiques |
| `test_preprocessing.py` | 20 | Prix, notes, remises, déduplication, normalisation |
| `test_ml.py` | 19 | Métriques RF/XGB, KMeans, DBSCAN, Apriori |
| `test_mcp.py` | 17 | Whitelist, permissions, path traversal, logging |
| `test_scraping.py` | 22 | Schéma produit, qualité données, multi-plateforme |
| `test_dashboard.py` | 18 | Contexte LLM, isolation données brutes |

```bash
python -m pytest tests/ -v
# 113 passed in 0.42s
```

---

## Références

- [Anthropic MCP Overview](https://www.anthropic.com/news/model-context-protocol)
- [MCP Specification 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26)
- [Kubeflow Pipelines v2](https://www.kubeflow.org/docs/components/pipelines/)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)

---

*Projet académique — FST Tanger · LSI · 2025/2026*