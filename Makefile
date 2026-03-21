# ═══════════════════════════════════════════════════════
#  Smart eCommerce Intelligence Pipeline — Makefile
#  Run from repo root: make <target>
# ═══════════════════════════════════════════════════════

PYTHON ?= python
PIP    ?= pip

.PHONY: help install install-dev scrape preprocess features score \
        pipeline train dashboard test lint clean \
        compile-kfp kfp-deploy kfp-port-forward kfp-stop-forward \
        kfp-verify kfp-status all

# ── Aide ─────────────────────────────────────────────
help:
	@echo ""
	@echo "  Smart eCommerce Intelligence Pipeline"
	@echo "  ══════════════════════════════════════"
	@echo "  Étape 1  — Scraping"
	@echo "    make scrape            Lance le scraping A2A (Shopify + WooCommerce)"
	@echo ""
	@echo "  Étape 2  — Preprocessing & Scoring"
	@echo "    make preprocess        Nettoyage et validation des données brutes"
	@echo "    make features          Feature engineering (remise, popularité, tiers...)"
	@echo "    make score             Scoring Top-K et sélection des meilleurs produits"
	@echo "    make pipeline          Enchaîne preprocess + features + score"
	@echo ""
	@echo "  Étape 3  — Machine Learning"
	@echo "    make train             Entraîne RF, XGBoost, KMeans, DBSCAN, Apriori"
	@echo ""
	@echo "  Étape 4  — Dashboard"
	@echo "    make dashboard         Lance le dashboard Streamlit (port 8501)"
	@echo ""
	@echo "  Kubeflow Pipelines"
	@echo "    make compile-kfp       Compile le pipeline en YAML"
	@echo "    make kfp-deploy        Déploie sur Minikube"
	@echo "    make kfp-port-forward  Active le port-forward UI"
	@echo "    make kfp-status        Vérifie le statut du cluster"
	@echo ""
	@echo "  Dev"
	@echo "    make install           Installe les dépendances"
	@echo "    make install-dev       Installe les dépendances + outils dev"
	@echo "    make test              Lance les 113 tests pytest"
	@echo "    make lint              Vérifie le style du code"
	@echo "    make clean             Supprime les fichiers temporaires"
	@echo "    make all               scrape + pipeline + train"
	@echo ""

# ── Installation ──────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install pytest pytest-cov ruff

# ── Étape 1 : Scraping A2A ────────────────────────────
scrape:
	cd scraping && $(PYTHON) run_all.py

# ── Étape 2 : Preprocessing & Scoring ────────────────
preprocess:
	cd data_processing && $(PYTHON) preprocess.py

features:
	cd data_processing && $(PYTHON) features.py

score:
	cd data_processing && $(PYTHON) scoring.py

pipeline:
	cd data_processing && $(PYTHON) run_pipeline.py

# ── Étape 3 : Machine Learning ────────────────────────
train:
	cd ml && $(PYTHON) classification.py
	cd ml && $(PYTHON) clustering.py
	cd ml && $(PYTHON) association_rules.py

# ── Étape 4 : Dashboard Streamlit ────────────────────
dashboard:
	PYTHONPATH=. streamlit run dashboard/app.py --server.port 8501

# ── Kubeflow Pipelines ────────────────────────────────
compile-kfp:
	cd kubeflow && $(PYTHON) pipeline.py
	@echo "✓ Pipeline KFP compilé"

kfp-deploy:
	kubectl apply -f kubeflow/kubeflow_smart_ecommerce_pipeline.yaml
	@echo "✓ Pipeline déployé sur Minikube"

kfp-port-forward:
	kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &
	@echo "✓ UI disponible sur http://localhost:8080"

kfp-stop-forward:
	pkill -f "kubectl port-forward" 2>/dev/null || true
	@echo "✓ Port-forward arrêté"

kfp-verify:
	kubectl get pods -n kubeflow --field-selector=status.phase=Running
	@echo "✓ Vérification terminée"

kfp-status:
	kubectl get pods -n kubeflow
	kubectl get workflows -n kubeflow 2>/dev/null || true

# ── Tests & Qualité ───────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m flake8 . --count --select=E9,F63,F7,F82 \
	          --exclude=venv,__pycache__,.git --show-source

# ── Nettoyage ─────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true

# ── Pipeline complet ──────────────────────────────────
all: scrape pipeline train
	@echo ""
	@echo "✅ Pipeline complet terminé"
	@echo "   Lance 'make dashboard' pour visualiser les résultats"