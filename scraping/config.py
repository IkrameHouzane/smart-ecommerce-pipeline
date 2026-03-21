# scraping/config.py
# ─────────────────────────────────────────────────────────
# Fichier de configuration central — le "carnet d'adresses".
# TOUTES les boutiques (Shopify + WooCommerce) sont ici.
# Si tu veux ajouter/modifier une boutique → ce fichier UNIQUEMENT.
# ─────────────────────────────────────────────────────────


# ══════════════════════════════════════
#  BOUTIQUES SHOPIFY
# ══════════════════════════════════════
# Accès via /products.json — public, sans clé API
# Boutiques confirmées et testées

SHOPIFY_STORES = [
    {
        "name": "allbirds",
        "url": "https://www.allbirds.com",
        # Chaussures running éco — 736 produits confirmés
    },
    {
        "name": "represent",
        "url": "https://representclo.com",
        # Vêtements sport lifestyle — 728 produits confirmés
    },
    {
        "name": "born_primitive",
        "url": "https://bornprimitive.com",
        # CrossFit et fitness — 1761 produits confirmés
    },
    {
        "name": "nobull",
        "url": "https://nobullproject.com",
        # Chaussures et vêtements sport intensif
    },
    {
        "name": "goat_tape",
        "url": "https://www.goattape.com",
        # Accessoires sport (straps, grip)
    },
    {
        "name": "ten_thousand",
        "url": "https://www.tenthousand.cc",
        # Vêtements entraînement intensif
    },
    {
        "name": "cuts_clothing",
        "url": "https://cutsclothing.com",
        # Vêtements sport premium hommes
    },
    {
        "name": "set_active",
        "url": "https://setactive.co",
        # Vêtements fitness femme
    },
]


# ══════════════════════════════════════
#  BOUTIQUES WOOCOMMERCE
# ══════════════════════════════════════
# Accès via Store API publique — /wp-json/wc/store/v1/products
# IMPORTANT : c'est différent de l'API REST classique /wp-json/wc/v3/
# La Store API ne nécessite AUCUNE clé API — accès public !
# Boutiques confirmées et testées

WOO_STORES = [
    {
        "name": "nalgene",
        "url": "https://nalgene.com",
        # Bouteilles sport — 92 produits confirmés
        "geography": "US",
    },
    {
        "name": "nutribullet",
        "url": "https://www.nutribullet.com",
        # Blenders nutrition sport — 184 produits confirmés
        "geography": "US",
    },
    {
        "name": "danosseasoning",
        "url": "https://danosseasoning.com",
        # Condiments sport/fitness — 125 produits confirmés
        "geography": "US",
    },
]


# ══════════════════════════════════════
#  PARAMÈTRES GÉNÉRAUX
# ══════════════════════════════════════

# Shopify : maximum 250 produits par page
PRODUCTS_PER_PAGE = 250

# WooCommerce Store API : 40 produits par page
WOO_PER_PAGE = 40

# WooCommerce : nombre maximum de pages à scraper par boutique
WOO_MAX_PAGES = 25

# Dossier de sauvegarde des CSV
# ".." = remonter d'un niveau depuis scraping/ vers smart_ecommerce/
OUTPUT_DIR = "../data"
