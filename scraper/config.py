# Configuration for DGSI databases
# Each entry: (db_name, view_id, court_label, approx_doc_count)

DATABASES = [
    {
        "db": "jstj.nsf",
        "view_id": "954f0ce6ad9dd8b980256b5f003fa814",
        "label": "Supremo Tribunal de Justiça",
        "short": "STJ",
        "approx_count": 72887,
    },
    {
        "db": "jsta.nsf",
        "view_id": "35fbbbf22e1bb1e680256f8e003ea931",
        "label": "Supremo Tribunal Administrativo",
        "short": "STA",
        "approx_count": 89526,
    },
    {
        "db": "jcon.nsf",
        "view_id": "35fbbbf22e1bb1e680256f8e003ea931",
        "label": "Tribunal dos Conflitos",
        "short": "TCON",
        "approx_count": 1232,
    },
    {
        "db": "atco1.nsf",
        "view_id": "904714e45043f49b802565fa004a5fd7",
        "label": "Tribunal Constitucional (até 1998)",
        "short": "TC",
        "approx_count": 6107,
    },
    {
        "db": "jtrp.nsf",
        "view_id": "56a6e7121657f91e80257cda00381fdf",
        "label": "Tribunal da Relação do Porto",
        "short": "TRP",
        "approx_count": 63320,
    },
    {
        "db": "jtrl.nsf",
        "view_id": "33182fc732316039802565fa00497eec",
        "label": "Tribunal da Relação de Lisboa",
        "short": "TRL",
        "approx_count": 59927,
    },
    {
        "db": "jtrc.nsf",
        "view_id": "8fe0e606d8f56b22802576c0005637dc",
        "label": "Tribunal da Relação de Coimbra",
        "short": "TRC",
        "approx_count": 16313,
    },
    {
        "db": "jtrg.nsf",
        "view_id": "86c25a698e4e7cb7802579ec004d3832",
        "label": "Tribunal da Relação de Guimarães",
        "short": "TRG",
        "approx_count": 15346,
    },
    {
        "db": "jtre.nsf",
        "view_id": "134973db04f39bf2802579bf005f080b",
        "label": "Tribunal da Relação de Évora",
        "short": "TRE",
        "approx_count": 17763,
    },
    {
        "db": "jtca.nsf",
        "view_id": "170589492546a7fb802575c3004c6d7d",
        "label": "Tribunal Central Administrativo Sul",
        "short": "TCAS",
        "approx_count": 30059,
    },
    {
        "db": "jtcn.nsf",
        "view_id": "89d1c0288c2dd49c802575c8003279c7",
        "label": "Tribunal Central Administrativo Norte",
        "short": "TCAN",
        "approx_count": 20312,
    },
    {
        "db": "cajp.nsf",
        "view_id": "954f0ce6ad9dd8b980256b5f003fa814",
        "label": "Julgados de Paz",
        "short": "JP",
        "approx_count": 7322,
    },
]

BASE_URL = "https://www.dgsi.pt"
TOTAL_APPROX = sum(db["approx_count"] for db in DATABASES)

# HTTP settings
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0
MAX_CHUNK_SIZE_MB = 80  # Max JSON file size before splitting
DOCS_PER_CHUNK = 15000  # Approx docs per output file

# Encoding
ENCODING = "windows-1252"  # DGSI uses cp1252 / latin-1
