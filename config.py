# config.py
# Configuración general del sistema
import os
from pathlib import Path

# Configuración de paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "uploads"

# Crear directorios si no existen
for directory in [DATA_DIR, CHROMA_DB_DIR, UPLOAD_DIR]:
    directory.mkdir(exist_ok=True)

# Configuración de modelos
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
LLM_MODEL = "gemma2:2b-instruct-q4_K_M"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Configuración de procesamiento
BATCH_SIZE = 32
MAX_WORKERS = 4
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
