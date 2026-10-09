"""
Central configuration for the RAG application.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data" / "raw")))
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(BASE_DIR / "faiss_index")))
LLAMA_CACHE_DIR = Path(os.getenv("LLAMA_INDEX_CACHE_DIR", str(PROCESSED_DIR / "llama_cache")))

# Create directories if they don't exist
for _dir in [DATA_DIR, PROCESSED_DIR, FAISS_INDEX_PATH, LLAMA_CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Google AI ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro")

# ── Chunking ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# ── Supported file extensions ─────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    "text": [".txt", ".md", ".rst", ".csv"],
    "pdf": [".pdf"],
    "word": [".docx", ".doc"],
    "image": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
}

ALL_SUPPORTED = [ext for exts in SUPPORTED_EXTENSIONS.values() for ext in exts]
