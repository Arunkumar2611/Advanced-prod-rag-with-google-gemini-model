"""
Ingestion Pipeline
==================
Orchestrates the full document ingestion flow:

  load → chunk → embed → store

Supports all chunking strategies and document types.
Can be run as a standalone script or imported as a module.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    GOOGLE_API_KEY,
    DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from src.ingestion.document_loader import DocumentLoaderDispatcher
from src.chunking.strategies import chunk_documents, STRATEGY_MAP
from src.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline.

    Parameters
    ----------
    chunking_strategy : str
        One of: fixed_size, recursive, token, semantic, markdown,
                html, table_aware, sentence_window, sliding_window
    pdf_mode : str
        One of: basic, table, diagram
    chunk_size : int
        Target chunk size in characters (or tokens for token strategy).
    chunk_overlap : int
        Overlap between adjacent chunks.
    google_api_key : str
        Google AI API key (falls back to env var).
    """

    def __init__(
        self,
        chunking_strategy: str = "recursive",
        pdf_mode: str = "diagram",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        google_api_key: str = GOOGLE_API_KEY,
        faiss_store: Optional[FAISSVectorStore] = None,
    ):
        if chunking_strategy not in STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy '{chunking_strategy}'. "
                f"Choose from: {list(STRATEGY_MAP.keys())}"
            )

        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.google_api_key = google_api_key
        self.pdf_mode = pdf_mode

        self.loader = DocumentLoaderDispatcher(pdf_mode=pdf_mode)
        self.vector_store = faiss_store or FAISSVectorStore(google_api_key=google_api_key)

    # ── Load ─────────────────────────────────────────────────────────────────

    def load_documents(
        self,
        source: str | Path,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load documents from a file or directory.
        Returns raw (un-chunked) Document objects.
        """
        source = Path(source)
        if source.is_file():
            docs = self.loader.load(source)
        elif source.is_dir():
            docs = self.loader.load_directory(source, recursive=recursive)
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        logger.info("Loaded %d raw document(s) from %s", len(docs), source)
        return docs

    # ── Chunk ─────────────────────────────────────────────────────────────────

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Apply the configured chunking strategy."""
        kwargs: Dict = {}

        # Semantic chunker needs the embeddings object
        if self.chunking_strategy == "semantic":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from config.settings import EMBEDDING_MODEL
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=self.google_api_key,
            )
            kwargs["embeddings"] = embeddings
        elif self.chunking_strategy == "sentence_window":
            kwargs["window_size"] = 2
        elif self.chunking_strategy == "sliding_window":
            kwargs["chunk_size"] = self.chunk_size
            kwargs["stride"] = self.chunk_size // 2
        elif self.chunking_strategy not in ("table_aware",):
            kwargs["chunk_size"] = self.chunk_size
            kwargs["chunk_overlap"] = self.chunk_overlap

        chunks = chunk_documents(docs, strategy=self.chunking_strategy, **kwargs)
        logger.info(
            "Strategy '%s': %d docs → %d chunks",
            self.chunking_strategy, len(docs), len(chunks),
        )
        return chunks

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(
        self,
        source: str | Path = DATA_DIR,
        recursive: bool = True,
        incremental: bool = False,
    ) -> Dict:
        """
        Full pipeline: load → chunk → embed → store.

        Args:
            source: file or directory to ingest
            recursive: whether to recurse into subdirectories
            incremental: if True, add to existing index; else rebuild

        Returns:
            dict with ingestion statistics
        """
        start = time.time()
        stats: Dict = {}

        # 1. Load
        docs = self.load_documents(source, recursive=recursive)
        stats["raw_documents"] = len(docs)

        if not docs:
            logger.warning("No documents found at %s", source)
            return stats

        # 2. Chunk
        chunks = self.chunk_documents(docs)
        stats["chunks"] = len(chunks)

        # Filter empty chunks
        chunks = [c for c in chunks if c.page_content.strip()]
        stats["non_empty_chunks"] = len(chunks)

        # 3. Embed & Store
        if incremental and self.vector_store.is_built():
            logger.info("Incremental mode: adding %d chunks to existing index.", len(chunks))
            self.vector_store.add_documents(chunks)
        else:
            logger.info("Building new FAISS index with %d chunks…", len(chunks))
            self.vector_store.build_from_documents(chunks)

        elapsed = time.time() - start
        stats["elapsed_seconds"] = round(elapsed, 2)
        stats["chunking_strategy"] = self.chunking_strategy
        stats["pdf_mode"] = self.pdf_mode

        logger.info("Ingestion complete: %s", stats)
        return stats

    def ingest_multiple_strategies(
        self,
        source: str | Path = DATA_DIR,
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Ingest the same source with multiple chunking strategies.
        Creates a separate FAISS index subdirectory for each strategy.
        Useful for benchmarking / A-B testing.
        """
        from config.settings import FAISS_INDEX_PATH

        strategies = strategies or list(STRATEGY_MAP.keys())
        results = {}

        for strategy in strategies:
            logger.info("=== Strategy: %s ===", strategy)
            try:
                strategy_path = FAISS_INDEX_PATH / strategy
                store = FAISSVectorStore(
                    index_path=strategy_path,
                    google_api_key=self.google_api_key,
                )
                pipeline = IngestionPipeline(
                    chunking_strategy=strategy,
                    pdf_mode=self.pdf_mode,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    google_api_key=self.google_api_key,
                    faiss_store=store,
                )
                stats = pipeline.ingest(source)
                results[strategy] = {"status": "ok", **stats}
            except Exception as e:
                logger.error("Strategy '%s' failed: %s", strategy, e)
                results[strategy] = {"status": "error", "error": str(e)}

        return results
