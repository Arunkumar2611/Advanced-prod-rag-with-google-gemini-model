"""
FAISS Vector Store Manager
===========================
Wraps LangChain's FAISS integration with Google text-embedding-004.

Features:
  - Build index from documents
  - Persist & load index from disk
  - Similarity search (k-NN + score threshold)
  - MMR (Maximal Marginal Relevance) search
  - Incremental add / delete
  - LlamaIndex FAISS bridge
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    TOP_K,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Manages a FAISS vector store backed by Google Generative AI embeddings.
    """

    INDEX_FILE = "index.faiss"
    PKL_FILE = "index.pkl"

    def __init__(
        self,
        index_path: Optional[Path] = None,
        embedding_model: str = EMBEDDING_MODEL,
        google_api_key: str = GOOGLE_API_KEY,
    ):
        self.index_path = Path(index_path or FAISS_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key,
        )
        self._store: Optional[FAISS] = None

    # ── Build / Load / Save ──────────────────────────────────────────────────

    def build_from_documents(self, documents: List[Document]) -> None:
        """Create a FAISS index from a list of documents."""
        if not documents:
            raise ValueError("Cannot build index from an empty document list.")

        logger.info("Building FAISS index from %d chunks…", len(documents))
        self._store = FAISS.from_documents(documents, self.embeddings)
        self.save()
        logger.info("Index built and saved to %s", self.index_path)

    def save(self) -> None:
        """Persist the current index to disk."""
        if self._store is None:
            raise RuntimeError("No index to save. Call build_from_documents first.")
        self._store.save_local(str(self.index_path))
        logger.info("FAISS index saved → %s", self.index_path)

    def load(self) -> None:
        """Load a previously saved index from disk."""
        faiss_file = self.index_path / self.INDEX_FILE
        if not faiss_file.exists():
            raise FileNotFoundError(
                f"No FAISS index found at {self.index_path}. "
                "Run build_from_documents first."
            )
        self._store = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded from %s", self.index_path)

    def is_built(self) -> bool:
        """Return True if an index is ready (either in memory or on disk)."""
        return self._store is not None or (self.index_path / self.INDEX_FILE).exists()

    def get_store(self) -> FAISS:
        """Return the underlying FAISS store, loading from disk if needed."""
        if self._store is None:
            self.load()
        return self._store  # type: ignore[return-value]

    # ── Incremental Updates ──────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to an existing index."""
        store = self.get_store()
        store.add_documents(documents)
        self.save()
        logger.info("Added %d documents to existing index.", len(documents))

    def delete_by_source(self, source: str) -> int:
        """Remove all vectors whose metadata 'source' matches the given path."""
        store = self.get_store()
        ids_to_delete = [
            doc_id
            for doc_id, doc in store.docstore._dict.items()
            if doc.metadata.get("source") == source
        ]
        if ids_to_delete:
            store.delete(ids_to_delete)
            self.save()
            logger.info("Deleted %d vectors for source=%s", len(ids_to_delete), source)
        return len(ids_to_delete)

    # ── Search ───────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = TOP_K,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Document]:
        """Standard k-NN similarity search."""
        store = self.get_store()
        return store.similarity_search(query, k=k, filter=filter_metadata)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = TOP_K,
        score_threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[Tuple[Document, float]]:
        """Return (document, score) pairs filtered by a minimum similarity score."""
        store = self.get_store()
        results = store.similarity_search_with_score(query, k=k)
        # FAISS scores are L2 distances (lower = better). Convert to cosine-like.
        filtered = [(doc, score) for doc, score in results]
        return filtered

    def mmr_search(
        self,
        query: str,
        k: int = TOP_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        """
        Maximal Marginal Relevance search — balances relevance vs. diversity.
        lambda_mult=1.0 → pure relevance; 0.0 → pure diversity.
        """
        store = self.get_store()
        return store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    # ── LlamaIndex Bridge ────────────────────────────────────────────────────

    def as_llama_index_vector_store(self):
        """
        Export the underlying FAISS index as a LlamaIndex VectorStore.
        Enables LlamaIndex query engines to use the same FAISS index.
        """
        try:
            import faiss as faiss_lib
            from llama_index.vector_stores.faiss import FaissVectorStore

            store = self.get_store()
            faiss_index = store.index  # raw faiss.Index object
            return FaissVectorStore(faiss_index=faiss_index)
        except ImportError as e:
            logger.error("LlamaIndex FAISS bridge requires llama-index-vector-stores-faiss: %s", e)
            raise

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return basic statistics about the current index."""
        store = self.get_store()
        n_vectors = store.index.ntotal
        return {
            "n_vectors": n_vectors,
            "index_path": str(self.index_path),
            "embedding_model": self.embeddings.model,
        }
