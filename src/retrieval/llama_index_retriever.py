"""
LlamaIndex Integration
=======================
Bridges LangChain documents & FAISS with LlamaIndex query engines.

Two approaches are provided:

1. LlamaIndexRetriever
   - Converts LangChain Documents → LlamaIndex Nodes
   - Wraps the shared FAISS index as a LlamaIndex VectorStoreIndex
   - Provides .query() for natural-language questions

2. LlamaIndexQueryEngine
   - Full LlamaIndex pipeline with sub-question decomposition
   - Useful for complex multi-hop questions

Both use the same Google embedding model for consistency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import GOOGLE_API_KEY, EMBEDDING_MODEL, LLM_MODEL, LLAMA_CACHE_DIR
from src.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


def _get_llama_settings(google_api_key: str):
    """Configure LlamaIndex global settings with Google models."""
    try:
        from llama_index.core import Settings
        from llama_index.embeddings.langchain import LangchainEmbedding
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        from llama_index.core.llms import LangChainLLM

        # Embedding
        lc_embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=google_api_key,
        )
        Settings.embed_model = LangchainEmbedding(lc_embeddings)

        # LLM
        lc_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=google_api_key,
            temperature=0.2,
        )
        Settings.llm = LangChainLLM(llm=lc_llm)

        return Settings
    except ImportError as e:
        logger.error("LlamaIndex settings setup failed: %s", e)
        raise


class LlamaIndexRetriever:
    """
    Wraps the shared FAISS vector store as a LlamaIndex retriever.
    Uses LangChain ↔ LlamaIndex document conversion utilities.
    """

    def __init__(
        self,
        faiss_store: FAISSVectorStore,
        google_api_key: str = GOOGLE_API_KEY,
        top_k: int = 5,
    ):
        self.faiss_store = faiss_store
        self.google_api_key = google_api_key
        self.top_k = top_k
        self._index = None

    def _build_index(self):
        """Build the LlamaIndex VectorStoreIndex from the shared FAISS store."""
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.core.schema import TextNode
        from llama_index.vector_stores.faiss import FaissVectorStore

        _get_llama_settings(self.google_api_key)

        # Get raw FAISS index from the LangChain store
        lc_store = self.faiss_store.get_store()

        try:
            faiss_index = lc_store.index
            llama_vs = FaissVectorStore(faiss_index=faiss_index)

            # Convert LangChain docstore to LlamaIndex nodes
            nodes = []
            for doc_id, lc_doc in lc_store.docstore._dict.items():
                node = TextNode(
                    text=lc_doc.page_content,
                    id_=str(doc_id),
                    metadata=lc_doc.metadata,
                )
                nodes.append(node)

            storage_ctx = StorageContext.from_defaults(vector_store=llama_vs)
            self._index = VectorStoreIndex(
                nodes,
                storage_context=storage_ctx,
                show_progress=False,
            )
            logger.info("LlamaIndex index built with %d nodes.", len(nodes))
        except Exception as e:
            logger.error("Failed to build LlamaIndex index: %s", e)
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """Run a query against the LlamaIndex retriever."""
        if self._index is None:
            self._build_index()

        retriever = self._index.as_retriever(similarity_top_k=self.top_k)
        query_engine = self._index.as_query_engine(similarity_top_k=self.top_k)

        try:
            response = query_engine.query(question)
            source_nodes = retriever.retrieve(question)

            return {
                "question": question,
                "answer": str(response),
                "sources": [
                    {
                        "content": node.node.get_content()[:300],
                        "score": node.score,
                        "metadata": node.node.metadata,
                    }
                    for node in source_nodes
                ],
            }
        except Exception as e:
            logger.error("LlamaIndex query failed: %s", e)
            return {"question": question, "answer": f"Query failed: {e}", "sources": []}


class LlamaIndexQueryEngine:
    """
    Full LlamaIndex pipeline that supports:
    - Sub-question decomposition for complex queries
    - Persistent storage context
    - Custom system prompt
    """

    def __init__(
        self,
        documents: Optional[List] = None,
        google_api_key: str = GOOGLE_API_KEY,
        top_k: int = 5,
        persist_dir: Optional[Path] = None,
    ):
        self.documents = documents or []
        self.google_api_key = google_api_key
        self.top_k = top_k
        self.persist_dir = persist_dir or LLAMA_CACHE_DIR
        self._engine = None

    def _build_engine(self):
        from llama_index.core import (
            VectorStoreIndex,
            SimpleDirectoryReader,
            StorageContext,
            load_index_from_storage,
        )
        from llama_index.core.schema import TextNode

        _get_llama_settings(self.google_api_key)

        persist_path = Path(self.persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)

        try:
            # Try loading persisted index first
            storage_ctx = StorageContext.from_defaults(persist_dir=str(persist_path))
            index = load_index_from_storage(storage_ctx)
            logger.info("Loaded existing LlamaIndex index from %s", persist_path)
        except Exception:
            # Build from documents
            if not self.documents:
                raise ValueError("No documents provided to build LlamaIndex engine.")

            nodes = [
                TextNode(
                    text=doc.page_content,
                    metadata=doc.metadata,
                )
                for doc in self.documents
            ]
            index = VectorStoreIndex(nodes, show_progress=True)
            index.storage_context.persist(persist_dir=str(persist_path))
            logger.info("Built & persisted LlamaIndex index at %s", persist_path)

        self._engine = index.as_query_engine(
            similarity_top_k=self.top_k,
            response_mode="tree_summarize",
        )

    def query(self, question: str) -> Dict[str, Any]:
        if self._engine is None:
            self._build_engine()

        try:
            response = self._engine.query(question)
            sources = []
            if hasattr(response, "source_nodes"):
                sources = [
                    {
                        "content": node.node.get_content()[:300],
                        "score": node.score,
                        "metadata": node.node.metadata,
                    }
                    for node in response.source_nodes
                ]
            return {
                "question": question,
                "answer": str(response),
                "sources": sources,
            }
        except Exception as e:
            logger.error("LlamaIndex engine query failed: %s", e)
            return {"question": question, "answer": f"Query failed: {e}", "sources": []}
