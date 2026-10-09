"""
Chunking Strategies Module
===========================
Implements all major text-chunking strategies used in RAG pipelines:

1. Fixed-Size (Character)        – simple, fast baseline
2. Recursive Character           – respects semantic boundaries (LangChain default)
3. Token-Based                   – respects LLM token limits
4. Semantic / Sentence           – groups sentences by embedding similarity
5. Markdown Header               – splits on ## headings
6. HTML Section                  – splits on HTML tags
7. Table-Aware                   – keeps table blocks intact
8. Sentence Window               – sentence-level with context window
9. Overlap Sliding Window        – classic sliding window

Each strategy is a class with a `split(docs)` method that returns
a list of LangChain Documents.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


# ── 1. Fixed-Size (Character) ─────────────────────────────────────────────────

class FixedSizeChunker:
    """Split text into fixed-size character chunks with optional overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
        )

    def split(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        return self._tag(chunks, "fixed_size")

    @staticmethod
    def _tag(chunks: List[Document], strategy: str) -> List[Document]:
        for i, c in enumerate(chunks):
            c.metadata["chunking_strategy"] = strategy
            c.metadata["chunk_index"] = i
        return chunks


# ── 2. Recursive Character ────────────────────────────────────────────────────

class RecursiveChunker:
    """
    Recursively split using a hierarchy of separators:
    paragraph → sentence → word → character.
    Best general-purpose strategy for prose text.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def split(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        return self._tag(chunks, "recursive")

    @staticmethod
    def _tag(chunks: List[Document], strategy: str) -> List[Document]:
        for i, c in enumerate(chunks):
            c.metadata["chunking_strategy"] = strategy
            c.metadata["chunk_index"] = i
        return chunks


# ── 3. Token-Based ────────────────────────────────────────────────────────────

class TokenChunker:
    """
    Split by token count rather than character count.
    Uses tiktoken (cl100k_base) so chunk sizes align with LLM context windows.
    Falls back to RecursiveChunker if tiktoken is unavailable.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._build_splitter()

    def _build_splitter(self):
        try:
            from langchain_text_splitters import TokenTextSplitter
            self.splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base",
            )
            self._use_token = True
        except Exception:
            logger.warning("TokenTextSplitter unavailable; using RecursiveChunker.")
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size * 4,
                chunk_overlap=self.chunk_overlap * 4,
            )
            self._use_token = False

    def split(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata["chunking_strategy"] = "token"
            c.metadata["chunk_index"] = i
        return chunks


# ── 4. Semantic / Sentence Similarity ────────────────────────────────────────

class SemanticChunker:
    """
    Group sentences by embedding similarity so each chunk is
    semantically coherent. Requires google embeddings at init.

    Algorithm:
      1. Split into sentences.
      2. Embed each sentence.
      3. Compute cosine distance between consecutive sentences.
      4. Break when distance exceeds threshold.
    """

    def __init__(self, embeddings, breakpoint_threshold: float = 0.3):
        self.embeddings = embeddings
        self.threshold = breakpoint_threshold

    def split(self, docs: List[Document]) -> List[Document]:
        try:
            from langchain_experimental.text_splitter import SemanticChunker as LC_SC
            splitter = LC_SC(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=self.threshold * 100,
            )
            chunks = splitter.split_documents(docs)
        except ImportError:
            logger.warning(
                "langchain_experimental not installed; falling back to RecursiveChunker."
            )
            chunks = RecursiveChunker().split(docs)

        for i, c in enumerate(chunks):
            c.metadata["chunking_strategy"] = "semantic"
            c.metadata["chunk_index"] = i
        return chunks


# ── 5. Markdown Header ───────────────────────────────────────────────────────

class MarkdownChunker:
    """
    Split Markdown documents on heading boundaries (H1–H3).
    Each chunk carries the header hierarchy in its metadata.
    """

    HEADERS = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS,
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, docs: List[Document]) -> List[Document]:
        all_chunks: List[Document] = []
        for doc in docs:
            try:
                md_docs = self.md_splitter.split_text(doc.page_content)
                for md_doc in md_docs:
                    md_doc.metadata.update(doc.metadata)
                sub_chunks = self.text_splitter.split_documents(md_docs)
                all_chunks.extend(sub_chunks)
            except Exception as e:
                logger.warning("MarkdownChunker fallback for doc: %s", e)
                all_chunks.extend(RecursiveChunker().split([doc]))

        for i, c in enumerate(all_chunks):
            c.metadata["chunking_strategy"] = "markdown_header"
            c.metadata["chunk_index"] = i
        return all_chunks


# ── 6. HTML Section ──────────────────────────────────────────────────────────

class HTMLChunker:
    """Split HTML content on section tags (<h1>–<h4>, <section>, <article>)."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, docs: List[Document]) -> List[Document]:
        try:
            from langchain_text_splitters import HTMLHeaderTextSplitter

            headers = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
            ]
            splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            all_chunks: List[Document] = []
            for doc in docs:
                try:
                    html_docs = splitter.split_text(doc.page_content)
                    for hd in html_docs:
                        hd.metadata.update(doc.metadata)
                    all_chunks.extend(text_splitter.split_documents(html_docs))
                except Exception:
                    all_chunks.extend(RecursiveChunker().split([doc]))

        except ImportError:
            logger.warning("HTMLHeaderTextSplitter unavailable; using RecursiveChunker.")
            all_chunks = RecursiveChunker(self.chunk_size, self.chunk_overlap).split(docs)

        for i, c in enumerate(all_chunks):
            c.metadata["chunking_strategy"] = "html_section"
            c.metadata["chunk_index"] = i
        return all_chunks


# ── 7. Table-Aware ───────────────────────────────────────────────────────────

class TableAwareChunker:
    """
    Keep markdown table blocks (produced by PDFTableLoader) intact inside
    a single chunk. Non-table text is split with RecursiveChunker.

    Strategy:
      - Detect [TABLE …] blocks using regex.
      - Extract them as standalone chunks.
      - Split the remaining prose normally.
      - Re-merge in document order.
    """

    TABLE_PATTERN = re.compile(
        r"(\[TABLE \d+\]\n(?:\|.+\|\n?)+)", re.MULTILINE
    )

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.prose_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, docs: List[Document]) -> List[Document]:
        all_chunks: List[Document] = []

        for doc in docs:
            parts = self.TABLE_PATTERN.split(doc.page_content)
            for part in parts:
                if not part.strip():
                    continue
                is_table = bool(self.TABLE_PATTERN.match(part.strip()))
                chunk_doc = Document(page_content=part, metadata=dict(doc.metadata))
                chunk_doc.metadata["is_table_chunk"] = is_table

                if is_table:
                    all_chunks.append(chunk_doc)
                else:
                    all_chunks.extend(self.prose_splitter.split_documents([chunk_doc]))

        for i, c in enumerate(all_chunks):
            c.metadata["chunking_strategy"] = "table_aware"
            c.metadata["chunk_index"] = i
        return all_chunks


# ── 8. Sentence Window ───────────────────────────────────────────────────────

class SentenceWindowChunker:
    """
    Split into individual sentences, then attach a surrounding window
    (k sentences before + after) as context stored in metadata.
    The chunk itself is ONE sentence; the window is used at retrieval time.
    """

    SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, window_size: int = 2):
        self.window = window_size

    def split(self, docs: List[Document]) -> List[Document]:
        all_chunks: List[Document] = []

        for doc in docs:
            sentences = self.SENTENCE_END.split(doc.page_content.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

            for i, sent in enumerate(sentences):
                start = max(0, i - self.window)
                end = min(len(sentences), i + self.window + 1)
                window_text = " ".join(sentences[start:end])

                chunk = Document(
                    page_content=sent,
                    metadata={
                        **doc.metadata,
                        "window_context": window_text,
                        "sentence_index": i,
                        "chunking_strategy": "sentence_window",
                        "chunk_index": len(all_chunks),
                    },
                )
                all_chunks.append(chunk)

        return all_chunks


# ── 9. Overlap Sliding Window ─────────────────────────────────────────────────

class SlidingWindowChunker:
    """
    Classic sliding-window over characters.
    Equivalent to FixedSizeChunker but makes the overlap explicit and
    adds positional metadata useful for debugging.
    """

    def __init__(self, chunk_size: int = 1000, stride: int = 500):
        self.chunk_size = chunk_size
        self.stride = stride

    def split(self, docs: List[Document]) -> List[Document]:
        all_chunks: List[Document] = []

        for doc in docs:
            text = doc.page_content
            start = 0
            chunk_idx = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk = Document(
                    page_content=text[start:end],
                    metadata={
                        **doc.metadata,
                        "char_start": start,
                        "char_end": end,
                        "chunking_strategy": "sliding_window",
                        "chunk_index": chunk_idx,
                    },
                )
                all_chunks.append(chunk)
                chunk_idx += 1
                if end == len(text):
                    break
                start += self.stride

        return all_chunks


# ── Strategy Registry & Factory ──────────────────────────────────────────────

STRATEGY_MAP: Dict[str, Any] = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "token": TokenChunker,
    "semantic": SemanticChunker,
    "markdown": MarkdownChunker,
    "html": HTMLChunker,
    "table_aware": TableAwareChunker,
    "sentence_window": SentenceWindowChunker,
    "sliding_window": SlidingWindowChunker,
}


def get_chunker(strategy: str, **kwargs):
    """
    Factory function to instantiate a chunker by strategy name.

    Args:
        strategy: one of the keys in STRATEGY_MAP
        **kwargs: passed to the chunker constructor

    Returns:
        A chunker instance with a .split(docs) method.
    """
    if strategy not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Choose from: {list(STRATEGY_MAP.keys())}"
        )
    return STRATEGY_MAP[strategy](**kwargs)


def chunk_documents(
    docs: List[Document],
    strategy: str = "recursive",
    **kwargs,
) -> List[Document]:
    """Convenience wrapper: load strategy, split docs, return chunks."""
    chunker = get_chunker(strategy, **kwargs)
    chunks = chunker.split(docs)
    logger.info(
        "Chunking strategy='%s' produced %d chunks from %d doc(s).",
        strategy, len(chunks), len(docs),
    )
    return chunks
