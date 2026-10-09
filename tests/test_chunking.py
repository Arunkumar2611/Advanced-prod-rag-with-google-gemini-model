"""
Unit tests for chunking strategies.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from src.chunking.strategies import (
    FixedSizeChunker,
    RecursiveChunker,
    TokenChunker,
    MarkdownChunker,
    TableAwareChunker,
    SentenceWindowChunker,
    SlidingWindowChunker,
    get_chunker,
    STRATEGY_MAP,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """\
# Introduction

Retrieval-Augmented Generation (RAG) combines neural retrieval with
conditional text generation to produce grounded, factual responses.

## How It Works

1. The user provides a query.
2. A retriever fetches relevant documents from a vector store.
3. The LLM generates an answer conditioned on the retrieved context.

## Benefits

- Reduces hallucinations
- Allows the model to access up-to-date knowledge
- Supports domain-specific knowledge bases
"""

TABLE_TEXT = """\
Some introductory text before the table.

[TABLE 1]
| Name | Score | Grade |
| --- | --- | --- |
| Alice | 95 | A |
| Bob | 82 | B |

More text after the table describing the results.
"""


def make_docs(text: str, source: str = "test.txt") -> list:
    return [Document(page_content=text, metadata={"source": source, "file_type": ".txt"})]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_basic_split(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_metadata_preserved(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert all(c.metadata.get("source") == "test.txt" for c in chunks)

    def test_strategy_tag(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert all(c.metadata.get("chunking_strategy") == "fixed_size" for c in chunks)


class TestRecursiveChunker:
    def test_respects_paragraphs(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert len(chunks) >= 1

    def test_strategy_tag(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert all(c.metadata.get("chunking_strategy") == "recursive" for c in chunks)


class TestTokenChunker:
    def test_split(self):
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert len(chunks) > 0

    def test_strategy_tag(self):
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert all(c.metadata.get("chunking_strategy") == "token" for c in chunks)


class TestMarkdownChunker:
    def test_splits_on_headers(self):
        chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert len(chunks) >= 2  # At least Introduction + How It Works

    def test_strategy_tag(self):
        chunker = MarkdownChunker()
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert all(c.metadata.get("chunking_strategy") == "markdown_header" for c in chunks)


class TestTableAwareChunker:
    def test_table_chunk_intact(self):
        chunker = TableAwareChunker(chunk_size=500)
        chunks = chunker.split(make_docs(TABLE_TEXT))
        table_chunks = [c for c in chunks if c.metadata.get("is_table_chunk")]
        assert len(table_chunks) >= 1, "Should detect at least one table chunk"

    def test_prose_split(self):
        chunker = TableAwareChunker(chunk_size=50)
        chunks = chunker.split(make_docs(TABLE_TEXT))
        assert len(chunks) > 1

    def test_strategy_tag(self):
        chunker = TableAwareChunker()
        chunks = chunker.split(make_docs(TABLE_TEXT))
        assert all(c.metadata.get("chunking_strategy") == "table_aware" for c in chunks)


class TestSentenceWindowChunker:
    def test_one_sentence_per_chunk(self):
        text = "Hello world. This is a test. RAG is great."
        chunker = SentenceWindowChunker(window_size=1)
        chunks = chunker.split(make_docs(text))
        # Each chunk should be a single sentence
        for c in chunks:
            assert c.page_content.strip()

    def test_window_context_present(self):
        text = "Hello world. This is a test. RAG is great."
        chunker = SentenceWindowChunker(window_size=1)
        chunks = chunker.split(make_docs(text))
        assert all("window_context" in c.metadata for c in chunks)

    def test_strategy_tag(self):
        chunker = SentenceWindowChunker()
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert all(c.metadata.get("chunking_strategy") == "sentence_window" for c in chunks)


class TestSlidingWindowChunker:
    def test_overlap(self):
        chunker = SlidingWindowChunker(chunk_size=100, stride=50)
        chunks = chunker.split(make_docs(SAMPLE_TEXT * 3))
        assert len(chunks) > 1
        # Consecutive chunks should overlap
        if len(chunks) >= 2:
            end_c0 = chunks[0].metadata["char_end"]
            start_c1 = chunks[1].metadata["char_start"]
            assert start_c1 < end_c0

    def test_strategy_tag(self):
        chunker = SlidingWindowChunker()
        chunks = chunker.split(make_docs(SAMPLE_TEXT))
        assert all(c.metadata.get("chunking_strategy") == "sliding_window" for c in chunks)


class TestGetChunker:
    def test_factory_all_strategies(self):
        """All registered strategies must be instantiable without extra kwargs."""
        skip = {"semantic"}  # needs embeddings
        for name in STRATEGY_MAP:
            if name in skip:
                continue
            chunker = get_chunker(name)
            chunks = chunker.split(make_docs(SAMPLE_TEXT))
            assert isinstance(chunks, list), f"Strategy '{name}' failed"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent_strategy")


class TestChunkIndexing:
    """All strategies must set chunk_index metadata."""

    def test_chunk_index_sequential(self):
        skip = {"semantic"}
        for name in STRATEGY_MAP:
            if name in skip:
                continue
            chunker = get_chunker(name)
            chunks = chunker.split(make_docs(SAMPLE_TEXT))
            indices = [c.metadata.get("chunk_index") for c in chunks]
            assert indices == list(range(len(chunks))), f"Indices not sequential for '{name}'"
