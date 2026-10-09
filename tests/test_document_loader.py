"""
Unit tests for document loaders.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from src.ingestion.document_loader import (
    TextLoader,
    DocumentLoaderDispatcher,
    PDFTableLoader,
    TableAwareChunker,
)


class TestTextLoader:
    def test_load_txt(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello, RAG world!\nLine two.", encoding="utf-8")

        docs = TextLoader().load(f)
        assert len(docs) == 1
        assert "Hello, RAG world!" in docs[0].page_content
        assert docs[0].metadata["file_type"] == ".txt"
        assert docs[0].metadata["loader"] == "TextLoader"

    def test_load_markdown(self, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# Title\n\nSome content.", encoding="utf-8")

        docs = TextLoader().load(f)
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == ".md"

    def test_metadata_source(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3", encoding="utf-8")

        docs = TextLoader().load(f)
        assert docs[0].metadata["source"] == str(f)


class TestDocumentLoaderDispatcher:
    def test_dispatch_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Dispatcher test content.")

        dispatcher = DocumentLoaderDispatcher()
        docs = dispatcher.load(f)
        assert len(docs) == 1
        assert "Dispatcher test" in docs[0].page_content

    def test_load_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("Document A content.", encoding="utf-8")
        (tmp_path / "b.md").write_text("Document B content.", encoding="utf-8")
        (tmp_path / "ignored.xyz").write_text("Not loaded.", encoding="utf-8")

        dispatcher = DocumentLoaderDispatcher()
        docs = dispatcher.load_directory(tmp_path, recursive=False)
        assert len(docs) == 2

    def test_load_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.txt").write_text("Top level.", encoding="utf-8")
        (sub / "nested.txt").write_text("Nested.", encoding="utf-8")

        dispatcher = DocumentLoaderDispatcher()
        docs = dispatcher.load_directory(tmp_path, recursive=True)
        assert len(docs) == 2

    def test_unsupported_falls_back(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("Some content here.")

        dispatcher = DocumentLoaderDispatcher()
        docs = dispatcher.load(f)
        # Falls back to TextLoader, so content should be present
        assert len(docs) == 1


class TestTableMarkdown:
    """Test the table-to-markdown conversion."""

    def test_basic_table(self):
        table = [["Name", "Score"], ["Alice", "95"], ["Bob", "80"]]
        md = PDFTableLoader._table_to_markdown(table)
        assert "| Name | Score |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 95 |" in md

    def test_empty_table(self):
        md = PDFTableLoader._table_to_markdown([])
        assert md == ""

    def test_none_cells(self):
        table = [["A", None], ["B", "C"]]
        md = PDFTableLoader._table_to_markdown(table)
        assert "| A |  |" in md
