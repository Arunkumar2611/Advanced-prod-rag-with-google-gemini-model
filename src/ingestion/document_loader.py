"""
Document Loader Module
======================
Supports loading: plain text, PDF (text-only), PDF with tables,
PDF with embedded graphs/flow-diagrams, DOCX, images (OCR).

Each loader returns a list of LangChain Document objects enriched
with metadata that downstream chunkers and the vector store rely on.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_import(module: str, package: Optional[str] = None):
    """Return module or None if not installed."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        pkg = package or module
        logger.warning("Optional dependency '%s' not installed. Some loaders may be unavailable.", pkg)
        return None


# ── Plain Text ────────────────────────────────────────────────────────────────

class TextLoader:
    """Load plain-text files (.txt, .md, .rst, .csv)."""

    def load(self, file_path: str | Path) -> List[Document]:
        file_path = Path(file_path)
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("TextLoader failed for %s: %s", file_path, e)
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "loader": "TextLoader",
                },
            )
        ]


# ── Basic PDF ─────────────────────────────────────────────────────────────────

class PDFLoader:
    """Load PDF files page-by-page using pypdf."""

    def load(self, file_path: str | Path) -> List[Document]:
        pypdf = _safe_import("pypdf")
        if pypdf is None:
            return []

        file_path = Path(file_path)
        docs: List[Document] = []
        try:
            reader = pypdf.PdfReader(str(file_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "file_type": ".pdf",
                            "page": i + 1,
                            "total_pages": len(reader.pages),
                            "loader": "PDFLoader",
                        },
                    )
                )
        except Exception as e:
            logger.error("PDFLoader failed for %s: %s", file_path, e)
        return docs


# ── PDF with Tables ───────────────────────────────────────────────────────────

class PDFTableLoader:
    """
    Extract text AND tables from PDFs.
    Uses pdfplumber for layout-aware extraction; falls back to pypdf if unavailable.
    Tables are serialised as markdown so they survive chunking intact.
    """

    def load(self, file_path: str | Path) -> List[Document]:
        pdfplumber = _safe_import("pdfplumber")
        file_path = Path(file_path)
        docs: List[Document] = []

        if pdfplumber is None:
            logger.warning("pdfplumber not available; falling back to basic PDFLoader.")
            return PDFLoader().load(file_path)

        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # --- plain text ---
                    text = page.extract_text() or ""

                    # --- tables ---
                    table_md_blocks: List[str] = []
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        if not table:
                            continue
                        md = self._table_to_markdown(table)
                        table_md_blocks.append(f"\n\n[TABLE {t_idx + 1}]\n{md}\n")

                    combined = text + "".join(table_md_blocks)

                    docs.append(
                        Document(
                            page_content=combined,
                            metadata={
                                "source": str(file_path),
                                "file_name": file_path.name,
                                "file_type": ".pdf",
                                "page": i + 1,
                                "total_pages": len(pdf.pages),
                                "has_tables": bool(tables),
                                "table_count": len(tables),
                                "loader": "PDFTableLoader",
                            },
                        )
                    )
        except Exception as e:
            logger.error("PDFTableLoader failed for %s: %s", file_path, e)
        return docs

    @staticmethod
    def _table_to_markdown(table: list) -> str:
        """Convert a list-of-lists table to a markdown table string."""
        if not table:
            return ""
        rows = []
        for r_idx, row in enumerate(table):
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            rows.append("| " + " | ".join(cells) + " |")
            if r_idx == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(rows)


# ── PDF with Graphs / Flow Diagrams ───────────────────────────────────────────

class PDFDiagramLoader:
    """
    Extract text from PDFs and detect pages that contain diagrams/graphs.
    Diagram pages are converted to images and OCR'd via pytesseract.
    A textual description tag is added to the metadata.
    """

    # Heuristic: if extracted text < MIN_TEXT_CHARS assume page is image-heavy
    MIN_TEXT_CHARS = 50

    def load(self, file_path: str | Path) -> List[Document]:
        file_path = Path(file_path)
        docs: List[Document] = []

        fitz = _safe_import("fitz", "pymupdf")          # PyMuPDF
        pytesseract = _safe_import("pytesseract")
        PIL = _safe_import("PIL.Image", "Pillow")

        # --- Primary extraction via PDFTableLoader so tables are kept ---
        base_docs = PDFTableLoader().load(file_path)

        for doc in base_docs:
            page_num = doc.metadata.get("page", 1)
            text_len = len(doc.page_content.strip())
            is_diagram_page = text_len < self.MIN_TEXT_CHARS

            if is_diagram_page and fitz is not None and pytesseract is not None and PIL is not None:
                try:
                    ocr_text = self._ocr_page(file_path, page_num - 1, fitz, pytesseract, PIL)
                    doc.page_content = (
                        f"[DIAGRAM / GRAPH PAGE {page_num}]\n"
                        f"OCR Extracted Text:\n{ocr_text}\n"
                    )
                    doc.metadata["has_diagram"] = True
                    doc.metadata["ocr_applied"] = True
                except Exception as e:
                    logger.warning("OCR failed for page %d of %s: %s", page_num, file_path, e)
                    doc.metadata["has_diagram"] = True
                    doc.metadata["ocr_applied"] = False
            else:
                doc.metadata["has_diagram"] = is_diagram_page
                doc.metadata["ocr_applied"] = False

            doc.metadata["loader"] = "PDFDiagramLoader"
            docs.append(doc)

        return docs

    @staticmethod
    def _ocr_page(file_path: Path, page_index: int, fitz, pytesseract, PIL) -> str:
        """Render PDF page to image and run OCR."""
        pdf = fitz.open(str(file_path))
        page = pdf[page_index]
        mat = fitz.Matrix(2.0, 2.0)          # 2x zoom for better OCR
        clip = page.get_pixmap(matrix=mat)
        img_bytes = clip.tobytes("png")
        image = PIL.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(image)
        pdf.close()
        return text


# ── DOCX ──────────────────────────────────────────────────────────────────────

class DocxLoader:
    """Load Microsoft Word (.docx) files."""

    def load(self, file_path: str | Path) -> List[Document]:
        docx = _safe_import("docx", "python-docx")
        file_path = Path(file_path)
        if docx is None:
            return []

        try:
            doc_obj = docx.Document(str(file_path))
            paragraphs = [p.text for p in doc_obj.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)

            # Also extract tables
            table_blocks = []
            for t_idx, table in enumerate(doc_obj.tables):
                rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                md = PDFTableLoader._table_to_markdown(rows)
                table_blocks.append(f"\n\n[TABLE {t_idx + 1}]\n{md}")

            full_text = text + "".join(table_blocks)
            return [
                Document(
                    page_content=full_text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": ".docx",
                        "has_tables": bool(doc_obj.tables),
                        "loader": "DocxLoader",
                    },
                )
            ]
        except Exception as e:
            logger.error("DocxLoader failed for %s: %s", file_path, e)
            return []


# ── Image / Diagram OCR ───────────────────────────────────────────────────────

class ImageOCRLoader:
    """Load image files and extract text via pytesseract OCR."""

    def load(self, file_path: str | Path) -> List[Document]:
        pytesseract = _safe_import("pytesseract")
        PIL = _safe_import("PIL.Image", "Pillow")
        file_path = Path(file_path)

        if pytesseract is None or PIL is None:
            logger.warning("pytesseract or Pillow not available; cannot load image.")
            return []

        try:
            image = PIL.open(str(file_path))
            text = pytesseract.image_to_string(image)
            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "loader": "ImageOCRLoader",
                        "ocr_applied": True,
                    },
                )
            ]
        except Exception as e:
            logger.error("ImageOCRLoader failed for %s: %s", file_path, e)
            return []


# ── Dispatcher ────────────────────────────────────────────────────────────────

class DocumentLoaderDispatcher:
    """
    Given a file path, pick the right loader automatically.

    loader_mode for PDFs:
      'basic'   → PDFLoader
      'table'   → PDFTableLoader   (default)
      'diagram' → PDFDiagramLoader
    """

    PDF_MODES = {
        "basic": PDFLoader,
        "table": PDFTableLoader,
        "diagram": PDFDiagramLoader,
    }

    def __init__(self, pdf_mode: str = "diagram"):
        self.pdf_mode = pdf_mode

    def load(self, file_path: str | Path) -> List[Document]:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext in (".txt", ".md", ".rst", ".csv"):
            return TextLoader().load(file_path)
        elif ext == ".pdf":
            loader_cls = self.PDF_MODES.get(self.pdf_mode, PDFDiagramLoader)
            return loader_cls().load(file_path)
        elif ext in (".docx", ".doc"):
            return DocxLoader().load(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            return ImageOCRLoader().load(file_path)
        else:
            logger.warning("Unsupported file type: %s — trying TextLoader.", ext)
            return TextLoader().load(file_path)

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """Load all supported documents from a directory."""
        from config.settings import ALL_SUPPORTED

        directory = Path(directory)
        allowed = extensions or ALL_SUPPORTED
        pattern = "**/*" if recursive else "*"
        all_docs: List[Document] = []

        for fp in directory.glob(pattern):
            if fp.is_file() and fp.suffix.lower() in allowed:
                logger.info("Loading: %s", fp)
                docs = self.load(fp)
                all_docs.extend(docs)

        logger.info("Loaded %d document(s) from %s", len(all_docs), directory)
        return all_docs
