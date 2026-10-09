# RAG Production Application

A production-grade **Retrieval-Augmented Generation (RAG)** system built with
**LangChain**, **LangGraph**, **LlamaIndex**, **Google Generative AI embeddings**,
and **FAISS** vector store.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION PIPELINE                   │
│                                                             │
│  [Documents]                                                │
│  .txt .md .pdf .docx .png .jpg                              │
│       │                                                     │
│       ▼                                                     │
│  [Document Loaders]                                         │
│  ├── TextLoader          (plain text, markdown, CSV)        │
│  ├── PDFLoader           (text-only PDFs)                   │
│  ├── PDFTableLoader      (PDFs with tables → markdown)      │
│  ├── PDFDiagramLoader    (PDFs with graphs → OCR)           │
│  ├── DocxLoader          (Word documents + tables)          │
│  └── ImageOCRLoader      (PNG/JPG via pytesseract)          │
│       │                                                     │
│       ▼                                                     │
│  [Chunking Strategies]                                      │
│  ├── fixed_size          CharacterTextSplitter              │
│  ├── recursive           RecursiveCharacterTextSplitter     │
│  ├── token               TokenTextSplitter (tiktoken)       │
│  ├── semantic            Embedding similarity grouping      │
│  ├── markdown            MarkdownHeaderTextSplitter         │
│  ├── html                HTMLHeaderTextSplitter             │
│  ├── table_aware         Keeps tables intact                │
│  ├── sentence_window     1 sentence + context window        │
│  └── sliding_window      Classic overlapping windows        │
│       │                                                     │
│       ▼                                                     │
│  [Google text-embedding-004]                                │
│       │                                                     │
│       ▼                                                     │
│  [FAISS Vector Store]  ←→  [LlamaIndex Bridge]             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE (LangGraph)               │
│                                                             │
│  [User Query]                                               │
│       │                                                     │
│       ▼                                                     │
│  [retrieve]  MMR search → top-k documents                  │
│       │                                                     │
│       ▼                                                     │
│  [grade_documents]  LLM grades each doc for relevance      │
│       │                                                     │
│    relevant?                                                │
│  ┌───┴───┐                                                  │
│ yes      no                                                 │
│  │       ▼                                                  │
│  │  [rewrite_query] → [retrieve]  (up to 2 retries)        │
│  ▼                                                          │
│  [generate]  Gemini 1.5 Pro generates answer               │
│       │                                                     │
│       ▼                                                     │
│  [grade_answer]  LLM verifies answer quality               │
│       │                                                     │
│     good?                                                   │
│  ┌───┴───┐                                                  │
│ yes      no → [rewrite_query]                               │
│  │                                                          │
│  ▼                                                          │
│  [Final Answer + Sources]                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
RAG_PROD/
├── main.py                        # CLI entry point
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py               # Centralised configuration
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py    # All document loaders
│   │   └── pipeline.py           # Orchestration pipeline
│   ├── chunking/
│   │   └── strategies.py         # 9 chunking strategies
│   ├── vectorstore/
│   │   └── faiss_store.py        # FAISS + Google embeddings
│   ├── graph/
│   │   └── rag_graph.py          # LangGraph workflow
│   └── retrieval/
│       └── llama_index_retriever.py  # LlamaIndex integration
├── data/
│   └── raw/                      # Place your documents here
├── faiss_index/                  # Auto-created on first ingest
└── tests/
    ├── test_chunking.py
    └── test_document_loader.py
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: For OCR support (image/diagram PDFs), also install:
> - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
> - [Poppler](https://poppler.freedesktop.org/) (for pdf2image)

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```

Get a free Google AI key at: https://aistudio.google.com/app/apikey

### 3. Add Your Documents

```
data/raw/
├── my_report.pdf
├── notes.txt
├── diagram.png
└── data_sheet.docx
```

### 4. Ingest Documents

```bash
# Default: recursive chunking, diagram-aware PDF mode
python main.py ingest

# Specific chunking strategy
python main.py ingest --strategy semantic

# Ingest a specific file
python main.py ingest --source path/to/file.pdf --strategy table_aware

# Run ALL chunking strategies (creates separate indexes)
python main.py ingest --all-strategies

# Add to existing index (incremental)
python main.py ingest --source new_doc.pdf --incremental
```

### 5. Query

```bash
# LangGraph pipeline (with document grading + query rewriting)
python main.py query What is retrieval-augmented generation?

# LlamaIndex pipeline
python main.py llama What are the benefits of semantic chunking?

# Compare both pipelines side-by-side
python main.py compare What embedding model should I use for RAG?
```

### 6. Other Commands

```bash
# Show vector store stats
python main.py stats

# List all chunking strategies
python main.py list
```

---

## Chunking Strategies

| Strategy | Best For |
|---|---|
| `fixed_size` | Fast baseline, uniform chunks |
| `recursive` | General prose (default) |
| `token` | Strict token budget alignment |
| `semantic` | Highest retrieval quality |
| `markdown` | Markdown documentation |
| `html` | HTML web content |
| `table_aware` | PDFs/DOCX with tables |
| `sentence_window` | Fine-grained Q&A |
| `sliding_window` | Dense technical text |

---

## Document Support

| File Type | Loader | Table Support | Diagram/OCR |
|---|---|---|---|
| `.txt`, `.md`, `.csv` | TextLoader | — | — |
| `.pdf` (text) | PDFLoader | — | — |
| `.pdf` (tables) | PDFTableLoader | Markdown tables | — |
| `.pdf` (diagrams) | PDFDiagramLoader | Markdown tables | pytesseract |
| `.docx` | DocxLoader | Markdown tables | — |
| `.png`, `.jpg`, `.tiff` | ImageOCRLoader | — | pytesseract |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *required* | Google AI API key |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |
| `LLM_MODEL` | `gemini-1.5-pro` | Generation model |
| `FAISS_INDEX_PATH` | `./faiss_index` | Vector store location |
| `DATA_DIR` | `./data/raw` | Document source directory |
| `DEFAULT_CHUNK_SIZE` | `1000` | Default chunk size |
| `DEFAULT_CHUNK_OVERLAP` | `200` | Default chunk overlap |
| `TOP_K` | `5` | Number of documents to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum similarity score |
