"""
RAG Production CLI
==================
Entry point for the RAG application.

Commands:
  ingest   – ingest documents into the vector store
  query    – ask a question (LangGraph pipeline)
  llama    – ask a question (LlamaIndex pipeline)
  compare  – compare LangGraph vs LlamaIndex answers
  stats    – show vector store statistics
  list     – list all chunking strategies
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config.settings import GOOGLE_API_KEY, DATA_DIR
from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore.faiss_store import FAISSVectorStore
from src.chunking.strategies import STRATEGY_MAP

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_api_key():
    if not GOOGLE_API_KEY:
        console.print(
            "[bold red]ERROR:[/bold red] GOOGLE_API_KEY is not set. "
            "Create a .env file (see .env.example) or export the variable."
        )
        sys.exit(1)


def _get_store() -> FAISSVectorStore:
    store = FAISSVectorStore(google_api_key=GOOGLE_API_KEY)
    if not store.is_built():
        console.print(
            "[yellow]No FAISS index found. Run 'python main.py ingest' first.[/yellow]"
        )
        sys.exit(1)
    return store


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    _check_api_key()
    console.rule("[bold blue]Data Ingestion[/bold blue]")

    source = Path(args.source) if args.source else DATA_DIR

    if args.all_strategies:
        console.print(f"[cyan]Ingesting with ALL strategies from:[/cyan] {source}")
        pipeline = IngestionPipeline(
            pdf_mode=args.pdf_mode,
            google_api_key=GOOGLE_API_KEY,
        )
        results = pipeline.ingest_multiple_strategies(source)
        _print_multi_results(results)
        return

    console.print(
        f"[cyan]Source:[/cyan] {source}\n"
        f"[cyan]Strategy:[/cyan] {args.strategy}\n"
        f"[cyan]PDF mode:[/cyan] {args.pdf_mode}\n"
        f"[cyan]Chunk size:[/cyan] {args.chunk_size} | "
        f"[cyan]Overlap:[/cyan] {args.chunk_overlap}"
    )

    pipeline = IngestionPipeline(
        chunking_strategy=args.strategy,
        pdf_mode=args.pdf_mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        google_api_key=GOOGLE_API_KEY,
    )

    with console.status("[bold green]Ingesting documents…"):
        stats = pipeline.ingest(source, incremental=args.incremental)

    _print_stats_table(stats, title="Ingestion Results")


def cmd_query(args):
    _check_api_key()
    console.rule("[bold blue]RAG Query (LangGraph)[/bold blue]")

    from src.graph.rag_graph import run_rag

    store = _get_store()
    question = " ".join(args.question)
    console.print(f"[bold]Q:[/bold] {question}\n")

    with console.status("[bold green]Thinking…"):
        result = run_rag(question, store, GOOGLE_API_KEY)

    _print_answer(result)


def cmd_llama(args):
    _check_api_key()
    console.rule("[bold blue]RAG Query (LlamaIndex)[/bold blue]")

    from src.retrieval.llama_index_retriever import LlamaIndexRetriever

    store = _get_store()
    question = " ".join(args.question)
    console.print(f"[bold]Q:[/bold] {question}\n")

    with console.status("[bold green]Thinking (LlamaIndex)…"):
        retriever = LlamaIndexRetriever(store, GOOGLE_API_KEY, top_k=5)
        result = retriever.query(question)

    _print_answer(result)


def cmd_compare(args):
    _check_api_key()
    console.rule("[bold blue]Compare: LangGraph vs LlamaIndex[/bold blue]")

    from src.graph.rag_graph import run_rag
    from src.retrieval.llama_index_retriever import LlamaIndexRetriever

    store = _get_store()
    question = " ".join(args.question)
    console.print(f"[bold]Q:[/bold] {question}\n")

    with console.status("[bold green]LangGraph pipeline…"):
        lg_result = run_rag(question, store, GOOGLE_API_KEY)

    with console.status("[bold green]LlamaIndex pipeline…"):
        retriever = LlamaIndexRetriever(store, GOOGLE_API_KEY, top_k=5)
        li_result = retriever.query(question)

    console.print(Panel(lg_result.get("answer", "—"), title="[bold cyan]LangGraph Answer[/bold cyan]"))
    console.print(Panel(li_result.get("answer", "—"), title="[bold magenta]LlamaIndex Answer[/bold magenta]"))


def cmd_stats(args):
    _check_api_key()
    console.rule("[bold blue]Vector Store Statistics[/bold blue]")
    store = _get_store()
    info = store.stats()
    _print_stats_table(info, title="FAISS Index Stats")


def cmd_list(_args):
    console.rule("[bold blue]Chunking Strategies[/bold blue]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Strategy", style="green")
    table.add_column("Description")

    descriptions = {
        "fixed_size":      "Split on fixed character count with overlap",
        "recursive":       "Recursive split on paragraphs → sentences → words (default)",
        "token":           "Split by token count (cl100k_base tokenizer)",
        "semantic":        "Group sentences by embedding similarity",
        "markdown":        "Split on Markdown heading boundaries (H1–H3)",
        "html":            "Split on HTML section tags (h1–h4)",
        "table_aware":     "Keep table blocks intact; split prose separately",
        "sentence_window": "One sentence per chunk with surrounding context window",
        "sliding_window":  "Classic sliding window with configurable stride",
    }

    for name, desc in descriptions.items():
        table.add_row(name, desc)

    console.print(table)


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_stats_table(data: dict, title: str = "Stats"):
    table = Table(show_header=True, header_style="bold cyan", title=title)
    table.add_column("Key", style="green")
    table.add_column("Value")
    for k, v in data.items():
        table.add_row(str(k), str(v))
    console.print(table)


def _print_multi_results(results: dict):
    table = Table(show_header=True, header_style="bold cyan", title="Multi-Strategy Ingestion")
    table.add_column("Strategy", style="green")
    table.add_column("Status")
    table.add_column("Chunks")
    table.add_column("Time (s)")
    for strategy, info in results.items():
        table.add_row(
            strategy,
            "[green]OK[/green]" if info.get("status") == "ok" else "[red]ERROR[/red]",
            str(info.get("non_empty_chunks", info.get("error", "—"))),
            str(info.get("elapsed_seconds", "—")),
        )
    console.print(table)


def _print_answer(result: dict):
    console.print(Panel(
        result.get("answer", "No answer."),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    ))

    sources = result.get("sources", [])
    if sources:
        console.print(f"\n[bold]Sources ({len(sources)}):[/bold]")
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            src_path = meta.get("source", "unknown")
            page = meta.get("page", "")
            page_info = f" (p.{page})" if page else ""
            excerpt = src.get("content", "")[:120].replace("\n", " ")
            console.print(
                f"  [cyan]{i}.[/cyan] {Path(src_path).name}{page_info}\n"
                f"     [dim]{excerpt}…[/dim]"
            )


# ── CLI Parser ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG Production Application — LangChain + LangGraph + LlamaIndex + FAISS",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- ingest ----
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the vector store")
    p_ingest.add_argument("--source", "-s", help="File or directory to ingest (default: data/raw)")
    p_ingest.add_argument(
        "--strategy", "-c",
        default="recursive",
        choices=list(STRATEGY_MAP.keys()),
        help="Chunking strategy (default: recursive)",
    )
    p_ingest.add_argument("--pdf-mode", default="diagram", choices=["basic", "table", "diagram"])
    p_ingest.add_argument("--chunk-size", type=int, default=1000)
    p_ingest.add_argument("--chunk-overlap", type=int, default=200)
    p_ingest.add_argument("--incremental", action="store_true", help="Add to existing index")
    p_ingest.add_argument("--all-strategies", action="store_true", help="Run all chunking strategies")
    p_ingest.set_defaults(func=cmd_ingest)

    # ---- query ----
    p_query = sub.add_parser("query", help="Ask a question using LangGraph RAG pipeline")
    p_query.add_argument("question", nargs="+", help="Your question")
    p_query.set_defaults(func=cmd_query)

    # ---- llama ----
    p_llama = sub.add_parser("llama", help="Ask a question using LlamaIndex pipeline")
    p_llama.add_argument("question", nargs="+", help="Your question")
    p_llama.set_defaults(func=cmd_llama)

    # ---- compare ----
    p_cmp = sub.add_parser("compare", help="Compare LangGraph vs LlamaIndex answers")
    p_cmp.add_argument("question", nargs="+", help="Your question")
    p_cmp.set_defaults(func=cmd_compare)

    # ---- stats ----
    p_stats = sub.add_parser("stats", help="Show vector store statistics")
    p_stats.set_defaults(func=cmd_stats)

    # ---- list ----
    p_list = sub.add_parser("list", help="List available chunking strategies")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
