"""
LangGraph RAG Workflow
=======================
Implements a stateful, multi-step RAG pipeline as a directed graph:

  [retrieve] → [grade_documents] → [generate] → [grade_answer]
                      ↓ (irrelevant)
               [rewrite_query] → [retrieve]  (up to max_retries)

State keys:
  question       – original user question
  rewritten_q    – optionally rewritten query
  documents      – retrieved Document objects
  generation     – LLM-generated answer
  retry_count    – number of retry loops executed
  metadata       – passthrough dict for tracing
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import GOOGLE_API_KEY, LLM_MODEL, TOP_K
from src.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


# ── State Schema ─────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    rewritten_q: Optional[str]
    documents: List[Document]
    generation: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]


# ── Prompt Templates ─────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant. Answer the question based ONLY on the
provided context. If the context does not contain enough information, say
so honestly.

Context:
{context}

Question: {question}

Provide a clear, accurate, and well-structured answer."""
)

GRADE_DOC_PROMPT = ChatPromptTemplate.from_template(
    """You are a relevance grader. Given the question and a retrieved document,
determine if the document is relevant.

Question: {question}
Document: {document}

Reply with exactly one word: "yes" if relevant, "no" if not."""
)

GRADE_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a quality grader. Does the answer address the question?

Question: {question}
Answer: {answer}

Reply with exactly one word: "yes" if it does, "no" if it does not."""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """The retrieved documents were not relevant for this question.
Rewrite the question to be clearer and more searchable.

Original question: {question}

Rewritten question (respond with ONLY the question text):"""
)


# ── Node Functions ────────────────────────────────────────────────────────────

def build_nodes(vector_store: FAISSVectorStore, llm: ChatGoogleGenerativeAI):

    # ---- retrieve ----
    def retrieve(state: RAGState) -> RAGState:
        query = state.get("rewritten_q") or state["question"]
        logger.info("[retrieve] query=%r", query)
        docs = vector_store.mmr_search(query, k=TOP_K)
        state["documents"] = docs
        state["metadata"]["retrieved_docs"] = len(docs)
        return state

    # ---- grade_documents ----
    def grade_documents(state: RAGState) -> RAGState:
        question = state.get("rewritten_q") or state["question"]
        grader = GRADE_DOC_PROMPT | llm | StrOutputParser()

        relevant = []
        for doc in state["documents"]:
            try:
                verdict = grader.invoke({
                    "question": question,
                    "document": doc.page_content[:1000],
                }).strip().lower()
            except Exception:
                verdict = "yes"  # conservative: keep on error

            if verdict.startswith("y"):
                relevant.append(doc)

        state["documents"] = relevant
        state["metadata"]["relevant_docs"] = len(relevant)
        logger.info(
            "[grade_documents] %d/%d docs passed grading.",
            len(relevant), state["metadata"].get("retrieved_docs", 0),
        )
        return state

    # ---- generate ----
    def generate(state: RAGState) -> RAGState:
        question = state.get("rewritten_q") or state["question"]
        context = "\n\n---\n\n".join(
            d.page_content for d in state["documents"]
        ) or "No relevant context found."

        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        state["generation"] = answer
        logger.info("[generate] answer length=%d chars", len(answer))
        return state

    # ---- grade_answer ----
    def grade_answer(state: RAGState) -> RAGState:
        grader = GRADE_ANSWER_PROMPT | llm | StrOutputParser()
        try:
            verdict = grader.invoke({
                "question": state["question"],
                "answer": state["generation"],
            }).strip().lower()
        except Exception:
            verdict = "yes"

        state["metadata"]["answer_graded"] = verdict
        logger.info("[grade_answer] verdict=%s", verdict)
        return state

    # ---- rewrite_query ----
    def rewrite_query(state: RAGState) -> RAGState:
        chain = REWRITE_PROMPT | llm | StrOutputParser()
        try:
            new_q = chain.invoke({"question": state["question"]}).strip()
        except Exception:
            new_q = state["question"]

        state["rewritten_q"] = new_q
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["metadata"]["rewritten_query"] = new_q
        logger.info("[rewrite_query] new_q=%r (retry #%d)", new_q, state["retry_count"])
        return state

    return retrieve, grade_documents, generate, grade_answer, rewrite_query


# ── Routing Logic ─────────────────────────────────────────────────────────────

MAX_RETRIES = 2


def route_after_grading(state: RAGState) -> str:
    """After grading documents: generate if we have relevant docs, else retry."""
    if state["documents"]:
        return "generate"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        logger.info("[route] Max retries reached. Generating with no context.")
        return "generate"
    return "rewrite_query"


def route_after_answer(state: RAGState) -> str:
    """After grading answer: end if good, retry if bad and retries remain."""
    verdict = state.get("metadata", {}).get("answer_graded", "yes")
    if verdict.startswith("y"):
        return END
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return END
    return "rewrite_query"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_rag_graph(vector_store: FAISSVectorStore, google_api_key: str = GOOGLE_API_KEY):
    """
    Construct and compile the LangGraph RAG workflow.

    Returns a compiled graph whose .invoke(state) method runs the pipeline.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=google_api_key,
        temperature=0.2,
    )

    retrieve, grade_documents, generate, grade_answer, rewrite_query = build_nodes(
        vector_store, llm
    )

    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("rewrite_query", rewrite_query)

    # Entry point
    graph.set_entry_point("retrieve")

    # Fixed edges
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("generate", "grade_answer")
    graph.add_edge("rewrite_query", "retrieve")

    # Conditional edges
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_conditional_edges(
        "grade_answer",
        route_after_answer,
        {END: END, "rewrite_query": "rewrite_query"},
    )

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────────────────

def run_rag(question: str, vector_store: FAISSVectorStore, google_api_key: str = GOOGLE_API_KEY) -> Dict[str, Any]:
    """Run the full RAG pipeline for a single question and return results."""
    app = build_rag_graph(vector_store, google_api_key)

    initial_state: RAGState = {
        "question": question,
        "rewritten_q": None,
        "documents": [],
        "generation": None,
        "retry_count": 0,
        "metadata": {},
    }

    final_state = app.invoke(initial_state)

    return {
        "question": final_state["question"],
        "answer": final_state.get("generation", "No answer generated."),
        "sources": [
            {
                "content": d.page_content[:200],
                "metadata": d.metadata,
            }
            for d in final_state.get("documents", [])
        ],
        "metadata": final_state.get("metadata", {}),
    }
