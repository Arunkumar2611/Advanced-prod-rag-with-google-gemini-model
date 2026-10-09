# Q3 2024 AI Technology Report

## Executive Summary

This report covers the latest developments in AI embedding models, vector databases,
and retrieval-augmented generation frameworks for Q3 2024. Key findings indicate a
30% improvement in retrieval accuracy with semantic chunking versus fixed-size chunking.

## Model Performance Comparison

The following table shows benchmark results on the BEIR dataset:

| Model | NDCG@10 | MRR@10 | Recall@100 |
| --- | --- | --- | --- |
| text-embedding-004 | 0.847 | 0.792 | 0.961 |
| text-embedding-3-large | 0.831 | 0.778 | 0.958 |
| cohere-embed-v3 | 0.822 | 0.769 | 0.952 |
| bge-m3 | 0.819 | 0.765 | 0.949 |

## Chunking Strategy Benchmarks

Experiments were conducted on a 10,000 document corpus from the medical domain.

| Strategy | Recall@5 | Precision@5 | Latency (ms) |
| --- | --- | --- | --- |
| Semantic | 0.74 | 0.68 | 45 |
| Sentence Window | 0.72 | 0.71 | 38 |
| Recursive | 0.69 | 0.65 | 12 |
| Fixed-size | 0.61 | 0.58 | 10 |
| Token-based | 0.65 | 0.62 | 15 |

## Architecture Recommendations

### For Production Systems

1. **Embedding Model**: Google text-embedding-004 for highest accuracy.
2. **Vector Store**: FAISS for on-premise; Pinecone for fully managed.
3. **Chunking**: Semantic chunking for accuracy-sensitive use cases;
   recursive for general purpose.
4. **Re-ranking**: Always apply a cross-encoder re-ranker on top-20 results.

### For Development / Prototyping

1. Fixed-size chunking with `chunk_size=512` is fast and predictable.
2. FAISS CPU is sufficient for corpora up to 1M vectors.
3. Use MMR retrieval to avoid redundant retrieved chunks.

## Cost Analysis

| Component | Cost per 1M tokens |
| --- | --- |
| text-embedding-004 | $0.025 |
| Gemini 1.5 Flash (input) | $0.075 |
| Gemini 1.5 Pro (input) | $1.25 |

## Flow: Data Ingestion Pipeline

```
[Raw Documents]
     │
     ▼
[Document Loader]  ← PDF, DOCX, TXT, Images
     │
     ▼
[Chunking Strategy]  ← recursive / semantic / table-aware
     │
     ▼
[Google Embedding]  ← text-embedding-004
     │
     ▼
[FAISS Vector Store]  ← persist to disk
```

## Conclusion

RAG systems continue to mature rapidly. The combination of semantic chunking,
high-quality embeddings, and MMR retrieval provides the best trade-off between
retrieval quality and computational cost.

Next steps for the team:
- Evaluate HyDE for improved retrieval on ambiguous queries.
- Implement query routing for multi-domain knowledge bases.
- Add streaming generation support for improved UX.
