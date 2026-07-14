# Near-Zero-Hallucination RAG Pipeline

A "retrieve, constrain, verify, abstain" RAG pipeline: hybrid (dense + BM25)
retrieval with reciprocal-rank fusion, citation-constrained generation,
atomic-claim faithfulness verification, and a calibrated abstention policy —
wired into a small self-correcting agent loop.

This is a portable, CPU-only, offline-runnable adaptation of the design
described in *"Building a RAG Pipeline for 10M+ Documents With Near-Zero
Hallucination."* The original uses a GPU, a local vLLM server, and
Hugging-Face-hosted embedding/reranker/generator models; this version swaps
those for components that need no GPU and no model download, while keeping
the same architecture and control flow, so you can run it immediately and
later drop in real models for production-grade answer quality.

## What's different from the original, and why

| Original | This version | Why |
|---|---|---|
| Qwen3-Embedding-4B (dense) | TF-IDF + Truncated SVD (scikit-learn) | No GPU / model download needed |
| Qwen3-Reranker-4B (cross-encoder) | LLM-prompted relevance scoring | Reuses whatever LLM backend you plug in |
| Qwen3-32B via local vLLM server | Pluggable `LLMClient` (mock / OpenAI / Anthropic) | Runs anywhere; swap in a real model with one flag |
| LangGraph agent graph | Plain Python loop (`RAGAgent.answer`) | Same control flow (route → retrieve → grade → refine/generate → verify → finalize), no extra dependency |
| LanceDB on-disk vector store | In-memory NumPy matrix | Fine at demo scale; see "Scaling" below for how to reach 10M+ |

The **contract is unchanged**: every sentence in an answer must carry a
citation to a real retrieved passage, every claim is checked against that
passage by a faithfulness judge, and the system abstains — returns "I do not
have enough supporting evidence..." — rather than emit an unsupported claim.

## Files

- `rag_pipeline.py` — the library: chunking, hybrid retrieval + RRF fusion,
  LLM reranking, routing/decomposition, cited generation, claim
  extraction, the verification gate, the abstention policy, and the
  `RAGAgent` self-correcting loop.
- `demo.py` — runs the whole pipeline over a small sample corpus (4
  passages) with one answerable and one unanswerable question, and prints
  the route/hops/status/citations for each.

## Quickstart (fully offline, no API key)

```bash
pip install scikit-learn rank_bm25 numpy
python demo.py
```

This uses `MockLLMClient`, a heuristic (non-neural) stand-in for the LLM so
the whole pipeline — router, generator, claim extractor, faithfulness judge —
runs with zero setup. Answer quality is naturally much lower than a real
model; it exists to prove the pipeline's control flow and abstention logic
work end-to-end.

## Running with a real LLM

```bash
pip install openai   # or: pip install anthropic
python demo.py --backend openai --api-key sk-... --model gpt-4o-mini
python demo.py --backend anthropic --api-key sk-ant-... --model claude-sonnet-5
```

`OpenAIClient` also works against any OpenAI-compatible endpoint (e.g. a
local vLLM server, exactly like the original design) by passing `base_url`.

## Using your own corpus

```python
from rag_pipeline import Passage, build_pipeline, OpenAIClient

passages = [
    Passage(id="doc1", title="...", text="..."),
    Passage(id="doc2", title="...", text="..."),
    # ...
]

llm = OpenAIClient(api_key="sk-...", model="gpt-4o-mini")
agent = build_pipeline(passages, llm, contextualize=True)

result = agent.answer("Your question here")
print(result["final"].status)     # "answered" or "abstained"
print(result["final"].answer)
print(result["final"].citations)
```

## Tuning

- `tau_claim` in `build_pipeline(...)` — the minimum per-claim faithfulness
  score to count as supported (default `0.3`, matching the article). Raise
  it to trade coverage for a lower hallucination rate.
- `crag_ok` / `crag_bad` / `max_hops` on `RAGAgent` — control the
  corrective-retrieval loop: evidence graded ≥ `crag_ok` generates
  immediately, evidence below `crag_bad` gives up early, and anything in
  between triggers a query refinement and another retrieval hop, up to
  `max_hops`.
- `retrieve_k` / `rerank_top_n` — how many candidates the fusion stage keeps
  before the (expensive) reranker, and how many survive reranking into the
  generation prompt.

## Scaling to millions of documents

The retrieval interfaces (`DenseIndex`, `BM25Index`) are intentionally
narrow (`build(chunks)` / `search(query, k)`), so the in-memory
implementations here can be swapped for production stores without touching
anything else in the pipeline:

- Swap `DenseIndex` for a real vector store (LanceDB, Qdrant, pgvector,
  etc.) backed by a neural embedding model — this is the change that
  matters most for retrieval quality, since TF-IDF/SVD has no notion of
  paraphrase or semantics.
- Swap `BM25Index` for `bm25s` or an Elasticsearch/OpenSearch BM25 index for
  large corpora.
- Everything downstream (fusion, reranking, routing, generation,
  verification, abstention) is retrieval-backend-agnostic and needs no
  changes.

## Architecture

```
question
   │
   ▼
 route (no_retrieval / single_hop / multi_hop)
   │
   ├─ no_retrieval ──────────────────────────► abstain
   │
   ▼
 retrieve (dense + BM25, fused by RRF) ──► rerank (top-k)
   │
   ▼
 grade evidence sufficiency
   │
   ├─ weak, hops < max ─► refine query (decompose) ─► retrieve (loop)
   ├─ weak, hops = max ─────────────────────────────► abstain
   │
   ▼ strong
 generate (cited, or abstain token)
   │
   ▼
 verify: split into atomic claims → score each vs. cited context
   │
   ├─ any claim unsupported ─► abstain
   │
   ▼
 answered (with citations + min claim support score)
```
