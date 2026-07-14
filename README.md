# 🚀 RAG Pipeline with Near-Zero Hallucination

A production-inspired **Retrieval-Augmented Generation (RAG) pipeline** designed to minimize hallucinations using a *retrieve → constrain → verify → abstain* architecture.

This project is a lightweight, CPU-friendly implementation inspired by large-scale RAG systems, adapted to run locally without GPUs while preserving the core design principles.

---

## 🔥 Key Features

- ✅ **Hybrid Retrieval**
  - Combines dense retrieval + BM25
  - Uses Reciprocal Rank Fusion (RRF)

- ✅ **LLM-based Reranking**
  - Improves relevance of retrieved documents

- ✅ **Grounded Answer Generation**
  - Every response is based on retrieved context
  - Includes citations for traceability

- ✅ **Faithfulness Verification**
  - Splits answers into atomic claims
  - Verifies each claim against source documents

- ✅ **Abstention Mechanism**
  - Avoids hallucination
  - Returns *“Not enough supporting evidence”* when needed

- ✅ **Self-Correcting Agent Loop**
  - Refines queries when retrieval is weak
  - Iterative reasoning until confident answer

---

## 🧠 Architecture

```
Question
   │
   ▼
Routing (single-hop / multi-hop / no-retrieval)
   │
   ▼
Hybrid Retrieval (Dense + BM25)
   │
   ▼
Reranking
   │
   ▼
Evidence Evaluation
   │
   ├── Weak → Query Refinement → Retrieve (loop)
   ├── Weak (max hops) → Abstain
   ▼
Strong
   ▼
Answer Generation (with citations)
   ▼
Claim Verification
   │
   ├── Unsupported → Abstain
   ▼
Final Answer
```

---

## 📁 Project Structure

```
RAG-Pipelines/
│
├── rag_pipeline.py     # Core RAG logic
├── run_pdfs.py         # Run pipeline on PDFs
├── demo.py             # Demo example
├── pdfs/               # Input documents
├── README.md
```

---

## ⚡ Quick Start (Offline Mode)

Run without API key:

```bash
pip install scikit-learn rank_bm25 numpy
python demo.py
```

👉 Uses a **mock LLM** for testing pipeline logic

---

## 🤖 Run with Real LLM

```bash
pip install openai
python run_pdfs.py --backend openai --api-key YOUR_API_KEY
```

---

## 📄 Using Your Own Documents

```python
from rag_pipeline import Passage, build_pipeline, OpenAIClient

passages = [
    Passage(id="doc1", title="Title 1", text="Content..."),
    Passage(id="doc2", title="Title 2", text="Content...")
]

llm = OpenAIClient(api_key="YOUR_KEY", model="gpt-4o-mini")
agent = build_pipeline(passages, llm)

result = agent.answer("Your question here")

print(result["final"].answer)
print(result["final"].citations)
```

---

## 🎯 Key Concepts

- Retrieval-Augmented Generation (RAG)
- Hybrid Search (Semantic + Keyword)
- Reciprocal Rank Fusion (RRF)
- LLM-based Reranking
- Grounded Answer Generation
- Claim-level Verification
- Hallucination Reduction
- Agentic Query Refinement

---

## 📊 Why This Project Matters

Most basic RAG systems:
- ❌ Hallucinate answers  
- ❌ Don’t verify outputs  
- ❌ Lack grounding  

This system:
- ✔ Verifies every claim  
- ✔ Uses citations  
- ✔ Abstains when unsure  
- ✔ Improves reliability  

---

## 🚀 Future Improvements

- Integrate vector DB (FAISS / Pinecone / Qdrant)
- Add real embedding models
- Build UI (Streamlit / React)
- Deploy as API

---

## 🧑‍💻 Author

**Santhosh Kumar Kathiresan**  
MS Computer Science — Illinois Institute of Technology  

---

## ⭐ Support

If you found this useful, give this repo a ⭐
