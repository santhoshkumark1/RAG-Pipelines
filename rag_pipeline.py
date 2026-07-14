"""
rag_pipeline.py
================
A self-contained implementation of a "retrieve, constrain, verify, abstain"
RAG pipeline: hybrid (dense + BM25) retrieval with reciprocal-rank fusion,
citation-constrained generation, atomic-claim faithfulness verification, and
a calibrated abstention policy, wired into a small self-correcting agent loop.

This is a portable, CPU-only adaptation. It does not require a GPU, a local
vLLM server, or a Hugging Face download:

  * "Dense" retrieval uses TF-IDF + Truncated SVD (scikit-learn) instead of a
    neural embedder, so it works fully offline.
  * "Sparse" retrieval uses BM25 (rank_bm25).
  * The LLM used for routing / generation / claim-checking is behind a small
    LLMClient interface. A MockLLMClient (extractive, offline, no API key)
    is used by default so the whole thing runs out of the box. Swap in
    OpenAIClient or AnthropicClient (stubs included) to use a real model.

Usage:
    python demo.py
"""

from __future__ import annotations

import re
import json
import uuid
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize
import numpy as np
from rank_bm25 import BM25Okapi


# --------------------------------------------------------------------------
# 0. LLM client interface (pluggable backend)
# --------------------------------------------------------------------------

class LLMClient(Protocol):
    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> str:
        ...


class MockLLMClient:
    """
    Fully offline, no-API-key fallback so the pipeline is runnable out of the
    box. It fakes the router/generator/judge with cheap heuristics instead of
    a real LLM. Swap this out (see OpenAIClient / AnthropicClient below) for
    real answer quality -- this class exists purely so demo.py works with
    zero setup.
    """

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> str:
        if "precise query classifier" in system.lower():
            return self._route(user)
        if "strict faithfulness grader" in system.lower():
            return self._grade_claim(user)
        if "grade retrieval sufficiency" in system.lower():
            return self._grade_retrieval(user)
        if "extract atomic factual claims" in system.lower():
            return self._extract_claims(user)
        if "precise query classifier" not in system.lower() and "context passages" in user.lower():
            return self._answer(user)
        return ""

    def _route(self, user: str) -> str:
        m = re.search(r"Question:\s*(.*)", user, re.S)
        q = (m.group(1) if m else user).strip().lower()
        if any(w in q for w in ["hi", "hello", "best programming language", "opinion", "your favorite"]):
            return "no_retrieval"
        if len(re.findall(r"\band\b|\bboth\b|,", q)) >= 1 and "?" in q:
            return "multi_hop"
        return "single_hop"

    def _grade_retrieval(self, user: str) -> str:
        # crude: more overlapping content words between question and context -> higher grade
        m = re.search(r"Question: (.*)", user)
        q = m.group(1) if m else ""
        qwords = set(re.findall(r"[a-zA-Z]{4,}", q.lower()))
        ctx = user.lower()
        hits = sum(1 for w in qwords if w in ctx)
        grade = min(1.0, hits / max(1, len(qwords)))
        return f"{grade:.2f}"

    def _extract_claims(self, user: str) -> str:
        m = re.search(r"ANSWER:\n(.*)", user, re.S)
        text = m.group(1) if m else user
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return "\n".join(s.strip() for s in sents if len(s.strip()) > 3)

    def _grade_claim(self, user: str) -> str:
        m = re.search(r"CONTEXT:\n(.*)\n\nCLAIM: (.*)", user, re.S)
        if not m:
            return "0.0"
        context, claim = m.group(1).lower(), m.group(2).lower()
        cwords = set(re.findall(r"[a-zA-Z]{4,}", claim))
        if not cwords:
            return "0.0"
        hits = sum(1 for w in cwords if w in context)
        score = hits / len(cwords)
        return f"{min(1.0, score):.2f}"

    def _answer(self, user: str) -> str:
        # naive extractive "answer": pick the passage sentence with the most
        # word overlap with the question, cite that passage's id.
        m = re.search(r"Context passages:\n(.*)\n\nQuestion: (.*)\n\nAnswer:", user, re.S)
        if not m:
            return "INSUFFICIENT_EVIDENCE"
        ctx_block, question = m.group(1), m.group(2)
        blocks = re.split(r"\n(?=\[[a-f0-9]{8,}\])", ctx_block.strip())
        qwords = set(re.findall(r"[a-zA-Z]{4,}", question.lower()))
        best_sent, best_id, best_score = None, None, 0
        for b in blocks:
            idm = re.match(r"\[([a-f0-9]{8,})\]\s*(.*)", b, re.S)
            if not idm:
                continue
            cid, text = idm.group(1), idm.group(2)
            for sent in re.split(r"(?<=[.!?])\s+", text):
                swords = set(re.findall(r"[a-zA-Z]{4,}", sent.lower()))
                score = len(swords & qwords)
                if score > best_score:
                    best_score, best_sent, best_id = score, sent.strip(), cid
        if not best_sent or best_score == 0:
            return "INSUFFICIENT_EVIDENCE"
        return f"{best_sent} [{best_id}]"


class OpenAIClient:
    """Real LLM backend for any OpenAI-compatible endpoint (OpenAI, vLLM, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1"):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return r.choices[0].message.content


class AnthropicClient:
    """Real LLM backend using the Anthropic API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-5"):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> str:
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if b.type == "text")


# --------------------------------------------------------------------------
# 1. Data model
# --------------------------------------------------------------------------

@dataclass
class Passage:
    id: str
    title: str
    text: str


@dataclass
class Chunk:
    id: str
    passage_id: str
    title: str
    text: str
    contextual_text: str = ""

    @property
    def indexed_text(self) -> str:
        return self.contextual_text or self.text


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    source: str  # "dense" | "sparse" | "hybrid" | "reranked"

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.indexed_text


@dataclass
class ClaimVerdict:
    claim: str
    score: float
    supported: bool


@dataclass
class GateResult:
    passed: bool
    verdicts: list[ClaimVerdict]
    min_support: float


@dataclass
class CitedAnswer:
    text: str
    cited_ids: list[str]
    abstained: bool
    raw: str = ""


@dataclass
class FinalAnswer:
    status: str  # "answered" | "abstained"
    answer: str
    citations: list[str]
    min_support: float
    reason: str


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def split_sentences(text: str) -> list[str]:
    # lightweight sentence splitter; good enough without a heavy NLP dependency
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------
# 3. Chunking + contextualization
# --------------------------------------------------------------------------

class StructureAwareChunker:
    """Packs whole sentences up to a word-count budget, respecting sentence
    boundaries so an answer-bearing sentence is never silently truncated."""

    def __init__(self, target_words: int = 180, overlap_sentences: int = 1):
        self.target_words = target_words
        self.overlap_sentences = overlap_sentences

    def chunk(self, passage: Passage) -> list[Chunk]:
        sents = split_sentences(passage.text) or [passage.text]
        chunks, cur, cur_words = [], [], 0
        for s in sents:
            w = len(s.split())
            if cur and cur_words + w > self.target_words:
                chunks.append(self._make(passage, cur))
                cur = cur[-self.overlap_sentences:] if self.overlap_sentences else []
                cur_words = sum(len(x.split()) for x in cur)
            cur.append(s)
            cur_words += w
        if cur:
            chunks.append(self._make(passage, cur))
        return chunks

    @staticmethod
    def _make(passage: Passage, sents: list[str]) -> Chunk:
        cid = uuid.uuid4().hex[:12]
        return Chunk(id=cid, passage_id=passage.id, title=passage.title, text=" ".join(sents))


class Contextualizer:
    """Prepends a one-line situating sentence to each chunk (contextual
    retrieval) so a chunk that mentions 'it grew 3 percent' is still
    searchable on its own. Uses the LLM client; skips silently on failure."""

    PROMPT = (
        "Here is a document titled '{title}':\n<document>\n{doc}\n</document>\n\n"
        "Here is a chunk from it:\n<chunk>\n{chunk}\n</chunk>\n\n"
        "Give a short, single-sentence context (<=25 words) that situates this "
        "chunk within the document so it can be retrieved on its own. "
        "Answer with the sentence only."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def contextualize(self, chunks: list[Chunk], doc_lookup: dict[str, str], workers: int = 16) -> list[Chunk]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _one(c: Chunk) -> None:
            try:
                ctx = self.llm.chat(
                    "You write concise retrieval context.",
                    self.PROMPT.format(title=c.title, doc=doc_lookup.get(c.passage_id, c.text)[:4000], chunk=c.text),
                    max_tokens=64,
                ).strip()
            except Exception:
                ctx = ""
            c.contextual_text = f"{ctx}\n{c.text}" if ctx else c.text

        total = len(chunks)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_one, c): c for c in chunks}
            for _ in as_completed(futures):
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  [contextualize] {done}/{total} chunks")
        return chunks


# --------------------------------------------------------------------------
# 4. Hybrid index: TF-IDF/SVD "dense" + BM25 sparse
# --------------------------------------------------------------------------

class DenseIndex:
    """TF-IDF followed by truncated SVD as an offline stand-in for a neural
    embedder. Captures topical/paraphrase similarity without any model
    download; swap for a real sentence-embedding model in production."""

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.matrix: Optional[np.ndarray] = None
        self.ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        texts = [c.indexed_text for c in chunks]
        self.ids = [c.id for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, max(2, min(tfidf.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        reduced = self.svd.fit_transform(tfidf)
        self.matrix = sk_normalize(reduced)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        qv = self.vectorizer.transform([query])
        qr = sk_normalize(self.svd.transform(qv))
        sims = (self.matrix @ qr[0])
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top]


class BM25Index:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.ids = [c.id for c in chunks]
        tokenized = [re.findall(r"[a-zA-Z0-9]+", c.indexed_text.lower()) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = re.findall(r"[a-zA-Z0-9]+", query.lower())
        scores = self.bm25.get_scores(q)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: combine ranked lists using rank only, since
    BM25 scores and cosine similarities live on incomparable scales."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, dense: DenseIndex, sparse: BM25Index, chunk_lookup: dict[str, Chunk], rrf_k: int = 60):
        self.dense, self.sparse, self.chunk_lookup, self.rrf_k = dense, sparse, chunk_lookup, rrf_k

    def retrieve(self, query: str, k: int = 30) -> list[RetrievedChunk]:
        dense_hits = self.dense.search(query, k)
        sparse_hits = self.sparse.search(query, k)
        fused = rrf_fuse([[i for i, _ in dense_hits], [i for i, _ in sparse_hits]], self.rrf_k)[:k]
        out = []
        for cid, score in fused:
            c = self.chunk_lookup.get(cid)
            if c:
                out.append(RetrievedChunk(chunk=c, score=score, source="hybrid"))
        return out


class LLMReranker:
    """Cross-encoder-style reranking via LLM relevance judgment. Slower and
    more accurate than the bi-encoder/BM25 fusion stage; run only over the
    fused top-k candidates, never over the whole corpus."""

    PROMPT = (
        "On a scale of 0 to 10, how directly does this passage answer the "
        "question? Reply with only a number.\n\nQuestion: {q}\nPassage: {p}"
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int = 8) -> list[RetrievedChunk]:
        scored = []
        for c in candidates:
            try:
                out = self.llm.chat("You score passage relevance.",
                                     self.PROMPT.format(q=query, p=c.text[:800]), max_tokens=4)
                m = re.search(r"\d+(\.\d+)?", out)
                score = float(m.group()) if m else c.score
            except Exception:
                score = c.score
            scored.append(RetrievedChunk(chunk=c.chunk, score=score, source="reranked"))
        return sorted(scored, key=lambda x: -x.score)[:top_n]


# --------------------------------------------------------------------------
# 5. Routing
# --------------------------------------------------------------------------

class QueryRouter:
    LABELS = {"no_retrieval", "single_hop", "multi_hop"}
    PROMPT = (
        "Classify the question into exactly one label:\n"
        "- no_retrieval: greetings/opinions or questions no document corpus could answer\n"
        "- single_hop: answerable by finding one fact\n"
        "- multi_hop: needs combining facts from multiple documents\n"
        "Question: {q}\nReply with only the label."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def route(self, query: str) -> str:
        out = self.llm.chat("You are a precise query classifier.",
                             self.PROMPT.format(q=query), max_tokens=8).strip().lower()
        for lbl in self.LABELS:
            if lbl in out:
                return lbl
        return "single_hop"


class Decomposer:
    PROMPT = (
        "Break this multi-hop question into 2-3 ordered, self-contained "
        "sub-questions, one per line, no numbering. If it is already simple, "
        "return it unchanged.\nQuestion: {q}"
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def decompose(self, query: str) -> list[str]:
        out = self.llm.chat("You decompose multi-hop questions.", self.PROMPT.format(q=query), max_tokens=120)
        subs = [s.strip(" -\t") for s in out.splitlines() if s.strip()]
        return subs or [query]


# --------------------------------------------------------------------------
# 6. Cited generation
# --------------------------------------------------------------------------

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"
_CITE_RE = re.compile(r"\[([a-f0-9]{8,})\]")

GENERATION_SYSTEM_PROMPT = (
    "You answer strictly from the numbered context passages. Rules:\n"
    "1. Use ONLY facts in the passages, never outside knowledge.\n"
    f"2. If the passages do not contain the answer, reply with exactly: {ABSTAIN_TOKEN}\n"
    "3. Every sentence MUST end with a citation to the passage id(s) it uses, like [abc123def456].\n"
    "4. Be concise and factual."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)


def parse_citations(text: str, valid_ids: set[str]) -> tuple[list[str], str]:
    found = _CITE_RE.findall(text)
    valid = [c for c in dict.fromkeys(found) if c in valid_ids]
    invalid = [c for c in dict.fromkeys(found) if c not in valid_ids]
    cleaned = text
    for bad in invalid:
        cleaned = cleaned.replace(f"[{bad}]", "")
    return valid, cleaned


class CitedGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> CitedAnswer:
        user = f"Context passages:\n{format_context(chunks)}\n\nQuestion: {question}\n\nAnswer:"
        raw = self.llm.chat(GENERATION_SYSTEM_PROMPT, user, max_tokens=400).strip()
        if ABSTAIN_TOKEN in raw:
            return CitedAnswer(text="", cited_ids=[], abstained=True, raw=raw)
        valid_ids = {c.id for c in chunks}
        cited, cleaned = parse_citations(raw, valid_ids)
        return CitedAnswer(text=cleaned.strip(), cited_ids=cited, abstained=False, raw=raw)


# --------------------------------------------------------------------------
# 7. Verification gate (claim extraction + faithfulness judge)
# --------------------------------------------------------------------------

CLAIM_DECOMP_PROMPT = (
    "Split the ANSWER below into short, atomic, independently checkable "
    "factual claims, one per line, with no extra commentary.\n\nANSWER:\n{a}"
)

JUDGE_PROMPT = (
    "You are a strict fact-checker. Decide whether the CONTEXT supports the CLAIM.\n\n"
    "CONTEXT:\n{context}\n\nCLAIM: {claim}\n\n"
    "Output ONLY a number: 1.0 if the context clearly states or entails the claim, "
    "0.0 if it contradicts or does not mention it, or a value in between."
)


class ClaimExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, answer: str) -> list[str]:
        clean = _CITE_RE.sub("", answer).strip()
        out = self.llm.chat("You extract atomic factual claims.", CLAIM_DECOMP_PROMPT.format(a=clean), max_tokens=300)
        claims = [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip(" -\t") for ln in out.splitlines() if ln.strip()]
        return [c for c in claims if len(c) > 3]


class FaithfulnessJudge:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def score(self, claim: str, context: str) -> float:
        out = self.llm.chat("You are a strict faithfulness grader.",
                             JUDGE_PROMPT.format(context=context[:6000], claim=claim), max_tokens=8)
        m = re.search(r"[01](?:\.\d+)?", out)
        return min(1.0, float(m.group())) if m else 0.0


class VerificationGate:
    """Splits an answer into atomic claims and checks each against its cited
    context; a single unsupported claim fails the whole answer, because an
    answer is only as trustworthy as its weakest sentence."""

    def __init__(self, extractor: ClaimExtractor, judge: FaithfulnessJudge, tau: float = 0.3):
        self.extractor, self.judge, self.tau = extractor, judge, tau

    def check(self, cited: CitedAnswer, chunks: list[RetrievedChunk]) -> GateResult:
        claims = self.extractor.extract(cited.text)
        used = [c for c in chunks if c.id in set(cited.cited_ids)] or chunks
        context = "\n\n".join(c.text for c in used)
        verdicts = []
        for cl in claims:
            s = self.judge.score(cl, context)
            verdicts.append(ClaimVerdict(cl, s, s >= self.tau))
        min_support = min((v.score for v in verdicts), default=0.0)
        passed = len(verdicts) > 0 and all(v.supported for v in verdicts)
        return GateResult(passed, verdicts, min_support)


# --------------------------------------------------------------------------
# 8. Abstention policy
# --------------------------------------------------------------------------

class AbstentionPolicy:
    def __init__(self, tau: float = 0.3):
        self.tau = tau

    def decide(self, route: str, cited: Optional[CitedAnswer], gate: Optional[GateResult]) -> FinalAnswer:
        if route == "no_retrieval":
            return self._abstain("routed_no_retrieval", gate)
        if cited is None or cited.abstained:
            return self._abstain("model_abstained", gate)
        if gate is None or not gate.passed or gate.min_support < self.tau:
            return self._abstain("unsupported_claims", gate)
        return FinalAnswer("answered", cited.text, cited.cited_ids, gate.min_support, "verified")

    @staticmethod
    def _abstain(reason: str, gate: Optional[GateResult]) -> FinalAnswer:
        return FinalAnswer("abstained",
                            "I do not have enough supporting evidence in the available sources to answer this confidently.",
                            [], gate.min_support if gate else 0.0, reason)


# --------------------------------------------------------------------------
# 9. Self-correcting agent (CRAG-style loop, plain Python)
# --------------------------------------------------------------------------

class RAGAgent:
    def __init__(self, retriever: HybridRetriever, router: QueryRouter, decomposer: Decomposer,
                 reranker: LLMReranker, generator: CitedGenerator, gate: VerificationGate,
                 policy: AbstentionPolicy, retrieve_k: int = 30, rerank_top_n: int = 8,
                 crag_ok: float = 0.7, crag_bad: float = 0.4, max_hops: int = 3,
                 grader_llm: Optional[LLMClient] = None):
        self.retriever, self.router, self.decomposer = retriever, router, decomposer
        self.reranker, self.generator, self.gate, self.policy = reranker, generator, gate, policy
        self.retrieve_k, self.rerank_top_n = retrieve_k, rerank_top_n
        self.crag_ok, self.crag_bad, self.max_hops = crag_ok, crag_bad, max_hops
        self.grader_llm = grader_llm

    def _grade_evidence(self, query: str, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        ctx = "\n".join(f"- {c.text[:200]}" for c in chunks[:8])
        out = self.grader_llm.chat("You grade retrieval sufficiency.",
                                    f"Question: {query}\nContext:\n{ctx}\nHow sufficient (0-1)?", max_tokens=8)
        m = re.search(r"[01](?:\.\d+)?", out)
        return float(m.group()) if m else 0.5

    def answer(self, question: str) -> dict:
        route = self.router.route(question)
        if route == "no_retrieval":
            final = self.policy.decide(route, None, None)
            return {"route": route, "hops": 0, "final": final}

        query, hops = question, 0
        chunks: list[RetrievedChunk] = []
        while True:
            fused = self.retriever.retrieve(query, k=self.retrieve_k)
            chunks = self.reranker.rerank(question, fused, top_n=self.rerank_top_n) if fused else []
            grade = self._grade_evidence(question, chunks)
            if grade >= self.crag_ok or hops >= self.max_hops:
                break
            if grade < self.crag_bad and hops >= 1:
                break
            hops += 1
            subs = self.decomposer.decompose(question)
            query = " ".join(subs)

        if not chunks or grade < self.crag_bad:
            final = self.policy.decide(route, None, None)
            return {"route": route, "hops": hops, "grade": grade, "final": final, "chunks": chunks}

        cited = self.generator.generate(question, chunks)
        gate_result = self.gate.check(cited, chunks) if not cited.abstained else None
        final = self.policy.decide(route, cited, gate_result)
        return {"route": route, "hops": hops, "grade": grade, "cited": cited, "gate": gate_result,
                "final": final, "chunks": chunks}


# --------------------------------------------------------------------------
# 10. Pipeline builder
# --------------------------------------------------------------------------

def build_pipeline(passages: list[Passage], llm: LLMClient, contextualize: bool = True,
                    dense_components: int = 128, tau_claim: float = 0.3) -> RAGAgent:
    chunker = StructureAwareChunker()
    all_chunks: list[Chunk] = []
    for p in passages:
        all_chunks.extend(chunker.chunk(p))

    if contextualize:
        doc_lookup = {p.id: p.text for p in passages}
        Contextualizer(llm).contextualize(all_chunks, doc_lookup)

    chunk_lookup = {c.id: c for c in all_chunks}

    dense = DenseIndex(n_components=dense_components)
    dense.build(all_chunks)
    sparse = BM25Index()
    sparse.build(all_chunks)

    retriever = HybridRetriever(dense, sparse, chunk_lookup)
    router = QueryRouter(llm)
    decomposer = Decomposer(llm)
    reranker = LLMReranker(llm)
    generator = CitedGenerator(llm)
    gate = VerificationGate(ClaimExtractor(llm), FaithfulnessJudge(llm), tau=tau_claim)
    policy = AbstentionPolicy(tau=tau_claim)

    return RAGAgent(retriever, router, decomposer, reranker, generator, gate, policy, grader_llm=llm)