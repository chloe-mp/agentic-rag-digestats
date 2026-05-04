# Agentic RAG for Regulatory Compliance

> **From prototype to cost-optimized production: a hybrid agentic RAG system shipped for a French SMB, evolved over three architectural iterations.**

A retrieval-augmented generation system that answers regulatory compliance questions over a 600+ page corpus of EU and French agricultural regulations. Built for a French organic fertilizer trader whose sales team needed faster, more reliable answers when checking compliance before signing deals.

This repository documents the full journey: V1 (prototype that worked) → V2 (production-grade quality) → V3 (cloud-native and cost-optimized).

---

## Live demo

**App URL** — *available on request*
**Demo password** — *available on request via [LinkedIn DM](#connect)*

The app is gated by a password to protect the client's compliance corpus from public crawling. Reach out and I'll give you a one-time access link.

---

## Quick stack overview

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (5-agent state graph) |
| LLM | Gemini 2.5 Flash |
| Embeddings | BGE-M3 (dense + sparse, 8k context) |
| Reranker | bge-reranker-v2-m3 (cross-encoder) |
| Vector store | Qdrant Cloud (managed, free tier) |
| Hybrid retrieval | BM25 + dense + RRF + cross-encoder reranking |
| Web fallback | Tavily |
| API | FastAPI |
| Deployment | GCP Cloud Run (scale-to-zero) |
| Secrets | Google Secret Manager + dedicated runtime SA |
| CI/CD | GitHub Actions |
| Monitoring | LangFuse (per-request tracing) |
| Evaluation | RAGAS |

---

## Why this project is interesting

It's not a tutorial RAG. It's a **production system that survived two architectural iterations**, was evaluated end-to-end with industry-standard metrics, and was actively cost-optimized after going live.

Key design decisions you'll find documented below:

- **Why hybrid retrieval beats dense-only** for regulatory text full of acronyms and exact references
- **Why I chose BGE-M3** over OpenAI embeddings (and why context window > dimensionality for legal text)
- **Why Qdrant Cloud free tier** instead of self-hosted or Pinecone for this volume
- **Why the V2 architecture cost €87/month and how V3 fixes it** (full SKU breakdown, anti-pattern diagnosis, lessons learned)

---

## Architecture (V3 — current)

```
User question
     │
     ▼
[Reformulator]  ← rewrites the question to optimize retrieval
     │
     ▼
[Retriever + Grader]  ← BM25 + dense + RRF + cross-encoder reranking
     │
     ├── Score sufficient? ──Yes──▶ [Generator] ──▶ Structured answer
     │
     └── Score insufficient ──▶ [Web search (Tavily)] ──▶ [Generator] ──▶ Answer
```

Implementation in `assemble_graph.py`:

```python
workflow = StateGraph(RAGState)
workflow.add_node("reformulate", reformulate_question)
workflow.add_node("retrieve", retrieve_and_score)
workflow.add_node("web_search", web_search_fallback)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("reformulate")
workflow.add_edge("reformulate", "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    route_after_retrieval,  # web_search if score too low, else generate
    {"web_search": "web_search", "generate": "generate"},
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
```

---

## Retrieval pipeline

```
Reformulated query
       │
       ├──▶ BM25 (lexical)            → Top-k by lexical relevance
       │
       ├──▶ Dense retrieval (cosine)  → Top-k by semantic similarity
       │
       ▼
[RRF — Reciprocal Rank Fusion]        → Merged ranking
       │
       ▼
[Cross-encoder reranking]              → Fine ranking, threshold > 0.5
       │
       ▼
Top-10 final documents → passed to the LLM as context
```

**Why hybrid + reranking, not just dense?**

Regulatory text is full of exact references that semantic similarity gets wrong: *"arrêté du 9 avril 2018"*, *"CDC DIG"*, *"Règlement (CE) n° 1069/2009"*. BM25 catches these instantly. Dense catches paraphrases. RRF fuses the two rankings without having to normalize their scores (which live on incompatible scales). The cross-encoder then does fine-grained query×document attention on the ~50 candidates left, keeping the top 10 above a 0.5 threshold.

---

## Evaluation (RAGAS, V2/V3)

| Metric | Score | Reading |
|---|---|---|
| **Faithfulness** | 0.898 | Answers stay grounded in retrieved context |
| **Answer relevancy** | 0.908 | Answers actually address the question |
| **Context recall** | 0.893 | Retriever finds most of the necessary information |
| **Context precision** | 0.696 | Some irrelevant chunks pass the reranker — the next thing I'd improve |

**Honest note on context precision (0.696):** the reranker filters at threshold 0.5, which is sometimes too lenient. The fix isn't a single dial — it's a combination of (a) a stricter LLM-based grading node, (b) revisiting chunk size for dense regulatory passages, and (c) tuning the reranker threshold per query type. Listed as the highest-priority improvement in the next iteration.

The evaluator (RAGAS, using a separate Gemini instance) is **distinct from the generator** to avoid self-evaluation bias.

---

## Architecture journey: V1 → V2 → V3

This project went through three architectural iterations driven by very different concerns at each step. Each transition was a deliberate decision, not a tech-stack flip.

### V1 — Prototype that worked

**Goal:** prove the system could answer real regulatory questions for the sales team.

| Component | Choice |
|---|---|
| Vector store | ChromaDB |
| Embeddings | multilingual-e5-large |
| Chunking | 800 tokens / 300 overlap |
| Reranker | ms-marco-MiniLM (English-first) |
| Evaluation | Vertex AI Eval + DeepEval |
| Deployment | Cloud Run, container with everything embedded |

**The hidden problem:** `multilingual-e5-large` has a **512-token context window**. With 800-token chunks, the model was **silently truncating** ~30-40% of every chunk. No error, no warning — just lost information at indexing time. Retrieval still worked, but the system was operating on a degraded representation of the corpus.

### V2 — Production-grade

**Goal:** fix the truncation bug, harden retrieval quality, professionalize the codebase.

| Component | V1 | V2 | Why |
|---|---|---|---|
| Vector store | ChromaDB | **Qdrant** | Native dense + sparse storage, advanced metadata filtering, cleaner API for hybrid search |
| Embeddings | e5-large (512 tokens) | **BGE-M3** (8192 tokens) | Stops truncation; native dense + sparse from one model; excellent multilingual (FR-heavy corpus) |
| Chunking | 800 / 300 (37% overlap) | **1500 / 200** (13% overlap) | Bigger chunks keep regulatory articles intact; less overlap stops the reranker from being flooded with near-duplicates |
| Reranker | ms-marco-MiniLM | **bge-reranker-v2-m3** | Multilingual, same family as the embedder, higher recall on FR text |
| Evaluation | Vertex AI + DeepEval | **RAGAS** | Standardized RAG metrics (faithfulness, relevancy, context precision/recall), comparable across projects and to literature |
| Linting / tests | None | **Ruff + graph compile tests** | CI-enforced quality |

V2 was good, in production, monitored via LangFuse, and fully evaluated. **It also cost €86.72/month**, which was the trigger for V3.

### V3 — Cloud-native cost optimization

**Goal:** make the production deployment sustainable and architecturally clean.

This wasn't a model or retrieval change — every quality metric stayed identical. It was a **FinOps + cloud architecture refactor** triggered by a billing analysis.

**Diagnosis (April 2026 bill, by SKU — not by service):**

| SKU | €/month | Root cause |
|---|---|---|
| Cloud Run Min Instance Memory | 41.01 | `min-instances=1` (forced by 2-5min cold start) |
| Artifact Registry Storage | 13.15 | 207 GB of accumulated Docker images (115 versions) |
| Cloud Run Min Instance CPU | 10.26 | Same idle compute issue |
| Gemini API | 12.00 | Actual usage (acceptable) |
| TVA + misc | ~10.00 | Proportional |
| **Total** | **86.72** | |

**The anti-pattern:** I had embedded BGE-M3 (2.3 GB) + the Qdrant index (SQLite) + 10 PDFs **inside the Cloud Run container** itself. Cloud Run is designed for stateless, lightweight, scale-to-zero workloads. By stuffing state into the container, I'd made cold starts unbearable (2-5 min to load BGE-M3 + scan the index), which forced `min-instances=1` and meant **paying for idle compute 24/7**. About 60% of the bill was just keeping a warm instance for an app used ~65 times per day.

**Four architectural decisions:**

1. **Decouple data from code.** Index migrated to Qdrant Cloud (free tier, 1 GB cluster — 0.6% utilization for 1,406 chunks). Ingestion pipeline (`indexing_pipeline.py --reindex`) separated from runtime. The client can re-index new PDFs without redeploying code.

2. **Activate scale-to-zero.** `min-instances=0`. Cold start of 30-60s accepted for this volume. Possible *only because* the image dropped from 5-7 GB to ~1 GB after data was externalized.

3. **Properly secure secrets.** Migration to Google Secret Manager (instead of plain env vars in `service.yaml`). Dedicated runtime Service Account `agentic-rag-runtime@...` (least privilege) instead of the default compute SA. Lazy initialization of the Qdrant client (`if URL and KEY: connect; else: warn`) so CI tests run without exposing secrets to the pipeline.

4. **Optimize artifact lifecycle.** Cleanup policy on Artifact Registry (keep 5 most recent). Trimmed `.dockerignore`. Git repo cleaned of data artifacts. Consolidated GCP projects (Gemini API key moved from the auto-created `Default Gemini Project` into `rag-digestats` — single billing, unified IAM).

**Results:**

| Metric | V2 (April 2026) | V3 (June 2026, projected) |
|---|---|---|
| Monthly GCP bill | €86.72 | ~€10-15 |
| Docker image | 5-7 GB | ~1 GB |
| Artifact Registry storage | 207 GB | < 30 GB |
| CI/CD pipeline | 15-20 min | 3-5 min |
| Code/data coupling | Tight (rebuild = re-index) | Decoupled |
| Secrets surface | Compute SA + env vars | Dedicated SA + Secret Manager |

> **Note:** V3 was deployed on May 4, 2026. The €10-15/month figure is an estimate based on per-SKU analysis. Validated billing data will be added at the end of June 2026.

---

## What I'd do next

Honest list of things I'd improve before claiming this is "done":

- **Pin dependency versions** in `requirements.txt` (`langchain-google-genai` is currently unpinned — silent breaking changes are a real risk on rebuild).
- **Set up a GCP budget alert at €20/month** to catch cost regressions early — I would have spotted the V2 cost issue much earlier with this in place.
- **Improve PDF parsing** (move from `pdfplumber` to PyMuPDF or Docling for better table/structure preservation in regulatory PDFs).
- **Improve context precision (0.696 → target 0.85+):** stricter LLM grading node, revisit chunk size for dense passages, per-query-type reranker threshold.
- **Use Gemini 2.5 Flash Lite for routing/grading nodes** (10× cheaper than Flash) — only the final generation needs Flash quality.
- **Evaluate self-hosted Qdrant on a small VPS** as a cost comparison once volume grows; Qdrant Cloud Standard becomes worth the upgrade only beyond ~96€/month equivalent.
- **A/B framework for retrieval changes** — currently relying on RAGAS deltas only, which is fine for now but won't scale once traffic grows.

---

## Run locally

```bash
# Clone
git clone https://github.com/[your-handle]/[repo].git
cd [repo]

# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Required env vars
export GOOGLE_API_KEY=...
export TAVILY_API_KEY=...
export QDRANT_URL=...        # your Qdrant Cloud cluster
export QDRANT_API_KEY=...
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export APP_PASSWORD=...

# Index your PDFs (one-time)
python indexing_pipeline.py --reindex

# Run
uvicorn app:app --reload --port 8080
```

---

## Repository structure

```
.
├── app.py                   # FastAPI entry + LangFuse callback + auth
├── assemble_graph.py        # LangGraph state graph assembly
├── indexing_pipeline.py     # Hybrid RAG (BGE-M3 + BM25 + Qdrant)
├── document_grader.py       # Relevance scoring of retrieved docs
├── generate_answer.py       # Final answer generation (system prompt + context)
├── query_pipeline.py        # Routing: direct generation or tool path
├── rewrite_question.py      # Question reformulation for retrieval
├── web_tool.py              # Tavily fallback (web search)
├── models.py                # Pydantic state models
├── eval.py                  # RAGAS evaluation harness
├── service.yaml             # Cloud Run deployment config
├── cleanup-policy.json      # Artifact Registry retention policy
├── Dockerfile
├── requirements.txt
└── .github/workflows/deploy.yml
```

---

## <a id="connect"></a>Connect with me

I'm currently a junior freelance ML/cloud engineer based in France, looking for opportunities in production ML, FinOps, or cloud architecture roles.

- LinkedIn: **[https://linkedin.com/in/chloe-petridis/]**
- Email: **[chloe.petridis@gmail.com]**

Happy to share the demo password, walk through any technical decision, or discuss how this approach could apply to your domain.

---

*Built and shipped solo over Q1-Q2 2026. Documentation up-to-date as of May 2026.*

---

## Acknowledgments

Regulatory texts are public domain (Légifrance, EUR-Lex). Built as a
personal project alongside ongoing research on multi-agent systems and
agentic AI ([ECAI 2025](https://hal.science/hal-05350815v1)).

---

*Made by [Chloé Petridis](https://github.com/chloe-mp) — happy to discuss
hybrid retrieval, RAG evaluation methodology, or LangGraph design tradeoffs.*
