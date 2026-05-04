# Agentic RAG — Digestate Regulatory Compliance

[![Deploy](https://github.com/chloe-mp/agentic-rag-digestats/actions/workflows/deploy.yml/badge.svg)](https://github.com/chloe-mp/agentic-rag-digestats/actions/workflows/deploy.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Cloud Run](https://img.shields.io/badge/deployed%20on-Cloud%20Run-4285F4.svg?logo=googlecloud&logoColor=white)](https://rag-digestats-gnnkfawr7q-ew.a.run.app)

LangGraph-based agentic RAG system answering questions about French and
European regulations on agricultural digestates — built on official
regulatory texts (CDC DIG, NF U 44-051, NF U 42-001, EU Regulation 2019/1009).

> **Live demo** → [rag-digestats-gnnkfawr7q-ew.a.run.app](https://rag-digestats-gnnkfawr7q-ew.a.run.app)
> *(password-protected, ask for access)*

> Hybrid retrieval (BGE-M3 + BM25) · LangGraph orchestration ·
> Gemini 2.5 Flash · Qdrant · Cloud Run · Langfuse tracing · DeepEval / RAGAS

---

## Why this project

The French and European regulatory framework for agricultural digestates
(byproducts of methanization used as organic fertilizer) is dense and
fragmented across multiple texts: the *Cahier des charges DIG* (CDC DIG),
the NF U 42-001 and NF U 44-051 standards, EU Regulation 2019/1009 and its
implementing acts.

Producers and agricultural operators routinely need to verify compliance
points across these texts: minimum residence time in mesophilic vs
thermophilic digestion, allowed input proportions, sanitary delays, labeling
requirements, etc.

This project is an **agentic RAG** system that answers such questions in
natural language, citing the exact regulatory passages that ground each answer.

It is built as a real engineering exercise: production deployment on
Cloud Run, hybrid retrieval, formal evaluation with RAGAS, observability
through Langfuse.

---

## Headline numbers

Evaluated on a custom benchmark of **50 regulatory questions** (factual,
multi-hop, edge cases) with [RAGAS](https://github.com/explodinggradients/ragas):

| Metric              | Mean   | Median |
|---------------------|--------|--------|
| Faithfulness        | **0.905** | 1.000 |
| Answer Relevancy    | **0.901** | 0.936 |
| Context Recall      | **0.907** | 1.000 |
| Context Precision   | 0.696  | 0.747 |

The high **faithfulness** (0.905) means the system rarely hallucinates —
when it answers, it stays grounded in retrieved regulatory text. The
median of 1.0 on faithfulness and context recall shows that most queries
are answered with full provenance.

**Context precision (0.696) is the weakest metric** and is honestly
flagged: hybrid retrieval (BGE-M3 + BM25) brings high recall but also
some noise. Reranking is the natural next step (see *Limitations*).

---

## Architecture

The system uses **LangGraph** to orchestrate a small agentic graph:

                ┌──────────────────────┐
                │   User question      │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │  Question rewrite    │
                │  (clarify intent)    │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │  Hybrid retrieval    │
                │  Qdrant BGE-M3 + BM25│
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │  Document grader     │
                │  (relevance check)   │
                └──────────┬───────────┘
                      ┌────┴────┐
                   relevant?    no
                      │         │
                      ▼         ▼
              ┌──────────┐  ┌──────────┐
              │ Generate │  │ Web tool │
              │ answer   │  │ fallback │
              └────┬─────┘  └────┬─────┘
                   └──────┬──────┘
                          ▼
                ┌──────────────────────┐
                │  Final answer +      │
                │  cited passages      │
                └──────────────────────┘

Indexed corpus: **10 official PDFs** (1406 chunks) covering EU regulations,
French *arrêtés* and norms governing digestate methanization and labeling.

### Why this stack

- **Hybrid retrieval (BGE-M3 + BM25)** — pure dense retrieval misses exact
  regulatory references (article numbers, decree numbers). BM25 catches
  these; BGE-M3 catches semantic paraphrases. RRF fusion combines both.
- **LangGraph over plain LangChain** — explicit state machine, easier to
  reason about and trace. Each node logs to Langfuse with its inputs/
  outputs/timings.
- **Gemini 2.5 Flash** — fast, low-cost, sufficient for grounded extraction.
  The bottleneck on this task is retrieval, not generation.
- **Qdrant** — embedded mode (no separate server), pre-built index shipped
  with the Docker image for cold-start parity in production.
- **Pre-built index in Docker image** — eliminates re-indexing at startup
  (~3 minutes saved on cold start), at the cost of a heavier image.
- **Cloud Run + Secret Manager** — autoscaling to zero, secrets injected at
  runtime, no credentials in code.

---

## Tech stack

| Layer              | Tools                                        |
|--------------------|----------------------------------------------|
| Orchestration      | LangGraph                                    |
| LLM                | Gemini 2.5 Flash                             |
| Embeddings         | BGE-M3 (multilingual)                        |
| Vector store       | Qdrant (embedded mode, pre-built index)      |
| Lexical retrieval  | BM25                                         |
| Web fallback       | Tavily                                       |
| Observability      | Langfuse                                     |
| Evaluation         | RAGAS, DeepEval                              |
| Frontend           | Streamlit                                    |
| Deployment         | Google Cloud Run + Artifact Registry         |
| CI / CD            | GitHub Actions (lint + tests + deploy)       |

---

## Running locally

```bash
git clone https://github.com/chloe-mp/agentic-rag-digestats
cd agentic-rag-digestats

# Create a .env from the template (you'll need GOOGLE_API_KEY,
# TAVILY_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, APP_PASSWORD)
cp .env.example .env  # then fill in the values

# Build & run
docker build -t agentic-rag-digestats .
docker run -p 8080:8080 --env-file .env agentic-rag-digestats

# Open http://localhost:8080
```

---

## Evaluation

The `evaluation/` folder contains a curated benchmark of 50 regulatory
questions split across:

- **Factual single-hop** (e.g. *"Quelle est la plage de pH requise pour
  un digestat conforme au CDC DIG ?"*)
- **Multi-hop** (combining info from multiple regulatory texts)
- **Edge cases** (rare or ambiguous regulatory configurations)

Reproduce the eval:

```bash
python evaluation/rag_eval.py    # run the RAG over all questions
python evaluation/rag_eval.py --metrics  # compute RAGAS scores
```

Results are written to `evaluation/ragas_results.json`.

---

## Limitations & next steps

### Retrieval
- **Context precision (0.696) is below where it should be.** 
- **No query routing.** Some questions are pure factual lookups (no
  reasoning needed) and could be answered with simple BM25; others require
  multi-step reasoning. Currently every question goes through the same
  graph. Routing on question type could lower latency and cost.

### Modeling
- **Single LLM (Gemini 2.5 Flash) for all nodes.** A larger model on the
  generation node and a smaller / cheaper one on document grading would
  optimize the cost/quality tradeoff.
- **No RAG fine-tuning.** The system relies on prompt engineering only.
  Fine-tuning Gemini or a smaller open model on the regulatory domain is
  a clear next experiment.

### Operations
- **No A/B framework.** Comparing retrieval variants today requires
  manual eval runs; a lightweight A/B layer would let multiple variants
  run in parallel and accumulate stats.
- **Cold start latency.** The pre-built Qdrant index keeps it acceptable
  (~10s) but a warm pool would be needed for low-latency usage.

---

## Acknowledgments

Regulatory texts are public domain (Légifrance, EUR-Lex). Built as a
personal project alongside ongoing research on multi-agent systems and
agentic AI ([ECAI 2025](https://hal.science/hal-05350815v1)).

---

*Made by [Chloé Petridis](https://github.com/chloe-mp) — happy to discuss
hybrid retrieval, RAG evaluation methodology, or LangGraph design tradeoffs.*
