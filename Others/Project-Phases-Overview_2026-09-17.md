# Project Phases Overview

## Phase 1: Stabilize the Current RAG

Implement:

1.  Basic tests.
2.  Document deletion.
3.  Duplicate prevention.
4.  Re-indexing.
5.  Metadata filters.
6.  Similarity score threshold.
7.  OCR for scanned PDFs.
8.  Better table-aware CSV/Excel handling.

Your current **streamlit\_app.py** should remain mostly unchanged. First, improve the backend and verify everything through the existing UI.

## Phase 2: Add Evaluation

Create 30–50 questions based on one or two sample documents. Start with only:

-   Retrieval hit rate.
-   Expected source chunk found or not.
-   Retrieval latency.

Then add:

-   Context precision.
-   Context recall.
-   Answer faithfulness.
-   Answer relevance.
-   Token usage.
-   Cost.

The first goal is to run one command such as:

> python evaluation/run\_evaluation.py

and receive:

> Retrieval hit rate: 86%  
> Average latency: 1.8 seconds

## Phase 3: Improve Retrieval

Use your evaluation results to compare:

1.  Current vector search.
2.  Different chunk sizes.
3.  Metadata filtering.
4.  Hybrid vector plus keyword search.
5.  Reranking.
6.  Query rewriting.
7.  MMR retrieval.

Implement one technique at a time and record the result in the **README**.

## Phase 4: Convert It into a Production Backend

Add these gradually:

1.  PostgreSQL for users and document records.
2.  Object storage for uploaded files.
3.  Authentication.
4.  User-specific document access.
5.  Rate limiting.
6.  Structured logging.
7.  Health checks.
8.  Background processing with Redis and Celery/Dramatiq.
9.  Docker deployment.
10.  CI/CD tests.

Use Django for users, administration, permissions, and business data. Keep FastAPI for the AI and retrieval service if that separation helps you learn.

## Phase 5: Learn Agents and Tools

Create tools for:

-   Searching documents.
-   Querying PostgreSQL.
-   Calling one external API.
-   Returning structured data.
-   Asking for approval before a write operation.

Then add conversation state and learn LangGraph. After that, learn MCP for connecting agents to external tools and data sources.

## Phase 6: Learn Core ML and LLM Concepts

Study and implement small examples for:

-   Tokenization.
-   Embeddings.
-   Cosine similarity.
-   Attention.
-   Transformer architecture.
-   Fine-tuning.
-   LoRA and QLoRA.
-   Quantization.
-   vLLM inference.
-   GPU fundamentals.

You do not need to train a large model. Focus on understanding how models work and how to serve or adapt them.

## Phase 7: Build One Strong Second Project

Start with one project, not four. The best choice for your background is:

> Django Support Ticket Copilot

It can include:

-   Django users and permissions.
-   Support ticket management.
-   RAG over company documentation.
-   Ticket classification.
-   Suggested replies.
-   Structured outputs.
-   PostgreSQL tools.
-   Human approval.
-   Audit logs.
-   Evaluation.