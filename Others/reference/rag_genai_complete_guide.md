# Generative AI and RAG: The Complete Engineering Guide

A from-scratch, practical guide to building retrieval-augmented generation systems.
It explains **why** each part exists, **how** the parts connect, the **best practice**
for each step, the **advanced techniques** that separate a demo from a production
system, and a **scenario troubleshooting** section for when things go wrong.

The running example is a legal-document assistant (statutes, judgments, contracts),
but every pattern is general.

---

## Contents

- Part 0 — Why RAG exists (and why you cannot just paste the file into the prompt)
- Part 1 — The pipeline map: how every piece relates
- Part 2 — Mental models that make it click
- Part 3 — Ingestion, step by step (load, chunk, embed, build Documents, store)
- Part 4 — The vector store and schema, done right
- Part 5 — Retrieval: where answer quality is won
- Part 6 — Generation: context assembly and prompting
- Part 7 — Evaluation: knowing if it actually works
- Part 8 — Production concerns
- Part 9 — Scenario Q&A: symptom, why, fix
- Part 10 — End-to-end implementation skeleton
- Part 11 — Cheat sheets and defaults

---

# Part 0 — Why RAG exists

## What a large language model is, and what it cannot do

A large language model (LLM) is a next-token predictor trained on a huge but **fixed**
text corpus. That gives it fluent language and broad knowledge, but it has four hard
limits:

1. **No private knowledge.** It never saw your contracts, your case files, your
   internal wiki.
2. **Stale knowledge.** It only knows up to its training cutoff.
3. **Hallucination.** When it does not know, it produces plausible, confident, wrong
   text. It has no built-in sense of "I have no source for this."
4. **Finite context.** It can only read a limited amount of text per request.

RAG addresses 1 to 3 by feeding the model the right source text at question time, and
it does so in a way that respects limit 4.

## The obvious question: why not just paste the whole document in?

This is the right instinct to interrogate. If the model can read text, why not dump
the entire document (or all documents) into the prompt and ask the question? Several
reasons make this fail in practice:

- **Context window limits.** A model can read only so many tokens at once. A corpus of
  thousands of judgments does not fit. Even one long contract plus a long
  conversation can overflow.
- **Cost.** You pay per token, every request. Sending a 200-page document on every
  question is enormous, recurring waste when the answer depends on two paragraphs.
- **Latency.** More input tokens means slower responses. Stuffing everything makes
  every query slow even when the relevant slice is tiny.
- **Attention dilution ("lost in the middle").** Models attend unevenly across a long
  input. Bury the key clause in the middle of 100 pages and accuracy drops. A focused
  prompt with only the relevant chunks outperforms a giant one.
- **It does not scale or update.** "Paste everything" has no story for "I just added
  500 new documents" or "search across the whole library."

RAG flips the approach: **store everything once, retrieve only the few most relevant
pieces per question, and put just those into the prompt.** You get scale, low cost per
query, fast responses, and focused, grounded answers. The vector database is the
machinery that makes "find the few most relevant pieces" fast across millions of
chunks.

> One-line definition: RAG = search your own data for the relevant pieces, then let the
> LLM write the answer using those pieces as its source.

---

# Part 1 — The pipeline map

RAG has two phases. Ingestion runs once per document. Query runs once per question.

```
INGESTION (offline, per document)
  raw file (PDF/CSV/DOCX/scan)
    └─ LOAD ──────────────► raw text (+ OCR if scanned)
        └─ SPLIT ─────────► chunks (overlapping passages)
            └─ EMBED ─────► one vector per chunk
                └─ STORE ─► vectors + text + metadata in the vector DB

QUERY (online, per question)
  user question
    └─ (optionally TRANSFORM the query)
        └─ EMBED the query ──────────► query vector
            └─ SEARCH the vector DB ─► top-k candidate chunks  (+ metadata filter)
                └─ (optionally RERANK) ► best chunks
                    └─ ASSEMBLE context ► prompt = system + context + question
                        └─ LLM ─────────► grounded answer (+ citations)
```

## Who does what

- **Loaders / parsers** turn a file into text. (PDF text extraction, OCR for scans,
  CSV row reading.)
- **Splitters** cut text into chunks of a retrievable size.
- **Embedding model** maps text to a vector that captures meaning.
- **Vector database (Milvus)** stores vectors and answers nearest-neighbour queries
  fast; also stores text and metadata, and supports metadata filtering.
- **Retriever** is the query-time component that takes a question and returns chunks.
  It can wrap plain similarity search, hybrid search, reranking, query transforms, and
  so on.
- **LLM** writes the final answer from the retrieved context.
- **LangChain** is the orchestration layer that wires these together with consistent
  interfaces (the `Document` object, the `Retriever` interface, LCEL chains). It is
  glue and convenience, not a database and not a model.

Keep the layers distinct in your head: **LangChain orchestrates, Milvus stores and
searches, the embedding model and LLM are the intelligence.** Most confusion comes
from blurring these.

---

# Part 2 — Mental models that make it click

## Embeddings: meaning as coordinates

An embedding model turns a piece of text into a point in high-dimensional space (a
list of, say, 1,024 numbers). The training objective places texts with similar meaning
near each other. "Bail conditions for economic offences" and "anticipatory bail in a
fraud case" land close together even with no shared words; "monsoon rainfall in Kerala"
lands far away.

Analogy: imagine a vast map where every sentence is a pin, and distance equals
difference in meaning. Search becomes "find the pins nearest to the question's pin."

## Similarity search and why a vector DB

To answer a query you want the nearest pins to the query pin. Comparing the query to
every stored vector one by one (brute force) is accurate but slow at scale. A vector
database builds an **index** (HNSW, IVF, and others) that finds approximate nearest
neighbours in milliseconds across millions of vectors. That speed, plus durable
storage, metadata filtering, and updates, is why you use Milvus rather than a Python
list or a relational database. A SQL database is built to match exact values; a vector
database is built to rank by closeness in meaning.

## The Document: the universal currency

Across the whole pipeline, a unit of data is a `Document`: a piece of text
(`page_content`) plus a `metadata` dictionary. Loaders produce Documents, splitters
produce smaller Documents, the vector store ingests Documents, and the retriever
returns Documents. Learning this one object explains most of the LangChain surface.

## Milvus never sees a "PDF" or a "CSV"

By the time data reaches Milvus, every source has been flattened into the same shape:
`text + vector + metadata`. A PDF page, a CSV row, and a DOCX paragraph all become the
same kind of object. Like a library shelf that holds hardcovers, paperbacks, and
magazines as "items with a title and a shelf code," Milvus stores chunks and does not
care what the original file looked like. This single idea answers many schema
questions later.

## You retrieve chunks, not files

The unit of retrieval is the chunk, not the document. A good chunk is small enough to
be a precise, relevant hit and large enough to be self-contained and meaningful. Most
ingestion design is really chunking design.

---

# Part 3 — Ingestion, step by step

The order matters, and each step sets up the next. Here is the chain with the reason
for each link.

## 3.1 Load: file to text

**Goal:** get clean text out of whatever the user uploaded.

- Digital PDFs and DOCX: extract text directly.
- Scanned or photographed PDFs (no text layer): run OCR. Preprocess the image first
  (deskew, denoise, adaptive threshold) so OCR accuracy is high; garbage text here
  poisons everything downstream.
- CSV/structured: each row (or a group of rows) becomes text, often with the column
  names included so the meaning survives.

**Best practice:** normalize whitespace, fix encoding, strip boilerplate
(headers/footers/page numbers) that would otherwise pollute chunks and embeddings. The
cleaner the text, the better every later step works.

**Before/after:** before this step you have bytes on disk; after it you have plain
strings ready to be split. Do not chunk or embed until extraction is clean.

## 3.2 Split (chunk): text to passages

**Goal:** cut text into retrievable units. This is the single highest-leverage choice
for retrieval quality.

Why chunk at all: embeddings represent a whole passage as one vector. Too large and the
vector is a blurry average of many ideas, so it matches weakly and wastes context. Too
small and a chunk loses the context needed to be meaningful.

**Strategies, from simple to advanced:**

- **Fixed-size with overlap.** Split every N characters/tokens with an overlap (e.g.
  ~800 tokens, ~100 overlap). Overlap prevents a key sentence from being cut in half at
  a boundary. Simple and a fine default.
- **Structure-aware (recursive).** Split on natural boundaries first (sections,
  paragraphs, sentences), falling back to size limits. Keeps ideas intact. Strongly
  preferred for documents with structure, which legal text has (sections, clauses).
- **Document-structure-aware.** Use the document's own hierarchy (a statute's
  section/sub-section, a contract's clause numbering) as chunk boundaries. Best fidelity
  for legal and technical material.
- **Semantic chunking.** Split where the topic shifts (detected via embedding
  similarity between adjacent sentences). More compute, sometimes better coherence.

**Best practice:** choose chunk size with your embedding model's context and your
query style in mind. Keep overlap modest. Preserve structure where you can. Store
enough metadata on each chunk (source, section, page) to trace and filter it later.

**Before/after:** chunking takes clean text and produces the passages you will embed.
Get this wrong and no amount of clever retrieval recovers it; get it right and even
plain similarity search performs well.

## 3.3 Embed: passages to vectors

**Goal:** turn each chunk into a vector with the embedding model.

**Choices that matter:**

- **Model choice.** Pick a model suited to your domain and languages (for a
  bilingual legal corpus, a multilingual model). The embedding model defines the
  "meaning space"; everything is measured in it.
- **Dimension.** Higher dimension can capture more nuance but costs more storage and
  compute. Use what the model outputs; do not truncate arbitrarily.
- **Normalization and metric.** If you use cosine similarity, the metric must match
  what you index in Milvus (set `metric_type="COSINE"`). Mismatched metrics give
  meaningless rankings.
- **Same model for ingest and query.** You must embed queries with the **same** model
  you embedded chunks with, or the vectors are not comparable.
- **Batching.** Embed in batches for throughput; handle rate limits and retries.

**Before/after:** embedding consumes chunks and produces the vectors the store will
index. The query side will reuse this exact model.

## 3.4 Build the Document: where text meets metadata

**Goal:** package each chunk as a `Document` with its metadata. This is where the
"metadata contract" (Part 4) is fulfilled.

```python
from langchain_core.documents import Document

def to_document(chunk_text, source_file, section=None, extra=None):
    return Document(
        page_content=chunk_text,
        metadata={
            "source": source_file,           # which file this came from
            "document_type": classify(source_file),  # "statute" | "judgment" | ...
            "jurisdiction": None,            # fill when known; nullable in schema
            "year": None,
            "section": section,              # e.g. "Section 6"
            **(extra or {}),                 # page number, row index, etc.
        },
    )
```

**Where the Document is used:**

- Created here, at the end of ingestion preprocessing.
- Passed to `vector_store.add_documents([...])` to be embedded and stored.
- Returned by the retriever at query time, so the same `metadata` you set here is what
  you filter on and cite later.

**Best practice:** decide your metadata schema now, and have one normalization function
produce it for every source. The metadata you do not attach here is metadata you cannot
filter or cite later.

## 3.5 Store: Documents into the vector DB

`add_documents` embeds each chunk and writes `id + text + vector + metadata` into the
collection. On the very first insert, if the collection does not exist, it is created.
How that collection is shaped is Part 4, which is where most real decisions live.

---

# Part 4 — The vector store and schema, done right

## What a schema is

A collection's schema declares its fields: the primary key, the vector field (with its
dimension), the text field, and any metadata fields, each with a type. The schema also
carries indexes (one for the vector field, optionally scalar indexes on metadata
fields). The schema is **frozen at creation**; you cannot change field types or add a
required field afterwards.

## Auto schema vs manual schema: which to use

- **Auto schema** (let the wrapper build it on first insert): fine when retrieval is
  "find similar text, maybe a light filter," and when your data is uniform. Fastest to
  start.
- **Manual schema** (you define fields and indexes): use when you filter results by
  metadata, need a partition key, or want specific types and nullability. This is the
  production choice for anything with structured filters, which legal retrieval is.

There is a recommended **middle path**: explicit indexed columns for the few fields you
will reliably filter on, plus dynamic field on for everything else.

## Two behaviours that surprise people

**The collection is created lazily, on the first upload.** Constructing the
`Milvus(...)` object does not create anything; it is like naming a notebook you have
not bought yet. The collection appears on the first `add_documents`. Consequence: a
search before any upload returns empty, not an error.

**The auto-schema infers required columns from the first batch's metadata.** With the
default `enable_dynamic_field=False`, each metadata key in the first inserted documents
becomes a **required** column. If a later document lacks that key, the insert fails.
This is the classic "missed an field" crash. The fix is dynamic field and/or nullable
fields, below.

## enable_dynamic_field and nullable: the two safety valves

- **`enable_dynamic_field=True`**: metadata keys you did not model go into a flexible
  JSON pocket instead of becoming rigid required columns. Inserts with varied metadata
  stop crashing.
- **`nullable=True`** on a modeled field: a document missing that value inserts fine
  instead of being rejected.

Together these let many sources with differing metadata live in one collection.

## Does a fixed schema lock me to one file type?

**No.** The schema describes the **chunk** (`text + vector + metadata`), not the source
file format. Because every file type is flattened to the same chunk shape before
storage (Part 2), one collection holds PDF, CSV, and DOCX chunks together. What a schema
constrains is **metadata consistency**, and nullable plus dynamic field makes that easy
to satisfy.

## What forces a separate collection

A collection has exactly one vector field, one dimension, one embedding model. A
**different embedding model** forces a new collection (the "one language per room"
rule). File format never forces a split. You also split when bodies of data are so
unrelated you would never search them together.

## Defining the schema: full code, with before/after

**Before this step:** you have decided which fields you filter on, your embedding model
is chosen, and pymilvus is a version compatible with your wrapper.

**Run this once as a migration step, not on app startup.** Provisioning is a deliberate
act; running it on every boot risks dropping or racing your collection.

```python
# create_collection.py  — run ONCE
from pymilvus import MilvusClient, DataType
from config import milvus_uri
from services.utils import get_embedding_model

COLLECTION = "LegalDocs"
client = MilvusClient(uri=milvus_uri)

# dimension comes from the real model; never hardcode it
dim = len(get_embedding_model().embed_query("dimension probe"))

if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)   # WARNING: deletes data; intended on a rebuild

schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

# fields the LangChain wrapper expects BY NAME (keep these names, or pass overrides)
schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

# filterable domain fields — EDIT to your needs. nullable avoids the "missed field" crash.
schema.add_field("source",        DataType.VARCHAR, max_length=512, nullable=True)
schema.add_field("document_type", DataType.VARCHAR, max_length=64,  nullable=True)
schema.add_field("jurisdiction",  DataType.VARCHAR, max_length=128, nullable=True)
schema.add_field("court",         DataType.VARCHAR, max_length=256, nullable=True)
schema.add_field("statute",       DataType.VARCHAR, max_length=256, nullable=True)
schema.add_field("year",          DataType.INT16,                   nullable=True)

# indexes: one for the vector, scalar indexes for the fields you filter on
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                       params={"M": 16, "efConstruction": 200})
for f in ("source", "document_type", "jurisdiction", "court", "statute", "year"):
    index_params.add_index(field_name=f, index_type="INVERTED")
    # INVERTED is a good general default; BITMAP is cheaper for very low-cardinality
    # fields (a handful of distinct values); STL_SORT suits pure numeric range filters.

client.create_collection(collection_name=COLLECTION, schema=schema,
                         index_params=index_params, consistency_level="Strong")
print("created", COLLECTION, "dim", dim)
```

**After this step:** the collection exists with the shape you chose. Now point the
LangChain wrapper at it. Because the collection already exists, the wrapper reuses your
schema rather than inventing one. Field names must match what the wrapper expects, which
is why they are `pk` / `text` / `vector`; if you rename them, pass `primary_field=`,
`text_field=`, `vector_field=`.

```python
# services/vector_db.py
from functools import lru_cache
from langchain_milvus import Milvus
from config import milvus_uri
from services.utils import get_embedding_model

embedding_model = get_embedding_model()


class VectorStore:
    def __init__(self):
        self.store = Milvus(
            embedding_function=embedding_model,
            collection_name="LegalDocs",
            connection_args={"uri": milvus_uri},
            enable_dynamic_field=True,   # must match how the collection was created
            auto_id=True,
            primary_field="pk", text_field="text", vector_field="vector",
        )

    def add_documents(self, docs):
        return self.store.add_documents(docs)

    def reindex_file(self, docs, file_name):
        # delete-by-source then add: clean re-ingest when auto_id is on
        self.store.delete(expr=f'source == "{file_name}"')
        return self.store.add_documents(docs)

    def search(self, query, top_k=5, expr=None):
        return self.store.similarity_search(query=query, k=top_k, expr=expr)


@lru_cache(maxsize=1)
def get_vector_store():
    return VectorStore()
```

## The metadata contract (your side of the deal)

When you define explicit fields, your ingestion code must fill them. The agreement:

- The keys you promoted to columns are the keys your Document builder writes.
- Have a value, it lands in the fast indexed column.
- No value, `nullable=True` lets it through as null.
- Extra, unmodeled keys fall into the dynamic field.

This contract is what keeps inserts reliable across heterogeneous sources. It is not
the library's job to guarantee it; it is yours, enforced in one normalization function.

## What to do when documents have different fields but share one schema

This is the common real situation: a statute chunk has a `section`, a judgment chunk
has a `court` and a `bench`, a CSV row has neither. You do **not** want a separate
schema per type. Do this instead:

1. **Model the shared, stable filter fields as nullable columns** (`document_type`,
   `jurisdiction`, `year`, `source`). Every type can supply these or leave them null.
2. **Keep dynamic field on** so type-specific extras (`bench`, `row_index`, `page`)
   are stored without being declared.
3. **Normalize in one place** so every type maps onto the contract.

```python
def normalize(chunk_text, source, doc_type, **type_specific):
    base = {
        "source": source,
        "document_type": doc_type,
        "jurisdiction": type_specific.pop("jurisdiction", None),
        "year": type_specific.pop("year", None),
    }
    base.update(type_specific)   # bench, page, row_index, ... → dynamic field
    return Document(page_content=chunk_text, metadata=base)
```

Now statutes, judgments, and CSV rows coexist in one collection, each filterable on the
shared fields, each free to carry its own extras. This is the practical proof that a
schema does not lock you to one file structure.

## Filtered search: the payoff

The reason to model fields as indexed columns is filtered retrieval. The filter narrows
the candidate set **before** the expensive vector comparison, which is both faster and
more correct on a structured corpus.

```python
store = get_vector_store()

# only Gujarat High Court judgments from 2015 onward
docs = store.search(
    "bail conditions for economic offences", top_k=5,
    expr="court == 'Gujarat High Court' && year >= 2015",
)

# only statutes, any jurisdiction
docs = store.search(
    "foreign direct investment remittance limits", top_k=5,
    expr="document_type == 'statute'",
)
```

Operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `like`, `&&`, `||`. On a legal
corpus where a 1999 statute and its 2015 amendment are different answers, this filter is
not an optimization; it is correctness.

---

# Part 5 — Retrieval: where answer quality is won

Generation can only be as good as what you retrieve. Most quality work happens here.

## 5.1 Baseline: dense similarity search

Embed the query, find the top-k nearest chunk vectors, return them. Good default;
struggles with exact terms (statute numbers, party names, rare tokens) because
embeddings capture meaning, not exact strings.

## 5.2 Hybrid search: dense + sparse (lexical)

Combine semantic similarity (dense vectors) with keyword matching (sparse/BM25). Dense
catches paraphrase and meaning; sparse catches exact tokens like "Section 138" or
"FEMA, 1999." Fusing both is the single biggest retrieval upgrade for legal and
technical corpora, where exact identifiers matter.

In `langchain_milvus`, dense plus a built-in BM25 sparse function gives hybrid search,
fused with a ranker (Reciprocal Rank Fusion is the robust default):

```python
from langchain_milvus import Milvus, BM25BuiltInFunction

store = Milvus(
    embedding_function=embedding_model,          # dense
    builtin_function=BM25BuiltInFunction(),      # sparse (lexical), computed in Milvus
    vector_field=["dense", "sparse"],            # one field per modality
    collection_name="LegalDocs",
    connection_args={"uri": milvus_uri},
)
# similarity_search fuses dense + sparse with RRF by default
docs = store.similarity_search("Section 138 dishonour of cheque", k=5)
```

## 5.3 Reranking: precision after recall

Retrieval is a two-stage game. First **recall**: cast a wide net (fetch top 20 to 50
candidates cheaply). Then **precision**: a heavier **cross-encoder reranker** reads each
candidate together with the query and scores true relevance, and you keep the top 5.

A bi-encoder (your embedding model) encodes query and chunk separately, so it is fast
but coarse. A cross-encoder reads them jointly, so it is slow but far more accurate.
Using it only on the shortlist gives accuracy without the cost of scoring everything.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

base = store.as_retriever(search_kwargs={"k": 30})           # wide recall
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"),
    top_n=5,                                                 # precise top 5
)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=base
)
docs = retriever.invoke("anticipatory bail in a cheque dishonour case")
```

Hosted rerankers (Cohere, Jina, Voyage) drop in the same way. Reranking is usually the
second biggest quality lever after hybrid search.

## 5.4 Query transformation: fix the question before searching

The user's phrasing is often a poor search query. Transform it first:

- **Multi-query.** Generate several paraphrases of the question, retrieve for each,
  union the results. Covers vocabulary the user did not use.
- **HyDE (hypothetical document embeddings).** Ask the LLM to draft a hypothetical
  answer, embed *that*, and search with it. The draft is closer in meaning to real
  passages than a terse question is.
- **Decomposition.** Break a multi-part question into sub-questions, retrieve per
  sub-question, then synthesize. Essential for "compare X and Y under Z."
- **Conversational contextualization.** In a chat, rewrite a follow-up that depends on
  history ("what about its penalty?") into a standalone query ("what is the penalty
  under Section 138 of the Negotiable Instruments Act?") before retrieving. Detect when
  a new question is topically independent so you do not wrongly fuse unrelated turns.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
mqr = MultiQueryRetriever.from_llm(retriever=store.as_retriever(), llm=llm)
```

## 5.5 Small-to-big: parent-document and sentence-window retrieval

There is tension between chunk sizes: small chunks retrieve precisely but lack context;
large chunks carry context but retrieve fuzzily. Resolve it by **retrieving small,
returning big**: index small chunks for precise matching, but feed the LLM the larger
parent passage (or a window around the hit) so it has full context. LangChain's
`ParentDocumentRetriever` implements this.

## 5.6 Contextual retrieval: chunks that remember their place

A chunk like "It shall be punishable with imprisonment up to two years" is ambiguous in
isolation. Before embedding, prepend a short, LLM-generated context line situating the
chunk in its document ("This clause is from Section 138 of the Negotiable Instruments
Act, on cheque dishonour penalties."). Embedding the contextualized chunk sharply
improves retrieval of otherwise ambiguous passages. Combine with hybrid search and
reranking for a strong stack.

## 5.7 Diversity: MMR

When top results are near-duplicates, you waste context. Maximal Marginal Relevance
(MMR) trades a little relevance for diversity, so the k chunks cover more ground.

```python
retriever = store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.5},
)
```

## 5.8 Metadata pre-filtering

Always prefer filtering at the database (the `expr` in Part 4) over fetching everything
and filtering in Python. Pre-filtering shrinks the search space and is both faster and
more correct. Pair every advanced technique above with the right metadata filter.

**How to layer these:** a strong production retriever is often hybrid search for recall,
then a cross-encoder reranker for precision, with a metadata pre-filter applied, and
query contextualization in front for chat. Add multi-query or HyDE if recall is still
the bottleneck. Add parent-document if chunks feel context-starved.

---

# Part 6 — Generation: turning chunks into an answer

## Assembling context

Concatenate the retrieved chunks into a context block, labelled with their source so the
model (and the user) can trace claims. Put the most relevant chunks where the model
attends best (near the start and end), and keep the block within budget after reranking
has trimmed it.

```python
context = "\n\n".join(
    f"[Source {i+1} | {d.metadata.get('source')} | {d.metadata.get('section','')}]\n{d.page_content}"
    for i, d in enumerate(docs)
)
```

## Prompting for grounded answers

The system prompt should: instruct the model to answer **only** from the provided
context, to cite the source numbers it used, and to say "I do not have enough
information" when the context does not cover the question. This last instruction is the
main defence against hallucination in RAG; without it the model fills gaps from its
parametric memory.

```
You are a legal research assistant. Answer using ONLY the context below.
Cite sources as [Source N]. If the context does not contain the answer, say so plainly.
Do not rely on outside knowledge.

Context:
{context}

Question: {question}
```

## Other generation concerns

- **Structured output** when the consumer is code, not a human (extract fields into
  JSON with a schema).
- **Streaming** for responsive UX on long answers.
- **No-context guardrail.** If retrieval returns nothing above a relevance threshold,
  short-circuit to "no relevant information found" rather than letting the model
  improvise.
- **Citations.** Map cited source numbers back to document metadata so the user can
  verify. For legal use this is not optional.

---

# Part 7 — Evaluation: knowing if it actually works

You cannot improve what you do not measure. Evaluate the two stages separately, because
a bad answer can come from bad retrieval or bad generation, and the fixes differ.

**Retrieval metrics** (did the right chunks come back?):

- **Recall@k**: was the relevant chunk in the top k?
- **Precision@k**: how many of the k are relevant?
- **MRR / nDCG**: did the relevant chunk rank near the top?

**Generation metrics** (did the answer use them well?):

- **Faithfulness / groundedness**: is every claim supported by the retrieved context?
- **Answer relevance**: does it address the question?
- **Context precision/recall**: did retrieval provide what generation needed?

**How to do it in practice:** build a small but real test set of question, ideal answer,
and the source chunk(s) that contain it. Run the pipeline, score with an automated
framework (RAGAS-style, often using an LLM as judge) plus spot human review. When you
change chunking, the embedding model, or the retriever, re-run the set and compare. This
turns "it feels better" into a number.

---

# Part 8 — Production concerns

- **Re-ingest, update, delete.** With `auto_id=True` there is no automatic dedup, so a
  re-upload duplicates chunks. Use delete-by-source then add (the `reindex_file` method
  in Part 4). To remove a document, `delete(expr='source == "file.pdf"')`.
- **Document versioning.** Store a version or ingested-at timestamp in metadata so you
  can prefer the latest and audit what was retrieved when.
- **Multi-tenancy / isolation.** For per-client or per-document scoping, declare a
  **partition key** at schema creation (`is_partition_key=True` and pass
  `partition_key_field=`). Milvus then physically segregates and scans only the
  relevant partition.
- **Caching.** Cache embeddings of repeated inputs, and cache full query results for
  hot queries, to cut latency and cost. Deterministic embedding (temperature 0 on any
  LLM-side query transform) plus input hashing makes caching safe.
- **Latency and cost.** Hybrid plus rerank adds latency; tune recall `k`, reranker
  size, and batch sizes. Keep one vector-store instance (cache it) rather than rebuilding
  per request.
- **Observability.** Trace each stage (transformed query, retrieved chunks and scores,
  final prompt, answer). When an answer is wrong you must see whether retrieval or
  generation failed.
- **Security.** Treat retrieved content as untrusted: it can carry prompt-injection
  text, so instruct the model to treat context as data, not instructions. Enforce
  access control by filtering on a permission/tenant field so users only retrieve what
  they may see. Be deliberate about PII in stored text and metadata.
- **Version compatibility.** The data-plane libraries (vector DB client, the wrapper)
  must agree. "Latest plus latest" is not a guarantee; pin to a tested combination and
  upgrade deliberately with the eval set as your safety net.

---

# Part 9 — Scenario Q&A: symptom, why, fix

**Insert fails: "missed an field X."**
Why: auto-schema made X a required column from the first batch; a later document lacks
it. Fix: rebuild the collection with `enable_dynamic_field=True` and nullable domain
fields.

**Startup error: "should create connection first."**
Why: a version mismatch between the wrapper and the vector-DB client, not your code.
Fix: pin the client to a version the wrapper was tested against; re-check after upgrades.

**Search returns nothing.**
Why/fix, in order: nothing uploaded yet (collection is created on first insert); custom
field names not passed to the wrapper; index metric does not match the search metric; an
over-tight metadata filter; weak consistency on a read right after a write (use
"Strong").

**Re-uploading a file duplicates chunks.**
Why: `auto_id=True` mints new keys each insert; no dedup. Fix: delete-by-source, then
add.

**Different document types have different fields.**
Why: you are letting loaders define metadata, so keys vary. Fix: one normalization
function to a shared nullable contract, dynamic field on for extras. One schema, many
types.

**Need a filter but everything is in dynamic JSON and slow.**
Why: unmodeled metadata is not an indexed column. Fix: promote the filter field to a
typed, indexed column (rebuild), then filter with `expr`.

**Existing collection has the wrong schema.**
Why: schema is frozen at creation. Fix: drop and rebuild if data is reproducible from
source; otherwise create a new collection and copy entities across.

**Changed the embedding model and results broke.**
Why: vectors from different models are not comparable, and dimensions differ. Fix: new
collection, re-embed everything from source, repoint the app.

**Retrieval brings back semantically close but wrong-identifier chunks** (e.g. wrong
section number). Why: dense embeddings match meaning, not exact tokens. Fix: add hybrid
(BM25) search so exact identifiers are matched; consider reranking.

**Answers hallucinate despite retrieval.**
Why: the prompt lets the model use outside knowledge, or retrieval missed the relevant
chunk. Fix: instruct "answer only from context, else say you do not know"; verify
retrieval actually returned the needed chunk (an eval-set job); add a no-context
guardrail and relevance threshold.

**Chunks lack context, answers are fragmentary.**
Why: chunks too small or stripped of surrounding text. Fix: parent-document or
sentence-window retrieval, or contextual retrieval that prepends a situating line before
embedding.

**Top results are near-duplicates, wasting context.**
Why: redundant neighbours. Fix: MMR for diversity.

**Filtered search is slow at scale, or tenants leak into each other.**
Why: a plain indexed column still scans the whole collection. Fix: a partition key on
the always-scoping field.

---

# Part 10 — End-to-end implementation skeleton

```
project/
  create_collection.py        # Part 4: run ONCE to provision the schema
  services/
    document_parser.py        # 3.1–3.4: load, OCR, clean, chunk, normalize → Documents
    vector_db.py              # 3.5 + 4: the VectorStore wrapper (add/reindex/search)
    retriever.py              # Part 5: hybrid + rerank + filters + query transforms
    llm_service.py            # Part 6: context assembly, grounded prompt, generation
  evals/                      # Part 7: test set + scoring
  main.py                     # API: /upload (ingest), /search (RAG), /chat (with history)
```

Ordered build:

1. Pin compatible versions; stand up Milvus.
2. Decide chunking and metadata contract; write `document_parser.py`.
3. Provision the collection once with `create_collection.py`.
4. Implement `vector_db.py`; ingest a sample; confirm `add_documents` works.
5. Implement retrieval: start with similarity + filters, then add hybrid, then rerank.
6. Implement generation with a grounded prompt and citations.
7. Build a small eval set; measure; iterate on chunking, embeddings, and retrieval.
8. Add production concerns: dedup/update, caching, tracing, access control, partitions.

---

# Part 11 — Cheat sheets and defaults

## Pipeline decisions

| Decision | Sensible default | Upgrade when |
|---|---|---|
| Chunking | structure-aware, ~800 tokens, ~100 overlap | use document structure; try semantic |
| Embedding | a strong multilingual model, full dimension | domain-tuned model; evaluate alternatives |
| Vector index | HNSW + COSINE | tune M/efConstruction for recall vs speed |
| Retrieval | similarity + metadata filter | hybrid (BM25) → rerank → query transforms |
| Context | top 5 after rerank, sources labelled | parent-document; contextual retrieval |
| Generation | grounded prompt, cite sources, no-context guard | structured output; streaming |

## Schema decisions

| Situation | Do |
|---|---|
| Uniform data, light filtering | auto schema |
| Filter by metadata | manual schema, indexed columns |
| Heterogeneous sources | shared nullable contract + dynamic field |
| Different embedding model | separate collection |
| Per-tenant / per-doc scoping | partition key |
| Provisioning | run once as migration, never on app startup |

## Retrieval technique selector

| Problem | Technique |
|---|---|
| Misses exact identifiers/terms | hybrid (dense + BM25) |
| Right candidates, wrong order | cross-encoder reranking |
| User phrasing is poor | multi-query / HyDE |
| Multi-part question | decomposition |
| Chat follow-ups | conversational contextualization |
| Chunks lack context | parent-document / contextual retrieval |
| Redundant results | MMR |

## The mental models to keep

- RAG = retrieve the few relevant pieces, then let the LLM write from them.
- LangChain orchestrates; Milvus stores and searches; the embedding model and LLM are
  the intelligence.
- Embeddings are meaning as coordinates; search is nearest neighbours.
- The `Document` (text + metadata) is the currency of the whole pipeline.
- Milvus stores chunks, not files; the schema describes chunks, not file formats.
- A different embedding model is the one thing that forces a separate collection.
- Quality is won in retrieval; correctness in generation is won by grounding the prompt.

---

*Field names, types, chunk sizes, and model choices here are sensible starting points.
Tune them against an evaluation set built from your own documents and questions.*
