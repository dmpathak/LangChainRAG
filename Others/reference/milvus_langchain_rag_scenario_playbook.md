# Milvus + LangChain RAG: A Practical Playbook

A working developer's guide to building a Retrieval-Augmented Generation (RAG) system
on Milvus through `langchain_milvus`. It explains the concepts in plain language, then
gives **"when this situation happens, do this"** guidance for the real problems you
hit in production.

Built around a legal-document system (statutes, judgments, agreements), but the
patterns apply to any RAG project.

---

## Table of contents

1. Part 1 — Foundations: how RAG, LangChain, and Milvus fit together
2. Part 2 — The decision questions (the FAQ that clears the concepts)
3. Part 3 — Real-world scenarios: when X happens, do Y
4. Part 4 — Implementation walkthrough: zero to a filtered RAG search
5. Part 5 — Cheat sheet

---

# Part 1 — Foundations

## What RAG actually is

A language model only knows what it was trained on. RAG is how you let it answer
questions about *your* documents: you **retrieve** the most relevant pieces of your
own data and hand them to the model as context, so it **generates** an answer grounded
in those pieces instead of guessing.

The flow, end to end:

```
  1. INGEST (done once per document)
     PDF / CSV / DOCX
        → split into chunks
        → embed each chunk into a vector (a list of numbers)
        → store vectors + text + metadata in Milvus

  2. QUERY (done per user question)
     user question
        → embed the question into a vector
        → Milvus finds the chunks whose vectors are closest (similarity search)
        → those chunks become "context"
        → context + question → LLM → grounded answer
```

Milvus is the **vector database**: it stores the vectors and answers the question
"which stored chunks are most similar to this query vector?" That nearest-neighbour
search is the "R" in RAG.

## What does LangChain's Milvus wrapper do for me?

You *can* talk to Milvus directly with `pymilvus`. The `langchain_milvus` wrapper sits
on top and does the repetitive glue work:

- Calls your embedding model for you on insert and on query.
- Auto-creates the collection and a default schema on your first insert.
- Maps a chunk's `metadata` dict to Milvus fields and back.
- Gives you a `Retriever` object that plugs into LangChain chains (LCEL).

The trade is convenience for control. The wrapper makes sensible default choices; this
guide is mostly about *when those defaults are not what you want* and how to take the
wheel.

## The single idea that prevents most confusion

**Milvus never sees a "PDF" or a "CSV".** Every source is flattened into the same
shape before storage: `text + vector + metadata`. A PDF page, a CSV row, and a DOCX
paragraph all become the same kind of object (`Document(page_content=..., metadata={...})`).

Think of a library. Books arrive as hardcovers, paperbacks, and magazines, but on the
shelf they are all just "items with a title and a shelf code." The shelf does not care
about the cover. Hold this picture; it answers most of Part 2.

---

# Part 2 — The decision questions

### Q: Do I need to write a schema myself?

**No, not to get started.** `langchain_milvus` builds one automatically on your first
upload. You only write a schema yourself when you need control the auto-path cannot
give you (filtering, partitioning, specific types). See Part 3, Scenario C, for when
that moment arrives.

### Q: When is the collection actually created?

**On the first `add_documents`, not when you construct the `Milvus(...)` object.**

Constructing `Milvus(...)` is like *choosing a name* for a notebook. The notebook is
not bought until you write the first page. Practical consequence: if you call
`similarity_search` before anything has ever been uploaded, the collection does not
exist yet and you get empty results, not an error.

### Q: Does a fixed schema restrict you to one file structure?

**No. This is the most important one to get right.**

A schema describes the **chunk** (`text + vector + metadata`), not the **source file
format**. Because every file type is flattened into the same chunk shape before it
reaches Milvus, one collection holds chunks from PDFs, CSVs, and DOCX side by side
without any problem.

What a schema actually constrains is **metadata consistency**, not file type. And you
keep that consistency easy to satisfy with two tools:

1. Make domain fields **nullable**, so a missing value is allowed, not fatal.
2. Keep **dynamic field on**, so any metadata key you did not model lands in a
   flexible JSON pocket instead of crashing the insert.

So the rule is never "one file type per collection." The rule is "every chunk must
satisfy the metadata contract," and nullable + dynamic field makes that contract
forgiving. A worked example of mixing file types is in Part 3, Scenario E.

### Q: One collection, or many? When do I split?

Use **one** collection for one body of related data in one embedding space.

The one thing that genuinely forces a **separate** collection is the embedding model.
A collection has exactly one vector field, one dimension, one embedding model. Change
the model, or want to mix two, and you need a new collection.

Analogy: a collection is a room where everyone speaks one language (the embedding
model). People can wear different clothes (file types) and it is fine. Someone
speaking a different language (a different embedding model) needs a different room,
because no one here can understand them.

You also split when data is so unrelated you would never search across it together
(your legal corpus vs. a product catalog).

### Q: Auto schema or manual schema?

Decide with one question: **do my searches involve structured filters alongside the
vector similarity?**

- "Find the most similar text, maybe a light filter" → **auto schema** is fine.
- "Find similar text, but only Gujarat High Court judgments from 2015 onward" →
  **manual schema** with indexed fields.

There is also a recommended middle path: define explicit indexed columns for the few
fields you are sure you will filter on, and leave dynamic field on for everything
else. Stable filters get fast columns; loose metadata stays safe in JSON.

---

# Part 3 — Real-world scenarios: when X happens, do Y

Each scenario follows the same shape: **Symptom → Why it happens → Do this.**

---

## Scenario A — Insert crashes: "missed an field `row_number`"

**Symptom**
```
DataNotMatchException: Insert missed an field `row_number` to collection
without set nullable==true or set default_value
```

**Why it happens**
Your collection was created earlier (auto-schema, `enable_dynamic_field=False`) from a
batch of documents that happened to carry a `row_number` metadata key. That made
`row_number` a **required** column. A later document (different source) has no
`row_number`, so Milvus rejects it: you promised every row has this field.

It is like a form where someone marked "Middle Name" as required. Now everyone without
a middle name is stuck.

**Do this**
Recreate the collection with dynamic field on and your domain fields made nullable.
Existing schema cannot be altered in place, so this is a rebuild. See Scenario D for
the migration steps, and Part 4 for the full provisioning script. The core change:

```python
Milvus(
    ...,
    enable_dynamic_field=True,   # unmodeled metadata goes to a JSON pocket, no crash
)
```

---

## Scenario B — Startup crash: "should create connection first"

**Symptom**
```
ConnectionNotExistException: should create connection first.
```
...even though Milvus is running and your URI is correct.

**Why it happens**
A version mismatch, not your code. `langchain_milvus` 0.3.3 still uses the older
ORM-style `Collection` internally. `pymilvus` 2.6.10 changed how `MilvusClient`
registers its connection, and the two stopped agreeing. "Latest + latest" is not a
compatibility guarantee when libraries version independently.

**Do this**
Pin pymilvus below 2.6.10. The highest safe version is 2.6.9, and it still satisfies
`langchain_milvus`'s own requirement (`pymilvus>=2.6.0,<3.0`):

```bash
pip install "pymilvus>=2.6.0,<2.6.10"
```

Verified boundary: 2.6.9 works, 2.6.10 does not. Re-check when a newer
`langchain_milvus` ships that drops the ORM `Collection` call; then you can unpin.

---

## Scenario C — I need to filter results, not just find similar text

**Symptom**
Your retrieval returns text-similar chunks, but you cannot say "only statutes" or
"only this jurisdiction," and stuffing that logic into the prompt is unreliable.

**Why it happens**
With the auto-schema, your metadata sits in the dynamic JSON pocket. Milvus *can*
filter on JSON, but it is slower and more limited than filtering on a real, indexed,
typed column. For correctness-critical filters (a 2015 judgment vs a 1999 one), you
want real columns.

**Do this**
Promote your filter fields to explicit, indexed columns in a manual schema (Part 4),
then pass an `expr` to the search:

```python
results = store.similarity_search(
    query="bail conditions for economic offences",
    top_k=5,
    expr="court == 'Gujarat High Court' && year >= 2015",
)
```

Why it is fast: an indexed column lets Milvus *first* shrink the candidate set to the
matching rows, *then* run the expensive vector comparison on that smaller set.

Operators you will use in `expr`: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `&&`, `||`.

---

## Scenario D — I already have a collection with the wrong schema

**Symptom**
You changed your mind about fields (or hit Scenario A), but the collection already
exists with the old, rigid schema. Flipping `enable_dynamic_field=True` in code does
nothing, because the wrapper reuses the existing collection's schema.

**Why it happens**
A Milvus collection's schema is **frozen at creation**. You cannot add a required
field, change a field's type, or turn on dynamic field after the fact.

**Do this — pick based on whether you can lose the data:**

*If the data is disposable (dev, or you can re-ingest from source files):* drop and
rebuild.
```python
client.drop_collection("MyLangChainCollection")   # deletes everything
# then run the provisioning script in Part 4, then re-upload your files
```

*If the data is precious and you cannot re-ingest:* create a **new** collection with
the correct schema, then copy the data across by reading old entities and re-inserting
them. Sketch:
```python
old = MilvusClient(uri=milvus_uri)
rows = old.query("OldCollection", filter="pk >= 0",
                 output_fields=["text", "vector", "source", "year"])  # page through if large
# build Documents from rows, then add_documents into the new collection
```
Re-embedding is only necessary if you are also changing the embedding model (Scenario
H); otherwise reuse the stored vectors.

Practical tip: keep your raw source files. If you can always re-ingest, every schema
change becomes a cheap "drop and rebuild" instead of a risky data migration.

---

## Scenario E — I'm mixing PDFs and CSVs and their metadata differs

**Symptom**
Different loaders emit different metadata keys (a CSV loader adds row indices, a PDF
loader adds page numbers), and inserts are inconsistent or crashing.

**Why it happens**
You are leaning on the source loaders to define your metadata, so the keys vary by
file type. The schema cannot absorb that if fields are required.

**Do this**
Introduce a **metadata contract** in your `DocumentProcessor`: a normalization step
that maps every source into the same set of keys before insert. Model the stable ones
as nullable columns; let the rest fall into the dynamic field.

```python
from langchain_core.documents import Document

def normalize(raw_text, source_file, extra=None):
    return Document(
        page_content=raw_text,
        metadata={
            # the contract: always present (nullable in schema, so None is OK)
            "source": source_file,
            "document_type": detect_type(source_file),   # "statute" | "judgment" | ...
            "jurisdiction": None,
            "year": None,
            # anything source-specific (page, row index, ...) goes here and lands
            # safely in the dynamic JSON field:
            **(extra or {}),
        },
    )
```

Now a PDF chunk and a CSV row both satisfy the same contract, and they coexist in one
collection. This is the practical proof of "a schema does not lock you to one file
type."

---

## Scenario F — Search returns nothing

**Symptom**
`similarity_search` returns an empty list even though you think data is there.

**Why it happens / Do this — run this checklist in order:**

1. **Was anything ever uploaded?** The collection is created on the first
   `add_documents` (Q in Part 2). No upload yet means no collection. Upload first.
2. **Field-name mismatch.** If you hand-built the schema with custom field names, the
   wrapper must be told: pass `primary_field=`, `text_field=`, `vector_field=` to
   match. A mismatch means the wrapper queries fields that do not exist.
3. **Metric mismatch.** The index `metric_type` (e.g. `COSINE`) must match what the
   search uses. Mixed metrics give meaningless or empty results.
4. **An over-tight filter.** If you passed an `expr`, loosen it. Maybe nothing matches
   `year >= 2030`.
5. **Consistency level.** On a brand-new insert with `consistency_level` left weak, a
   read immediately after a write can miss it. Use `"Strong"` if you query right after
   inserting.

---

## Scenario G — Re-uploading the same file creates duplicates

**Symptom**
You upload `fema_1999.pdf` twice and now every chunk exists twice, polluting results.

**Why it happens**
With `auto_id=True`, Milvus assigns a fresh primary key every insert, so it has no way
to know a chunk is a re-upload. There is no automatic dedup.

**Do this — delete the old version by source, then add the new one:**

```python
class VectorStore:
    def reindex_file(self, documents, file_name):
        # remove any existing chunks from this file first
        self.vector_store.delete(expr=f'source == "{file_name}"')
        # then insert the fresh chunks
        return self.vector_store.add_documents(documents)
```

This "delete-by-source then add" pattern is the clean re-ingest approach when
`auto_id=True`. (True upsert-by-id needs `auto_id=False` and deterministic ids you
generate yourself, which is more bookkeeping than most pipelines want.)

---

## Scenario H — I want to change the embedding model

**Symptom**
You switch from one embedding model to another (different provider, or a model with a
different vector dimension) and searches break or return nonsense.

**Why it happens**
Vectors from different models are not comparable, and dimensions may differ. The
collection is locked to one embedding space (the "one language per room" rule).

**Do this**
Treat it as a new collection: provision a fresh collection (the new dimension comes
out automatically from `len(embed_query("probe"))` in the Part 4 script),
re-embed **all** documents from source, and insert them there. Point your app at the
new collection name. Do not try to mix old and new vectors in one place.

---

## Scenario I — Filtered search is slow, or I need tenant isolation

**Symptom**
Filters work but are slow at scale, or you need each client/tenant's data kept
separate and never mixed in results.

**Why it happens**
A plain indexed column still scans across the whole collection. For a field that
*always* scopes the query (a tenant id, or a single document id), Milvus can do better
by physically partitioning the data.

**Do this**
Declare a **partition key** at schema creation. Milvus then stores rows grouped by
that key and only scans the relevant partition at query time.

```python
# in the provisioning script:
schema.add_field("tenant_id", DataType.VARCHAR, max_length=64, is_partition_key=True)

# tell the wrapper which field is the partition key:
Milvus(..., partition_key_field="tenant_id")
```

Use a partition key for the dimension that scopes nearly every query (tenancy, or
per-document chat). Use ordinary indexed columns for dimensions you filter on *some* of
the time (jurisdiction, year).

---

# Part 4 — Implementation walkthrough: zero to a filtered RAG search

The ordered steps to a production-shaped setup. Field names and types are a template;
edit the domain fields to match what you filter on.

### Step 1 — Pin a compatible pymilvus (avoids Scenario B)

```bash
pip install "pymilvus>=2.6.0,<2.6.10" langchain_milvus
```

### Step 2 — Provision the collection ONCE (manual schema)

Run this as a migration script, not on every app start.

```python
# create_collection.py
from pymilvus import MilvusClient, DataType
from config import milvus_uri
from services.utils import get_embedding_model

COLLECTION = "MyLangChainCollection"
client = MilvusClient(uri=milvus_uri)

# dimension comes from your real model — never hardcode it
dim = len(get_embedding_model().embed_query("dimension probe"))

if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)   # WARNING: deletes data; intended on rebuild

schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

# fields the wrapper expects BY NAME (keep these exact names)
schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

# your filterable domain fields — EDIT THIS LIST. nullable=True avoids Scenario A.
schema.add_field("source",        DataType.VARCHAR, max_length=512, nullable=True)
schema.add_field("document_type", DataType.VARCHAR, max_length=64,  nullable=True)
schema.add_field("jurisdiction",  DataType.VARCHAR, max_length=128, nullable=True)
schema.add_field("court",         DataType.VARCHAR, max_length=256, nullable=True)
schema.add_field("statute",       DataType.VARCHAR, max_length=256, nullable=True)
schema.add_field("year",          DataType.INT16,                   nullable=True)

index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                       params={"M": 16, "efConstruction": 200})
# INVERTED for general filters; BITMAP is cheaper for very low-cardinality fields
for f in ("source", "document_type", "jurisdiction", "court", "statute", "year"):
    index_params.add_index(field_name=f, index_type="INVERTED")

client.create_collection(collection_name=COLLECTION, schema=schema,
                         index_params=index_params, consistency_level="Strong")
print("created", COLLECTION, "dim", dim)
```

### Step 3 — Wire the wrapper to the existing collection

```python
# services/vector_db.py
from functools import lru_cache
from langchain_milvus import Milvus
from config import milvus_uri
from services.utils import get_embedding_model

embedding_model = get_embedding_model()


class VectorStore:
    def __init__(self):
        self.vector_store = Milvus(
            embedding_function=embedding_model,
            collection_name="MyLangChainCollection",
            connection_args={"uri": milvus_uri},
            enable_dynamic_field=True,   # MUST match how the collection was created
            auto_id=True,
            primary_field="pk", text_field="text", vector_field="vector",
            # no index_params / drop_old: the collection is already provisioned
        )

    def add_documents(self, documents):
        return self.vector_store.add_documents(documents)

    def reindex_file(self, documents, file_name):           # Scenario G
        self.vector_store.delete(expr=f'source == "{file_name}"')
        return self.vector_store.add_documents(documents)

    def similarity_search(self, query, top_k=5, expr=None):  # Scenario C
        return self.vector_store.similarity_search(query=query, k=top_k, expr=expr)


@lru_cache(maxsize=1)            # reuse one instance instead of rebuilding per request
def get_vector_store():
    return VectorStore()
```

### Step 4 — Enforce the metadata contract on ingest (Scenario E)

Your `DocumentProcessor` must write the modeled keys into each chunk's metadata.
Missing values are fine because the columns are nullable; unmodeled keys fall into the
dynamic field.

### Step 5 — Retrieve with filters and feed the LLM

```python
store = get_vector_store()
docs = store.similarity_search(
    query=user_query, top_k=5,
    expr="document_type == 'statute' && jurisdiction == 'India'",
)
context = "\n\n".join(d.page_content for d in docs)
# context + user_query → your LLM chain → grounded answer
```

That filtered retrieval is the whole point: it restricts to the right documents
*before* similarity ranking, which is where answer quality on a structured corpus
comes from.

---

# Part 5 — Cheat sheet

| Situation | What to do |
|---|---|
| Just starting, data is uniform | Let the wrapper auto-create the schema. |
| `ConnectionNotExistException` at startup | Pin `pymilvus>=2.6.0,<2.6.10`. |
| `Insert missed an field ...` crash | Rebuild with `enable_dynamic_field=True` + nullable fields. |
| Need to filter by metadata | Manual schema with indexed columns; pass `expr` to search. |
| Existing collection has wrong schema | Drop + rebuild (or copy to a new collection if data is precious). |
| Mixing PDFs / CSVs / DOCX | One collection + a metadata-contract normalization step. |
| Search returns nothing | Check: uploaded yet? field names match? metric match? filter too tight? consistency? |
| Re-upload makes duplicates | `delete(expr='source == "file"')` then `add_documents`. |
| Changing embedding model | New collection, re-embed everything from source. |
| Slow filters / tenant isolation | Partition key (`is_partition_key=True` + `partition_key_field=`). |
| Does a fixed schema lock me to one file type? | No. Schema describes the chunk, not the file format. |
| What forces a separate collection? | A different embedding model / dimension. |
| Safe versions | `langchain_milvus` 0.3.3 with `pymilvus` 2.6.9. |

---

*Field names, types, and the domain field list are a starting template. Tune them to
the dimensions you actually filter and partition on.*
