# Build a Production RAG System: Step-by-Step (with Code)

A minimal but production-shaped RAG pipeline on Milvus + LangChain. Every API here was
checked against a live install, so the code runs as written. Stack: LangChain 1.x,
`langchain-milvus` 0.3.3, `pymilvus` 2.6.x (pinned, see Step 0).

The build order is deliberate: **install → provision once → ingest → retrieve+generate
→ serve**. Each step depends on the one before it.

---

## Step 0 — Install (the version pin matters)

```bash
pip install \
  "langchain>=0.3" "langchain-openai" "langchain-text-splitters" \
  "langchain-community" "langchain-milvus==0.3.3" \
  "pymilvus>=2.6.0,<2.6.10" \
  pypdf
```

**Why the pin:** `langchain-milvus` 0.3.3 still uses an internal API that `pymilvus`
2.6.10 broke. With 2.6.10+ you get `ConnectionNotExistException: should create
connection first` on startup. 2.6.9 is the highest version that works with 0.3.3.

Run Milvus locally (Docker is simplest):

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start    # serves on localhost:19530
```

---

## Step 1 — Config and shared models

Keep one place for settings and for the embedding/LLM clients. Cache the clients so you
do not rebuild them per request.

```python
# config.py
import os

MILVUS_URI   = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION   = os.getenv("MILVUS_COLLECTION", "LegalDocs")

LLM_MODEL        = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL")   # set this for OpenRouter / vLLM / Azure
```

```python
# models.py
from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
import config

@lru_cache(maxsize=1)
def get_llm():
    return init_chat_model(config.LLM_MODEL, model_provider="openai", temperature=0)

@lru_cache(maxsize=1)
def get_embeddings():
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,   # None = real OpenAI; set for compatible servers
    )
```

**Best practice:** `temperature=0` for the answer model makes outputs stable and
cacheable. The embedding model chosen here defines your whole "meaning space"; the query
side must use the exact same model.

---

## Step 2 — Provision the collection ONCE

Create the collection with an explicit, flexible schema. **Run this as a one-time
migration script, never on app startup.** Running it on boot risks dropping or racing
your data.

```python
# provision.py  — run once
from pymilvus import MilvusClient, DataType
import config
from models import get_embeddings

def provision(drop_existing=False):
    client = MilvusClient(uri=config.MILVUS_URI)

    if client.has_collection(config.COLLECTION):
        if not drop_existing:
            print("already exists"); return
        client.drop_collection(config.COLLECTION)   # deletes data

    # dimension comes from the real model; never hardcode it
    dim = len(get_embeddings().embed_query("dimension probe"))

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    # names the LangChain wrapper expects by default:
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    # filterable fields — EDIT to your needs. nullable=True so a missing value is OK.
    schema.add_field("source",        DataType.VARCHAR, max_length=512, nullable=True)
    schema.add_field("document_type", DataType.VARCHAR, max_length=64,  nullable=True)
    schema.add_field("jurisdiction",  DataType.VARCHAR, max_length=128, nullable=True)
    schema.add_field("year",          DataType.INT16,                   nullable=True)

    index = client.prepare_index_params()
    index.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                    params={"M": 16, "efConstruction": 200})
    for f in ("source", "document_type", "jurisdiction", "year"):
        index.add_index(field_name=f, index_type="INVERTED")

    client.create_collection(config.COLLECTION, schema=schema,
                             index_params=index, consistency_level="Strong")
    print("created", config.COLLECTION, "dim", dim)

if __name__ == "__main__":
    provision(drop_existing=True)
```

**Why explicit schema:** modeling `jurisdiction`, `year`, etc. as real indexed columns
makes metadata filtering fast and correct. `enable_dynamic_field=True` lets any extra
metadata you did not model still be stored, and `nullable=True` stops the "missed an
field" insert crash when documents differ.

---

## Step 3 — Ingest: load, split, build Documents, store

One module turns any uploaded file into stored chunks. The chunk size and the metadata
contract are the two quality-critical choices here.

```python
# ingest.py
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_milvus import Milvus
import config
from models import get_embeddings

@lru_cache(maxsize=1)
def get_store():
    return Milvus(
        embedding_function=get_embeddings(),
        collection_name=config.COLLECTION,
        connection_args={"uri": config.MILVUS_URI},
        enable_dynamic_field=True,    # must match how the collection was created
        auto_id=True,
        primary_field="pk", text_field="text", vector_field="vector",
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150,    # split on natural boundaries, with overlap
    separators=["\n\n", "\n", ". ", " ", ""],
)

def load_chunks(path, document_type, jurisdiction=None, year=None):
    pages = PyPDFLoader(path).load()       # one Document per page, with page metadata
    chunks = splitter.split_documents(pages)
    # the metadata CONTRACT: every chunk gets the same modeled fields, filled the same way
    for c in chunks:
        c.metadata = {
            "source": path.split("/")[-1],
            "document_type": document_type,
            "jurisdiction": jurisdiction,
            "year": year,
            "page": c.metadata.get("page"),   # extra key → stored in dynamic field
        }
    return chunks

def ingest_file(path, document_type, **meta):
    store = get_store()
    chunks = load_chunks(path, document_type, **meta)
    store.delete(expr=f'source == "{path.split("/")[-1]}"')  # idempotent re-ingest
    store.add_documents(chunks)
    return len(chunks)
```

**Best practices baked in here:**

- **Structure-aware splitting with overlap** keeps each chunk a coherent thought and
  prevents key sentences being cut at a boundary.
- **One normalization point** fills the modeled fields identically for every file, so
  mixed sources coexist in one collection. The schema describes chunks, not file
  formats, so PDFs and (later) CSVs live together.
- **Delete-before-add by `source`** makes re-uploading a file idempotent instead of
  creating duplicates (Milvus auto-ids do not dedup on their own).
- **Cache the store** with `lru_cache` so you reuse one connection instead of opening a
  new one per request.

---

## Step 4 — Retrieve and generate (the RAG chain)

Embed the question with the same model, fetch the closest chunks (optionally filtered),
hand them to the LLM with a grounding instruction.

```python
# rag.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from models import get_llm
from ingest import get_store

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a legal research assistant. Answer using ONLY the context below. "
     "Cite sources as [Source N]. If the answer is not in the context, say you do not "
     "have enough information. Do not use outside knowledge.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

def _format(docs):
    return "\n\n".join(
        f"[Source {i+1} | {d.metadata.get('source')}] {d.page_content}"
        for i, d in enumerate(docs)
    )

def answer(question, expr=None, k=5):
    store = get_store()
    # filter at the database BEFORE the vector comparison (faster, more correct)
    retrieve = RunnableLambda(lambda q: store.similarity_search(q, k=k, expr=expr))
    chain = (
        {"context": retrieve | _format, "question": RunnablePassthrough()}
        | PROMPT | get_llm() | StrOutputParser()
    )
    docs = retrieve.invoke(question)
    if not docs:
        return {"answer": "No relevant information found.", "sources": []}
    return {
        "answer": chain.invoke(question),
        "sources": sorted({d.metadata.get("source") for d in docs}),
    }
```

Usage:

```python
answer("penalty for cheque dishonour")
answer("bail in economic offences", expr="jurisdiction == 'India' && year >= 2015")
```

**Best practice:** the "answer only from context, else say you do not know" instruction
is the main defence against hallucination. The no-document short-circuit stops the model
improvising when retrieval finds nothing.

---

## Step 5 — Serve (minimal FastAPI)

```python
# main.py
from fastapi import FastAPI
from ingest import ingest_file
from rag import answer

app = FastAPI()

@app.post("/ingest")
def ingest(path: str, document_type: str, jurisdiction: str = None, year: int = None):
    n = ingest_file(path, document_type, jurisdiction=jurisdiction, year=year)
    return {"status": "ok", "chunks_indexed": n}

@app.post("/ask")
def ask(question: str, expr: str = None, k: int = 5):
    return answer(question, expr=expr, k=k)
```

That is a working RAG service: provision once, `/ingest` your files, `/ask` questions.

---

## Upgrades you add later (only when needed)

The baseline above is solid. Improve the **retrieval** step when quality demands it,
because the answer can only be as good as the chunks you fetch:

- **Hybrid search** (semantic + keyword) so exact identifiers like "Section 138" match
  alongside paraphrases. Add a sparse function to the store:
  ```python
  from langchain_milvus import Milvus, BM25BuiltInFunction
  store = Milvus(embedding_function=get_embeddings(),
                 builtin_function=BM25BuiltInFunction(),
                 vector_field=["vector", "sparse"], collection_name=config.COLLECTION,
                 connection_args={"uri": config.MILVUS_URI})
  ```
- **Reranking**: fetch a wide net (k=30), then a cross-encoder picks the best few. Wrap
  the retriever in `ContextualCompressionRetriever` with a `CrossEncoderReranker`.
- **Query rewriting** for chat: rewrite a follow-up into a standalone question before
  retrieving.

Add these one at a time, measure, keep what helps.

---

## Best practices (the short list)

1. **Pin the data-plane versions** (`langchain-milvus` + `pymilvus`) to a tested pair.
   Latest plus latest is not a guarantee.
2. **Provision the schema once**, as a migration, not on app startup.
3. **Keep the schema flexible**: dynamic field on, domain fields nullable.
4. **One normalization function** fills the metadata contract for every source.
5. **Same embedding model** for ingest and query, always.
6. **Filter at the database** with `expr`, not in Python after fetching.
7. **Idempotent ingest**: delete by `source`, then add.
8. **Cache** the embedding client, the LLM, and the vector store.
9. **Ground the prompt**: answer only from context, cite sources, admit when unknown.
10. **Use `consistency_level="Strong"`** when you read right after a write.

---

## Pitfalls and concerns (symptom → why → fix)

**`ConnectionNotExistException: should create connection first` at startup.**
Why: `pymilvus` 2.6.10+ broke an internal API that `langchain-milvus` 0.3.3 still uses.
Fix: pin `pymilvus>=2.6.0,<2.6.10`.

**`DataNotMatchException: Insert missed an field X`.**
Why: the collection was created with strict columns and a later document lacks field X.
Fix: recreate with `enable_dynamic_field=True` and nullable fields. A schema is frozen
at creation, so changing it means rebuilding the collection.

**Search returns nothing.**
Why/fix, in order: nothing ingested yet (the collection is created on the first insert,
not at object construction); custom field names not passed to the wrapper
(`primary_field`/`text_field`/`vector_field`); index metric does not match the search;
a too-tight `expr`; weak consistency on a read right after a write (use Strong).

**Re-uploading a file duplicates chunks.**
Why: auto-ids do not dedup. Fix: delete by `source` before adding (shown in Step 3).

**Changed the embedding model and results broke.**
Why: vectors from different models are not comparable, and dimensions differ. Fix: a new
collection, re-embed everything from source. A different embedding model is the one
thing that forces a separate collection.

**Documents of different types have different metadata.**
Why: letting loaders define metadata makes keys vary by file type. Fix: the
normalization contract plus dynamic field. One schema holds all types.

**Retrieval brings semantically close but wrong-identifier chunks.**
Why: dense embeddings match meaning, not exact tokens. Fix: add hybrid (BM25) search.

**Answers hallucinate despite retrieval.**
Why: the prompt allowed outside knowledge, or retrieval missed the chunk. Fix: the
grounding instruction and no-context short-circuit; verify the right chunk was actually
retrieved.

**Treat retrieved text as untrusted.**
Concern: a stored document can contain prompt-injection text. Keep the instruction that
context is data to answer from, not commands to follow, and enforce access control by
filtering on a permission or tenant field.
