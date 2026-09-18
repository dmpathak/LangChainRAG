import json
from functools import lru_cache

from langchain_core.documents import Document
from langchain_milvus import Milvus

from app.config import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    MILVUS_URI,
    DEFAULT_COLLECTION_NAME,
)
from app.rag_services.llm.AI_models import get_embedding_model


class VectorStore:
    def __init__(self, collection_name):
        self.vector_store = Milvus(
            embedding_function=get_embedding_model(),
            collection_name=collection_name,
            enable_dynamic_field=True,
            connection_args={"uri": MILVUS_URI},
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": HNSW_M,
                    "efConstruction": HNSW_EF_CONSTRUCTION,
                },
            },
        )

    def add_documents(self, documents):
        product_ids = sorted({
            str(document.metadata["product_id"])
            for document in documents
            if document.metadata.get("product_id") is not None
        })
        if product_ids and self.vector_store.col is not None:
            self.vector_store.col.delete(
                expr=f"product_id in {json.dumps(product_ids)}"
            )
        return self.vector_store.add_documents(documents)

    def similarity_search_with_score(self, query, top_k, file_name=None):
        params = {
            "metric_type": "COSINE",
            "params": {"ef": max(HNSW_EF_SEARCH, top_k)},
        }
        kwargs = {"query": query, "k": top_k, "param": params}
        if file_name:
            kwargs["expr"] = f"file_name == {json.dumps(file_name)}"
        return self.vector_store.similarity_search_with_score(**kwargs)

    def has_file_hash(self, file_hash):
        if self.vector_store.col is None:
            return False
        try:
            rows = self.vector_store.col.query(
                expr=f"file_hash == {json.dumps(file_hash)}",
                output_fields=["file_hash"],
                limit=1,
            )
        except Exception:
            return False
        return bool(rows)

    def delete_by_file_name(self, file_name):
        if self.vector_store.col is None:
            return 0
        result = self.vector_store.col.delete(
            expr=f"file_name == {json.dumps(file_name)}"
        )
        self.vector_store.col.flush()
        return getattr(result, "delete_count", 0)

    def list_documents(self):
        if self.vector_store.col is None:
            return []
        rows = self.vector_store.col.query(
            expr="",
            output_fields=["file_name", "file_hash"],
            limit=10000,
        )
        documents = {}
        for row in rows:
            file_name = row.get("file_name") or "unknown"
            summary = documents.setdefault(
                file_name,
                {"file_name": file_name, "file_hash": row.get("file_hash"), "chunks": 0},
            )
            summary["chunks"] += 1
        return list(documents.values())

    def get_all_documents(self, file_name=None):
        if self.vector_store.col is None:
            return []
        expression = f"file_name == {json.dumps(file_name)}" if file_name else ""
        rows = self.vector_store.col.query(
            expr=expression,
            output_fields=["*"],
            limit=10000,
        )
        documents = []
        for row in rows:
            page_content = row.pop("text", "")
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"vector", "pk", "id"}
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents


@lru_cache(maxsize=32)
def get_vector_store(collection_name=DEFAULT_COLLECTION_NAME):
    return VectorStore(collection_name=collection_name)
