import json
from functools import lru_cache

from langchain_core.documents import Document
from langchain_milvus import Milvus
from config import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    MILVUS_URI,
    collection_name,
)
from services.AI_models import get_embedding_model


class VectorStore:
    def __init__(self):
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
        self._available_fields = set()

    def add_documents(self, documents):
        product_id_set = set()
        for document in documents:
            product_id = document.metadata.get("product_id")
            if product_id is not None:
                product_id_set.add(str(product_id))

        product_ids = sorted(product_id_set)
        if product_ids and self.vector_store.col is not None:
            expression = f"product_id in {json.dumps(product_ids)}"
            self.vector_store.col.delete(expr=expression)

        for document in documents:
            self._available_fields.update(document.metadata)
        return self.vector_store.add_documents(documents)

    def similarity_search_with_score(self, query, top_k, filter_expression=None):
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": max(HNSW_EF_SEARCH, top_k)},
        }
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            param=search_params,
            expr=filter_expression,
        )

    def all_documents(self, filter_expression=None):
        if self.vector_store.col is None:
            return []

        iterator = self.vector_store.col.query_iterator(
            expr=filter_expression or "pk >= 0",
            output_fields=["*"],
            batch_size=1000,
        )

        documents = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break

                for row in batch:
                    page_content = row.pop("text", "")
                    row.pop("vector", None)
                    documents.append(Document(page_content=page_content, metadata=row))
        finally:
            iterator.close()
        return documents

    def available_fields(self):
        if self._available_fields:
            return sorted(self._available_fields)
        if self.vector_store.col is None:
            return []

        rows = self.vector_store.col.query(
            expr="pk >= 0",
            output_fields=["*"],
            limit=1,
        )
        if rows:
            self._available_fields.update(rows[0])
        self._available_fields.difference_update({"pk", "text", "vector"})
        return sorted(self._available_fields)


@lru_cache(maxsize=1)
def get_vector_store():
    return VectorStore()
