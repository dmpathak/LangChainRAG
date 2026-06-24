from functools import lru_cache

from pymilvus import connections
from langchain_milvus import Milvus
from config import milvus_uri
from services.utils import get_embedding_model

embedding_model = get_embedding_model()


# responsible for Milvus connection.
class VectorStore:
    def __init__(self):
        self.vector_store = Milvus(
            embedding_function=embedding_model,
            collection_name="MyLangChainCollection",
            connection_args={
                "uri": milvus_uri
            },
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE"
            },
        )

    def add_documents(self, documents):
        return self.vector_store.add_documents(documents)

    def similarity_search(self, query, top_k=5):
        return self.vector_store.similarity_search(query=query, k=top_k)


@lru_cache(maxsize=1)
def get_vector_store():
    return VectorStore()
