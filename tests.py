from pymilvus import connections
from langchain_milvus import Milvus

from services.utils import get_embedding_model

connections.connect(
    alias="MyLangChainCollection",
    db_name="MyLangChainCollection",
    uri="http://localhost:19530"
)

embedding_model = get_embedding_model()

print("Embedding ready")


vector_store = Milvus(
    embedding_function=embedding_model,
    collection_name="MyLangChainCollection",
    connection_args={
        "uri": "http://localhost:19530"
    },
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE"
    },
)

print("SUCCESS", vector_store.__dict__)
