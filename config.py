import os

from dotenv import load_dotenv

load_dotenv()

# Milvus
MILVUS_URI = "http://localhost:19530"
collection_name = "MyLangChainCollection"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# LLM
LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"
LLM_MODEL_PROVIDER = "nvidia"

# Embeddings
EMBEDDING_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "inclusionai/ling-3.0-flash-fin:free"
EMBEDDING_MODEL_PROVIDER = "openrouter"
EMBEDDING_DOCUMENT_INPUT_TYPE = "passage"
EMBEDDING_QUERY_INPUT_TYPE = "query"

# Retrieval
FINAL_CONTEXT_DOCUMENTS = 8
MIN_DENSE_SCORE = 0.20
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# HNSW
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100
