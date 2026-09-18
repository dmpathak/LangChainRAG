import os

from dotenv import load_dotenv

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME", "MyLangChainCollection")
DOCUMENT_COLLECTION_NAME = os.getenv("DOCUMENT_COLLECTION_NAME", "documents")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
APP_API_KEY = os.getenv("APP_API_KEY", "")

LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"
LLM_MODEL_PROVIDER = "nvidia"
EMBEDDING_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
EMBEDDING_DOCUMENT_INPUT_TYPE = "passage"
EMBEDDING_QUERY_INPUT_TYPE = "query"

FINAL_CONTEXT_DOCUMENTS = 8
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.0"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "30"))
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100
