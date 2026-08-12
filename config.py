"""Simple configuration file."""
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

# Milvus
MILVUS_URI = "http://localhost:19530"
milvus_token = ""
database_name = "ProductSearchRAG"
collection_name = "MyLangChainCollection"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# LLM
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "nvidia/nemotron-3.5-lightning:free"
LLM_MODEL_PROVIDER = "openrouter"

# embeddings
EMBEDDING_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
EMBEDDING_MODEL_PROVIDER = "openrouter"

# # Reranking (Jina). Leave the key empty to run dense-only retrieval.
# rerank_enabled: bool = True
# jina_api_key= ""
# rerank_model = "jina-reranker-v2-base-multilingual"

UPLOAD_DIR = "data/uploads"
