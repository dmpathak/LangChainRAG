from langchain.chat_models import init_chat_model
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_URL,
    LLM_MODEL,
    LLM_MODEL_PROVIDER,
    NVIDIA_API_KEY,
    OPENROUTER_API_KEY,
)

from services.openrouter_embeddings import OpenRouterEmbeddings


def get_llm_model():
    return init_chat_model(  # init_chat_model ==> this is for chat model only
        model=LLM_MODEL,
        model_provider=LLM_MODEL_PROVIDER,
        api_key=NVIDIA_API_KEY,
        temperature=0,
    )


def get_embedding_model():
    return OpenRouterEmbeddings(
        api_key=OPENROUTER_API_KEY,
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_URL,
    )
