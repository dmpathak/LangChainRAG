from functools import lru_cache
from langchain.chat_models import init_chat_model
from config import LLM_MODEL, EMBEDDING_MODEL, OPENROUTER_API_KEY, LLM_MODEL_PROVIDER, EMBEDDING_URL
from langchain_openai import OpenAIEmbeddings


@lru_cache(maxsize=1)
def get_llm_model():
    return init_chat_model(  # init_chat_model ==> this is for chat model only
        model=LLM_MODEL,
        model_provider=LLM_MODEL_PROVIDER,
        temperature=0.2,
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=EMBEDDING_URL,
        check_embedding_ctx_length=False,
        model_kwargs={
            "encoding_format": "float"
        }
    )