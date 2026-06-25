from functools import lru_cache
from langchain.chat_models import init_chat_model
from config import LLM_MODEL, EMBEDDING_MODEL, OPENROUTER_API_KEY, LLM_MODEL_PROVIDER, EMBEDDING_URL
from langchain_openai import OpenAIEmbeddings


@lru_cache(maxsize=1)
def get_llm_model():
    return init_chat_model(  # init_chat_model ==> this is for chat model only
        model=LLM_MODEL,
        model_provider=LLM_MODEL_PROVIDER,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=EMBEDDING_URL,
        model_kwargs={"encoding_format": "float"}  # Only set if your embedding response format is wrong without it.
    )
# ********** What encoding_format="float" actually guarantees *************
# It says => “Give me raw float vector output suitable for vector DBs”
#
# So your pipeline (FAISS, Milvus, Weaviate, etc.) always gets:
# e.g: [0.0123, -0.98, 0.334, ...]
#
# instead of encoded or wrapped formats.
