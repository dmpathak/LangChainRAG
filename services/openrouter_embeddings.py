from typing import List

from openai import OpenAI
from langchain_core.embeddings import Embeddings


class OpenRouterEmbeddings(Embeddings):
    """
    LangChain-compatible embedding implementation for OpenRouter.

    Why this wrapper?
    -----------------
    `langchain_openai.OpenAIEmbeddings` currently has a compatibility issue
    with some OpenRouter embedding providers (e.g. NVIDIA Nemotron), where
    requests are sent with `encoding_format="base64"` even when
    `encoding_format="float"` is configured.

    NVIDIA's embedding endpoint only accepts `encoding_format="float"` (or
    omits the parameter), causing requests through `OpenAIEmbeddings` to fail.

    This wrapper calls the OpenAI client directly and explicitly sends
    `encoding_format="float"` to ensure compatibility while still implementing
    LangChain's `Embeddings` interface.
    """

    def __init__(
            self,
            api_key: str,
            model: str,
            base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Explicitly set encoding_format="float" because NVIDIA embedding
        # models on OpenRouter do not support the default base64 encoding.
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        # Keep query embeddings consistent with document embeddings.
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )

        return response.data[0].embedding
