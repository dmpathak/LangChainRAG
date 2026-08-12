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
            batch_size: int = 256,
    ):
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")

        self.model = model
        self.batch_size = batch_size

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Explicitly set encoding_format="float" because NVIDIA embedding
        # models on OpenRouter do not support the default base64 encoding.
        embeddings: List[List[float]] = []

        # OpenRouter accepts at most 256 inputs in one embeddings request.
        # LangChain vector stores may pass the entire document set here, so
        # batch at this boundary instead of relying on each caller to do it.
        for start in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(
                model=self.model,
                input=texts[start:start + self.batch_size],
                encoding_format="float",
            )

            # The API exposes an index specifically for restoring input order.
            batch_data = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in batch_data)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # Keep query embeddings consistent with document embeddings.
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )

        return response.data[0].embedding
