from openai import OpenAI
from langchain_core.embeddings import Embeddings


class OpenRouterEmbeddings(Embeddings):
    """OpenRouter embeddings that always request float vectors."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        batch_size: int = 256,
        document_input_type: str | None = None,
        query_input_type: str | None = None,
    ):
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")

        self.model = model
        self.batch_size = batch_size
        self.document_input_type = document_input_type
        self.query_input_type = query_input_type

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
                extra_body=self._input_type(self.document_input_type),
            )

            batch_data = sorted(response.data, key=lambda item: item.index)
            for item in batch_data:
                embeddings.append(item.embedding)

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
            extra_body=self._input_type(self.query_input_type),
        )

        return response.data[0].embedding

    @staticmethod
    def _input_type(value: str | None) -> dict:
        if value:
            return {"input_type": value}
        return {}
