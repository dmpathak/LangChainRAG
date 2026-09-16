"""
Retrieve context using the user's complete question.
"""

from functools import lru_cache

from config import FINAL_CONTEXT_DOCUMENTS, collection_name as default_collection_name
from services.vector_db import get_vector_store


class RetrievalService:
    def __init__(self, collection_name):
        self.vector_store = get_vector_store(collection_name)

    def add_documents(self, documents):
        return self.vector_store.add_documents(documents)

    def search(self, query, top_k=FINAL_CONTEXT_DOCUMENTS):
        # Preserve the user's purpose and preferences in the embedding query.
        scored_documents = self.vector_store.similarity_search_with_score(
            query=query,
            top_k=top_k,
        )
        return [document for document, score in scored_documents]


@lru_cache(maxsize=32)
def get_retrieval_service(collection_name=default_collection_name):
    return RetrievalService(collection_name=collection_name)
