"""Retrieval flow: vector candidates -> hybrid scores -> reranking -> filter."""

import re
from functools import lru_cache

from app.config import (
    FINAL_CONTEXT_DOCUMENTS,
    KEYWORD_WEIGHT,
    MIN_SIMILARITY_SCORE,
    RERANK_ENABLED,
    RETRIEVAL_MODE,
    VECTOR_WEIGHT,
    DEFAULT_COLLECTION_NAME,
)
from app.rag_services.retrieval.vector_db import get_vector_store


class RetrievalService:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.vector_store = get_vector_store(collection_name)

    # Document management used by upload/list/delete operations.
    def add_documents(self, documents):
        return self.vector_store.add_documents(documents)

    def has_file_hash(self, file_hash):
        return self.vector_store.has_file_hash(file_hash)

    def delete_file(self, file_name):
        return self.vector_store.delete_by_file_name(file_name)

    def list_documents(self):
        return self.vector_store.list_documents()

    def get_all_documents(self, file_name=None):
        return self.vector_store.get_all_documents(file_name=file_name)

    # Main search path used by /search.
    def search_with_scores(
        self,
        query,
        top_k=FINAL_CONTEXT_DOCUMENTS,
        min_score=MIN_SIMILARITY_SCORE,
        file_name=None,
    ):
        if RETRIEVAL_MODE == "hybrid":
            results = self._hybrid_search(query, top_k, file_name)
        else:
            results = self._vector_search(query, top_k, file_name)

        return [(document, score) for document, score in results if score >= min_score]

    def search(self, query, top_k=FINAL_CONTEXT_DOCUMENTS):
        return [
            document for document, score in self.search_with_scores(query, top_k)
        ]

    def _vector_search(self, query, top_k, file_name=None):
        return self.vector_store.similarity_search_with_score(
            query=query,
            top_k=top_k,
            file_name=file_name,
        )

    def _hybrid_search(self, query, top_k, file_name=None):
        candidate_k = min(max(top_k * 3, top_k), 100)
        vector_results = self.vector_store.similarity_search_with_score(
            query=query,
            top_k=candidate_k,
            file_name=file_name,
        )
        all_documents = self.vector_store.get_all_documents(file_name=file_name)
        if not all_documents:
            return vector_results[:top_k]

        query_terms = _terms(query)
        candidates = self._combine_candidates(vector_results, all_documents, query_terms)
        ranked = []
        for candidate in candidates.values():
            document = candidate["document"]
            keyword_score = _keyword_score(query, document.page_content, query_terms)
            hybrid_score = _weighted_score(candidate["vector_score"], keyword_score)
            final_score = hybrid_score
            if RERANK_ENABLED:
                final_score = 0.6 * hybrid_score + 0.4 * keyword_score
            ranked.append((document, final_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _combine_candidates(vector_results, all_documents, query_terms):
        scores = [score for _, score in vector_results]
        minimum = min(scores, default=0.0)
        score_range = max(scores, default=1.0) - minimum or 1.0
        candidates = {}

        for document, score in vector_results:
            candidates[_document_key(document)] = {
                "document": document,
                "vector_score": (score - minimum) / score_range,
            }

        for document in all_documents:
            if _keyword_score(" ".join(query_terms), document.page_content, query_terms) <= 0:
                continue
            candidates.setdefault(
                _document_key(document),
                {"document": document, "vector_score": 0.0},
            )
        return candidates


@lru_cache(maxsize=32)
def get_retrieval_service(collection_name=DEFAULT_COLLECTION_NAME):
    return RetrievalService(collection_name=collection_name)


STOP_WORDS = {
    "a", "an", "and", "are", "about", "does", "how", "is", "of", "the",
    "this", "what", "which", "with", "from", "in", "to", "for", "on",
}


def _terms(text):
    return {
        term for term in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if term not in STOP_WORDS
    }


def _keyword_score(query, document_text, query_terms):
    if not query_terms:
        return 0.0
    overlap = len(query_terms & _terms(document_text)) / len(query_terms)
    phrase_bonus = 0.2 if query.lower().strip() in document_text.lower() else 0.0
    return min(overlap + phrase_bonus, 1.0)


def _weighted_score(vector_score, keyword_score):
    total_weight = VECTOR_WEIGHT + KEYWORD_WEIGHT
    if total_weight <= 0:
        return 0.0
    return (VECTOR_WEIGHT * vector_score + KEYWORD_WEIGHT * keyword_score) / total_weight


def _document_key(document):
    return document.metadata.get("document_id") or document.page_content
