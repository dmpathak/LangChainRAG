"""Retrieval flow used by the FastAPI search endpoint.

The search path is intentionally kept in this file:

    query
      -> vector candidates from Milvus
      -> optional keyword candidates
      -> combine scores
      -> optional reranking
      -> minimum score filter
      -> documents sent to the LLM

The service is created once per Milvus collection. The collection selected in
the UI is passed into ``get_retrieval_service`` and is never mixed with another
collection.
"""

import re
from functools import lru_cache

from config import (
    FINAL_CONTEXT_DOCUMENTS,
    KEYWORD_WEIGHT,
    MIN_SIMILARITY_SCORE,
    RERANK_ENABLED,
    RETRIEVAL_MODE,
    VECTOR_WEIGHT,
    collection_name as default_collection_name,
)
from services.vector_db import get_vector_store


class RetrievalService:
    """Add documents and retrieve relevant documents for one collection."""

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.vector_store = get_vector_store(collection_name)

    # ---------- Document management ----------

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

    # ---------- Main retrieval flow ----------

    def search_with_scores(
        self,
        query,
        top_k=FINAL_CONTEXT_DOCUMENTS,
        min_score=MIN_SIMILARITY_SCORE,
        file_name=None,
    ):
        """Return the best documents and their final retrieval scores."""
        if RETRIEVAL_MODE == "hybrid":
            results = self._hybrid_search(query, top_k, file_name)
        else:
            results = self._vector_search(query, top_k, file_name)

        # This is the last step. Low-confidence documents never reach the LLM.
        return [
            (document, score)
            for document, score in results
            if score >= min_score
        ]

    def search(self, query, top_k=FINAL_CONTEXT_DOCUMENTS):
        """Compatibility helper that returns documents without scores."""
        return [
            document
            for document, score in self.search_with_scores(query, top_k)
        ]

    # ---------- Retrieval strategies ----------

    def _vector_search(self, query, top_k, file_name=None):
        """Current baseline: semantic vector search only."""
        return self.vector_store.similarity_search_with_score(
            query=query,
            top_k=top_k,
            file_name=file_name,
        )

    def _hybrid_search(self, query, top_k, file_name=None):
        """Combine semantic search and simple keyword matching."""
        # Retrieve extra vector candidates because the final result is only
        # selected after vector and keyword scores are combined.
        candidate_k = min(max(top_k * 3, top_k), 100)
        vector_results = self.vector_store.similarity_search_with_score(
            query=query,
            top_k=candidate_k,
            file_name=file_name,
        )

        # Keyword search is implemented over stored rows. This is simple and
        # useful for exact product IDs, names, and technical terms.
        all_documents = self.vector_store.get_all_documents(file_name=file_name)
        if not all_documents:
            return vector_results[:top_k]

        query_terms = _terms(query)
        candidates = self._combine_candidates(vector_results, all_documents, query_terms)

        scored_candidates = []
        for candidate in candidates.values():
            document = candidate["document"]
            keyword_score = _keyword_score(query, document.page_content, query_terms)
            hybrid_score = _weighted_score(
                candidate["vector_score"],
                keyword_score,
            )

            # This is a small local reranker. It does not call another model;
            # it gives exact query terms and phrases extra importance.
            final_score = hybrid_score
            if RERANK_ENABLED:
                final_score = (0.6 * hybrid_score) + (0.4 * keyword_score)

            scored_candidates.append((document, final_score))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        return scored_candidates[:top_k]

    @staticmethod
    def _combine_candidates(vector_results, all_documents, query_terms):
        """Create one candidate set from vector and keyword retrieval."""
        vector_scores = [score for _, score in vector_results]
        minimum = min(vector_scores, default=0.0)
        maximum = max(vector_scores, default=1.0)
        score_range = maximum - minimum or 1.0

        candidates = {}
        for document, score in vector_results:
            candidates[_document_key(document)] = {
                "document": document,
                "vector_score": (score - minimum) / score_range,
            }

        for document in all_documents:
            keyword_score = _keyword_score(
                " ".join(query_terms),
                document.page_content,
                query_terms,
            )
            if keyword_score <= 0:
                continue
            candidates.setdefault(
                _document_key(document),
                {"document": document, "vector_score": 0.0},
            )

        return candidates


@lru_cache(maxsize=32)
def get_retrieval_service(collection_name=default_collection_name):
    """Return one cached retrieval service for the selected collection."""
    return RetrievalService(collection_name=collection_name)


# ---------- Small keyword scoring helpers ----------

STOP_WORDS = {
    "a", "an", "and", "are", "about", "does", "how", "is", "of", "the",
    "this", "what", "which", "with", "from", "in", "to", "for", "on",
}


def _terms(text):
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if term not in STOP_WORDS
    }


def _keyword_score(query, document_text, query_terms):
    if not query_terms:
        return 0.0

    document_terms = _terms(document_text)
    overlap = len(query_terms & document_terms) / len(query_terms)
    phrase_bonus = 0.2 if query.lower().strip() in document_text.lower() else 0.0
    return min(overlap + phrase_bonus, 1.0)


def _weighted_score(vector_score, keyword_score):
    total_weight = VECTOR_WEIGHT + KEYWORD_WEIGHT
    if total_weight <= 0:
        return 0.0
    return (
        VECTOR_WEIGHT * vector_score + KEYWORD_WEIGHT * keyword_score
    ) / total_weight


def _document_key(document):
    return document.metadata.get("document_id") or document.page_content
