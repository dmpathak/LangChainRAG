"""
Minimal retrieval flow: plan once, then use one Milvus route.
"""

from functools import lru_cache

from config import FINAL_CONTEXT_DOCUMENTS, MIN_DENSE_SCORE
from services.query_planner import get_query_planner
from services.search_filters import CombinedFilters
from services.structured_search import StructuredSearch
from services.vector_db import get_vector_store


class RetrievalService:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.query_planner = get_query_planner()
        self.structured_search = StructuredSearch(self.vector_store)

    def add_documents(self, documents):
        return self.vector_store.add_documents(documents)

    def search(self, query, filters, top_k=FINAL_CONTEXT_DOCUMENTS):
        available_fields = tuple(self.vector_store.available_fields())
        plan = self.query_planner.create_plan(query, available_fields)

        combined_filters = CombinedFilters(filters, plan.filters)

        if plan.mode == "structured":
            documents = self.structured_search.execute(plan, combined_filters, top_k)
            if documents is not None:
                return self._attach_plan(documents, plan)

        scored_documents = self.vector_store.similarity_search_with_score(
            query=plan.semantic_query or query,
            top_k=top_k,
            filter_expression=combined_filters.to_milvus_expression(),
        )

        documents = []
        for document, score in scored_documents:
            if score >= MIN_DENSE_SCORE:
                documents.append(document)

        return self._attach_plan(documents, plan)

    @staticmethod
    def _attach_plan(documents, plan):
        plan_data = plan.model_dump()
        for document in documents:
            document.metadata["search_plan"] = plan_data
        return documents


@lru_cache(maxsize=1)
def get_retrieval_service():
    return RetrievalService()
