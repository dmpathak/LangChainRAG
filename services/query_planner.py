"""Use the LLM as a planner; it chooses retrieval but does not calculate results."""

from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from services.AI_models import get_llm_model


class FilterCondition(BaseModel):
    field: str = Field(description="Normalized metadata field, such as brand or price")
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains"] = "eq"
    value: str | float | int | bool


class SearchPlan(BaseModel):
    mode: Literal["semantic", "structured"] = "semantic"
    operation: Literal[
        "search", "filter", "sort", "minimum", "maximum", "count", "average", "sum"
    ] = "search"
    field: str | None = Field(
        default=None,
        description="Field used for sorting or calculation, such as price or rating",
    )
    sort_direction: Literal["ascending", "descending"] = "ascending"
    filters: list[FilterCondition] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=20)
    semantic_query: str | None = Field(
        default=None,
        description="Clean wording used only for semantic retrieval",
    )


PLANNER_PROMPT = """
You plan searches over uploaded documents and product rows. Return a search plan,
not an answer.

Choose mode "structured" when correctness requires examining metadata: exact
filtering, numeric comparisons, sorting, minimum, maximum, count, average, or sum.
Choose mode "semantic" for meaning, descriptions, features, recommendations,
text comparisons, or exact product/SKU lookup. Semantic mode uses dense and
keyword retrieval together.

Metadata names are lowercase snake_case. Use only fields supplied with the user
query. Use operation "filter" for filtering without an aggregate and "search"
with semantic mode. Put the calculation or sort field in `field`. Extract every
clear constraint into `filters`; use "contains" for partial category/name matches.
Do not invent constraints, fields, or values.

Important examples of planning behavior:
- "highest price from all" means structured + maximum + field price + no filters.
- "lowest rating overall" means structured + minimum + field rating + no filters.
- "average price of Samsung products" means structured + average + field price,
  with only the explicitly requested Samsung filter.
- Words such as "all", "overall", and "entire catalogue" mean filters must be empty
  unless the user also states a real constraint.
""".strip()


class QueryPlanner:
    def __init__(self, model=None):
        model = model or get_llm_model()
        self.planner = model.with_structured_output(SearchPlan)

    def create_plan(self, query: str, available_fields: tuple[str, ...] = ()):
        try:
            messages = [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(
                    content=(
                        f"Available metadata fields: {', '.join(available_fields)}\n"
                        f"User query: {query}"
                    )
                ),
            ]
            return self.planner.invoke(messages)
        except Exception as exc:
            print(f"Query planner fallback: {exc}")
            return SearchPlan(semantic_query=query)


@lru_cache(maxsize=1)
def get_query_planner():
    return QueryPlanner()
