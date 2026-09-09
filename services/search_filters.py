"""Safe metadata filters shared by semantic and structured retrieval."""

from dataclasses import dataclass
import re


@dataclass
class SearchFilters:
    brand: str | None = None
    category: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = None
    in_stock: bool | None = None

    def to_milvus_expression(self):
        parts = []
        if self.brand:
            parts.append(f'brand == "{self._escape(self.brand)}"')
        if self.category:
            parts.append(f'category == "{self._escape(self.category)}"')
        if self.min_price is not None:
            parts.append(f"price >= {self.min_price}")
        if self.max_price is not None:
            parts.append(f"price <= {self.max_price}")
        if self.min_rating is not None:
            parts.append(f"rating >= {self.min_rating}")
        if self.in_stock:
            parts.append("stock > 0")
        return " and ".join(parts) or None

    @staticmethod
    def _escape(value):
        return value.replace("\\", "\\\\").replace('"', '\\"')


class CombinedFilters:
    """Combine explicit UI filters with conditions selected by the planner."""

    SAFE_FIELD = re.compile(r"^[a-z_][a-z0-9_]*$")

    def __init__(self, explicit_filters, conditions):
        self.explicit_filters = explicit_filters
        self.conditions = conditions

    def to_milvus_expression(self):
        parts = []
        explicit_expression = self.explicit_filters.to_milvus_expression()
        if explicit_expression:
            parts.append(explicit_expression)

        for condition in self.conditions:
            expression = self._condition_expression(condition)
            if expression:
                parts.append(expression)

        wrapped_parts = []
        for part in parts:
            wrapped_parts.append(f"({part})")
        return " and ".join(wrapped_parts) or None

    @classmethod
    def _condition_expression(cls, condition):
        field = cls._field_name(condition.field)
        if not field:
            return None

        if condition.operator == "contains":
            value = SearchFilters._escape(str(condition.value))
            return f'{field} like "%{value}%"'

        operators = {
            "eq": "==",
            "ne": "!=",
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
        }
        value = condition.value
        if isinstance(value, str):
            value = f'"{SearchFilters._escape(value)}"'
        return f"{field} {operators[condition.operator]} {value}"

    @classmethod
    def _field_name(cls, value):
        field = "_".join(str(value).strip().lower().replace("-", " ").split())
        if cls.SAFE_FIELD.fullmatch(field):
            return field
        return None
