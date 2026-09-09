"""Execute validated LLM plans over the complete local catalogue."""

from langchain_core.documents import Document


class StructuredSearch:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def execute(self, plan, filters, top_k):
        documents = self.vector_store.all_documents(filters.to_milvus_expression())
        operation = plan.operation

        if operation == "count":
            return [self._calculation_document(plan, len(documents), len(documents))]

        if operation in {"average", "sum"}:
            values = self._numeric_values(documents, plan.field)
            if not values:
                return []

            if operation == "sum":
                result = sum(values)
            else:
                result = sum(values) / len(values)
            return [self._calculation_document(plan, result, len(values))]

        if operation in {"minimum", "maximum"}:
            reverse = operation == "maximum"
            ranked = self._ranked_documents(documents, plan.field, reverse)
            if not ranked:
                return []

            winning_value = ranked[0][0]
            results = []
            for value, document in ranked:
                if value == winning_value:
                    results.append(document)
            return results[:top_k]

        if operation == "sort":
            reverse = plan.sort_direction == "descending"
            ranked = self._ranked_documents(documents, plan.field, reverse)
            limit = min(plan.limit, top_k)
            results = []
            for value, document in ranked[:limit]:
                results.append(document)
            return results

        if operation == "filter":
            limit = min(plan.limit, top_k)
            return documents[:limit]

        return None

    @staticmethod
    def _ranked_documents(documents, field, reverse):
        if not field:
            return []

        field = StructuredSearch._field_name(field)
        ranked = []

        for document in documents:
            value = document.metadata.get(field)
            if value is None:
                continue
            try:
                sortable_value = (0, float(value))
            except (TypeError, ValueError):
                sortable_value = (1, str(value).lower())
            ranked.append((sortable_value, document))
        ranked.sort(key=lambda result: result[0], reverse=reverse)
        return ranked

    @staticmethod
    def _numeric_values(documents, field):
        if not field:
            return []

        field = StructuredSearch._field_name(field)
        values = []
        for document in documents:
            try:
                values.append(float(document.metadata[field]))
            except (KeyError, TypeError, ValueError):
                continue
        return values

    @staticmethod
    def _field_name(value):
        return "_".join(str(value).strip().lower().replace("-", " ").split())

    @staticmethod
    def _calculation_document(plan, value, matched_rows):
        field_label = ""
        if plan.field:
            field_label = f" for {plan.field}"

        document_id = f"calculation:{plan.operation}:{plan.field or 'all'}"
        content = (
            f"Exact structured calculation\n"
            f"operation: {plan.operation}{field_label}\n"
            f"result: {value}\n"
            f"rows_used: {matched_rows}"
        )
        return Document(
            page_content=content,
            metadata={
                "document_id": document_id,
                "file_name": "Calculated from indexed products",
                "operation": plan.operation,
                "field": plan.field,
            },
        )
