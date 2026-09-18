"""Deterministic answers for simple numeric questions over tabular data."""

import re


def answer_price_question(query, documents):
    normalized_query = query.lower()
    is_lowest = any(word in normalized_query for word in ("lowest", "cheapest", "minimum"))
    is_highest = any(word in normalized_query for word in ("highest", "most expensive", "maximum"))
    if not is_price_question(query):
        return None

    products = []
    for document in documents:
        metadata = document.metadata
        price = _number(metadata.get("price"))
        if price is None:
            continue
        product_name = next(
            (
                metadata.get(field)
                for field in ("product_name", "name", "title", "product")
                if metadata.get(field)
            ),
            "Unknown product",
        )
        products.append((price, str(product_name), document))

    if not products:
        return None

    selected = min(products, key=lambda item: item[0]) if is_lowest else max(products, key=lambda item: item[0])
    price, product_name, document = selected
    product_id = document.metadata.get("product_id")
    identity = f" (product_id {product_id})" if product_id else ""
    adjective = "lowest-priced" if is_lowest else "highest-priced"
    return {
        "answer": f"The {adjective} product is {product_name}{identity}, priced at {price}.",
        "document": document,
    }


def is_price_question(query):
    normalized_query = query.lower()
    return "price" in normalized_query and (
        any(word in normalized_query for word in ("lowest", "cheapest", "minimum"))
        or any(word in normalized_query for word in ("highest", "most expensive", "maximum"))
    )


def _number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None
