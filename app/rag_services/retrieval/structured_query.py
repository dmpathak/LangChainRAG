"""Deterministic answers for simple numeric questions over tabular data."""

import re


def is_price_question(query):
    text = query.lower()
    comparison = ("lowest", "cheapest", "minimum", "highest", "most expensive", "maximum")
    return "price" in text and any(word in text for word in comparison)


def answer_price_question(query, documents):
    text = query.lower()
    lowest = any(word in text for word in ("lowest", "cheapest", "minimum"))
    products = []

    for document in documents:
        price = _number(document.metadata.get("price"))
        if price is None:
            continue
        product_name = next(
            (
                document.metadata.get(field)
                for field in ("product_name", "name", "title", "product")
                if document.metadata.get(field)
            ),
            "Unknown product",
        )
        products.append((price, str(product_name), document))

    if not products:
        return None

    selected = min(products) if lowest else max(products)
    price, product_name, document = selected
    product_id = document.metadata.get("product_id")
    identity = f" (product_id {product_id})" if product_id else ""
    adjective = "lowest-priced" if lowest else "highest-priced"
    return {
        "answer": f"The {adjective} product is {product_name}{identity}, priced at {price}.",
        "document": document,
    }


def _number(value):
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None
