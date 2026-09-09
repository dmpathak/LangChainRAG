"""Create the Milvus collection. Run with: python -m Others.schema_migration"""

import argparse
import sys

from pymilvus import DataType, MilvusClient

from config import HNSW_EF_CONSTRUCTION, HNSW_M, MILVUS_URI, collection_name
from services.AI_models import get_embedding_model


def provision(drop_existing=False):
    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(collection_name):
        if not drop_existing:
            print(f"Collection '{collection_name}' already exists.")
            return

        client.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}.")

    dimension = len(get_embedding_model().embed_query("dimension test"))
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

    schema.add_field(
        field_name="pk",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=dimension,
    )
    schema.add_field(
        field_name="product_id",
        datatype=DataType.VARCHAR,
        max_length=128,
        nullable=True,
    )
    schema.add_field(
        field_name="product_name",
        datatype=DataType.VARCHAR,
        max_length=512,
        nullable=True,
    )
    schema.add_field(
        field_name="brand",
        datatype=DataType.VARCHAR,
        max_length=128,
        nullable=True,
    )
    schema.add_field(
        field_name="category",
        datatype=DataType.VARCHAR,
        max_length=128,
        nullable=True,
    )
    schema.add_field(field_name="price", datatype=DataType.FLOAT, nullable=True)
    schema.add_field(field_name="rating", datatype=DataType.FLOAT, nullable=True)
    schema.add_field(field_name="stock", datatype=DataType.INT64, nullable=True)
    schema.add_field(
        field_name="features",
        datatype=DataType.VARCHAR,
        max_length=2048,
        nullable=True,
    )
    schema.add_field(
        field_name="description",
        datatype=DataType.VARCHAR,
        max_length=4096,
        nullable=True,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
    )

    indexed_fields = ["product_name", "brand", "category", "price", "rating", "stock"]
    for field in indexed_fields:
        index_params.add_index(field_name=field, index_type="INVERTED")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )
    print(f"Created collection '{collection_name}' with dimension {dimension}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the Milvus collection")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Delete and recreate the collection",
    )
    args = parser.parse_args()

    if args.drop_existing:
        confirmation = input(
            "This will DELETE all collection data. Type 'yes' to continue: "
        )
        if confirmation.strip().lower() != "yes":
            print("Operation cancelled.")
            sys.exit(0)

    provision(drop_existing=args.drop_existing)
