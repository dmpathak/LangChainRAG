"""Create the Milvus collection for documents.

Run with:
    python -m Others.schema_migration_documents

To recreate the collection:
    python -m Others.schema_migration_documents --drop-existing
"""

import argparse
import sys

from pymilvus import DataType, MilvusClient

from app.config import (
    DOCUMENT_COLLECTION_NAME as document_collection_name,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    MILVUS_URI,
)
from app.rag_services.llm.AI_models import get_embedding_model


def provision(drop_existing=False):
    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(document_collection_name):
        if not drop_existing:
            print(
                f"Collection '{document_collection_name}' already exists."
            )
            return

        client.drop_collection(document_collection_name)
        print(
            f"Dropped existing collection: "
            f"{document_collection_name}."
        )

    # Get embedding dimension dynamically
    dimension = len(
        get_embedding_model().embed_query("dimension test")
    )

    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )

    # ---------------------------------------------------------
    # Primary key
    # ---------------------------------------------------------
    schema.add_field(
        field_name="pk",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )

    # ---------------------------------------------------------
    # Document chunk text
    # ---------------------------------------------------------
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )

    # ---------------------------------------------------------
    # Embedding vector
    # ---------------------------------------------------------
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=dimension,
    )

    # ---------------------------------------------------------
    # Source document
    # ---------------------------------------------------------
    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=1024,
        nullable=True,
    )

    # ---------------------------------------------------------
    # Page number
    # ---------------------------------------------------------
    schema.add_field(
        field_name="page",
        datatype=DataType.INT64,
        nullable=True,
    )

    # ---------------------------------------------------------
    # Chunk identifier
    # ---------------------------------------------------------
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        max_length=256,
        nullable=True,
    )

    # ---------------------------------------------------------
    # Indexes
    # ---------------------------------------------------------
    index_params = client.prepare_index_params()

    # Vector index
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M": HNSW_M,
            "efConstruction": HNSW_EF_CONSTRUCTION,
        },
    )

    # Scalar indexes
    index_params.add_index(
        field_name="source",
        index_type="INVERTED",
    )

    index_params.add_index(
        field_name="page",
        index_type="INVERTED",
    )

    # ---------------------------------------------------------
    # Create collection
    # ---------------------------------------------------------
    client.create_collection(
        collection_name=document_collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )

    print(
        f"Created collection '{document_collection_name}' "
        f"with dimension {dimension}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the Milvus document collection"
    )

    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Delete and recreate the collection",
    )

    args = parser.parse_args()

    if args.drop_existing:
        confirmation = input(
            "This will DELETE all collection data. "
            "Type 'yes' to continue: "
        )

        if confirmation.strip().lower() != "yes":
            print("Operation cancelled.")
            sys.exit(0)

    provision(
        drop_existing=args.drop_existing
    )
