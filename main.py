import hashlib
import logging
from hmac import compare_digest
from io import BytesIO
from types import SimpleNamespace

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from config import APP_API_KEY, MAX_UPLOAD_SIZE_MB, MIN_SIMILARITY_SCORE
from config import collection_name as default_collection_name
from services.document_parser import DocumentProcessor
from services.llm_service import LLMService
from services.retrieval_service import get_retrieval_service
from services.structured_query import answer_price_question, is_price_question

app = FastAPI()
logger = logging.getLogger(__name__)
document_processor = DocumentProcessor()
llm_service = LLMService()


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Require a key only when APP_API_KEY is configured."""
    if APP_API_KEY and (not x_api_key or not compare_digest(x_api_key, APP_API_KEY)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def validate_upload(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file name is required")
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is {MAX_UPLOAD_SIZE_MB} MB.",
        )


def process_upload(file_name, file_bytes, collection_name):
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    retrieval_service = get_retrieval_service(collection_name)

    if retrieval_service.has_file_hash(file_hash):
        return {"status": "skipped", "reason": "duplicate", "documents_indexed": 0}

    try:
        # Re-uploading a changed file replaces the previous file with the same name.
        retrieval_service.delete_file(file_name)
        uploaded_file = SimpleNamespace(
            filename=file_name,
            file=BytesIO(file_bytes),
        )
        documents = document_processor.process(
            file=uploaded_file,
            metadata={"file_name": file_name, "file_hash": file_hash},
        )
        retrieval_service.add_documents(documents)
        return {"status": "success", "documents_indexed": len(documents)}
    except Exception as error:
        logger.exception("Document processing failed for %s", file_name)
        raise


@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form(default_collection_name),
    _: None = Depends(require_api_key),
):
    validate_upload(file)
    return process_upload(file.filename, file.file.read(), collection_name)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(
    user_query: str,
    top_k: int = 8,
    collection_name: str = default_collection_name,
    file_name: str | None = None,
    min_score: float = MIN_SIMILARITY_SCORE,
    _: None = Depends(require_api_key),
):
    """Search the indexed documents and answer the user's question."""
    retrieval_service = get_retrieval_service(collection_name)
    structured_result = None
    if is_price_question(user_query):
        structured_result = answer_price_question(
            user_query,
            retrieval_service.get_all_documents(file_name=file_name),
        )
    if structured_result:
        document = structured_result["document"]
        return {
            "answer": structured_result["answer"] + " [Document 1]",
            "sources_found": 1,
            "sources": [
                {
                    "label": "Document 1",
                    "document_id": document.metadata.get("document_id"),
                    "file_name": document.metadata.get("file_name"),
                    "page_number": document.metadata.get("page_number"),
                    "row_number": document.metadata.get("row_number"),
                    "score": 1.0,
                }
            ],
        }

    scored_docs = retrieval_service.search_with_scores(
        query=user_query,
        top_k=min(max(top_k, 1), 30),
        min_score=max(min_score, -1.0),
        file_name=file_name,
    )
    retrieved_docs = [document for document, score in scored_docs]
    if not retrieved_docs:
        return {
            "answer": "No relevant information found or no documents have been indexed yet.",
            "reason": "no_results_above_threshold",
            "min_score": min_score,
            "collection_name": collection_name,
        }

    context_parts = []
    for index, (document, score) in enumerate(scored_docs, start=1):
        context_parts.append(f"[Document {index}]\n{document.page_content}")
    context = "\n\n".join(context_parts)

    print("Invoking LLM...")
    answer = llm_service.get_response(query=user_query, context=context)

    sources = []
    for index, (document, score) in enumerate(scored_docs, start=1):
        source = {
            "label": f"Document {index}",
            "document_id": document.metadata.get("document_id"),
            "file_name": document.metadata.get("file_name"),
            "page_number": document.metadata.get("page_number"),
            "row_number": document.metadata.get("row_number"),
            "score": round(score, 4),
        }
        sources.append(source)

    return {
        "answer": answer,
        "sources_found": len(retrieved_docs),
        "sources": sources,
    }


@app.get("/documents")
def list_documents(
    collection_name: str = default_collection_name,
    _: None = Depends(require_api_key),
):
    return {"documents": get_retrieval_service(collection_name).list_documents()}


@app.delete("/documents")
def delete_documents(
    file_name: str,
    collection_name: str = default_collection_name,
    _: None = Depends(require_api_key),
):
    deleted_chunks = get_retrieval_service(collection_name).delete_file(file_name)
    return {
        "status": "success",
        "file_name": file_name,
        "deleted_chunks": deleted_chunks,
    }
