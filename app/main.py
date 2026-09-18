import hashlib
import logging
from hmac import compare_digest
from io import BytesIO
from types import SimpleNamespace

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from app.config import APP_API_KEY, DEFAULT_COLLECTION_NAME, MAX_UPLOAD_SIZE_MB
from app.config import MIN_SIMILARITY_SCORE
from app.rag_services.documents.document_parser import DocumentProcessor
from app.rag_services.llm.llm_service import LLMService
from app.rag_services.retrieval.retrieval_service import get_retrieval_service
from app.rag_services.retrieval.structured_query import (
    answer_price_question,
    is_price_question,
)

app = FastAPI()
logger = logging.getLogger(__name__)
document_processor = DocumentProcessor()
llm_service = LLMService()


def require_api_key(x_api_key: str | None = Header(default=None)):
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
        retrieval_service.delete_file(file_name)
        uploaded_file = SimpleNamespace(filename=file_name, file=BytesIO(file_bytes))
        documents = document_processor.process(
            file=uploaded_file,
            metadata={"file_name": file_name, "file_hash": file_hash},
        )
        retrieval_service.add_documents(documents)
        return {"status": "success", "documents_indexed": len(documents)}
    except Exception:
        logger.exception("Document processing failed for %s", file_name)
        raise


@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form(DEFAULT_COLLECTION_NAME),
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
    collection_name: str = DEFAULT_COLLECTION_NAME,
    file_name: str | None = None,
    min_score: float = MIN_SIMILARITY_SCORE,
    _: None = Depends(require_api_key),
):
    retrieval_service = get_retrieval_service(collection_name)

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
                "sources": [{
                    "label": "Document 1",
                    "document_id": document.metadata.get("document_id"),
                    "file_name": document.metadata.get("file_name"),
                    "page_number": document.metadata.get("page_number"),
                    "row_number": document.metadata.get("row_number"),
                    "score": 1.0,
                }],
            }

    scored_documents = retrieval_service.search_with_scores(
        query=user_query,
        top_k=min(max(top_k, 1), 30),
        min_score=max(min_score, -1.0),
        file_name=file_name,
    )
    if not scored_documents:
        return {
            "answer": "No relevant information found or no documents have been indexed yet.",
            "reason": "no_results_above_threshold",
            "min_score": min_score,
            "collection_name": collection_name,
        }

    context = "\n\n".join(
        f"[Document {index}]\n{document.page_content}"
        for index, (document, score) in enumerate(scored_documents, start=1)
    )
    answer = llm_service.get_response(query=user_query, context=context)

    sources = []
    for index, (document, score) in enumerate(scored_documents, start=1):
        sources.append({
            "label": f"Document {index}",
            "document_id": document.metadata.get("document_id"),
            "file_name": document.metadata.get("file_name"),
            "page_number": document.metadata.get("page_number"),
            "row_number": document.metadata.get("row_number"),
            "score": round(score, 4),
        })

    return {"answer": answer, "sources_found": len(sources), "sources": sources}


@app.get("/documents")
def list_documents(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    _: None = Depends(require_api_key),
):
    return {"documents": get_retrieval_service(collection_name).list_documents()}


@app.delete("/documents")
def delete_documents(
    file_name: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    _: None = Depends(require_api_key),
):
    deleted_chunks = get_retrieval_service(collection_name).delete_file(file_name)
    return {
        "status": "success",
        "file_name": file_name,
        "deleted_chunks": deleted_chunks,
    }
