from fastapi import FastAPI, File, Form, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage

from config import collection_name as default_collection_name
from schema import Movie
from services.AI_models import get_llm_model
from services.document_parser import DocumentProcessor
from services.llm_service import LLMService
from services.prompt import SYSTEM_PROMPT
from services.retrieval_service import get_retrieval_service

app = FastAPI()
llm_model = get_llm_model()
document_processor = DocumentProcessor()
llm_service = LLMService()


@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form(default_collection_name),
):
    documents = document_processor.process(
        file=file,
        metadata={"file_name": file.filename},
    )
    get_retrieval_service(collection_name).add_documents(documents)
    return {"status": "success", "documents_indexed": len(documents)}


@app.post("/search")
def search(
    user_query: str,
    top_k: int = 8,
    collection_name: str = default_collection_name,
):
    """Search the indexed documents and answer the user's question."""
    retrieved_docs = get_retrieval_service(collection_name).search(
        query=user_query,
        top_k=min(max(top_k, 1), 30),
    )
    if not retrieved_docs:
        return {
            "answer": "No relevant information found or no documents have been indexed yet."
        }

    context_parts = []
    for index, document in enumerate(retrieved_docs, start=1):
        context_parts.append(f"[Document {index}]\n{document.page_content}")
    context = "\n\n".join(context_parts)

    print("Invoking LLM...")
    answer = llm_service.get_response(query=user_query, context=context)

    sources = []
    for index, document in enumerate(retrieved_docs, start=1):
        source = {
            "label": f"Document {index}",
            "document_id": document.metadata.get("document_id"),
            "file_name": document.metadata.get("file_name"),
            "page_number": document.metadata.get("page_number"),
            "row_number": document.metadata.get("row_number"),
        }
        sources.append(source)

    return {
        "answer": answer,
        "sources_found": len(retrieved_docs),
        "sources": sources,
    }


@app.post("/invoke")
async def invoke(user_query: str):
    """Direct LLM demo without RAG."""
    conversation = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_query)]
    model_with_structure = llm_model.with_structured_output(Movie)
    return model_with_structure.invoke(conversation)


@app.post("/stream")
async def stream(user_query: str):
    """Direct streaming LLM demo without RAG."""
    conversation = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]
    return llm_model.stream(conversation)
