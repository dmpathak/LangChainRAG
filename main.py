from fastapi import FastAPI, File, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage

from schema import Movie
from services.AI_models import get_llm_model
from services.document_parser import DocumentProcessor
from services.llm_service import LLMService
from services.prompt import SYSTEM_PROMPT
from services.retrieval_service import get_retrieval_service
from services.search_filters import SearchFilters

app = FastAPI()
llm_model = get_llm_model()
document_processor = DocumentProcessor()
llm_service = LLMService()


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    documents = document_processor.process(
        file=file,
        metadata={"file_name": file.filename},
    )
    get_retrieval_service().add_documents(documents)
    return {"status": "success", "documents_indexed": len(documents)}


@app.post("/search")
def search(
    user_query: str,
    top_k: int = 8,
    brand: str | None = None,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    in_stock: bool | None = None,
):
    """Search the indexed documents and answer the user's question."""
    filters = SearchFilters(
        brand=brand,
        category=category,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        in_stock=in_stock,
    )
    retrieved_docs = get_retrieval_service().search(
        query=user_query,
        filters=filters,
        top_k=min(max(top_k, 1), 20),
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
        "search_plan": retrieved_docs[0].metadata.get("search_plan"),
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
