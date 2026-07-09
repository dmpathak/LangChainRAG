from fastapi import FastAPI, UploadFile, File

from schema import Movie
from services.document_parser import DocumentProcessor
from services.llm_service import LLMService
from services.prompt import SYSTEM_PROMPT
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from services.AI_models import get_llm_model
from services.vector_db import get_vector_store

app = FastAPI()
llm_model = get_llm_model()
document_processor = DocumentProcessor()
llm_service = LLMService()


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    documents = document_processor.process(
        file=file,
        metadata={
            "file_name": file.filename,
        },
    )

    # This is where
    get_vector_store().add_documents(documents)

    return {
        "status": "success",
        "documents_indexed": len(documents),
    }


@app.post("/search")
def search(user_query: str, top_k: int = 5):
    """
    Search products.
    This is a RAG (Retrieval Augmented Generation) system.
    """
    # Step 1: Retrieve relevant documents
    retrieved_docs = get_vector_store().similarity_search(
        query=user_query,
        top_k=top_k,
    )
    if not retrieved_docs:
        return {
            "answer": "No relevant information found or no documents have been indexed yet."
        }

    # Step 2: Create context
    context = "\n\n".join([
        f"Document {idx + 1}:\n{doc.page_content}"
        for idx, doc in enumerate(retrieved_docs)
    ])

    # Send to LLM:
    print("Invoking LLM...")
    response = llm_service.get_response(
        query=user_query,
        context=context,
    )
    return {
        "answer": response,
        "sources_found": len(retrieved_docs),
    }


@app.post("/invoke")
async def invoke(user_query: str):
    """
    ******* Not Related To RAG & lagchain flow *******
    Direct Q/A with LLM.  (NO RAG)
    """
    conversation = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_query)]

    model_with_structure = llm_model.with_structured_output(Movie)
    response = model_with_structure.invoke(conversation)
    return response


@app.post("/stream")
async def stream(user_query: str):
    """
    ******* Not Related To RAG & lagchain flow *******
    Direct Q/A with LLM.  (NO RAG)

    You need to use socket for this.
    """
    conversation = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]

    response = llm_model.stream(conversation)
    return response
