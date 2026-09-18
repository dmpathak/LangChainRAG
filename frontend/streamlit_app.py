import sys
from pathlib import Path

import requests
import streamlit as st

# Streamlit adds ``frontend/`` to the import path when this file is run
# directly. Add the project root so the sibling ``app`` package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    APP_API_KEY,
    DEFAULT_COLLECTION_NAME,
    DOCUMENT_COLLECTION_NAME,
    RERANK_ENABLED,
    RETRIEVAL_MODE,
)

API_URL = "http://localhost:8000"
API_HEADERS = {"X-API-Key": APP_API_KEY} if APP_API_KEY else {}

st.set_page_config(page_title="Document RAG Assistant", page_icon="💬", layout="wide")
st.title("💬 Document RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "documents_collection" not in st.session_state:
    st.session_state.documents_collection = None


def show_error(response):
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def load_documents(collection):
    response = requests.get(
        f"{API_URL}/documents",
        params={"collection_name": collection},
        headers=API_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("documents", [])


with st.sidebar:
    st.header("Upload Document")
    collection_name = st.selectbox(
        "Document collection",
        [DOCUMENT_COLLECTION_NAME, DEFAULT_COLLECTION_NAME],
    )
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "csv", "xls", "xlsx"],
    )

    if uploaded_file and st.button("Upload Document"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            response = requests.post(
                f"{API_URL}/upload",
                files=files,
                data={"collection_name": collection_name},
                headers=API_HEADERS,
                timeout=120,
            )
            if response.ok:
                result = response.json()
                if result.get("status") == "skipped":
                    st.info("This file is already indexed.")
                else:
                    st.success(f"Indexed {result.get('documents_indexed', 0)} chunks.")
                st.session_state.documents = load_documents(collection_name)
                st.session_state.documents_collection = collection_name
            else:
                st.error(show_error(response))
        except requests.RequestException as error:
            st.error(str(error))

    st.divider()
    st.header("Indexed documents")
    if st.session_state.documents_collection != collection_name:
        try:
            st.session_state.documents = load_documents(collection_name)
            st.session_state.documents_collection = collection_name
        except requests.RequestException:
            st.session_state.documents = []

    if st.button("Refresh documents"):
        try:
            st.session_state.documents = load_documents(collection_name)
            st.session_state.documents_collection = collection_name
        except requests.RequestException as error:
            st.error(str(error))

    documents = st.session_state.documents
    if documents:
        document_names = [document["file_name"] for document in documents]
        selected_document = st.selectbox("Select a file", document_names)
        if st.button("Delete selected file"):
            response = requests.delete(
                f"{API_URL}/documents",
                params={"file_name": selected_document, "collection_name": collection_name},
                headers=API_HEADERS,
                timeout=30,
            )
            if response.ok:
                st.success(f"Deleted {selected_document}.")
                st.session_state.documents = load_documents(collection_name)
                st.rerun()
            else:
                st.error(show_error(response))
    else:
        st.caption("No indexed documents found in this collection.")

    st.divider()
    retrieval_label = RETRIEVAL_MODE.title()
    if RERANK_ENABLED:
        retrieval_label += " + reranking"
    st.caption(f"Retrieval mode: {retrieval_label}")
    top_k = st.number_input("Retrieved chunks", 1, 50, 30)
    min_score = st.number_input("Minimum similarity score", 0.0, 1.0, 0.0, 0.05)
    file_options = ["All files"] + [document["file_name"] for document in documents]
    search_file = st.selectbox("Search within", file_options)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask anything about your indexed documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            params = {
                "user_query": prompt,
                "top_k": top_k,
                "min_score": min_score,
                "collection_name": collection_name,
            }
            if search_file != "All files":
                params["file_name"] = search_file
            response = requests.post(
                f"{API_URL}/search",
                params=params,
                headers=API_HEADERS,
                timeout=120,
            )
            if not response.ok:
                st.error(show_error(response))
                st.stop()

            response_data = response.json()
            st.markdown(response_data["answer"])
            sources = response_data.get("sources", [])
            source_names = sorted({source.get("file_name") for source in sources if source.get("file_name")})
            if source_names:
                st.caption("Sources: " + ", ".join(source_names))
        except requests.RequestException as error:
            st.error(str(error))
