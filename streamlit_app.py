import requests
import streamlit as st

from config import (
    APP_API_KEY,
    RERANK_ENABLED,
    RETRIEVAL_MODE,
    collection_name,
    document_collection_name,
)

API_URL = "http://localhost:8000"
API_HEADERS = {"X-API-Key": APP_API_KEY} if APP_API_KEY else {}

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="💬",
    layout="wide",
)
st.title("💬 Document RAG Assistant")

# Session State
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


def load_documents(selected_collection):
    response = requests.get(
        f"{API_URL}/documents",
        params={"collection_name": selected_collection},
        headers=API_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("documents", [])

# Sidebar
with st.sidebar:
    st.header("Upload Document")

    collection_name = st.selectbox(
        "Document collection",
        ["documents", "MyLangChainCollection"],
        key="collection_selector",
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "csv", "xls", "xlsx"],
    )
    if uploaded_file and st.button("Upload Document"):
        with st.spinner("Uploading document..."):
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
                        st.success(
                            f"Document indexed: {result.get('documents_indexed', 0)} chunks."
                        )
                    st.session_state.documents = load_documents(collection_name)
                    st.session_state.documents_collection = collection_name
                else:
                    st.error(show_error(response))
            except requests.RequestException as exc:
                st.error(str(exc))

    st.divider()
    st.header("Indexed documents")
    if st.button("Refresh documents"):
        try:
            st.session_state.documents = load_documents(collection_name)
        except requests.RequestException as exc:
            st.error(str(exc))

    if st.session_state.documents_collection != collection_name:
        try:
            st.session_state.documents = load_documents(collection_name)
            st.session_state.documents_collection = collection_name
        except requests.RequestException:
            st.session_state.documents = []

    documents = st.session_state.documents
    if documents:
        document_names = [item["file_name"] for item in documents]
        selected_document = st.selectbox("Select a file", document_names)
        if st.button("Delete selected file"):
            try:
                response = requests.delete(
                    f"{API_URL}/documents",
                    params={
                        "file_name": selected_document,
                        "collection_name": collection_name,
                    },
                    headers=API_HEADERS,
                    timeout=30,
                )
                if response.ok:
                    st.success(f"Deleted {selected_document}.")
                    st.session_state.documents = load_documents(collection_name)
                    st.rerun()
                else:
                    st.error(show_error(response))
            except requests.RequestException as exc:
                st.error(str(exc))
    else:
        st.caption("Click Refresh documents to view indexed files.")

    st.divider()
    retrieval_label = RETRIEVAL_MODE.title()
    if RERANK_ENABLED:
        retrieval_label += " + reranking"
    st.caption(f"Retrieval mode: {retrieval_label}")
    top_retrieve = st.number_input("Retrieved chunks", 1, 50, 30)
    min_score = st.number_input(
        "Minimum similarity score",
        0.0,
        1.0,
        0.0,
        0.05,
        help="Start at 0.0. Increase this only after checking returned scores.",
    )
    file_options = ["All files"] + [item["file_name"] for item in documents]
    search_file = st.selectbox("Search within", file_options)

# Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat
prompt = st.chat_input("Ask anything about your indexed documents...")

if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            params = {
                "user_query": prompt,
                "collection_name": collection_name,
            }
            params.update(
                {
                    "top_k": top_retrieve,
                    "min_score": min_score,
                }
            )
            if search_file != "All files":
                params["file_name"] = search_file

            response = requests.post(
                f"{API_URL}/search",
                params=params,
                headers=API_HEADERS,
                timeout=120,
            )

            if response.ok:
                response_data = response.json()
                answer = response_data["answer"]

                placeholder.markdown(answer)

                sources = response_data.get("sources", [])
                grouped_sources = {}
                for source in sources:
                    source_name = source.get("file_name") or source.get("label")
                    if not source_name:
                        continue

                    grouped = grouped_sources.setdefault(
                        source_name,
                        {"locations": [], "best_score": source.get("score")},
                    )
                    score = source.get("score")
                    if score is not None and (
                        grouped["best_score"] is None or score > grouped["best_score"]
                    ):
                        grouped["best_score"] = score

                    location = source.get("page_number") or source.get("row_number")
                    if location is not None and location not in grouped["locations"]:
                        grouped["locations"].append(location)

                if grouped_sources:
                    st.caption("Sources")
                    for file_name, source in grouped_sources.items():
                        details = []
                        if source["locations"]:
                            details.append(
                                "locations: " + ", ".join(map(str, source["locations"]))
                            )
                        if source["best_score"] is not None:
                            details.append(f"best score: {source['best_score']}")
                        st.caption(f"- {file_name}" + (f" ({'; '.join(details)})" if details else ""))
                elif response_data.get("reason") == "no_results_above_threshold":
                    st.warning(
                        "No chunks passed the similarity threshold. "
                        "Keep the threshold at 0.0 first, and confirm that the "
                        "document was uploaded to the selected collection."
                    )

                assistant_message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(assistant_message)
            else:
                placeholder.error(show_error(response))
        except requests.RequestException as exc:
            placeholder.error(str(exc))
