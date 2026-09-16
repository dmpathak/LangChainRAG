import requests
import streamlit as st

from config import collection_name, document_collection_name

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="💬",
    layout="wide",
)
st.title("💬 Document RAG Assistant")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Upload Document")

    collection_name = st.selectbox(
        "Document collection",
        [f"{collection_name} -> CSV, XLS, XLSX", f"{document_collection_name} -> PDF, DOCX"],
        key="collection_selector",
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "csv", "xls", "xlsx"],
    )
    top_retrieve = st.number_input("Final context documents", 1, 30, 15)

    if uploaded_file and st.button("Upload Document"):
        with st.spinner("Indexing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    data={"collection_name": collection_name},
                )

                if response.ok:
                    st.success("Document indexed successfully.")
                else:
                    st.error(response.text)
            except requests.RequestException as exc:
                st.error(str(exc))

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
            response = requests.post(
                f"{API_URL}/search",
                params={
                    "user_query": prompt,
                    "top_k": top_retrieve,
                    "collection_name": collection_name,
                },
            )

            if response.ok:
                response_data = response.json()
                answer = response_data["answer"]

                placeholder.markdown(answer)

                sources = response_data.get("sources", [])
                source_names = []

                for source in sources:
                    source_name = source.get("file_name") or source.get("label")
                    if source_name and source_name not in source_names:
                        source_names.append(source_name)

                if source_names:
                    st.caption("Sources: " + ", ".join(source_names))

                assistant_message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(assistant_message)
            else:
                error_message = response.text

                # Backend should return an appropriate message when
                # no documents have been indexed yet.
                try:
                    error_message = response.json().get("detail", error_message)
                except Exception:
                    pass

                placeholder.error(error_message)
        except requests.RequestException as exc:
            placeholder.error(str(exc))
