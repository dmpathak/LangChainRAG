import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="💬",   # Chat Assistant
    layout="wide",
)

st.title("💬 Document RAG Assistant")

# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "csv", "xls", "xlsx"],
    )

    if uploaded_file:

        if st.button("Upload Document"):

            with st.spinner("Indexing document..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    )
                }

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                )

                if response.status_code == 200:
                    st.success("Document indexed successfully.")
                    st.session_state.document_uploaded = True
                else:
                    st.error(
                        f"Upload failed: {response.text}"
                    )

# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Input
# -----------------------------

if prompt := st.chat_input(
    "Ask anything about the uploaded document..."
):

    if not st.session_state.document_uploaded:
        st.warning(
            "Please upload a document first."
        )
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        try:
            response = requests.post(
                f"{API_URL}/search",
                params={
                    "user_query": prompt,
                    "top_k": 5,
                },
            )

            if response.status_code == 200:

                data = response.json()

                answer = (
                    data["answer"]
                    if isinstance(data, dict)
                    else str(data)
                )

                placeholder.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            else:
                placeholder.error(
                    f"Error: {response.text}"
                )

        except Exception as exc:
            placeholder.error(str(exc))