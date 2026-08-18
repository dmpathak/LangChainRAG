import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Document RAG Assistant")

# ---------------------------------
# Session State
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------
# Sidebar
# ---------------------------------

with st.sidebar:
    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "csv", "xls", "xlsx"],
    )
    top_retrieve = st.number_input("Number of documents to retrieve", 1, 500, 50)

    if uploaded_file and st.button("Upload Document"):

        with st.spinner("Indexing document..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
            }

            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                )

                if response.ok:
                    st.success("Document indexed successfully.")
                else:
                    st.error(response.text)

            except requests.RequestException as exc:
                st.error(str(exc))

# ---------------------------------
# Chat History
# ---------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------
# Chat
# ---------------------------------

if prompt := st.chat_input(
    "Ask anything about your indexed documents..."
):

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
                    "top_k": top_retrieve,
                },
            )

            if response.ok:
                answer = response.json()["answer"]

                placeholder.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            else:
                error_message = response.text

                # Backend should return an appropriate message when
                # no documents have been indexed yet.
                try:
                    error_message = response.json().get(
                        "detail",
                        error_message,
                    )
                except Exception:
                    pass

                placeholder.error(error_message)

        except requests.RequestException as exc:
            placeholder.error(str(exc))
