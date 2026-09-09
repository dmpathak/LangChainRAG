import requests
import streamlit as st

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

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "csv", "xls", "xlsx"],
    )
    top_retrieve = st.number_input("Final context documents", 1, 30, 15)

    with st.expander("Optional product filters"):
        brand = st.text_input("Brand")
        category = st.text_input("Category")
        min_price = st.number_input("Minimum price", min_value=0.0, value=None)
        max_price = st.number_input("Maximum price", min_value=0.0, value=None)
        min_rating = st.number_input(
            "Minimum rating", min_value=0.0, max_value=5.0, value=None
        )
        in_stock = st.checkbox("In stock only")

    if uploaded_file and st.button("Upload Document"):
        with st.spinner("Indexing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

            try:
                response = requests.post(f"{API_URL}/upload", files=files)

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
                    "brand": brand or None,
                    "category": category or None,
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_rating": min_rating,
                    "in_stock": in_stock or None,
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

                if response_data.get("search_plan"):
                    with st.expander("LLM search plan"):
                        st.json(response_data["search_plan"])

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
