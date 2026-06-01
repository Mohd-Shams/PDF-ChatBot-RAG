import tempfile
import streamlit as st

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector import create_vector_store
from src.retriver import get_retriever
from src.rag_chain import generate_answer


# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------------------
# SESSION STATE
# ---------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ---------------------------
# SIDEBAR
# ---------------------------

with st.sidebar:

    st.title("📚 PDF Assistant")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    if st.session_state.retriever:
        st.success("✅ PDF Loaded")
    else:
        st.warning("⚠ Upload a PDF")

# ---------------------------
# MAIN TITLE
# ---------------------------

st.title("📄 PDF AI Chatbot")
st.caption("Upload a PDF and ask unlimited questions.")

# ---------------------------
# PDF PROCESSING
# ---------------------------

if uploaded_file:

    if "processed_pdf" not in st.session_state or \
       st.session_state.processed_pdf != uploaded_file.name:

        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            documents = load_pdf(pdf_path)

            chunks = split_documents(documents)

            embeddings = get_embedding_model()

            vectordb = create_vector_store(
                chunks,
                embeddings
            )

            retriever = get_retriever(vectordb)

            st.session_state.retriever = retriever
            st.session_state.processed_pdf = uploaded_file.name

        st.success("PDF processed successfully!")

# ---------------------------
# DISPLAY CHAT HISTORY
# ---------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------
# USER QUESTION
# ---------------------------

if st.session_state.retriever:

    user_question = st.chat_input(
        "Ask anything about the PDF..."
    )

    if user_question:

        # Show user message

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate answer

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = generate_answer(
                    user_question,
                    st.session_state.retriever
                )

                st.markdown(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )