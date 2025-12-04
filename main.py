import streamlit as st
import os
import time
from supporting_functions import create_rag_chain, create_vector_store, PERSIST_DIR
from langchain_community.document_loaders import PyMuPDFLoader
import shutil



st.set_page_config(page_title="RAG with Ollama & ChromaDB", layout="wide")
st.title("📄 RAG Project with Ollama & ChromaDB")

st.write("""
Upload a PDF and ask questions about its content.
This optimized version loads PDFs faster and avoids re-processing.
""")

# ---- Session state initialization ----
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None

# ---- Sidebar ----
with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    process_btn = st.button("Process")

# ---- Document Processing ----
if process_btn and uploaded_file is not None:

    if st.session_state.last_file != uploaded_file.name and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    # Avoid reprocessing same file
    if st.session_state.last_file == uploaded_file.name:
        st.warning("⚠️ This PDF was already processed. Ready to answer queries.")
    else:
        start_time = time.time()
        with st.spinner("🚀 Processing PDF..."):

            # Save file
            os.makedirs("temp_docs", exist_ok=True)
            file_path = os.path.join("temp_docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())


            # Create vector store ONCE
            st.session_state.vector_store = create_vector_store(file_path)
            st.session_state.rag_chain = create_rag_chain(st.session_state.vector_store)

            st.session_state.last_file = uploaded_file.name

        end_time = time.time()
        st.success(f"✔ PDF processed in **{end_time - start_time:.2f} seconds**")

# ---- Q&A Section ----
if st.session_state.rag_chain:

    st.header("Ask a Question")
    user_q = st.text_input("Your question:")

    if st.button("Get Answer"):

        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            start_time1 = time.time()
            with st.spinner("🤖 Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_q})

                    st.subheader("Answer")
                    st.write(response["answer"])

                    with st.expander("Retrieved Context"):
                        for i, doc in enumerate(response["context"], start=1):
                            st.markdown(f"**Chunk {i}:**")
                            st.info(doc.page_content)

                except Exception as e:
                    st.error(f"Error: {e}")
            end_time1 = time.time()
            st.success(f"✔Answered in **{end_time1 - start_time1:.2f} seconds**")

else:
    st.info("Upload and process a PDF to begin.")
