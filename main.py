import streamlit as st
import os
import time
import shutil
import gc
from langchain_community.llms import Ollama
import uuid
from supporting_functions import create_rag_chain, create_vector_store, set_persona

st.set_page_config(page_title="RAG with Ollama & FAISS", layout="wide")
st.title("📄 RAG Project with Ollama & FAISS")

st.write("""
Upload a PDF and ask questions about its content.
This optimized version loads PDFs faster and avoids re-processing.
""")

# ---- Session state initialization ----
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "persona" not in st.session_state:
    st.session_state.persona = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None

# ---- Function to safely cleanup FAISS files ----
def cleanup_faiss_files():
    st.session_state.vector_store = None
    st.session_state.rag_chain = None
    gc.collect()  # Ensure Windows releases file handles

    for f in ["faiss.index", "faiss_store.pkl"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except PermissionError:
                # Rename instead of deleting (Windows allows renaming open files)
                os.rename(f, f"old_{uuid.uuid4()}_{f}")

# ---- Sidebar ----
with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    persona_choice = st.selectbox(
    "Select your persona",
    ("RESEARCH", "BUSINESS", "EDUCATION"),)
    process_btn = st.button("Process")

# ---- Document Processing ----
if process_btn and uploaded_file is not None:

    # Cleanup FAISS to ensure fresh index
    cleanup_faiss_files()

    start_time = time.time()
    with st.spinner("🚀 Processing PDF..."):

        # Save file temporarily
        os.makedirs("temp_docs", exist_ok=True)
        file_path = os.path.join("temp_docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create vector store and RAG chain
        st.session_state.vector_store = create_vector_store(file_path)
        persona_choice = set_persona(st.session_state.persona)
        st.session_state.rag_chain = create_rag_chain(st.session_state.vector_store,persona_choice)

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
            st.subheader("Answer (Streaming)")
            answer_placeholder = st.empty()

            streamed_text = ""
            final_response = None

            with st.spinner("🤖 Thinking..."):
                try:
                    llm = Ollama(model="llama3.2:1b")
                    # five = llm.invoke(f"""
                    # Rewrite this question into 3 sharper, more proper versions:
                    # "{user_q}"
                    # """)
                    # chosen = llm.invoke(f"""Select the best question for retrieval.
                    # Return only the selected question:
                    # {five}
                    # """)
                    st.info(f"Selected prompt(best):{user_q}")
                    # Stream answer tokens
                    for chunk in st.session_state.rag_chain.stream({"input": user_q}):
                        if "answer" in chunk:
                            streamed_text += chunk["answer"]
                            answer_placeholder.markdown(streamed_text)

                        if "context" in chunk:
                            final_response = chunk  # capture final context

                    # Show retrieved chunks (at least 5 if available)
                    if final_response and "context" in final_response:
                        context_chunks = final_response["context"]
                        num_chunks = min(5, len(context_chunks))
                        with st.expander(f"Retrieved Context (Top {num_chunks} Chunks)"):
                            for i, doc in enumerate(context_chunks[:num_chunks], start=1):
                                st.markdown(f"**Chunk {i}:**")
                                st.info(doc.page_content)

                except Exception as e:
                    st.error(f"Error: {e}")

            end_time1 = time.time()
            st.success(f"✔ Answered in **{end_time1 - start_time1:.2f} seconds**")

else:
    st.info("Upload and process a PDF to begin.")
