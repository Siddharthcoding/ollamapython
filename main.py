import streamlit as st
import os
from supporting_functions import create_rag_chain, create_vector_store
import time

st.set_page_config(page_title="RAG with Ollama & ChromaDB", layout="wide")
st.title("📄 RAG Project with Ollama & ChromaDB")

st.write("""
Welcome! Upload a PDF and ask any question about its content.
The system uses a local Ollama model (LLaMA3.2) and ChromaDB for retrieval.
""")

# Sidebar upload area
with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF file and click 'Process'", type=['pdf'])
    process_btn = st.button("Process")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

st_time = time.time()
# Process the uploaded document
if process_btn and uploaded_file is not None:
    with st.spinner("Processing document... this may take a few minutes"):
        temp_dir = "temp_docs"
        os.makedirs(temp_dir, exist_ok=True)

        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create vector store and RAG chain
        st.session_state.vector_store = create_vector_store(file_path)
        st.session_state.rag_chain = create_rag_chain(st.session_state.vector_store)

        st.success("Document processed successfully! You can now ask questions.")
end_time = time.time()

st.write("Prcessed time:",end_time-st_time)

st1_time = time.time()
# Show Q&A section
if st.session_state.rag_chain is not None:
    st.header("Ask your question")
    qna = st.text_input("Enter your question:", key="user_question")

    if st.button("Get Answer"):
        if qna.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": qna})

                    st.write("### Answer")
                    st.write(response["answer"])

                    with st.expander("Show Retrieved Context"):
                        st.write("The following context was used:")
                        for doc in response["context"]:
                            st.info(doc.page_content)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("Please upload and process a PDF file to continue.")


end1_time = time.time()

st.write("Prcessed time:",end1_time-st1_time)