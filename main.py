import streamlit as st
import os
import time
import gc
import uuid
from supporting_functions import create_rag_chain, analyze_image_with_vision_llm, create_vector_store, add_to_vector_store


st.set_page_config(page_title="RAG with Ollama & FAISS", layout="wide")
st.title("📄 RAG Project with Ollama & FAISS")


st.write("""
Upload multiple PDFs and ask questions about their combined content.
This optimized version loads PDFs faster and avoids re-processing.
""")


# ---- Session state initialization ----
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "persona" not in st.session_state:
    st.session_state.persona = "EDUCATION"
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # ✅ Track all processed files
if "processed_images" not in st.session_state:
    st.session_state.processed_images = set()  # Track all the images
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "analyze_image_with_vision_llm" not in st.session_state:
    st.session_state.analyze_image_with_vision_llm = None
if "answer_generated" not in st.session_state:
    st.session_state.answer_generated = False  # ✅ Track if answer was shown
if "persona_changed" not in st.session_state:
    st.session_state.persona_changed = False  # ✅ Track persona changes



# ---- Function to safely cleanup FAISS files ----
def cleanup_faiss_files():
    st.session_state.vector_store = None
    st.session_state.analyze_image_with_vision_llm = None
    st.session_state.rag_chain = None
    st.session_state.pdf_processed = False
    st.session_state.processed_files = set()  # ✅ Clear file tracking
    st.session_state.processed_images = set()  # Clear images tracking
    st.session_state.last_question = None
    st.session_state.answer_generated = False
    gc.collect()


    for f in ["faiss.index", "faiss_store.pkl"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except PermissionError:
                os.rename(f, f"old_{uuid.uuid4()}_{f}")



# ---- Sidebar ----
with st.sidebar:
    st.header("Upload Your Documents")


    # ✅ Enable multiple file upload
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type=["pdf"],
        accept_multiple_files=True  # ✅ KEY CHANGE
    )

    upload_images = st.file_uploader(
        "Upload Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )


    persona_choice = st.selectbox(
        "Select your persona",
        ("RESEARCH", "BUSINESS", "EDUCATION"),
        index=("RESEARCH", "BUSINESS", "EDUCATION").index(st.session_state.persona)
    )


    # ✅ Persona switch = rebuild ONLY RAG chain + trigger regeneration
    if persona_choice != st.session_state.persona:
        st.session_state.persona = persona_choice
        st.session_state.persona_changed = True  # ✅ Mark that persona changed
        if st.session_state.vector_store:
            st.session_state.rag_chain = create_rag_chain(
                st.session_state.vector_store,
                st.session_state.persona
            )


    # ✅ Show processed files count
    if st.session_state.processed_files:
        st.info(f"📚 {len(st.session_state.processed_files)} file(s) currently loaded")
        with st.expander("View loaded files"):
            for fname in st.session_state.processed_files:
                st.write(f"✓ {fname}")


    process_btn = st.button("Process Documents")
    
    # ✅ Clear all button
    if st.button("Clear All Documents"):
        cleanup_faiss_files()
        st.success("All documents cleared!")
        st.rerun()



# ---- Document Processing (Multiple PDFs) ----
if process_btn and (uploaded_files or upload_images):
    
    # ✅ Identify new files to process
    uploaded_file_names = {f.name for f in uploaded_files}
    uploaded_image_names = {f.name for f in upload_images}
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    new_images = [f for f in upload_images if f.name not in st.session_state.processed_images]

    removed_files = st.session_state.processed_files - uploaded_file_names - uploaded_image_names


    # ✅ Handle removed files - rebuild from scratch if any removed
    if removed_files:
        st.warning(f"Detected {len(removed_files)} removed file(s). Rebuilding vector store...")
        cleanup_faiss_files()
        new_files = uploaded_files  # Process all files
        new_images = upload_images


    if new_files or new_images:
        start_time = time.time()
        
        os.makedirs("temp_docs", exist_ok=True)
        os.makedirs("temp_images", exist_ok=True)


        # ✅ Process each new file
        for idx, uploaded_file in enumerate(new_files):
            with st.spinner(f"🚀 Processing {uploaded_file.name} ({idx+1}/{len(new_files)})..."):
                
                file_path = os.path.join("temp_docs", uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())


                # ✅ First file: create vector store
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = create_vector_store(file_path)
                else:
                    # ✅ Subsequent files: merge into existing store
                    st.session_state.vector_store = add_to_vector_store(
                        st.session_state.vector_store,
                        file_path
                    )
                
                st.session_state.processed_files.add(uploaded_file.name)
        
        for idx, upload_images in enumerate(new_images):
            with st.spinner(f"🚀 Processing {upload_images.name} ({idx+1}/{len(new_images)})..."):

                img_file_path = os.path.join("temp_images", upload_images.name)

                with open (img_file_path, "wb") as f:
                    f.write(upload_images.getbuffer())

                image_findings = analyze_image_with_vision_llm(img_file_path)

                findings_txt_path = os.path.join(
                    "temp_docs", f"{upload_images.name}_findings.txt"
                )

                with open(findings_txt_path, "w", encoding="utf-8") as f:
                    f.write(image_findings)

                if st.session_state.vector_store is None:
                    st.session_state.vector_store = create_vector_store(findings_txt_path)
                else:
                    st.session_state.vector_store = add_to_vector_store(
                        st.session_state.vector_store,
                        findings_txt_path
                    )
                
                st.session_state.processed_images.add(upload_images.name)


        # ✅ Create/update RAG chain with combined vector store
        st.session_state.rag_chain = create_rag_chain(
            st.session_state.vector_store,
            st.session_state.persona
        )
        st.session_state.pdf_processed = True


        st.success(
                f"✔ {len(new_files)} PDF(s) and {len(new_images)} image(s) processed "
                f"in **{time.time() - start_time:.2f} seconds**"
            )
        st.success(f"📚 Total documents loaded: {len(st.session_state.processed_files)}")

    else:
        st.info("📄 All uploaded PDFs already processed. Ready for questions.")


elif process_btn and not uploaded_files:
    st.warning("Please upload at least one PDF file.")



# ---- Q&A Section ----
if st.session_state.rag_chain:

    st.header("Ask a Question")
    st.caption(f"Querying across {len(st.session_state.processed_files)} document(s)")
    user_q = st.text_input("Your question:")

    if st.button("Get Answer"):
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.last_question = user_q
            st.session_state.answer_generated = True
            st.session_state.persona_changed = False 


    # ---- Auto-Generate Answer on Persona Change ----
    if st.session_state.persona_changed and st.session_state.last_question and st.session_state.answer_generated:
        st.session_state.persona_changed = False  


    # ---- Generate / Regenerate Answer ----
    if st.session_state.last_question and st.session_state.answer_generated:


        st.subheader(f"Answer ({st.session_state.persona})")
        answer_placeholder = st.empty()
        streamed_text = ""
        final_response = None

        with st.spinner("🤖 Thinking..."):
            try:
                for chunk in st.session_state.rag_chain.stream(
                    {"input": st.session_state.last_question}
                ):
                    if "answer" in chunk:
                        streamed_text += chunk["answer"]
                        answer_placeholder.markdown(streamed_text)


                    if "context" in chunk:
                        final_response = chunk


                if final_response and "context" in final_response:
                    context_chunks = final_response["context"]
                    num_chunks = min(5, len(context_chunks))
                    with st.expander(f"Retrieved Context (Top {num_chunks} Chunks)"):
                        for i, doc in enumerate(context_chunks[:num_chunks], 1):
                            st.markdown(f"**Chunk {i}:**")
                            # ✅ Show source document if available
                            source = doc.metadata.get('source', 'Unknown')
                            st.caption(f"Source: {source}")
                            st.info(doc.page_content)

            except Exception as e:
                st.error(f"Error: {e}")


else:
    st.info("Upload and process PDF(s) to begin.")
