# main.py
import streamlit as st
import os
import time
import gc
import uuid
from typing import List, Set

from supporting_functions import (
    create_rag_chain,
    analyze_image_with_vision_llm,
    create_vector_store,
    add_to_vector_store,
    try_load_faiss_store
)


st.set_page_config(page_title="RAG with Ollama & FAISS (Medical-ready)", layout="wide")
st.title("📄 RAG Project with Ollama & FAISS (Medical-ready)")

st.write(
    """
Upload multiple PDFs (medical reports, scanned lab results) and images (scans) and ask
questions about their combined content. This app avoids re-processing already added files
and supports incremental indexing.
"""
)

# -------------------------
# Session state initialization
# -------------------------
if "vector_store" not in st.session_state:
    # try to load existing store from disk if present
    st.session_state.vector_store = try_load_faiss_store()
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "persona" not in st.session_state:
    st.session_state.persona = "MEDICAL"  # default to medical persona
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # type: Set[str]
if "processed_images" not in st.session_state:
    st.session_state.processed_images = set()  # type: Set[str]
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "answer_generated" not in st.session_state:
    st.session_state.answer_generated = False
if "persona_changed" not in st.session_state:
    st.session_state.persona_changed = False


# -------------------------
# Function to safely cleanup FAISS files
# -------------------------
def cleanup_faiss_files():
    st.session_state.vector_store = None
    st.session_state.rag_chain = None
    st.session_state.pdf_processed = False
    st.session_state.processed_files = set()
    st.session_state.processed_images = set()
    st.session_state.last_question = None
    st.session_state.answer_generated = False
    gc.collect()

    for f in ["faiss.index", "faiss_store.pkl"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except PermissionError:
                os.rename(f, f"old_{uuid.uuid4()}_{f}")
            except Exception:
                # best-effort delete
                pass


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Upload Your Documents")

    # multiple file uploader may be None if user didn't upload anything
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    upload_images = st.file_uploader(
        "Upload Images (scans/photos)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    persona_choice = st.selectbox(
        "Select your persona",
        ("MEDICAL", "RESEARCH", "BUSINESS", "EDUCATION"),
        index=("MEDICAL", "RESEARCH", "BUSINESS", "EDUCATION").index(st.session_state.persona
                                                                       if st.session_state.persona in ("MEDICAL", "RESEARCH", "BUSINESS", "EDUCATION")
                                                                       else 0)
    )

    # If persona changed, mark and rebuild chain (not vector store)
    if persona_choice != st.session_state.persona:
        st.session_state.persona = persona_choice
        st.session_state.persona_changed = True
        if st.session_state.vector_store:
            st.session_state.rag_chain = create_rag_chain(
                st.session_state.vector_store,
                st.session_state.persona
            )

    # Status of currently loaded files
    if st.session_state.processed_files or st.session_state.processed_images:
        st.info(f"📚 {len(st.session_state.processed_files) + len(st.session_state.processed_images)} file(s) currently loaded")
        with st.expander("View loaded files/images"):
            for fname in sorted(st.session_state.processed_files):
                st.write(f"📄 {fname}")
            for iname in sorted(st.session_state.processed_images):
                st.write(f"🖼️ {iname}")

    process_btn = st.button("Process Documents")

    if st.button("Clear All Documents"):
        cleanup_faiss_files()
        st.success("All documents cleared!")
        st.rerun()


# -------------------------
# Document Processing (Multiple PDFs + Images)
# -------------------------
# Convert None -> empty list for easier set ops
uploaded_files = uploaded_files or []
upload_images = upload_images or []

if process_btn:
    if not uploaded_files and not upload_images:
        st.warning("Please upload at least one PDF or image to process.")
    else:
        uploaded_file_names = {f.name for f in uploaded_files}
        uploaded_image_names = {f.name for f in upload_images}

        # new inputs (not processed earlier)
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        new_images = [f for f in upload_images if f.name not in st.session_state.processed_images]

        # detect removed files (user removed some from UI) -> rebuild required
        removed_files = (st.session_state.processed_files | st.session_state.processed_images) - (uploaded_file_names | uploaded_image_names)
        if removed_files:
            st.warning(f"Detected {len(removed_files)} removed file(s). Rebuilding vector store from current uploads.")
            cleanup_faiss_files()
            # Re-process everything currently present in the uploader
            new_files = list(uploaded_files)
            new_images = list(upload_images)

        if not new_files and not new_images and st.session_state.vector_store is not None:
            st.info("No new files to add — vector store already contains uploaded documents.")
        else:
            start_time = time.time()
            os.makedirs("temp_docs", exist_ok=True)
            os.makedirs("temp_images", exist_ok=True)

            # create or extend vector store
            try:
                # process PDFs
                for idx, uploaded_file in enumerate(new_files):
                    with st.spinner(f"Processing PDF {uploaded_file.name} ({idx+1}/{len(new_files)})..."):
                        file_path = os.path.join("temp_docs", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        if st.session_state.vector_store is None:
                            st.session_state.vector_store = create_vector_store(file_path)
                        else:
                            st.session_state.vector_store = add_to_vector_store(
                                st.session_state.vector_store, file_path
                            )

                        st.session_state.processed_files.add(uploaded_file.name)

                # process images
                for idx, img in enumerate(new_images):
                    with st.spinner(f"Processing Image {img.name} ({idx+1}/{len(new_images)})..."):
                        img_file_path = os.path.join("temp_images", img.name)
                        with open(img_file_path, "wb") as f:
                            f.write(img.getbuffer())

                        # run vision LLM to extract findings, stored as a small text doc
                        image_findings = analyze_image_with_vision_llm(img_file_path)
                        findings_txt_path = os.path.join("temp_docs", f"{img.name}_findings.txt")
                        with open(findings_txt_path, "w", encoding="utf-8") as f:
                            f.write(image_findings)

                        if st.session_state.vector_store is None:
                            st.session_state.vector_store = create_vector_store(findings_txt_path)
                        else:
                            st.session_state.vector_store = add_to_vector_store(
                                st.session_state.vector_store, findings_txt_path
                            )

                        st.session_state.processed_images.add(img.name)

                # create/update rag chain
                if st.session_state.vector_store:
                    st.session_state.rag_chain = create_rag_chain(
                        st.session_state.vector_store, st.session_state.persona
                    )

                st.session_state.pdf_processed = True
                elapsed = time.time() - start_time
                st.success(f"✔ {len(new_files)} PDF(s) and {len(new_images)} image(s) processed in {elapsed:.2f} s")
                st.success(f"📚 Total documents loaded: {len(st.session_state.processed_files)} PDFs, {len(st.session_state.processed_images)} images")

            except Exception as e:
                st.error(f"Error while processing documents: {e}")


# -------------------------
# Q&A Section
# -------------------------
if st.session_state.rag_chain:
    st.header("Ask a Question")
    st.caption(f"Querying across {len(st.session_state.processed_files) + len(st.session_state.processed_images)} document(s)")

    user_q = st.text_input("Your question:")

    if st.button("Get Answer"):
        if not user_q or not user_q.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.last_question = user_q.strip()
            st.session_state.answer_generated = True
            st.session_state.persona_changed = False

    # If last_question and answer_generated, invoke chain
    if st.session_state.last_question and st.session_state.answer_generated:
        st.subheader(f"Answer ({st.session_state.persona})")
        with st.spinner("Thinking..."):
            try:
                # retrieval chain uses .invoke / .run (non-streaming)
                response = st.session_state.rag_chain.invoke({"input": st.session_state.last_question})
                # Some chain types return dict, some return string. Normalize:
                if isinstance(response, dict):
                    answer_text = response.get("answer") or response.get("output_text") or str(response)
                    context = response.get("context") or response.get("source_documents") or []
                else:
                    answer_text = str(response)
                    context = []

                st.markdown(answer_text)

                if context:
                    with st.expander("Retrieved Context (top documents)"):
                        for i, doc in enumerate(context[:6], 1):

                            if hasattr(doc, "page_content"):
                                page_content = doc.page_content
                                metadata = doc.metadata
                            elif isinstance(doc, dict):
                                page_content = doc.get("page_content", str(doc))
                                metadata = doc.get("metadata", {})
                            else:
                                page_content = str(doc)
                                metadata = {}

                            source = metadata.get("source", "Unknown")

                            st.markdown(f"**Doc {i} — Source:** {source}")
                            st.info(page_content)


            except Exception as e:
                st.error(f"Failed to produce answer: {e}")
else:
    st.info("Upload and process PDF(s) or images to begin.")
