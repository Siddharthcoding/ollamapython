# main.py
import streamlit as st
import os
import time
import gc
import uuid
from typing import Set

from supporting_functions import (
    create_rag_chain,
    analyze_image_with_vision_llm,
    create_vector_store,
    add_to_vector_store,
    try_load_faiss_store,
    run_brain_tumor_scratch_model,
    transcribe_audio_file,
    ensure_brain_tumor_model_trained,
    add_text_to_vector_store
)

st.set_page_config(page_title="RAG with Ollama & FAISS (Medical-ready)", layout="wide")
st.title("📄 RAG Project with Ollama & FAISS (Medical-ready)")

st.write(
    """
Upload multiple PDFs (medical reports, scanned lab results) and images (scans) and ask
questions about their combined content. This app avoids re-processing already added files
and supports incremental indexing.

**Important medical disclaimer:** The app is for research/educational purposes only.
It *does not* provide medical diagnoses. Always consult a licensed clinician for
any medical decision. The brain-tumor detection flow is a *scratch/prototype* — see notes below.
"""
)

# -------------------------
# Top-level page selector
# -------------------------
page = st.sidebar.radio("Select Page", ["RAG", "Medical"], index=0)

# -------------------------
# Shared session state initialization (used by RAG)
# -------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = try_load_faiss_store()
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "persona" not in st.session_state:
    st.session_state.persona = "MEDICAL"
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
                pass

# -------------------------
# RAG Page (existing)
# -------------------------
if page == "RAG":
    with st.sidebar:
        st.header("Upload Your Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="rag_pdf_uploader"
        )

        upload_images = st.file_uploader(
            "Upload Images (scans/photos)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="rag_image_uploader"
        )

        persona_choice = st.selectbox(
            "Select your persona",
            ("MEDICAL", "RESEARCH", "BUSINESS", "EDUCATION"),
            index=("MEDICAL", "RESEARCH", "BUSINESS", "EDUCATION").index(
                st.session_state.persona if st.session_state.persona in ("MEDICAL", "RESEARCH", "BUSINESS", "EDUCATION") else 0
            )
        )

        if persona_choice != st.session_state.persona:
            st.session_state.persona = persona_choice
            st.session_state.persona_changed = True
            if st.session_state.vector_store:
                st.session_state.rag_chain = create_rag_chain(
                    st.session_state.vector_store,
                    st.session_state.persona
                )

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

    uploaded_files = uploaded_files or []
    upload_images = upload_images or []

    if process_btn:
        if not uploaded_files and not upload_images:
            st.warning("Please upload at least one PDF or image to process.")
        else:
            uploaded_file_names = {f.name for f in uploaded_files}
            uploaded_image_names = {f.name for f in upload_images}

            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            new_images = [f for f in upload_images if f.name not in st.session_state.processed_images]

            removed_files = (st.session_state.processed_files | st.session_state.processed_images) - (uploaded_file_names | uploaded_image_names)
            if removed_files:
                st.warning(f"Detected {len(removed_files)} removed file(s). Rebuilding vector store from current uploads.")
                cleanup_faiss_files()
                new_files = list(uploaded_files)
                new_images = list(upload_images)

            if not new_files and not new_images and st.session_state.vector_store is not None:
                st.info("No new files to add — vector store already contains uploaded documents.")
            else:
                start_time = time.time()
                os.makedirs("temp_docs", exist_ok=True)
                os.makedirs("temp_images", exist_ok=True)

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

                            image_findings = analyze_image_with_vision_llm(img_file_path)
                            findings_txt_path = os.path.join("temp_docs", f"{img.name}_findings.txt")
                            with open(findings_txt_path, "w", encoding="utf-8") as f:
                                f.write(image_findings)

                            # Add text findings directly into vector store (robust)
                            if st.session_state.vector_store is None:
                                st.session_state.vector_store = add_text_to_vector_store(image_findings, source=img.name)
                            else:
                                st.session_state.vector_store = add_text_to_vector_store(image_findings, source=img.name, existing_store=st.session_state.vector_store)

                            st.session_state.processed_images.add(img.name)

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

        if st.session_state.last_question and st.session_state.answer_generated:
            st.subheader(f"Answer ({st.session_state.persona})")
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": st.session_state.last_question})
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

# -------------------------
# Medical Page (new)
# -------------------------
elif page == "Medical":
    st.header("Medical Tools & Prototype Models")
    st.write(
        """
This page contains tools specific to medical workflows:
- image / scan analysis via Vision LLM (LLava/Ollama)
- brain tumor detection prototype (will train if model absent)
- audio transcription input (optional, placeholder)

**Strong disclaimer**: outputs are for research/educational use only and *not* clinical diagnosis.
"""
    )

    med_tabs = st.tabs(["Image / Scan Analysis", "Brain Tumor (Scratch Prototype)", "Audio Notes"])

    # Image / Scan Analysis
    with med_tabs[0]:
        st.subheader("Image / Scan Analysis (Vision LLM)")
        med_images = st.file_uploader("Upload medical image(s) (MRI/CT/X-ray) for analysis",
                                      type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="med_imgs")

        if med_images:
            for img in med_images:
                st.write(f"**{img.name}**")
                st.image(img, width=400)
            if st.button("Run Vision LLM Analysis"):
                os.makedirs("temp_med_images", exist_ok=True)
                for img in med_images:
                    path = os.path.join("temp_med_images", img.name)
                    with open(path, "wb") as f:
                        f.write(img.getbuffer())
                    with st.spinner(f"Analyzing {img.name}..."):
                        try:
                            findings = analyze_image_with_vision_llm(path)
                            st.markdown("**Findings (Vision LLM):**")
                            st.info(findings)
                        except Exception as e:
                            st.error(f"Vision LLM analysis failed: {e}")

    # Brain Tumor Scratch Prototype
    with med_tabs[1]:
        st.subheader("Brain Tumor Detection — Scratch Prototype")
        st.write(
            """
This flow attempts to run a small prototype CNN model. If the model is absent:
1. The app will attempt to download a public dataset via Kaggle (if credentials present).
2. If download is not possible, a synthetic dataset will be generated automatically.
3. The scratch CNN will train (progress shown) and be saved to `models/brain_tumor_model.h5`.
"""
        )

        tumor_image = st.file_uploader("Upload single brain scan (jpg/png).", type=["png", "jpg", "jpeg"], key="tumor_img_uploader")

        if tumor_image:
            st.image(tumor_image, caption="Uploaded image", width=400)

            if st.button("Run Brain Tumor Prototype"):
                os.makedirs("temp_med_images", exist_ok=True)
                img_path = os.path.join("temp_med_images", tumor_image.name)
                with open(img_path, "wb") as f:
                    f.write(tumor_image.getbuffer())

                st.info("Checking dataset/model. If absent, dataset will be downloaded or generated and the model will be trained (this may take several minutes).")
                progress_bar = st.progress(0)
                status = st.empty()

                try:
                    ensure_brain_tumor_model_trained(progress_bar=progress_bar, status=status)
                except Exception as e:
                    st.error(f"Failed while ensuring model trained: {e}")
                    st.stop()

                st.success("Model ready. Running prediction...")

                with st.spinner("Running brain tumor prototype..."):
                    try:
                        result = run_brain_tumor_scratch_model(img_path)

                        st.markdown("## 🧠 Prototype Result")

                        if result.get("note"):
                            st.warning(result["note"])

                        if result.get("prediction"):
                            st.success(f"Prediction: **{result['prediction']}**")

                        if result.get("confidence") is not None:
                            st.write(f"Confidence: **{result['confidence']:.3f}**")

                        if result.get("heatmap_path") and os.path.exists(result["heatmap_path"]):
                            st.image(result["heatmap_path"], caption="Grad-CAM Heatmap", width=400)

                        if result.get("analysis_text"):
                            st.info(result["analysis_text"])

                    except Exception as e:
                        st.error(f"Brain tumor prototype failed: {e}")

    # Audio tab
    with med_tabs[2]:
        st.subheader("Audio Notes (transcription placeholder)")
        st.write("Upload an audio note (e.g., clinician dictation). The app will attempt to transcribe it if a transcription backend is available.")
        uploaded_audio = st.file_uploader("Upload audio (wav, mp3, m4a)", type=["wav", "mp3", "m4a"], accept_multiple_files=False, key="audio_upload")

        if uploaded_audio:
            audio_path = os.path.join("temp_med_images", uploaded_audio.name)
            os.makedirs("temp_med_images", exist_ok=True)
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())

            st.audio(audio_path)
            if st.button("Transcribe Audio"):
                with st.spinner("Transcribing..."):
                    try:
                        transcription, note = transcribe_audio_file(audio_path)
                        if note:
                            st.warning(note)
                        st.markdown("**Transcription:**")
                        st.write(transcription)
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
