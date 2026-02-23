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
    try_load_faiss_store,
    run_pickle_model_prediction
)

st.set_page_config(page_title="RAG with Ollama & FAISS", layout="wide")

# ─────────────────────────────────────────────
# Session state initialisation (shared)
# ─────────────────────────────────────────────
def _init_ns(ns: str):
    """Initialise a namespaced session-state bucket."""
    key = f"_state_{ns}"
    if key not in st.session_state:
        st.session_state[key] = {
            "vector_store": None,
            "rag_chain": None,
            "persona": "MEDICAL" if ns == "medical" else "EDUCATION",
            "processed_files": set(),
            "processed_images": set(),
            "last_question": None,
            "pdf_processed": False,
            "answer_generated": False,
        }
    return st.session_state[key]


def cleanup_ns(ns: str):
    s = st.session_state[f"_state_{ns}"]
    s["vector_store"] = None
    s["rag_chain"] = None
    s["pdf_processed"] = False
    s["processed_files"] = set()
    s["processed_images"] = set()
    s["last_question"] = None
    s["answer_generated"] = False
    gc.collect()

    suffix = f"_{ns}"
    for f in [f"faiss.index{suffix}", f"faiss_store{suffix}.pkl"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except PermissionError:
                os.rename(f, f"old_{uuid.uuid4()}_{f}")
            except Exception:
                pass


# ─────────────────────────────────────────────
# Helper: try load FAISS store per namespace
# ─────────────────────────────────────────────
def _try_load(ns: str):
    index_path = f"faiss.index_{ns}"
    store_path = f"faiss_store_{ns}.pkl"
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    if os.path.exists(index_path):
        try:
            emb = OllamaEmbeddings(model="mxbai-embed-large")
            return FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────
# Shared helpers re-using supporting_functions
# but with namespaced FAISS paths
# ─────────────────────────────────────────────
def _create_vs(file_path, ns):
    """Wrapper that saves FAISS to namespaced path."""
    import pickle
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from supporting_functions import extract_images_from_pdf, analyze_image_with_vision_llm
    from langchain_core.documents import Document
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    index_path = f"faiss.index_{ns}"
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs = [Document(page_content=content, metadata={"source": os.path.basename(file_path)})]
    else:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No text extracted.")
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
        image_paths = extract_images_from_pdf(file_path)
        def _proc(ip):
            try:
                vt = analyze_image_with_vision_llm(ip)
                return Document(page_content=f"Image Analysis:\n{vt}",
                                metadata={"source": os.path.basename(file_path), "type": "image"})
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=2) as ex:
            image_docs = list(ex.map(_proc, image_paths))
        docs.extend([d for d in image_docs if d])

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(index_path)
    return store


def _add_vs(existing_store, file_path, ns):
    import pickle
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from supporting_functions import extract_images_from_pdf, analyze_image_with_vision_llm
    from langchain_core.documents import Document
    from concurrent.futures import ThreadPoolExecutor

    index_path = f"faiss.index_{ns}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs = [Document(page_content=content, metadata={"source": os.path.basename(file_path)})]
    else:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No text extracted.")
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
        image_paths = extract_images_from_pdf(file_path)
        def _proc(ip):
            try:
                vt = analyze_image_with_vision_llm(ip)
                return Document(page_content=f"Image Analysis:\n{vt}",
                                metadata={"source": os.path.basename(file_path), "type": "image"})
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=2) as ex:
            image_docs = list(ex.map(_proc, image_paths))
        docs.extend([d for d in image_docs if d])

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    existing_store.add_documents(chunks)
    existing_store.save_local(index_path)
    return existing_store


# ─────────────────────────────────────────────
# Shared Q&A widget
# ─────────────────────────────────────────────
def render_qa(s: dict):
    if not s["rag_chain"]:
        st.info("Upload and process documents to begin Q&A.")
        return

    st.markdown("---")
    st.header("💬 Ask a Question")
    st.caption(f"Querying across {len(s['processed_files']) + len(s['processed_images'])} document(s)")

    user_q = st.text_input("Your question:", key=f"q_{id(s)}")

    col1, col2 = st.columns([1, 6])
    with col1:
        ask = st.button("Get Answer", key=f"ask_{id(s)}")

    if ask:
        if not user_q or not user_q.strip():
            st.warning("Please enter a question.")
        else:
            s["last_question"] = user_q.strip()
            s["answer_generated"] = True

    if s["last_question"] and s["answer_generated"]:
        st.subheader(f"Answer ({s['persona']})")
        with st.spinner("Thinking…"):
            try:
                response = s["rag_chain"].invoke({"input": s["last_question"]})
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
                            st.markdown(f"**Doc {i} — Source:** `{source}`")
                            st.info(page_content)
            except Exception as e:
                st.error(f"Failed to produce answer: {e}")


# ══════════════════════════════════════════════
#  PAGE: General RAG
# ══════════════════════════════════════════════
def page_general():
    ns = "general"
    s = _init_ns(ns)
    if s["vector_store"] is None:
        s["vector_store"] = _try_load(ns)

    st.title("📄 RAG with Ollama & FAISS")
    st.write(
        "Upload multiple PDFs and images and ask questions about their combined content. "
        "Supports incremental indexing — already-processed files are skipped."
    )

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.header("📁 Upload Documents")

        uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="gen_pdfs")
        upload_images = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="gen_imgs")

        persona_options = ("MEDICAL", "RESEARCH", "BUSINESS", "EDUCATION")
        persona_choice = st.selectbox("Select persona", persona_options,
                                      index=persona_options.index(s["persona"]) if s["persona"] in persona_options else 0,
                                      key="gen_persona")
        if persona_choice != s["persona"]:
            s["persona"] = persona_choice
            if s["vector_store"]:
                s["rag_chain"] = create_rag_chain(s["vector_store"], s["persona"])

        if s["processed_files"] or s["processed_images"]:
            st.info(f"📚 {len(s['processed_files']) + len(s['processed_images'])} file(s) loaded")
            with st.expander("View loaded files"):
                for f in sorted(s["processed_files"]):
                    st.write(f"📄 {f}")
                for f in sorted(s["processed_images"]):
                    st.write(f"🖼️ {f}")

        process_btn = st.button("Process Documents", key="gen_proc")
        if st.button("Clear All Documents", key="gen_clear"):
            cleanup_ns(ns)
            st.success("Cleared!")
            st.rerun()

    # ── Processing ───────────────────────────
    uploaded_files = uploaded_files or []
    upload_images = upload_images or []

    if process_btn:
        if not uploaded_files and not upload_images:
            st.warning("Please upload at least one file.")
        else:
            uploaded_names = {f.name for f in uploaded_files}
            uploaded_img_names = {f.name for f in upload_images}

            new_files = [f for f in uploaded_files if f.name not in s["processed_files"]]
            new_images = [f for f in upload_images if f.name not in s["processed_images"]]

            removed = (s["processed_files"] | s["processed_images"]) - (uploaded_names | uploaded_img_names)
            if removed:
                st.warning(f"Detected {len(removed)} removed file(s). Rebuilding.")
                cleanup_ns(ns)
                new_files = list(uploaded_files)
                new_images = list(upload_images)

            if not new_files and not new_images and s["vector_store"] is not None:
                st.info("No new files to add.")
            else:
                start = time.time()
                os.makedirs("temp_docs", exist_ok=True)
                os.makedirs("temp_images", exist_ok=True)
                try:
                    for idx, uf in enumerate(new_files):
                        with st.spinner(f"Processing PDF {uf.name} ({idx+1}/{len(new_files)})…"):
                            fp = os.path.join("temp_docs", uf.name)
                            with open(fp, "wb") as f:
                                f.write(uf.getbuffer())
                            if s["vector_store"] is None:
                                s["vector_store"] = _create_vs(fp, ns)
                            else:
                                s["vector_store"] = _add_vs(s["vector_store"], fp, ns)
                            s["processed_files"].add(uf.name)

                    for idx, img in enumerate(new_images):
                        with st.spinner(f"Processing image {img.name} ({idx+1}/{len(new_images)})…"):
                            ip = os.path.join("temp_images", img.name)
                            with open(ip, "wb") as f:
                                f.write(img.getbuffer())
                            try:
                                cnn = run_pickle_model_prediction(ip)
                                st.success(f"🧠 CNN: {cnn['prediction']} ({cnn['confidence']*100:.1f}%)")
                                ft = f"CNN MODEL PREDICTION:\nClass: {cnn['prediction']}\nConfidence: {cnn['confidence']*100:.2f}%"
                            except Exception as e:
                                st.error(f"CNN failed: {e}")
                                ft = "CNN prediction failed."
                            try:
                                ft += "\n\nVISION MODEL ANALYSIS:\n" + analyze_image_with_vision_llm(ip)
                            except Exception:
                                ft += "\n\nVision analysis failed."
                            txt_path = os.path.join("temp_docs", f"{img.name}_findings.txt")
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(ft)
                            if s["vector_store"] is None:
                                s["vector_store"] = _create_vs(txt_path, ns)
                            else:
                                s["vector_store"] = _add_vs(s["vector_store"], txt_path, ns)
                            s["processed_images"].add(img.name)

                    if s["vector_store"]:
                        s["rag_chain"] = create_rag_chain(s["vector_store"], s["persona"])
                    s["pdf_processed"] = True
                    st.success(f"✔ Done in {time.time()-start:.2f}s — {len(new_files)} PDF(s), {len(new_images)} image(s)")
                except Exception as e:
                    st.error(f"Error: {e}")

    render_qa(s)


# ══════════════════════════════════════════════
#  PAGE: Medical
# ══════════════════════════════════════════════
def page_medical():
    ns = "medical"
    s = _init_ns(ns)
    if s["vector_store"] is None:
        s["vector_store"] = _try_load(ns)
    s["persona"] = "MEDICAL"   # always Medical persona on this page

    st.title("🏥 Medical Analysis Hub")
    st.write(
        "Dedicated space for medical imaging, reports, and audio. "
        "Brain tumour scans are first classified by the local CNN model, "
        "then analysed with the vision LLM — all answers use the **MEDICAL** persona."
    )

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.header("🩺 Upload Medical Files")

        med_pdfs = st.file_uploader("Medical Reports (PDF)", type=["pdf"], accept_multiple_files=True, key="med_pdfs")
        med_images = st.file_uploader(
            "Medical Images (scans / photos)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="med_imgs"
        )
        med_audio = st.file_uploader(
            "Audio Notes (mp3 / wav — transcription coming soon)",
            type=["mp3", "wav", "m4a"],
            accept_multiple_files=True,
            key="med_audio"
        )

        if s["processed_files"] or s["processed_images"]:
            st.info(f"📚 {len(s['processed_files']) + len(s['processed_images'])} file(s) loaded")
            with st.expander("View loaded files"):
                for f in sorted(s["processed_files"]):
                    st.write(f"📄 {f}")
                for f in sorted(s["processed_images"]):
                    st.write(f"🖼️ {f}")

        process_btn = st.button("Process Medical Files", key="med_proc")
        if st.button("Clear All Medical Data", key="med_clear"):
            cleanup_ns(ns)
            st.success("Medical data cleared!")
            st.rerun()

    # ── Audio placeholder notice ──────────────
    if med_audio:
        st.info(
            f"🎙️ {len(med_audio)} audio file(s) uploaded. "
            "Automatic transcription (Whisper) support coming soon — "
            "files are stored but not yet indexed."
        )
        os.makedirs("temp_audio", exist_ok=True)
        for af in med_audio:
            ap = os.path.join("temp_audio", af.name)
            with open(ap, "wb") as f:
                f.write(af.getbuffer())

    # ── Processing ───────────────────────────
    med_pdfs = med_pdfs or []
    med_images = med_images or []

    if process_btn:
        if not med_pdfs and not med_images:
            st.warning("Please upload at least one PDF or image.")
        else:
            uploaded_names = {f.name for f in med_pdfs}
            uploaded_img_names = {f.name for f in med_images}

            new_files = [f for f in med_pdfs if f.name not in s["processed_files"]]
            new_images = [f for f in med_images if f.name not in s["processed_images"]]

            removed = (s["processed_files"] | s["processed_images"]) - (uploaded_names | uploaded_img_names)
            if removed:
                st.warning(f"Detected {len(removed)} removed file(s). Rebuilding.")
                cleanup_ns(ns)
                new_files = list(med_pdfs)
                new_images = list(med_images)

            if not new_files and not new_images and s["vector_store"] is not None:
                st.info("No new files to add.")
            else:
                start = time.time()
                os.makedirs("temp_docs", exist_ok=True)
                os.makedirs("temp_images", exist_ok=True)
                try:
                    # PDFs
                    for idx, uf in enumerate(new_files):
                        with st.spinner(f"Processing report {uf.name} ({idx+1}/{len(new_files)})…"):
                            fp = os.path.join("temp_docs", uf.name)
                            with open(fp, "wb") as f:
                                f.write(uf.getbuffer())
                            if s["vector_store"] is None:
                                s["vector_store"] = _create_vs(fp, ns)
                            else:
                                s["vector_store"] = _add_vs(s["vector_store"], fp, ns)
                            s["processed_files"].add(uf.name)

                    # Images — CNN then Vision LLM (unchanged workflow)
                    for idx, img in enumerate(new_images):
                        with st.spinner(f"Analysing scan {img.name} ({idx+1}/{len(new_images)})…"):
                            ip = os.path.join("temp_images", img.name)
                            with open(ip, "wb") as f:
                                f.write(img.getbuffer())

                            # Step 1: local CNN classification
                            try:
                                cnn = run_pickle_model_prediction(ip)
                                st.success(f"🧠 CNN Prediction: **{cnn['prediction']}**")
                                st.info(f"Confidence: {cnn['confidence']*100:.2f}%")
                                findings_text = (
                                    f"CNN MODEL PREDICTION:\n"
                                    f"Predicted Class: {cnn['prediction']}\n"
                                    f"Confidence: {cnn['confidence']*100:.2f}%\n"
                                )
                            except Exception as e:
                                st.error(f"CNN prediction failed: {e}")
                                findings_text = "CNN prediction failed.\n"

                            # Step 2: vision LLM analysis
                            try:
                                vision_out = analyze_image_with_vision_llm(ip)
                                findings_text += "\n\nVISION MODEL ANALYSIS:\n" + vision_out
                            except Exception:
                                findings_text += "\n\nVision analysis failed."

                            # Save findings and index
                            txt_path = os.path.join("temp_docs", f"{img.name}_findings.txt")
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(findings_text)

                            if s["vector_store"] is None:
                                s["vector_store"] = _create_vs(txt_path, ns)
                            else:
                                s["vector_store"] = _add_vs(s["vector_store"], txt_path, ns)

                            s["processed_images"].add(img.name)

                    # Build RAG chain with MEDICAL persona
                    if s["vector_store"]:
                        s["rag_chain"] = create_rag_chain(s["vector_store"], "MEDICAL")
                    s["pdf_processed"] = True
                    st.success(f"✔ Done in {time.time()-start:.2f}s — {len(new_files)} report(s), {len(new_images)} scan(s)")

                except Exception as e:
                    st.error(f"Error while processing: {e}")

    # ── Inline image preview ──────────────────
    if med_images:
        with st.expander("🔬 Preview Uploaded Scans"):
            cols = st.columns(min(len(med_images), 4))
            for i, img in enumerate(med_images):
                with cols[i % 4]:
                    st.image(img, caption=img.name, use_column_width=True)

    render_qa(s)


# ══════════════════════════════════════════════
#  Navigation
# ══════════════════════════════════════════════
PAGES = {
    "📄 General RAG": page_general,
    "🏥 Medical": page_medical,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection]()