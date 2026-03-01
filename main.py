"""
main.py  —  RAG + Medical + SLI Audio
──────────────────────────────────────────────────────────────────────────────
Three pages reachable via the sidebar:
  📄  General RAG   – multi-persona PDF / image Q&A
  🏥  Medical       – brain-tumour CNN + vision LLM + PDF reports
  🎙️  SLI Audio     – LANNA-trained classifier, acoustic features,
                       transcription, RAG Q&A on audio findings
──────────────────────────────────────────────────────────────────────────────
"""

import os
import io
import time
import gc
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from typing import Set

from supporting_functions import (
    create_rag_chain,
    analyze_image_with_vision_llm,
    run_pickle_model_prediction,
)

# ── SLI audio helpers ────────────────────────────────────────────────────────
try:
    from sli_audio_functions import (
        load_wav,
        trim_silence,
        extract_all_features,
        predict_sli,
        transcribe_audio,
        build_audio_findings_text,
        plot_waveform,
        plot_spectrogram,
        plot_mfcc,
        plot_feature_radar,
        plot_probability_bar,
        FEATURE_NAMES,
    )
    SLI_AVAILABLE = True
except ImportError as _e:
    SLI_AVAILABLE = False
    _SLI_IMPORT_ERROR = str(_e)


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="RAG + Medical + SLI", layout="wide")


# ═════════════════════════════════════════════
#  SHARED UTILITIES
# ═════════════════════════════════════════════

def _init_ns(ns: str) -> dict:
    key = f"_state_{ns}"
    if key not in st.session_state:
        st.session_state[key] = {
            "vector_store":    None,
            "rag_chain":       None,
            "persona":         "MEDICAL" if ns == "medical" else "EDUCATION",
            "processed_files": set(),
            "processed_images": set(),
            "last_question":   None,
            "pdf_processed":   False,
            "answer_generated": False,
        }
    return st.session_state[key]


def cleanup_ns(ns: str):
    s = st.session_state[f"_state_{ns}"]
    s.update({
        "vector_store": None, "rag_chain": None,
        "pdf_processed": False, "processed_files": set(),
        "processed_images": set(), "last_question": None,
        "answer_generated": False,
    })
    gc.collect()
    for f in [f"faiss.index_{ns}", f"faiss_store_{ns}.pkl"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except PermissionError:
                os.rename(f, f"old_{uuid.uuid4()}_{f}")
            except Exception:
                pass


def _try_load(ns: str):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    idx = f"faiss.index_{ns}"
    if os.path.exists(idx):
        try:
            emb = OllamaEmbeddings(model="mxbai-embed-large")
            return FAISS.load_local(idx, emb, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None


def _create_vs(file_path: str, ns: str):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from concurrent.futures import ThreadPoolExecutor
    from supporting_functions import extract_images_from_pdf

    idx  = f"faiss.index_{ns}"
    emb  = OllamaEmbeddings(model="mxbai-embed-large")
    ext  = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs = [Document(page_content=content,
                         metadata={"source": os.path.basename(file_path)})]
    else:
        loader = PyMuPDFLoader(file_path)
        docs   = loader.load()
        if not docs:
            raise ValueError("No text extracted.")
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
        def _proc(ip):
            try:
                vt = analyze_image_with_vision_llm(ip)
                return Document(page_content=f"Image Analysis:\n{vt}",
                                metadata={"source": os.path.basename(file_path),
                                          "type": "image"})
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=2) as ex:
            image_docs = list(ex.map(_proc, extract_images_from_pdf(file_path)))
        docs.extend([d for d in image_docs if d])

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100).split_documents(docs)
    store  = FAISS.from_documents(chunks, emb)
    store.save_local(idx)
    return store


def _add_vs(existing_store, file_path: str, ns: str):
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from concurrent.futures import ThreadPoolExecutor
    from supporting_functions import extract_images_from_pdf

    idx = f"faiss.index_{ns}"
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs = [Document(page_content=content,
                         metadata={"source": os.path.basename(file_path)})]
    else:
        loader = PyMuPDFLoader(file_path)
        docs   = loader.load()
        if not docs:
            raise ValueError("No text extracted.")
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
        def _proc(ip):
            try:
                vt = analyze_image_with_vision_llm(ip)
                return Document(page_content=f"Image Analysis:\n{vt}",
                                metadata={"source": os.path.basename(file_path),
                                          "type": "image"})
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=2) as ex:
            image_docs = list(ex.map(_proc, extract_images_from_pdf(file_path)))
        docs.extend([d for d in image_docs if d])

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100).split_documents(docs)
    existing_store.add_documents(chunks)
    existing_store.save_local(idx)
    return existing_store


def render_qa(s: dict, key_suffix: str = ""):
    if not s["rag_chain"]:
        st.info("Upload and process documents to begin Q&A.")
        return

    st.markdown("---")
    st.header("💬 Ask a Question")
    st.caption(
        f"Querying across "
        f"{len(s['processed_files']) + len(s['processed_images'])} document(s)"
    )

    user_q = st.text_input("Your question:", key=f"q_{key_suffix}")
    if st.button("Get Answer", key=f"ask_{key_suffix}"):
        if not user_q or not user_q.strip():
            st.warning("Please enter a question.")
        else:
            s["last_question"]   = user_q.strip()
            s["answer_generated"] = True

    if s["last_question"] and s["answer_generated"]:
        st.subheader(f"Answer ({s['persona']})")
        with st.spinner("Thinking…"):
            try:
                response = s["rag_chain"].invoke({"input": s["last_question"]})
                if isinstance(response, dict):
                    answer_text = (response.get("answer") or
                                   response.get("output_text") or str(response))
                    context = (response.get("context") or
                               response.get("source_documents") or [])
                else:
                    answer_text = str(response)
                    context     = []

                st.markdown(answer_text)
                if context:
                    with st.expander("Retrieved Context"):
                        for i, doc in enumerate(context[:6], 1):
                            if hasattr(doc, "page_content"):
                                pc, meta = doc.page_content, doc.metadata
                            elif isinstance(doc, dict):
                                pc   = doc.get("page_content", str(doc))
                                meta = doc.get("metadata", {})
                            else:
                                pc, meta = str(doc), {}
                            st.markdown(
                                f"**Doc {i} — Source:** `{meta.get('source','?')}`"
                            )
                            st.info(pc)
            except Exception as e:
                st.error(f"Failed to produce answer: {e}")


# ═════════════════════════════════════════════
#  PAGE 1 — General RAG
# ═════════════════════════════════════════════

def page_general():
    ns = "general"
    s  = _init_ns(ns)
    if s["vector_store"] is None:
        s["vector_store"] = _try_load(ns)

    st.title("📄 RAG with Ollama & FAISS")
    st.write(
        "Upload multiple PDFs and images and ask questions about their "
        "combined content. Supports incremental indexing."
    )

    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF(s)", type=["pdf"],
            accept_multiple_files=True, key="gen_pdfs"
        )
        upload_images = st.file_uploader(
            "Upload Images", type=["png","jpg","jpeg"],
            accept_multiple_files=True, key="gen_imgs"
        )
        persona_opts = ("MEDICAL","RESEARCH","BUSINESS","EDUCATION")
        persona_choice = st.selectbox(
            "Select persona", persona_opts,
            index=persona_opts.index(s["persona"])
            if s["persona"] in persona_opts else 0,
            key="gen_persona"
        )
        if persona_choice != s["persona"]:
            s["persona"] = persona_choice
            if s["vector_store"]:
                s["rag_chain"] = create_rag_chain(
                    s["vector_store"], s["persona"])

        if s["processed_files"] or s["processed_images"]:
            st.info(
                f"📚 {len(s['processed_files'])+len(s['processed_images'])} "
                "file(s) loaded"
            )
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

    uploaded_files = uploaded_files or []
    upload_images  = upload_images  or []

    if process_btn:
        if not uploaded_files and not upload_images:
            st.warning("Please upload at least one file.")
        else:
            uploaded_names    = {f.name for f in uploaded_files}
            uploaded_img_names = {f.name for f in upload_images}
            new_files  = [f for f in uploaded_files if f.name not in s["processed_files"]]
            new_images = [f for f in upload_images  if f.name not in s["processed_images"]]
            removed = (s["processed_files"] | s["processed_images"]) - \
                      (uploaded_names | uploaded_img_names)
            if removed:
                st.warning(f"Detected {len(removed)} removed file(s). Rebuilding.")
                cleanup_ns(ns); new_files = list(uploaded_files); new_images = list(upload_images)

            if not new_files and not new_images and s["vector_store"] is not None:
                st.info("No new files to add.")
            else:
                start = time.time()
                os.makedirs("temp_docs",   exist_ok=True)
                os.makedirs("temp_images", exist_ok=True)
                try:
                    for idx, uf in enumerate(new_files):
                        with st.spinner(f"Processing PDF {uf.name} ({idx+1}/{len(new_files)})…"):
                            fp = os.path.join("temp_docs", uf.name)
                            with open(fp, "wb") as f: f.write(uf.getbuffer())
                            if s["vector_store"] is None:
                                s["vector_store"] = _create_vs(fp, ns)
                            else:
                                s["vector_store"] = _add_vs(s["vector_store"], fp, ns)
                            s["processed_files"].add(uf.name)

                    for idx, img in enumerate(new_images):
                        with st.spinner(f"Processing image {img.name} ({idx+1}/{len(new_images)})…"):
                            ip = os.path.join("temp_images", img.name)
                            with open(ip, "wb") as f: f.write(img.getbuffer())
                            try:
                                cnn = run_pickle_model_prediction(ip)
                                st.success(f"🧠 CNN: {cnn['prediction']} ({cnn['confidence']*100:.1f}%)")
                                ft = f"CNN MODEL PREDICTION:\nClass: {cnn['prediction']}\nConfidence: {cnn['confidence']*100:.2f}%"
                            except Exception as e:
                                st.error(f"CNN failed: {e}"); ft = "CNN prediction failed."
                            try:
                                ft += "\n\nVISION MODEL ANALYSIS:\n" + analyze_image_with_vision_llm(ip)
                            except Exception:
                                ft += "\n\nVision analysis failed."
                            txt_path = os.path.join("temp_docs", f"{img.name}_findings.txt")
                            with open(txt_path, "w", encoding="utf-8") as f: f.write(ft)
                            if s["vector_store"] is None:
                                s["vector_store"] = _create_vs(txt_path, ns)
                            else:
                                s["vector_store"] = _add_vs(s["vector_store"], txt_path, ns)
                            s["processed_images"].add(img.name)

                    if s["vector_store"]:
                        s["rag_chain"] = create_rag_chain(s["vector_store"], s["persona"])
                    s["pdf_processed"] = True
                    st.success(
                        f"✔ Done in {time.time()-start:.2f}s — "
                        f"{len(new_files)} PDF(s), {len(new_images)} image(s)"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    render_qa(s, key_suffix="general")


# ═════════════════════════════════════════════
#  PAGE 2 — Medical (Brain Tumour)
# ═════════════════════════════════════════════

def page_medical():
    ns = "medical"
    s  = _init_ns(ns)
    if s["vector_store"] is None:
        s["vector_store"] = _try_load(ns)
    s["persona"] = "MEDICAL"

    st.title("🏥 Medical Analysis Hub")
    st.write(
        "Dedicated space for medical imaging and reports. "
        "Brain tumour scans → CNN classification → vision LLM analysis → "
        "indexed for Q&A under the **MEDICAL** persona."
    )

    with st.sidebar:
        st.header("🩺 Upload Medical Files")
        med_pdfs   = st.file_uploader("Medical Reports (PDF)", type=["pdf"],
                                       accept_multiple_files=True, key="med_pdfs")
        med_images = st.file_uploader("Medical Images (scans)", type=["png","jpg","jpeg"],
                                       accept_multiple_files=True, key="med_imgs")
        if s["processed_files"] or s["processed_images"]:
            st.info(
                f"📚 {len(s['processed_files'])+len(s['processed_images'])} file(s) loaded"
            )
            with st.expander("View loaded files"):
                for f in sorted(s["processed_files"]):  st.write(f"📄 {f}")
                for f in sorted(s["processed_images"]): st.write(f"🖼️ {f}")

        process_btn = st.button("Process Medical Files", key="med_proc")
        if st.button("Clear All Medical Data", key="med_clear"):
            cleanup_ns(ns); st.success("Medical data cleared!"); st.rerun()

    med_pdfs   = med_pdfs   or []
    med_images = med_images or []

    if process_btn:
        if not med_pdfs and not med_images:
            st.warning("Please upload at least one PDF or image.")
        else:
            uploaded_names    = {f.name for f in med_pdfs}
            uploaded_img_names = {f.name for f in med_images}
            new_files  = [f for f in med_pdfs   if f.name not in s["processed_files"]]
            new_images = [f for f in med_images  if f.name not in s["processed_images"]]
            removed = (s["processed_files"] | s["processed_images"]) - \
                      (uploaded_names | uploaded_img_names)
            if removed:
                st.warning(f"Detected {len(removed)} removed file(s). Rebuilding.")
                cleanup_ns(ns); new_files = list(med_pdfs); new_images = list(med_images)

            if not new_files and not new_images and s["vector_store"] is not None:
                st.info("No new files to add.")
            else:
                start = time.time()
                os.makedirs("temp_docs",   exist_ok=True)
                os.makedirs("temp_images", exist_ok=True)
                try:
                    for idx, uf in enumerate(new_files):
                        with st.spinner(f"Processing report {uf.name} ({idx+1}/{len(new_files)})…"):
                            fp = os.path.join("temp_docs", uf.name)
                            with open(fp, "wb") as f: f.write(uf.getbuffer())
                            if s["vector_store"] is None:
                                s["vector_store"] = _create_vs(fp, ns)
                            else:
                                s["vector_store"] = _add_vs(s["vector_store"], fp, ns)
                            s["processed_files"].add(uf.name)

                    for idx, img in enumerate(new_images):
                        with st.spinner(f"Analysing scan {img.name} ({idx+1}/{len(new_images)})…"):
                            ip = os.path.join("temp_images", img.name)
                            with open(ip, "wb") as f: f.write(img.getbuffer())

                            # ── Step 1: CNN classification ────────────────
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

                            # ── Step 2: Vision LLM analysis ───────────────
                            try:
                                vision_out = analyze_image_with_vision_llm(ip)
                                findings_text += "\n\nVISION MODEL ANALYSIS:\n" + vision_out
                            except Exception:
                                findings_text += "\n\nVision analysis failed."

                            txt_path = os.path.join("temp_docs", f"{img.name}_findings.txt")
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(findings_text)
                            if s["vector_store"] is None:
                                s["vector_store"] = _create_vs(txt_path, ns)
                            else:
                                s["vector_store"] = _add_vs(s["vector_store"], txt_path, ns)
                            s["processed_images"].add(img.name)

                    if s["vector_store"]:
                        s["rag_chain"] = create_rag_chain(s["vector_store"], "MEDICAL")
                    s["pdf_processed"] = True
                    st.success(
                        f"✔ Done in {time.time()-start:.2f}s — "
                        f"{len(new_files)} report(s), {len(new_images)} scan(s)"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    if med_images:
        with st.expander("🔬 Preview Uploaded Scans"):
            cols = st.columns(min(len(med_images), 4))
            for i, img in enumerate(med_images):
                with cols[i % 4]:
                    st.image(img, caption=img.name, use_column_width=True)

    render_qa(s, key_suffix="medical")


# ═════════════════════════════════════════════
#  PAGE 3 — SLI Audio
# ═════════════════════════════════════════════

def page_sli_audio():
    ns = "sli"
    s  = _init_ns(ns)
    if s["vector_store"] is None:
        s["vector_store"] = _try_load(ns)
    s["persona"] = "MEDICAL"

    st.title("🎙️ SLI Speech Analysis")
    st.write(
        "Upload a child's speech WAV file. The pipeline will:\n"
        "1. **Classify** it as *Healthy* or *SLI* (with severity) using the "
        "LANNA-trained model.\n"
        "2. **Extract & visualise** acoustic features (MFCCs, pitch, formants, "
        "jitter/shimmer, spectral features).\n"
        "3. **Transcribe** speech (requires `openai-whisper`).\n"
        "4. **Index** the findings report and let you ask Q&A questions about it."
    )

    if not SLI_AVAILABLE:
        st.error(f"sli_audio_functions.py could not be imported: {_SLI_IMPORT_ERROR}")
        return

    # ── model check & auto-train ──────────────────────────────────────────
    def _model_ready() -> bool:
        return (
            os.path.exists("models/sli_classifier.pkl") and
            os.path.exists("models/sli_scaler.pkl")     and
            os.path.exists("models/sli_meta.json")
        )

    if not _model_ready():
        st.warning("⚠️ SLI classifier model not found in `models/`.")
        st.markdown("### 🏋️ Train the model automatically")
        st.write(
            "Point to your LANNA `Data/` folder and click **Train Now**. "
            "Training runs in the background — logs stream in real time below."
        )

        default_data = os.path.join(os.getcwd(), "Data")
        data_root_input = st.text_input(
            "Path to LANNA `Data/` folder",
            value=default_data,
            key="sli_data_root",
            help="Folder that contains the `Healthy/` and `Patient/` subfolders"
        )

        col_sev, col_est = st.columns(2)
        with col_sev:
            use_severity = st.checkbox(
                "4-class severity model (Healthy / Mild / Moderate / Severe)",
                value=True, key="sli_train_severity"
            )
        with col_est:
            n_estimators = st.slider(
                "Number of trees", min_value=50, max_value=500,
                value=300, step=50, key="sli_n_est"
            )

        train_btn = st.button(
            "🚀 Train Now", key="sli_train_btn", type="primary",
            disabled=not data_root_input.strip()
        )

        if train_btn:
            data_root_val = data_root_input.strip()
            if not os.path.isdir(data_root_val):
                st.error(
                    f"Directory not found: `{data_root_val}`\n\n"
                    "Please enter the correct path to the LANNA `Data/` folder."
                )
            else:
                import subprocess, sys, threading, queue as _queue

                severity_flag = "--severity" if use_severity else "--no_severity"
                cmd = [
                    sys.executable, "train_sli_model.py",
                    "--data_root",    data_root_val,
                    "--model_dir",    "models",
                    "--n_estimators", str(n_estimators),
                    severity_flag,
                ]

                st.info(f"Running:\n```\n{' '.join(cmd)}\n```")
                log_area   = st.empty()
                status_box = st.empty()
                log_lines: list = []

                def _stream(proc, q):
                    for line in iter(proc.stdout.readline, ""):
                        q.put(line)
                    proc.stdout.close()
                    q.put(None)

                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    q: _queue.Queue = _queue.Queue()
                    threading.Thread(target=_stream, args=(proc, q),
                                     daemon=True).start()
                    status_box.info("⏳ Training in progress…")

                    while True:
                        try:
                            line = q.get(timeout=0.2)
                        except _queue.Empty:
                            log_area.code("".join(log_lines[-60:]), language="bash")
                            continue
                        if line is None:
                            break
                        log_lines.append(line)
                        log_area.code("".join(log_lines[-60:]), language="bash")

                    proc.wait()
                    log_area.code("".join(log_lines), language="bash")

                    if proc.returncode == 0 and _model_ready():
                        status_box.success(
                            "✅ Training complete! Model saved to `models/`. "
                            "Reloading…"
                        )
                        time.sleep(1)
                        st.rerun()
                    else:
                        status_box.error(
                            f"❌ Training failed (exit code {proc.returncode}). "
                            "Check the log above for details."
                        )
                except FileNotFoundError:
                    st.error(
                        "`train_sli_model.py` not found in the current directory."
                    )
                except Exception as exc:
                    st.error(f"Unexpected error launching training: {exc}")

        # don't render the rest of the page until model exists
        if not _model_ready():
            return

    model_ready = _model_ready  # callable alias used in sidebar disabled= guard

    # ── sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎧 Upload Audio")
        audio_files = st.file_uploader(
            "Child speech WAV files",
            type=["wav", "mp3", "m4a"],
            accept_multiple_files=True,
            key="sli_wavs"
        )

        transcribe_opt = st.checkbox(
            "Transcribe speech (Whisper)", value=False,
            key="sli_transcribe"
        )
        lang_opt = st.selectbox(
            "Transcription language",
            ["cs (Czech)", "en (English)", "auto"],
            index=0, key="sli_lang"
        )

        if s["processed_files"] or s["processed_images"]:
            st.info(f"📚 {len(s['processed_files'])} audio file(s) analysed")
            with st.expander("Analysed files"):
                for f in sorted(s["processed_files"]): st.write(f"🎙️ {f}")

        process_btn = st.button("Analyse Audio", key="sli_proc",
                                 disabled=not model_ready())
        if st.button("Clear SLI Data", key="sli_clear"):
            cleanup_ns(ns); st.success("SLI data cleared!"); st.rerun()

    audio_files = audio_files or []

    if not audio_files:
        st.info("Upload one or more WAV files in the sidebar to begin.")
        render_qa(s, key_suffix="sli")
        return

    # ── audio player ──────────────────────────────────────────────────────
    with st.expander("🔊 Audio Players", expanded=True):
        for af in audio_files:
            st.caption(af.name)
            st.audio(af, format="audio/wav")

    # ── processing ────────────────────────────────────────────────────────
    if process_btn and model_ready():
        new_files = [f for f in audio_files if f.name not in s["processed_files"]]

        if not new_files:
            st.info("All uploaded files have already been analysed.")
        else:
            os.makedirs("temp_audio", exist_ok=True)
            os.makedirs("temp_docs",  exist_ok=True)

            lang_map = {"cs (Czech)": "cs", "en (English)": "en", "auto": None}
            lang = lang_map.get(lang_opt, "cs")

            for af in new_files:
                st.markdown(f"---\n### 🎙️ Analysing: `{af.name}`")

                # save to disk
                ap = os.path.join("temp_audio", af.name)
                with open(ap, "wb") as f:
                    f.write(af.getbuffer())

                # ── 1. Load & visualise waveform ──────────────────────────
                with st.spinner("Loading waveform…"):
                    try:
                        signal, sr = load_wav(ap)
                        signal_trimmed = trim_silence(signal)

                        col_wave, col_spec = st.columns(2)
                        with col_wave:
                            fig = plot_waveform(
                                signal_trimmed, sr,
                                title=f"Waveform — {af.name}"
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                        with col_spec:
                            fig = plot_spectrogram(
                                signal_trimmed, sr,
                                title="Spectrogram"
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                    except Exception as e:
                        st.error(f"Waveform loading failed: {e}")
                        signal, sr, signal_trimmed = None, 16000, None

                # ── 2. MFCC visualisation ─────────────────────────────────
                if signal_trimmed is not None and len(signal_trimmed) > 1600:
                    with st.spinner("Computing MFCCs…"):
                        try:
                            fig = plot_mfcc(signal_trimmed, sr)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.warning(f"MFCC plot failed: {e}")

                # ── 3. SLI Classification ─────────────────────────────────
                prediction = None
                with st.spinner("Classifying (SLI vs Healthy)…"):
                    try:
                        prediction = predict_sli(ap)

                        label    = prediction["label"]
                        severity = prediction["severity"]
                        conf     = prediction["confidence"] * 100

                        # colour-coded result box
                        if label == "Healthy":
                            st.success(
                                f"✅ **Prediction: {label}** — "
                                f"Confidence: {conf:.1f}%"
                            )
                        else:
                            sev_str = f" — Severity: **{severity}**" if severity else ""
                            st.error(
                                f"⚠️ **Prediction: {label}**{sev_str} — "
                                f"Confidence: {conf:.1f}%"
                            )

                        # probability bar chart + feature radar side by side
                        col_prob, col_radar = st.columns(2)
                        with col_prob:
                            fig = plot_probability_bar(prediction["probabilities"])
                            st.pyplot(fig)
                            plt.close(fig)
                        with col_radar:
                            fig = plot_feature_radar(prediction["features"])
                            st.pyplot(fig)
                            plt.close(fig)

                    except Exception as e:
                        st.error(f"Classification failed: {e}")

                # ── 4. Acoustic feature table ─────────────────────────────
                if prediction is not None:
                    feats = prediction["features"]
                    with st.expander("📊 Full Acoustic Feature Table"):
                        import pandas as pd
                        feat_df = pd.DataFrame({
                            "Feature": FEATURE_NAMES[:len(feats)],
                            "Value":   [f"{v:.4f}" for v in feats]
                        })
                        st.dataframe(feat_df, use_container_width=True,
                                     height=400)

                # ── 5. Transcription ──────────────────────────────────────
                transcription = ""
                if transcribe_opt:
                    with st.spinner("Transcribing with Whisper…"):
                        transcription = transcribe_audio(
                            ap, language=lang or "cs"
                        )
                    if transcription.startswith("["):
                        st.warning(transcription)
                    else:
                        st.subheader("📝 Transcription")
                        st.text_area(
                            "Transcript", transcription,
                            height=120, key=f"trans_{af.name}"
                        )
                else:
                    transcription = "[Transcription not requested]"

                # ── 6. Build findings text & index ────────────────────────
                if prediction is not None:
                    findings_text = build_audio_findings_text(
                        ap, prediction, transcription
                    )
                    txt_path = os.path.join(
                        "temp_docs", f"{af.name}_sli_findings.txt"
                    )
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(findings_text)

                    with st.spinner("Indexing findings for Q&A…"):
                        try:
                            if s["vector_store"] is None:
                                s["vector_store"] = _create_vs(txt_path, ns)
                            else:
                                s["vector_store"] = _add_vs(
                                    s["vector_store"], txt_path, ns
                                )
                            s["rag_chain"] = create_rag_chain(
                                s["vector_store"], "MEDICAL"
                            )
                            st.success("✔ Findings indexed — Q&A ready below")
                        except Exception as e:
                            st.warning(f"Indexing failed: {e}")

                    s["processed_files"].add(af.name)

            st.success(
                f"✔ Analysed {len(new_files)} new audio file(s). "
                f"Total: {len(s['processed_files'])}"
            )

    # ── Q&A ───────────────────────────────────────────────────────────────
    render_qa(s, key_suffix="sli")


# ═════════════════════════════════════════════
#  NAVIGATION
# ═════════════════════════════════════════════

PAGES = {
    "📄 General RAG": page_general,
    "🏥 Medical":     page_medical,
    "🎙️ SLI Audio":  page_sli_audio,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection]()