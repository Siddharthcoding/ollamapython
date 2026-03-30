"""
main.py  —  RAG + Medical + SLI Audio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three pages via sidebar:
  📄  General RAG   – multi-persona PDF / image Q&A
  🏥  Medical       – brain-tumour CNN + vision LLM + PDF reports
  🎙️  SLI Audio     – LANNA hierarchical classifier, task-level profiles,
                       acoustic visualisations, transcription, RAG Q&A
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, io, time, gc, uuid, threading, subprocess, sys
import queue as _queue
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

# ── SLI audio helpers ─────────────────────────────────────────────────────────
try:
    from sli_audio_functions import (
        load_wav, trim_silence,
        predict_from_task_map, predict_single_wav,
        transcribe_audio, build_audio_findings_text,
        plot_waveform, plot_spectrogram, plot_mfcc,
        plot_task_quality, plot_complexity_profile,
        plot_disorder_gauge, plot_binary_donut, plot_severity_probs,
        TASK_ORDER, TASK_LABELS, FEATURE_NAMES, N_FEATURES,
    )
    SLI_AVAILABLE = True
except ImportError as _e:
    SLI_AVAILABLE = False
    _SLI_IMPORT_ERROR = str(_e)

st.set_page_config(page_title="RAG + Medical + SLI", layout="wide")


# ═════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _init_ns(ns: str) -> dict:
    key = f"_state_{ns}"
    if key not in st.session_state:
        st.session_state[key] = {
            "vector_store": None, "rag_chain": None,
            "persona": "MEDICAL" if ns in ("medical","sli") else "EDUCATION",
            "processed_files": set(), "processed_images": set(),
            "last_question": None, "pdf_processed": False,
            "answer_generated": False,
        }
    return st.session_state[key]


def cleanup_ns(ns: str):
    s = st.session_state[f"_state_{ns}"]
    s.update({"vector_store": None, "rag_chain": None, "pdf_processed": False,
               "processed_files": set(), "processed_images": set(),
               "last_question": None, "answer_generated": False})
    gc.collect()
    for f in [f"faiss.index_{ns}", f"faiss_store_{ns}.pkl"]:
        if os.path.exists(f):
            try: os.remove(f)
            except PermissionError: os.rename(f, f"old_{uuid.uuid4()}_{f}")
            except Exception: pass


def _try_load(ns: str):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    idx = f"faiss.index_{ns}"
    if os.path.exists(idx):
        try:
            return FAISS.load_local(idx, OllamaEmbeddings(model="mxbai-embed-large"),
                                    allow_dangerous_deserialization=True)
        except Exception: return None
    return None


def _create_vs(file_path: str, ns: str):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from concurrent.futures import ThreadPoolExecutor
    from supporting_functions import extract_images_from_pdf

    idx = f"faiss.index_{ns}"
    emb = OllamaEmbeddings(model="mxbai-embed-large")
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs = [Document(page_content=content,
                         metadata={"source": os.path.basename(file_path)})]
    else:
        loader = PyMuPDFLoader(file_path)
        docs   = loader.load()
        if not docs: raise ValueError("No text extracted.")
        for doc in docs: doc.metadata["source"] = os.path.basename(file_path)
        def _proc(ip):
            try:
                vt = analyze_image_with_vision_llm(ip)
                return Document(page_content=f"Image Analysis:\n{vt}",
                                metadata={"source": os.path.basename(file_path),
                                          "type": "image"})
            except Exception: return None
        with ThreadPoolExecutor(max_workers=2) as ex:
            image_docs = list(ex.map(_proc, extract_images_from_pdf(file_path)))
        docs.extend([d for d in image_docs if d])

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100).split_documents(docs)
    store = FAISS.from_documents(chunks, emb)
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
        if not docs: raise ValueError("No text extracted.")
        for doc in docs: doc.metadata["source"] = os.path.basename(file_path)
        def _proc(ip):
            try:
                vt = analyze_image_with_vision_llm(ip)
                return Document(page_content=f"Image Analysis:\n{vt}",
                                metadata={"source": os.path.basename(file_path),
                                          "type": "image"})
            except Exception: return None
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
    st.caption(f"Querying across "
               f"{len(s['processed_files'])+len(s['processed_images'])} document(s)")
    user_q = st.text_input("Your question:", key=f"q_{key_suffix}")
    if st.button("Get Answer", key=f"ask_{key_suffix}"):
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
                    answer_text = (response.get("answer") or
                                   response.get("output_text") or str(response))
                    context = (response.get("context") or
                               response.get("source_documents") or [])
                else:
                    answer_text = str(response); context = []
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
                            st.markdown(f"**Doc {i} — Source:** `{meta.get('source','?')}`")
                            st.info(pc)
            except Exception as e:
                st.error(f"Failed to produce answer: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — General RAG
# ═════════════════════════════════════════════════════════════════════════════

def page_general():
    ns = "general"
    s  = _init_ns(ns)
    if s["vector_store"] is None: s["vector_store"] = _try_load(ns)

    st.title("📄 RAG with Ollama & FAISS")
    st.write("Upload multiple PDFs and images and ask questions about their "
             "combined content. Supports incremental indexing.")

    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"],
                                           accept_multiple_files=True, key="gen_pdfs")
        upload_images  = st.file_uploader("Upload Images", type=["png","jpg","jpeg"],
                                           accept_multiple_files=True, key="gen_imgs")
        persona_opts = ("MEDICAL","RESEARCH","BUSINESS","EDUCATION")
        persona_choice = st.selectbox(
            "Select persona", persona_opts,
            index=persona_opts.index(s["persona"]) if s["persona"] in persona_opts else 0,
            key="gen_persona")
        if persona_choice != s["persona"]:
            s["persona"] = persona_choice
            if s["vector_store"]:
                s["rag_chain"] = create_rag_chain(s["vector_store"], s["persona"])
        if s["processed_files"] or s["processed_images"]:
            st.info(f"📚 {len(s['processed_files'])+len(s['processed_images'])} file(s) loaded")
            with st.expander("View loaded files"):
                for f in sorted(s["processed_files"]):  st.write(f"📄 {f}")
                for f in sorted(s["processed_images"]): st.write(f"🖼️ {f}")
        process_btn = st.button("Process Documents", key="gen_proc")
        if st.button("Clear All Documents", key="gen_clear"):
            cleanup_ns(ns); st.success("Cleared!"); st.rerun()

    uploaded_files = uploaded_files or []; upload_images = upload_images or []
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
                os.makedirs("temp_docs", exist_ok=True)
                os.makedirs("temp_images", exist_ok=True)
                try:
                    for idx, uf in enumerate(new_files):
                        with st.spinner(f"Processing PDF {uf.name} ({idx+1}/{len(new_files)})…"):
                            fp = os.path.join("temp_docs", uf.name)
                            with open(fp, "wb") as f: f.write(uf.getbuffer())
                            s["vector_store"] = (_create_vs(fp, ns) if s["vector_store"] is None
                                                 else _add_vs(s["vector_store"], fp, ns))
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
                            try: ft += "\n\nVISION MODEL ANALYSIS:\n" + analyze_image_with_vision_llm(ip)
                            except Exception: ft += "\n\nVision analysis failed."
                            txt_path = os.path.join("temp_docs", f"{img.name}_findings.txt")
                            with open(txt_path, "w", encoding="utf-8") as f: f.write(ft)
                            s["vector_store"] = (_create_vs(txt_path, ns) if s["vector_store"] is None
                                                 else _add_vs(s["vector_store"], txt_path, ns))
                            s["processed_images"].add(img.name)
                    if s["vector_store"]:
                        s["rag_chain"] = create_rag_chain(s["vector_store"], s["persona"])
                    s["pdf_processed"] = True
                    st.success(f"✔ Done in {time.time()-start:.2f}s — "
                               f"{len(new_files)} PDF(s), {len(new_images)} image(s)")
                except Exception as e:
                    st.error(f"Error: {e}")
    render_qa(s, key_suffix="general")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Medical
# ═════════════════════════════════════════════════════════════════════════════

def page_medical():
    ns = "medical"
    s  = _init_ns(ns)
    if s["vector_store"] is None: s["vector_store"] = _try_load(ns)
    s["persona"] = "MEDICAL"

    st.title("🏥 Medical Analysis Hub")
    st.write("Brain tumour scans → CNN → vision LLM → Q&A under MEDICAL persona.")

    with st.sidebar:
        st.header("🩺 Upload Medical Files")
        med_pdfs   = st.file_uploader("Medical Reports (PDF)", type=["pdf"],
                                       accept_multiple_files=True, key="med_pdfs")
        med_images = st.file_uploader("Medical Images (scans)", type=["png","jpg","jpeg"],
                                       accept_multiple_files=True, key="med_imgs")
        if s["processed_files"] or s["processed_images"]:
            st.info(f"📚 {len(s['processed_files'])+len(s['processed_images'])} file(s) loaded")
            with st.expander("View loaded files"):
                for f in sorted(s["processed_files"]):  st.write(f"📄 {f}")
                for f in sorted(s["processed_images"]): st.write(f"🖼️ {f}")
        process_btn = st.button("Process Medical Files", key="med_proc")
        if st.button("Clear All Medical Data", key="med_clear"):
            cleanup_ns(ns); st.success("Cleared!"); st.rerun()

    med_pdfs = med_pdfs or []; med_images = med_images or []
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
                st.warning(f"Detected {len(removed)} removed. Rebuilding.")
                cleanup_ns(ns); new_files = list(med_pdfs); new_images = list(med_images)
            if not new_files and not new_images and s["vector_store"] is not None:
                st.info("No new files to add.")
            else:
                start = time.time()
                os.makedirs("temp_docs", exist_ok=True)
                os.makedirs("temp_images", exist_ok=True)
                try:
                    for idx, uf in enumerate(new_files):
                        with st.spinner(f"Processing {uf.name}…"):
                            fp = os.path.join("temp_docs", uf.name)
                            with open(fp, "wb") as f: f.write(uf.getbuffer())
                            s["vector_store"] = (_create_vs(fp, ns) if s["vector_store"] is None
                                                 else _add_vs(s["vector_store"], fp, ns))
                            s["processed_files"].add(uf.name)
                    for idx, img in enumerate(new_images):
                        with st.spinner(f"Analysing {img.name}…"):
                            ip = os.path.join("temp_images", img.name)
                            with open(ip, "wb") as f: f.write(img.getbuffer())
                            try:
                                cnn = run_pickle_model_prediction(ip)
                                st.success(f"🧠 CNN: **{cnn['prediction']}**")
                                st.info(f"Confidence: {cnn['confidence']*100:.2f}%")
                                ft = (f"CNN MODEL PREDICTION:\n"
                                      f"Predicted Class: {cnn['prediction']}\n"
                                      f"Confidence: {cnn['confidence']*100:.2f}%\n")
                            except Exception as e:
                                st.error(f"CNN failed: {e}"); ft = "CNN prediction failed.\n"
                            try: ft += "\n\nVISION MODEL ANALYSIS:\n" + analyze_image_with_vision_llm(ip)
                            except Exception: ft += "\n\nVision analysis failed."
                            txt_path = os.path.join("temp_docs", f"{img.name}_findings.txt")
                            with open(txt_path, "w", encoding="utf-8") as f: f.write(ft)
                            s["vector_store"] = (_create_vs(txt_path, ns) if s["vector_store"] is None
                                                 else _add_vs(s["vector_store"], txt_path, ns))
                            s["processed_images"].add(img.name)
                    if s["vector_store"]:
                        s["rag_chain"] = create_rag_chain(s["vector_store"], "MEDICAL")
                    s["pdf_processed"] = True
                    st.success(f"✔ Done in {time.time()-start:.2f}s")
                except Exception as e:
                    st.error(f"Error: {e}")

    if med_images:
        with st.expander("🔬 Preview Scans"):
            cols = st.columns(min(len(med_images), 4))
            for i, img in enumerate(med_images):
                with cols[i % 4]: st.image(img, caption=img.name, use_column_width=True)
    render_qa(s, key_suffix="medical")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — SLI Audio
# ═════════════════════════════════════════════════════════════════════════════

def page_sli_audio():
    ns = "sli"
    s  = _init_ns(ns)
    if s["vector_store"] is None: s["vector_store"] = _try_load(ns)
    s["persona"] = "MEDICAL"

    st.title("🎙️ SLI Speech Analysis")
    st.markdown(
        "Analyse children's speech recordings for **Specific Language Impairment (SLI)**. "
        "Upload WAV files grouped by task type — the system uses the LANNA progression logic:\n"
        "- **Healthy**: good quality across all tasks\n"
        "- **Mild SLI**: vowels/consonants ok, words/sentences break down\n"
        "- **Moderate SLI**: vowels ok, consonants/syllables break down\n"
        "- **Severe SLI**: even isolated vowel phonation is impaired")

    if not SLI_AVAILABLE:
        st.error(f"sli_audio_functions.py import error: {_SLI_IMPORT_ERROR}")
        return

    def _model_ready():
        from sli_audio_functions import (SLI_BINARY_CLF_PATH, SLI_BINARY_SCL_PATH,
            SLI_SEVERITY_CLF_PATH, SLI_SEVERITY_SCL_PATH, SLI_META_PATH)
        return all(os.path.exists(p) for p in [
            SLI_BINARY_CLF_PATH, SLI_BINARY_SCL_PATH,
            SLI_SEVERITY_CLF_PATH, SLI_SEVERITY_SCL_PATH, SLI_META_PATH])

    # ── sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎛️ Options")
        transcribe_opt = st.checkbox("Transcribe speech (Whisper)", value=False, key="sli_tr")
        lang_opt = st.selectbox("Language", ["cs (Czech)", "en (English)", "auto"],
                                index=0, key="sli_lang")
        if _model_ready():
            try:
                import json as _json
                meta = _json.load(open("models/sli_meta.json"))
                st.success("✅ Trained model loaded")
                bm = meta.get("binary_metrics", {})
                sm = meta.get("severity_metrics", {})
                if bm:
                    st.caption(f"Binary   — BalAcc: {bm.get('balanced_accuracy',0):.3f}  F1: {bm.get('f1_macro',0):.3f}")
                if sm and sm.get("f1_macro",0) > 0:
                    st.caption(f"Severity — BalAcc: {sm.get('balanced_accuracy',0):.3f}  F1: {sm.get('f1_macro',0):.3f}")
                sd = meta.get("severity_dist", {})
                if sd:
                    total = sum(sd.values())
                    dist_str = "  ".join(f"{k.capitalize()}:{v}" for k,v in sorted(sd.items()))
                    st.caption(f"Training: {meta.get('n_healthy',0)}H / {meta.get('n_patients',0)}P  [{dist_str}]")
            except Exception: pass
        else:
            st.info("🔢 Rule-based mode\n(no ML model needed)")

        with st.expander("🏋️ Train on LANNA data", expanded=False):
            data_root_input = st.text_input("Data/ folder", value=os.path.join(os.getcwd(),"Data"), key="sli_data")
            use_nn  = st.checkbox("Test MLP Neural Network", value=False, key="sli_nn")
            n_trees = st.slider("Trees", 50, 500, 300, 50, key="sli_trees")
            if st.button("🚀 Train", key="sli_train_btn", type="primary"):
                if not os.path.isdir(data_root_input):
                    st.error(f"Not found: {data_root_input}")
                else:
                    cmd = [sys.executable, "train_sli_model.py",
                           "--data_root", data_root_input, "--model_dir", "models",
                           "--n_estimators", str(n_trees)]
                    if use_nn: cmd.append("--use_nn")
                    log_area = st.empty(); log_lines = []
                    def _stream(proc, q):
                        for line in iter(proc.stdout.readline, ""): q.put(line)
                        proc.stdout.close(); q.put(None)
                    try:
                        import queue as _queue
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT, text=True, bufsize=1)
                        q = _queue.Queue()
                        threading.Thread(target=_stream, args=(proc, q), daemon=True).start()
                        while True:
                            try: line = q.get(timeout=0.2)
                            except _queue.Empty:
                                log_area.code("".join(log_lines[-60:]), language="bash"); continue
                            if line is None: break
                            log_lines.append(line)
                            log_area.code("".join(log_lines[-60:]), language="bash")
                        proc.wait()
                        if proc.returncode == 0: st.success("✅ Done!"); time.sleep(1); st.rerun()
                        else: st.error(f"Failed (exit {proc.returncode})")
                    except Exception as exc: st.error(f"Error: {exc}")

        if s["processed_files"]: st.info(f"📚 {len(s['processed_files'])} session(s)")
        if st.button("🗑️ Clear", key="sli_clear"): cleanup_ns(ns); st.success("Cleared!"); st.rerun()

    lang_map = {"cs (Czech)": "cs", "en (English)": "en", "auto": None}
    lang = lang_map.get(lang_opt, "cs")

    # ── UPLOAD MODE SELECTION ─────────────────────────────────────────────
    st.subheader("📁 Upload Speech Recordings")

    upload_mode = st.radio(
        "Upload mode",
        ["🏷️ Label by task type (best accuracy — matches LANNA structure)",
         "📋 Unlabelled WAVs (quick screen — assigned to sentence task)"],
        key="sli_upload_mode")

    if "Label by task" in upload_mode:
        _render_labelled_upload(s, ns, transcribe_opt, lang)
    else:
        _render_quick_upload(s, ns, transcribe_opt, lang)

    render_qa(s, key_suffix="sli")


def _render_labelled_upload(s, ns, transcribe_opt, lang):
    """Upload mode: user assigns each file (or batch) to a task type."""
    st.markdown(
        "Upload WAV files for each task. You can upload multiple files per task — "
        "they will be averaged. Tasks you skip will be marked as 'not tested'.")
    st.caption("💡 Tip: upload all recordings from one child session for most accurate results.")

    task_wav_map = {}
    cols = st.columns(2)
    for i, task in enumerate(TASK_ORDER):
        with cols[i % 2]:
            label_str = TASK_LABELS.get(task, task)
            complexity = ("🟢 Easy" if task in ["SAMHOL","SOUHL"]
                          else "🟡 Medium" if task in ["1SL","2SL"]
                          else "🔴 Hard")
            files = st.file_uploader(
                f"{complexity}  **{label_str}** (`{task}`)",
                type=["wav"], accept_multiple_files=True,
                key=f"sli_task_{task}")
            if files:
                task_wav_map[task] = files

    n_tasks = len(task_wav_map)
    n_files = sum(len(v) for v in task_wav_map.values())

    if task_wav_map:
        st.info(f"📎 {n_files} file(s) across {n_tasks} task(s) ready for analysis")
        with st.expander("🔊 Audio preview", expanded=False):
            for task, files in task_wav_map.items():
                st.caption(f"**{TASK_LABELS.get(task,task)}**")
                for f in files: st.audio(f)

    analyse_btn = st.button(
        f"🔍 Analyse ({n_files} file(s), {n_tasks} task(s))" if task_wav_map else "🔍 Analyse",
        type="primary", disabled=not task_wav_map, key="sli_label_btn")

    if analyse_btn and task_wav_map:
        os.makedirs("temp_audio", exist_ok=True)
        saved_map = {}
        for task, files in task_wav_map.items():
            saved_paths = []
            for f in files:
                p = os.path.join("temp_audio", f"{task}_{f.name}")
                with open(p, "wb") as fh: fh.write(f.getbuffer())
                saved_paths.append(p)
            saved_map[task] = saved_paths

        all_wavs = [p for paths in saved_map.values() for p in paths]
        session_name = f"{n_tasks} tasks ({list(task_wav_map.keys())[0]}…)"

        with st.spinner(f"Extracting per-task features from {n_files} file(s)…"):
            try:
                result = predict_from_task_map(saved_map)
            except Exception as e:
                st.error(f"Analysis failed: {e}"); return

        _render_sli_results(result, session_name, all_wavs, s, ns, transcribe_opt, lang)


def _render_quick_upload(s, ns, transcribe_opt, lang):
    """Quick upload: unlabelled WAVs assigned to VSL (sentence) task."""
    st.caption(
        "Files are assigned to the **Sentences** task (hardest level). "
        "For better accuracy, use the labelled upload mode above.")

    files = st.file_uploader(
        "WAV files", type=["wav"], accept_multiple_files=True, key="sli_quick_wavs")
    files = files or []

    if files:
        with st.expander(f"🔊 Audio ({len(files)} file(s))", expanded=False):
            for f in files: st.caption(f.name); st.audio(f)

    btn = st.button(
        f"🔍 Quick Screen ({len(files)} file(s))" if files else "🔍 Quick Screen",
        type="primary", disabled=not files, key="sli_quick_btn")

    if btn and files:
        os.makedirs("temp_audio", exist_ok=True)
        paths = []
        for f in files:
            p = os.path.join("temp_audio", f.name)
            with open(p, "wb") as fh: fh.write(f.getbuffer())
            paths.append(p)

        # Assign to VSL (most complex task — conservative)
        task_map = {"VSL": paths}
        session  = f"Quick screen ({files[0].name})"

        with st.spinner(f"Analysing {len(files)} file(s) as sentence task…"):
            try:
                result = predict_from_task_map(task_map)
            except Exception as e:
                st.error(f"Analysis failed: {e}"); return

        st.info("ℹ️ Only the Sentences (VSL) slot was tested. "
                "Use labelled upload for full task-level profiling.")
        _render_sli_results(result, session, paths, s, ns, transcribe_opt, lang)


def _render_sli_results(result, session_name, wav_paths, s, ns,
                         transcribe_opt, lang):
    label         = result["label"]
    severity      = result.get("severity")
    disorder_score = result["disorder_score"]
    task_scores   = result.get("task_scores", {})

    st.markdown("---")
    st.subheader(f"📊 Results — {session_name}")

    # 1. Disorder gauge
    fig = plot_disorder_gauge(disorder_score, label, severity)
    st.pyplot(fig); plt.close(fig)

    # 2. Badge
    score_pct = disorder_score * 100
    if label == "Healthy":
        st.success(f"✅ **Healthy**  —  Disorder score: {score_pct:.0f}/100")
    elif (severity or "").lower() == "mild":
        st.warning(f"⚠️ **SLI — Mild**  —  Disorder score: {score_pct:.0f}/100")
    elif (severity or "").lower() == "moderate":
        st.error(f"🔶 **SLI — Moderate**  —  Disorder score: {score_pct:.0f}/100")
    else:
        st.error(f"🔴 **SLI — Severe**  —  Disorder score: {score_pct:.0f}/100")

    st.caption("Severity: determined from RELATIVE quality drop across task complexity (consonants → syllables → words → sentences)")

    # 3. Classification probabilities
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_binary_donut(result["binary_probs"])
        st.pyplot(fig); plt.close(fig)
    with col2:
        sp = result.get("severity_probs")
        if sp:
            fig = plot_severity_probs(sp)
            if fig: st.pyplot(fig); plt.close(fig)
        else:
            st.info("Severity chart: available for SLI cases")

    # 4. Task quality bar chart  (CORE view)
    st.subheader("📊 Task-Level Quality Profile")
    fig = plot_task_quality(task_scores, label, severity or "")
    st.pyplot(fig); plt.close(fig)

    # 5. Complexity profile line chart
    fig2 = plot_complexity_profile(task_scores)
    if fig2:
        st.subheader("📈 Quality Across Complexity Levels")
        st.pyplot(fig2); plt.close(fig2)

    # Complexity breakdown
    from sli_audio_functions import EASY_TASKS, MEDIUM_TASKS, HARD_TASKS
    present = {t: s for t, s in task_scores.items() if s is not None}
    if present:
        colA, colB, colC = st.columns(3)
        easy_scores = [s for t,s in present.items() if t in EASY_TASKS]
        mid_scores  = [s for t,s in present.items() if t in MEDIUM_TASKS]
        hard_scores = [s for t,s in present.items() if t in HARD_TASKS]
        import numpy as _np
        with colA:
            q = float(_np.mean(easy_scores)) if easy_scores else None
            emoji = "🟢" if q and q>=0.65 else "🟡" if q and q>=0.45 else "🔴" if q else "⚪"
            st.metric("Easy (vowels/consonants)", f"{q*100:.0f}%" if q else "—", help="SAMHOL, SOUHL")
        with colB:
            q2 = float(_np.mean(mid_scores)) if mid_scores else None
            emoji2 = "🟢" if q2 and q2>=0.55 else "🟡" if q2 and q2>=0.40 else "🔴" if q2 else "⚪"
            st.metric("Medium (syllables/2-syl)", f"{q2*100:.0f}%" if q2 else "—", help="1SL, 2SL")
        with colC:
            q3 = float(_np.mean(hard_scores)) if hard_scores else None
            emoji3 = "🟢" if q3 and q3>=0.50 else "🟡" if q3 and q3>=0.35 else "🔴" if q3 else "⚪"
            st.metric("Hard (words/sentences)", f"{q3*100:.0f}%" if q3 else "—", help="3SL, 4SL, VSL")

    # 6. Clinical profile
    with st.expander("📋 Clinical Profile & Therapy Recommendations", expanded=True):
        st.text(result.get("profile", "No profile generated."))

    # 7. Acoustic visualisations
    if wav_paths:
        viz_path = wav_paths[0]
        try:
            sig, sr = load_wav(viz_path)
            sig_t   = trim_silence(sig, sr)
            with st.expander("🔉 Acoustic Visualisations", expanded=False):
                st.caption(f"Sample: {os.path.basename(viz_path)}")
                c1, c2 = st.columns(2)
                with c1:
                    fig = plot_waveform(sig_t, sr, "Waveform"); st.pyplot(fig); plt.close(fig)
                with c2:
                    fig = plot_spectrogram(sig_t, sr, "Spectrogram"); st.pyplot(fig); plt.close(fig)
                if len(sig_t) > sr * 0.2:
                    fig = plot_mfcc(sig_t, sr); st.pyplot(fig); plt.close(fig)
        except Exception as e:
            st.caption(f"(Acoustic viz: {e})")

    # 8. Per-task feature table
    with st.expander("📈 Per-Task Acoustic Detail", expanded=False):
        import pandas as pd
        from sli_audio_functions import TASK_FEAT_DIM, _PER_TASK_NAMES
        feats = result.get("features")
        if feats is not None:
            rows = []
            for i, task in enumerate(TASK_ORDER):
                off = i * TASK_FEAT_DIM
                block = feats[off: off+TASK_FEAT_DIM]
                if block.sum() == 0: continue
                tq = task_scores.get(task)
                row = {"Task": TASK_LABELS.get(task, task),
                       "Quality %": f"{tq*100:.0f}%" if tq else "—",
                       "Voiced ratio": f"{feats[off]*100:.0f}%",
                       "HNR (dB)": f"{feats[off+1]:.1f}",
                       "Jitter (%)": f"{feats[off+2]:.2f}",
                       "Shimmer (%)": f"{feats[off+3]:.1f}",
                       "F2/F1": f"{feats[off+4]:.2f}",
                       "Speech rate": f"{feats[off+5]:.2f}"}
                rows.append(row)
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

    # 9. Transcription
    transcription = ""
    if transcribe_opt and wav_paths:
        with st.spinner("Transcribing…"):
            transcription = transcribe_audio(wav_paths[0], language=lang or "cs")
        if transcription.startswith("["):
            st.warning(transcription)
        else:
            st.subheader("📝 Transcription")
            st.text_area("Transcript", transcription, height=120,
                         key=f"trans_{session_name[:30]}")

    # 10. Index for RAG
    findings = build_audio_findings_text(session_name, result, transcription)
    os.makedirs("temp_docs", exist_ok=True)
    safe = session_name.replace("/","_").replace("\\","_").replace(" ","_")[:40]
    txt_path = os.path.join("temp_docs", f"{safe}_sli_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f: f.write(findings)
    try:
        s["vector_store"] = (_create_vs(txt_path, ns) if s["vector_store"] is None
                             else _add_vs(s["vector_store"], txt_path, ns))
        s["rag_chain"] = create_rag_chain(s["vector_store"], "MEDICAL")
        st.success("✔ Report indexed — ask questions below")
    except Exception as e:
        st.caption(f"(RAG indexing: {e})")
    s["processed_files"].add(session_name)



# ═════════════════════════════════════════════════════════════════════════════
#  NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════

PAGES = {
    "📄 General RAG": page_general,
    "🏥 Medical":     page_medical,
    "🎙️ SLI Audio":  page_sli_audio,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection]()