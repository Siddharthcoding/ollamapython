import os
import pickle
import uuid
import shutil
import tempfile
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import json
import stat

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms import Ollama
import ollama
from functools import partial
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import fitz
from langchain_core.documents import Document

# ----------------------------------------------------------------
# Config / constants
# ----------------------------------------------------------------
llm_model = "llama3.2:1b"
llm_model1 = "deepseek-r1:8b"
embedding_model = "mxbai-embed-large"
FAISS_INDEX_PATH = "faiss.index"
FAISS_STORE_PATH = "faiss_store.pkl"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "brain_tumor_class_names.json")
DATASET_DIR = "brain_tumor_dataset"
DEFAULT_TRAIN_EPOCHS = 8

embeddings = OllamaEmbeddings(model=embedding_model)

def _load_docs_from_file(file_path):
    """
    Return a list of langchain Document objects for given file path.
    Supports PDFs (via PyMuPDFLoader) and plain text files.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        for d in docs:
            if not d.metadata:
                d.metadata = {}
            d.metadata["source"] = os.path.basename(file_path)
        return docs
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            with open(file_path, "rb") as f:
                txt = f.read().decode("utf-8", errors="ignore")
        doc = Document(page_content=txt, metadata={"source": os.path.basename(file_path)})
        return [doc]

# -------------------------
# Basic RAG utilities
# -------------------------
def try_load_faiss_store():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            embeddings_local = OllamaEmbeddings(model=embedding_model)
            return FAISS.load_local(FAISS_INDEX_PATH, embeddings_local, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None

def process_image(image_path, file_path):
    try:
        vision_text = analyze_image_with_vision_llm(image_path)
        return Document(
            page_content=f"Image Analysis:\n{vision_text}",
            metadata={
                "source": os.path.basename(file_path),
                "type": "image"
            }
        )
    except Exception as e:
        print(f"Image extraction failed: {e}")
        return None

def create_vector_store(file_path):
    docs = _load_docs_from_file(file_path)
    if not docs:
        raise ValueError("No text extracted from file.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    faiss_store = FAISS.from_documents(chunks, embeddings)
    faiss_store.save_local(FAISS_INDEX_PATH)
    with open(FAISS_STORE_PATH, "wb") as f:
        pickle.dump({"embedding": embedding_model}, f)
    return faiss_store

def add_to_vector_store(existing_store, file_path):
    docs = _load_docs_from_file(file_path)
    if not docs:
        raise ValueError("No text extracted from file.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    existing_store.add_documents(chunks)
    existing_store.save_local(FAISS_INDEX_PATH)
    return existing_store

def add_text_to_vector_store(text, source="text_input", existing_store=None):
    """
    Add a plain text string as a document into the vector store.
    If existing_store is None, create a new FAISS store and return it.
    """
    doc = Document(page_content=text, metadata={"source": source})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents([doc])
    if existing_store is None:
        faiss_store = FAISS.from_documents(chunks, embeddings)
        faiss_store.save_local(FAISS_INDEX_PATH)
        with open(FAISS_STORE_PATH, "wb") as f:
            pickle.dump({"embedding": embedding_model}, f)
        return faiss_store
    else:
        existing_store.add_documents(chunks)
        existing_store.save_local(FAISS_INDEX_PATH)
        return existing_store

def create_rag_chain(vector_store, persona):
    if persona == "BUSINESS":
        llm = Ollama(model="qwen2.5:7b", temperature=0.4)
    elif persona == "RESEARCH":
        llm = Ollama(model="deepseek-r1:1.5b", temperature=0.2)
    elif persona == "MEDICAL":
        llm = Ollama(model="deepseek-r1:1.5b", temperature=0.1)
    else:
        llm = Ollama(model="llama3.2:1b", temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", set_persona(persona)),
        ("human", """
You must answer STRICTLY using only the provided context below.

If the context does NOT contain sufficient information,
respond exactly with:
"The document does not contain sufficient medical information."

Do NOT:
- Invent diseases
- Invent lab values
- Assume symptoms
- Provide prescriptions

Context:
{context}

User Question:
{input}

Structured Answer:
""")
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def set_persona(choice):
    if choice == "MEDICAL":
        return """
You are a clinical medical reasoning assistant.

STRICT MEDICAL RULES:
- Use ONLY information present in the provided context.
- Do NOT make definitive diagnoses.
- Use probability-based language (may indicate, could suggest).
- Do NOT prescribe medications.
- If insufficient data, respond exactly:
  "The document does not contain sufficient medical information."
- Always recommend consulting a licensed physician.

You must respond in this structured format:

Patient Summary:
Observed Symptoms:
Abnormal Findings:
Possible Conditions (ranked by likelihood):
Recommended Tests:
Suggested Care Plan:
Urgency Level (Low / Moderate / High / Emergency):
Confidence Level:
"""
    elif choice == "RESEARCH":
        return "You are a research expert. Your answers must be highly technical, citation-style."
    elif choice == "BUSINESS":
        return "You are a business financial analyst and advisor."
    elif choice == "EDUCATION":
        return "You are a friendly teacher."
    return ""

# --------------------------------------------------
# Vision LLM image analysis
# --------------------------------------------------
def image_resize(image_path, max_size=448):
    """
    Resize image in-place to fit within max_size (keeping aspect ratio).
    This avoids sending or displaying huge images.
    """
    try:
        img = Image.open(image_path)
        img.thumbnail((max_size, max_size))
        img.save(image_path)
    except Exception:
        pass

def analyze_image_with_vision_llm(image_path):
    prompt = """
You are an advanced document and medical image analysis assistant. Examine the provided image carefully and extract all relevant information.

1. Identify the image type: MRI, CT, X-ray, Ultrasound, chart, table, diagram, or text.
2. Extract visible text exactly as written.
3. Describe charts, tables, or diagrams, summarizing key points.
4. For medical images:
   - Note visible anatomical structures and abnormalities.
   - Provide a ranked differential diagnosis based only on visible findings (High / Moderate / Low likelihood).
   - Mention uncertainty or limitations.
5. Ignore irrelevant visual details.

STRICT RULES:
- Do not invent measurements, findings, or patient information.
- Do not give definitive diagnoses or treatment advice.
- Clearly state uncertainty where applicable.
"""
    image_resize(image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    resp = ollama.chat(
        model="llava-phi3:3.8b",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_bytes]
        }],
        options={
            "num_predict": 400,
            "temperature": 0.2
        }
    )
    return resp["message"]["content"]

def extract_images_from_pdf(file_path, output_folder="temp_images"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(file_path)
    image_path = []
    for page_ind in range(len(doc)):
        page = doc[page_ind]
        image_list = page.get_images(full=True)
        for img_ind, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image['image']
            image_ext = base_image['ext']
            if base_image.get("width", 0) < 300 or base_image.get("height", 0) < 300:
                continue
            image_filename = f"{output_folder}/{uuid.uuid4()}.{image_ext}"
            with open(image_filename, "wb") as f:
                f.write(image_bytes)
            if os.path.getsize(image_filename) > 50_000:
                image_path.append(image_filename)
    return image_path

# ===================================================
# Brain tumor dataset utilities (Kaggle + synthetic)
# ===================================================
def _dataset_has_images(dataset_dir=DATASET_DIR):
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                return True
    return False

def _write_kaggle_json_from_env():
    """
    If KAGGLE_USERNAME and KAGGLE_KEY exist in env, write ~/.kaggle/kaggle.json so the kaggle CLI works.
    """
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        return False
    home = os.path.expanduser("~")
    kaggle_dir = os.path.join(home, ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    data = {"username": username, "key": key}
    try:
        with open(kaggle_json, "w") as f:
            json.dump(data, f)
        try:
            os.chmod(kaggle_json, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass
        return True
    except Exception:
        return False

def _download_kaggle_dataset(status=None):
    """
    Download sartajbhuvaji/brain-tumor-classification-mri from Kaggle using kaggle CLI.
    Writes ~/.kaggle/kaggle.json if env vars present.
    """
    slug = "sartajbhuvaji/brain-tumor-classification-mri"
    tmp = "kaggle_tmp"
    try:
        # ensure kaggle config
        _write_kaggle_json_from_env()
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        if status:
            status.text("Downloading dataset from Kaggle...")
        subprocess.check_call(["kaggle", "datasets", "download", "-d", slug, "-p", tmp, "--unzip"])
        train_root = os.path.join(DATASET_DIR, "train")
        if os.path.exists(train_root):
            shutil.rmtree(train_root)
        os.makedirs(train_root, exist_ok=True)

        expected_classes = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
        found_any = False
        for root, dirs, files in os.walk(tmp):
            for d in dirs:
                if d.lower() in expected_classes:
                    src = os.path.join(root, d)
                    dst = os.path.join(train_root, d.lower())
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    found_any = True

        if not found_any:
            for entry in os.listdir(tmp):
                p = os.path.join(tmp, entry)
                if os.path.isdir(p):
                    imgs = [f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    if imgs:
                        dst = os.path.join(train_root, entry)
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(p, dst)
                        found_any = True

        shutil.rmtree(tmp)
        if not found_any:
            if status:
                status.text("Downloaded, but no class folders detected.")
            return False

        if status:
            status.text("Kaggle dataset downloaded and prepared.")
        return True

    except subprocess.CalledProcessError as e:
        if status:
            status.text(f"Kaggle CLI failed: {e}")
        if os.path.exists(tmp):
            try:
                shutil.rmtree(tmp)
            except Exception:
                pass
        return False
    except Exception as e:
        if status:
            status.text(f"Kaggle download failed: {e}")
        if os.path.exists(tmp):
            try:
                shutil.rmtree(tmp)
            except Exception:
                pass
        return False

def _generate_synthetic_dataset(dataset_dir=DATASET_DIR, n_per_class=300, img_size=224):
    """
    Generate synthetic 4-class dataset (simple simulation).
    """
    np.random.seed(42)
    classes = ["glioma", "meningioma", "pituitary_tumor", "no_tumor"]
    train_root = os.path.join(dataset_dir, "train")
    if os.path.exists(train_root):
        shutil.rmtree(train_root)
    for cl in classes:
        os.makedirs(os.path.join(train_root, cl), exist_ok=True)

    for cl in classes:
        for i in range(n_per_class):
            base = np.random.normal(loc=0.5, scale=0.12, size=(img_size, img_size))
            base = np.clip(base, 0, 1)
            img = Image.fromarray(np.uint8(base * 255)).convert("L")
            if cl != "no_tumor":
                draw = ImageDraw.Draw(img)
                rx = np.random.randint(img_size//12, img_size//6)
                ry = np.random.randint(img_size//12, img_size//6)
                cx = np.random.randint(rx, img_size - rx)
                cy = np.random.randint(ry, img_size - ry)
                bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
                intensity = int(180 + 50 * np.random.rand())
                draw.ellipse(bbox, fill=intensity)
                img = img.filter(ImageFilter.GaussianBlur(radius=2))
            img_rgb = Image.merge("RGB", (img, img, img))
            fname = os.path.join(train_root, cl, f"{cl}_{i}.jpg")
            img_rgb.save(fname, quality=85)

# ===================================================
# Training (transfer learning using MobileNetV2)
# ===================================================
def _train_transfer_model(dataset_dir=DATASET_DIR, model_out_path=MODEL_PATH, epochs=DEFAULT_TRAIN_EPOCHS,
                          progress_bar=None, status=None):
    """
    Train a transfer-learning model (MobileNetV2 backbone) for multi-class classification.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception as e:
        raise RuntimeError("TensorFlow is required for training. Install tensorflow to continue.") from e

    train_dir = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")

    img_size = (224, 224)
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        validation_split=0.15,
        subset="training",
        seed=123
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        validation_split=0.15,
        subset="validation",
        seed=123
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Base model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
    base_model.trainable = False  # freeze for initial training

    inputs = layers.Input(shape=(224,224,3))
    x = layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Simple progress callback (updates progress bar if provided)
    class _ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, pb, status_box, total_epochs):
            super().__init__()
            self.pb = pb
            self.status_box = status_box
            self.total = total_epochs

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            frac = int(((epoch + 1) / self.total) * 100)
            if self.pb is not None:
                try:
                    self.pb.progress(min(frac, 100))
                except Exception:
                    pass
            if self.status_box is not None:
                acc = logs.get("accuracy", 0)
                val_acc = logs.get("val_accuracy", 0)
                loss = logs.get("loss", 0)
                val_loss = logs.get("val_loss", 0)
                self.status_box.text(f"Epoch {epoch+1}/{self.total} — loss:{loss:.3f}, acc:{acc:.3f}, val_loss:{val_loss:.3f}, val_acc:{val_acc:.3f}")

    callbacks = [_ProgressCallback(progress_bar, status, epochs)]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    model.save(model_out_path)

    # Save class names for future mapping
    class_names_file = CLASS_NAMES_PATH
    with open(class_names_file, "w") as f:
        json.dump(class_names, f)

def ensure_brain_tumor_model_trained(progress_bar=None, status=None, epochs=DEFAULT_TRAIN_EPOCHS):
    """
    Ensure a model exists: if not, attempt Kaggle download; if that fails, generate synthetic dataset,
    then train the transfer-learning model.
    """
    # If model already exists, nothing to do
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
        if progress_bar:
            progress_bar.progress(100)
        if status:
            status.text("Model already trained and available.")
        return

    # Ensure dataset
    if not _dataset_has_images(DATASET_DIR):
        if status:
            status.text("No dataset found locally — attempting Kaggle download...")
        ok = _download_kaggle_dataset(status=status)
        if not ok:
            if status:
                status.text("Kaggle download failed or not configured — generating synthetic dataset.")
            _generate_synthetic_dataset(DATASET_DIR, n_per_class=300)
            if status:
                status.text("Synthetic dataset generated.")

    # Train model
    _train_transfer_model(DATASET_DIR, MODEL_PATH, epochs=epochs, progress_bar=progress_bar, status=status)

# ===================================================
# Prediction + Grad-CAM (robust overlay)
# ===================================================
def _load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            return json.load(f)
    # fallback: attempt to read folder names under DATASET_DIR/train
    train_dir = os.path.join(DATASET_DIR, "train")
    if os.path.exists(train_dir):
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        return classes
    return ["class_0"]

def _find_last_conv_layer_name(model):
    """
    Attempt to find the last Conv2D layer name (search nested models).
    Returns layer.name or None.
    """
    try:
        import tensorflow as tf
    except Exception:
        return None

    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for inner in reversed(layer.layers):
                if isinstance(inner, tf.keras.layers.Conv2D) or 'conv' in inner.name.lower():
                    return inner.name
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            return layer.name
    return None

def run_brain_tumor_scratch_model(image_path, model_path=MODEL_PATH):
    """
    Load saved model and run prediction & Grad-CAM.
    Returns dict:
      - prediction: class name
      - confidence: float
      - heatmap_path: path or None
      - note: optional
      - analysis_text: optional fallback analysis
    """
    result = {"heatmap_path": None}

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")

    # If no model exists, fallback to Vision-LLM analysis (non-diagnostic)
    if not os.path.exists(model_path):
        result["note"] = f"No local model found at {model_path}."
        try:
            result["analysis_text"] = analyze_image_with_vision_llm(image_path)
        except Exception:
            result["analysis_text"] = "Vision LLM analysis failed."
        return result

    try:
        import tensorflow as tf
    except Exception:
        result["note"] = "TensorFlow not installed. Install tensorflow to run local model predictions."
        try:
            result["analysis_text"] = analyze_image_with_vision_llm(image_path)
        except Exception:
            result["analysis_text"] = "Vision LLM analysis failed."
        return result

    # load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        result["note"] = f"Failed to load model: {e}"
        try:
            result["analysis_text"] = analyze_image_with_vision_llm(image_path)
        except Exception:
            result["analysis_text"] = "Vision LLM analysis failed."
        return result

    class_names = _load_class_names()

    target_size = (224, 224)
    img_resized = img.resize(target_size)
    x = np.array(img_resized).astype("float32") / 255.0
    x_input = np.expand_dims(x, axis=0)

    # prediction
    preds = model.predict(x_input)
    if preds.ndim == 1:
        # degrade to binary style
        probs = np.array([1 - preds[0], preds[0]])
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    else:
        probs = preds[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

    predicted_class = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    result["prediction"] = predicted_class
    # small calibration: clip to [0,1]
    result["confidence"] = float(np.clip(confidence, 0.0, 1.0))

    # Grad-CAM (robust)
    try:
        import cv2
        import tensorflow as tf

        last_conv_name = _find_last_conv_layer_name(model)
        if not last_conv_name:
            result["note"] = result.get("note", "") + " No conv layer found for Grad-CAM."
            return result

        # Create model that outputs conv layer and final predictions
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_name).output, model.output])

        inp = tf.convert_to_tensor(x_input)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(inp)
            loss = predictions[:, idx]

        grads = tape.gradient(loss, conv_outputs)  # shape (1, h, w, channels)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))  # channels
        conv_outputs = conv_outputs[0]  # h, w, channels

        # weight channels by pooled grads
        weighted = conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :]
        heatmap = tf.reduce_sum(weighted, axis=-1)

        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val.numpy() > 0:
            heatmap = heatmap / max_val
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, target_size)

        try:
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_uint8 = cv2.GaussianBlur(heatmap_uint8, (7,7), 0)
        except Exception:
            heatmap_uint8 = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET) 

        img_np = np.array(img_resized)
        try:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        except Exception:
            img_bgr = img_np

        alpha = 0.5
        beta = 1 - alpha
        if heatmap_color.shape[:2] != img_bgr.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (img_bgr.shape[1], img_bgr.shape[0]))

        superimposed = cv2.addWeighted(img_bgr, beta, heatmap_color, alpha, 0)

        try:
            superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        except Exception:
            superimposed_rgb = superimposed

        tmpdir = tempfile.mkdtemp(prefix="gradcam_")
        heatmap_path = os.path.join(tmpdir, f"heatmap_{uuid.uuid4().hex}.png")

        plt.figure(figsize=(3.5, 3.5), dpi=100)
        plt.imshow(superimposed_rgb)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        result["heatmap_path"] = heatmap_path

    except Exception as e:
        result["note"] = result.get("note", "") + f" Grad-CAM failed: {e}"

    return result

# --------------------------------------------------
# Audio transcription placeholder
# --------------------------------------------------
def transcribe_audio_file(audio_path):
    """
    Try to transcribe audio. This function is a *placeholder*:
    - If `whisper` (openai/whisper) is installed, attempt to use it.
    - Otherwise return a helpful note and empty transcription.
    Returns (transcription_text, note)
    """
    try:
        import whisper
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        text = result.get("text", "")
        return text, None
    except Exception as e:
        note = ("No local transcription backend available (whisper not installed or failed). "
                "To enable audio transcription, install 'openai-whisper' and its dependencies, "
                "or provide an external transcription service and modify transcribe_audio_file().")
        return "", note