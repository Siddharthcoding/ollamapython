from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms import Ollama
import ollama
from functools import partial
from langchain_core.prompts import ChatPromptTemplate
import os
import pickle
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import fitz
import uuid
from langchain_core.documents import Document
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


llm_model       = "llama3.2:1b"
llm_model1      = "deepseek-r1:8b"
embedding_model = "mxbai-embed-large"
FAISS_INDEX_PATH = "faiss.index"
FAISS_STORE_PATH = "faiss_store.pkl"

PICKLE_MODEL_PATH = "model.pkl"
CLASS_NAMES_PATH  = "models/brain_tumor_class_names.json"


# --------------------------------------------------
# Try loading existing FAISS store (legacy / General page)
# --------------------------------------------------
def try_load_faiss_store():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            embeddings = OllamaEmbeddings(model=embedding_model)
            return FAISS.load_local(
                FAISS_INDEX_PATH, embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            return None
    return None


embeddings = OllamaEmbeddings(model=embedding_model)


def process_image(image_path, file_path):
    try:
        vision_text = analyze_image_with_vision_llm(image_path)
        return Document(
            page_content=f"Image Analysis:\n{vision_text}",
            metadata={"source": os.path.basename(file_path), "type": "image"}
        )
    except Exception as e:
        print(f"Image extraction failed: {e}")
        return None


# --------------------------------------------------
# Vector Store Creation
# --------------------------------------------------
def create_vector_store(file_path):
    loader = PyMuPDFLoader(file_path)
    docs   = loader.load()
    if not docs:
        raise ValueError("No text extracted from PDF.")
    for doc in docs:
        doc.metadata['source'] = os.path.basename(file_path)

    image_paths  = extract_images_from_pdf(file_path)
    process_func = partial(process_image, file_path=file_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        image_docs = list(executor.map(process_func, image_paths))
    docs.extend([doc for doc in image_docs if doc])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    faiss_store = FAISS.from_documents(chunks, embeddings)
    faiss_store.save_local(FAISS_INDEX_PATH)
    with open(FAISS_STORE_PATH, "wb") as f:
        pickle.dump({"embedding": embedding_model}, f)
    return faiss_store


# --------------------------------------------------
# Add to Existing Vector Store
# --------------------------------------------------
def add_to_vector_store(existing_store, file_path):
    loader = PyMuPDFLoader(file_path)
    docs   = loader.load()
    if not docs:
        raise ValueError("No text extracted from PDF.")
    for doc in docs:
        doc.metadata['source'] = os.path.basename(file_path)

    image_paths  = extract_images_from_pdf(file_path)
    process_func = partial(process_image, file_path=file_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        image_docs = list(executor.map(process_func, image_paths))
    docs.extend([doc for doc in image_docs if doc])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    existing_store.add_documents(chunks)
    existing_store.save_local(FAISS_INDEX_PATH)
    return existing_store


# --------------------------------------------------
# RAG Chain (Persona-aware)
# --------------------------------------------------
def create_rag_chain(vector_store, persona):
    if persona == "BUSINESS":
        llm = Ollama(model="qwen2.5:7b",      temperature=0.4)
    elif persona == "RESEARCH":
        llm = Ollama(model="deepseek-r1:1.5b", temperature=0.2)
    elif persona == "MEDICAL":
        llm = Ollama(model="deepseek-r1:1.5b", temperature=0.1)
    else:
        llm = Ollama(model="llama3.2:1b",      temperature=0.5)

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

    retriever      = vector_store.as_retriever(search_kwargs={"k": 6})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


# --------------------------------------------------
# Persona Prompts
# --------------------------------------------------
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

If the question is about symptoms, analyze patterns.
If about lab values, interpret high/low relative to context.
If about scans, reason from extracted findings.
"""

    elif choice == "RESEARCH":
        return """You are a research expert.
Your answers must be:
- highly technical
- citation-style
- structured
- formal
- synthesize information from multiple sources when available
"""

    elif choice == "BUSINESS":
        return """
You are a business financial analyst and advisor. The user will upload detailed
financial data for a chosen period.

STRICT NUMERIC RULES:
- Use ONLY numbers explicitly in the context.
- Revenue = units × selling price.
- Profit = revenue − cost.
- Do NOT invent numbers for any product, cost, or employee.
- If any required number is missing, respond exactly:
  "The document does not contain this information."
- Show all calculation steps explicitly in the following format:
    Original value:
    Calculation:
    Result:
- Never mix units with currency.
- When analyzing multiple documents, consolidate data appropriately.

1) Data assumptions — detect currency, period, aggregation level.
2) Analysis — Executive Summary, P&L, Cashflow, Labour, Drivers.
3) Insights & diagnostics — flag 3–6 actionable problems.
4) Recommendations — top 5 prioritised actions.
5) Forecast & plan — base vs action scenario, 30–90 day plan.
6) KPIs & monitoring — recommend 6 KPIs.

Tone & style: short, actionable, business-first.
"""

    elif choice == "EDUCATION":
        return """
You are a friendly teacher.
Your answers must be:
- easy to understand
- step-by-step
- include examples
- friendly tone
- combine information from multiple sources when relevant
"""
    return ""


# --------------------------------------------------
# Image utilities
# --------------------------------------------------
def image_resize(image_path, max_size=448):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    img.save(image_path)


def analyze_image_with_vision_llm(image_path: str,
                                   findings_text: str = "") -> str:
    """
    Run the vision LLM (llava-phi3) on an image.

    Parameters
    ----------
    image_path    : path to the image file (required)
    findings_text : optional extra context to prepend to the prompt
                    (e.g. CNN prediction text).  Defaults to "".
    """
    base_prompt = """
You are an advanced document and medical image analysis assistant.
Examine the provided image carefully and extract all relevant information.

1. Identify the image type: MRI, CT, X-ray, Ultrasound, chart, table, diagram, or text.
2. Extract visible text exactly as written.
3. Describe charts, tables, or diagrams, summarizing key points.
4. For medical images:
   - Note visible anatomical structures and abnormalities.
   - Provide a ranked differential diagnosis based only on visible findings
     (High / Moderate / Low likelihood).
   - Mention uncertainty or limitations.
5. Ignore irrelevant visual details.

STRICT RULES:
- Do not invent measurements, findings, or patient information.
- Do not give definitive diagnoses or treatment advice.
- Clearly state uncertainty where applicable.
"""
    full_prompt = base_prompt
    if findings_text:
        full_prompt = f"{base_prompt}\n\nAdditional context:\n{findings_text}"

    image_resize(image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    resp = ollama.chat(
        model="llava-phi3:3.8b",
        messages=[{
            "role":    "user",
            "content": full_prompt,
            "images":  [image_bytes]
        }],
        options={"num_predict": 400, "temperature": 0.2}
    )
    return resp["message"]["content"]


# --------------------------------------------------
# PDF image extraction
# --------------------------------------------------
def extract_images_from_pdf(file_path, output_folder="temp_images"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(file_path)
    image_paths = []
    for page_ind in range(len(doc)):
        page       = doc[page_ind]
        image_list = page.get_images(full=True)
        for img in image_list:
            xref       = img[0]
            base_image = doc.extract_image(xref)
            if base_image["width"] < 300 or base_image["height"] < 300:
                continue
            image_filename = f"{output_folder}/{uuid.uuid4()}.{base_image['ext']}"
            with open(image_filename, "wb") as f:
                f.write(base_image["image"])
            if os.path.getsize(image_filename) > 50_000:
                image_paths.append(image_filename)
    return image_paths


# --------------------------------------------------
# Brain tumour CNN model
# --------------------------------------------------
def load_class_name():
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError("Class names file not found.")
    import json
    with open(CLASS_NAMES_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [k for k, v in sorted(data.items(), key=lambda item: item[1])]
    return data


def processes_image(image_path, target_size=(150, 150)):
    img       = Image.open(image_path).convert("RGB")
    img       = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)


# Load once at import time
model = load_model("model.h5")


def run_pickle_model_prediction(image_path):
    class_name = load_class_name()
    x    = processes_image(image_path)
    pred = model.predict(x)

    if pred.ndim == 1:
        probs = np.array([1 - pred[0], pred[0]])
    else:
        probs = pred[0]

    idx        = int(np.argmax(probs))
    confidence = float(probs[idx])
    return {
        "prediction": class_name[idx],
        "confidence": float(np.clip(confidence, 0.0, 1.0))
    }


print(f"Brain tumour model loaded: {type(model)}")