from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import ollama
from langchain_core.prompts import ChatPromptTemplate
import os
import pickle
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import fitz  
import uuid
from langchain_core.documents import Document
from PIL import Image

llm_model = "llama3.2:1b"
llm_model1 = "deepseek-r1:8b"
embedding_model = "mxbai-embed-large"
FAISS_INDEX_PATH = "faiss.index"
FAISS_STORE_PATH = "faiss_store.pkl"


# --------------------------------------------------
# Try loading existing FAISS store
# --------------------------------------------------
def try_load_faiss_store():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            embeddings = OllamaEmbeddings(model=embedding_model)
            return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None


# --------------------------------------------------
# Vector Store Creation
# --------------------------------------------------
def create_vector_store(file_path):

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        raise ValueError("No text extracted from PDF.")

    for doc in docs:
        doc.metadata['source'] = os.path.basename(file_path)

    image_paths = extract_images_from_pdf(file_path)

    for image_path in image_paths:
        try:
            vision_text = analyze_image_with_vision_llm(image_path)

            image_doc = Document(
                page_content=f"Image Analysis:\n{vision_text}",
                metadata={"source": os.path.basename(file_path), "type": "image"}
            )

            docs.append(image_doc)
        
        except Exception as e:
            print(f"Image analysis failed: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model)
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
    docs = loader.load()

    if not docs:
        raise ValueError("No text extracted from PDF.")

    for doc in docs:
        doc.metadata['source'] = os.path.basename(file_path)

    
    image_paths = extract_images_from_pdf(file_path)

    for image_path in image_paths:
        try:
            vision_text = analyze_image_with_vision_llm(image_path)

            image_doc = Document(
                page_content=f"Image Analysis:\n{vision_text}",
                metadata={"source": os.path.basename(file_path), "type": "image"}
            )

            docs.append(image_doc)
        
        except Exception as e:
            print(f"Image analysis failed: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model)
    new_store = FAISS.from_documents(chunks, embeddings)

    existing_store.merge_from(new_store)
    existing_store.save_local(FAISS_INDEX_PATH)

    return existing_store


# --------------------------------------------------
# RAG Chain (Persona-aware)
# --------------------------------------------------
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
You are a business financial analyst and advisor. The user will upload detailed financial data for a chosen period (week/month/year) — e.g., transaction ledger, sales data, payroll, supplier invoices, bank statements, and/or P&L/Balance sheet. Do the following automatically after ingest:


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

1) Data assumptions
 - Detect currency, period, and aggregation level from file. If ambiguous, state your assumed currency and period.
 - Clean obvious duplicates or zero-value test rows and summarize data quality issues.

2) Analysis (numbers-first)
 - Produce an Executive Summary (1–3 lines).
 - Compact P&L summary: Revenue, COGS, Gross Profit, Opex, EBITDA, Net Profit.
 - Cashflow snapshot and runway.
 - Labour cost breakdown.
 - Top cost and revenue drivers.


3) Insights & diagnostics
 - Flag 3–6 actionable problems with numeric signals.


4) Recommendations
 - Top 5 prioritized actions with financial impact.


5) Forecast & plan
 - Base vs Action scenario forecast.
 - 30–90 day action plan.


6) KPIs & monitoring
 - Recommend 6 KPIs.


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

def image_resize(image_path, max_size=1024):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    img.save(image_path)


def analyze_image_with_vision_llm(image_path):

    prompt = """
You are a radiology interpretation assistant.

STRICT:
- Extract only visible findings.
- Do NOT diagnose.
- No treatment advice.
- Mention uncertainty if applicable.

Output format:
- Image Type:
- Observed Abnormalities:
- Measurements (if visible):
- Possible Clinical Relevance:
- Confidence Level:
"""

    image_resize(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    resp = ollama.chat(
        model="llava:7b",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_bytes]
        }]
    )

    return resp["message"]["content"]


def extract_images_from_pdf(file_path, output_folder="temp_images"):
    os.makedirs(output_folder,exist_ok = True)

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

            if base_image["width"] < 300 or base_image["height"] < 300:
                continue

            image_filename = f"{output_folder}/{uuid.uuid4()}.{image_ext}"

            with open(image_filename, "wb") as f:
                f.write(image_bytes)

            image_path.append(image_filename)
    return image_path