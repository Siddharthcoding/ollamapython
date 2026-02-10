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


llm_model = "llama3.2:1b"
llm_model1 = "deepseek-r1:8b"
embedding_model = "mxbai-embed-large"
PERSIST_DIR = "db"
FAISS_INDEX_PATH = "faiss.index"
FAISS_STORE_PATH = "faiss_store.pkl"



# --------------------------------------------------
# Vector Store Creation (First PDF)
# --------------------------------------------------
def create_vector_store(file_path):
    """Create initial FAISS vector store from first PDF"""
    
    print(f"⚡ Creating new FAISS vector store from {file_path}...")


    loader = PyMuPDFLoader(file_path)
    docs = loader.load()


    # ✅ Add source metadata to track which PDF chunks came from
    for doc in docs:
        doc.metadata['source'] = os.path.basename(file_path)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)


    embeddings = OllamaEmbeddings(model=embedding_model)


    faiss_store = FAISS.from_documents(chunks, embeddings)


    # ✅ Save to disk
    faiss_store.save_local(FAISS_INDEX_PATH)
    with open(FAISS_STORE_PATH, "wb") as f:
        pickle.dump({"embedding": embeddings}, f)


    print(f"✔ FAISS index created with {len(chunks)} chunks")
    return faiss_store



# --------------------------------------------------
# Add Documents to Existing Vector Store
# --------------------------------------------------
def add_to_vector_store(existing_store, file_path):
    """Add new PDF to existing FAISS vector store"""
    
    print(f"📄 Adding {file_path} to existing vector store...")


    loader = PyMuPDFLoader(file_path)
    docs = loader.load()


    # ✅ Add source metadata
    for doc in docs:
        doc.metadata['source'] = os.path.basename(file_path)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)


    # ✅ Get embeddings (must use same model as original)
    embeddings = OllamaEmbeddings(model=embedding_model)


    # ✅ Create temporary store for new PDF
    new_store = FAISS.from_documents(chunks, embeddings)


    # ✅ Merge into existing store
    existing_store.merge_from(new_store)


    # ✅ Save updated store
    existing_store.save_local(FAISS_INDEX_PATH)
    with open(FAISS_STORE_PATH, "wb") as f:
        pickle.dump({"embedding": embeddings}, f)


    print(f"✔ Added {len(chunks)} chunks. Vector store updated.")
    return existing_store



# --------------------------------------------------
# RAG Chain
# --------------------------------------------------
def create_rag_chain(vector_store, persona):


    if persona == "BUSINESS":
        llm = Ollama(model="qwen2.5:7b", temperature=0.4)
    elif persona == "RESEARCH":
        llm = Ollama(model="deepseek-r1:1.5b", temperature=0.2)
    else:
        llm = Ollama(model="llama3.2:1b", temperature=0.5)


    prompt = ChatPromptTemplate.from_messages([
        ("system", set_persona(persona)),
        ("human", """
You are an expert assistant. Answer the user's question using ONLY the information provided in the context below. 
If the answer cannot be found in the context, respond exactly with:
"The document does not contain this information."


Context:
{context}


Question:
{input}


Answer format instructions:
- Provide a clear and concise answer.
- Use complete sentences.
- Do NOT include any information not in the context.
- If the answer is not present, respond exactly as instructed above.
- When information comes from multiple documents, synthesize it coherently.


Answer:
""")
    ])


    # ✅ Retrieve from multiple documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)


    return retrieval_chain



# --------------------------------------------------
# Persona Prompts
# --------------------------------------------------
def set_persona(choice):


    if choice == "RESEARCH":
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







def analyze_image_with_vision_llm(image_path):
    prompt = """
    You are a medical image analysis assistant.

    Analyze the given image carefully.

    Instructions:
    - Extract only visible findings from the image.
    - Do NOT diagnose.
    - Do NOT assume patient history.
    - Mention uncertainty if present.

    Output format:
    - Image type:
    - Key visible findings (bullet points):
    - Possible clinical relevance (non-diagnostic):
    - Confidence level (low / medium / high)
    """


    resp = ollama.chat(
        model="llava:7b",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path]
            }
        ]
    )

    return resp["message"]["content"]