from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import os
import pickle
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


llm_model = "llama3.2:1b"
embedding_model = "mxbai-embed-large"
PERSIST_DIR="db"
FAISS_INDEX_PATH = "faiss.index"
FAISS_STORE_PATH = "faiss_store.pkl"

def create_vector_store(file_path):


    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_STORE_PATH):
        print("🔄 Loading existing FAISS vector store...")
        with open(FAISS_STORE_PATH,"rb") as f:
            stored = pickle.load(f)
        faiss_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            stored["embedding"],
            allow_dangerous_deserialization=True
        )
        return faiss_store
    
    print("⚡ Creating new FAISS vector store... Processing PDF...")


    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model)

    faiss_store = FAISS.from_documents(chunks, embeddings)

    faiss_store.save_local(FAISS_INDEX_PATH)
    with open(FAISS_STORE_PATH,"wb") as f:
        pickle.dump({"embedding":embeddings},f)

    print("✔ FAISS index saved.")

    return faiss_store

def create_rag_chain(vector_store,persona):
    llm = Ollama(model="llama3.2:1b")

    prompt = ChatPromptTemplate.from_messages([
        ("system",persona),
        ("human","""
    You are an expert assistant. Answer the user's question using ONLY the information provided in the context below. 
    If the answer cannot be found in the context, respond exactly with: "The document does not contain this information."

    Context:
    {context}

    Question:
    {input}

    Answer format instructions:
    - Provide a clear and concise answer.
    - Use complete sentences.
    - Do NOT include any information not in the context.
    - If the answer is not present, respond exactly as instructed above.

    Answer:
    """)
    ])


    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def set_persona(choice):
    if choice == "RESEARCH":
        return """You are a research expert.
Your answers must be:
- highly technical
- citation-style
- structured  
- formal"""
    
    elif choice == "BUSINESS":
        return """
You are a business consultant.
Your answers must be:
- simple
- actionable
- problem → solution oriented
- clear and concise
"""

    elif choice == "EDUCATION":
        return """
You are a friendly teacher.
Your answers must be:
- easy to understand
- step-by-step
- include examples
- friendly tone
"""

    return ""

