from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


llm_model = "llama3.2:1b"
embedding_model = "mxbai-embed-large"
PERSIST_DIR="db"

def create_vector_store(file_path):


    if os.path.exists(PERSIST_DIR):
        print("🔄 Loading existing ChromaDB...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OllamaEmbeddings(model=embedding_model)
        )
    
    print("⚡ Creating new ChromaDB... Processing PDF...")


    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   # smaller chunks -> faster embedding
        chunk_overlap=200 # reduce overlap
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model)

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    vector_store.persist()
    print("✔ ChromaDB created and saved.")

    return vector_store

def create_rag_chain(vector_store):
    llm = Ollama(model=llm_model)


    prompt = ChatPromptTemplate.from_template("""
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


    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
