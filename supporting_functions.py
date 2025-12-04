from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


llm_model = "llama3"
embedding_model = "nomic-embed-text"

def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model)

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="db"
    )

    return vector_store

def create_rag_chain(vector_store):
    llm = Ollama(model=llm_model)

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using ONLY the provided context.
    If the answer is not present, say "The document does not contain this information."

    <context>
    {context}
    </context>

    Question: {input}
    Answer based only on the context.
    """)

    retriever = vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
