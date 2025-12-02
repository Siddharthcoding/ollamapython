from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


llm_model = llama3.2:1b
embedding_model = nomic-embed-text


def create_vector_store(file_pathh):
    fileloader = PyPDFLoader(file_path=file_pathh)
    docs = fileloader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model)

    vector_store = Chroma.from_documents(documents=splits,embedding=embeddings)

    return vector_store

def create_rag_chain(vector_store):
    llm = Ollama(model=llm_model)

    prompt=ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Your goal is to provide a detailed and comprehensive answer.
    Extract all relevant information from the context to formulate your response.
    Think step by step and structure your answer logically.
    If the context does not contain the answer to the question, state that the information is not available in the provided document. Do not attempt to make up information.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    retriever = vector_store.as_retriever()

    document_chain = create_stuff_documents_chain(llm , prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain