from langchain_chroma import Chroma
from config import COLLECTION_NAME, PERSIST_DIR

def get_vectorstore(embeddings):
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

def add_documents(vectorstore, docs):
    vectorstore.add_documents(docs)