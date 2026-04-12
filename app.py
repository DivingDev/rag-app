import streamlit as st

from rag.loader import load_documents_from_urls
from rag.splitter import split_documents
from rag.embeddings import get_embeddings
from rag.vectorstore import get_vectorstore, add_documents
from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.chain import build_chain


st.set_page_config(page_title="RAG QA App", page_icon="🧠", layout="centered")

st.title("🧠 RAG Question Answering")
st.markdown("Ask a question based on the loaded article")


urls_input = st.text_area(
    "🌐 Enter URLs (one per line)",
    height=120,
    placeholder="https://example.com/article1\nhttps://example.com/article2",
)

# Initialize pipeline once
@st.cache_resource
def initialize_pipeline(urls: tuple[str, ...]):
    docs = load_documents_from_urls(list(urls))
    split_docs = split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    add_documents(vectorstore, split_docs)

    retriever = get_retriever(vectorstore)
    llm = get_llm()
    chain = build_chain(retriever, llm)
    return chain, retriever


# UI Input
question = st.text_input("💬 Enter your question")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question")
    else:
        urls = tuple(u.strip() for u in urls_input.splitlines() if u.strip())
        chain, retriever = initialize_pipeline(urls)
        with st.spinner("Thinking..."):
            response = chain.invoke(question)

        st.success("✅ Answer")
        st.write(response.message)

        if response.source:
            st.markdown(f"🔗 Source: {response.source}")