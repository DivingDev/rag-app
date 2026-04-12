import nest_asyncio
from langchain_community.document_loaders import WebBaseLoader

nest_asyncio.apply()


def load_documents_from_urls(urls: list[str]):
    loader = WebBaseLoader(urls)
    loader.requests_per_second = 2
    return loader.load()


