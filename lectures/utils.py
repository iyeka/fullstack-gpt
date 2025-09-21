import os
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS

# for type hints:
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pathlib import Path
from langchain.schema import Document


@st.cache_data(show_spinner="Loading file...")
def split_file(file: UploadedFile, file_path: str | Path) -> list[Document]:
    file_content = file.read()
    file_path = file_path
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Embedding file...")
def embed_file(docs: list[Document], cache_dir: str | Path):
    cache_dir = LocalFileStore(cache_dir)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
