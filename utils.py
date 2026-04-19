from typing import List

from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_core.embeddings import Embeddings


class Embedder:
    def split_file(self, file: UploadedFile, file_dir: str):
        self.file = file
        file_content = self.file.read()
        file_path = f"{file_dir}/{self.file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        self.docs = loader.load_and_split(text_splitter=splitter)
        return self.docs

    def embed_file(
        self,
        embedding_dir: str,
        embedding_model: Embeddings,
    ):
        embeddings = embedding_model
        cache_dir = LocalFileStore(f"{embedding_dir}/{self.file.name}")
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vectorstore = FAISS.from_documents(self.docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        return retriever


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)
