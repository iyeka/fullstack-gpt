from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_core.embeddings import Embeddings


class Embedder:
    def embed_file(
        file: UploadedFile,
        file_dir: str,
        embedding_dir: str,
        embedding_model: Embeddings,
    ):
        file_content = file.read()
        file_path = f"{file_dir}/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        cache_dir = LocalFileStore(f"{embedding_dir}/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = embedding_model
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        return retriever
