from typing import Literal

from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.title("PrivateGPT")

st.set_page_config(page_title="PrivateGPT", page_icon="🤫")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


memory = ConversationBufferWindowMemory(
    return_messages=True,
    k=10,
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, model: str):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(model=model)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role: Literal["human", "ai"]):
    st.session_state["messages"].append({"message": message, "role": role})
    if role == "human":
        memory.chat_memory.add_user_message(message)
    elif role == "ai":
        memory.chat_memory.add_ai_message(message)


def send_message(message, role: Literal["human", "ai"], save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def load_memory():
    return memory.load_memory_variables({})["history"]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.
    
    Context: {context}
    Question: {question}
    """,
)

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        label="Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    selected_model = st.selectbox(
        label="Select a model",
        options=["llama3.1", "mistral", "falcon"],
    )


if file and selected_model:
    model = f"{selected_model}:latest"
    llm = ChatOllama(
        model=model,
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )
    retriever = embed_file(file, model)
    send_message("I'm ready. Ask Away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file.")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "history": RunnableLambda(load_memory),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []
