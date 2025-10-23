# CODE CHALLENGE: ADD MEMORY TO CHAIN

import streamlit as st
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

# for type hints:
from typing import Literal

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)
st.title("DocumentGPT")
st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your file on the sidebar.
"""
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def save_file(file, file_path):
    with open(file_path, "wb") as f:
        file_content = file.read()
        f.write(file_content)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file_path, _cache_dir):
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings()
    cached_embbedings = CacheBackedEmbeddings.from_bytes_store(embeddings, _cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embbedings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    memory = st.session_state["memory"]
    if role == "human":
        memory.chat_memory.add_user_message(message)
    elif role == "ai":
        memory.chat_memory.add_ai_message(message)


def send_message(message: str, role: Literal["ai", "human"], save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# def load_memory(_):
#     return st.session_state["memory"].load_memory_variables({})["history"]


def paint_memory():
    for message in st.session_state["memory"].load_memory_variables({})["history"]:
        send_message(message.content, message.type, save=False)


llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        # MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
if file:
    FILE_PATH = f"./.cache/files/{file.name}"
    CACHE_DIR = LocalFileStore(f"./.cache/embeddings/{file.name}")

    save_file(file, FILE_PATH)
    retriever = embed_file(FILE_PATH, CACHE_DIR)
    send_message("I'm ready! Ask away", "ai", save=False)
    paint_memory()

    message = st.chat_input("Ask anything about your file")
    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever
                | RunnableLambda(
                    lambda docs: "\n\n".join(doc.page_content for doc in docs)
                ),
                # "history": RunnableLambda(load_memory),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
