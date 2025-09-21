import streamlit as st
from typing import Literal
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="🔒",
)


# CallbackHandler have functions for listening events on the llm
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):  # 몇 개의 args와 kwargs를 받든 상관없어
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# @st.cache_data sees the file parameter of embed_file to detect whether it has changed. if the file is same, Streamlit will not gonna re-run the function.
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, model):
    file_content = file.read()
    file_path = f"./lectures/#8_files/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./lectures/#8_files/private_embeddings/{file.name}")

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    file = loader.load_and_split(text_splitter=splitter)

    embeddings = OllamaEmbeddings(model=model)  # 사전 다운로드
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(file, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role: Literal["user", "assistant", "ai", "human"]):
    st.session_state["messages"].append({"message": message, "role": role})
    (
        memory.chat_memory.add_user_message(message)
        if role == "user"
        else memory.chat_memory.add_ai_message(message)
    )


def send_message(message, role: Literal["user", "assistant", "ai", "human"], save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def load_memory():
    memory.load_memory_variables({})["history"]


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


st.title("PrivateGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

1. Upload your files on the sidebar.
2. Choose the model you want to chat.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
    )
    model = st.selectbox(
        "Choose the model you want to use.",
        ("llama2-uncensored:7b", "falcon3:1b", "mistral:7b"),
    )

memory = ConversationBufferMemory(return_messages=True)

llm = ChatOllama(
    model=model,
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

# finetune the prompt to instruct.
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.
    
    Context: {context}
    Question: {question}
    """,
)

if file:
    retriever = embed_file(file, model)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever
                | RunnableLambda(
                    lambda docs: "\n\n".join(doc.page_content for doc in docs)
                ),
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
