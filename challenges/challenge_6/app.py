from typing import Literal
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import streamlit as st


st.set_page_config(page_title="DocumentGPT Challenge", page_icon="📃")
st.title("Fullstack GPT: #7.0 ~ #7.10")

with st.expander("challenges"):
    st.markdown(
        """
        - [x] Migrate the RAG pipeline you implemented in the previous assignments to Streamlit.
        - [x] Implement file upload and chat history.
        - [x] Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
        - [x] Using st.sidebar put a link to the Github repo with the code of your Streamlit app.
    """
    )


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")


def save_message(message: str, role: Literal["human", "ai"]):
    st.session_state["messages"].append({"message": message, "role": role})
    if role == "human":
        memory.chat_memory.add_user_message(message)
    elif role == "ai":
        memory.chat_memory.add_ai_message(message)


def send_message(message: str, role: Literal["human", "ai"], save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file: UploadedFile, api_key: str):
    file_path = f".cache/files/{file.name}"
    file_content = file.read()
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    cache_dir = LocalFileStore(f".cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


memory = ConversationBufferMemory(return_messages=True)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using only the following context and chat history. If you don't know the answer, just say you don't know. Don't make it up:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

with st.sidebar:
    file = st.file_uploader(
        label="FILE",
        type=["txt", "pdf", "docx", "png"],
    )

    api_key = st.text_input(
        label="OPENAI API KEY",
        placeholder="Type your api key here",
    )

    st.link_button(
        label="🔗GITHUB REPO",
        url="https://github.com/iyeka/fullstack-gpt/tree/master/challenges/challenge_6",
    )

if file and api_key:
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )

    retriever = embed_file(
        file=file,
        api_key=api_key,
    )
    send_message("The file is ready. Ask away!", "ai", save=False)

    paint_history()

    message = st.chat_input("Ask anything about your file.")
    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever,
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
    send_message(
        message="""
            Welcome to DocumentGPT!

            Use this chatbot to ask questions about your files.

            Get started by upload your file and OpenAI api key on the sidebar.
        """,
        role="ai",
        save=False,
    )
