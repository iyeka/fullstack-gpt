import streamlit as st
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from typing import Literal

st.set_page_config(page_title="DocumentGPT", page_icon="📜")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    # listen to events
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)  # == with self.message_box: st.markdown


llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(
    show_spinner="Embedding file..."
)  # file이 바뀌지 않으면 다시 실행하지 않고 예전 retriever를 그대로 반환하는 데코레이터
def embed_file(file):
    file_content = file.read()
    file_path = f"./lectures/.cache/files/{file.name}"
    with open(file_path, "wb") as f:  # mode: write the file in binary
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings()
    cache_dir = LocalFileStore(f"./lectures/.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


def save_message(message, role: Literal["user", "assistant", "ai", "human"]):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role: Literal["user", "assistant", "ai", "human"], save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs) -> str:
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
    Answer the question using ONLY the following context. If you don't know the answer, just say you don't know. Don't make it up.
    Context:{context}
""",
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")
st.markdown(
    """
Welcome!

Use this chatbot to ask about you're file and get a professional answer!

Upload your file on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload file: .txt, .pdf, .docx, .epub, .html",
        type=["pdf", "txt", "docx", "epub", "html"],
    )

if file:
    retriever = embed_file(file)
    send_message(
        "I'm ready. Ask away!",
        "ai",
        save=False,
    )
    paint_history()
    message = st.chat_input("Ask anything about your file here.")
    if message:
        send_message(message, "human")
        # chain은 자동으로 retriever.invoke(message)
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
