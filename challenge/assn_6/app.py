import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from typing import Literal

st.title("Fullstack GPT: #7.0 ~ #7.10")
with st.expander("challenge"):
    st.markdown(
        """
        - [x] Migrate the RAG pipeline you implemented in the previous assignments to Streamlit.
        - [x] Implement file upload and chat history.
        - [x] Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
        - [x] Using st.sidebar put a link to the Github repo with the code of your Streamlit app.
        - [x] 여기에서 계정을 개설하세요: https://share.streamlit.io/
        - [ ] 다음 단계를 따르세요: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app#deploy-your-app-1
        - [x]앱의 구조가 아래와 같은지 확인하고 배포 양식의 Main file path 에 app.py를 작성하세요.
            your-repo/
            ├── .../
            ├── app.py
            └── requirements.txt
        - [ ] 과제 제출 링크는 반드시 streamlit.app URL 이 되도록 하세요.
    """
    )


def save_file(file, file_path: str):
    with open(file_path, "wb") as f:
        content = file.read()
        f.write(content)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file_path: str, _cache_dir: str, api_key: str):

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    cached_embbedings = CacheBackedEmbeddings.from_bytes_store(embeddings, _cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embbedings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message: str, role: Literal["human", "ai"]):
    memory = st.session_state["memory"]
    if role == "human":
        memory.chat_memory.add_user_message(message)
    elif role == "ai":
        memory.chat_memory.add_ai_message(message)


def send_message(message: str, role: Literal["human", "ai"], save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message()


def paint_memory():
    for message in st.session_state["memory"].load_memory_variables({})["history"]:
        send_message(message.content, message.type, save=False)


def invoke_chain(message):
    response = chain.invoke(message)
    send_message(response.content, "ai")


with st.sidebar:
    file = st.file_uploader(
        label="UPLOAD FILE",
        type=[".txt", ".pdf", ".docx"],
    )
    api_key = st.text_input(
        label="OPENAI API KEY",
        type="password",
    )
    git_link = st.text_input("Put your Git commit link")
    st.markdown(f"[{git_link}]({git_link})")

if file and api_key:
    FILE_PATH = f"./.cache/files/{file.name}"
    CACHE_DIR = LocalFileStore(f"./.cache/embeddings/{file.name}")

    save_file(file, FILE_PATH)
    retriever = embed_file(FILE_PATH, CACHE_DIR, api_key)
    send_message("I'm ready! Ask away", "ai", save=False)
    paint_memory()

    message = st.chat_input("Ask anything about your file")
    if message:
        send_message(message, "human")

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.1,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Answer human question according to following context. If you don't know don't make it up and just say so.\n\n
                    {context}
                    """,
                ),
                ("human", "{question}"),
            ]
        )

        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        invoke_chain(message)
elif not file:
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
