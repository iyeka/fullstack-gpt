import git
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough

# for type hint:
from typing import Literal

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

with st.expander("Tasks"):
    st.markdown(
        """
        # Fullstack GPT: #7.0 ~ #7.10

        - [x] 이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
        - 파일 업로드 및 채팅 기록을 구현합니다.
            - [x] 파일 업로드를 구현하기 위해 [st.file_uploader](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader)를 사용합니다. type 매개변수를 통해 허용할 파일 확장자를 지정할 수 있습니다.
            - [x] 채팅 기록을 저장하기 위해 [Session State](https://docs.streamlit.io/library/api-reference/session-state)를 사용합니다.
        - [x] 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
            - [x] 사용자의 OpenAI API 키를 사용하기 위해 [st.text_input](https://docs.streamlit.io/library/api-reference/widgets/st.text_input)을 이용하여 API 키를 입력받습니다. 그런 다음, ChatOpenAI와 OpenAIEmbeddings 클래스를 사용할 때 해당 API 키를 openai_api_key 매개변수로 넘깁니다.
            - with [st.sidebar](https://docs.streamlit.io/library/api-reference/layout/st.sidebar)를 사용하면 사이드바와 관련된 코드를 더욱 깔끔하게 정리할 수 있습니다.
        - [x] st.sidebar를 사용하여 [스트림릿 앱](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app#deploy-your-app-1)의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
            - [x] 과제 제출 링크는 반드시 streamlit.app URL 이 되도록 하세요.
        - [x] 앱의 구조가 아래와 같은지 확인하고 배포 양식의 Main file path 에 app.py를 작성하세요.
            ```
            your-repo/
            ├── .../
            ├── app.py
            └── requirements.txt
    """
    )

st.title("DocumentGPT")
st.markdown(
    """
    Welcome to Document GPT!

    Upload your file on sidebar and ask anything about the file.
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
    st.markdown([git_link](git_link))

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
