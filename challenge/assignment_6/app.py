import os
import streamlit as st
from dotenv import load_dotenv
from typing import Literal
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_store = MessageStore()

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        self.message_store.save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


class MessageStore:
    def send_message(self, message, role: Literal["human", "ai"], save=True):
        with st.chat_message(role):
            st.write(message)
        if save:
            self.save_message(message, role)

    def save_message(self, message, role: Literal["human", "ai"]):
        st.session_state["messages"].append({"message": message, "role": role})

    def load_message(self):
        for message in st.session_state["messages"]:
            self.send_message(message["message"], message["role"], save=False)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    FILE_PATH = f"./challenge/assignment_6/files/{file.name}"
    CACHE_DIR = LocalFileStore(f"./challenge/assignment_6/cache/{file.name}")

    file_content = file.read()
    with open(FILE_PATH, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(FILE_PATH)
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    file = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, CACHE_DIR)
    vectorstore = FAISS.from_documents(file, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def instructions():
    with st.sidebar:
        api_key = st.text_input(
            label="OPENAI_API_KEY",
            type="password",
            placeholder="Type your OPENAI API KEY and press enter.",
        )
        file = st.file_uploader(
            label="File Upload",
            type=["pdf", "txt", "docx"],
        )

        with st.expander("Tasks"):
            st.markdown(
                """
                - [x] 이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
                - 파일 업로드 및 채팅 기록을 구현합니다.
                    - [x] [파일 업로드](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader)를 구현하기 위해 st.file_uploader를 사용합니다. type 매개변수를 통해 허용할 파일 확장자를 지정할 수 있습니다.
                    - [x] 채팅 기록을 저장하기 위해 [Session State](https://docs.streamlit.io/library/api-reference/session-state)를 사용합니다.
                - [x] 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
                    - [x] 사용자의 OpenAI API 키를 사용하기 위해 [st.text_input](https://docs.streamlit.io/library/api-reference/widgets/st.text_input)을 이용하여 API 키를 입력받습니다. 그런 다음, ChatOpenAI와 OpenAIEmbeddings 클래스를 사용할 때 해당 API 키를 openai_api_key 매개변수로 넘깁니다.
                    - [x] with st.sidebar를 사용하면 사이드바와 관련된 코드를 더욱 깔끔하게 정리할 수 있습니다.(https://docs.streamlit.io/library/api-reference/layout/st.sidebar)
                - [ ] st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
                    - [ ] 코드를 공개 Github 리포지토리에 푸시합니다. 단. OpenAI API 키를 Github 리포지토리에 푸시하지 않도록 주의하세요.
                    - [x] [여기에서 계정을 개설하세요](https://share.streamlit.io/)
                    - [ ] [다음 단계를 따르세요](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app#deploy-your-app-1)
                    - [ ] 앱의 구조가 아래와 같은지 확인하고 배포 양식의 Main file path 에 app.py를 작성하세요.
                    ```python
                    your-repo/
                    ├── .../
                    ├── app.py
                    └── requirements.txt
                    ```
                    - [ ] 과제 제출 링크는 반드시 streamlit.app URL 이 되도록 하세요.
                """
            )

    st.markdown(
        """
        Welcome!
                
        Use this chatbot to ask questions to an AI about your files!

        1. Upload your OPENAI api key on the sidebar.
        2. Upload your file on the sidebar.
        3. When embedding is finish, Ask question to chatbot.
        """
    )
    return api_key, file


def main(api_key, file):
    llm = ChatOpenAI(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL_NAME"),
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the question only with given context. If you don't know the answer just say you don't know, don't make it up.

                context: {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    retriever = embed_file(file)
    message_store = MessageStore()
    message_store.send_message("I'm ready! Ask away!", "ai", save=False)

    message_store.load_message()
    query = st.chat_input("Ask about your file!")
    if query:
        message_store.send_message(message=query, role="human")
        chain = (
            {
                "context": retriever
                | RunnableLambda(
                    lambda docs: "\n\n".join(doc.page_content for doc in docs)
                ),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(query)


load_dotenv()
st.set_page_config(
    page_title="풀스택 GPT: #7.0~7.10",
    page_icon="📜",
)
st.title("풀스택 GPT: #7.0~7.10")

api_key, file = instructions()
if api_key and file:
    main(api_key, file)
elif not file:
    st.session_state["messages"] = []
