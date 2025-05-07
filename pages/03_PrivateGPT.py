from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="🔒",
)

# custom CallBackHandler. 응답을 실시간으로 화면에 표시하기 위해 제작.
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    # unlimited arguments and keywords
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty() # 나중에 뭔가 담을 수 있는 빈 공간

    # llm이 새롭게 생성하는 모든 token에 대해 listen.
    def on_llm_new_token(self, token, *args, **kwargs):
        # 빈 메시지에 토큰을 실시간으로 추가
        self.message += token # == self.message = f"{self.message}{token}"
        self.message_box.markdown(self.message) # ex) self.message = "a", message_box("a") -> self.message = "b", message_box("ab")

    # llm이 작업을 끝내는 시점 감지
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

llm = ChatOllama(
    model="llama2-uncensored:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

# decorator for caching result of a function. 파일이 바뀌지 않는 한 함수가 다시 실행되지 않는다.
@st.cache_data(show_spinner=True) # show_spinner="showing message"
def embed_file(file):
    file_content = file.read()
    # 이 경로에 유저가 올린 file 저장
    file_path = f"./.cache/files/{file.name}"
    with open(file=file_path, mode="wb") as f: # write the file in binary mode
        f.write(file_content)

    # embeddings 저장 경로
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(
        model="llama2-uncensored:latest"
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

# Send AI chat message to user
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False,)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_template(
    """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
        
        Context: {context}
        (question: {question}
        """
)

st.title("DocumentGPT")

# Ask user to upload file
st.markdown("""
Willkommen!
Use this chatbot to ask questions to an AI about your files.
""")

with st.sidebar:
    file = st.file_uploader(label="Upload a .txt .pdf .docx type file", type=["txt", "pdf", "docx"])

if file:
    retriever = embed_file(file)
    send_message("File is uploaded.", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file to chatbot.")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []