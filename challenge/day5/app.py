import streamlit as st
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from typing import Literal


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./challenge/day5/.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings()
    cache_dir = LocalFileStore(f"./challenge/day5/.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_human_message(message):
    st.session_state["memory"].chat_memory.add_user_message(message)


def save_ai_message(message):
    st.session_state["memory"].chat_memory.add_ai_message(message)


def send_message(message, role: Literal["user", "assistant", "ai", "human"], save=True):
    with st.chat_message(role):
        st.markdown(message)
    if role == "human" and save:
        save_human_message(message)
    elif role == "ai" and save:
        save_ai_message(message)


def chat_history():
    for message in st.session_state["memory"].chat_memory.messages:
        send_message(message.content, message.type, save=False)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the human question only with following context. If you don't know the answer, don't make it up:\n\n{context}",
        ),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.1,
)

st.title("RAG Pipeline with Streamlit")
st.markdown(
    """
Upload your document to the sidebar.\n
Chatbot will read the document and answer your questions on your behalf.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "upload file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm READY. Ask away!", "ai", save=False)
    chat_history()
    question = st.chat_input("Ask anything about your file.")
    if question:
        send_message(question, "human")
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
        response = chain.invoke(question)
        send_message(response.content, "ai")
else:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
