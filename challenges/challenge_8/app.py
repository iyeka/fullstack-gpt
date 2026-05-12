from typing import Literal
from bs4 import BeautifulSoup

from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import SitemapLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field
import streamlit as st

st.set_page_config(page_title="Fullstack GPT: #10.0 ~ #10.6", page_icon="🌐")

st.title("SiteGPT Challenge")

with st.expander("challenges"):
    tab1, tab2 = st.tabs(["Challenge Challenge", "Course Challenge"])

    with tab1:
        st.markdown("""
            - [x] Cloudflare 공식문서를 위한 SiteGPT 버전을 만드세요.
            - 챗봇은 아래 프로덕트의 문서에 대한 질문에 답변할 수 있어야 합니다:
                - [x] AI Gateway
                - [x] Cloudflare Vectorize
                - [x] Workers AI
            - [x] 사이트맵을 사용하여 각 제품에 대한 공식문서를 찾아보세요.
            - 여러분이 제출한 내용은 다음 질문으로 테스트됩니다:
                - [x]"What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?"
                - [x]"What can I do with Cloudflare’s AI Gateway?"
                - [x]"How many indexes can a single account have in Vectorize?"
            - [x] 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
            - [x] st.sidebar를 사용하여 Streamlit app과 함께 깃허브 리포지토리에 링크를 넣습니다.
            """)

    with tab2:
        st.markdown("""
            - 챗봇 구현
                - [x] Add memory and history
                - [x] Add streaming
            - [ ] Cache users' similar questions
                1. Save the first user question and response
                2. 다음 질문을 받으면 llm에게 "이런 질문을 받은 적이 있는지" 먼저 물어본다. question list에서 user question과 비슷한 것을 선택하라는 function calling을 사용한다.
                3. 있다면 그에 대한 답변을 반환한다.
                4. 없다면 retriever와 map re-rank를 작동시킨다.
            """)


url = "https://www.cloudflare.com/sitemap.xml"
memory = ConversationBufferMemory()


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


def send_message(message: str, role: Literal["ai", "human"], save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def load_message():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def parse_page(soup: BeautifulSoup):
    header = soup.find("div", attrs={"data-qa": "TemplateHeader"})
    if header:
        header.decompose()
    contact = soup.find("dialog", attrs={"data-qa": "BlockModal"})
    if contact:
        contact.decompose()
    footer = soup.find("footer")
    if footer:
        footer.decompose()

    return soup.get_text(
        separator=" ",
        strip=True,
    )


@st.cache_data(show_spinner="Loading Website...")
def load_website(url: str, _api_key: str):
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=150,
        chunk_overlap=30,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    cache_dir = LocalFileStore(f".cache/site_embeddings")
    embeddings = CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(api_key=_api_key), cache_dir
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            Use ONLY the following context to answer the human question.
            if you can't, just say you don't know. Don't make things up.
            
            Then, score the answer between 0 to 5.
            If the answer answers user question, score should be high.
            Else, it should be low.
            Make sure to always include the score even if it is 0.

            Context: {context}
            
            Examples: 

            Question: How far away is the moon?
            Answer: The moon is 384,400 km away.
            Score: 5
                                                        
            Question: How far away is the sun?
            Answer: I don't know
            Score: 0
            
            Your turn!
        """,
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": chain.invoke(
                    {"context": doc.page_content, "question": question}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]

    choose_answer_llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Use ONLY the following pre-existing answers to answer the user's question.

                Use the answers that have the highest score (more helpful) and favor the most recent ones.

                Cite sources and return the sources of the answers as they are, do not change them.

                Answers: {answers}
                """,
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | choose_answer_llm

    condensed = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}"
        for answer in answers
    )
    result = chain.invoke({"answers": condensed, "question": question})
    return result.content


def get_question_list():
    messages = st.session_state["messages"]

    return [
        {"id": i, "question": message["message"]}
        for i, message in enumerate(messages)
        if message["role"] == "human"
    ]


"""
def get_question_list():
    messages = st.session_state["messages"]

    return [
        {
            "id": pair_id,
            "question": question["message"],
            "answer": answer["message"],
        }
        for pair_id, (question, answer) in enumerate(zip(messages[::2], messages[1::2]))
    ]
"""


def find_similar_question(question):
    class FindSimilarQuestionSchemaArgs(BaseModel):
        is_matched: bool = Field(
            description="True if a similar question exists, false otherwise."
        )
        id: int | None = Field(description="The ID of the matching question.")

    function = {
        "name": "find_similar_question",
        "description": "Check if duplicated question in the list, and return whether it exists and it's id.",
        "parameters": FindSimilarQuestionSchemaArgs.model_json_schema(),
    }

    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.1,
    ).bind(function_call={"name": "find_similar_question"}, function=[function])

    prompt = ChatPromptTemplate.from_template("""
    Find if a similar question is in the question list.

    question: {question}

    quesiton list: {question_list}
    """)

    chain = (
        {
            "question": question,
            "question_list": RunnableLambda(get_question_list),
        }
        | prompt
        | llm
    )

    result = chain.invoke()
    return result.additional_kwargs["function_call"]["arguments"]


with st.sidebar:
    api_key = st.text_input("OPENAI API KEY")
    st.link_button(
        label="🔗GITHUB REPO",
        url="https://github.com/iyeka/fullstack-gpt/tree/master/challenges/challenge_8",
    )

if api_key:

    retriever = load_website(url, api_key)
    send_message("Cloudflare Chatbot is ready. Ask away!", "ai", save=False)
    query = st.chat_input()
    if query:
        send_message(query, "human")
        load_message()
        chain = (
            {"docs": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            chain.invoke(query)

else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.markdown("""
    This is SiteGPT for Cloudflare.

    You can ask questions about:
    - AI Gateway
    - Cloudflare Vectorize
    - Workers AI

    Start by enter your OpenAI API key on the sidebar.
    """)
