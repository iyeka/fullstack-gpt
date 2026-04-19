from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnableLambda
import streamlit as st
import json

from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.schema import Document

st.set_page_config(
    page_title="Fullstack GPT: #9.0 ~ #9.9",
    page_icon="❓",
)
st.title("QuizGPT Challenge")

with st.expander("challenges"):
    st.markdown(
        """
        - QuizGPT를 구현하되 다음 기능을 추가합니다:
            - [x] 함수 호출을 사용합니다.
            - [x] 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
            - [x] 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
            - [x] 만점이면 st.ballons를 사용합니다.
            - [ ] 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
            - [ ] st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.
"""
    )


@st.cache_data(show_spinner="Loading file...")
def split_file(file: UploadedFile):
    file_content = file.read()
    file_path = f".cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(topic: str):
    retriever = WikipediaRetriever(top_k_results=1)
    docs = retriever.get_relevant_documents(topic)
    return docs


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_data(show_spinner="Making quiz...")
def invoke_chain(_docs: list[Document], quiz_level: str, topic: str):
    chain = (
        {
            "quiz_level": RunnableLambda(lambda _: quiz_level),
            "context": format_docs,
        }
        | prompt
        | llm
    )
    result = chain.invoke(_docs)
    return result


function = {
    "name": "create_quiz",
    "description": "a function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant that is role playing as a teacher.

Based ONLY on the following context, make 10 questions to test the user's knowledge about the text.
There are two quiz levels, easy and hard. Adjust the difficulty of questions according to the level.

Each question should have four answers, three of them must be incorrect and one should be correct.
The correct answer should be placed randomly among four answers.
\n\n
{quiz_level}
{context}
""",
        )
    ]
)


with st.sidebar:
    api_key = st.text_input("Insert your OpenAI api key")

    docs = None
    source = st.selectbox(
        label="Choose source of the quiz", options=["File", "Wikipedia"]
    )

    if source == "File":
        file = st.file_uploader(
            label="FILE",
            label_visibility="collapsed",
            type=["txt", "pdf", "docx"],
        )

        if file:
            docs = split_file(file)

    elif source == "Wikipedia":
        topic = st.text_input("search on topic")
        if topic:
            docs = wiki_search(topic)

    quiz_level = st.select_slider("Quiz Level", ["easy", "hard"])

    st.link_button(
        label="🔗GITHUB REPO",
        url="https://github.com/iyeka/fullstack-gpt/tree/master/challenges/challenge_7",
    )

if docs and api_key:
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.1,
    ).bind(
        function_call={"name": "create_quiz"},
        functions=[function],
    )

    response = invoke_chain(docs, quiz_level, topic if topic else file.name)
    json_response = json.loads(response.additional_kwargs["function_call"]["arguments"])

    if "form_key" not in st.session_state:
        st.session_state["form_key"] = 0

    with st.form(key=f"questions_form_{st.session_state['form_key']}"):
        count_answers = 0

        for i, question in enumerate(json_response["questions"]):
            st.write(f"**Q. {question['question']}**")
            value = st.radio(
                label="Select an answer",
                options=[answer["answer"] for answer in question["answers"]],
                index=None if st.session_state[f"radio{i}"] is None else None,
                key=f"radio{i}",
            )

            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct")
                count_answers += 1
            elif value is not None:
                st.error("Try again")

        submit_button = st.form_submit_button()

        if count_answers == len(json_response["questions"]):
            st.balloons()

    retry_button = st.button("Try Again")
    if retry_button:
        st.session_state["form_key"] += 1
        st.rerun()


else:
    st.markdown(
        """
Welcome to QuizGPT.
            
I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
            
Get started by:
1. Insert your OpenAI api key
2. Uploading a file or searching on Wikipedia in the sidebar.
"""
    )
