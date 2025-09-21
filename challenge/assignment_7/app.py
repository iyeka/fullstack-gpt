import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever

# for system:
import os
import importlib.util
from dotenv import load_dotenv

# for type hints:
from langchain.schema import Document

load_dotenv()
st.set_page_config(
    page_title="풀스택 GPT: #9.0~9.9",
    page_icon="❔",
)
st.title("풀스택 GPT: #9.0~9.9")


def import_utils():
    spec = importlib.util.spec_from_file_location("utils", os.getenv("utils_path"))
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(query: str) -> list[Document]:
    retriever = WikipediaRetriever(top_k_results=1)
    docs = retriever.invoke(query)
    return docs


def retrieve_docs(source: str) -> tuple[list[Document], str]:
    docs = None
    meta = None
    if source == "File":
        file = st.file_uploader("Upload a file here.", ("txt", "pdf", "docx"))
        if file:
            meta = file.name
            docs = utils.split_file(file, f"./challenge/assignment_7/{meta}")
    else:
        query = st.text_input("Type keyword to search Wikipedia.")
        if query:
            docs = wiki_search(query)
            meta = query
    return docs, meta


@st.cache_data(show_spinner="Making quiz...")
def invoke_chain(
    _docs: list[Document], meta: str, test_level: str, api_key: str
) -> dict:

    schema = {
        "name": "format_json",
        "description": "format the multiple-choice question and answers into JSON type.",
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

    llm = ChatOpenAI(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL_NAME"),
        temperature=0.1,
    ).bind(
        functions=[schema],
        function_call={"name": "format_json"},
    )
    easy_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are a helpful assistant role playing as a teacher.
                
            Based ONLY on the following context make 10 easy questions to test the user's knowledge about the text.
            
            Each question should have 4 answers, three of them must be incorrect and one should be correct.
                
            Use (o) to signal the correct answer.
                
            - Question examples:
                
                Question: What is the capital of Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut

            - context: {context}
            """,
            )
        ]
    )
    hard_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are a helpful assistant role playing as a teacher.
                
            Based ONLY on the following context make 10 tricky questions to test the user's knowledge about the text.
            
            Each question should have 4 answers, three of them must be incorrect and one should be correct.
                
            Use (o) to signal the correct answer.
                
            - Question examples:
                
                Question: What is the capital of Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut

            - context: {context}
            """,
            )
        ]
    )

    if test_level == "easy":
        chain = {"context": utils.format_docs} | llm | easy_prompt
    else:
        chain = {"context": utils.format_docs} | llm | hard_prompt
    response = chain.invoke(_docs)
    return response.additional_kwargs["function_call"]["arguments"]


def paint_questions(questions: dict[str, list]):
    with st.form("questions_form"):
        correct = 0

        for question in questions:
            st.write(question["question"])
            value = st.radio(
                "Select an answer.",
                [answer["answer"] for answer in question["answers"]],
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success()
                correct += 1
            elif value is not None:
                st.error("Wrong! Try again.")

        submit = st.form_submit_button()

        if submit:
            if len(questions) == correct:
                st.balloons()


utils = import_utils()
with st.sidebar:
    source = st.selectbox(
        "Choose a source to make quiz.", ("File", "Wikipedia article")
    )
    docs, meta = retrieve_docs(source)
    level = st.selectbox("Select test level.", ("easy", "hard"))
    api_key = st.text_input("OPENAI_API_KEY")
    github = st.link_button("🔗 GitHub Repository", "")

    with st.expander("Tasks"):
        st.markdown(
            """
                - [x] QuizGPT를 구현하되 다음 기능을 추가합니다:
                    - 함수 호출을 사용합니다.
                        - [x] [함수 호출 (Function Calling)](https://python.langchain.com/v0.1/docs/expression_language/primitives/binding/#attaching-openai-functions)을 활용하여 모델의 응답을 원하는 형식으로 변환합니다.
                    - [x] 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
                        - [x] 유저가 시험의 난이도를 선택할 수 있도록 [st.selectbox](https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox)를 사용합니다.
                    - [x] 퀴즈를 화면에 표시하여 유저가 풀 수 있도록 [st.radio](https://docs.streamlit.io/develop/api-reference/widgets/st.radio)를 사용합니다.
                    - [x] 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
                    - 만점이면 st.ballons를 사용합니다.
                        - [x] 만점 여부를 확인하기 위해 문제의 총 개수와 정답 개수가 같은지 비교합니다. 만약 같으면 [st.balloons](https://docs.streamlit.io/develop/api-reference/status/st.balloons)를 사용합니다.
                    - [x] 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
                        - [x] 유저의 자체 OpenAI API 키를 사용하기 위해 [st.text_input](https://docs.streamlit.io/library/api-reference/widgets/st.text_input) 등 을 이용하여 API 키를 입력받습니다. 그런 다음, ChatOpenAI 클래스를 사용할 때 해당 API 키를 openai_api_key 매개변수로 넘깁니다.
                    - [ ] st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.
                - [ ] 제출은 streamlit.app URL로 이루어져야 합니다.
                """
        )


if docs and level:
    response = invoke_chain(docs, meta, level, api_key)
    paint_questions(response)
else:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    1. Upload a file or searching on Wikipedia in the sidebar.
    2. Select test level between easy and hard.
    3. Write down your OPENAI_API_KEY and press enter.
    """
    )
