import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# 과제
md = st.markdown("""
    <details>
    <summary>풀스택 GPT: #7.0 ~ #7.10</summary>
    ## Tasks:\n
    - [x] 이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
    - [x] 파일 업로드 및 채팅 기록을 구현합니다.
        > - [x] 파일 업로드를 구현하기 위해 st.file_uploader를 사용합니다. type 매개변수를 통해 허용할 파일 확장자를 지정할 수 있습니다. [st.file_uploader 공식 문서](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader)
        > - [x] 채팅 기록을 저장하기 위해 Session State를 사용합니다. [Session State 공식 문서](https://docs.streamlit.io/library/api-reference/session-state)
    - [x] 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
        > - [x] 사용자의 OpenAI API 키를 사용하기 위해 st.text_input을 이용하여 API 키를 입력받습니다. 그런 다음, ChatOpenAI와 OpenAIEmbeddings 클래스를 사용할 때 해당 API 키를 openai_api_key 매개변수로 넘깁니다. [st.text_input 공식 문서](https://docs.streamlit.io/library/api-reference/widgets/st.text_input)
    - [ ] st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
        > - [ ] with st.sidebar를 사용하면 사이드바와 관련된 코드를 더욱 깔끔하게 정리할 수 있습니다. [st.sidebar 공식 문서](https://docs.streamlit.io/library/api-reference/layout/st.sidebar)
    - [x] 코드를 공개 Github 리포지토리에 푸시합니다. 
    - [x] 단. OpenAI API 키를 Github 리포지토리에 푸시하지 않도록 주의하세요.
    - [x] 여기에서 계정을 개설하세요: [https://share.streamlit.io/](https://share.streamlit.io/)
    - [x] 다음 단계를 따르세요: [https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app#deploy-your-app-1](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app#deploy-your-app-1)
    - [ ] ~~앱의 구조가 아래와 같은지 확인하고 배포 양식의 Main file path 에 app.py를 작성하세요.~~
        ```
        your-repo/
        ├── .../
        ├── app.py
        └── requirements.txt
        ```
    - [x] 과제 제출 링크는 반드시 streamlit.app URL 이 되도록 하세요.
    </details>
""",
    unsafe_allow_html=True
)
# 해설
md = st.markdown("""
    <details>
    <summary>Solution</summary>
    [https://huq8hcz9ktqsatygmrjsdm.streamlit.app/](https://huq8hcz9ktqsatygmrjsdm.streamlit.app/)
    [https://github.com/fullstack-gpt-python/assignment-15/blob/main/app.py](https://github.com/fullstack-gpt-python/assignment-15/blob/main/app.py)
    1. 사용자의 OpenAI API 키 이용하기
        - st.text_input을 사용하여 사용자의 OpenAI API 키를 입력받습니다.
        - 입력받은 API 키를 ChatOpenAI와 OpenAIEmbeddings 클래스를 사용할 때 openai_api_key 매개변수로 넘깁니다.
    2. 파일 업로드
        - st.file_uploader를 사용하여 사용자가 파일을 업로드할 수 있도록 합니다.
        - 업로드할 수 있는 파일의 확장자는 pdf, txt, docx로 지정합니다.
        - 업로드된 파일을 임베딩하고 vectorstore에 저장한 후, 이를 retriever로 변환하여 체인에서 사용합니다.
        - 이전과 같은 파일을 선택했을 때 임베딩 과정을 다시 하지 않도록 하기 위해 embed_file 함수에 st.cache_data 데코레이터를 추가하였습니다. [st.cache_data 공식 문서](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
    3. 채팅 기록
        - 채팅 기록을 저장하기 위해 Session State를 사용합니다.
        - 솔루션에서는 st.session_state["messages"]를 리스트로 초기화하고, 메시지를 추가하는 방법으로 구현했습니다. (save_message 함수 참고)
        - 저장된 채팅 기록을 페이지에 출력하기 위해 st.session_state["messages"] 리스트에 있는 메시지들을 하나씩 출력합니다. (paint_history 함수 참고)
    </details>
""",
    unsafe_allow_html=True
)

# 유저가 올린 파일 가져오기
def getting_file(file):
    file_content = file.read() # 유저가 올린 파일 내용을 메모리로 가져오기
    file_path = f"./.cache/files/{file.name}" # 유저가 올린 파일을 저장할 경로
    with open(file=file_path, mode="wb") as f:
        f.write(file_content) # 지정한 경로에 복사한 파일 내용 저장
    # LangChain으로 파일 읽기
    loader = UnstructuredFileLoader(file_path) 
    full_doc = loader.load()
    return full_doc # return type: a list of Document object

# Split the file text
def split_file(doc):
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    split_docs = splitter.split_documents(doc)
    return split_docs # return type: a list of Document objects

# 텍스트를 숫자로 변환하고, 이를 캐시와 vectorstore에 저장
def embed_file(docs):
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
    )
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# 메시지 내역 화면에 보이기
def show_history():
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["message"])

# session_state에 메시지 내역 저장
def save_message(message, role, toggle=True):
    st.session_state["chat_history"].append({"message":message, "role":role})
    if toggle:
        show_history()

# UI
st.title("DocumentGPT")
st.write("Welcome! Upload your file and ask chatbot about the file.")

# 사이드 바에 파일 올리는 버튼
with st.sidebar:
    file = st.file_uploader(label="Upload a .txt .pdf .docx type file", type=["txt", "pdf", "docx"])
    api_key=st.text_input("Enter Your OpenAI API Key.")

llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.1,
    api_key=api_key,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Answer the question using ONLY the following context. If you don't know the answer, DON'T make it up.
        
        Context: {context}
        """),
    ("human", "{question}"),
])

if file and api_key:
    doc = getting_file(file) # 유저가 올린 파일 가져오기
    docs = split_file(doc) # Split the file text
    retriever = embed_file(docs) # get relevant docs
    message = st.chat_input("Ask anything about your file.") # 유저 질문란
    if message:
        save_message(message, "human", toggle=True)
        chain = {"context": retriever | RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs)), 
                 "question":RunnablePassthrough()} | prompt | llm
        response = chain.invoke(message)
        with st.chat_message("ai"):
            st.write(response.content)
        save_message(response.content, "ai", toggle=False)
else:
    st.session_state["chat_history"] = [] # 파일이 없으면 chat_history 초기화