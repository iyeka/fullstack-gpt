# FULLSTACK-GPT

## 2.3 OpenAI Requirements

### 사용하는 서비스

1. LangChain

   1. Open AI API에 비견하여 LangChain의 장점

   - Memory Module을 직접 구현하지 않아도 된다.
   - 코드 변경 없이 어플리케이션 Component 변경 가능 -> 다양한 Models 사용 가능
   - 다양한 모듈과 에이전트에 연결할 수 있는 도구가 이미 만들어져 있다.

2. OpenAI API
3. Streamlit : Python code로 UI 생성
4. Pinecone: Database for Vectors
5. HuggingFace: GPT4 말고 다른 모델 가져오는 법
6. FastAPI: ChatGPT plugin의 API 구축

## 2.6 Jupyter Notebooks의 장점

- print()를 하지 않아도 마지막 줄에 쓰는 변수가 실행된다.
- 코드를 실행하면 메모리에 저장해 다음 셀에서 쓸 수 있다.

## 3.0 LLMs and Chat Models

### 차이

- LLMs:

  - using text-davinci-003

- Chat Models:

  - using gpt-3.5-turbo
  - price is 1/10 of text-davinci-003.
  - more conversational

### What .env file does do?

- environment variable will be loaded to the memory by default when Jupyter Notebook is running.
- LLM과 Chat Model이 .env 라는 이름의 파일에서 OPENAI_API_KEY 라는 이름의 변수를 찾아본다.

## 4.0 MODEL IO: Introduction

### 모듈이란:

- pre-made pieces

### LangChain Modules

- Model I/O

  - Model Inputs: Prompts
  - Language Models
  - Model Outputs: Output Parsers

- Retrieval: 외부 데이터에 접근하여 모델에 제공
- Chains
- Agents: AI 자동화. chain이 필요한 도구들을 직접 선택하여 사용한다.
- Memory: Add memory to chatbot.
- Callbacks: model이 하고 있는 일을 중간에 확인

# 4.4 Serialization and Composition

- Serialize: 저장, 불러오기
- compose: 작은 프롬프트 조각들을 결합하는 것.

## Serialization

- two types of prompts:
  - JSON
  - YAML

#7.1 Magic

## [Streamlit Widgets](https://docs.streamlit.io/develop/api-reference)

### streamlit.write

- 유저가 무엇을 넘겨주던지 화면에 나타낸다.
- st.write(ClassName): Class의 주석과 property 목록까지 함께 화면에 보여준다.

## Streamlit Magic

- st.write(parameter) -> parameter 만 남겨도 전자와 동일하게 화면에 표시된다.
- 변수 이름만 적어도 화면에 표시된다.
- mention a variable or Class or whatever.

#7.2 Data Flow

- data가 변경될 때마다(사용자가 뭔갈 입력하거나, 슬라이더를 드래그하거나 모든 경우) python file 전체가 재실행된다.

## 데이터 숨기는 방법: If-Else

```
import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")
st.title(today)

model = st.selectbox("Choose your model", ("GPT-4o", "Claude-3.0",))
st.write(model)


if model == "Claude-3.0":
    st.write("Cheap")
else:
    st.write("Not cheap.")
    name = st.text_input("What is your name?")
    name

    value = st.slider("temparature", min_value=0.1, max_value=1.0,)
    value
```

#7.3 Multi Page

## 이번 강의 목표:

- Streamlit으로 Sidebar 만들기

1.

```
st.sidebar.title("sidebar title")
```

2. ==with keyword==

```
with st.sidebar:
  st.title("sidebar ttile")
  st.time_input("sidebar chat")
```

\*similar to:

```
tab_one, tab_two, tab_three = st.tabs(["A","B"])
with tab_one:
  st.write("a")

with tab_two:
  st.write("b")
```

- Application 별 Home 만들기

- Pages 만들기: Streamlit is looking for pages folder. So it is important to name it as pages.
  - 만든 pages는 sidebar에 나타나는데, 제목 앞에 숫자를 적으면 순서는 생기지만 숫자 자체는 보이지 않는다.

# 7.4 Chat Messages

## Document GPT Chatbot 만들기

### Streamlit이 가진 Chat Elements

```
import time
import streamlit as st

st.title("DocumentGPT")

with st.chat_message("human"):
    st.write("Hello")

with st.chat_message("ai"):
    st.write("how are you")

# 진행 상황 알리미
with st.status("Embedding File...", expanded=True) as status:
    time.sleep(2)
    st.write("Getting the file")
    time.sleep(2)
    st.write("Embedding the file")
    time.sleep(2)
    st.write("Caching the file")
    status.update(label="error", state="error")

st.chat_input("Send a messsage to AI")
```

### user의 chat message history 보여주기

- Session State
  - 페이지가 re-load 돼도 데이터 보존
  - 형태: list of messages and role [{"message":message, "role":role}]

```
# session_sate에 message key가 없다면: initialize it with empty list and messages key.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def send_message(message, role, save=True):
  with st.chat_message(role):
      st.write(message)
  if save:
      st.session_state["messages"].append({"message":message, "role":role})

# paint messages
for message in st.session_state["messages"]:
    send_message(
        message["message"],
        message["role"],
        save=False,
    )


message = st.chat_input("Send a message to the AI")
if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said {message}", "ai")

    # keep track of cache
    with st.sidebar:
        st.write(st.session_state)
```

# 7.6 Uploading Documents

## 목표:

- user가 Streamlit UI를 통해 질문할 때마다 chain 실행
- Streamlit에서 무거운 연산 cache하는 법

## Steps:

- User가 upload한 file을 가져와 .cache 폴더에 저장

# 7.6 Uploading Documents

- Jupyter Notebook은 .env file에서 OPENAI API KEY를 자동으로 찾지만, streamlit은 못 찾음으로(지금은 수정됨) streamlit이 .env 파일을 읽도록
  1. main folder/ > .streamlit/ > secrets.toml 파일을 만든다.
  2. .env variables를 secrets 파일에 복붙한다.
  3. streamlit 서버를 재시작한다.

# 7.7 Chat History

## Cache Streamlit Results(embed file function)

- Use @st.cache_data decorator on the def embed_file function: The decorator looking for embed_file(file) function's file argument. if file exist, Streamlit will skip that function and return the previous value.

# 8.4 Ollama

- Ollama에서 다운받을 수 있는 Ollama run falcon:180b는 GPT4 만큼 성능을 내는 오픈소스 모델
- Code challenge: sidebar에 llama, mistral, falcon 중 model 선택할 수 있는 dropbox 만들기
