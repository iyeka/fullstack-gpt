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
